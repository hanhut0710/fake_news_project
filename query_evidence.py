import pandas as pd
import wikipedia
import re
import time
import torch
from sentence_transformers import SentenceTransformer, util
from typing import List
import concurrent.futures
from tqdm import tqdm

# ====================== CONFIG ======================
INPUT_CSV = "processed_fakenews_data.csv"
OUTPUT_CSV = "evidence_retrieved_live.csv"
TOP_K_EVIDENCE = 5
WIKI_LANGUAGE = "en"
MODEL_NAME = 'all-MiniLM-L6-v2'

# THE SPEED LIMITER: Reduced to 3 to avoid Wikipedia rate limiting
# Wikipedia blocks around 10 concurrent requests, so 3 is safe
MAX_WORKERS = 15
RETRY_ATTEMPTS = 2
BACKOFF_FACTOR = 1  # seconds to wait between retries
# ===================================================

def parse_entities(entities_str: str) -> List[str]:
    if not entities_str or pd.isna(entities_str) or str(entities_str).strip() in ["[]", ""]:
        return []
    
    # Xử lý chuỗi list từ CSV (cả hai dạng bạn đưa ra đều được)
    cleaned = str(entities_str).strip("[]")
    entities = [e.strip().strip("'\"") for e in cleaned.split(",") if e.strip()]
    
    # Tách các thực thể ghép với '&' hoặc 'and'
    split_entities = []
    for entity in entities:
        entity = entity.replace('"', '')

        # Split theo & hoặc and (case-insensitive)
        parts = re.split(r'\s*&\s*|\s+and\s+', entity, flags=re.IGNORECASE)
        
        split_entities.extend([p.strip() for p in parts if p.strip()])
    
    return split_entities

def is_valid_entity(e):
    return (
        len(e) > 3 and
        not e.islower() and
        not e.isdigit()
    )
    
def split_sentences(text: str) -> List[str]:
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if len(s.strip()) > 15]

def fetch_and_evaluate(row_data):
    """This function is run by multiple workers simultaneously"""
    idx, claim, entities, model = row_data
    documents = []
    wikipedia.set_lang(WIKI_LANGUAGE)

    # 1. Ping Wikipedia (Network Bound) - Try each entity individually
    for entity in entities:
        for attempt in range(RETRY_ATTEMPTS):
            try:
                # Increased sleep between requests to reduce rate limiting
                time.sleep(0.5)
                results = wikipedia.search(entity, results=1)
                if results:
                    page = wikipedia.page(results[0], auto_suggest=False)
                documents.append(page.content[:5000])  # Cap at 5000 chars to save memory
                break  # Success, move to next entity
            except wikipedia.exceptions.DisambiguationError as e:
                # Pick the first option from disambiguation
                try:
                    time.sleep(0.5)
                    first_option = e.options[0] if e.options else entity
                    page = wikipedia.page(first_option, auto_suggest=False)
                    documents.append(page.content[:5000])
                    break
                except Exception as e2:
                    if attempt == RETRY_ATTEMPTS - 1:
                        pass  # Skip this entity
            except wikipedia.exceptions.PageError:
                if attempt == RETRY_ATTEMPTS - 1:
                    pass  # Entity doesn't exist, skip
            except Exception as e:
                # Rate limiting or network error
                if attempt < RETRY_ATTEMPTS - 1:
                    time.sleep(BACKOFF_FACTOR * (2 ** attempt))  # Exponential backoff
                else:
                    pass  # Give up on this entity

    # Fallback search if entities fail (search for claim keywords)
    if not documents:
        for attempt in range(RETRY_ATTEMPTS):
            try:
                time.sleep(0.5)
                search_results = wikipedia.search(claim, results=3)
                for title in search_results:
                    try:
                        time.sleep(0.5)
                        page = wikipedia.page(title, auto_suggest=True)
                        documents.append(page.content[:5000])
                        break  # Got one document, stop searching
                    except Exception:
                        pass
                if documents:
                    break
            except Exception as e:
                if attempt < RETRY_ATTEMPTS - 1:
                    time.sleep(BACKOFF_FACTOR * (2 ** attempt))
                else:
                    pass

    if not documents:
        return idx, "No evidence found from Wikipedia."

    full_text = " ".join(documents)
    corpus = split_sentences(full_text)[:200]  # Cap sentences for GPU speed

    if len(corpus) == 0:
        return idx, "No evidence found from Wikipedia."

    # 2. GPU Semantic Search (Compute Bound)
    corpus_embeddings = model.encode(corpus, convert_to_tensor=True)
    query_embedding = model.encode(claim, convert_to_tensor=True)

    cosine_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
    
    top_k = min(TOP_K_EVIDENCE, len(corpus))
    top_results = torch.topk(cosine_scores, k=top_k)
    
    top_indices = top_results.indices.tolist()
    top_evidence = [corpus[i] for i in top_indices]
    
    return idx, " ||| ".join(top_evidence)

# ====================== MAIN ======================
if __name__ == "__main__":
    if torch.cuda.is_available():
        device = "cuda"
        gpu_name = torch.cuda.get_device_name(torch.cuda.current_device())
        print(f"🚀 Initializing model on GPU: {gpu_name}")
    else:
        device = "cpu"
        print("⚠️ No GPU detected. Using CPU.")
    
    retriever_model = SentenceTransformer(MODEL_NAME, device=device)

    print("📂 Loading data...")
    df = pd.read_csv(INPUT_CSV)
    
    # Pre-allocate lists to keep results in the exact same order as the CSV
    evidence_results = [""] * len(df)
    split_entities_results = [""] * len(df)
    
    # Package the data for the workers
    tasks = []
    for idx, row in df.iterrows():
        claim = str(row['claim'])
        entities = parse_entities(row.get('entities', ""))
         # 🔥 LỌC ENTITY Ở ĐÂY
        entities = [e for e in entities if is_valid_entity(e)]
        
        # 🔥 GIỚI HẠN SỐ LƯỢNG (rất quan trọng)
        entities = list(set(entities))[:3]
        
        split_entities_results[idx] = str(entities)
        
        tasks.append((idx, claim, entities, retriever_model))

    print(f"🌐 Pinging Wikipedia with {MAX_WORKERS} concurrent threads...")
    print(f"⏱️  Expecting ~{len(df) * 0.5 // 60:.0f}-{len(df) * 0.5 // 60 + 30:.0f} minutes (including retry logic)...\n")
    
    # Launch the Thread Pool
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # executor.map automatically assigns tasks to free workers
        futures = executor.map(fetch_and_evaluate, tasks)
        
        # Wrap the output in tqdm for a live progress bar
        for idx, evidence_str in tqdm(futures, total=len(df), desc="Retrieving Evidence"):
            evidence_results[idx] = evidence_str

            
    df['entities'] = split_entities_results  # Update entities with split version
    df['evidence'] = evidence_results
    df.to_csv(OUTPUT_CSV, index=False)
    
    # Print summary
    found_count = sum(1 for e in evidence_results if e != "No evidence found from Wikipedia.")
    print(f"\n✅ Live Scraping Complete! File output: {OUTPUT_CSV}")
    print(f"📊 Summary: {found_count}/{len(df)} records have evidence ({found_count/len(df)*100:.1f}%)")
    if found_count / len(df) < 0.5:
        print("⚠️  Low evidence retrieval rate. Consider:")
        print("   - Increasing RETRY_ATTEMPTS in config")
        print("   - Increasing sleep durations between requests")
        print("   - Reducing MAX_WORKERS further")