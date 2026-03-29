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

# THE SPEED LIMITER: 15 is fast, but safe enough to avoid IP bans
MAX_WORKERS = 15 
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

def split_sentences(text: str) -> List[str]:
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if len(s.strip()) > 15]

def fetch_and_evaluate(row_data):
    """This function is run by multiple workers simultaneously"""
    idx, claim, entities, model = row_data
    documents = []
    wikipedia.set_lang(WIKI_LANGUAGE)

    # 1. Ping Wikipedia (Network Bound)
    for entity in entities:
        try:
            # Added a tiny sleep to space out requests slightly within the thread
            time.sleep(0.1) 
            page = wikipedia.page(entity, auto_suggest=True)
            documents.append(page.content[:5000]) # Cap at 5000 chars to save memory
        except Exception:
            pass

    # Fallback search if entities fail
    if not documents:
        try:
            search_results = wikipedia.search(claim, results=2)
            for title in search_results:
                try:
                    time.sleep(0.1)
                    page = wikipedia.page(title, auto_suggest=True)
                    documents.append(page.content[:5000])
                except Exception:
                    pass
        except Exception:
            pass

    if not documents:
        return idx, "No evidence found from Wikipedia."

    full_text = " ".join(documents)
    corpus = split_sentences(full_text)[:200] # Cap sentences for GPU speed

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
        split_entities_results[idx] = str(entities)  # Store split entities
        tasks.append((idx, claim, entities, retriever_model))

    print(f"🌐 Pinging Wikipedia with {MAX_WORKERS} concurrent threads...")
    
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
    
    print(f"\n✅ Live Scraping Complete! File output: {OUTPUT_CSV}")