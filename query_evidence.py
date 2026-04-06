import pandas as pd
import wikipedia
import re
import time
import torch
from sentence_transformers import SentenceTransformer, util
from typing import List
import concurrent.futures
from tqdm import tqdm
from preprocessing import extract_entities

# ====================== CONFIG ======================
INPUT_CSV = "processed_fakenews_data.csv"
OUTPUT_CSV = "evidence_retrieved_live.csv"
TOP_K_EVIDENCE = 5
WIKI_LANGUAGE = "en"
MODEL_NAME = 'all-MiniLM-L6-v2'

# THE SPEED LIMITER: Reduced to 3 to avoid Wikipedia rate limiting
# Wikipedia blocks around 10 concurrent requests, so 3 is safe
MAX_WORKERS = 3
RETRY_ATTEMPTS = 2
BACKOFF_FACTOR = 1  # seconds to wait between retries
# ===================================================

# Lightweight in-process cache to avoid re-querying Wikipedia repeatedly
_EVIDENCE_CACHE = {}

# Device and shared retriever model (instantiate once)

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
    # Better sentence splitter: keep sentences that contain some informative length
    sentences = re.split(r'(?<=[.!?])\s+', text)
    cleaned = [s.strip() for s in sentences if len(s.strip()) > 15]
    return cleaned


def clean_text(text: str) -> str:
    # Remove wiki-style headings and weird artifacts, truncate long repeated sections
    if not isinstance(text, str):
        return ""
    # remove section headers like == Early life ==
    text = re.sub(r'=+\s*[^=]+\s*=+', ' ', text)
    # replace pipeline separators used elsewhere
    text = text.replace('|||', '. ')
    # remove multiple newlines and weird unicode
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def extract_ordinals(text: str) -> List[int]:
    # Find ordinals like 45th, 1st, 2nd, 3rd etc.
    nums = []
    if not isinstance(text, str):
        return nums
    for m in re.findall(r"(\d+)(?:st|nd|rd|th)\b", text, flags=re.IGNORECASE):
        try:
            nums.append(int(m))
        except:
            pass
    return nums

def fetch_and_evaluate(row_data):
    """This function is run by multiple workers simultaneously"""
    idx, claim, entities, model = row_data
    documents = []
    wikipedia.set_lang(WIKI_LANGUAGE)

    # 1. Ping Wikipedia (Network Bound) - Try each entity individually
    for entity in entities:
        for attempt in range(RETRY_ATTEMPTS):
            try:
                # polite small delay to avoid aggressive scraping
                time.sleep(0.2)
                results = wikipedia.search(entity, results=1)
                if results:
                    page = wikipedia.page(results[0], auto_suggest=False)
                # Clean and cap content
                documents.append(clean_text(page.content)[:2000])  # Cap at 2000 chars to save memory
                break  # Success, move to next entity
            except wikipedia.exceptions.DisambiguationError as e:
                # Pick the first option from disambiguation
                try:
                    first_option = e.options[0] if e.options else entity
                    page = wikipedia.page(first_option, auto_suggest=False)
                    documents.append(clean_text(page.content)[:2000])
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
                time.sleep(0.2)
                search_results = wikipedia.search(claim, results=2)
                for title in search_results:
                    try:
                        page = wikipedia.page(title, auto_suggest=True)
                        documents.append(clean_text(page.content)[:2000])
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
    corpus = split_sentences(full_text)[:100]  # Cap sentences for speed

    # Basic relevance filter: keep sentences that share tokens with claim or contain entity
    claim_tokens = set([w.lower() for w in re.findall(r'\w+', claim)][:12])
    filtered = []
    for s in corpus:
        s_low = s.lower()
        if any(tok in s_low for tok in claim_tokens) or any(e.lower() in s_low for e in entities):
            filtered.append(s)

    if len(filtered) == 0:
        # fallback to using original corpus
        filtered = corpus

    # 2. Semantic Search (Compute Bound)
    # Use the model instance passed in the row_data tuple (parameter name: model)
    corpus_embeddings = model.encode(filtered, convert_to_tensor=True, show_progress_bar=False)
    query_embedding = model.encode(claim, convert_to_tensor=True, show_progress_bar=False)

    cosine_scores = util.cos_sim(query_embedding, corpus_embeddings)[0].cpu()

    # Boost sentences that contain ordinals or years matching the claim
    claim_ord = extract_ordinals(claim)
    year_match = re.findall(r"\b(18|19|20)\d{2}\b", claim)

    adjusted_scores = cosine_scores.numpy().tolist()
    for i, s in enumerate(filtered):
        boost = 0.0
        s_ord = extract_ordinals(s)
        # exact ordinal match -> boost
        if any(o in s_ord for o in claim_ord):
            boost += 0.25
        # year presence
        if year_match and any(ym in s for ym in year_match):
            boost += 0.15
        # entity exact match
        if any(e.lower() in s.lower() for e in entities):
            boost += 0.10
        adjusted_scores[i] += boost

    top_k = min(TOP_K_EVIDENCE, len(filtered))
    top_idx = sorted(range(len(adjusted_scores)), key=lambda i: adjusted_scores[i], reverse=True)[:top_k]
    top_evidence = [filtered[i] for i in top_idx]

        # Clean and truncate each evidence sentence to reduce noise
    cleaned = []
    for t in top_evidence:
        t2 = clean_text(t)
        if len(t2) > 700:
            t2 = t2[:700].rsplit(' ', 1)[0] + '...'
        cleaned.append(t2)

    return idx, " ||| ".join(cleaned)


def query_evidence(claim, retriever_model, nlp):

    if claim in _EVIDENCE_CACHE:
        return _EVIDENCE_CACHE[claim]

    entities = extract_entities(claim, nlp)
    entities = list(dict.fromkeys(entities))[:3]  # Limit to top 3 unique entities

    print("Extracted entities:", entities)

    row_data = (0, claim, entities, retriever_model)

    print("Row data:", row_data)
    _, evidence_str = fetch_and_evaluate(row_data)

    _EVIDENCE_CACHE[claim] = evidence_str
    return evidence_str


# ====================== MAIN ======================
if __name__ == "__main__":
    # Instantiate a local retriever model for script mode
    retriever_device = "cuda" if torch.cuda.is_available() else "cpu"
    if retriever_device == "cuda":
        gpu_name = torch.cuda.get_device_name(torch.cuda.current_device())
        print(f"🚀 Using GPU for retrieval model: {gpu_name}")
    else:
        print("⚠️ No GPU detected. Retrieval will use CPU.")
    retriever_model = SentenceTransformer(MODEL_NAME, device=retriever_device)

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