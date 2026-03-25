import pandas as pd
import sqlite3
import re
import torch
from sentence_transformers import SentenceTransformer, util
from typing import List
from tqdm import tqdm

# ====================== CONFIG ======================
INPUT_CSV = "processed_fakenews_data.csv"
OUTPUT_CSV = "evidence_retrieved.csv"
DB_NAME = "local_wikipedia.db"
TOP_K_EVIDENCE = 5
MODEL_NAME = 'all-MiniLM-L6-v2'
# ===================================================

def parse_entities(entities_str: str) -> List[str]:
    if not entities_str or pd.isna(entities_str) or str(entities_str).strip() in ["[]", ""]:
        return []
    cleaned = str(entities_str).strip("[]")
    return [e.strip() for e in cleaned.split(",") if e.strip()]

def split_sentences(text: str) -> List[str]:
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if len(s.strip()) > 15]

def get_local_evidence(claim: str, entities: List[str], model: SentenceTransformer, cursor: sqlite3.Cursor) -> List[str]:
    """Fetches text instantly from local DB, then uses GPU Semantic Search"""
    documents = []

    # 1. Local Database Lookup (Instant)
    for entity in entities:
        # Try exact match first
        cursor.execute("SELECT text FROM wiki WHERE title = ? COLLATE NOCASE", (entity,))
        row = cursor.fetchone() # Just grab the best match
        
        if row:
            # Keep only the first 5000 characters to prevent GPU bottlenecking
            documents.append(row[0][:5000])
        else:
            # Fallback: "Starts with" search (This uses the index and is INSTANT)
            cursor.execute("SELECT text FROM wiki WHERE title LIKE ? COLLATE NOCASE LIMIT 1", (f"{entity}%",))
            row = cursor.fetchone()
            if row:
                documents.append(row[0][:5000])

    if not documents:
        return ["No evidence found in local Wikipedia."]

    full_text = " ".join(documents)
    corpus = split_sentences(full_text)
    
    # Hard cap at 200 sentences to ensure GPU runs at lightning speed
    corpus = corpus[:200]

    if len(corpus) == 0:
        return ["No evidence found in local Wikipedia."]

    # 2. GPU Accelerated Retrieval
    # PyTorch processes this batch instantly now
    corpus_embeddings = model.encode(corpus, convert_to_tensor=True)
    query_embedding = model.encode(claim, convert_to_tensor=True)

    cosine_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
    
    top_k = min(TOP_K_EVIDENCE, len(corpus))
    top_results = torch.topk(cosine_scores, k=top_k)
    
    top_indices = top_results.indices.tolist()
    return [corpus[i] for i in top_indices]

# ====================== MAIN ======================
if __name__ == "__main__":
    # 1. Initialize GPU Model
    if torch.cuda.is_available():
        device = "cuda"
        gpu_name = torch.cuda.get_device_name(torch.cuda.current_device())
        print(f"🚀 Initializing model on GPU: {gpu_name}")
    else:
        device = "cpu"
        print("⚠️ No GPU detected. Using CPU.")
    
    retriever_model = SentenceTransformer(MODEL_NAME, device=device)

    # 2. Connect to Local Database
    try:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
    except sqlite3.OperationalError:
        print(f"❌ Error: Could not find {DB_NAME}. Did you run the builder script first?")
        exit()

    # 3. Load Data
    print("📂 Loading data from preprocessing...")
    df = pd.read_csv(INPUT_CSV)
    evidence_list = []
    
    # 4. Process Loop with Progress Bar
    print(f"⚙️ Processing {len(df)} claims locally...")
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Fact Checking"):
        claim = str(row['claim'])
        entities = parse_entities(row.get('entities', ""))
        
        evidence = get_local_evidence(claim, entities, retriever_model, cursor)
        evidence_list.append(" ||| ".join(evidence))
        
    # 5. Save Results
    df['evidence'] = evidence_list
    df.to_csv(OUTPUT_CSV, index=False)
    conn.close()
    
    print(f"\n✅ Pipeline Complete! File output: {OUTPUT_CSV}")