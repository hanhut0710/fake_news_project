import pandas as pd
import wikipedia
import re
import time
from rank_bm25 import BM25Okapi
from typing import List

# ====================== CONFIG ======================
INPUT_CSV = "processed_fakenews_data.csv"   # <-- ĐỔI TÊN FILE CỦA BẠN Ở ĐÂY
OUTPUT_CSV = "evidence_retrieved.csv"
TOP_K_EVIDENCE = 5                            # Số evidence tốt nhất lấy về cho mỗi claim
WIKI_LANGUAGE = "en"                          # FakeNewsNet là tiếng Anh
# ===================================================

def parse_entities(entities_str: str) -> List[str]:
    """Parse chuỗi entities từ CSV thành list"""
    if not entities_str or entities_str.strip() == "[]" or entities_str.strip() == "":
        return []
    # Xóa dấu [] và split theo dấu phẩy
    cleaned = entities_str.strip("[]")
    return [e.strip() for e in cleaned.split(",") if e.strip()]

def split_sentences(text: str) -> List[str]:
    """Tách thành câu đơn giản và sạch"""
    # Regex tốt cho tiếng Anh
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if len(s.strip()) > 15]  # loại câu quá ngắn

def get_wikipedia_evidence(claim: str, entities: List[str]) -> List[str]:
    """Lấy evidence từ Wikipedia + BM25"""
    documents = []
    wikipedia.set_lang(WIKI_LANGUAGE)

    # 1. Lấy trang của từng entity (ưu tiên)
    for entity in entities:
        try:
            page = wikipedia.page(entity, auto_suggest=True)
            documents.append(page.content)          # content = toàn bộ bài viết (không markup)
            time.sleep(0.5)                         # tránh rate limit
        except (wikipedia.exceptions.PageError, wikipedia.exceptions.DisambiguationError):
            pass
        except Exception:
            pass

    # 2. Search theo claim để bổ sung thêm trang liên quan
    try:
        search_results = wikipedia.search(claim, results=3)
        for title in search_results:
            try:
                page = wikipedia.page(title, auto_suggest=True)
                documents.append(page.content)
                time.sleep(0.5)
            except Exception:
                pass
    except Exception:
        pass

    if not documents:
        return ["No evidence found from Wikipedia."]

    # Ghép tất cả nội dung lại và tách thành câu
    full_text = " ".join(documents)
    corpus = split_sentences(full_text)

    if len(corpus) == 0:
        return ["No evidence found from Wikipedia."]

    # BM25 Retrieval
    tokenized_corpus = [doc.split() for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    tokenized_query = claim.lower().split()

    scores = bm25.get_scores(tokenized_query)
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:TOP_K_EVIDENCE]
    
    top_evidence = [corpus[i] for i in top_indices]
    return top_evidence

# ====================== MAIN ======================
if __name__ == "__main__":
    print("Đang load dữ liệu từ preprocessing...")
    df = pd.read_csv(INPUT_CSV)
    
    print(f"Đang xử lý {len(df)} claims...")
    evidence_list = []
    
    for idx, row in df.iterrows():
        claim = row['claim']
        entities_str = str(row['entities'])
        entities = parse_entities(entities_str)
        
        print(f"[{idx+1}/{len(df)}] Processing: {claim[:80]}...")
        
        evidence = get_wikipedia_evidence(claim, entities)
        
        # Ghép thành string để lưu CSV
        evidence_str = " ||| ".join(evidence)
        evidence_list.append(evidence_str)
        
        # Nghỉ 1 giây để tránh bị Wikipedia block
        time.sleep(1)
    
    df['evidence'] = evidence_list
    
    # Xuất file theo đúng yêu cầu (claim, evidence, label) + các cột khác để tiện debug
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n✅ Hoàn thành! File output: {OUTPUT_CSV}")
    print(f"   - Cột mới: 'evidence' (top {TOP_K_EVIDENCE} câu tốt nhất)")
    print(f"   - Định dạng cho Verification Model: claim | evidence | label")