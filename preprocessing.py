import pandas as pd
import spacy
import re
import ast

# 1. Load mô hình ngôn ngữ của spaCy
print("Đang tải mô hình spaCy...")
nlp = spacy.load("en_core_web_sm")

def split_combined_entities(entities_list):
    """
    Tách entity dạng: A & B, A and B, A&B,...
    """
    split_entities = []
    
    for entity in entities_list:
        # Xóa dấu ngoặc kép
        entity = entity.replace('"', '')  # giữ lại dấu '
        
        # Split bằng regex (xử lý mọi case)
        parts = re.split(r'\s*&\s*|\s+and\s+', entity, flags=re.IGNORECASE)
        
        split_entities.extend([p.strip() for p in parts if p.strip()])
    
    return split_entities

def extract_entities(text):
    if not isinstance(text, str):
        return "[]"
    
    doc = nlp(text)
    entities = []
    
    # Lọc các loại thực thể mang thông tin kiểm chứng được (Fact-checking)
    target_labels = ['PERSON', 'ORG', 'GPE', 'LOC', 'EVENT', 'WORK_OF_ART']
    
    for ent in doc.ents:
        if ent.label_ in target_labels:
            entities.append(ent.text)
    
    # Tách các thực thể ghép
    entities = split_combined_entities(entities)
    
    # Xóa trùng lặp và trả về định dạng chuỗi của list
    return str(list(set(entities)))

def process_fakenews_file(file_path, source_name, label_value):
    # Đọc và xử lý một file dữ liệu đơn lẻ
    print(f"Đang xử lý: {file_path}...")
    
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Không tìm thấy file {file_path}. Bỏ qua.")
        return pd.DataFrame()

    # Kiểm tra xem file có cột 'id' và 'title' không
    id_col = 'id' if 'id' in df.columns else 'news_id'
    
    if id_col not in df.columns or 'title' not in df.columns:
        print(f"Lỗi: File {file_path} không có cột {id_col} hoặc 'title'")
        return pd.DataFrame()

    # 1. Trích xuất Claim (Sử dụng tiêu đề)
    df['claim'] = df['title']
    
    # 2. Trích xuất Entities
    df['entities'] = df['claim'].apply(extract_entities)
    
    # 3. Gán nhãn và Nguồn
    df['label'] = label_value
    df['source'] = source_name
    
    # 4. Đổi tên cột id cho đồng nhất và chuẩn bị output
    df = df.rename(columns={id_col: 'id'})
    
    # Trả về DataFrame với đúng 5 cột yêu cầu
    return df[['id', 'claim', 'entities', 'label', 'source']]


# Bước 3: Chạy Pipeline và Gộp dữ liệu

if __name__ == "__main__":
    
    data_files = [
        {"path": "data/raw/gossipcop_fake.csv", "source": "gossipcop", "label": "fake"},
        {"path": "data/raw/gossipcop_real.csv", "source": "gossipcop", "label": "real"},
        {"path": "data/raw/politifact_fake.csv", "source": "politifact", "label": "fake"},
        {"path": "data/raw/politifact_real.csv", "source": "politifact", "label": "real"}
    ]
    
    processed_dfs = []
    
    for item in data_files:
        processed_df = process_fakenews_file(item["path"], item["source"], item["label"])
        if not processed_df.empty:
            processed_dfs.append(processed_df)
            
    if processed_dfs:
        # Gộp tất cả dữ liệu lại
        final_dataset = pd.concat(processed_dfs, ignore_index=True)
        
        # Lưu ra file CSV cuối cùng
        output_file = "processed_fakenews_data.csv"
        final_dataset.to_csv(output_file, index=False)
        print(f"\nHoàn thành! Đã lưu file tại: {output_file}")

        print("\nVí dụ 1 record từ tập dữ liệu vừa tạo:")
        print(final_dataset.iloc[0].to_dict())
    else:
        print("\nKhông có dữ liệu nào được xử lý. Hãy kiểm tra lại đường dẫn file.")