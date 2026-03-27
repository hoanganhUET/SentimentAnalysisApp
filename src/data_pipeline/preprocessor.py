import json
import os
import gzip

# Đường dẫn cho file dữ liệu (Review 5-core data và Metadata)
reviews_file_path = os.path.join('data/raw/reviews_5_core.json.gz')
metadata_file_path = os.path.join('data/raw/meta_data.json.gz')

# Data được sử dụng là Amazon Product Data (McAuley Lab - UCSD)
def load_data_in_chunks(file_path, chunk_size=10000, data_type='review'):
    """
    Đọc dữ liệu theo từng chunk và lọc các trường không cần thiết để tránh OOM. 
    data_type có thể là 'review' hoặc 'metadata'.
    """
    # Các trường cần giữ lại (Whitelist) để tối ưu RAM
    REVIEW_KEEP_COLS = {'reviewerID', 'asin', 'overall', 'reviewText', 'summary', 'unixReviewTime'}
    META_KEEP_COLS = {'asin', 'title', 'categories'}
    
    keep_cols = REVIEW_KEEP_COLS if data_type == 'review' else META_KEEP_COLS
    chunk = []
    
    # Hỗ trợ đọc trực tiếp file .json.gz mà không cần giải nén
    open_func = gzip.open if file_path.endswith('.gz') else open
    mode = 'rt' if file_path.endswith('.gz') else 'r'
    
    with open_func(file_path, mode, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                raw_item = json.loads(line)
                # Chỉ lọc và giữ lại các giá trị quan trọng
                filtered_item = {k: v for k, v in raw_item.items() if k in keep_cols}
                chunk.append(filtered_item)
            except json.JSONDecodeError:
                continue # Bỏ qua dòng bị lỗi parse
            
            if len(chunk) >= chunk_size:
                yield chunk
                chunk = []
        
        # Trả về nốt phần dữ liệu còn lại
        if chunk:
            yield chunk

