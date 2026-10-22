# Data Directory Structure

- `raw/`: Chứa các tài liệu thô tải về từ PubMed (chỉ lấy abstract $\le 31/12/2018$) và Hetionet v1.0.
- `processed/`: Dữ liệu sau khi làm sạch, chuẩn hóa thực thể và quan hệ.
- `cache/`: Cache kết quả gọi PubTator 3.0 API và Groq LLM theo PMID.
- `samples/`: Các tập mẫu nhỏ dùng để chạy unit tests và smoke test.
