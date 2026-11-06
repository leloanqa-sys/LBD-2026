# Hệ Thống Kiến Trúc Toàn Diện (System Architecture)

Dự án áp dụng nguyên tắc kiến trúc **Offline Research Pipeline & Online Query Application**.

## 1. Khối Chức Năng
- **ETL & Data Ingestion**: Thu thập tài liệu y văn từ PubMed $\le 31/12/2018$, đối chiếu PMC và ClinicalTrials.gov.
- **Biomedical NLP (NER & RE)**: Nhận diện thực thể bằng PubTator 3.0 API; trích xuất quan hệ có cấu trúc và dấu sinh học thông qua Groq LLM (Llama 3.1 70B/8B).
- **Knowledge Graph Integration**: Hợp nhất đồ thị nền tĩnh Hetionet v1.0 với các cạnh trích xuất từ y văn vào NetworkX / Neo4j.
- **Metapath-Constrained LBD & Path Ranking**: Duyệt đồ thị DFS theo khuôn mẫu siêu đường đi (Metapath), tính điểm suy luận đa nhân tố ($EQ, Penalty, DirectionFit$).
- **Explanation & Evidence Retrieval**: Truy vết bằng chứng nguyên văn kèm mã PMID hỗ trợ thẩm định Human-in-the-Loop.
