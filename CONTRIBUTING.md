# Hướng Dẫn Đóng Góp Dự Án (Contribution Guide)

## Phân công trách nhiệm nhóm (Team Roles)
- **Đặng Quang Huy (Huy)**: Data & ETL Engineer (Contract A, Contract C, Contract D, Thu thập PubMed, Đối chiếu chéo ID, Ghép đồ thị Hetionet + NLP).
- **Nguyễn Khang (Khang)**: NLP Engineer (PubTator 3.0 NER, Groq LLM Relation Extraction, Contract B, Validation).
- **Nguyễn Nhật Lâm (Liam)**: Knowledge Graph Reasoning & Ranking Engineer (Metapath-Constrained DFS, Path Search, Ranking Engine, Contract E).
- **Nguyễn Công Gia Phước (Phước)**: Evaluation & Application Specialist (Ground Truth, Temporal Backtesting, Baseline Comparison, Smoke Test).

## Quy trình làm việc (Data Contract Workflow)
1. Mọi module phải tuân thủ nghiêm ngặt Schema trong thư mục `contracts/`.
2. Dữ liệu khi bàn giao giữa các thành viên bắt buộc phải vượt qua validation gate tương ứng.
3. Chạy `pytest tests/` và `python scripts/run_smoke_test.py` trước khi merge code.
