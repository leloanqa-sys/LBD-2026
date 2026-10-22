# Dapagliflozin Literature-Based Discovery (LBD)

Hệ thống Hỗ trợ Khám phá Tri thức Y văn và Tái định vị thuốc (Drug Repurposing) ứng dụng AI/NLP và Đồ thị tri thức (Knowledge Graph) cho trường hợp nghiên cứu **Dapagliflozin**.

## Mục tiêu Dự án
- Tự động đọc và bóc tách thực thể & quan hệ y sinh từ PubMed (mốc cắt $T = 31/12/2018$).
- Hợp nhất đồ thị Hetionet v1.0 và các quan hệ y văn trích xuất.
- Tìm kiếm đường đi sinh học đa chặng ($Drug \to Gene \to Pathway \to Disease$) với kiểm soát lan truyền dấu (Signed Rules).
- Xếp hạng bệnh ứng viên (Candidate Ranking) và giải thích nguồn gốc minh bạch với trích dẫn PMID.
- Kiểm nghiệm hồi cứu (Temporal Backtesting) trên 2 nhãn vàng:
  - **Ground Truth 1**: Suy tim (HFrEF, FDA phê duyệt 05/2020)
  - **Ground Truth 2**: Bệnh thận mạn (CKD, FDA phê duyệt 04/2021)

## Cấu trúc Dự án
- `configs/`: File cấu hình YAML cho đường dẫn, dữ liệu, NLP, đồ thị, ranking, evaluation.
- `contracts/`: JSON Schemas kiểm soát chất lượng dữ liệu giữa các luồng bàn giao (A -> E).
- `src/lbd/`: Mã nguồn module chuẩn hóa (Common, Data, NLP, Graph, Ranking, Evaluation).
- `notebooks/`: Chuỗi Jupyter Notebooks thực nghiệm tuần tự từ data audit đến đánh giá.
- `data/`: Dữ liệu mẫu (samples) và tài liệu cấu trúc thư mục data.
- `artifacts/`: Nơi lưu trữ các kết quả trung gian và mô hình.
- `tests/`: Bộ kiểm thử tự động (Unit tests & Contract tests).
- `docs/`: Kiến trúc chi tiết, từ điển dữ liệu, quy tắc sinh học, quyết định kỹ thuật.
- `scripts/`: Scripts kiểm tra smoke test và kiểm định chất lượng artifact.
