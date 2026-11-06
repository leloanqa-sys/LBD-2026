# Hướng Dẫn Tích Hợp Phân Hệ NLP Tái Định Vị Thuốc (Dành Cho Khang)

Tài liệu này tóm tắt toàn bộ phân hệ NLP (Tầng trích xuất quan hệ văn bản & Hợp nhất Đồ thị tri thức) đã được nâng cấp, kiểm định chất lượng và đóng gói sẵn sàng để Khang tích hợp vào luồng Tìm đường (Path Search) và Xếp hạng Giả thuyết (Candidate Ranking) của hệ thống chính `dapagliflozin-lbd`.

---

## 1. Vị trí Các Thành phần Đã Đóng Gói trong Repo `dapagliflozin-lbd`

### Mã nguồn NLP cốt lõi:
- **`src/lbd/nlp/v6_industrial/`**: Kiến trúc Neuro-Symbolic 5 Tầng:
  - `layer1_zoning_and_grounding.py`: Phân vùng cấu trúc (`RESULTS`, `CONCLUSIONS`), chuẩn hóa Hetionet GO Ontology (`GO:0006814`, `GO:0006006`, `GO:0006954`), đồng tham chiếu thuốc xuyên câu.
  - `layer3_event_and_blinded_llm.py`: Trích xuất sự kiện siêu đồ thị N-ary (`BioEventFrame`) & thẩm định LLM mù thực thể (Groq `qwen/qwen3.8-27b`).
  - `layer4_polarity_and_grade.py`: Đại số cực tính ($\text{Sign} = \Delta_{\text{verb}} \times V_{\text{entity}}$), bộ lọc từ phỏng đoán (Hedge Detector) và phân tách mệnh đề.
  - `layer5_bayesian_fusion.py`: Hợp nhất Bayesian Beta-Binomial ở cấp độ PMID ($\mu$ và $\text{LCB}_{90}$), chống bão hòa giả tạo và lưu vết kiểm toán (Provenance Fix).
- **`src/lbd/nlp/bio_re_engine_v3.py`**: Mô hình Tầng A ML đã huấn luyện (Zoned Pooling TruncatedSVD + Neural MLP + Softmax Ensemble) trên BioRED (F1 69.3%, F2 77.1%).
- **`src/lbd/nlp/integrated_repurposing_pipeline.py`**: Pipeline tích hợp end-to-end từ văn bản thô ra Contract B & Contract D.
- **`src/lbd/nlp/evaluate_ood_unseen_corpus.py`**: Kịch bản kiểm chứng học vẹt Out-of-Distribution trên tập bài báo hoàn toàn mới.

### Dữ liệu & Artifacts Chuẩn (Contracts):
- **`data/processed/v6_contract_b_relations.jsonl`**: Quan hệ trích xuất cấp độ câu (Khớp 100% `contracts/relation.schema.json`).
- **`data/processed/v6_contract_d_kg_edges.jsonl`**: Cạnh đồ thị tri thức đã hợp nhất Bayesian kèm dấu sinh học `sign \in {-1, 0, 1}` và `tier \in {1, 2, 3}` (Khớp 100% `contracts/graph_edge.schema.json`).
- **`data/processed/ood_unseen_abstracts_human_review_sample_v2.csv`**: Bảng kiểm toán 41 cạnh trích xuất từ 70 bài báo OOD mới, đã sửa dứt điểm lỗi Provenance (PMID khớp chính xác câu bằng chứng).

---

## 2. Quy Ước 3 Tầng Merge Cạnh vào Đồ Thị Hetionet (Dành Cho Module Đồ Thị của Khang)

Khi nạp (merge) các cạnh từ `data/processed/v6_contract_d_kg_edges.jsonl` vào đồ thị Hetionet nền:

| Phân tầng (Tier) | Thuộc tính trên Cạnh | Số lượng | Nguyên tắc Xử lý của Thuật toán Tìm đường |
| :--- | :--- | :---: | :--- |
| **Tier 1 (Canonical Core)** | `tier: 1`, `sign: -1` hoặc `+1`, `partition: "ADMITTED_KG"` | **16 cạnh** | **Nạp thẳng vào Neo4j/NetworkX graph nền.** Đây là các mắt xích cốt lõi đã có đa PMID khẳng định (`Dapagliflozin -> SLC5A2`, `Glucosuria`, `Natriuresis GO:0006814`, `Albuminuria`, `Heart Failure`, `CKD`, `NAFLD`). Thuật toán DFS/BFS dùng trực tiếp các cạnh này để nhân dấu Metapath. |
| **Tier 2 (Contextual & Baseline)** | `tier: 2`, `sign: 0`, `partition: "HISTORICAL_BASELINE_KG"` hoặc `"COHORT_CONTEXT_KG"` | **9 cạnh** | **Nạp với thuộc tính ngữ cảnh.** Mang `sign: 0` (ngữ cảnh bệnh nền T2D hoặc đoàn hệ thử nghiệm). Không tham gia đảo dấu điều trị, dùng để hiển thị thông tin bổ trợ trên UI. |
| **Tier 3 (Provisional Queue)** | `tier: 3`, `partition: "PROVISIONAL_QUEUE"` | **14 cạnh** | **Cách ly vào hàng đợi thăm dò.** KHÔNG nạp mặc định vào đồ thị tìm đường tự động. Chỉ hiển thị trên Web UI dưới dạng cạnh nét đứt khi người dùng bật bộ lọc *"Khám phá giả thuyết sớm (1 PMID)"*. |
| **Contraindication Lock** | `partition: "CONTRAINDICATED_LOCK"`, `sign: -1` | **1 cạnh** | Khóa chống chỉ định lâm sàng ở bệnh nhân suy thận nặng (`Renal Impairment`). Tuyệt đối không để thuật toán DFS gán nhãn điều trị. |

---

## 3. Cách Khang Chạy Lại Kiểm Tra Nhanh (Smoke Test)

Trong thư mục `dapagliflozin-lbd`:
```bash
# Kiểm tra chạy pipeline trên tập OOD mới
python -m src.lbd.nlp.evaluate_ood_unseen_corpus
```

Kết quả sẽ sinh ra các cạnh Contract D hợp lệ tại `data/processed/` để script `build_graph.py` có thể đọc trực tiếp.
