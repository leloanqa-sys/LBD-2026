# Ground Truth, Train Data & Out-of-Sample Benchmark Registry

**Người phụ trách chính (Owner):** Nguyễn Công Gia Phước (Evaluation, Ground Truth & Benchmarks)  
**Phối hợp kiểm định (Co-owners):** Đặng Quang Huy (Data Cutoff & KG Session), Nguyễn Nhật Lâm (Ranking Models), Nguyễn Khang (BioRED NLP Grounding)

Thư mục `data/ground_truth/` lưu trữ toàn bộ **nhãn chuẩn (Gold Standard Labels)**, **thông tin truy xuất nguồn gốc (Credentials & Provenance)**, và **4 bộ kiểm định ngoài vùng huấn luyện (Out-of-Sandbox Test Suites)** của dự án Tái định vị thuốc.

---

## 1. Danh mục tệp dữ liệu trong `data/ground_truth/`

| Tên tệp | Loại dữ liệu | Quy mô | Nguồn gốc chuẩn (Credentials / DOI) | Mục đích sử dụng |
| :--- | :--- | :--- | :--- | :--- |
| [`gold_labels_from_hetionet.csv`](file:///c:/Users/Admin/Downloads/tái%20định%20vị%20thuốc/dapagliflozin-lbd/data/ground_truth/gold_labels_from_hetionet.csv) | Tập Train / Nested CV | 1.145 cặp có nhãn (`755 CtD` dương, `390 CpD` loại trừ) trên tổng 212.624 cặp `(Compound, Disease)` | **PharmacotherapyDB v1.0 / Hetionet v1.0** (`DOI: 10.7554/eLife.26726`, CC0 1.0) | Huấn luyện & kiểm định chéo (5-fold Outer / 3-fold Inner Nested CV theo cụm cấu trúc `CrC`) cho các mô hình `M0`, `M2`, `M3` trong `artifacts/ranking_m2/`. |
| [`prospective_temporal_test.json`](file:///c:/Users/Admin/Downloads/tái%20định%20vị%20thuốc/dapagliflozin-lbd/data/ground_truth/prospective_temporal_test.json) | **Suite A** — Kiểm định Tương lai (Post-2018) | 10 cặp tái định vị dương tính mới (2019–2023) | **FDA Drug Approvals & ClinicalTrials.gov Phase 3** | Đo khả năng dự đoán tương lai (Prospective Prediction) cho các chỉ định được duyệt sau mốc khóa thời gian `2018-12-31`. |
| [`rare_disease_test.json`](file:///c:/Users/Admin/Downloads/tái%20định%20vị%20thuốc/dapagliflozin-lbd/data/ground_truth/rare_disease_test.json) | **Suite B** — Bệnh hiếm (Rare Diseases) | 11 cặp ở các nút bệnh có bậc $\le 30$ | **Orphanet & Hetionet v1.0 Low-Degree Split** | Kiểm tra độ nhạy của các metapath nhỏ (Micro-metapaths) và thuật toán khử thiên lệch bậc ($\beta = 0.6$) khi đồ thị thưa. |
| [`clinical_failure_untrue_test.json`](file:///c:/Users/Admin/Downloads/tái%20định%20vị%20thuốc/dapagliflozin-lbd/data/ground_truth/clinical_failure_untrue_test.json) | **Suite C** — Thất bại lâm sàng (Hard Negatives) | 184 cặp thất bại thử nghiệm lâm sàng (`label = 0`) | **repoDB** (`Brown & Patel, Sci Data 2017, DOI: 10.1038/sdata.2017.29`) | Kiểm tra khả năng của **Signed Graph (NTR Flux)** trong việc phát hiện cơ chế sinh học thuận chiều bệnh sinh (`pathogenic`) và đẩy các thuốc thất bại lâm sàng xuống hạng thấp. |
| [`landmark_4_classes_test.json`](file:///c:/Users/Admin/Downloads/tái%20định%20vị%20thuốc/dapagliflozin-lbd/data/ground_truth/landmark_4_classes_test.json) | **Suite D** — 4 Dòng thuốc Kinh điển (Landmark Repurposing) | 7 cặp đại diện cho 4 cơ chế nhảy đồ thị (Cross-Graph Jump) | **FDA Approvals 2019–2023** (Spravato, Lodoco, Farxiga, Invokana, Wegovy/STEP-1) | Bài kiểm tra toàn diện từ cào bài báo (`landmark_curated_abstracts.csv`) $\rightarrow$ trích xuất NLP BioRED $\rightarrow$ nạp vào phiên tạm Hetionet $\rightarrow$ đo `MRR`, `Recall@K`, `Avg Rank`. |
| [`case_study_gold.csv`](file:///c:/Users/Admin/Downloads/tái%20định%20vị%20thuốc/dapagliflozin-lbd/data/ground_truth/case_study_gold.csv) | Bảng tổng hợp Gold Standard | 17 cặp chuẩn mực | Tổng hợp từ Suite A & Suite D | Bảng CSV đối chiếu nhanh cho báo cáo và UI Human-in-the-Loop. |
| [`ground_truth_pairs.csv`](file:///c:/Users/Admin/Downloads/tái%20định%20vị%20thuốc/dapagliflozin-lbd/data/ground_truth/ground_truth_pairs.csv) | Bảng gộp toàn bộ Test Suites | 205 cặp kiểm định (`Suite A + B + C`) | Tổng hợp có kiểm duyệt (`VERIFIED`) | Dùng cho các script thống kê tổng thể của Phước. |
| [`ground_truth_credentials.json`](file:///c:/Users/Admin/Downloads/tái%20định%20vị%20thuốc/dapagliflozin-lbd/data/ground_truth/ground_truth_credentials.json) | Hồ sơ chứng thực & Checksum | Metadata JSON | SHA-256 & DOI Registry | Lưu trữ mã băm SHA-256 của đồ thị gốc và thông tin pháp lý/học thuật của từng tập dữ liệu. |

---

## 2. Quy tắc Chống rò rỉ dữ liệu (Zero-Leakage Guarantee)

1. **Khóa thời gian văn bản (Temporal Cutoff $\le$ `2018-12-31`):**
   - Mọi bài báo PubMed nạp vào đường ống NLP chỉ được phép chứa kiến thức sinh học cơ bản trước năm 2019 (ví dụ: *Ketamine ức chế GRIN1/GRIN2A*, *Colchicine ức chế NLRP3/IL1B*, *Dapagliflozin ức chế SLC5A2*).
   - Tuyệt đối không nạp cạnh `Compound-treats-Disease (CtD)` trực tiếp từ văn bản vào đồ thị; hệ thống chỉ nạp các cạnh trung gian (`Compound -> Gene` hoặc `Gene -> Disease`) vào phiên RAM tạm thời (`augment_session_edges`).
2. **Bảo toàn tuyệt đối đồ thị gốc (Zero Disk Mutation):**
   - Tệp `data/external/hetionet/hetionet-v1.0-edges.sif.gz` luôn giữ nguyên mã băm SHA-256:
     `611f0411ac666be4e0270e54bbca67407e14720bfc3099311f0f5eec25c5a947`.
3. **Phân cụm cấu trúc thuốc khi Train (`split_indices.npz`):**
   - Tập huấn luyện `features.npz` (212.624 cặp $\times$ 68 đặc trưng) được chia theo cụm đồng phân cấu trúc (`CrC` connected components) để đảm bảo không có hai thuốc cùng họ hóa học vừa nằm ở tập Train vừa nằm ở tập Test.

---

## 3. Lệnh chạy kiểm định nhanh trên Ground Truth

```powershell
# Chạy kiểm định toàn diện của Phước (Ground Truth + 4 Landmark Cases + Ablation)
python scripts/run_team_verification.py --member phuoc

# Chạy Ablation Study chi tiết trên 4 dòng thuốc kinh điển (Ketamine, Colchicine, SGLT2i, GLP-1)
python scripts/run_landmark_ablation_study.py

# Chạy kiểm định E2E từ nạp văn bản NLP vào Session Graph đến đo MRR/Hit@K
python scripts/run_live_e2e_repurposing_benchmark.py
```
