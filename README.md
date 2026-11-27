# Hệ thống Tái định vị Thuốc trên Đồ thị Tri thức Động & Suy luận Nhân quả (Dynamic Causal KG & Literature-Based Discovery)

> **🎓 ĐỒ ÁN MÔN HỌC: GIẢI PHÁP TRÍ TUỆ NHÂN TẠO (AI SOLUTIONS)**  
> **🏛️ ĐƠN VỊ ĐÀO TẠO:** Khoa Hệ thống Thông tin / Công nghệ Thông tin Kinh doanh — **Trường Đại học Kinh tế - Luật (UEL), Đại học Quốc gia TP.HCM (VNU-HCM)**  
> **🏷️ DỰ ÁN:** Nghiên cứu và Thiết kế Hệ thống Suy luận Nhân quả Hỗ trợ Tái định vị Thuốc (Computational Drug Repurposing via Dynamic Knowledge Graphs & Causal Inference)  
> **📌 TRIẾT LÝ HỌC THUẬT:** "Học trong sự nghèo nàn tri thức" (Hetionet mốc 2017), White-Box Dual-Head Causal Ranker, 0% rò rỉ dữ liệu hậu-2017.

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![UEL VNU-HCM](https://img.shields.io/badge/institution-UEL%20VNU--HCM-003366.svg)]()
[![Course: AI Solutions](https://img.shields.io/badge/course-Gi%E1%BA%A3i%20Ph%C3%A1p%20AI-orange.svg)]()
[![Tests: 70 Passed](https://img.shields.io/badge/pytest-70%20passed-brightgreen.svg)]()
[![Single Source of Truth: Verified](https://img.shields.io/badge/benchmark%20sync-49%2F49%20PASS-success.svg)]()
[![Zero Post-2017 Leakage](https://img.shields.io/badge/temporal%20boundary-Hetionet%202017-red.svg)]()

---

## 1. Phân chia Khối Công việc & Quyền Quản lý File của 4 Thành viên

Mỗi thành viên chịu trách nhiệm hoàn toàn trên các module, hợp đồng dữ liệu (Contracts A–E) và lệnh tự kiểm định riêng:

| Thành viên | Vai trò chuyên môn | Các thư mục & Tệp mã nguồn cần nắm vững | Hợp đồng dữ liệu (Contracts) | Lệnh tự kiểm tra độc lập |
| :--- | :--- | :--- | :--- | :--- |
| **1. Đặng Quang Huy** | **Data Engineering, Bộ lọc KG & Nạp Session Tạm Hetionet** | • [`src/lbd/data/fetch_pubmed.py`](file:///c:/Users/Admin/Downloads/tái%20định%20vị%20thuốc/dapagliflozin-lbd/src/lbd/data/fetch_pubmed.py)<br>• [`src/lbd/data/load_hetionet.py`](file:///c:/Users/Admin/Downloads/tái%20định%20vị%20thuốc/dapagliflozin-lbd/src/lbd/data/load_hetionet.py)<br>• [`src/lbd/data/normalize.py`](file:///c:/Users/Admin/Downloads/tái%20định%20vị%20thuốc/dapagliflozin-lbd/src/lbd/data/normalize.py), [`src/lbd/common/ids.py`](file:///c:/Users/Admin/Downloads/tái%20định%20vị%20thuốc/dapagliflozin-lbd/src/lbd/common/ids.py)<br>• [`scripts/collect_multidrug_abstracts.py`](file:///c:/Users/Admin/Downloads/tái%20định%20vị%20thuốc/dapagliflozin-lbd/scripts/collect_multidrug_abstracts.py)<br>• [`scripts/filter_huy.py`](file:///c:/Users/Admin/Downloads/tái%20định%20vị%20thuốc/dapagliflozin-lbd/scripts/filter_huy.py), [`configs/kg_filter.json`](file:///c:/Users/Admin/Downloads/tái%20định%20vị%20thuốc/dapagliflozin-lbd/configs/kg_filter.json)<br>• [`src/lbd/graph/build_graph.py`](file:///c:/Users/Admin/Downloads/tái%20định%20vị%20thuốc/dapagliflozin-lbd/src/lbd/graph/build_graph.py)<br>• Hàm `clone_for_session()` & `augment_session_edges()` trong [`src/lbd/ranking/m2_ranker.py`](file:///c:/Users/Admin/Downloads/tái%20định%20vị%20thuốc/dapagliflozin-lbd/src/lbd/ranking/m2_ranker.py#L532-L645) | • **Contract A** (`abstract.schema.json`)<br>• **Contract C** (`entity.schema.json`)<br>• **Contract D** (`kg_edge.schema.json` & `kg_ingestion_edge.schema.json`) | `python scripts/run_team_verification.py --member huy` |
| **2. Nguyễn Khang** | **Biomedical NLP, BioRED Relation Extraction & Verbatim Grounding** | • Toàn bộ [`src/lbd/nlp/`](file:///c:/Users/Admin/Downloads/tái%20định%20vị%20thuốc/dapagliflozin-lbd/src/lbd/nlp/):<br>  - [`engine.py`](file:///c:/Users/Admin/Downloads/tái%20định%20vị%20thuốc/dapagliflozin-lbd/src/lbd/nlp/engine.py), [`classifier.py`](file:///c:/Users/Admin/Downloads/tái%20định%20vị%20thuốc/dapagliflozin-lbd/src/lbd/nlp/classifier.py)<br>  - [`zoning.py`](file:///c:/Users/Admin/Downloads/tái%20định%20vị%20thuốc/dapagliflozin-lbd/src/lbd/nlp/zoning.py), [`ner.py`](file:///c:/Users/Admin/Downloads/tái%20định%20vị%20thuốc/dapagliflozin-lbd/src/lbd/nlp/ner.py)<br>  - [`events.py`](file:///c:/Users/Admin/Downloads/tái%20định%20vị%20thuốc/dapagliflozin-lbd/src/lbd/nlp/events.py), [`polarity.py`](file:///c:/Users/Admin/Downloads/tái%20định%20vị%20thuốc/dapagliflozin-lbd/src/lbd/nlp/polarity.py), [`fusion.py`](file:///c:/Users/Admin/Downloads/tái%20định%20vị%20thuốc/dapagliflozin-lbd/src/lbd/nlp/fusion.py)<br>  - [`model_v3_biored_multiclass.pkl`](file:///c:/Users/Admin/Downloads/tái%20định%20vị%20thuốc/dapagliflozin-lbd/src/lbd/nlp/model_v3_biored_multiclass.pkl)<br>  - [`crosswalk_mesh_doid.json`](file:///c:/Users/Admin/Downloads/tái%20định%20vị%20thuốc/dapagliflozin-lbd/src/lbd/nlp/crosswalk_mesh_doid.json)<br>• [`scripts/run_nlp_extraction.py`](file:///c:/Users/Admin/Downloads/tái%20định%20vị%20thuốc/dapagliflozin-lbd/scripts/run_nlp_extraction.py) | • **Contract B** (`extracted_relation.schema.json` — bắt buộc câu trích dẫn nguyên văn `evidence` & dấu sinh học `sign` $\in \{-1, 0, +1\}$) | `python scripts/run_team_verification.py --member khang` |
| **3. Nguyễn Nhật Lâm** | **Thuật toán Xếp hạng M2 DWPC, Khử thiên lệch bậc ($\beta=0.6$), Signed Graph NTR & Micro-DFS** | • Toàn bộ [`src/lbd/ranking/`](file:///c:/Users/Admin/Downloads/tái%20định%20vị%20thuốc/dapagliflozin-lbd/src/lbd/ranking/):<br>  - [`m2_ranker.py`](file:///c:/Users/Admin/Downloads/tái%20định%20vị%20thuốc/dapagliflozin-lbd/src/lbd/ranking/m2_ranker.py) (Trái tim thuật toán)<br>  - [`features.py`](file:///c:/Users/Admin/Downloads/tái%20định%20vị%20thuốc/dapagliflozin-lbd/src/lbd/ranking/features.py), [`score_paths.py`](file:///c:/Users/Admin/Downloads/tái%20định%20vị%20thuốc/dapagliflozin-lbd/src/lbd/ranking/score_paths.py), [`rank_candidates.py`](file:///c:/Users/Admin/Downloads/tái%20định%20vị%20thuốc/dapagliflozin-lbd/src/lbd/ranking/rank_candidates.py)<br>• [`src/lbd/graph/filter_subgraph.py`](file:///c:/Users/Admin/Downloads/tái%20định%20vị%20thuốc/dapagliflozin-lbd/src/lbd/graph/filter_subgraph.py), [`path_search.py`](file:///c:/Users/Admin/Downloads/tái%20định%20vị%20thuốc/dapagliflozin-lbd/src/lbd/graph/path_search.py)<br>• [`artifacts/ranking_m2/`](file:///c:/Users/Admin/Downloads/tái%20định%20vị%20thuốc/dapagliflozin-lbd/artifacts/ranking_m2/) (`features.npz`, `M2_model.pkl`, `M3_model.pkl`, `split_indices.npz`) | • **Contract E** (`ranked_candidate.schema.json`) | `python scripts/run_team_verification.py --member lam` |
| **4. Nguyễn Công Gia Phước** | **Kiểm định (Evaluation), Ground Truth, Ablation Study & Giao diện Human-in-the-Loop UI** | • Toàn bộ [`src/lbd/evaluation/`](file:///c:/Users/Admin/Downloads/tái%20định%20vị%20thuốc/dapagliflozin-lbd/src/lbd/evaluation/):<br>  - [`ground_truth.py`](file:///c:/Users/Admin/Downloads/tái%20định%20vị%20thuốc/dapagliflozin-lbd/src/lbd/evaluation/ground_truth.py), [`metrics.py`](file:///c:/Users/Admin/Downloads/tái%20định%20vị%20thuốc/dapagliflozin-lbd/src/lbd/evaluation/metrics.py), [`run_experiment.py`](file:///c:/Users/Admin/Downloads/tái%20định%20vị%20thuốc/dapagliflozin-lbd/src/lbd/evaluation/run_experiment.py)<br>• Toàn bộ [`data/ground_truth/`](file:///c:/Users/Admin/Downloads/tái%20định%20vị%20thuốc/dapagliflozin-lbd/data/ground_truth/) & [`artifacts/benchmarks/`](file:///c:/Users/Admin/Downloads/tái%20định%20vị%20thuốc/dapagliflozin-lbd/artifacts/benchmarks/)<br>• [`scripts/run_landmark_ablation_study.py`](file:///c:/Users/Admin/Downloads/tái%20định%20vị%20thuốc/dapagliflozin-lbd/scripts/run_landmark_ablation_study.py)<br>• [`scripts/run_live_e2e_repurposing_benchmark.py`](file:///c:/Users/Admin/Downloads/tái%20định%20vị%20thuốc/dapagliflozin-lbd/scripts/run_live_e2e_repurposing_benchmark.py)<br>• [`ui/serve.py`](file:///c:/Users/Admin/Downloads/tái%20định%20vị%20thuốc/dapagliflozin-lbd/ui/serve.py), [`ui/explorer.html`](file:///c:/Users/Admin/Downloads/tái%20định%20vị%20thuốc/dapagliflozin-lbd/ui/explorer.html) | • Giám sát toàn bộ **Contracts A–E**, kiểm định `MRR`, `Recall@K`, `NDCG@K` & phê duyệt lâm sàng trên UI | `python scripts/run_team_verification.py --member phuoc` |

---

## 2. Hướng dẫn Cài đặt Chuẩn chỉ (Step-by-Step Setup Guide)

### Bước 1: Khởi tạo môi trường ảo Python ($\ge$ 3.10)
Mở PowerShell hoặc Terminal tại thư mục gốc `dapagliflozin-lbd`:

```powershell
cd "c:\Users\Admin\Downloads\tái định vị thuốc\dapagliflozin-lbd"

# Tạo môi trường ảo (khuyến nghị)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Cài đặt đầy đủ thư viện runtime + công cụ kiểm thử (editable mode)
pip install -r requirements.txt
pip install -e ".[dev]"
```

### Bước 2: Cấu hình biến môi trường (`.env` — tùy chọn cho giải thích AI)
Hệ thống chạy **100% offline** đối với toàn bộ quy trình Trích xuất NLP BioRED, Nạp Session Hetionet, Xếp hạng M2/Signed Graph và Đo kiểm Benchmark. Nếu muốn bật tính năng tổng hợp giải thích bằng LLM trên giao diện Web UI:

```powershell
# Tạo tệp .env tại thư mục gốc dapagliflozin-lbd
Set-Content -Path .env -Value "GROQ_API_KEY=gsk_your_api_key_here"
```

### Bước 3: Chạy kiểm định toàn hệ thống (Unit Tests + 4-Member Self-Test)

```powershell
# 1. Chạy 34 bài kiểm thử đơn vị & bất biến kiến trúc (Pytest)
python -m pytest tests/ -v

# 2. Chạy kịch bản tự kiểm định toàn diện cho cả 4 thành viên (Huy, Khang, Lâm, Phước)
python scripts/run_team_verification.py --member all
```

---

## 3. Cấu trúc Thư mục Chuẩn hóa (`dapagliflozin-lbd/`)

```text
dapagliflozin-lbd/
├── configs/
│   ├── pipeline.yaml                        # Cấu hình tham số toàn cục (cutoff 2018-12-31, top_m, damping w=0.4)
│   └── kg_filter.json                       # Quy tắc lọc cạnh nghiêm ngặt của Huy (Contract D gate)
├── contracts/                               # 6 JSON Schemas khóa chặt giao tiếp giữa 4 thành viên
│   ├── abstract.schema.json                 # Contract A (Huy -> Khang)
│   ├── extracted_relation.schema.json       # Contract B (Khang -> Huy)
│   ├── entity.schema.json                   # Contract C (Huy <-> Khang)
│   ├── kg_edge.schema.json                  # Contract D (Huy -> Lâm)
│   ├── kg_ingestion_edge.schema.json        # Contract D Strict (filter_huy.py)
│   └── ranked_candidate.schema.json         # Contract E (Lâm -> Phước)
├── data/
│   ├── external/hetionet/                   # Đồ thị gốc Hetionet v1.0 (Bất biến - Zero Disk Mutation)
│   ├── raw/
│   │   ├── landmark_curated_abstracts.csv   # 10 bài báo y văn chuẩn mực (pre-2019) cho 4 dòng thuốc kinh điển
│   │   └── dapagliflozin_abstracts.csv      # Tập bài báo mở rộng của Dapagliflozin
│   └── ground_truth/                        # [PHƯỚC QUẢN LÝ] Toàn bộ nhãn chuẩn, Train labels & 4 Test Suites
│       ├── README_GROUND_TRUTH.md           # Hồ sơ chứng thực nguồn gốc (DOI, SHA-256, chống rò rỉ)
│       ├── ground_truth_credentials.json    # Metadata chứng thực của từng tập dữ liệu
│       ├── gold_labels_from_hetionet.csv    # Nhãn huấn luyện gốc từ PharmacotherapyDB / Hetionet v1.0
│       ├── prospective_temporal_test.json   # Suite A: 10 cặp thuốc tái định vị được duyệt sau năm 2018
│       ├── rare_disease_test.json           # Suite B: 11 cặp bệnh hiếm (bậc nút <= 30)
│       ├── clinical_failure_untrue_test.json# Suite C: 184 cặp thất bại thử nghiệm lâm sàng từ repoDB
│       ├── landmark_4_classes_test.json     # Suite D: 4 dòng thuốc kinh điển (Ketamine, Colchicine, SGLT2i, GLP-1)
│       ├── case_study_gold.csv              # Bảng CSV đối chiếu nhanh các ca chuẩn mực
│       └── ground_truth_pairs.csv           # Bảng CSV gộp toàn bộ 205 cặp kiểm định
├── artifacts/
│   ├── ranking_m2/                          # [LÂM QUẢN LÝ] Trọng số mô hình & Ma trận đặc trưng đã huấn luyện
│   │   ├── features.npz                     # Ma trận X (212.624 cặp x 68 đặc trưng DWPC + Signed features)
│   │   ├── split_indices.npz                # Chỉ số chia 5-fold Nested CV theo cụm cấu trúc CrC
│   │   ├── M0_model.pkl                     # Mô hình Baseline chỉ dùng bậc nút (Degree Prior)
│   │   ├── M2_model.pkl                     # Mô hình DWPC Topological ElasticNet (65 đặc trưng)
│   │   ├── M3_model.pkl                     # Mô hình Signed Graph (68 đặc trưng gồm L1000 & Directional)
│   │   └── data_manifest.json               # Ánh xạ chỉ số thuốc (1.552), bệnh (137) và tên 68 đặc trưng
│   └── benchmarks/                          # [PHƯỚC QUẢN LÝ] Báo cáo JSON kết quả thực nghiệm & Ablation
│       ├── e2e_live_landmark_benchmark_report.json
│       ├── landmark_ablation_study_report.json
│       ├── live_pipeline_session_audit.json
│       └── signed_graph_integration_benchmark_report.json
├── src/lbd/
│   ├── common/                              # Cấu hình, logging, валиdation & ID Crosswalk (Huy)
│   ├── data/                                # Tải PubMed, Hetionet & chuẩn hóa thực thể (Huy)
│   ├── nlp/                                 # Trích xuất quan hệ BioRED, Zoning, NER, Cực tính (Khang)
│   ├── graph/                               # Dựng đồ thị, lọc ego-subgraph & DFS cơ bản (Huy & Lâm)
│   ├── ranking/                             # M2RepurposingRanker, DWPC, β=0.6, Signed NTR & Micro-DFS (Lâm)
│   └── evaluation/                          # Ground Truth, Recall@K, MRR, NDCG@K (Phước)
├── scripts/
│   ├── run_team_verification.py             # [SCRIPT CHÍNH] Tự kiểm tra theo từng thành viên (--member huy|khang|lam|phuoc|all)
│   ├── run_explorer.py                      # Đường ống E2E nối từ bài báo -> NLP -> Session KG -> Ranking
│   ├── test_live_session_pipeline.py        # Kiểm định đối chứng Có NLP vs Không có NLP (Zero Disk Mutation)
│   ├── run_live_e2e_repurposing_benchmark.py# Chạy E2E từ nạp cạnh vào đồ thị đến đo MRR trên 4 dòng thuốc
│   ├── run_landmark_ablation_study.py       # Chạy Ablation Study (C1 -> C6) trên 4 dòng thuốc kinh điển
│   └── run_signed_graph_integration_benchmark.py # Đo kiểm Signed Graph trên Suite A, B, C
├── ui/
│   ├── serve.py                             # Máy chủ Web Human-in-the-Loop (`/api/rerank`, `/api/review`, `/api/explain`)
│   └── explorer.html                        # Giao diện trực quan hóa Signed Graph, thanh trượt β / λ_ntr & bác sĩ thẩm định
└── tests/                                   # 34 bài kiểm thử tự động (Unit, Contract, Acceptance Invariants)
```

---

## 4. Tóm tắt Kết quả Kiểm định Thực nghiệm (Ground Truth Benchmarks)

### 4.1. Thực nghiệm E2E từ Nạp cạnh Y văn vào Session Hetionet đến Đo MRR (4 Dòng thuốc Kinh điển)
Chạy lệnh `python scripts/run_live_e2e_repurposing_benchmark.py`:

| Dòng thuốc & Đích bệnh (FDA Approved Post-2018) | Cạnh NLP nạp vào Session | Hạng (Không NLP) | Hạng (Có NLP Session) | Dịch chuyển hạng | Điểm số (No NLP $\rightarrow$ With NLP) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **1. Ketamine (`DB01221`) $\rightarrow$ Trầm cảm (`DOID:1595`)** | 3 cạnh (`GRIN1`, `GRIN2A`, `BDNF`) | #24 | **#21** | **$\uparrow +3$ bậc** | `0.8588` $\rightarrow$ `0.8806` |
| **2a. Colchicine (`DB01394`) $\rightarrow$ Bệnh mạch vành (`DOID:3393`)** | 5 cạnh (`CdG:INHIBITS NLRP3`, `IL1B`) | #35 | **#8** | **$\uparrow +27$ bậc (Top 10)** | `0.6254` $\rightarrow$ `0.7885` |
| **2b. Colchicine (`DB01394`) $\rightarrow$ Xơ vữa động mạch (`DOID:1936`)** | 5 cạnh (`CdG:INHIBITS NLRP3`, `IL1B`) | #35 | **#7** | **$\uparrow +28$ bậc (Top 10)** | `0.6226` $\rightarrow$ `0.7867` |
| **3a. Dapagliflozin (`DB06292`) $\rightarrow$ Suy tim (`DOID:12930`)** | 9 cạnh (`CdG:INHIBITS SLC5A2`) | #39 | **#5** | **$\uparrow +34$ bậc (Top 5)** | `0.7131` $\rightarrow$ `0.9355` |
| **3b. Dapagliflozin (`DB06292`) $\rightarrow$ Suy thận mạn (`DOID:784`)** | 9 cạnh (`CdG:INHIBITS SLC5A2`) | #6 | **#3** | **$\uparrow +3$ bậc (Top 3)** | `0.9462` $\rightarrow$ `0.9963` |
| **4. Nhóm Peptide GPCR / GLP-1 (`DB00104`) $\rightarrow$ Béo phì (`DOID:9970`)** | 2 cạnh (`GLP1R` multi-tissue axis) | #71 | **#52** | **$\uparrow +19$ bậc** | `0.3407` $\rightarrow$ `0.4090` |
| **Tổng hợp toàn bộ (Global Metrics)** | **Bảo toàn SHA-256 (Zero Disk Mutation)** | **MRR = `0.0627`** | **MRR = `0.1303` (+108.0%)** | **Avg Rank: `35.0` $\rightarrow$ `16.0`** | **Hit@5: `0%` $\rightarrow$ `33.3%`** |

### 4.2. Khởi chạy Giao diện Human-in-the-Loop Web UI

```powershell
python ui/serve.py --port 8000
```
Sau đó mở trình duyệt tại `http://localhost:8000` để:
1. Chạy truy vấn đa thuốc theo thời gian thực hoặc chế độ Offline.
2. Kéo thanh trượt **Signed Graph NTR Weight ($\lambda_{\text{ntr}}$)** và **Degree Debiasing ($\beta$)** để tái xếp hạng tức thì (<50ms).
3. Xem nhãn dấu sinh học (`[Đảo ngược bệnh sinh (-1)]` vs `[Thuận chiều bệnh sinh (+1)]`) và lưu quyết định thẩm định của chuyên gia (`APPROVED`, `REJECTED`, `NEEDS_TRIAL`).
