# START HERE — Hướng dẫn Nhanh cho 4 Thành viên (Huy, Khang, Lâm, Phước)

Chào mừng bạn đến với kho mã nguồn chính thức `dapagliflozin-lbd` (Phiên bản v3.0 — Đã tích hợp **Signed Graph NTR Flux**, **Degree Debiasing $\beta=0.6$**, **Ephemeral Session KG**, và **4 Bộ Kiểm định Ground Truth**).

---

## 1. Ba Lệnh Cần Chạy Ngay Sau Khi Clone

```powershell
# 1. Cài đặt môi trường
pip install -r requirements.txt
pip install -e ".[dev]"

# 2. Chạy tự kiểm định theo tên thành viên của bạn (huy | khang | lam | phuoc | all)
python scripts/run_team_verification.py --member all

# 3. Chạy toàn bộ 34 bài kiểm thử đơn vị & hợp đồng dữ liệu
python -m pytest tests/ -v
```

---

## 2. Bản Đồ Nhanh Theo Từng Thành Viên

### 👤 Đặng Quang Huy — Data, Bộ lọc KG & Nạp Session Tạm Hetionet
- **File chính cần nắm:**
  - [`src/lbd/common/ids.py`](file:///c:/Users/Admin/Downloads/tái%20định%20vị%20thuốc/dapagliflozin-lbd/src/lbd/common/ids.py) & [`src/lbd/data/normalize.py`](file:///c:/Users/Admin/Downloads/tái%20định%20vị%20thuốc/dapagliflozin-lbd/src/lbd/data/normalize.py): Ánh xạ tên thuốc/gen/bệnh sang mã chuẩn Hetionet (`DBxxxxx`, Entrez Gene ID, `DOID:xxxx`).
  - [`scripts/collect_multidrug_abstracts.py`](file:///c:/Users/Admin/Downloads/tái%20định%20vị%20thuốc/dapagliflozin-lbd/scripts/collect_multidrug_abstracts.py) & [`data/raw/landmark_curated_abstracts.csv`](file:///c:/Users/Admin/Downloads/tái%20định%20vị%20thuốc/dapagliflozin-lbd/data/raw/landmark_curated_abstracts.csv): Thu thập bài báo trước mốc khóa thời gian `2018-12-31`.
  - [`scripts/filter_huy.py`](file:///c:/Users/Admin/Downloads/tái%20định%20vị%20thuốc/dapagliflozin-lbd/scripts/filter_huy.py) & [`configs/kg_filter.json`](file:///c:/Users/Admin/Downloads/tái%20định%20vị%20thuốc/dapagliflozin-lbd/configs/kg_filter.json): Chặn câu giả thuyết/phủ định và bài báo vượt mốc thời gian.
  - Hàm `clone_for_session()` và `augment_session_edges()` trong [`src/lbd/ranking/m2_ranker.py`](file:///c:/Users/Admin/Downloads/tái%20định%20vị%20thuốc/dapagliflozin-lbd/src/lbd/ranking/m2_ranker.py#L532-L645): Nạp cạnh mới từ NLP vào bản sao RAM của Hetionet (Zero Disk Mutation).
- **Lệnh test riêng của Huy:**
  ```powershell
  python scripts/run_team_verification.py --member huy
  python scripts/test_live_session_pipeline.py
  ```

### 👤 Nguyễn Khang — Biomedical NLP, BioRED Engine & Verbatim Grounding
- **File chính cần nắm:**
  - [`src/lbd/nlp/engine.py`](file:///c:/Users/Admin/Downloads/tái%20định%20vị%20thuốc/dapagliflozin-lbd/src/lbd/nlp/engine.py), [`classifier.py`](file:///c:/Users/Admin/Downloads/tái%20định%20vị%20thuốc/dapagliflozin-lbd/src/lbd/nlp/classifier.py), [`model_v3_biored_multiclass.pkl`](file:///c:/Users/Admin/Downloads/tái%20định%20vị%20thuốc/dapagliflozin-lbd/src/lbd/nlp/model_v3_biored_multiclass.pkl): Mô hình phân loại quan hệ y sinh.
  - [`src/lbd/nlp/zoning.py`](file:///c:/Users/Admin/Downloads/tái%20định%20vị%20thuốc/dapagliflozin-lbd/src/lbd/nlp/zoning.py) & [`src/lbd/nlp/ner.py`](file:///c:/Users/Admin/Downloads/tái%20định%20vị%20thuốc/dapagliflozin-lbd/src/lbd/nlp/ner.py): Chia vùng IMRaD và nhận diện thực thể lai (PubTator 3.0 + từ điển regex nội bộ).
  - [`src/lbd/nlp/events.py`](file:///c:/Users/Admin/Downloads/tái%20định%20vị%20thuốc/dapagliflozin-lbd/src/lbd/nlp/events.py), [`polarity.py`](file:///c:/Users/Admin/Downloads/tái%20định%20vị%20thuốc/dapagliflozin-lbd/src/lbd/nlp/polarity.py), [`fusion.py`](file:///c:/Users/Admin/Downloads/tái%20định%20vị%20thuốc/dapagliflozin-lbd/src/lbd/nlp/fusion.py): Xác định dấu tương tác sinh học (`-1` ức chế/điều trị, `+1` kích hoạt/gây bệnh, `0` liên kết).
- **Lệnh test riêng của Khang:**
  ```powershell
  python scripts/run_team_verification.py --member khang
  python -m pytest tests/test_relation_extraction.py -v
  ```

### 👤 Nguyễn Nhật Lâm — Thuật toán DWPC, Khử thiên lệch ($\beta=0.6$) & Signed Graph NTR
- **File chính cần nắm:**
  - [`src/lbd/ranking/m2_ranker.py`](file:///c:/Users/Admin/Downloads/tái%20định%20vị%20thuốc/dapagliflozin-lbd/src/lbd/ranking/m2_ranker.py): Chứa toàn bộ công thức ma trận DWPC ($w=0.4$), khử thiên lệch bậc bệnh ($\beta=0.6$), tính luồng dấu sinh học **Net Therapeutic Ratio (NTR Flux)**, và truy vết đường đi **Meet-in-the-Middle Micro-DFS**.
  - [`artifacts/ranking_m2/`](file:///c:/Users/Admin/Downloads/tái%20định%20vị%20thuốc/dapagliflozin-lbd/artifacts/ranking_m2/): Lưu trữ `features.npz` (212.624 cặp $\times$ 68 đặc trưng), `M2_model.pkl`, `M3_model.pkl`, `split_indices.npz`.
- **Lệnh test riêng của Lâm:**
  ```powershell
  python scripts/run_team_verification.py --member lam
  python scripts/run_signed_graph_integration_benchmark.py
  ```

### 👤 Nguyễn Công Gia Phước — Kiểm định (Evaluation), Ground Truth & Web UI
- **File chính cần nắm:**
  - [`data/ground_truth/`](file:///c:/Users/Admin/Downloads/tái%20định%20vị%20thuốc/dapagliflozin-lbd/data/ground_truth/) & [`data/ground_truth/README_GROUND_TRUTH.md`](file:///c:/Users/Admin/Downloads/tái%20định%20vị%20thuốc/dapagliflozin-lbd/data/ground_truth/README_GROUND_TRUTH.md): Chứa hồ sơ chứng thực (`ground_truth_credentials.json`), tập nhãn train (`gold_labels_from_hetionet.csv`), và 4 bộ Test Suites (`prospective_temporal_test.json`, `rare_disease_test.json`, `clinical_failure_untrue_test.json`, `landmark_4_classes_test.json`).
  - [`src/lbd/evaluation/`](file:///c:/Users/Admin/Downloads/tái%20định%20vị%20thuốc/dapagliflozin-lbd/src/lbd/evaluation/): Tính `Recall@5`, `Recall@10`, `Precision@K`, `NDCG@10`, `MRR`.
  - [`ui/serve.py`](file:///c:/Users/Admin/Downloads/tái%20định%20vị%20thuốc/dapagliflozin-lbd/ui/serve.py) & [`ui/explorer.html`](file:///c:/Users/Admin/Downloads/tái%20định%20vị%20thuốc/dapagliflozin-lbd/ui/explorer.html): Giao diện Human-in-the-Loop cho phép chuyên gia chỉnh tham số và lưu quyết định lâm sàng.
- **Lệnh test riêng của Phước:**
  ```powershell
  python scripts/run_team_verification.py --member phuoc
  python scripts/run_landmark_ablation_study.py
  python scripts/run_live_e2e_repurposing_benchmark.py
  ```
