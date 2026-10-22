# HƯỚNG DẪN THIẾT KẾ VÀ QUY CHUẨN LÀM VIỆC DÀNH CHO TEAM (TEAM DESIGN & WORKFLOW GUIDE)

Tài liệu này là cẩm nang bắt buộc dành cho 4 thành viên nhóm nghiên cứu:
- **Đặng Quang Huy (Huy)** – *Data & ETL Engineer*
- **Nguyễn Khang (Khang)** – *Biomedical NLP & LLM Engineer*
- **Nguyễn Nhật Lâm (Liam)** – *Knowledge Graph & Algorithm Engineer*
- **Nguyễn Công Gia Phước (Phước)** – *Evaluation & Application Specialist*

---

## 1. Bản Đồ Không Gian Sở Hữu (Code Ownership Matrix)

Mỗi thành viên làm chủ một không gian module độc lập trong hệ thống để tránh xung đột mã nguồn (merge conflicts) và đảm bảo tính trách nhiệm cao nhất:

```
dapagliflozin-lbd/
├── configs/
│   ├── data.yaml                 <-- Huy
│   ├── nlp.yaml                  <-- Khang
│   ├── graph.yaml, ranking.yaml  <-- Liam
│   └── evaluation.yaml           <-- Phước
│
├── contracts/
│   ├── paper.schema.json         <-- Huy (Chủ quản Contract A)
│   ├── relation.schema.json      <-- Khang (Chủ quản Contract B & C)
│   ├── graph_edge.schema.json    <-- Huy & Liam (Chủ quản Contract D)
│   └── candidate.schema.json     <-- Liam & Phước (Chủ quản Contract E)
│
├── src/lbd/
│   ├── common/                   <-- Toàn đội sử dụng chung
│   ├── data/                     <-- ĐẶNG QUANG HUY SỞ HỮU
│   ├── nlp/                      <-- NGUYỄN KHANG SỞ HỮU
│   ├── graph/                    <-- NGUYỄN NHẬT LÂM SỞ HỮU
│   ├── ranking/                  <-- NGUYỄN NHẬT LÂM SỞ HỮU
│   └── evaluation/               <-- NGUYỄN CÔNG GIA PHƯỚC SỞ HỮU
│
├── notebooks/
│   ├── 00_data_audit.ipynb       <-- Huy
│   ├── 01_nlp_baseline.ipynb     <-- Khang
│   ├── 02_relation_extraction.ipynb <-- Khang
│   ├── 03_graph_build.ipynb      <-- Liam
│   ├── 04_path_search.ipynb      <-- Liam
│   └── 05_evaluation.ipynb       <-- Phước
│
└── tests/
    ├── test_contracts.py         <-- Huy & Khang
    ├── test_normalization.py     <-- Huy
    ├── test_relation_extraction.py <-- Khang
    ├── test_signed_rules.py      <-- Liam
    ├── test_path_search.py       <-- Liam
    └── test_ranking.py           <-- Liam & Phước
```

---

## 2. Nhiệm Vụ Chi Tiết & Đầu Việc Cần Triển Khai Tiếp Theo

### A. Đặng Quang Huy (Data & ETL)
- **Khu vực phụ trách**: `src/lbd/data/`, `configs/data.yaml`, `contracts/paper.schema.json`, `notebooks/00_data_audit.ipynb`.
- **Nhiệm vụ cốt lõi**:
  1. **PubMed API Collector**: Triển khai hoàn chỉnh script tải 664 abstracts thực tế của Dapagliflozin qua Entrez E-utilities với câu truy vấn chuẩn.
  2. **Temporal Cutoff Guard**: Tuyệt đối đảm bảo trường `publication_date <= 2018-12-31`. Đánh dấu `temporal_filter_status = "passed"` hoặc `"rejected"`.
  3. **ID Crosswalk**: Hoàn thiện bảng ánh xạ `id_crosswalk.csv` đồng bộ giữa DrugBank (`DB06292`), Entrez Gene (`6524`), MeSH và Disease Ontology (DOID).
  4. **Hetionet v1.0 Subgraph**: Tải và trích xuất mạng con bán kính $k$-hop quanh Dapagliflozin từ dữ liệu tĩnh Hetionet 2017.
- **Sản phẩm bàn giao**: File `data/processed/papers_pre2018.jsonl` vượt qua validation của `contracts/paper.schema.json` (Contract A).

### B. Nguyễn Khang (Biomedical NLP & LLM)
- **Khu vực phụ trách**: `src/lbd/nlp/`, `configs/nlp.yaml`, `contracts/relation.schema.json`, `notebooks/01_...`, `notebooks/02_...`.
- **Nhiệm vụ cốt lõi**:
  1. **PubTator 3.0 NER**: Gọi API PubTator 3.0 theo batch (50 PMIDs/lần), lưu cache cục bộ trong `data/cache/pubtator/` để tránh gọi lặp lại.
  2. **Groq LLM Relation Extraction**: Tích hợp Groq API với model `llama-3.1-70b-versatile` (fallback `llama-3.1-8b-instant`), thiết lập `temperature = 0.0`.
  3. **Verbatim Evidence Verification**: Kiểm tra câu bằng chứng (`evidence`) phải tồn tại nguyên văn trong abstract nguồn.
  4. **Manual Annotation Sample**: Lấy mẫu ngẫu nhiên 100 - 200 quan hệ ra `data/review_sample.csv` để cả nhóm chấm chéo độ chuẩn xác (Precision $\approx 80\%$).
- **Sản phẩm bàn giao**: File `data/processed/relations_extracted.jsonl` đạt chuẩn Contract B & C.

### C. Nguyễn Nhật Lâm (Knowledge Graph & Ranking)
- **Khu vực phụ trách**: `src/lbd/graph/`, `src/lbd/ranking/`, `configs/graph.yaml`, `configs/ranking.yaml`, `notebooks/03_...`, `notebooks/04_...`.
- **Nhiệm vụ cốt lõi**:
  1. **Knowledge Graph Merge**: Hợp nhất các cạnh tĩnh từ Hetionet v1.0 và các cạnh NLP mới; gắn trường `source_dataset` (`hetionet_v1.0` vs `nlp_literature`).
  2. **Signed Rules Mapping**: Thiết lập bảng dấu sinh học:
     - $INHIBITS (-1)$, $ACTIVATES (+1)$, $DECREASES (-1)$, $CAUSES (+1)$, $ASSOCIATED\_WITH (0)$.
  3. **Metapath-Constrained DFS**: Dò tìm đường đi độ dài $L \in [3, 5]$ theo mẫu $Compound \to Gene \to [BP/PW] \to Disease$. **Bắt buộc cắt tỉa ngay nhánh đi qua `DOID:9352` (Type 2 Diabetes)**.
  4. **Ranking Engine**: Tính điểm $PathScore(p)$ với hệ số khớp chiều $DirectionFit$ (1.5 cho -1, 1.0 cho 0, 0.0 cho +1) và khử siêu nút bằng hệ số $Penalty(p) = \gamma^{L-1} \prod deg(v)^w$. Tổng hợp Top 5 đường đi cốt lõi ra điểm $Score(d)$ và trừ phạt mâu thuẫn $\lambda \ln(1 + |U_{neg}|)$.
- **Sản phẩm bàn giao**: File `artifacts/subgraph_dapa.json` và `artifacts/candidate_rankings.json` chuẩn Contract D & E.

### D. Nguyễn Công Gia Phước (Evaluation & Application)
- **Khu vực phụ trách**: `src/lbd/evaluation/`, `configs/evaluation.yaml`, `notebooks/05_evaluation.ipynb`, `scripts/run_smoke_test.py`.
- **Nhiệm vụ cốt lõi**:
  1. **5-Tier Ground Truth Benchmark**: Khởi tạo tập ground truth đối chứng độc lập (FDA, ClinicalTrials.gov, repoDB, TDC, RePairDB):
     - **Hard Positives**: Suy tim (`DOID:9333`), Bệnh thận mạn (`DOID:784`).
     - **Historical**: Đái tháo đường T2D (`DOID:9352`) $\to$ Kiểm tra xem thuật toán có lọc sạch không.
     - **Soft Positives**: NAFLD/NASH (`DOID:0080358`).
     - **True Negatives**: Các thử nghiệm thất bại do độc tính hoặc thiếu hiệu quả.
  2. **Evaluation Metrics**: Tính $Recall@5$, $Recall@10$, và $MRR$ (Mean Reciprocal Rank).
  3. **Ablation Study**: Chạy thực nghiệm bóc tách 4 kịch bản:
     - Bỏ DirectionFit (coi mọi cạnh đều dương).
     - Bỏ phạt mâu thuẫn (Contradiction Penalty).
     - Chỉ dùng Shortest Path thay vì Metapath DFS.
     - Chỉ dùng NLP không dùng đồ thị Hetionet.
- **Sản phẩm bàn giao**: Báo cáo thực nghiệm `artifacts/evaluation_report.json` và bảng phân tích sai số (Error Analysis).

---

## 3. Những Nội Dung Bắt Buộc Phải Đọc & Nắm Vững

Mọi thành viên trước khi code bắt buộc phải đọc kỹ các tài liệu sau trong thư mục `docs/`:
1. [**`docs/ĐẲNG CẤP GIẢI PHÁP CHO AI.pdf`**](file:///c:/Users/Admin/Downloads/t%C3%A1i%20%C4%91%E1%BB%8Bnh%20v%E1%BB%8B%20thu%E1%BB%91c/dapagliflozin-lbd/docs/%C4%90%E1%BA%B2NG%20C%E1%BA%A4P%20GI%E1%BA%A2I%20PH%C3%81P%20CHO%20AI.pdf): Đề cương đề tài gốc, nắm vững 5 khối chức năng và lý do chọn Dapagliflozin.
2. [**`docs/architecture.md`**](file:///c:/Users/Admin/Downloads/t%C3%A1i%20%C4%91%E1%BB%8Bnh%20v%E1%BB%8B%20thu%E1%BB%91c/dapagliflozin-lbd/docs/architecture.md): Nguyên tắc Offline Research Pipeline & Online Query Application.
3. [**`docs/biological_rules.md`**](file:///c:/Users/Admin/Downloads/t%C3%A1i%20%C4%91%E1%BB%8Bnh%20v%E1%BB%8B%20thu%E1%BB%91c/dapagliflozin-lbd/docs/biological_rules.md): Hai đường suy luận mẫu ("Đường đi vàng" - Golden Paths):
   - **Golden Path 1 (Suy tim)**: Dapagliflozin -[INHIBITS (-1)]-> SLC5A2 -[DECREASES (-1)]-> Natriuresis -[DECREASES (-1)]-> Plasma Volume/Preload -[CAUSES (+1)]-> Heart Failure.
     $$(-1) \times (-1) \times (-1) \times (+1) = -1 \implies \text{Thuốc làm giảm tác nhân gây bệnh}$$
   - **Golden Path 2 (Bệnh thận mạn)**: Tác động qua cơ chế phản hồi cầu - ống thận (TGF) làm co tiểu động mạch đến và giảm áp lực nội cầu thận.
     $$(-1) \times (-1) \times (+1) \times (+1) \times (-1) \times (+1) = -1 \implies \text{Khớp chiều điều trị}$$
4. [**`docs/decisions/0001_temporal_cutoff_choice.md`**](file:///c:/Users/Admin/Downloads/t%C3%A1i%20%C4%91%E1%BB%8Bnh%20v%E1%BB%8B%20thu%E1%BB%91c/dapagliflozin-lbd/docs/decisions/0001_temporal_cutoff_choice.md): Tuyệt đối không nạp y văn hay dữ liệu sau ngày 31/12/2018.

---

## 4. Quy Tắc Làm Việc Trên GitHub (Git Workflow & Standards)

### A. Repository & Cấu hình Remote
- Repository URL: `https://github.com/leloanqa-sys/LBD-2026.git`
- Nhánh chính: `main` (chỉ merge qua Pull Request, không push trực tiếp).

### B. Quy chuẩn đặt tên nhánh (Branching Convention)
Mỗi thành viên khi làm việc phải tạo nhánh riêng từ `main`:
- `feat/data-pubmed-scraper` (Huy)
- `feat/nlp-groq-extractor` (Khang)
- `feat/graph-metapath-dfs` (Liam)
- `feat/eval-backtesting-metrics` (Phước)
- `fix/contract-validation-bug`

### C. Quy chuẩn Commit Message (Conventional Commits)
- `feat: integrate pubtator 3.0 batch query`
- `fix: correct sign propagation for DECREASES edge`
- `docs: update RACI matrix in team_design_guide.md`
- `test: add contract validation test for paper schema`

### D. Quy định nghiêm ngặt về Dữ liệu & Bảo mật
1. **KHÔNG commit API Keys**: Tuyệt đối không commit file `.env` chứa `GROQ_API_KEY` hay `NCBI_API_KEY`. Sử dụng file `.env.example` làm mẫu.
2. **KHÔNG commit Raw Data quá 50MB**: Thư mục `data/raw/` và file nhị phân lớn (`.parquet`, `.pt`, `.pkl`) đã được chặn trong `.gitignore`. Chỉ commit các file mẫu nhỏ trong `data/samples/`.
3. **Validation Gate trước khi tạo PR**:
   Trước khi push code và tạo Pull Request lên GitHub, thành viên bắt buộc phải chạy và pass 2 lệnh sau trên máy cá nhân:
   ```bash
   pytest tests/
   python scripts/run_smoke_test.py
   ```
