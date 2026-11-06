# Script nào gọi phần nào?

Chạy từ thư mục `dapagliflozin-lbd`. Đây là bản đồ chức năng đang có, không phải runner tuần tự mới.

## 1. Dữ liệu

`src/lbd/data/fetch_pubmed.py` có `create_document_record()` và `filter_by_cutoff()`, chưa là collector canonical hoàn chỉnh. Tải PubMed live có trong script demo, nhưng script đó có graph giả và không phù hợp làm lệnh nghiên cứu chính.

## 2. NLP canonical

```powershell
python scripts/run_nlp_extraction.py
```

Điểm gọi: `run()` → `NLPEngine.load_default()` → `extract_relations_from_text()` → `normalize_relation_record()`.

Input: CSV trong `data/raw/`, fallback sang `../nlp/`. Output: `data/processed/relations_extracted.jsonl`, `relations_normalized.jsonl` và review CSV. Có thể ghi đè output cũ. Chưa tự chạy bước merge KG.

## 3. KG

`src/lbd/graph/build_graph.py:create_knowledge_graph()` nhận snapshot nền và quan hệ NLP. `src/lbd/data/load_hetionet.py:load_hetionet_subgraph()` yêu cầu `graph_path`; hiện tải snapshot được truyền vào, chưa tự trích graph theo thuốc. Chưa có lệnh canonical nối output bước 2 đến snapshot KG mới.

## 4. Thuật toán và ranking

```powershell
python scripts/run_original_graph_dfs_and_ranking.py
python scripts/run_lbd_m2_ranking.py --drugs Dapagliflozin --top-k 5
```

Lệnh đầu: `search_metapaths()` + `score_individual_path()` + `score_disease_candidate()`, đọc `data/processed/merged_graph.json`, ghi graph reports trong `artifacts/`.

Lệnh sau: `M2RepurposingRanker.rank_drug()`, đọc model/feature tại `artifacts/ranking_m2/` và graph raw. Hai tuyến dùng cách tính điểm khác nhau; M2 chưa lấy feature từ KG NLP vừa tạo. Dùng `--output <đường_dẫn.json>` khi muốn lưu JSON của CLI M2.

## 5. Đánh giá ground truth

```powershell
python scripts/evaluate_gold_ground_truth.py --no-llm
```

Đọc gold mặc định trong `data/samples/`; đây là đánh giá NLP với entity spans/IDs cho trước. `--no-llm` tắt gọi live, không đảm bảo bỏ cache LLM cũ. Không dùng metric này thay cho nghiệm thu NER, KG hoặc ranking toàn hệ thống.

Metric ranking hiện có tại `src/lbd/evaluation/run_experiment.py:evaluate_rankings()`. Báo cáo M2 có trong `artifacts/ranking_m2/`; chưa có CLI thống nhất nhận run hiện tại và mọi loại gold.

## 6. Giao diện

```powershell
python ui/serve.py
```

Phục vụ `ui/index.html`; giao diện hiện còn dữ liệu viết sẵn, chưa đọc kết quả ranking mới.

## 7. Kiểm tra code

```powershell
python -m pytest tests -q
python scripts/run_smoke_test.py
```

Smoke dùng fixture có nhãn; không đo độ chính xác y văn. Trong audit 03/11/2026, Python hệ thống chạy đạt 34 test; `.venv` bên ngoài chưa có pytest.

## 8. Các file nghiên cứu và tương thích

`run_nlp_v3_extraction.py`, `run_nlp_benchmark.py`, `run_v6_end_to_end_benchmark.py` phục vụ những tuyến/phiên bản khác. `demo_end_to_end_repurposing.py` phục vụ trình diễn có dữ liệu giả. Không chạy chúng thay thế canonical engine mà bỏ qua sự khác biệt cấu hình và nguồn dữ liệu.
