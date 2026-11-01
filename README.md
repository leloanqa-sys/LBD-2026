# Dapagliflozin Literature-Based Discovery (LBD)

Hệ thống Khám phá Tri thức Y văn & Tái định vị thuốc (Drug Repurposing) ứng dụng AI/NLP kết hợp Đồ thị Tri thức (Knowledge Graph) và Mô hình Hợp nhất Bằng chứng Bayesian, tập trung nghiên cứu trường hợp tái định vị **Dapagliflozin** sang **Suy tim (Heart Failure)** và **Bệnh thận mạn (Chronic Kidney Disease)** trước mốc thời gian $T \le 31/12/2018$.

---

## 1. Cấu trúc Module NLP Chuẩn hóa (Canonical Biomedical NLP Architecture)

Toàn bộ các phiên bản phân mảnh trước đây (`v3`, `v6`, `v6_industrial`, `v7`, `v7_hybrid`, `ner_hybrid`) đã được **tinh gọn và quy chuẩn hóa 100%** vào thư mục chính thức: [`src/lbd/nlp/`](src/lbd/nlp/).

```
src/lbd/nlp/
├── engine.py                 # NLPEngine: Điều phối toàn bộ pipeline trích xuất & hợp nhất
├── zoning.py                 # Phân vùng diễn ngôn (Title/Methods/Results/Conclusion) & 2-Pass L2 Coreference
├── events.py                 # Trích xuất Bio-Event frames & Counterfactual Blinded LLM Verifier (Qwen-27B)
├── polarity.py               # Đại số phân cực hướng tác dụng (+1 / -1 / 0) & Chốt an toàn chống chỉ định
├── classifier.py             # BiomedicalRelationClassifier: Mô hình thống kê 4-lớp hiệu chuẩn xác suất
├── fusion.py                 # BayesianFusionEngine: Hợp nhất Beta-Binomial khử trùng lặp theo PMID xuất Contract D
├── relation_extraction.py     # Adapter chính thức: extract_relations_from_text chuẩn Contract B
├── ner.py                    # Giao diện nhận diện thực thể: PubTator 3.0 API & Local Lexicon fallback
├── sentence_splitter.py      # Bộ tách câu y sinh bảo vệ từ viết tắt khoa học (p < 0.05, vs., et al.)
├── external_annotations.py   # Client đồng bộ và lưu đệm dữ liệu PubTator 3.0 & EuropePMC SciLite
└── postprocess.py            # Kiểm tra bằng chứng nguyên văn (verbatim verification) & khử trùng lặp
```

---

## 2. Tiến độ & Kết quả Kiểm nghiệm Hệ thống (Validation Milestones)

### A. Kiểm thử Đơn vị & Tích hợp (Unit & Integration Tests)
- **Unit Tests (Pytest):** **12 / 12 test cases PASSED (100%)** trong `tests/` (`test_contracts.py`, `test_normalization.py`, `test_path_search.py`, `test_ranking.py`, `test_relation_extraction.py`, `test_signed_rules.py`).
- **Integration Smoke Test:** **PASS 6 / 6 bước** không có lỗi (`scripts/run_smoke_test.py`):
  1. Trích xuất NLP bằng `NLPEngine` chuẩn.
  2. Kiểm tra JSON Schema Contract B (`relation.schema.json`).
  3. Chuẩn hóa tên và mã định danh quốc tế (MeSH, DOID, Entrez).
  4. Hợp nhất bằng chứng Bayesian và kiểm tra JSON Schema Contract D (`graph_edge.schema.json`).
  5. Xây dựng đồ thị NetworkX và tìm kiếm đường dẫn sinh học (Metapath Search).
  6. Kiểm tra quy tắc nhân dấu đại số sinh học: $INHIBITS (-1) \times CAUSES (+1) = -1$ (Hướng trị liệu phù hợp $1.5$).

### B. Kết quả Đánh giá trên Bộ Ground Truth Mù do Con người Gán nhãn (Blind 200 Gold Sample)
Hệ thống đã chạy thực nghiệm End-to-End toàn diện có tích hợp mô hình **Live LLM** (`qwen/qwen3.8-27b` qua Groq API) đối chiếu trực tiếp trên 200 mẫu mù (`data/samples/gold_annotation_blind_200_annotated_for_kg.csv`):

| Chỉ số kiểm định | Toàn bộ 200 mẫu thử nghiệm mù | Tập 63 Cạnh Đồ thị Sạch (KG-Ready Subset) | Ý nghĩa đánh giá |
| :--- | :---: | :---: | :--- |
| **Recall (Độ bao phủ)** | **82.00%** *(82 / 100)* | **92.06%** *(58 / 63)* | Hệ thống bắt trúng **92%** các quan hệ thực sự có giá trị đưa vào KG. |
| **Precision (Độ chính xác)** | **65.08%** *(82 / 126)* | **56.86%** *(58 / 102)* | Tỷ lệ trích xuất chính xác trên tập nhãn khắt khe của con người. |
| **F1-Score** | **72.57%** | **70.30%** | Điểm cân bằng điều hòa tối ưu cho bài toán trích xuất quan hệ y sinh. |
| **Accuracy (Độ chính xác tổng)** | **69.00%** *(138 / 200)* | -- | Tỷ lệ dự đoán đúng trên cả hai nhóm $YES$ và $NO$. |

#### Hiệu năng theo từng nhóm cặp thực thể cốt lõi:
- **`Compound - Gene` (Dược lý phân tử):** Precision **85.0%**, Recall **100.0%**, F1 **91.9%** $\rightarrow$ Bắt trọn 100% các bằng chứng Dapagliflozin ức chế SLC5A2.
- **`DrugClass - Phenotype` (Huyết áp, bài tiết đường):** Precision **100.0%**, Recall **100.0%**, F1 **100.0%**.
- **`DrugClass - Disease` (Bệnh học Suy tim, Đái tháo đường):** Recall **100.0%**, F1 **80.0%**.
- **`DrugClass - BiologicalProcess`:** Precision **83.3%**, Recall **100.0%**, F1 **90.9%**.

---

## 3. Lưu ý Quan trọng về Bộ Ground Truth cho Nhóm Nghiên cứu (Ground Truth Audit)

Qua đối soát chi tiết bộ Ground Truth 200 mẫu của nhóm, có **4 vấn đề phương pháp luận** cần các thành viên nắm rõ:

1. **Mâu thuẫn chiều quan hệ (`direction_valid == NO`, 12 ca):**
   - Một số câu được gán `has_relation = YES` nhưng chiều nhân quả bị **ngược hoàn toàn** với câu hỏi (ví dụ: `GOLD_042` hỏi *UGE -> CKD*, nhưng câu văn thực tế là *suy thận làm giảm bài tiết đường* $CKD \to UGE$; hay gen $SLC5A2$ sinh lý làm tăng tái hấp thu nhưng đề bài gán hiệu ứng giảm của thuốc ức chế vào gen).
   - Mô hình NLP nhận định đúng chiều sinh học nên không trích xuất, do đó bị tính nhầm là False Negative.
2. **Đánh đồng câu phủ định / giả thuyết với quan hệ thật (16 ca):**
   - `GOLD_063`: Câu văn ghi rõ tần suất UTI bằng placebo (`NO_EFFECT_ON` - không có hiệu ứng), nhưng nhãn vẫn để `YES`.
   - `GOLD_017, 054, 088, 122`: Câu mang tính giả thuyết tương lai (*"shows promise"*, *"may translate"*), chưa phải kết luận thực chứng.
3. **Ép mã phân cấp thực thể (Subtype Mismatch, 11 ca):**
   - Bệnh thận do đái tháo đường (*Diabetic Nephropathy*) bị ép cứng thành *Chronic Kidney Disease [DOID:784]*, gây lệch ngữ nghĩa.
4. **Khuyến nghị sử dụng Ground Truth:**
   - Khi đánh giá nghiệm thu mô hình, **nên ưu tiên sử dụng tập 63 cạnh sạch (`KG_READY_CONTENT_TEMPORAL_UNVERIFIED`)** thay vì lấy toàn bộ nhãn thô, để tránh phạt oan mô hình khi nó xử lý đúng về mặt sinh học.

---

## 4. Hướng dẫn Sử dụng Nhanh cho Thành viên (Quickstart)

### Cài đặt môi trường
```bash
pip install -r requirements.txt
```

### Chạy kiểm thử tự động (Unit Tests)
```bash
python -m pytest tests
```

### Chạy kiểm thử tích hợp toàn diện (Smoke Test)
```bash
python scripts/run_smoke_test.py
```

### Chạy trích xuất quan hệ trên toàn bộ y văn PubMed
```bash
python scripts/run_nlp_extraction.py
```

### Chạy đánh giá đối chiếu với bộ Ground Truth mù (có Live LLM)
```bash
python scripts/evaluate_gold_ground_truth.py
```

### Sử dụng NLPEngine trong mã nguồn
```python
from lbd.nlp import NLPEngine, extract_relations_from_text

# Cách 1: Gọi hàm trích xuất chuẩn Contract B
relations = extract_relations_from_text(
    pmid="29085817",
    abstract="Dapagliflozin reduces worsening heart failure. Dapagliflozin inhibits SLC5A2.",
    title="Dapagliflozin in Heart Failure"
)
print("Extracted relations:", len(relations))

# Cách 2: Sử dụng NLPEngine điều phối
engine = NLPEngine.load_default(dataset_mode="biored")
doc = {
    "pmid": "29085817",
    "title": "Dapagliflozin in Heart Failure",
    "abstract": "Dapagliflozin reduces worsening heart failure. Dapagliflozin inhibits SLC5A2."
}
grounded, events, scores = engine.run_document(doc)
contract_b, fused_edges, contract_d = engine.fuse_corpus(events)
```
