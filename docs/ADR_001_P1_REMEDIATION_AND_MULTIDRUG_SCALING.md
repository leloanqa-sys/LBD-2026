# ADR-001: Cải cách P1 (Module-Boundary Remediation `v3.1-p1-sentence-directed`) & Đánh giá Kiến trúc Mở rộng Đa thuốc (Multi-Drug LBD Scaling)

- **Trạng thái:** Đã triển khai & Kiểm chứng (`16/16` unit tests pass, đã trích xuất lại toàn bộ `664` bài báo PubMed)
- **Ngày cập nhật:** 2026-10-28
- **Phiên bản Schema NLP:** `v3.1-p1-sentence-directed`

---

## 1. Bối cảnh & Lý do Sửa đổi Quyết định Kiến trúc (Decision Log)

Báo cáo kiểm toán độc lập trên `2.210` quan hệ đầu ra của phiên bản `v3.0-biored-zoned-mlp-ensemble` chỉ ra rằng dù mô hình học máy V3 không có lỗi cơ học, chất lượng đầu ra thực tế bị suy giảm nghiêm trọng do **đặt sai bài toán tại điểm nối giữa các module** (`bio_re_engine_v3.py` $\leftrightarrow$ `relation_extraction.py` $\leftrightarrow$ `postprocess.py` $\leftrightarrow$ `normalize.py`).

Bảng dưới đây ghi nhận các **Quyết định Kiến trúc cũ (Deprecated Decisions)**, **Quyết định Kiến trúc mới trong P1 (New Decisions)** và **Kết quả đo lường Trước / Sau trên 664 bài báo PubMed**:

| # | Hạng mục Kiến trúc | Quyết định cũ (`v3.0` - Bị bác bỏ) | Quyết định mới (`v3.1-p1` - Đã áp dụng) | Số liệu Trước P1 (`N=2210`) | Số liệu Sau P1 (`N=2944`) |
|---|---|---|---|---|---|
| **D1** | **Đơn vị trích xuất & Bằng chứng (`evidence`)** | Gom toàn bộ abstract thành 1 mẫu document-level; thiếu khóa `evidence` trong `meta_list` dẫn đến fallback `abstract[:200]`. | Tách câu (`split_doc_sentences`), trích xuất theo từng câu có đồng xuất hiện thực thể (`sentence-level mini-docs` gom batch theo bài báo) và lưu đúng câu bằng chứng vào `meta["evidence"]`. | `100%` dòng dài đúng 200 ký tự (`2210/2210`); `0/596` PMID có >1 câu bằng chứng. | Chỉ `0,71%` (`21/2944`) ngẫu nhiên dài 200 ký tự; **`80,6%` (`465/577`) PMID có nhiều câu bằng chứng khác biệt**. |
| **D2** | **Chiều quan hệ (`source` $\rightarrow$ `target`)** | Sắp xếp theo thứ tự từ điển của chuỗi ID (`canon_pair`: `source_id <= target_id`), khiến `6524` (`SLC5A2`) luôn đứng trước `DB06292` (`Dapagliflozin`). | Áp dụng phân cấp chiều sinh học chuẩn (`TYPE_HIERARCHY_RANK`: `Compound (0) -> DrugClass (1) -> Gene (2) -> BiologicalProcess (3) -> Disease (4)`). Với cặp cùng bậc, xét vị trí chủ ngữ/tân ngữ và cấu trúc bị động (`by @ENTITY`). | `100%` dòng có `source_id <= target_id`; `464` dòng `Gene -> Compound`, `0` dòng `Compound -> Gene`. | **`222` dòng `Compound -> Gene`, `66` dòng `DrugClass -> Gene`, `0` dòng `Gene -> Compound`**. |
| **D3** | **Ánh xạ Polarity (`+1/-1/0`) sang Động từ Lâm sàng (`relation_type`)** | Dùng `if-else` cứng ánh xạ mọi nhãn `POSITIVE (+1)` của cặp `Compound-Disease` thành `CAUSES`, mọi `NEGATIVE (-1)` thành `TREATS`. | **Tách biệt Dấu tương quan (`polarity`) và Nhãn động từ (`relation_type`)**: Nhãn động từ mạnh (`TREATS`, `CAUSES`, `INHIBITS`, `ACTIVATES`, `DECREASES`, `INCREASES`) chỉ được gán khi câu bằng chứng có động từ/cấu trúc ngữ법 tương ứng + kiểm tra phủ định (`not/no/without`) + lọc câu mô tả quần thể nền (`"in patients with T2D..."`). Nếu không đủ bằng chứng động từ trực tiếp, hạ về `ASSOCIATED_WITH` nhưng vẫn bảo toàn vector xác suất `[P(+), P(-), P(0)]`. | `Dapa -> T2D` bị gán `CAUSES` tới `30,2%` (`214/709`); `CAUSES` đúng `0/278`. | `Dapa -> T2D` gán `CAUSES` giảm xuống **`0,97%` (`7/718`)** (`TREATS`: `307`, `ASSOCIATED_WITH`: `404`). |
| **D4** | **Xử lý Thực thể `BiologicalProcess` (GO Terms)** | Giả dạng `BiologicalProcess` thành `Disease` để lách bộ lọc `VALID_BIORED_PAIR_TYPES`, làm loại bỏ toàn bộ cạnh nối tiếp `BiologicalProcess -> Disease` ($B \to C$). | Không đẩy cặp chứa `BiologicalProcess` qua bộ phân loại BioRED (vốn không có lớp GO). Phân tích trực tiếp các cặp `(Compound/Gene, BiologicalProcess)` và `(BiologicalProcess, Disease)` bằng bộ phân tích cú pháp – hướng – dấu câu (`_score_process_sentence_pair`). | `0` cạnh có nguồn là `BiologicalProcess`; đứt gãy hoàn toàn cầu nối $A \to B \to C$. | Khôi phục **`206` cạnh `BiologicalProcess -> Disease`** và **`70` cạnh `BiologicalProcess -> BiologicalProcess`**. |
| **D5** | **Phân tách Lớp thuốc (`DrugClass`) vs. Gene đích (`SLC5A2`)** | Gán chung chuỗi `"SGLT2 inhibitors"` vào `entity_id = "6524"` (`Gene: SLC5A2`). | Bổ sung thực thể `CLASS:SGLT2i` (`type = "DrugClass"`) trong `ids.py` và bộ hậu xử lý ngữ cảnh trong `ner.py` (tự động nâng cấp span `SGLT2` có hậu tố `inhibitor(s)` kể cả dạng viết tắt trong ngoặc `"sodium-glucose cotransporter 2 (SGLT2) inhibitors"` thành `CLASS:SGLT2i`). | `53,5%` (`1183/2210`) quan hệ có nguồn là `Gene` (đa số là lớp thuốc bị nhận nhầm). | Tỷ lệ dòng có `Gene` làm nguồn giảm còn **`7,2%` (`212/2944`)**; tách riêng **`814` dòng có nguồn là `DrugClass` (`CLASS:SGLT2i`)**. |
| **D6** | **Phân tách mã bệnh `DOID:9744` & Cờ kiểm định (`verbatim` / `normalization`)** | Gán trùng `Diabetic Ketoacidosis` và `Type 1 Diabetes` vào `DOID:9744`. Cờ `evidence_verbatim_status` và `normalization_status` luôn trả về `"verified"` (`100%`). | Tách `Diabetic Ketoacidosis` sang mã chuẩn `DOID:0050155`. `verify_evidence_verbatim` kiểm tra độ dài tối thiểu, từ chối chuỗi cắt cụt 200 ký tự và kiểm tra sự hiện diện của thực thể trong câu. `normalize_relations` phân loại 3 trạng thái thực chất (`verified`, `ambiguous` cho `CLASS:SGLT2i`, `unresolved`) và chuẩn hóa tên thực thể chuẩn. | `100%` dòng `verified` giả tạo; file `normalized` giống hệt file `extracted` (`0` dòng thay đổi). | `DOID:9744` chỉ còn đúng `Type 1 Diabetes` (`67` dòng); `normalization_status` phân tách **`2130 verified` / `814 ambiguous`**; **`2179/2944` dòng được chuẩn hóa tên thực thể chuẩn**. |
| **D7** | **Cô lập Luật Sinh lý Đặc thù Miền (`allow_domain_rules`)** | Nhúng cứng luật sinh lý HF/CKD và bơm thực thể ảo vào mọi chế độ chạy, gây nguy cơ rò rỉ tri thức (data leakage) khi backtest. | Bổ sung tham số `allow_domain_rules: bool = True` (mặc định tắt `allow_domain_rules=False` khi chạy kiểm định `strict_backtest`), vô hiệu hóa hoàn toàn cả luật sinh lý đặc thù lẫn bơm thực thể ảo (`virtual context anchors`). | Không thể tắt luật miền khi chạy đánh giá độc lập. | Đã kiểm chứng qua unit test `test_p1_domain_rule_isolation`: `allow_domain_rules=False` chặn 100% suy diễn ngầm. |

---

## 2. Chi tiết Các Thay đổi Mã nguồn trong Bản Cập nhật `v3.1-p1`

1. [`src/lbd/common/ids.py`](file:///c:/Users/Admin/Downloads/t%C3%A1i%20%C4%91%E1%BB%8Bnh%20v%E1%BB%8B%20thu%E1%BB%91c/dapagliflozin-lbd/src/lbd/common/ids.py):
   - Tách `Diabetic Ketoacidosis` sang `DOID:0050155`, giữ `DOID:9744` độc quyền cho `Type 1 Diabetes`.
   - Thêm thực thể `CLASS:SGLT2i` (`type: "DrugClass"`, tên chuẩn `"SGLT2 Inhibitors"`) và `DB09038` (`Empagliflozin`).
2. [`src/lbd/nlp/ner.py`](file:///c:/Users/Admin/Downloads/t%C3%A1i%20%C4%91%E1%BB%8Bnh%20v%E1%BB%8B%20thu%E1%BB%91c/dapagliflozin-lbd/src/lbd/nlp/ner.py):
   - Bổ sung bước hậu xử lý sau `resolve_overlaps`: nếu một span `Gene` (`6524`) được theo sau bởi `inhibitor/inhibitors/inhibition` (hoặc `)... inhibitors`), tự động nâng cấp span đó thành `DrugClass` (`CLASS:SGLT2i`) và mở rộng `end` bao trùm toàn bộ cụm danh từ.
3. [`src/lbd/nlp/bio_re_engine_v3.py`](file:///c:/Users/Admin/Downloads/t%C3%A1i%20%C4%91%E1%BB%8Bnh%20v%E1%BB%8B%20thu%E1%BB%91c/dapagliflozin-lbd/src/lbd/nlp/bio_re_engine_v3.py):
   - Sửa lỗi lệch 1 ký tự offset khi nối `title` + `abstract` qua tham số `title_sep_len`.
   - Trả về đúng câu `best_evidence` vào `meta_list["evidence"]`.
   - Sửa `predict_untrained` dùng một ma trận chiếu ngẫu nhiên cố định $W \in \mathbb{R}^{d \times 4}$ (bất biến với thứ tự dòng) và thêm `_align_proba_4class` chống lỗi thiếu lớp khi tập train nhỏ.
4. [`src/lbd/nlp/relation_extraction.py`](file:///c:/Users/Admin/Downloads/t%C3%A1i%20%C4%91%E1%BB%8Bnh%20v%E1%BB%8B%20thu%E1%BB%91c/dapagliflozin-lbd/src/lbd/nlp/relation_extraction.py):
   - Thiết kế lại `extract_relations_v3` chạy ở cấp độ câu (`v3.1-p1-sentence-directed`), định hướng theo `TYPE_HIERARCHY_RANK`, lọc câu mô tả bệnh nền của bệnh nhân (`_is_cohort_context_only`), tách riêng nhánh phân tích `BiologicalProcess`, tích hợp cổng kiểm tra động từ trước khi gán nhãn mạnh, và hỗ trợ `GroqRelationVerifier` (Tier-2 LLM có giới hạn `max_calls=50` + SHA-256 disk cache).
5. [`src/lbd/nlp/postprocess.py`](file:///c:/Users/Admin/Downloads/t%C3%A1i%20%C4%91%E1%BB%8Bnh%20v%E1%BB%8B%20thu%E1%BB%91c/dapagliflozin-lbd/src/lbd/nlp/postprocess.py) & [`src/lbd/data/normalize.py`](file:///c:/Users/Admin/Downloads/t%C3%A1i%20%C4%91%E1%BB%8Bnh%20v%E1%BB%8B%20thu%E1%BB%91c/dapagliflozin-lbd/src/lbd/data/normalize.py):
   - Kiểm định nghiêm ngặt `evidence_verbatim_status` và `normalization_status` 3 trạng thái (`verified`, `ambiguous`, `unresolved`).
6. [`src/lbd/data/load_hetionet.py`](file:///c:/Users/Admin/Downloads/t%C3%A1i%20%C4%91%E1%BB%8Bnh%20v%E1%BB%8B%20thu%E1%BB%91c/dapagliflozin-lbd/src/lbd/data/load_hetionet.py):
   - Cô lập việc bơm hạt giống `Gene::6524`, `Gene::6523` và 10 GO terms thận/tim mạch vào đúng điều kiện `if clean_drug_id == "DB06292":`, ngăn chặn lỗi tiêm nhầm đích SGLT2 cho các thuốc khác khi mở rộng đa thuốc.

---

## 3. Kiểm toán Các Phân hệ Còn lại (`data`, `graph`, `ranking`, `evaluation`) & Đánh giá Khả năng Mở rộng Đa Thuốc (Multi-Drug Scaling)

Khi mở rộng hệ thống từ bài toán đơn thuốc (Dapagliflozin) sang **Nền tảng Tái định vị Đa thuốc (Multi-Drug LBD Platform)**, việc sửa riêng phân hệ NLP là chưa đủ. Kiểm toán mã nguồn ở 4 phân hệ còn lại phát hiện **8 điểm nghẽn thuật toán và đồ thị** cần giải quyết trong giai đoạn P2/P3:

### 3.1. Phân hệ Nạp Tri thức Nền (`src/lbd/data/load_hetionet.py`)
1. **Đảo chiều cạnh biểu hiện gene làm sai lệch dấu nhân quả (`DuG`/`DdG` $\rightarrow$ `GuD`/`GdD`) — *Mức độ: Nghiêm trọng***
   - **Hiện trạng ([`load_hetionet.py#L22-L26`](file:///c:/Users/Admin/Downloads/t%C3%A1i%20%C4%91%E1%BB%8Bnh%20v%E1%BB%8B%20thu%E1%BB%91c/dapagliflozin-lbd/src/lbd/data/load_hetionet.py#L22-L26)):** Cạnh Hetionet `Disease --[DuG (+1)]--> Gene` (bệnh làm tăng biểu hiện gene $G$) đang bị đảo chiều cơ học thành `Gene --[GuD (+1)]--> Disease` với giả định rằng gene $G$ thúc đẩy bệnh.
   - **Lỗ hổng sinh học:** Trong sinh học bệnh lý, rất nhiều gene tăng biểu hiện khi bị bệnh (`DuG`) là **phản ứng bù trừ bảo vệ cơ thể** (ví dụ: BNP/`NPPB` tăng cao trong Suy tim để lợi tiểu bảo vệ tim), không phải nguyên nhân gây bệnh! Nếu đảo thành `Gene --(+1)--> Disease`, một loại thuốc ức chế (`-1`) gene bảo vệ đó sẽ có tích dấu $(-1) \times (+1) = -1$ (được chấm điểm là "điều trị tốt"), trong khi thực tế lâm sàng nó làm bệnh nặng thêm.
   - **Giải pháp cho Multi-Drug:** Không đảo chiều trực tiếp `DuG/DdG` thành cạnh nhân quả hạng 1; chỉ cho phép đi qua cạnh biểu hiện gene khi có bằng chứng hướng từ GWAS/OpenTargets hoặc gán trọng số tin cậy thấp hơn (`w_metaedge = 0.35`) kèm cờ `is_expression_correlation = True`.
2. **Danh sách Gene Dược động học (ADME) gán cứng chỉ có 11 gene ([`load_hetionet.py#L140`](file:///c:/Users/Admin/Downloads/t%C3%A1i%20%C4%91%E1%BB%8Bnh%20v%E1%BB%8B%20thu%E1%BB%91c/dapagliflozin-lbd/src/lbd/data/load_hetionet.py#L140)) — *Mức độ: Cao***
   - **Hiện trạng:** `adme_genes = {"1543", "1544", "1559", "1576", "1577", "1565", "54658", "54578", "54600", "5243", "9429"}` chỉ lọc được một vài enzyme CYP/UGT của Dapagliflozin.
   - **Rủi ro khi mở rộng đa thuốc:** Hàng trăm thuốc khác là cơ chất (substrate) của các transporter họ `SLC22A*` (OAT/OCT), `SLCO1B1` (OATP), `ABCC*` (MRP) hoặc các CYP khác. Nếu không lọc đầy đủ, đồ thị sẽ nối mọi thuốc dùng chung enzyme chuyển hóa gan/thận vào chung một đường đi giả tạo (hub-node explosion).
   - **Giải pháp:** Tải danh sách chuẩn ~300 gene ADME (PharmaADME) hoặc phân biệt rõ vai trò `pharmacological_target` vs `enzyme/transporter/carrier`.

### 3.2. Phân hệ Xây dựng Đồ thị Tri thức (`src/lbd/graph/build_graph.py`)
3. **Đa cạnh song song trái dấu trong `nx.MultiDiGraph` không được hợp nhất theo đa số — *Mức độ: Cao***
   - **Hiện trạng ([`build_graph.py#L114-L163`](file:///c:/Users/Admin/Downloads/t%C3%A1i%20%C4%91%E1%BB%8Bnh%20v%E1%BB%8B%20thu%E1%BB%91c/dapagliflozin-lbd/src/lbd/graph/build_graph.py#L114-L163)):** Các cạnh NLP được gom nhóm theo khóa `(src, tgt, rel_type, assertion)`. Do đó, nếu giữa cùng cặp `(u, v)` có 15 bài báo nói `DECREASES (-1)` và 1 bài báo nhiễu nói `INCREASES (+1)`, đồ thị vẫn giữ song song cả 2 cạnh `e_-` và `e_+`.
4. **Sổ cái bằng chứng phản bác (`contradiction_pmids`) chỉ nhìn 1-hop trực tiếp — *Mức độ: Trung bình***
   - **Hiện trạng ([`build_graph.py#L121-L133`](file:///c:/Users/Admin/Downloads/t%C3%A1i%20%C4%91%E1%BB%8Bnh%20v%E1%BB%8B%20thu%E1%BB%91c/dapagliflozin-lbd/src/lbd/graph/build_graph.py#L121-L133)):** Chỉ ghi nhận cạnh trực tiếp `Compound --[CAUSES]--> Disease` hoặc `negated TREATS` vào `contradiction_pmids`. Các đường đi cơ chế 2–3 bước có dấu tổng hợp `+1` (thuốc kích hoạt cơ chế gây bệnh) chưa được đưa vào bộ phạt phản bác.

### 3.3. Phân hệ Tìm kiếm Đường đi (`src/lbd/graph/path_search.py`)
5. **Thuật toán DFS lặp lại $O(|D|)$ lần từ cùng một nút nguồn — *Mức độ: Cao khi mở rộng đa thuốc***
   - **Hiện trạng ([`path_search.py#L119-L143`](file:///c:/Users/Admin/Downloads/t%C3%A1i%20%C4%91%E1%BB%8Bnh%20v%E1%BB%8B%20thu%E1%BB%91c/dapagliflozin-lbd/src/lbd/graph/path_search.py#L119-L143)):** `discover_all_disease_paths` lặp qua từng bệnh $d \in D$ và gọi lại `find_metapaths_abc(G, source_node, d_node)` từ đầu. Với 1 thuốc $\times$ 100 bệnh, cây DFS độ sâu 3 từ `source_node` bị duyệt lặp lại 100 lần. Khi mở rộng lên 500 thuốc $\times$ 500 bệnh ($250.000$ cặp), chi phí tính toán tăng gấp hàng trăm lần không cần thiết.
   - **Giải pháp:** Thay bằng **Single-Pass Rooted Bounded DFS** xuất phát 1 lần duy nhất từ `source_node` (độ sâu $\le 3$) và phân loại mọi đường đi chạm tới bất kỳ nút `Disease` nào vào `paths_by_disease[target_id]`. Tốc độ tăng gấp $50\times - 100\times$.
6. **Gán cứng bệnh lịch sử `HISTORICAL_DISEASES = {"DOID:9352"}` ([`path_search.py#L14`](file:///c:/Users/Admin/Downloads/t%C3%A1i%20%C4%91%E1%BB%8Bnh%20v%E1%BB%8B%20thu%E1%BB%91c/dapagliflozin-lbd/src/lbd/graph/path_search.py#L14)) — *Mức độ: Nghiêm trọng khi mở rộng đa thuốc***
   - **Hiện trạng:** Mọi loại thuốc đều bị chặn chung một bệnh duy nhất là Tiểu đường tuýp 2 (`DOID:9352`).
   - **Giải pháp:** Chuyển thành bảng tra cứu theo từng thuốc `historical_indications_by_drug[drug_id]` (lấy từ cạnh `CtD` của Hetionet hoặc nhãn FDA trước năm cutoff $t_0$).

### 3.4. Phân hệ Chấm điểm & Xếp hạng (`src/lbd/ranking/`)
7. **Lọc bỏ âm thầm đường đi `+1` kết hợp khử trùng lặp chuỗi nút gây khuếch đại nhiễu thiểu số — *Mức độ: Nghiêm trọng***
   - **Hiện trạng ([`rank_candidates.py#L85-L92`](file:///c:/Users/Admin/Downloads/t%C3%A1i%20%C4%91%E1%BB%8Bnh%20v%E1%BB%8B%20thu%E1%BB%91c/dapagliflozin-lbd/src/lbd/ranking/rank_candidates.py#L85-L92)):**
     ```python
     if p.get("net_sign") == 1:
         continue
     ```
     Giả sử trên chuỗi nút $A \to B \to C$, có 10 bài báo xác nhận $B$ làm nặng bệnh $C$ (`net_sign = +1`, đường đi có hại) và 1 bài báo trích xuất nhầm thành $B$ giảm bệnh $C$ (`net_sign = -1`). Khi chạy `rank_candidates`, đường đi `+1` (10 bài báo) bị `continue` bỏ qua âm thầm, còn đường đi `-1` (1 bài báo nhiễu) lại được giữ lại và cộng điểm dương vào `therapeutic_paths`!
   - **Giải pháp:** Phải hợp nhất dấu ở cấp độ cặp nút `(u, v)` trước khi chấm điểm đường đi (tính $\text{net\_polarity}(u,v) = \frac{W_- - W_+}{W_- + W_+ + W_0}$), hoặc trừ trực tiếp trọng số của các đường đi `net_sign == +1` vào điểm tổng hợp của bệnh ứng viên.
8. **Bẫy triệt tiêu dấu `0` (`Zero-Annihilation Trap`) & Bỏ qua `confidence_score` của NLP — *Mức độ: Cao***
   - **Hiện trạng ([`score_paths.py#L22-L23`](file:///c:/Users/Admin/Downloads/t%C3%A1i%20%C4%91%E1%BB%8Bnh%20v%E1%BB%8B%20thu%E1%BB%91c/dapagliflozin-lbd/src/lbd/ranking/score_paths.py#L22-L23) & [`features.py#L51-L65`](file:///c:/Users/Admin/Downloads/t%C3%A1i%20%C4%91%E1%BB%8Bnh%20v%E1%BB%8B%20thu%E1%BB%91c/dapagliflozin-lbd/src/lbd/ranking/features.py#L51-L65)):**
     - Mọi cạnh nền Hetionet (`GpBP`, `DaG`, `CbG` không rõ hướng) đều có `sign = 0`. Hàm `calculate_net_sign` quy định `if 0 in signs: return 0`. Hệ quả là mọi đường đi 2–3 bước có chạm vào dù chỉ 1 cạnh Hetionet không dấu đều bị归 về `net_sign = 0`, làm mất hoàn toàn thông tin hướng ở các bước còn lại.
     - Hàm `edge_weight` trong `features.py` chỉ nhân `w_tier * w_study * sat_factor(num_pmids)` mà **bỏ qua hoàn toàn `confidence_score` và `soft_sign`** của mô hình NLP.
   - **Giải pháp:** Sử dụng vector xác suất mềm $[P(+), P(-), P(0)]$ của V3 để truyền dấu xác suất qua đường đi (Probabilistic Sign Propagation) và nhân `mean(confidence_score)` vào `edge_weight`.
