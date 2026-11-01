# Hướng Dẫn Gán Nhãn Ground Truth Chuẩn Mực 2026 (Biomedical Relation Gold Standard)

**Tệp dữ liệu gán nhãn:** [`dapagliflozin-lbd/data/samples/ground_truth_sample_2026.csv`](file:///c:/Users/Admin/Downloads/t%C3%A1i%20%C4%91%E1%BB%8Bnh%20v%E1%BB%8B%20thu%E1%BB%91c/dapagliflozin-lbd/data/samples/ground_truth_sample_2026.csv)  
**Quy mô:** Đúng 200 câu văn nguyên bản trích xuất từ 664 bài báo PubMed về Dapagliflozin (giai đoạn tiền 2019).  
**Mục tiêu:** Cung cấp tập dữ liệu kiểm nghiệm chuẩn mực (Ground Truth Benchmark) để đo đạc chính xác **Recall, Precision, F1-Score** và khả năng sẵn sàng tích hợp Knowledge Graph (`KG-Viability`) của toàn bộ hệ thống NLP / LLM.

---

## 1. Bối cảnh & 4 bài học cốt lõi từ đợt kiểm định trước

Ở đợt kiểm định 200 mẫu thử trước (`gold_annotation_blind_200`), hệ thống đạt **Recall 82.00%**, **Precision 65.08%**, **F1 72.57%**, nhưng kiểm toán chuyên sâu phát hiện **4 lỗ hổng cấu trúc**:

1. **Lỗi đảo chiều quan hệ (12 trường hợp):** Khai báo `SRC -> TGT` nhưng ngữ nghĩa thực tế trong câu là `TGT -> SRC` (ví dụ: *Suy thận/CKD làm giảm bài tiết Dapagliflozin* bị gán thành *Dapagliflozin làm giảm CKD*).
2. **Lỗi nhập nhằng giữa Phủ định / Chưa xác quyết với Khẳng định (16 trường hợp):** Các câu phủ định (*"không làm tăng UTI"*) hoặc giả thuyết/mục tiêu nghiên cứu (*"chúng tôi đánh giá liệu..."*) bị đánh đồng thành cạnh khẳng định, khiến đồ thức tri thức (KG) bị nhiễm bẩn nếu nạp vào.
3. **Lỗi gộp sai độ hạt thực thể (11 trường hợp):** *Diabetic Nephropathy* (Bệnh thận đái tháo đường) hoặc *Albuminuria* bị ép cưỡng bức thành generic *Chronic Kidney Disease (DOID:784)*.
4. **Nhiễm bẩn đồng xuất hiện bệnh nền (Cohort Context):** Bệnh nhân T2D hay Tăng huyết áp chỉ là đối tượng tham gia thử nghiệm, câu văn không khẳng định thuốc điều trị hay gây bệnh trong mệnh đề đó, nhưng vẫn bị gán `has_relation = 1`.

Tập dữ liệu **Ground Truth 2026** này được thiết kế phân tầng chuẩn hóa (Stratified Sampling) trên 6 nhóm độc lập và bổ sung đầy đủ các cột kiểm soát chất lượng để loại trừ hoàn toàn 4 lỗi trên.

---

## 2. Cấu trúc bảng dữ liệu `ground_truth_sample_2026.csv`

Mỗi dòng tương ứng với **1 câu văn nguyên gốc** từ PubMed với 2 thực thể đã được đánh dấu nhãn:
- `[[SRC:từ_nguồn]]`: Thực thể nguồn (ví dụ: Thuốc `Dapagliflozin`, Lớp thuốc `SGLT2 Inhibitors`, Gene `SLC5A2`, Quá trình sinh học `Urinary Glucose Excretion`).
- `[[TGT:từ_đích]]`: Thực thể đích (ví dụ: Bệnh `Heart Failure`, `Diabetic Nephropathy`, Triệu chứng/Biomarker `HbA1c`, `Blood Pressure`, Biến cố `Diabetic Ketoacidosis`, `Urinary Tract Infection`).

Các cột từ A đến O là **thông tin ngữ cảnh & gợi ý hệ thống** (tham khảo):
- `sample_id`: Mã mẫu định danh duy nhất (`GT2026_001` đến `GT2026_200`).
- `pmid`: Mã bài báo PubMed.
- `section_zone`: Vùng diễn ngôn của câu trong bài báo (`RESULTS` = Kết quả, `CONCLUSIONS` = Kết luận, `BACKGROUND` = Đặt vấn đề / Mục tiêu, `METHODS` = Phương pháp).
- `source_name`, `source_id`, `source_type`: Tên và ID định danh chuẩn hóa của thực thể nguồn.
- `target_name`, `target_id`, `target_type`: Tên và ID định danh chuẩn hóa của thực thể đích.
- `sentence_highlighted`: Câu văn đầy đủ có tag `[[SRC:...]]` và `[[TGT:...]]`.
- `system_pred_relation`, `system_confidence`, `system_assertion_status`: Dự đoán của AI hiện tại (để đối chiếu sau khi gán nhãn).

---

## 3. Hướng dẫn chi tiết điền 6 cột dành cho chuyên gia gán nhãn (Cột P đến U)

Người gán nhãn cần hoàn thành 6 cột sau:

| Tên cột | Giá trị cho phép | Ý nghĩa & Tiêu chuẩn gán nhãn |
|---|---|---|
| **`human_has_relation`** | `1` hoặc `0` | **BẮT BUỘC.**<br>• Điền `1` nếu câu văn khẳng định hoặc mô tả một tác động sinh học, dược lý, nhân quả hoặc tương quan trực tiếp giữa `SRC` và `TGT`.<br>• Điền `0` nếu hai thực thể chỉ vô tình nằm chung một câu (như tiêu chuẩn chọn bệnh nhân tham gia thử nghiệm, giả thuyết chưa kết luận, hoặc thuộc 2 mệnh đề không liên quan). |
| **`human_direction_valid`** | `YES` hoặc `NO` | • `YES`: Chiều tác động đi từ `[[SRC]]` $\to$ `[[TGT]]` (đúng chiều khai báo).<br>• `NO`: Ngược chiều (thực tế trong câu là `[[TGT]]` tác động lên `[[SRC]]`) hoặc không thể xác định chiều. |
| **`human_assertion_status`** | `ASSERTED`, `UNCERTAIN`, `NEGATED` | • `ASSERTED`: Khẳng định thực tế/kết quả đã chứng minh (*"reduced", "caused", "inhibits"*).<br>• `UNCERTAIN`: Giả thuyết, mục tiêu bài báo, khuynh hướng chưa chắc chắn (*"aimed to evaluate", "may potentially", "remains controversial"*).<br>• `NEGATED`: Phủ định, không thấy hiệu ứng, kết quả âm tính (*"did not reduce", "no significant difference", "failed to prevent"*). |
| **`human_relation_type`** | 1 trong 8 nhãn chuẩn (chi tiết bên dưới) | Loại quan hệ y sinh chính xác nhất thể hiện bản chất liên kết giữa `SRC` và `TGT`. Điền `NONE` nếu `human_has_relation = 0`. |
| **`human_kg_viable`** | `YES` hoặc `NO` | **Tiêu chuẩn sẵn sàng nạp vào Đồ thức tri thức (Knowledge Graph):**<br>Chỉ điền `YES` khi đồng thời thỏa mãn:<br>1. `human_has_relation = 1`<br>2. `human_direction_valid = YES`<br>3. `human_assertion_status = ASSERTED`<br>4. Quan hệ mang giá trị dược lý/cơ chế tin cậy.<br>Nếu là kết quả phủ định (`NEGATED`), giả thuyết (`UNCERTAIN`), hoặc ngược chiều (`NO`), BẮT BUỘC điền `NO`. |
| **`human_annotator_notes`** | Văn bản tự do | Ghi chú thêm nếu phát hiện sắc thái đặc biệt, câu đảo ngữ, hoặc nhận xét về thực thể. |

---

## 4. Danh mục 8 nhãn quan hệ chuẩn (`human_relation_type`)

1. **`TREATS`**: Thuốc hoặc Lớp thuốc điều trị, cải thiện bệnh lý hoặc giảm biến cố lâm sàng (`Compound / DrugClass -> Disease`).
   - *Ví dụ:* Dapagliflozin làm giảm nhập viện do suy tim $\to$ `TREATS`.
2. **`INHIBITS`**: Thuốc ức chế thụ thể, kênh vận chuyển, protein hoặc gen (`Compound / DrugClass -> Gene`).
   - *Ví dụ:* Dapagliflozin ức chế chọn lọc SGLT2 $\to$ `INHIBITS`.
3. **`CAUSES`**: Thuốc gây ra tác dụng phụ, biến cố bất lợi hoặc một quá trình bệnh lý gây ra bệnh (`Compound -> Disease` hoặc `Process -> Disease`).
   - *Ví dụ:* Dapagliflozin làm tăng nguy cơ nhiễm trùng sinh dục hoặc DKA $\to$ `CAUSES`.
4. **`DECREASES`**: Làm giảm mức độ, chỉ số hoặc nồng độ của một quá trình/biomarker/triệu chứng (`SRC -> Phenotype / BiologicalProcess`).
   - *Ví dụ:* Dapagliflozin làm giảm huyết áp tâm thu, giảm HbA1c, giảm cân nặng $\to$ `DECREASES`.
5. **`INCREASES`**: Làm tăng mức độ của một quá trình sinh lý hoặc bài tiết (`SRC -> BiologicalProcess / Phenotype`).
   - *Ví dụ:* Dapagliflozin làm tăng đào thải glucose qua nước tiểu (`Urinary Glucose Excretion`) $\to$ `INCREASES`.
6. **`PALLIATES`**: Làm chậm tiến trình xấu đi của bệnh hoặc giảm tổn thương cơ quan mà không trị dứt điểm hoàn toàn.
   - *Ví dụ:* SGLT2i làm chậm suy giảm eGFR ở bệnh nhân suy thận mãn $\to$ `PALLIATES`.
7. **`ASSOCIATED_WITH`**: Có mối liên quan, tương quan dịch tễ hoặc đồng biến thiên nhưng câu văn không nêu rõ cơ chế tác động tăng/giảm đơn hướng.
8. **`NONE`**: Không có quan hệ trực tiếp trong câu văn đang xét (`human_has_relation = 0`).

---

## 5. Các ví dụ minh họa mẫu mực giải quyết 4 tình huống khó

### Tình huống 1: Câu mô tả quần thể bệnh nhân (Cohort Background)
- **Câu văn:** *"In a double-blind study of patients with [[TGT:type 2 diabetes]] and baseline hypertension, [[SRC:dapagliflozin]] was well tolerated."*
- **Phân tích:** Mệnh đề này chỉ mô tả tiêu chuẩn chọn mẫu bệnh nhân (bệnh nhân T2D tham gia thử nghiệm), câu không trình bày kết quả dapagliflozin điều trị T2D ở câu này.
- **Điền nhãn chuẩn:**
  - `human_has_relation` = `0`
  - `human_direction_valid` = `NO`
  - `human_assertion_status` = `ASSERTED`
  - `human_relation_type` = `NONE`
  - `human_kg_viable` = `NO`
  - `human_annotator_notes` = `"Cohort population inclusion criteria; no causal treatment claim in this sentence."`

### Tình huống 2: Kết quả âm tính / Không có tác dụng (Negated finding)
- **Câu văn:** *"Treatment with [[SRC:dapagliflozin]] did not increase the incidence of [[TGT:urinary tract infections]] compared with placebo."*
- **Phân tích:** Câu văn khẳng định một kết quả âm tính/phủ định (không thấy sự khác biệt về nhiễm trùng tiểu).
- **Điền nhãn chuẩn:**
  - `human_has_relation` = `1` (câu có đánh giá quan hệ)
  - `human_direction_valid` = `YES`
  - `human_assertion_status` = `NEGATED` (bắt buộc chọn NEGATED)
  - `human_relation_type` = `NONE`
  - `human_kg_viable` = `NO` (không xuất cạnh gây bệnh vào KG!)
  - `human_annotator_notes` = `"Null finding vs placebo; safe from UTI risk."`

### Tình huống 3: Ngược chiều tác động (Reverse causality)
- **Câu văn:** *"Severe [[SRC:renal impairment]] substantially blunts the glucosuric response to [[TGT:dapagliflozin]]."*
- **Phân tích:** Suy thận (`SRC`) làm giảm tác dụng bài tiết đường của Dapagliflozin (`TGT`). Đây là chiều Bệnh $\to$ Dược động học của thuốc, không phải Thuốc tác động lên Bệnh.
- **Điền nhãn chuẩn:**
  - `human_has_relation` = `1`
  - `human_direction_valid` = `NO` (cặp khai báo theo chiều Dapa $\to$ Bệnh sẽ bị ngược!)
  - `human_assertion_status` = `ASSERTED`
  - `human_relation_type` = `DECREASES`
  - `human_kg_viable` = `NO` (chặn không cho tạo cạnh sai hướng)
  - `human_annotator_notes` = `"Reverse direction: renal impairment blunts drug efficacy."`

### Tình huống 4: Cạnh điều trị tim thận kinh điển (Cardiorenal Confirmed Positive)
- **Câu văn:** *"In patients with chronic kidney disease, [[SRC:dapagliflozin]] significantly reduced the risk of composite renal failure and hospitalization for [[TGT:heart failure]]."*
- **Phân tích:** Kết quả thử nghiệm lâm sàng khẳng định thuốc giảm nhập viện do suy tim.
- **Điền nhãn chuẩn:**
  - `human_has_relation` = `1`
  - `human_direction_valid` = `YES`
  - `human_assertion_status` = `ASSERTED`
  - `human_relation_type` = `TREATS`
  - `human_kg_viable` = `YES` (100% sẵn sàng nạp vào Discovery Knowledge Graph)
  - `human_annotator_notes` = `"Confirmed clinical endpoint reduction in cardiorenal outcome."`

---

## 6. Tiến độ & Kế hoạch tiếp theo sau khi hoàn tất gán nhãn

1. **Sau khi nhóm gán nhãn hoàn tất tệp `ground_truth_sample_2026.csv`:**
   - Hệ thống sẽ chạy bộ script đánh giá tự động: `scripts/evaluate_gold_ground_truth.py`.
   - Tính toán ma trận nhầm lẫn (Confusion Matrix), tính riêng:
     - **NLP Relation Precision & Recall** (trên tập các câu có quan hệ thực tế).
     - **Specificity** (khả năng loại trừ các câu đồng xuất hiện nền / Cohort Background).
     - **Assertion Accuracy** (độ chính xác phát hiện câu phủ định `NEGATED` vs khẳng định `ASSERTED`).
     - **KG-Ready Yield** (tỷ lệ trích xuất đúng và an toàn nạp vào Knowledge Graph).
2. Kết quả sẽ được xuất thành báo cáo Markdown/JSON chính thức để báo cáo hội đồng nghiệm thu dự án.
