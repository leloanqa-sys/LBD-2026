# Hướng Dẫn Gán Nhãn Tập Vàng Mù (Blind Gold Standard Annotation Guide)

**Tệp dữ liệu gán nhãn:** [`dapagliflozin-lbd/data/gold_annotation_blind_200.csv`](file:///c:/Users/Admin/Downloads/t%C3%A1i%20%C4%91%E1%BB%8Bnh%20v%E1%BB%8B%20thu%E1%BB%91c/dapagliflozin-lbd/data/gold_annotation_blind_200.csv)  
**Quy mô:** 200 câu văn nguyên gốc từ PubMed (đã cố định 99 câu `DEV` và 101 câu `TEST` theo mã băm PMID ở tệp khóa nội bộ riêng biệt).  
**Nguyên tắc thiết kế mù (Double-Blind Protocol):** Tệp CSV này **không chứa bất kỳ dự đoán, điểm tin cậy hay nhãn gợi ý nào của AI** để tránh hoàn toàn hiệu ứng mỏ neo (anchoring bias) đối với người gán nhãn.

---

## 1. Cấu trúc mỗi dòng trong `gold_annotation_blind_200.csv`

Mỗi dòng là **1 câu văn thực tế duy nhất** từ bài báo PubMed, trong đó cả 2 thực thể đã được đánh dấu trực tiếp trong câu bằng ký hiệu:
- `[[SRC:...]]` — Thực thể nguồn (`source_entity`, thuộc nhóm: `Compound`, `DrugClass`, `Gene`, `BiologicalProcess`).
- `[[TGT:...]]` — Thực thể đích (`target_entity`, thuộc nhóm: `Gene`, `BiologicalProcess`, `Disease`).

> **Quy tắc vàng:** Chỉ gán nhãn dựa trên **những gì câu văn đang hiển thị nói trực tiếp**. Tuyệt đối **không** dùng kiến thức y khoa bên ngoài câu văn để điền `has_relation = 1` nếu câu văn chỉ liệt kê hai thực thể cạnh nhau mà không khẳng định mối liên hệ.

---

## 2. Hướng dẫn điền 5 cột gán nhãn

Người gán nhãn chỉ cần điền vào 5 cột từ cột H đến cột L (`has_relation`, `polarity_sign`, `clinical_verb`, `direction_valid`, `assertion_status`):

### Cột 1: `has_relation` (Bắt buộc — Điền `1` hoặc `0`)
- **`1` (Có quan hệ):** Câu văn khẳng định hoặc khảo sát trực tiếp tác động, cơ chế, sự điều hòa hoặc mối liên hệ nhân quả/tương quan giữa `[[SRC:...]]` và `[[TGT:...]]`.
- **`0` (Không có quan hệ / Chỉ đồng xuất hiện):**
  - Hai thực thể nằm ở hai mệnh đề độc lập không tác động lên nhau.
  - `[[TGT:...]]` chỉ là **bệnh nền của quần thể bệnh nhân** mà câu văn đang xét tác động lên một đích khác (Ví dụ: *"In patients with [[TGT:type 2 diabetes]], [[SRC:dapagliflozin]] reduced blood pressure"* $\rightarrow$ Câu này nói Dapa giảm huyết áp, còn T2D chỉ là tiêu chuẩn chọn bệnh nhân trong câu đó $\rightarrow$ đối với cặp `Dapa -> T2D` ở riêng câu này, điền `0`).

> **Lưu ý nhanh:** Nếu điền `has_relation = 0`, anh/chị có thể để trống các cột còn lại (hoặc điền `0` / `NONE`).

---

### Cột 2: `polarity_sign` (Bắt buộc khi `has_relation = 1` — Điền `-1`, `+1`, hoặc `0`)
Xác định **dấu tác động sinh học** của `[[SRC]]` lên `[[TGT]]`:

| Giá trị | Ý nghĩa sinh học / Dược lý | Các động từ/cụm từ điển hình trong câu |
| :---: | :--- | :--- |
| **`-1`** | **Giảm / Ức chế / Điều trị / Bảo vệ (Negative Correlation)** | *inhibits, blocks, reduces, decreases, lowers, attenuates, prevents, ameliorates, improves (disease), treats, protects against* |
| **`+1`** | **Tăng / Kích hoạt / Gây ra tác dụng phụ / Thúc đẩy bệnh sinh (Positive Correlation)** | *increases, induces, promotes, stimulates, enhances, activates, causes, triggers, elevates risk of (DKA/UTI)* |
| **`0`** | **Liên kết trung tính / Không rõ chiều tăng giảm (Neutral Association)** | *is associated with, correlates with, was evaluated in, binds to (without direction)* |

> **Cẩn trọng với bẫy đảo dấu ngữ pháp ("Inhibition of X"):**
> - Nếu câu văn viết: *"Inhibition of `[[SRC:SGLT2]]` increases `[[TGT:natriuresis]]`"*
> - Hãy tự hỏi: Bản thân **protein/gene `SGLT2` khi hoạt động bình thường** làm tăng hay làm giảm `natriuresis`? Vì **ức chế** SGLT2 làm **tăng** thải muối, nên bản thân `SGLT2` có tác động **làm giảm (`-1`)** thải muối (`SGLT2 --(-1)--> Natriuresis`). Hãy ghi chú `"inversion"` vào cột `annotator_notes` nếu gặp câu dạng này.

---

### Cột 3: `clinical_verb` (Điền 1 trong các nhãn chuẩn)
Chọn nhãn cụ thể nhất phản ánh bản chất quan hệ:
- Nhóm dấu `-1`:
  - `TREATS`: Thuốc/Lớp thuốc điều trị, phòng ngừa hoặc cải thiện bệnh (`Compound/DrugClass -> Disease`).
  - `INHIBITS`: Ức chế gen/protein/transporter (`Compound/DrugClass -> Gene`).
  - `DECREASES`: Làm giảm mức độ của một quá trình sinh lý hoặc bệnh lý (`-> BiologicalProcess / Disease`).
- Nhóm dấu `+1`:
  - `CAUSES`: Thuốc gây tác dụng phụ/biến cố bất lợi (như DKA, UTI) hoặc tiến trình sinh lý gây bệnh (`-> Disease`).
  - `ACTIVATES`: Hoạt hóa gen/protein (`-> Gene`).
  - `INCREASES`: Làm tăng mức độ quá trình sinh lý (như tăng bài tiết glucose niệu `glucosuria`, tăng `natriuresis`).
- Nhóm dấu `0`:
  - `ASSOCIATED_WITH`: Có liên quan nhưng không khẳng định chiều tăng/giảm.
- Không có quan hệ:
  - `NONE`

---

### Cột 4: `direction_valid` (Kiểm tra chiều `Source -> Target`)
- **`1`**: Chiều `Source -> Target` đã đúng chuẩn sinh học (`Source` tác động lên `Target`).
- **`-1`**: Chiều thực tế trong câu bị ngược (`Target` tác động lên `Source`).
- **`0`**: Quan hệ hai chiều / tương quan vô hướng.

---

### Cột 5: `assertion_status` (Độ xác quyết của câu văn)
- `asserted`: Câu văn khẳng định kết quả hoặc cơ chế thực tế (mặc định nếu để trống khi `has_relation = 1`).
- `negated`: Câu văn **phủ định** mối quan hệ (*"did not reduce", "failed to prevent", "was not associated with"*).
- `uncertain`: Giả thuyết hoặc mục tiêu nghiên cứu chưa có kết luận (*"We aimed to investigate whether...", "may potentially...", "remains unclear"*).

---

## 3. Ví dụ minh họa thực tế

| `source_entity` | `target_entity` | `highlighted_sentence` | `has_relation` | `polarity_sign` | `clinical_verb` | `direction_valid` | `assertion_status` |
|---|---|---|---|---|---|---|---|
| `Dapagliflozin [DB06292]` | `Heart Failure [DOID:6000]` | *"[[SRC:Dapagliflozin]] reduced the risk of hospitalization for [[TGT:heart failure]] in patients with T2D."* | `1` | `-1` | `TREATS` | `1` | `asserted` |
| `Dapagliflozin [DB06292]` | `Diabetic Ketoacidosis [DOID:0050155]` | *"Rare cases of [[TGT:euglycemic ketoacidosis]] have been reported with [[SRC:dapagliflozin]] therapy."* | `1` | `+1` | `CAUSES` | `1` | `asserted` |
| `Dapagliflozin [DB06292]` | `Urinary Glucose Excretion [GO:1904659]` | *"[[SRC:Dapagliflozin]] selectively inhibits SGLT2 to promote [[TGT:urinary glucose excretion]]."* | `1` | `+1` | `INCREASES` | `1` | `asserted` |
| `Dapagliflozin [DB06292]` | `Type 2 Diabetes [DOID:9352]` | *"We aimed to evaluate renal outcomes of [[SRC:dapagliflozin]] in patients with [[TGT:type 2 diabetes]]."* | `0` | `0` | `NONE` | `0` | `uncertain` |
| `Dapagliflozin [DB06292]` | `Heart Failure [DOID:6000]` | *"[[SRC:Dapagliflozin]] did not increase the risk of [[TGT:heart failure]] hospitalization."* | `1` | `0` | `NONE` | `1` | `negated` |
