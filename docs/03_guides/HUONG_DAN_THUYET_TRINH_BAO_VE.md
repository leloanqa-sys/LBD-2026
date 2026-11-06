# BẢN THUYẾT TRÌNH BẢO VỆ ĐỒ ÁN
## ĐỀ TÀI: HỆ THỐNG KHÁM PHÁ TRI THỨC VÀ TÁI ĐỊNH VỊ THUỐC TỰ ĐỘNG BẰNG HỌC MÁY TRÊN ĐỒ THỊ TRI THỨC Y SINH & XỬ LÝ NGÔN NGỮ TỰ NHIÊN (LBD + M2 GRAPH RANKING)

---

# PHẦN I: PHÂN TÍCH QUY TRÌNH PIPELINE TOÀN DIỆN
### *(ĐÃ DÙNG GÌ — LÀM GÌ — TẠI SAO LẠI CHỌN CÔNG NGHỆ ĐÓ?)*

Toàn bộ hệ thống được xây dựng theo kiến trúc liên hoàn **Hai Tầng Phối Hợp (Two-Stage Decoupled Architecture)** nhằm giải quyết trọn vẹn cả 2 yêu cầu: **Tốc độ & Độ chính xác toàn cục (Global Ranking Accuracy)** và **Khả năng giải thích cơ chế sinh học phân tử (Mechanistic Interpretability)**.

```mermaid
flowchart TD
    subgraph STAGE 1: DỮ LIỆU NỀN & XỬ LÝ TRI THỨC
        D1["Đồ thị Hetionet v1.0<br/>(47.031 Nút, 2.250.197 Cạnh)"] --> D2["Bộ lọc Khóa nhãn Zero Leakage<br/>(Loại bỏ CtD/CpD khỏi đồ thị)"]
        P1["Kho văn bản PubMed<br/>(Abstracts ≤ 2018)"] --> P2["Biomedical NLP Engine<br/>(BioRED NER + Dependency Parser)"]
    end

    subgraph STAGE 2: XẾP HẠNG VĨ MÔ (MACRO RANKING - M2)
        D2 --> M1["Vector hóa Ma trận Thưa DWPC<br/>(w = 0.4, 65 đặc trưng topological)"]
        M1 --> M2["Học máy có giám sát M2<br/>(arcsinh + StandardScaler + ElasticNet)"]
        M2 --> M3["Inference 137 Bệnh lý trong 0,05s<br/>Xác suất P(Treat)"]
        M3 --> M4["Gating Filter<br/>Tách Chỉ định gốc vs. Top Novel Candidates"]
    end

    subgraph STAGE 3: GIẢI THÍCH VI MÔ (MICRO DFS & EVIDENCE)
        M4 --> E1["Top 5 Bệnh Ứng viên Mới"]
        E1 --> E2["Meet-in-the-Middle Micro-DFS<br/>Truy vết chuỗi 3-nút và 5-nút cụ thể"]
        E2 --> E3["Chấm điểm từng con đường:<br/>Contrib = β_metapath × w(path)"]
        P2 -.-> E4["Đính kèm Bằng chứng Văn bản<br/>(Câu văn trích xuất & PMID ≤ 2018)"]
        E3 --> E4
        E4 --> OUT["Báo cáo Khoa học & UI Tương tác"]
    end
```

---

## 1. Tầng Dữ liệu Nền & Chuẩn bị (Data Ingestion & Integrity Boundary)

### Đã dùng gì?
- **Đồ thị tri thức Hetionet v1.0**: Tích hợp 29 cơ sở dữ liệu y sinh học công cộng chuẩn quốc tế (DrugBank, Entrez Gene, Disease Ontology, GO, Reactome, LINCS L1000, SIDER,...), gồm `47.031` nút (thuộc 11 loại thực thể) và `2.250.197` cạnh (thuộc 24 loại quan hệ / metaedge).
- **Kho bài báo khoa học y sinh (PubMed Abstracts)**: Dữ liệu bài báo chuyên ngành đến mốc thời gian lịch sử tháng 12/2018.

### Đã làm gì?
1. **Thiết lập Vùng đệm Thời gian (Temporal Boundary $\le$ 2018)**: Đóng băng toàn bộ đồ thị và văn bản y văn trước năm 2019. Mục đích là tạo bài kiểm tra "mù" (blind benchmark) lịch sử: Liệu hệ thống có thể từ bằng chứng trước năm 2019 mà dự đoán chính xác các phát hiện lâm sàng lớn của thế giới trong giai đoạn 2019–2023 hay không?
2. **Khóa nhãn tuyệt đối (Zero Label Leakage Protocol)**: 
   - Tách toàn bộ `755` cạnh `CtD` (Compound-treats-Disease) và `390` cạnh `CpD` (Compound-palliates-Disease) ra khỏi đồ thị trước khi tính toán bất kỳ bậc nút hay đường đi nào.
   - Nhãn `CpD` (chỉ giảm triệu chứng mà không chữa căn nguyên) được gán nhãn `-1` để loại khỏi tập huấn luyện, tránh làm nhiễu nhãn âm thực sự.

### Tại sao sử dụng?
- **Không tự tạo dữ liệu giả lập (No Synthetic Data)**: Nhiều đồ án sinh viên mắc lỗi tự tạo đồ thị con nhỏ vài chục nút hoặc tự gán nhãn. Sử dụng Hetionet nguyên bản đảm bảo tính khách quan và độ phức tạp tương đương bài toán thật ở các viện nghiên cứu dược học thế giới.
- **Tránh rò rỉ dữ liệu (Information Leakage)**: Nếu không tách `CtD`/`CpD` trước khi tính bậc nút, mô hình sẽ "học vẹt" (cheating) bằng cách nhìn thấy liên kết trực tiếp, dẫn đến kết quả đánh giá ảo nhưng vô giá trị khi áp dụng cho thuốc mới.

---

## 2. Tầng Trích xuất Đặc trưng Đồ thị & Học máy Xếp hạng (Macro-Matrix M2 Ranker)

### Đã dùng gì?
- **Thuật toán Đếm đường đi có Trọng số Phạt Bậc (Degree-Weighted Path Count - DWPC)** với tham số giảm chấn $w = 0.4$.
- **Biến đổi toán học arcsinh (`TrainAsinh`)**: Biến đổi phi tuyến bảo toàn zero.
- **Mô hình Hồi quy Logistic phạt hỗn hợp ElasticNet ($L_1 + L_2$)**: Tối ưu hóa bằng solver `saga` đa chu kỳ.
- **Chia tập Stratified Group K-Fold theo cụm hóa học Tanimoto**: Gom cụm thuốc bằng Morgan Fingerprint.

### Đã làm gì?
1. **Nhân ma trận thưa tính 63 Metapath DWPC**: Cho toàn bộ `212.624` cặp thuốc - bệnh (`1.552` thuốc $\times$ `137` bệnh), nhân các ma trận liên thuộc dạng `scipy.sparse.csr_matrix` của các chuỗi quan hệ 3 nút ($C-G-D$) và 5 nút ($C-G-BP-G-D$, $C-G-PW-G-D$,...).
2. **Trừng phạt các siêu nút (Hub Nodes)**: Mỗi bước nhảy giữa nút $u$ và nút $v$ trên quan hệ $m$ đều được nhân trọng số:
   $$w(u, m, v) = \left(\max(\text{deg}_{\text{out}}(u, m), 1) \cdot \max(\text{deg}_{\text{in}}(v, m), 1)\right)^{-0.4}$$
3. **Huấn luyện Mô hình M2**: 
   - Đầu vào gồm 65 đặc trưng: `log_drug_degree`, `log_disease_degree` + 63 đặc trưng DWPC.
   - Chuẩn hóa qua hàm $\text{arcsinh}(x / \text{scale})$ để nén các giá trị DWPC lệch phải mà không làm mất tính thưa (sparse), tiếp nối bởi `StandardScaler`.
   - Học máy tự động gán trọng số $\beta_j$ cho từng loại metapath.

### Tại sao sử dụng?
- **Tại sao bỏ công thức Heuristic thủ công ($S_{\text{orig}}$)?** 
  - Thực nghiệm kiểm toán độc lập trên tập Locked Test cho thấy công thức cũ chỉ đạt `PR-AUC = 0.0259` (thấp hơn cả việc chỉ đếm bậc nút thuần túy `M0: 0.0566`). Nguyên nhân: Công thức thủ công bị nhiễu loạn bởi hàng nghìn đường đi qua enzyme gan (`CYP3A4`, `CYP2C9`, `UGT1A9`) có mặt ở khắp mọi nơi.
- **Tại sao là DWPC thay vì đếm đường đi thô (Path Count)?**
  - Các gen phổ biến như *TP53* hay *TNF* có hàng nghìn kết nối. Nếu đếm đường thông thường, bệnh nào cũng sẽ có điểm cao giả tạo. Số mũ $w = 0.4$ phạt các nút quá phổ biến, ưu tiên các liên kết chuyên biệt và có giá trị sinh học thực sự.
- **Tại sao là ElasticNet thay vì Random Forest hay Deep Learning phức tạp?**
  - ElasticNet kết hợp $L_1$ (Lasso - triệt tiêu đặc trưng vô dụng về 0) và $L_2$ (Ridge - kiểm soát đa cộng tuyến).
  - Quan trọng nhất: ElasticNet có **tính giải thích tuyệt đối (Glass-box model)**. Mỗi trọng số $\beta_j$ là một hệ số log-odds rõ ràng, giúp bác sĩ và dược sĩ biết chính xác tại sao mô hình lại chấm điểm cao cho một con đường sinh học, thay vì là một "hộp đen" (black-box).

---

## 3. Tầng Giải thích Vi mô & Bằng chứng Văn bản (Micro-DFS & Evidence Grounding)

### Đã dùng gì?
- **Thuật toán Duyệt đồ thị Meet-in-the-Middle Micro-DFS**: Tìm kiếm hai chiều gặp nhau ở giữa.
- **Biomedical NLP Engine**: Pipeline xử lý văn bản y sinh tích hợp BioRED NER, Bộ lọc phân cực quan hệ (Polarity Filtering) và Kiểm định phủ định (Negation Detection).

### Đã làm gì?
1. Sau khi Tầng 1 tính điểm và xếp hạng 137 bệnh trong 0,05 giây, lấy ra **Top 5 bệnh ứng viên mới**.
2. Với mỗi bệnh trong Top 5:
   - Thay vì duyệt toàn bộ đồ thị từ đầu (mất 5 phút với hàng triệu đường), thuật toán lập chỉ mục ngược từ tập gen của bệnh (`DaG`/`DuG`/`DdG` chỉ có khoảng 20–50 gen) qua các thuật ngữ sinh học (`b_map`).
   - Duyệt từ tập gen của thuốc để tìm điểm giao nhau trong $0,05$ giây.
3. Chấm điểm từng đường đi cụ thể:
   $$\text{Contribution} = \beta_{\text{metapath}} \times \prod_{e \in \text{path}} w(e)$$
   Sắp xếp và trích xuất Top 2–3 đường đi có đóng góp dương lớn nhất vào logit.
4. Gắn kèm các câu trích dẫn thực tế từ các bài báo PubMed trước 2019 làm căn cứ minh chứng.

### Tại sao sử dụng?
- **Sự kết hợp hoàn hảo giữa Ma trận và DFS**: Ma trận thưa rất nhanh nhưng chỉ cho ra một con số vô hồn $\sum \text{DWPC}$. Ngược lại, DFS rất trực quan (vẽ được chuỗi gen) nhưng lại quá chậm nếu chạy toàn cục. Phương pháp **"Ma trận lọc ứng viên $\rightarrow$ Micro-DFS giải thích cơ chế"** lấy được ưu điểm tốc độ của ma trận và ưu điểm trực quan của DFS.

---

# PHẦN II: KỊCH BẢN THUYẾT TRÌNH TỪNG SLIDE
### *(DÀNH CHO 10–12 PHÚT TRÌNH BÀY TRƯỚC HỘI ĐỒNG)*

---

### SLIDE 1: MỞ ĐẦU & TÍNH CẤP THIẾT CỦA ĐỀ TÀI
- **Tiêu đề Slide**: *Hệ Thống Tái Định Vị Thuốc Bằng Học Máy Trên Đồ Thị Tri Thức Y Sinh (LBD & M2 Graph Ranking)*
- **Lời thuyết trình**:
  > *"Kính thưa Thầy/Cô Chủ tịch Hội đồng và quý Thầy/Cô phản biện,*
  >
  > *Quy trình phát triển một loại thuốc mới thông thường tiêu tốn từ 10 đến 15 năm và hàng tỷ USD, nhưng tỷ lệ thất bại trong thử nghiệm lâm sàng lên đến hơn 90%. Do đó, **Tái định vị thuốc (Drug Repurposing)** — tìm chỉ định mới cho các thuốc đã được chứng minh an toàn — đang là chiến lược hàng đầu của ngành dược phẩm.*
  >
  > *Tuy nhiên, đa số các phát hiện tái định vị trước đây đều mang tính tình cờ hoặc phụ thuộc vào trực giác cá nhân của nhà nghiên cứu. Mục tiêu đồ án của em là: **Xây dựng một hệ thống phát hiện tri thức tự động (Literature-Based Discovery) kết hợp mô hình học máy trên đồ thị tri thức y sinh quy mô lớn (Hetionet v1.0), vừa có khả năng xếp hạng chính xác các bệnh ứng viên, vừa diễn giải được cơ chế sinh học phân tử bằng các chuỗi tương tác gen và y văn thực tế**."*

---

### SLIDE 2: THÁCH THỨC KHOA HỌC & BƯỚC NGOẶT KIỂM TOÁN THỰC NGHIỆM
- **Tiêu đề Slide**: *Sự Thất Bại Của Công Thức Thủ Công (Heuristic) Và Bài Học Kiểm Toán Dữ Liệu Lớn*
- **Lời thuyết trình**:
  > *"Kính thưa Thầy/Cô, khi bắt đầu dự án, nhóm nghiên cứu đã áp dụng một công thức chấm điểm thủ công truyền thống: duyệt đường đi DFS, tính chất lượng cạnh, nhân dấu hướng tác động (+1/-1) và trừ phạt bậc nút.*
  >
  > *Trên các đồ thị giả lập nhỏ, công thức này chạy rất mượt. Nhưng khi đưa lên **đồ thị Hetionet thật với 2,25 triệu cạnh và 47.000 nút**, một hiện tượng nghiêm trọng đã xảy ra:*
  > 
  > 1. *Chỉ một hoạt chất như Dapagliflozin đã bùng nổ hơn 8.200 đường đi nối tới 125/137 bệnh, trong đó có tới 96% đường đi bị kéo qua các enzyme chuyển hóa thuốc ở gan như CYP3A4, CYP2C9 chứ không đi qua đích tác dụng dược lý chính.*
  > 2. *Khi kiểm toán độc lập trên tập Locked Test, công thức thủ công này sụp đổ với chỉ số **PR-AUC chỉ đạt 0.0259** — thậm chí còn kém hơn cả một mô hình cơ sở M0 chỉ đếm bậc của nút thuốc và bệnh (0.0566).*
  > 
  > *Phát hiện này buộc nhóm em phải thay đổi tư duy: **Không thể dùng các hệ số cảm tính do con người tự đặt ra. Hệ thống bắt buộc phải chuyển sang học máy có giám sát (Supervised Machine Learning) để thuật toán tự học trọng số của từng khuôn mẫu đường đi từ dữ liệu y sinh thực tế**."*

---

### SLIDE 3: KIẾN TRÚC GIẢI PHÁP HAI TẦNG (MACRO M2 + MICRO DFS)
- **Tiêu đề Slide**: *Kiến Trúc Hai Tầng: Tốc Độ Ma Trận & Độ Sâu Sinh Học Của DFS*
- **Lời thuyết trình**:
  > *"Để giải quyết bài toán trên, em đã thiết kế kiến trúc 2 tầng phối hợp:*
  >
  > * **Tầng 1 - Xếp hạng Vĩ mô (Macro Ranking - M2)**: Dựa trên thuật toán DWPC (Degree-Weighted Path Count) với số mũ giảm chấn $w=0.4$. Em vector hóa toàn bộ 212.624 cặp thuốc - bệnh thành ma trận 65 đặc trưng topological. Mô hình ElasticNet học từ 755 liên kết điều trị thật để gán trọng số cho từng loại đường đi. Nhờ tối ưu ma trận thưa, tầng này quét toàn bộ 137 bệnh chỉ mất **0,05 giây**, tách bỏ hoàn toàn các chỉ định nền đã biết để đưa ra Top 5 bệnh ứng viên mới.*
  >
  > * **Tầng 2 - Giải thích Vi mô (Micro-DFS & NLP Evidence)**: Với Top 5 bệnh tiềm năng nhất vừa tìm được, thuật toán Meet-in-the-Middle DFS sẽ truy vết ngược các chuỗi gen cụ thể. Điểm sáng tạo ở đây là: **Điểm đóng góp của từng đường đi được tính bằng chính trọng số $\beta$ mà mô hình M2 đã học nhân với trọng số cạnh DWPC**. Sau đó, engine NLP đính kèm các câu trích dẫn từ bài báo PubMed để bác sĩ có thể kiểm chứng ngay lập tức."*

---

### SLIDE 4: KẾT QUẢ ABLATION STUDY — BẰNG CHỨNG HỌC MÁY TRÊN TẬP KIỂM THỬ ĐỘC LẬP
- **Tiêu đề Slide**: *Kết Quả Nghiên Cứu Thực Nghiệm (Ablation Study) Trên Tập Khóa Độc Lập*
- **Lời thuyết trình**:
  > *"Bảng trên màn hình phản ánh kết quả thực nghiệm hoàn toàn trung thực trên tập Locked Test gồm 134 thuốc và 18.358 cặp bệnh được chia theo cấu trúc hóa học phân tử:*
  >
  > * *Trong đồ thị thực tế, tỷ lệ cặp điều trị thật cực kỳ hiếm, chỉ chiếm **0,355%** (1:280), do đó thước đo khoa học chuẩn xác nhất là **PR-AUC (Độ chính xác trung bình)**.*
  > * *Mô hình ngẫu nhiên đạt `PR-AUC = 0.0029`.*
  > * *Công thức heuristic cũ (M1) chỉ đạt `0.0259`.*
  > * *Mô hình cơ sở chỉ dùng bậc nút (M0) đạt `0.0566`.*
  > * *Và **Mô hình M2 DWPC của em đạt `PR-AUC = 0.1681` và `ROC-AUC = 0.8691`**.*
  >
  > *Kết quả này chứng minh: Mô hình M2 làm giàu tín hiệu dự đoán gấp **58 lần** so với ngẫu nhiên, và tăng gấp **6,5 lần** so với công thức cũ. Chỉ số **Mean Recall@20 đạt 61,36%**, nghĩa là hơn 61% các chỉ định điều trị thật nằm trọn vẹn trong Top 20 bệnh được gợi ý đầu tiên."*

---

### SLIDE 5: "BỘ NÃO" MÔ HÌNH HỌC ĐƯỢC GÌ TỪ DỮ LIỆU Y SINH?
- **Tiêu đề Slide**: *Giải Mã Trọng Số Mô Hình: Cơ Chế Đối Kháng Sinh Học Tự Động*
- **Lời thuyết trình**:
  > *"Thưa Thầy/Cô, điều làm em tâm đắc nhất ở mô hình M2 là tính minh bạch toán học. Khi trích xuất các trọng số $\beta$ lớn nhất, mô hình hoàn toàn không học vẹt mà đã tự động tìm ra **quy luật đối kháng sinh học (Therapeutic Reversal)**:*
  >
  > 1. *Hệ số dương cao nhất thuộc về đặc trưng `dwpc:CdG|GpBP|DuG` với $\beta = +1.0311$. Nghĩa là: Nếu thuốc làm giảm biểu hiện một gen trong cùng quá trình sinh học mà bệnh lý làm tăng biểu hiện gen đó $\rightarrow$ Mô hình cộng điểm điều trị mạnh nhất!*
  > 2. *Ngược lại, với `dwpc:CdG|GpBP|DdG`, khi thuốc làm giảm thêm một gen trong quá trình vốn đã suy giảm của bệnh $\rightarrow$ Mô hình tự động phạt điểm âm với $\beta = -0.4593$.*
  > 3. *Các đường đi gắn kết đích phân tử trực tiếp `CbG|direct|DaG` nhận trọng số dương vững chắc $\beta = +0.4757$.*
  >
  > *Mô hình đã chứng minh được cơ chế tác động dược lý thông qua toán học thống kê mà không cần con người can thiệp tham số."*

---

### SLIDE 6: ĐỐI CHIẾU LÂM SÀNG THỰC TẾ TRÊN 6 HOẠT CHẤT (2019–2023)
- **Tiêu đề Slide**: *Kiểm Chứng Độc Lập Bằng Các Thử Nghiệm Lâm Sàng Thực Tế*
- **Lời thuyết trình**:
  > *"Để kiểm tra giá trị thực tiễn, em đã đóng băng dữ liệu trước 2019 và chạy dự đoán trên 6 hoạt chất, đối chiếu với các phê duyệt của FDA diễn ra sau đó:*
  >
  > 1. ***Với Baricitinib (thuốc ức chế JAK1/JAK2)***:
  >    * *Hetionet trước 2017 chỉ biết thuốc trị ung thư huyết học. Hệ thống của em đã xếp **Crohn's disease ở Hạng #4 (0.9805)**, **Lupus ban đỏ ở Hạng #5 (0.9761)** và **Viêm khớp dạng thấp ở Hạng #16 (0.9248)**.*
  >    * *Thực tế: Tháng 6/2018, FDA đã duyệt Baricitinib (Olumiant) chính xác cho Viêm khớp dạng thấp, và hiện nay đang là thuốc chủ đạo cho Crohn và Lupus.*
  > 2. ***Với Colchicine (thuốc trị Gout từ cổ đại)***:
  >    * *Hệ thống xếp **Bệnh mạch vành (Coronary artery disease - CAD) ở Hạng #12/137 (0.8807)**.*
  >    * *Thực tế: Thử nghiệm COLCOT 2019 và LoDoCo2 2020 chứng minh Colchicine giảm 30% biến cố tim mạch; và tháng 6/2023, FDA đã chính thức phê duyệt Lodoco (Colchicine) cho bệnh tim mạch xơ vữa.*
  > 3. ***Với Dapagliflozin (Case study SGLT2i khóa mù)***:
  >    * *Dù hoàn toàn không có trong tập Train, mô hình vẫn tự động định vị **Tăng huyết áp ở Hạng #2 (0.8805)** và **Bệnh thận mạn (CKD) ở Hạng #9 (0.7662)** qua đích phân tử SLC5A2.*
  > 4. ***Với Aspirin***:
  >    * *Mô hình tìm ra ứng viên số 1 là **Vảy nến (Psoriasis - 0.9740)** và truy vết chính xác con đường ức chế phức hợp NF-$\kappa$B qua chuỗi `Aspirin -> NFKBIA -> Psoriasis`.*
  >
  > *Sự trùng khớp giữa dự báo của hệ thống với các mốc phê duyệt quốc tế khẳng định tính đúng đắn và tiềm năng ứng dụng to lớn của đề tài."*

---

### SLIDE 7: KẾT LUẬN & ĐÓNG GÓP CỦA ĐỒ ÁN
- **Tiêu đề Slide**: *Tổng Kết Đóng Góp & Hướng Phát Triển*
- **Lời thuyết trình**:
  > *"Tóm lại, đồ án đã đạt được 3 đóng góp cốt lõi:*
  >
  > 1. *Xây dựng thành công quy trình nghiên cứu trung thực khoa học, phát hiện và vượt qua giới hạn của công thức Heuristic thủ công trên đồ thị lớn Hetionet.*
  > 2. *Thiết kế kiến trúc Hai Tầng phối hợp giữa **Xếp hạng vĩ mô M2 bằng ma trận DWPC** và **Giải thích vi mô bằng Meet-in-the-Middle DFS**, đạt PR-AUC 0.1681 và tốc độ suy luận dưới 0,1 giây.*
  > 3. *Đóng gói hoàn chỉnh thành một thư viện module hóa cao trong `dapagliflozin-lbd`, vượt qua 100% các bài kiểm thử unit tests tự động, sẵn sàng cung cấp công cụ hỗ trợ các nhà nghiên cứu dược lý sàng lọc thuốc tái định vị.*
  >
  > *Em xin chân thành cảm ơn quý Thầy/Cô đã lắng nghe và em xin phép được lắng nghe các câu hỏi đóng góp từ Hội đồng ạ!"*

---

# PHẦN III: BẢN GIẢI THÍCH THUẬT NGỮ CHUYÊN MÔN
### *(DÀNH CHO BẠN KHI THUYẾT MINH HOẶC TRẢ LỜI CÂU HỎI THẦY CÔ)*

1. **Tái định vị thuốc (Drug Repurposing / Drug Reprofiling)**:
   - *Cách diễn giải*: Là việc tìm ra công dụng chữa bệnh mới cho một viên thuốc vốn đã được cấp phép cho bệnh khác (ví dụ: thuốc trị tiểu đường Dapagliflozin đem đi chữa suy tim, thuốc trị Gout Colchicine đem đi ngừa nhồi máu cơ tim). Điểm lợi lớn nhất là thuốc đã vượt qua các thử nghiệm an toàn và độc tính trên người, tiết kiệm 5–7 năm nghiên cứu.

2. **Đồ thị tri thức Y sinh (Biomedical Knowledge Graph - Hetionet)**:
   - *Cách diễn giải*: Là một mạng lưới khổng lồ kết nối tất cả các đối tượng y học: Thuốc, Gen, Bệnh lý, Con đường sinh học, Triệu chứng,... Mỗi mối quan hệ là một cạnh nối có tên gọi rõ ràng (ví dụ: *Thuốc gắn vào Gen*, *Gen gây ra Bệnh*).

3. **Metapath (Khuôn mẫu đường đi / Siêu đường đi)**:
   - *Cách diễn giải*: Là một "công thức" trừu tượng mô tả con đường đi từ Thuốc tới Bệnh. Ví dụ: `Compound -> Gene -> Disease` là metapath 3 nút (thuốc tác động trực tiếp lên gen của bệnh); `Compound -> Gene -> Pathway -> Gene -> Disease` là metapath 5 nút (thuốc tác động lên con đường sinh hóa mà bệnh lý đang làm biến đổi).

4. **DWPC (Degree-Weighted Path Count - Đếm đường đi phạt bậc nút)**:
   - *Cách diễn giải*: Nếu đếm số đường đi bình thường, những gen "ngôi sao" mà ai cũng nghiên cứu (như TP53, TNF) sẽ tạo ra hàng vạn con đường rác tới tất cả các bệnh. DWPC giải quyết việc này bằng cách đặt ra quy tắc: "Càng đi qua ngã tư có quá nhiều đường rẽ (nút bậc cao), giá trị của bước đi đó càng bị giảm giá trị". Nhờ vậy, thuật toán giữ lại được những con đường chuyên biệt, có giá trị sinh học đích thực.

5. **PR-AUC (Precision-Recall Area Under Curve)**:
   - *Cách diễn giải*: Trong một bài toán mất cân bằng dữ liệu cực đoan (cứ 280 cặp thuốc-bệnh chỉ có 1 cặp chữa bệnh thật), chỉ số Accuracy hay ROC-AUC rất dễ bị "đánh lừa" bởi số lượng nhãn âm khổng lồ. PR-AUC đo lường khả năng bắt trúng các cặp dương tính thật trên tổng số các cặp mà mô hình dự đoán. Do đó, PR-AUC là thước đo khắt khe và đáng tin cậy nhất trong y sinh học.

6. **Meet-in-the-Middle DFS (Tìm kiếm theo chiều sâu gặp nhau ở giữa)**:
   - *Cách diễn giải*: Thay vì để một người thám tử đi dò từ Thuốc qua hàng nghìn gen rồi hàng vạn con đường sinh hóa để tìm đến Bệnh (rất dễ bị lạc và mất thời gian), thuật toán cho hai mũi thám tử đi cùng lúc: Một mũi đi từ Thuốc, một mũi đi từ Bệnh, và họ chỉ tìm điểm hẹn gặp nhau ở các con đường sinh học trung gian. Kỹ thuật này giảm thời gian tìm kiếm từ vài phút xuống còn 0,05 giây.

7. **Zero Label Leakage (Chống rò rỉ nhãn)**:
   - *Cách diễn giải*: Là nguyên tắc đạo đức khoa học tối cao: Không bao giờ được cho mô hình nhìn thấy "đáp án" trong lúc đang làm bài thi. Trước khi tính toán bất kỳ chỉ số mạng nào, toàn bộ các cạnh điều trị đã biết đều phải được bóc tách và niêm phong riêng biệt.
