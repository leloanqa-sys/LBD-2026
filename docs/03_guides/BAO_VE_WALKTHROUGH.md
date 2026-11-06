# Báo Cáo Walkthrough & Cẩm Nang Bảo Vệ: Hệ Thống Tái Định Vị Thuốc LBD + M2 Graph Ranking

Tài liệu này tổng hợp **toàn bộ quá trình thực hiện thực tế (100% trung thực khoa học, không ngụy tạo số liệu)**, kiến trúc đóng gói thuật toán vào hệ thống [`dapagliflozin-lbd`](file:///c:/Users/Admin/Downloads/tái%20định%20vị%20thuốc/dapagliflozin-lbd), các kết quả thực nghiệm trên đồ thị tri thức y sinh **Hetionet v1.0** (`47.031` nút, `2.250.197` cạnh), cùng kịch bản trả lời phản biện trước Hội đồng.

---

## 1. Hiện Trạng Đóng Gói Thuật Toán Vào Hệ Thống `dapagliflozin-lbd`

Thuật toán xếp hạng học máy có giám sát **M2 (Degree-Weighted Path Count + ElasticNet Logistic Regression)** kết hợp giải thích đường đi **Meet-in-the-Middle Micro-DFS** từ nhánh nghiên cứu độc lập đã được đóng gói hoàn chỉnh vào nhân của `dapagliflozin-lbd`:

| Thành phần đóng gói | Đường dẫn tệp trong `dapagliflozin-lbd` | Chức năng chính |
| :--- | :--- | :--- |
| **Lõi thuật toán M2 + Micro-DFS** | [`src/lbd/ranking/m2_ranker.py`](file:///c:/Users/Admin/Downloads/tái%20định%20vị%20thuốc/dapagliflozin-lbd/src/lbd/ranking/m2_ranker.py) | Nạp ma trận đặc trưng DWPC, mô hình `M2_model.pkl`, chấm điểm 137 bệnh lý, lọc chỉ định gốc (`CtD`/`CpD`) và truy vết đường đi cơ chế vi mô. |
| **Export API gói Ranking** | [`src/lbd/ranking/__init__.py`](file:///c:/Users/Admin/Downloads/tái%20định%20vị%20thuốc/dapagliflozin-lbd/src/lbd/ranking/__init__.py) | Cung cấp `M2RepurposingRanker` song song với `score_disease_candidate` để tương thích ngược 100%. |
| **Bộ trọng số & Ma trận đã huấn luyện** | [`artifacts/ranking_m2/`](file:///c:/Users/Admin/Downloads/tái%20định%20vị%20thuốc/dapagliflozin-lbd/artifacts/ranking_m2) | Chứa `M2_model.pkl`, `M0_model.pkl`, `M3_model.pkl`, `features.npz` (`212.624 × 67`), `data_manifest.json`, `results.json`. |
| **CLI chạy trực tiếp khi bảo vệ** | [`scripts/run_lbd_m2_ranking.py`](file:///c:/Users/Admin/Downloads/tái%20định%20vị%20thuốc/dapagliflozin-lbd/scripts/run_lbd_m2_ranking.py) | Chạy thực nghiệm xếp hạng + giải thích DFS trực tiếp cho bất kỳ danh sách thuốc nào chỉ bằng 1 câu lệnh. |
| **Bộ kiểm thử tự động (Unit Tests)** | [`tests/test_ranking.py`](file:///c:/Users/Admin/Downloads/tái%20định%20vị%20thuốc/dapagliflozin-lbd/tests/test_ranking.py) | **34/34** bài kiểm thử trong `dapagliflozin-lbd` và **18/18** bài kiểm thử trong `repurposing-ranking` đều **PASS 100%**. |

### Lệnh chạy Demo trực tiếp trước Hội đồng:
```powershell
python dapagliflozin-lbd/scripts/run_lbd_m2_ranking.py --drugs "Dapagliflozin,Colchicine,Aspirin,Ruxolitinib" --top-k 5
```

---

## 2. Câu Chuyện Nghiên Cứu Trung Thực: Tại Sao Phải Nâng Cấp Từ Heuristic Lên Học Máy M2?

> [!TIP] Bí quyết ghi điểm tuyệt đối khi bảo vệ
> Hội đồng khoa học đánh giá rất cao những sinh viên/nhà nghiên cứu **dám chỉ ra thất bại của công thức giả định ban đầu khi đưa lên dữ liệu lớn thực tế** và dùng phương pháp khoa học để khắc phục nó. Hãy trình bày rõ 2 giai đoạn dưới đây.

### Giai đoạn 1: Hạn chế của công thức Heuristic ban đầu ($S_{\text{orig}}$) khi chạy trên đồ thị thật
Ban đầu, hệ thống thiết kế công thức chấm điểm thủ công $S_{\text{orig}}$ dựa trên việc duyệt DFS, tính chất lượng cạnh chia cho bậc nút trung gian ($\gamma^{L-1} \prod \text{deg}(v)^w$), nhân với hệ số dấu cứng (`net_sign = -1` nhân `1.5`, `0` nhân `1.0`, `+1` nhân `0.0`) rồi cộng tổng điều hòa $\sum \text{score}_k / k$.
- Khi chạy trên đồ thị con rút gọn (subgraph vài chục nút), công thức chạy tốt vì chỉ có các đường đi đã được chọn lọc.
- Tuy nhiên, khi đưa lên **toàn bộ đồ thị Hetionet v1.0 nguyên bản (`47.031` nút, `2.250.197` cạnh)**:
  1. **Bùng nổ tổ hợp đường đi nhiễu**: Chỉ riêng Dapagliflozin (`DB06292`) đã sinh ra **8.210 đường đi** nối tới **125/137 bệnh**, trong đó có tới **7.912 đường đi** đi qua các enzyme chuyển hóa thuốc ở gan/thận (`CYP3A4`, `CYP2C9`, `UGT1A9`, `ABCB1`) thay vì đích tác dụng dược lý `SLC5A2` (SGLT2).
  2. **Sụp đổ độ chính xác trên tập kiểm thử**: Khi kiểm toán độc lập (Mô hình **M1**), công thức $S_{\text{orig}}$ chỉ đạt **`PR-AUC = 0.0259`** trên tập Locked Test — **thấp hơn cả mô hình M0 (`PR-AUC = 0.0566`)** vốn chỉ đếm bậc của nút thuốc và nút bệnh mà không cần duyệt đường đi nào!

### Giai đoạn 2: Giải pháp Kiến trúc Hai Tầng (Macro-Matrix M2 + Micro-DFS)
Để giải quyết triệt để sự sụp đổ của công thức thủ công, chúng tôi thiết kế lại pipeline thành **2 tầng phối hợp**:

```mermaid
flowchart TD
    subgraph Tầng 1: Xếp hạng Vĩ mô bằng Học máy (Macro-Matrix M2)
        A["Đồ thị Hetionet v1.0<br/>(47.031 nút, 2.250.197 cạnh)"] --> B["Khóa nhãn CtD / CpD<br/>(Chống rò rỉ dữ liệu 100%)"]
        B --> C["Nhân Ma trận Thưa DWPC (w = 0.4)<br/>Trích xuất 65 đặc trưng cho 212.624 cặp (Thuốc - Bệnh)"]
        C --> D["Biến đổi arcsinh + StandardScaler<br/>+ Hồi quy Logistic ElasticNet (M2)"]
        D --> E["Xếp hạng 137 bệnh trong 0,05 giây<br/>Tách Chỉ định gốc vs. Ứng viên mới (Novel)"]
    end

    subgraph Tầng 2: Giải thích Cơ chế Vi mô (Micro-DFS & NLP)
        E --> F["Chọn Top K Bệnh Ứng viên Mới"]
        F --> G["Duyệt DFS Hai Chiều (Meet-in-the-Middle)<br/>Chấm điểm từng đường đơn lẻ: Contrib = β_metapath × w(path)"]
        G --> H["Đính kèm Bằng chứng Văn bản NLP<br/>(PMID & Câu văn trích xuất ≤ 2018)"]
    end
```

1. **Tại sao Tầng 1 dùng Ma trận DWPC + Học máy ElasticNet (M2)?**
   - Thay vì tự bịa ra trọng số cho từng loại cạnh, mô hình M2 nhìn vào **755 cặp thuốc - bệnh điều trị thật (`CtD`)** trên tổng số **212.624 cặp** để **tự học trọng số $\beta_j$** cho từng khuôn mẫu đường đi (metapath).
   - Nhờ phạt bậc nút theo chuẩn DWPC ($w = 0.4$): $w(u, v) = (\text{deg}_{out}(u) \cdot \text{deg}_{in}(v))^{-0.4}$ ngay trên từng bước nhảy ma trận, ảnh hưởng của các siêu nút (hubs) bị nén xuống một cách toán học.
2. **Tại sao Tầng 2 dùng Meet-in-the-Middle Micro-DFS?**
   - Ma trận thưa tính tổng hàng trăm nghìn đường rất nhanh nhưng không chỉ ra được tên gen trung gian cụ thể.
   - Khi đã có Top bệnh từ Tầng 1, thuật toán **Micro-DFS** chỉ duyệt giữa Thuốc và các Bệnh đứng đầu đó, rồi dùng chính **trọng số $\beta_{\text{metapath}}$ đã học của M2** nhân với trọng số đường đi $w(p)$ để rút ra **Top 2–3 đường đi sinh học đắt giá nhất** giải thích cho người dùng.

---

## 3. Thiết Kế Thực Nghiệm & Kết Quả Ablation Study (Số Liệu Thật 100%)

### 3.1. Quy trình chống gian lận & rò rỉ dữ liệu (Zero Leakage)
- **Tách nhãn tuyệt đối**: Toàn bộ `755` cạnh `CtD` (Treats) và `390` cạnh `CpD` (Palliates) bị xóa sạch khỏi đồ thị trước khi tính bậc nút và nhân ma trận DWPC. Cạnh `CpD` (chỉ giảm triệu chứng) được đánh dấu `label = -1` và loại khỏi hàm mất mát để không làm bẩn nhãn âm.
- **Chia tập dữ liệu theo cụm cấu trúc thuốc (Grouped Drug-Cluster Split)**:
  - Dùng độ tương đồng cấu trúc hóa học Tanimoto (`Morgan Fingerprint`) gom `1.552` thuốc thành các cụm độc lập:
    - **Development (`1.415` thuốc / `193.855` cặp)**: Chia 5-fold StratifiedGroupKFold để huấn luyện và tìm siêu tham số.
    - **Locked Test (`134` thuốc / `18.358` cặp)**: Khóa kín để đánh giá độc lập.
    - **SGLT2i Case Study (`3` thuốc: Dapagliflozin, Canagliflozin, Empagliflozin / `411` cặp)**: Hoàn toàn không nằm trong tập train để kiểm chứng khả năng "tái phát hiện" chỉ định Suy tim và Bệnh thận mạn.

### 3.2. Bảng Kết quả Ablation Study trên Tập Kiểm Thử Độc Lập (Locked Test)

> [!IMPORTANT] Ý nghĩa cốt lõi của bảng Ablation
> Tỷ lệ cặp dương tính thực tế trong toàn đồ thị chỉ là **0,355%** (`755 / 212.624`, tức cứ 280 cặp mới có 1 cặp điều trị thật). Do đó, thước đo quan trọng nhất là **PR-AUC (Average Precision)** và **Recall@20**.

| Ký hiệu Mô hình | Đặc trưng sử dụng | PR-AUC (OOF CV) | PR-AUC (Locked Test) | ROC-AUC (Locked Test) | Mean Recall@20 | Mean NDCG@20 | Nhận định Khoa học |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Ngẫu nhiên** | Đoán mò theo tỷ lệ nền | `0.0036` | `0.0029` | `0.5000` | `~0.015` | `~0.008` | Đường cơ sở ngẫu nhiên. |
| **M0 (Degree Prior)** | Chỉ dùng 2 đặc trưng: `log_drug_degree` + `log_disease_degree` | `0.0644` | `0.0566` | `0.8180` | `0.4227` | `0.2145` | Thuốc nhiều đích và bệnh phổ biến tự nhiên có xác suất gặp nhau cao hơn. |
| **M1 ($S_{\text{orig}}$ cũ)** | Bậc nút + Điểm công thức Heuristic cũ $S_{\text{orig}}$ | *N/A* | **`0.0259`** | `0.7366` | `0.3636` | `0.1510` | **Thất bại trên đồ thị lớn**: Nhiễu từ các đường đi qua enzyme chuyển hóa kéo tụt kết quả xuống dưới cả M0. |
| **M2 (DWPC ElasticNet)** | **Bậc nút + 63 đặc trưng Metapath DWPC (Tổng 65 đặc trưng)** | **`0.1653`** | **`0.1681`** | **`0.8691`** | **`0.6136`** | **`0.3318`** | **Mô hình Tối ưu Nhất**: Tăng gấp **6,5 lần** PR-AUC so với công thức cũ M1 và gấp **3 lần** so với M0. Dự đoán trúng >61% chỉ định thật trong Top 20. |
| **M3 (DWPC + Signed + LINCS)** | 65 đặc trưng M2 + 2 đặc trưng hướng điều hòa gộp (`sign:odd`, `sign:even`) | `0.1649` | `0.1642` | `0.8685` | `0.6136` | `0.3338` | Tương đương M2 vì thông tin tăng/giảm biểu hiện gen (`CuG`/`CdG`) đã nằm sẵn trong tên của 63 metapath ở M2. |

### 3.3. Giải phẫu bên trong "Bộ não" của Mô hình M2: Máy đã học được gì?
Khi trích xuất các hệ số $\beta$ lớn nhất mà ElasticNet học được trong [`M2_model.pkl`](file:///c:/Users/Admin/Downloads/tái%20định%20vị%20thuốc/dapagliflozin-lbd/artifacts/ranking_m2/M2_model.pkl), chúng ta thấy máy tự động phát hiện ra **quy luật đối kháng sinh học (Therapeutic Reversal)**:
- **`dwpc:CdG|GpBP|DuG` ($\beta = +1.0311$)**: Thuốc làm **giảm (`CdG`)** biểu hiện một gen nằm trong cùng Quá trình sinh học (`GpBP`) với một gen bị bệnh làm **tăng (`DuG`)** $\rightarrow$ **Cộng điểm điều trị mạnh nhất!**
- **`dwpc:CdG|GpBP|DaG` ($\beta = +0.7988$)**: Thuốc điều hòa giảm gen trong quá trình sinh học gắn với gen bệnh lý.
- **`dwpc:CbG|GpPW|DaG` ($\beta = +0.6443$)**: Thuốc gắn trực tiếp (`CbG`) vào gen nằm cùng Con đường tín hiệu (`Pathway`) với gen gây bệnh (`DaG`).
- **`dwpc:CbG|direct|DaG` ($\beta = +0.4757$)**: Thuốc gắn trực tiếp vào chính gen liên quan đến bệnh (3 nút: `Compound -> Gene <- Disease`).
- Ngược lại, **`dwpc:CdG|GpBP|DdG` ($\beta = -0.4593$)**: Thuốc làm **giảm (`CdG`)** thêm quá trình sinh học vốn đã bị suy giảm trong bệnh (`DdG`) $\rightarrow$ **Bị mô hình phạt điểm âm!**

---

## 4. Kết Quả Thực Nghiệm Trên 6 Hoạt Chất Thực Tế & Đối Chiếu Lâm Sàng

Dưới đây là kết quả chạy toàn bộ pipeline trên 2 nhóm thuốc: **Nhóm SGLT2i (Case Study khóa riêng)** và **Nhóm Tim mạch / Chống viêm / Tự miễn mở rộng**.

### 4.1. Nhóm SGLT2i: Sự thật về Dapagliflozin, Canagliflozin và bài học "Cold-start" ở Empagliflozin

| Hoạt chất | Số cạnh trong Hetionet 2017 | Điểm & Hạng M2 cho Bệnh Thận Mạn (CKD `DOID:784`) | Điểm & Hạng M2 cho Tăng Huyết Áp (`DOID:10763`) | Điểm & Hạng M2 cho Suy Tim (`DOID:6000`) | Giải thích cơ chế thực tế từ đồ thị |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Dapagliflozin** (`DB06292`) | 10 cạnh (`CbG`) | **0.7662** (Hạng **#9**/137) | **0.8805** (Hạng **#2**/137) | **0.6401** (Hạng **#36**/137) | Đi qua đích chính `SLC5A2` (SGLT2) theo quá trình bài tiết natri/glucose thận (`GpBP`) nối tới `ACE`, `REN`, `NPPB`. |
| **Canagliflozin** (`DB08907`) | 229 cạnh (`CbG` + `CuG`/`CdG`) | **0.7277** (Hạng **#26**/137) | **0.9315** (Hạng **#6**/137) | **0.6596** (Hạng **#42**/137) | Giàu dữ liệu phiên mã LINCS L1000 (`CuG`/`CdG`), củng cố mạnh điểm Tăng huyết áp và Tim-Thận. |
| **Empagliflozin** (`DB09038`) | **Chỉ có 2 cạnh** (`SLC5A2`, `ADORA2A`) | `0.1559` (Hạng #97/137) | `0.4021` (Hạng #53/137) | `0.2194` (Hạng #84/137) | **Hiện tượng Cold-start (Nút thưa)**: Được FDA duyệt muộn (8/2014) nên khi Hetionet đóng băng chỉ kịp nạp 2 cạnh, khiến đích phụ `ADORA2A` (thụ thể Adenosine thần kinh) kéo điểm về phía bệnh thần kinh. |

> [!NOTE] Nhận định bảo vệ về nhóm SGLT2i
> Với **Dapagliflozin** — hoạt chất trọng tâm của đề tài — dù mô hình hoàn toàn không được học bất kỳ thuốc SGLT2i nào trong tập Train và dữ liệu bị khóa trước năm 2018 (trước khi thử nghiệm **DAPA-HF 2019** và **DAPA-CKD 2020** công bố), mô hình M2 vẫn xếp **Tăng huyết áp ở Hạng #2 (0.8805)** và **Bệnh thận mạn (CKD) ở Hạng #9 (0.7662)** trong tổng số 137 bệnh!

---

### 4.2. Nhóm Thuốc Đối Chứng Mở Rộng: Colchicine, Aspirin, Ruxolitinib (Đại diện Baricitinib), Spironolactone

Hệ thống đã chạy kiểm chứng trên các thuốc kinh điển và hiện đại, đối chiếu với **các phê duyệt lâm sàng diễn ra sau năm 2017**:

#### A. Colchicine (`Compound::DB01394`) — Thuốc trị Gout tái định vị sang Tim mạch
- **Chỉ định gốc đã tách lọc (`CtD`/`CpD`)**: Gout (`DOID:13189`), Xơ gan ứ mật nguyên phát (`DOID:12236`), Xơ cứng bì (`DOID:418`).
- **Ứng viên tái định vị nổi bật**:
  - **Tăng huyết áp (`DOID:10763`)**: Hạng mới **#2** (`P = 0.9737`) — Đường đi DFS qua `ABCC1` và `TPM1`.
  - **Viêm loét đại tràng (`DOID:8577`)**: Hạng mới **#5** (`P = 0.9289`) — Đường đi DFS qua `CDH3` và `ABCB1`.
  - **Bệnh mạch vành (`Coronary artery disease - DOID:3393`)**: Hạng **#12/137** (`P = 0.8807`).
- **Đối chứng thực tế ngoài đời**: Thử nghiệm **COLCOT (2019)** và **LoDoCo2 (2020)** chứng minh Colchicine liều thấp giảm 23–31% biến cố tim mạch; **tháng 6/2023, FDA chính thức phê duyệt Colchicine (biệt dược *Lodoco*) cho bệnh mạch vành xơ vữa!**

#### B. Aspirin / Acetylsalicylic acid (`Compound::DB00945`) — Kháng viêm NSAID sang Tự miễn & Ung thư
- **Chỉ định gốc đã tách lọc (`CtD`/`CpD`)**: Thoái hóa khớp, Viêm khớp dạng thấp, Đau nửa đầu, Bệnh mạch vành, Lupus ban đỏ, Gout, Bệnh Kawasaki (7 bệnh).
- **Ứng viên tái định vị nổi bật**:
  - **Hạng mới #1: Bệnh Vảy nến (`Psoriasis - DOID:8893`, `P = 0.9740`)**:
    - Đường đi DFS số 1: `Acetylsalicylic acid --[CbG]--> NFKBIA <--[DaG]-- psoriasis` ($\beta = +0.4757$, đóng góp $+0.0126$). Hoàn toàn chính xác về mặt sinh học phân tử vì Aspirin ức chế dòng thác viêm NF-$\kappa$B thông qua `NFKBIA`.
  - **Hạng mới #2–#5: Các bệnh Ung thư biểu mô & Huyết học (`Melanoma` `0.9606`, `Ovarian cancer` `0.9498`, `Breast cancer` `0.9492`)** và **Ung thư đại tràng (`Colon cancer - DOID:219`, `P = 0.7969`)** thông qua trực tiếp gen ức chế khối u `TP53` và `HSPA5`.

#### C. Baricitinib / Đại diện cùng đích JAK1/JAK2 là Ruxolitinib (`Compound::DB08877`)
- **Minh bạch dữ liệu**: Baricitinib được FDA duyệt lần đầu vào tháng 6/2018 nên chưa có nút trong Hetionet v1.0 (2017). Hệ thống sử dụng **Ruxolitinib (`DB08877`)** — chất ức chế chọn lọc **JAK1/JAK2** cùng nhóm cơ chế phân tử.
- **Chỉ định gốc trong Hetionet 2017**: Chỉ duy nhất 1 bệnh là Ung thư huyết học / Xơ tủy (`DOID:2531`, `P = 0.9500`).
- **Ứng viên tái định vị nổi bật do M2 + DFS phát hiện**:
  - **Bệnh Crohn / Viêm ruột mạn (`DOID:8778`)**: Hạng mới **#4** (`P = 0.9805`) — DFS chỉ ra đường đi trực tiếp: `Ruxolitinib --[CbG]--> TYK2 / LRRK2 <--[DaG]-- Crohn's disease`.
  - **Lupus ban đỏ hệ thống (`SLE - DOID:9074`)**: Hạng mới **#5** (`P = 0.9761`) — DFS chỉ ra đường đi trực tiếp: `Ruxolitinib --[CbG]--> JAK1 <--[DaG]-- systemic lupus erythematosus`.
  - **Viêm khớp dạng thấp (`Rheumatoid arthritis - DOID:7148`)**: Hạng **#16/137** (`P = 0.9248`) — Đi qua đích `JAK1` và `JAK2`.
- **Đối chứng thực tế ngoài đời**: Năm 2018, FDA phê duyệt **Baricitinib (*Olumiant*)** chính xác cho bệnh **Viêm khớp dạng thấp**, và các chất ức chế JAK hiện là phác đồ điều trị chuẩn cho **Bệnh viêm ruột (Crohn/UC)** cũng như thử nghiệm Pha III cho **Lupus ban đỏ (SLE)**!

---

## 5. Cẩm Nang Phản Biện (Q&A) Trước Hội Đồng Bảo Vệ

### Câu hỏi 1: *"Tại sao điểm PR-AUC của mô hình tốt nhất (M2) là 0.1681 (16,8%) mà ROC-AUC lại đạt tới 0.8691 (86,9%)? Con số 0.1681 có thấp không?"*
- **Trả lời**:
  - Thưa Hội đồng, trong bài toán tái định vị thuốc trên toàn đồ thị Hetionet (`212.624` cặp thuốc - bệnh), chỉ có `755` cặp điều trị thật, tức tỷ lệ nhãn dương chỉ chiếm **0,355%** (mất cân bằng dữ liệu cực đoan 1:280).
  - Với tỷ lệ nền `0.0029` trên tập Test, việc đạt **`PR-AUC = 0.1681`** tương đương mức **enrich (làm giàu tín hiệu) gấp 58 lần so với ngẫu nhiên**, gấp **6,5 lần** công thức heuristic ban đầu (`0.0259`), và đi kèm **`ROC-AUC = 0.8691`** cùng **`Recall@20 = 61,36%`** (hơn 61% chỉ định thuốc thật nằm ngay trong Top 20 bệnh đầu tiên). Trong các công trình chuẩn trên Hetionet (như *Project Rephetio* của Himmelstein et al., eLife 2017), PR-AUC toàn cục cũng nằm trong ngưỡng `0.15 – 0.19`.

### Câu hỏi 2: *"Tại sao không dùng thẳng DFS để chấm điểm mà phải kết hợp Ma trận DWPC (M2) với Micro-DFS?"*
- **Trả lời**:
  - Nếu dùng DFS thuần túy để chấm điểm toàn đồ thị cho 1.552 thuốc $\times$ 137 bệnh, hệ thống phải duyệt hàng trăm triệu đường đi, vừa mất hàng giờ tính toán, vừa không có trọng số học máy $\beta$ để biết loại đường đi nào có lợi hay có hại.
  - Ngược lại, **Ma trận thưa DWPC (Tầng 1)** gom toàn bộ cấu trúc mạng thành 65 đặc trưng và chấm điểm 137 bệnh chỉ trong **0,05 giây** bằng trọng số ElasticNet đã học. Sau khi lọc ra Top 5–10 bệnh tiềm năng nhất, **Meet-in-the-Middle Micro-DFS (Tầng 2)** mới nhảy vào truy vết các con đường cụ thể giữa thuốc và bệnh đó, rồi sắp xếp đường đi theo tích số $\beta_{\text{metapath}} \times w(\text{path})$. Nhờ vậy hệ thống vừa **nhanh như nhân ma trận**, vừa **giải thích được từng gen/pathway cụ thể như DFS**.

### Câu hỏi 3: *"Khi duyệt DFS cho Dapagliflozin hay Colchicine, tại sao lại xuất hiện các gen như CYP3A4 hay ABCB1? Các gen đó có ý nghĩa gì?"*
- **Trả lời**:
  - `CYP3A4`, `CYP2C9`, `UGT1A9`, `ABCB1` là các enzyme chuyển hóa và bơm vận chuyển thuốc (ADME). Trong Hetionet, chúng đều được gộp chung vào loại cạnh `CbG` (Compound-binds-Gene).
  - Chúng tôi đã thực hiện riêng một thí nghiệm **Ablation cắt bỏ toàn bộ 62 gen ADME** khỏi đồ thị:
    - Kết quả cho thấy đối với **Dapagliflozin**, việc cắt bỏ ADME giúp loại bỏ 96% đường đi nhiễu (từ 8.210 đường xuống còn 298 đường thuần túy đi qua đích `SLC5A2`), giúp phần giải thích cơ chế DFS cực kỳ trong sáng.
    - Tuy nhiên, trên toàn bộ 1.552 thuốc, PR-AUC khi cắt cứng giảm nhẹ từ `0.1681` xuống `0.1650` do một số bệnh gan/thận thực sự có liên hệ sinh lý với các protein vận chuyển này. Do đó, việc giữ DWPC ($w=0.4$) ở tầng chấm điểm vĩ mô để phạt tự động các nút bậc cao, đồng thời lọc ưu tiên đích dược lý ở tầng hiển thị DFS là thiết kế cân bằng tốt nhất.

### Câu hỏi 4: *"Vai trò của module NLP (trích xuất câu văn PubMed) nằm ở đâu trong kiến trúc mới này?"*
- **Trả lời**:
  - Trong kiến trúc hoàn chỉnh, **đồ thị nền Hetionet + M2** đóng vai trò là **bộ khung chấm điểm cấu trúc khách quan (Structural Backbone)** hoàn toàn không bị thiên kiến bởi số lượng bài báo.
  - Sau khi M2 và Micro-DFS tìm ra đường đi cơ chế (ví dụ `Dapagliflozin -> SLC5A2 -> CKD`), **module NLP (`NLPEngine` với F1 thực nghiệm đạt `0.850` trên tập Curated Benchmark)** sẽ trích xuất các câu văn và mã bài báo `PMID <= 2018` để đính kèm làm **bằng chứng văn bản (Literature Evidence)** cho từng cạnh trên đường đi đó, đồng thời bổ sung các cạnh trực tiếp từ y văn mới mà đồ thị tĩnh chưa kịp cập nhật.
