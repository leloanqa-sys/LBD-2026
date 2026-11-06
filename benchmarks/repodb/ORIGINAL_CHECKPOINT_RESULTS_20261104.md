# Model M2 gốc trên repoDB — kết quả thực thi 04/11/2026

**Đã chạy checkpoint gốc, không huấn luyện lại. Có tín hiệu ranking trên thuốc ngoài fit, nhưng số ca nhỏ và khả năng phân biệt Approved/failed-status còn yếu.**

## Model và quy trình

- Model: `artifacts/ranking_m2/M2_model.pkl`; SHA256 `57d4d6d95454fd05fddf2d99c010b68de5e8e0626f38a499b4b67fd20d218ad6`. Hash trước/sau bằng nhau.
- Dùng nguyên scaler, arcsinh scale, coefficients, intercept và 65 feature (63 DWPC/metapath + 2 degree), w=0.4. Không fit hoặc chỉnh threshold/hyperparameter.
- Feature và raw Hetionet 47.031 node / 2.250.197 cạnh khớp hash của manifest gốc.
- Gold giữ nguyên 636 cặp từ benchmark repoDB trước; không sửa mapping/split theo kết quả mới. Unknown không gán negative.
- Ranking trên toàn 137 bệnh; track novel-relative-KG loại CtD/CpD khỏi candidate set.
- M0 là checkpoint degree gốc; random dùng seed42. Không so sánh trực tiếp chênh lệch với model retrain ở lần trước như hiệu quả cải tiến, vì training labels và split khác nhau.

## Kết quả thuật toán

Recall là macro theo thuốc có ít nhất một positive hợp lệ. Số thuốc của subset có thể lớn hơn số thuốc góp vào Recall nếu một số thuốc chỉ có nhãn failed-status.

### Toàn bộ repoDB ánh xạ được — có overlap huấn luyện
636 cặp, 431 thuốc, 452 positive; overlap fit groups=92.
- Original_M2: Recall@10 toàn bộ **57.00%**; ngoài CtD/CpD **46.94%**; MRR toàn bộ 0.3316.
- Original_M0_degree: Recall@10 toàn bộ **20.15%**; ngoài CtD/CpD **22.40%**; MRR toàn bộ 0.0751.
- Random: Recall@10 toàn bộ **8.30%**; ngoài CtD/CpD **6.17%**; MRR toàn bộ 0.0517.
M2 toàn bộ: Recall@5 45.43%, Recall@20 70.95%; 347 thuốc góp vào Recall / 452 positive. CI95 Recall@10 bootstrap theo cluster [43.70%, 60.99%].
M2 ngoài CtD/CpD: chỉ 127 thuốc / 149 positive. CI95 [38.46%, 60.47%]. Không diễn giải điểm cao của subset rất nhỏ thành khả năng tổng quát chắc chắn.

### Nhóm thuốc thuộc locked test gốc — không fit
34 cặp, 23 thuốc, 18 positive; overlap fit groups=0.
- Original_M2: Recall@10 toàn bộ **53.57%**; ngoài CtD/CpD **75.00%**; MRR toàn bộ 0.3381.
- Original_M0_degree: Recall@10 toàn bộ **14.29%**; ngoài CtD/CpD **25.00%**; MRR toàn bộ 0.0686.
- Random: Recall@10 toàn bộ **0.00%**; ngoài CtD/CpD **0.00%**; MRR toàn bộ 0.0336.
M2 toàn bộ: Recall@5 46.43%, Recall@20 67.86%; 14 thuốc góp vào Recall / 18 positive. CI95 Recall@10 bootstrap theo cluster [28.12%, 80.02%].
M2 ngoài CtD/CpD: chỉ 4 thuốc / 4 positive. CI95 [25.00%, 100.00%]. Không diễn giải điểm cao của subset rất nhỏ thành khả năng tổng quát chắc chắn.

### Mọi nhóm ngoài fit — test gốc và reserved
36 cặp, 25 thuốc, 20 positive; overlap fit groups=0.
- Original_M2: Recall@10 toàn bộ **59.38%**; ngoài CtD/CpD **83.33%**; MRR toàn bộ 0.3584.
- Original_M0_degree: Recall@10 toàn bộ **12.50%**; ngoài CtD/CpD **16.67%**; MRR toàn bộ 0.0632.
- Random: Recall@10 toàn bộ **6.25%**; ngoài CtD/CpD **16.67%**; MRR toàn bộ 0.0379.
M2 toàn bộ: Recall@5 53.12%, Recall@20 71.88%; 16 thuốc góp vào Recall / 20 positive. CI95 Recall@10 bootstrap theo cluster [33.33%, 83.33%].
M2 ngoài CtD/CpD: chỉ 6 thuốc / 6 positive. CI95 [40.00%, 100.00%]. Không diễn giải điểm cao của subset rất nhỏ thành khả năng tổng quát chắc chắn.

## Phân biệt ranking và Approved/failed-status

Trên toàn 636 cặp, M2 AP≈0,7042 với prevalence≈0,7107; AUROC≈0,5076. Dù Recall@10 57% trên candidate universe 137 bệnh, khả năng phân biệt Approved với failed-status ở tập labeled pairs gần ngẫu nhiên. Hai metric trả lời hai câu hỏi khác nhau.

Ở locked test gốc: AP≈0,6245, AUROC≈0,5938; ở mọi nhóm ngoài fit: AP≈0,6474, AUROC≈0,6063. Đây là tín hiệu vừa phải với sample nhỏ; không đủ kết luận clinical success.

## Kiểm tra overlap và thay đổi nhãn

- Trong 636 cặp repoDB, **517 cặp từng được dùng fit** model gốc; 31 cặp thuộc locked test row; 88 cặp excluded/reserved.
- Có **174 cặp tham gia fit có nhãn khác repoDB**. Nguyên nhân cần phân tích từng ca; model gốc dùng CtD positive và closed-world unknown=0, trong khi repoDB dùng Approved/failed-status. Không xem hai bộ nhãn là tương đương.
- 406/431 thuốc thuộc development gốc. Vì vậy 57,00% trên all-mapped không phải external independent validation.
- Locked test gốc không dùng để fit, nhưng kết quả của tập đó đã được xem trong thí nghiệm trước; không gọi là blind test mới hoàn toàn.
- repoDB và CtD có thể chia sẻ nguồn upstream; tách nhóm fit giảm overlap nhưng không biến nguồn thành hoàn toàn độc lập.

## Dapagliflozin

Top score của checkpoint:
- #1 hypertension: 0.9602.
- #2 type 2 diabetes mellitus: 0.9584.
- #3 lung cancer: 0.9204.
- #4 breast cancer: 0.8884.
- #5 Parkinson's disease: 0.8768.

CKD (`Disease::DOID:784`) đứng hạng 9. HF theo ID DOID:9333 không có trong candidate universe hiện tại; cần crosswalk kiểm chứng nếu muốn so với ID khác. RepoDB gốc chỉ có gold T2D cho Dapagliflozin, hạng 2. Do đó không tính được Recall HF/CKD từ gold này.

T2D vẫn xuất hiện trong track novel-relative-KG vì không có cạnh CtD/CpD tương ứng cho Dapagliflozin ở snapshot. Đây minh họa rõ **absent-in-KG không đồng nghĩa chỉ định mới ngoài thực tế**. Score cao của các bệnh khác chỉ là output model, không phải xác nhận hiệu quả điều trị.

## Kiểm chứng thực thi

- Đối chiếu 137 bệnh × 3 thuốc (Dapagliflozin, Aspirin, Colchicine) với API `M2RepurposingRanker.rank_drug()`: khớp trong sai số làm tròn 0,00005. API check tắt DFS để chỉ kiểm score; novelty filter được thực hiện riêng từ raw graph thật.
- So dự đoán trên toàn locked test với `M2_test_predictions.npy` gốc: max delta ≈2,09e-7.
- Tính lại 24 giá trị Recall@10 từ CSV predictions/rankings cho 4 subsets × 3 model × 2 tracks: khớp report.
- Hash nguồn/checkpoint/feature/gold khớp; checkpoint giữ nguyên. Runtime inference + metric + export khoảng 18,98 giây sau import.
- Một lần khởi động gặp DLL import bị Application Control chặn; lần thử lại import thành công, không đổi model hay dependency.

## Kết luận

Model gốc của bạn đạt kết quả ranking tốt hơn baseline degree/random trên phần thuốc không tham gia fit. Con số phù hợp để báo hiện tại là **Recall@10 53,57% trên 14 thuốc có positive (18 positive), thuộc locked test gốc**. Nhóm ngoài CtD/CpD đạt 75% nhưng chỉ 4 ca dương.

Đây là bằng chứng thử nghiệm có giới hạn, chưa là nghiệm thu hệ thống: coverage repoDB thấp, sample ngoài fit nhỏ, label policy khác nhau và phân biệt Approved/failed-status còn yếu. Không suy metric này sang NLP, KG admission hay discovery theo thời gian.

[Report JSON](original_checkpoint_runs/20261104T020021Z/report.json) · [Predictions gold pairs](original_checkpoint_runs/20261104T020021Z/repoDB_predictions.csv) · [Ranking đủ 137 bệnh](original_checkpoint_runs/20261104T020021Z/rankings_all_mapped_drugs.csv) · [Verification](original_checkpoint_runs/20261104T020021Z/verification.json) · [Manifest](original_checkpoint_runs/20261104T020021Z/manifest.json)

Chạy lại từ root LBD: `benchmarks/repodb/.venv/Scripts/python.exe -u benchmarks/repodb/evaluate_original_checkpoint.py`.
