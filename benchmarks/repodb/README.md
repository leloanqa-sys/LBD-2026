# Benchmark repoDB cho ranking trên KG thật

**Mới 04/11/2026:** [Kết quả giữ nguyên checkpoint M2 của người dùng](ORIGINAL_CHECKPOINT_RESULTS_20261104.md). Chạy bằng `benchmarks/repodb/.venv/Scripts/python.exe -u benchmarks/repodb/evaluate_original_checkpoint.py`. File `LATEST_ORIGINAL_CHECKPOINT_RUN.txt` trỏ kết quả riêng; các kết quả retrain trước đây vẫn giữ nguyên.

Bộ chạy độc lập, không import hoặc sửa `src/lbd`, scripts, configs, graph hay model hiện có. Huấn luyện **model mới theo đặc trưng M0/M2**, không lấy metric của checkpoint LBD cũ để gọi là external test.

## Đọc kết quả

`LATEST_RUN.txt` chứa đường dẫn run mới nhất. Trong run:

- `report.json`: kết quả fixed split, coverage và giới hạn.
- `grouped_diagnostic.json`: kiểm tra độ ổn định 5-fold theo nhóm, thực hiện sau khi xem fixed split; chỉ là phân tích bổ sung.
- `heldout_rankings.csv`: score từng thuốc–bệnh test, gold nếu có, trạng thái đã biết trong KG.
- `gold_pairs.json`, `mapping_audit.json`, `conflicting_pairs.json`: nhãn và mọi trường hợp bỏ qua/xung đột.
- `manifest.json`, `protocol.json`, `split.json`: hash nguồn, thiết kế và phân chia train/test.
- `M0_degree.pkl`, `M2_DWPC.pkl`: model mới huấn luyện bằng repoDB; score chưa calibration.

Xem [báo cáo diễn giải](RESULTS_20261103.md).

## Chạy lại

Từ thư mục `dapagliflozin-lbd`:

```powershell
python -m venv --system-site-packages benchmarks/repodb/.venv
benchmarks/repodb/.venv/Scripts/python.exe -m pip install -r benchmarks/repodb/requirements-benchmark.txt
powershell -File benchmarks/repodb/download_sources.ps1
benchmarks/repodb/.venv/Scripts/python.exe -m pytest benchmarks/repodb/test_benchmark.py -q
benchmarks/repodb/.venv/Scripts/python.exe -u benchmarks/repodb/run_benchmark.py
benchmarks/repodb/.venv/Scripts/python.exe -u benchmarks/repodb/run_grouped_diagnostic.py
```

Script chính mặc định **tính lại feature từ KG thật**. Có thể dùng `--reuse-features <thư_mục_feature>` để tiếp tục từ stage đã hoàn tất, vẫn kiểm graph hash. Không tự tải graph: cần `data/raw/hetionet-v1.0-nodes.tsv` và `hetionet-v1.0-edges.sif.gz`.

`feature_builder_reference.py` trích nguyên các hàm tạo đặc trưng từ framework nghiên cứu trong workspace; hash nguồn gốc ghi trong file. Không chạy `main()`/training của framework cũ. Feature builder có xuất y Hetionet để giữ đúng code tham chiếu, nhưng benchmark **không đọc y/valid/eligible đó**: nhãn huấn luyện/đánh giá chỉ từ repoDB.

## Protocol

- repoDB: snapshot RData từ repository gốc của tác giả, pinned commit; không tuyên bố là bản hosted 2023.
- Disease Ontology: snapshot pinned; ánh xạ exact UMLS CUI tới disease IDs của KG, không fuzzy tên hoặc mở rộng ancestor. Một CUI tới nhiều bệnh thì loại; Approved và failure cùng cặp sau ánh xạ thì loại.
- Unknown không thành negative. Terminated/Withdrawn/Suspended là failure-status proxy, không phải xác nhận vô hiệu sinh học.
- Feature: log degree + 63 DWPC, w=0.4; loại CtD/CpD trước tính degree/matrix; không dùng label edges làm feature.
- Split theo connected components CrC, seed=42, 20% số **nhóm** cho test. Không bảo đảm 20% số thuốc/cặp vì kích thước nhóm không đều.
- Model L2 logistic, C=1 cố định trước chạy; transform/scaler fit trên train. Không chỉnh hyperparameter hoặc seed theo test.
- Recall/nDCG/MRR xếp trên toàn 137 bệnh KG (hoặc loại CtD/CpD ở track novel), không chỉ trên những cặp có nhãn. AP/AUROC chỉ tính trên labeled pairs và báo prevalence.
- Model xáo nhãn chỉ xáo train để kiểm soát. Bootstrap theo drug cluster, không coi các thuốc trong cùng cluster là độc lập.

## Phạm vi

Đây là benchmark ranking trên feature KG thật, chưa phải raw-abstract→NLP→KG→ranking end-to-end. Không phải temporal discovery trước 2019. DO snapshot mới và repoDB gốc không cung cấp temporal ground truth đầy đủ; nhãn novelty chỉ có nghĩa không có CtD/CpD trong KG.

## Nguồn và ghi công

- [repoDB — mã nguồn của Brown và Patel](https://github.com/adam-sam-brown/repoDB), [bài báo Scientific Data](https://www.nature.com/articles/sdata201729).
- [Trang dữ liệu repoDB](https://unmtid-shinyapps.net/shiny/repodb/) nêu nguồn DrugCentral/ClinicalTrials.gov và giấy phép CC BY 4.0. Khi công bố cần ghi công các nguồn liên quan theo điều kiện dữ liệu.
- [Disease Ontology](https://github.com/DiseaseOntology/HumanDiseaseOntology).
- [Hetionet](https://github.com/hetio/hetionet).

Lần đầu đọc RData bằng encoding mặc định gặp lỗi; đã sửa bộ benchmark mới để đọc UTF-8 và tiếp tục từ feature stage, có nhật ký `stage_status.json` trong run đầu. Không sửa dữ liệu gốc hoặc dùng dữ liệu giả thay thế.
