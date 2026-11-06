# Đánh giá kiến trúc, độ trưởng thành và pipeline — 03/11/2026

## Phạm vi và kết luận

Đọc README, architecture, data dictionary, experiment protocol, team design guide, V7 và remediation; đối chiếu mã canonical, các runner, UI, dữ liệu và artifact hiện có. Repository thực tế là `dapagliflozin-lbd`, nhánh **Dev**, HEAD `440d7f1`, có nhiều thay đổi chưa commit. Không thấy nhánh `lbd` trong refs local/remote đang có; báo cáo áp dụng cho working tree này, không phải một nhánh lbd đã được xác minh. Không fetch remote, không sửa code, corpus hoặc graph trong lần đánh giá này.

**Đây là nguyên mẫu nghiên cứu có nhiều module hoạt động thật, nhưng chưa phải pipeline tái định vị thuốc end-to-end được nghiệm thu.** NLP là phân khu phát triển sâu nhất về cấu trúc; M2 có bằng chứng thực nghiệm định lượng rõ hơn thuật toán cũ; KG và output là các điểm nghẽn tích hợp. UI và demo còn tạo ấn tượng kết quả hoàn chỉnh vượt quá dữ liệu thực tế.

Mục tiêu theo docs: dùng y văn trước 31/12/2018 và graph nền để tìm giả thuyết bệnh mới cho Dapagliflozin, loại chỉ định lịch sử T2D, trả ranking kèm đường suy luận và bằng chứng PMID, rồi đánh giá bằng kết quả độc lập. Cấu trúc hợp lý là **4 phân khu xử lý: thu thập → NLP → KG → thuật toán**, với **output/evaluation là tầng tiêu thụ và kiểm chứng**.

Thang trưởng thành dùng ở đây là nhận định kỹ thuật, không phải tỷ lệ hoàn thành: M0 ý tưởng; M1 có thành phần chạy riêng; M2 nguyên mẫu tích hợp một phần; M3 tái lập được toàn luồng trên dữ liệu thật với kiểm chứng độc lập; M4 vận hành ổn định. Không phân khu nào được chứng nhận M4.

## 1. Cấu trúc và công nghệ

- `src/lbd/data`: tạo DocumentRecord, cutoff, chuẩn hóa, loader snapshot. `fetch_pubmed.py` hiện không phải collector HTTP hoàn chỉnh; tải live nằm trong script demo.
- `src/lbd/nlp`: NLPEngine canonical, zoning/grounding/coreference, classifier scikit-learn, event và polarity rules, verifier Groq tùy chọn, fusion Beta-Binomial. PubTator được cấp annotation ngoài; canonical engine không tự gọi PubTator khi chạy mặc định.
- `src/lbd/graph`: NetworkX MultiDiGraph, merge và DFS. Neo4j xuất hiện trong config/design nhưng đường chạy đã kiểm tra dùng NetworkX.
- `src/lbd/ranking`: công thức điểm đường có dấu và M2 dùng feature DWPC được tính sẵn, arcsinh, scaler và classifier học sẵn. Đây là hai tuyến khác nhau.
- `src/lbd/evaluation`: metric và ground truth cho tuyến cũ; kết quả thực nghiệm M2 nằm trong artifacts được chuyển vào từ phân hệ ngoài.
- `ui`: HTML/JavaScript tĩnh, Python SimpleHTTPRequestHandler; chưa có API nối tới engine hoặc artifact.
- Công nghệ nền: Python, NumPy/SciPy/scikit-learn, NetworkX, requests/urllib, JSON Schema, YAML, JSONL/CSV/NPZ/pickle. Không thấy orchestration thống nhất hoặc run manifest xuyên A→E.

Ưu điểm: tách domain module khá rõ, có schema, checkpoint, config và tests. Hạn chế: phần việc nghiệp vụ vẫn nằm trong scripts; configs và contracts có hai bản; artifacts dùng tên cố định; code và corpus/checkpoint/evaluator trải qua nhiều thư mục ngoài package.

## 2. Độ trưởng thành từng phân khu

### Thu thập — M2, dữ liệu thật có sẵn nhưng ETL chưa khép kín

Có 664 dòng `papers_pre2018.jsonl`, Contract A và hàm cutoff từ chối ngày không hợp lệ. Có corpus local và annotation cache hỗ trợ offline. Tuy nhiên, chưa có một collector canonical chịu trách nhiệm query, pagination, retry/rate limit, checkpoint tải, dedup, manifest và kiểm chứng temporal trước bàn giao cho NLP.

`scripts/run_nlp_extraction.py` đọc CSV thay vì dùng Contract A làm nguồn chuẩn. Nó đọc `print_year` vào biến `year` nhưng không kiểm tra cutoff và không truyền năm qua adapter. Demo nhận `year_cutoff` nhưng không dùng để lọc query hoặc record, đồng thời gán năm mặc định 2018 nếu thiếu. Vì vậy có guard trong thư viện không đồng nghĩa mọi entry point đã được bảo vệ.

Để lên M3: một đầu vào A duy nhất, mọi record phải qua temporal gate; lưu query, fetched_at, hash, nguồn và lý do reject; có thể resume và chạy lại cùng snapshot.

### NLP — M2 khá phát triển, chưa có nghiệm thu end-to-end độc lập

Có zoning, entity grounding nội bộ/ngoài, classifier học sẵn, lớp event/polarity, verbatim postprocess, dedup PMID và fusion. Đây là code thực, không thể quy toàn bộ NLP thành mock. Live LLM mặc định tắt; lỗi API trả symbolic fallback có nhãn. Classifier unavailable cũng có cảnh báo degraded mode.

Các tuyến chưa thống nhất: canonical V7 dùng classifier gate; benchmark V6 symbolic có hành vi khác. Không thể gộp metric hai tuyến. Gold với spans/IDs cho trước không đo NER từ văn bản thô, direction hoặc admission KG đầy đủ. Chưa tái chạy corpus/benchmark khoa học trong audit này.

Dữ liệu đầu ra canonical hiện chỉ có **4 quan hệ, cùng PMID 25516459**, trong khi A có 664 record. Không đủ căn cứ gọi đó là output toàn corpus; có thể là lần chạy giới hạn/ghi đè. Một quan hệ `Dapagliflozin INHIBITS INS` dẫn câu nói về tác dụng độc lập insulin: đây là ứng viên lỗi gán quan hệ cần review, dù evidence tồn tại nguyên văn. Verbatim chỉ chứng minh câu có trong nguồn, không chứng minh câu suy ra cạnh.

Heuristic ontology, valence, coreference, prior, threshold và confidence symbolic vẫn là chính sách nghiên cứu. Cần gold riêng cho NER/linking, direction, negation, section, drug specificity, evidence entailment và admission. Chưa thể gán một phần trăm accuracy chung cho NLP.

### KG — M2 về API, M1–M2 về dữ liệu bàn giao

Graph builder giữ attributes và nhận B/D; loader snapshot kiểm tra trùng node, endpoint và hash. Fusion có phân vùng admitted/provisional/rejected. Nhưng graph thực tế chưa phản ánh việc ghép NLP:

- `merged_graph.json`: **721 node, 2.804 edge; 100% source_dataset=hetionet_v1.0**.
- `v6_contract_d_kg_edges.jsonl`: **50 record riêng**, chưa xuất hiện như cạnh NLP trong snapshot trên.
- Graph có 2.095 cạnh sign=0, 356 cạnh sign=-1, 353 cạnh sign=+1. Theo báo cáo DFS hiện có, cả 4.755 đường đều có net_sign=0; không có nghĩa mọi cạnh đều không dấu.
- Báo cáo acceptance lưu sẵn ghi không cạnh nào có PMID hoặc evidence sentence. Chưa kiểm định lại upstream authenticity của snapshot nhỏ.

Thiếu một bước materialize KG có version, admission kiểm chứng thật, node metadata/ID crosswalk, edge dedup và evidence store bền vững. `to_contract_d_records()` bỏ partition, posterior/LCB, tên/type thực thể và evidence sentence khỏi record D. Những thông tin này cần được giữ ở graph hoặc nối qua evidence store/manifest; chỉ có PMID chưa đủ truy lại đúng câu gốc.

### Thuật toán — tuyến cũ M1–M2; M2 học máy ở mức M2

Tuyến cũ có DFS, sign algebra, hub penalty, path aggregation và metrics, nhưng `search_metapaths()` thực tế liệt kê simple paths, không áp khuôn node type. Runner chẩn đoán mới lọc metapath sau tìm kiếm. Báo cáo lưu sẵn: **4.755 đường, 0 khớp metapath cấu hình**, vì topology thường là Compound→Gene→BP/PW→Gene→Disease, khác khuôn Compound→Gene→BP/PW→Disease. DFS khớp oracle NetworkX chỉ xác nhận duyệt đường, không xác nhận hợp lệ cơ chế.

M2 có model/feature thật và CLI đa thuốc. Artifact `measurement_audit.json` ghi test macro Recall@10 M0≈0,1755; M2≈0,4174, AP M2≈0,0578; 27 thuốc có positive và 86 positives. Đây là **kết quả artifact lịch sử, không phải train lại trong audit**. CI chênh M3−M2 chứa 0 nên chưa đủ chứng cứ khẳng định M3 tốt hơn. Ground truth từ CtD Hetionet là tác vụ khôi phục quan hệ, không tự chứng minh phát hiện chỉ định tương lai.

M2 đang đọc feature matrix tĩnh + graph Hetionet thô; không nhận KG NLP mới làm đầu vào tính feature. Vì vậy thêm cạnh NLP vào NetworkX không làm score M2 cập nhật. Đây có thể là thiết kế hợp lệ nếu NLP chỉ dùng hậu kiểm, nhưng docs cần chọn rõ: NLP là feature ranking hay evidence sau ranking.

Để lên M3: chốt topology/metapath, temporal split và label semantics; định nghĩa interface feature build→train→infer→evidence; có run ID/hashes; đánh giá riêng held-out repurposing và calibration.

### Output và evaluation — M1, phần UI chủ yếu là trình diễn

CLI/artifact có output thật nhưng nhiều định dạng (`ranking_score`, `score`, `paths`, `top_paths`) chưa có adapter E thống nhất nối đến UI. UI không đọc JSON ranking, không gọi API; nút investigate chỉ hiện alert cố định. Evidence và PMID trong UI là literal, chưa được chứng minh là trích dẫn nguyên văn của các bài tương ứng.

`artifacts/evaluation_report.json` lưu graph 727 node/2.938 edge/2.104 relation và Recall@K=1, khác graph hiện tại 721/2.804 và 4 relation canonical. Không kết luận báo cáo đó giả; kết luận **không được dùng nó mô tả snapshot hiện tại khi chưa nối được manifest/version/hash**.

## 3. Pipeline thực tế và chỗ đứt

Luồng A: corpus CSV → run_nlp_extraction → B/C JSONL → dừng; chỉ validate bản ghi C đầu tiên và không xuất KG tích hợp.

Luồng B: benchmark symbolic → events → fusion → B/D và báo cáo riêng; khác classifier configuration tuyến A.

Luồng C: merged_graph baseline → DFS/ranking cũ → reports; không thấy NLP trong snapshot này, filtered ranking rỗng theo report đã lưu.

Luồng D: features.npz + M2_model.pkl + Hetionet raw → M2 ranking + micro-DFS → CLI/JSON; không lấy feature từ graph đã fuse NLP.

Luồng E: UI literals và demo có graph anchor viết sẵn → trình diễn. Chưa có chuỗi A→B/C→D→E dùng cùng một run/snapshot và evidence provenance.

**Mức trưởng thành tổng: M2 nguyên mẫu tích hợp một phần.** Không lấy trung bình mức hoàn thiện module để gọi tổng thể M3: chất lượng cuối bị giới hạn bởi temporal gate, admission, manifest và output đang thiếu.

## 4. Phát hiện ưu tiên, mock và hardcode

### P1 — có thể làm sai kết luận nghiên cứu hoặc đầu ra

1. **Demo dùng graph giả, điểm tự đặt, tuyên bố thành công vô điều kiện.** `scripts/demo_end_to_end_repurposing.py:178–250`: tự chèn SLC5A2→HF/CKD, cố định drug DB06292 dù CLI cho đổi thuốc; score=`min(0.98, valid_count*0.20+0.65)` hoặc 0.25; cuối luôn in SUCCESS. Cần chuyển demo thành fixture có nhãn hoặc chạy dữ liệu/thuật toán canonical thật và fail khi không đủ evidence.
2. **Temporal leakage ở demo/runner.** Demo `year_cutoff` không sử dụng, missing year thành 2018. Runner canonical không lọc ngày, adapter bỏ năm. Cần gate tại boundary A trước mọi NLP; năm không biết phải reject/quarantine, không giả định trước cutoff.
3. **UI hardcode kết quả và có lỗi phép nhân dấu.** `ui/index.html:658–824`: score, path, evidence, PMID viết sẵn; alert không dùng query để tính. Tại dòng 699, `(-1)*(+1)*(-1)*(+1)` thực tế bằng **+1**, UI ghi **-1**. Cần render Contract E thật và tính dấu từ edges.
4. **Admission có thể bị vượt qua.** `src/lbd/graph/build_graph.py:33–53`: có source/target là xem như D và đặt `admission_verified=True`; không validate schema, evidence, assertion hoặc ontology trước add. Repro in-memory: `{source:x,target:y,assertion_status:negated}` vẫn thành cạnh admitted với evidence rỗng. B imports tuy gắn admission_verified=False vẫn đi vào cùng graph. Cần admission gate riêng, record B không đi vào active graph trước duyệt.
5. **ID quan hệ không duy nhất xuyên corpus.** `fusion.py:90–100` reset REL_00001 mỗi lần gọi; adapter fuse từng document và runner ghép list. Repro hai PMID khác nhau cùng câu đều trả REL_00001. Dữ liệu hiện chỉ một PMID nên chưa có duplicate trong file 4 dòng, nhưng code sẽ đụng ID khi ghép nhiều bài. Dùng ID từ PMID+span+entities+predicate hoặc UUID/hash có namespace run.
6. **Metapath và loại chỉ định lịch sử chưa đúng contract.** `graph/path_search.py:6–48` không kiểm node type, length tính theo node. Dòng 28 cho T2D đi qua nếu nó là target; repro với drug→gene→DOID:9352 vẫn có 1 path khi filter_historical=True. Cần phân biệt lọc target novel và traversal, chốt đơn vị chiều dài.
7. **M2 bỏ lọc known indication khi graph không tải.** `m2_ranker.py:_load_hetionet_edges/rank_drug`: thiếu edges file thì return im lặng; hoặc load_graph_for_dfs=False làm ctd/cpd rỗng; rank_drug vẫn gọi các cặp là NOVEL_CANDIDATE. Cần tải indication gate độc lập DFS, hoặc trả trạng thái novelty_unknown/fail. Test hiện dùng graph=False nên không bảo vệ tiêu chí novelty.
8. **M2 micro-path contribution không phải phân rã chính xác logit.** `explain_pair_dfs` dùng beta*raw_path_weight, trong khi infer dùng beta*zscore(arcsinh(sum_weight/scale)). Transform phi tuyến và intercept/scaler khiến tổng path contributions không bằng score/logit. Hàm `compute_feature_contributions` có transform đúng ở cấp feature; cần đổi nhãn micro-path thành heuristic evidence hoặc thiết kế attribution phân bổ có kiểm chứng.

### P2 — làm giảm độ tin cậy, tái lập hoặc khả năng bảo trì

9. `normalize.py` gắn verified chỉ vì có hai chuỗi ID; repro ID tùy ý `not_a_real_id` vẫn verified. Synonym collapsing diabetic nephropathy/renal failure→CKD có thể xóa specificity. Phân biệt provided/resolved/ontology_verified và kiểm crosswalk có nguồn/version.
10. `ranking/features.py` cố định study weight=0.9, đếm số dòng PMID thay vì số PMID độc lập. Repro cùng PMID lặp làm edge weight tăng từ 0.27138 lên 0.33637. `rank_candidates.py` cũng len(list) cho contradiction. Đọc config/study_type và dedup set.
11. `events.py:262–270,321` verifier cache key không chứa model/prompt version hoặc mọi context type; cache được dùng trước kiểm live flag. Model `qwen/qwen3.8-27b` cố định trong code, config cũ ghi model khác và được README đánh dấu inactive. Tắt live không đồng nghĩa symbolic-only nếu cache có kết quả LLM cũ. Tách cache policy, lưu model/config/input hashes.
12. Fusion prior 0.85/0.45, conflict_discount 0.35, admission thresholds và normal-approx LCB là heuristic chưa calibration. Không gọi posterior/score là xác suất hiệu quả điều trị. Contract D export thiếu partition và evidence context gây mất khả năng hậu kiểm nếu không có store đi kèm.
13. Output và metric khác snapshot chưa có khóa liên kết; filenames cố định dễ overwrite. Cần `runs/<run_id>/manifest.json`, git dirty state, graph/corpus/model/config hash và tổng record ở mỗi gate.
14. M2 phụ thuộc `artifacts/ranking_m2` và `data/raw` ngoài package resources; wheel metadata chỉ đóng configs/contracts/NLP checkpoint. Source-tree test không chứng minh wheel M2 self-contained. Chuyển đường dẫn artifact thành tham số bắt buộc hoặc manifest-managed assets.
15. Hardcode `baricitinib_proxy`→Ruxolitinib không phải ID alias của cùng phân tử. Proxy cần trường explicit `requested_drug`, `actual_drug`, `proxy_reason`; không lẫn vào báo cáo thuốc đích.

### Mock hợp lệ nên giữ

`run_smoke_test.py` ghi rõ FIXTURE MODE và test Contract D vào graph rỗng trước fixture; HTTP mocks phục vụ lỗi API; dữ liệu giả trong tests kiểm invariants; random projection baseline ghi rõ untrained. Không xóa chúng chỉ vì tìm thấy từ mock/random. Trọng số chính sách, seed và ontology registry cũng không tự động là lỗi: lỗi là chúng bị trình bày như xác suất đã học hoặc kết quả xác thực.

## 5. Tệp thừa, trùng và lịch sử

- **Trùng byte đã xác nhận:** relations_extracted.jsonl = relations_extracted_v3.jsonl; relations_normalized.jsonl = relations_normalized_v3.jsonl. Đây là compatibility output có chủ đích; nên giữ một canonical file và mapping/deprecation, tránh hai tên làm người dùng tưởng hai model.
- **Trùng byte đã xác nhận:** 6 configs ngoài và trong src/lbd/configs; 5 schemas ngoài và trong src/lbd/contracts. Đóng gói resources là hợp lý, nhưng cần sinh tự động hoặc CI kiểm đồng bộ, không chỉnh tay cả hai.
- **Có thể dọn sau xác nhận tái tạo:** build/, .pytest_cache/, __pycache__/, wheel build cũ, `merged_graph.json.bak_prior_reset`. Không xóa backup trước khi xác định run/report nào còn phụ thuộc.
- **Nên tách khỏi đường chạy chính:** legacy_trash/, archive/, notebooks/thí nghiệm và output lịch sử trong ../nlp. REMEDIATION ghi 31 import đứt nằm ở legacy; audit này không chạy lại toàn bộ scanner để xác nhận con số đó.
- **Không xem là rác:** ../repurposing-ranking có source/train/evaluation framework và artifacts thực; ../nlp có corpus, checkpoints/evaluator cần thiết; audit_backups là bản truy vết. Chuyển sang quản lý dependency và lưu trữ lịch sử thay vì xóa cả thư mục.
- **Artifact lịch sử cần nhãn/version:** candidate_rankings.json, evaluation_report.json và các báo cáo trong samples/output. Không có provenance hiện hành thì không dùng làm dashboard trạng thái mới nhất.
- **Docs phân tán:** docs ở workspace ngoài và docs của dự án, README/tài liệu lịch sử có số liệu khác nhau. Chọn docs trong package làm nguồn chính, docs ngoài trỏ liên kết. Không cần xóa bản thiết kế gốc; đánh dấu planned/implemented/verified.

## 6. Thứ tự hoàn thiện đề xuất

1. Ngăn kết luận sai: gắn nhãn mock demo/UI, bỏ SUCCESS vô điều kiện, sửa dấu, gate cutoff, admission và novelty.
2. Chốt kiến trúc ranking: M2 background-only + NLP hậu kiểm, hoặc feature từ fused KG. Không gọi hai thiết kế là cùng một pipeline nếu dữ liệu vào khác nhau.
3. Hoàn thiện A→C→D: IDs toàn cục, temporal/source manifest, validate mọi record, node metadata, dedup và evidence store; xuất snapshot KG mới có thống kê background/NLP/rejected.
4. Hợp nhất E: adapter score/rank/path/evidence, run provenance, UI đọc artifact/API thật, trạng thái thiếu bằng chứng rõ ràng.
5. Nghiệm thu M3: một lệnh offline từ snapshot cố định đến output thật; kiểm counts/hashes và rerun tái lập; benchmark held-out NER/RE/direction/admission; ranking holdout theo thời gian/thuốc; ablation dùng cùng snapshot. Không thêm cạnh mẫu để ép HF/CKD lên top.
6. Sau đó mới dọn lịch sử, chuẩn hóa config/resources và khóa môi trường theo Python/platform.

## 7. Kiểm chứng trong lần audit

- `python -m pytest tests -q`: **34 passed, 24.30s**, chạy bằng Python hệ thống trong project.
- `python scripts/run_smoke_test.py`: **PASS**, fixture synthetic đã gắn nhãn; không phải validation y văn.
- Python của `.venv` workspace báo **No module named pytest**; môi trường ảo này chưa đủ để tái chạy bộ test. Không cài/sửa dependency trong lần audit.
- Đọc/đếm trực tiếp A, B/C, D và graph; so byte config/schema/output tương thích.
- Repro in-memory: REL ID lặp giữa bài; bypass admission bằng D hình thức; T2D target lọt filter; ID vô nghĩa vẫn verified; PMID duplicate làm tăng weight. Không ghi các record repro vào dữ liệu nghiên cứu.
- Các số 4.755 paths/oracle và metric M2 được trích từ artifact hiện có, không chạy lại DFS đầy đủ hoặc train model. Không gọi API live, không xác thực PMID/ontology upstream, không nghiệm thu browser UI hoặc wheel clean install trong lần này.

Ưu tiên tiếp theo là nối và bảo vệ các boundary dữ liệu. Việc thêm model/LLM mới trước khi giải quyết các lỗi này sẽ chưa làm pipeline tổng trưởng thành hơn.
