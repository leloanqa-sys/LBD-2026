# LBD: đánh giá sâu thiết kế và độ trưởng thành

Ngày 03/11/2026. Đối tượng là `dapagliflozin-lbd`, working tree nhánh Dev, không phải nhánh Git tên lbd. Bổ sung cho AUDIT_MATURITY_20261103.md. Không sửa code trong lần đánh giá này. Các mức dưới đây là nhận định kỹ thuật theo bằng chứng, không phải điểm benchmark hay phần trăm hoàn thành.

## 1. Phán quyết tổng thể

**Một modular research prototype có năng lực NLP và ranking thật, nhưng chưa là hệ thống LBD tích hợp đáng tin cậy.** Chia thư mục theo domain đã đúng hướng. Điểm yếu chính không phải thiếu thư mục hoặc model: thiếu hợp đồng ngữ nghĩa xuyên module, một nguồn định danh chuẩn và sự liên kết input→model→graph→output→evaluation theo run.

Có thể dùng hiện trạng để nghiên cứu module và chạy thực nghiệm có kiểm soát. Chưa đủ để trình bày một output bất kỳ như phát hiện được tái lập từ corpus pre-2019 bằng pipeline hoàn chỉnh.

Thang: M0 ý tưởng, M1 chạy riêng, M2 tích hợp một phần, M3 toàn luồng dữ liệu thật có kiểm chứng/tái lập, M4 vận hành ổn định. Tổng thể M2; mức trưởng thành ngữ nghĩa và evaluation thấp hơn độ phong phú code.

## 2. Thiết kế cấp hệ thống: đúng domain, thiếu application layer

Các domain `data`, `nlp`, `graph`, `ranking`, `evaluation` có ý nghĩa rõ, phù hợp một modular monolith offline. NetworkX, JSONL và scikit-learn đủ hợp lý cho nguyên mẫu; chưa có bằng chứng cần microservices hay đổi sang Neo4j để giải quyết các lỗi hiện tại.

Tuy nhiên, orchestration nằm ở các script riêng. Thiếu một lớp application chịu trách nhiệm lựa chọn snapshot, dependency, config, validation, retry/resume và lineage. Vì thế cùng tên nghiệp vụ có nhiều ý nghĩa: extraction canonical, benchmark symbolic, demo, ranking cũ và M2. Điều này tạo nhiều nguồn chân lý mà người dùng phải tự ghép.

`NLPEngine` là facade tốt ở cấp NLP, nhưng không phải facade toàn LBD. `common` đã có công cụ dùng chung, song manifest chỉ là dict khai báo, validation chỉ kiểm hình dạng, ID crosswalk vẫn cục bộ. Đặt tên class/contract industrial không chứng minh boundary đã được bảo vệ.

Đánh giá: module organization M2; orchestration M1; pipeline provenance M1. Giữ domain packages, bổ sung runner và artifact interfaces sẽ có giá trị hơn đổi tên toàn bộ module.

## 3. Lỗi thiết kế nghiêm trọng: dấu số học thay thế ngữ nghĩa

Repro offline trong RAM, không lưu dữ liệu giả vào corpus:

1. Cho `NLPEngine()` đọc câu fixture `RESULTS: Dapagliflozin is contraindicated in chronic kidney disease.`
2. NLP tạo `canonical_relation=CONTRAINDICATED_FOR`, sign=-1.
3. Fusion giữ `partition=CONTRAINDICATED_LOCK`.
4. Export D bỏ canonical_relation và partition, chỉ giữ `relation_type=ASSOCIATED_WITH`, assertion=negated, sign=-1.
5. Graph builder nhận cạnh đó; DFS trả cạnh; `calculate_net_sign()` và `get_direction_fit()` cho **-1 và 1.5**.

Nghĩa là từng lớp có vẻ làm đúng phần riêng, nhưng kết hợp lại biến contraindication thành đường được điểm therapeutic. Repro dùng min_length=2 để cô lập cạnh trực tiếp; không khẳng định runner mặc định đã xếp bệnh đó lên top. Nếu cạnh nằm trên đường dài được phép, thiết kế scorer cũng không kiểm khóa/negation để bảo vệ.

Nguồn: `nlp/polarity.py:281–294`, `nlp/events.py:682–684`, `nlp/fusion.py:to_contract_d_records`, `graph/build_graph.py`, `graph/path_search.py`, `ranking/score_paths.py`.

Cần tách ít nhất: tác động nhân quả increase/decrease; assertion asserted/negated/uncertain; quan hệ lâm sàng treats/causes/contraindicated/cohort; partition admissibility. Không dùng cùng -1 để biểu diễn cả inhibition, therapeutic effect và contraindication. Scorer cần biết predicate, context và admissibility trước khi áp sign algebra.

Đây là ưu tiên cao nhất vì làm sai diễn giải dù schema và unit tests vẫn pass.

## 4. Contracts A–E: có schema, chưa có type system nghiệp vụ

Điểm tốt là có JSON Schema, enum và validation tập trung. Nhưng B và C dùng chung schema, trong khi C đáng ra bảo đảm chuẩn hóa thành công. IDs không required, chuỗi không có minLength, evidence không cần nonempty. Repro một record với relation_id/source/target/evidence/pmid đều chuỗi rỗng, không có standard IDs, vẫn pass `relation.schema.json`.

D không bắt buộc assertion, partition, timestamp, ontology namespace hoặc evidence nonempty. Một record đạt schema không đồng nghĩa đủ điều kiện vào active graph. `admission_verified` hiện suy từ hình dạng source/target thay vì kết quả kiểm định.

E bắt buộc rank, drug_id, paths với nodes/relations/path_score/net_sign; output hàm candidate cũ và M2 chưa có adapter chung bảo đảm các trường này. Không được ép M2 path topology thành directed therapeutic path để vừa schema: E cần thể hiện loại explanation và score semantics của từng backend.

Thiết kế cần ba gate khác nhau: schema validation; referential integrity giữa IDs/evidence/source; semantic admission. Hiện lớp đầu có, hai lớp sau chưa hoàn chỉnh.

## 5. Định danh: rủi ro xuyên toàn pipeline

`common/ids.py` dùng Heart Failure DOID:9333; adapter PubTator trong `external_annotations.py` map MeSH D006333 sang DOID:6000. Trong repro contraindication, registry trả CKD DOID:0080828, trong khi hard positives dùng DOID:784. Không kết luận các ontology ID này sai về y sinh khi chưa kiểm upstream; điều xác nhận được là **cùng khái niệm nhãn trong dự án không được quy về một ID thống nhất**.

Đồng thời tuyến M2 dùng `Compound::DB...`, `Disease::DOID:...`; tuyến cũ dùng DB.../DOID:... không prefix. Join string trực tiếp có thể làm mất cạnh, trượt gold hoặc chia một bệnh thành hai node.

Crosswalk cần canonical ID + namespace/version + alias/source mapping + quan hệ exact/broader/narrower; không gom các bệnh liên quan thành synonym không điều kiện. Đây là yêu cầu cấu trúc dữ liệu, không chỉ thêm vài mục vào dict.

## 6. Phân khu dữ liệu: snapshot có thật, acquisition chưa là sản phẩm

Có 664 DocumentRecord và cutoff helper. Nhưng thư viện `fetch_pubmed.py` chủ yếu tạo/filter record, còn live fetch ở demo; runner extraction chọn CSV bằng path fallback và không gate cutoff. `create_dataset_manifest()` tự gán `audited_pre_cutoff` từ tham số, không thực hiện audit trước đó. Trạng thái audited vì vậy là khai báo, chưa là bằng chứng.

Cần collector riêng có manifest và gate, sau đó A là canonical record; CSV chỉ là export tiện xem. Khả năng retry/atomic cache trong annotation adapter là điểm tốt, nên tái dùng qua service interface thay vì viết lại HTTP trong nhiều scripts.

Mức M2 về dữ liệu local, M1 về acquisition/orchestration và chứng nhận provenance.

## 7. NLP: lớp phức tạp nhất, cũng tập trung nhiều chính sách nhất

Có classifier học sẵn, facade NLPEngine, event dataclasses, symbolic polarity, external annotations và fusion. Việc tách relation-existence probability khỏi directed clinical predicate là hướng thiết kế tốt.

Tuy nhiên các file lớn: classifier 765 dòng, events 691, zoning 648, polarity 581 ở lần đọc này. Số dòng không tự chứng minh thiết kế xấu; vấn đề là events vừa điều phối candidates, policy, LLM HTTP/cache và mapping contract. Zoning kết hợp registry với xử lý văn bản. Thay một policy có thể ảnh hưởng nhiều layer mà không có config/model provenance tương ứng.

Khi được phép refactor: tách annotation provider, entity resolver, relation scorer, policy/reasoner, verifier, evidence aggregator. Giữ facade hiện tại để callers không đổi. Không chia nhỏ file chỉ để giảm số dòng nếu chưa có boundary nghiệp vụ rõ.

Fusion có dedup PMID là điểm tốt, nhưng representative assertion, prior/LCB và export làm mất semantics cần được kiểm chứng riêng. Benchmark oracle entities không thay thế NER end-to-end. Mức M2 mạnh về implementation, chưa M3 về evidence validation.

## 8. KG và graph traversal: cấu trúc đồ thị chưa bằng mô hình tri thức

MultiDiGraph phù hợp cạnh nhiều nguồn. Tuy nhiên add_edge có thể tự tạo node không metadata; source/target endpoints có mặt không đồng nghĩa type/ontology hợp lệ. Node IDs, source-backed evidence và policy partition cần được quản lý như những entity có lifecycle, không chỉ edge attributes tùy ý.

`filter_ego_subgraph()` có thật: duyệt G.neighbors theo hướng outgoing, tạo induced subgraph. Cần phân biệt nó với `load_hetionet_subgraph()` vốn tải toàn snapshot. Vì vậy nhận định đúng là **đã có helper lọc ego, chưa có stage canonical trích graph theo thuốc với policy nghiệm thu**, không phải hoàn toàn không có hàm trích graph.

Helper chưa áp node type whitelist/metaedge policy, không báo truncation, thiếu root thì trả graph rỗng; radius âm không bị từ chối rõ. Với Hetionet, quan hệ membership/symmetric có thể cần traversal ngược theo whitelist; không được đảo tất cả cạnh và xem là causal.

DFS cũ không enforce metapaths; M2 topology có graph semantics riêng. Cần một GraphView/TraversalPolicy thống nhất trước khi chia sẻ thuật toán.

## 9. Ranking: hai câu hỏi nghiên cứu khác nhau đang cùng một package

Signed-path hỏi có tuyến cơ chế nào có dấu phù hợp; M2 hỏi đặc trưng topology dự đoán association/indication đến đâu. Chúng không thay thế nhau trực tiếp. M2 có artifacts huấn luyện và kiểm định, nhưng cần data/model interface xác thực hash, số chiều, thứ tự drugs×diseases, feature_names/columns và snapshot tương ứng trước inference.

M2 hiện coupling với layout filesystem và feature matrix tĩnh. Thay graph hoặc damping exponent của explainer không tự thay feature/model. Default settings có thể hợp lệ, nhưng API không nên cho người dùng tưởng đổi parameter là infer một model mới nhất quán.

Micro-DFS attribution beta×raw path weight là heuristic, không exact logit contribution qua arcsinh/scaler. `compute_feature_contributions()` mới phân rã ở cấp transformed feature. Output cần phân biệt feature attribution, structural path và biomedical evidence.

Novelty gate phải hoạt động khi không cần DFS và phải báo unknown khi thiếu graph indication. Không chứng nhận novelty bằng việc không tìm thấy nhãn trong một tập chưa tải.

Mức M2; experiment maturity tốt hơn tuyến signed cũ, integration maturity chưa theo kịp.

## 10. Evaluation và tests: kiểm code tốt hơn kiểm tuyên bố khoa học

34 tests và smoke đã pass trong lần audit trước cùng ngày; không chạy lại ở lần đọc sâu vì code không đổi. Có tests hữu ích: oracle DFS với cycle/parallel edges, permutation invariance, PMID dedup, HTTP fallback, section persistence và D preservation.

Nhưng tests hiện chưa chặn contraindication→therapeutic, B/C thiếu normalized IDs, crosswalk bất nhất, relation ID collision xuyên bài và novelty gate thiếu graph. Test M2 kiểm một case CKD probability>0.70 dựa artifact hiện hành có giá trị regression, không là validation held-out độc lập.

`ground_truth.py` là dict case-study với hai hard positives và một negative `TRAP:TERMINATED_01`, chưa là bộ gold đa thuốc có source/adjudication đầy đủ. MRR hiện tính reciprocal rank của positive đầu tiên cho một danh sách; gọi mean đúng khi aggregate nhiều query bên ngoài. Missing gold hiện trả 0 ở helper, cần phân biệt no-ground-truth với model miss ở evaluator.

Không thấy `.github` CI trong repo đang mở. Clean installation, version matrix, pinned environment và artifact compatibility chưa được chứng nhận bằng pipeline CI hiện có.

Evaluation M1–M2; chưa có bằng chứng M3 toàn hệ thống.

## 11. Đánh giá cấu trúc sau khi gom tài liệu

Mục lục mới và START_HERE cải thiện khả năng tìm file. Runtime vẫn giữ nguyên vì người dùng yêu cầu không đụng code. Do đó mức dễ đọc tài liệu đã tăng, còn coupling imports/path, schema và graph semantics chưa thay đổi.

Điểm nên giữ: src-layout, domain modules, engine facade, explicit offline mode, structured fallback, tests dựa invariants. Điểm nên cải tổ khi có phạm vi sửa code: application runner, artifact registry, canonical IDs, typed semantic contracts và evaluation tracks. Không cần xóa toàn bộ hoặc viết lại từ đầu.

## 12. Thứ tự trưởng thành nên theo

1. **Ngữ nghĩa:** khóa contraindication/negation khỏi therapeutic scoring; giữ canonical relation/partition/context xuyên D/E.
2. **Identity và integrity:** ontology crosswalk, IDs toàn cục, B/C tách trách nhiệm, evidence/source references có thể resolve.
3. **Một luồng có provenance:** canonical inputs, run manifest, stage registry, model-feature-graph compatibility; fail rõ thay vì synthetic fallback.
4. **Backend/output rõ nghĩa:** signed mechanism và M2 topology có interface chung nhưng explanation/score type khác nhau.
5. **Validation độc lập:** benchmark raw-text và supplied-entities riêng, edge-admission gold, drug/time holdout ranking, evidence entailment.
6. **Tái lập và vận hành:** lock, CI, source/wheel install, atomic outputs, resume đúng hash và UI đọc run thật.

Đạt M3 khi một run từ snapshot thật đến E được tái lập, semantic invariants không bị vi phạm và từng claim accuracy có benchmark đúng task/version. Chạy hết các script hoặc top kết quả trùng HF/CKD không đủ thay thế các điều kiện này.
