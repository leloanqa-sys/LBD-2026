# Kế hoạch gom thư mục, điều phối pipeline và đánh giá

Ngày: 03/11/2026. Phạm vi: working tree `dapagliflozin-lbd`, nhánh Dev và các thư mục workspace đang được nó tham chiếu. Đây là bản kế hoạch; chưa di dời file hoặc thay đổi hành vi chương trình.

## 1. Quyết định kiến trúc

Chọn `dapagliflozin-lbd/` làm repository và điểm làm việc chính. Bốn phân khu xử lý là `data`, `nlp`, `graph`, `ranking`; `evaluation` kiểm chứng chúng, `output` chuyển kết quả cho UI. Giữ các package domain đang có, tránh đổi tên hàng loạt chỉ để làm đẹp folder.

Mọi logic thực thi chuyển vào `src/lbd/`; CLI gọi các hàm ở đây. `scripts/` chỉ giữ launcher tương thích hoặc công cụ bảo trì, không giữ một bản nghiệp vụ thứ hai. Dùng một CLI `python -m lbd` để truy cập tất cả bước.

Chọn hướng mặc định ban đầu: **M2 dùng graph nền để xếp hạng, NLP dùng làm bằng chứng hậu kiểm**. Lý do: đúng với feature/model hiện có. KG hợp nhất vẫn được tạo và có thể trích subgraph để kiểm chứng cơ chế hoặc chạy tuyến signed-path riêng. Không đưa fused KG vào model M2 cũ mà không tính lại feature và thẩm định/train lại. Trong manifest phải ghi `ranking_graph=background` và `evidence_graph=fused`; không gọi hai graph này là một.

Nếu mục tiêu nghiên cứu yêu cầu NLP trực tiếp thay đổi ranking, mở profile thí nghiệm riêng `fused_graph_retrain`, có feature rebuild, train/holdout/ablation riêng. Đây không phải thao tác di dời file đơn thuần.

## 2. Cấu trúc đích

```text
dapagliflozin-lbd/
  README.md                     # Cài đặt, 3–5 lệnh chính, link tới docs
  pyproject.toml                # Metadata và dependency duy nhất
  .env.example                  # Chỉ tên biến/mẫu, không chứa key thật
  configs/                      # Profile chạy, không sao chép default package
    research.yaml
    smoke.yaml
  src/lbd/
    __main__.py                 # python -m lbd
    cli.py                      # Parse lệnh, gọi pipeline
    common/                     # Logging, IDs, config, validation, manifest
    pipeline/
      runner.py                 # Dependency graph, gate, resume
      stages.py                 # Registry tên bước → callable
      context.py                # RunContext, ArtifactRef, đường dẫn tuyệt đối
    data/                       # Collector, cutoff, normalization, snapshot loader
    nlp/                        # Engine canonical và các lớp NLP hiện có
    graph/                      # Merge, admission, subgraph, traversal
    ranking/                    # Feature build, inference, evidence adapter
    output/                     # Contract E serializer, provenance, explanation
    evaluation/                 # NLP gold, KG gold, ranking gold, metrics
    resources/
      defaults/                 # Config mặc định đóng gói
      contracts/                # Schema duy nhất, đóng gói bằng resources
  assets/
    models/                     # Checkpoint/model theo version + manifest
  data/
    raw/                        # Snapshot bất biến của PubMed/Hetionet
    reference/                  # Crosswalk/ontology có nguồn và version
    gold/                       # Gold và annotation guide, không lẫn output
    cache/                      # Có namespace nguồn/model/prompt version
  runs/
    <run_id>/
      manifest.json
      config.resolved.yaml
      logs/
      01_collect/               # papers.jsonl, abstracts.csv, rejected.jsonl
      02_nlp/                   # events.jsonl, relations.jsonl, rejected.jsonl
      03_normalize/             # relations_normalized.jsonl, unresolved.jsonl
      04_graph/                 # fused_graph, evidence store, admission report
      05_subgraph/              # subgraph theo drug, traversal metadata
      06_ranking/               # ranking, feature/model references
      07_output/                # candidates.json đúng Contract E
      08_evaluation/            # report + errors + coverage
  ui/                           # Đọc Contract E thật
  tests/
    unit/
    integration/
    acceptance/
    fixtures/                   # Synthetic có nhãn, không chung data thật
  scripts/
    maintenance/                # Migration/inspection/validation
  docs/
    architecture/               # Kiến trúc hiện hành, contracts, quyết định
    guides/                     # Setup, chạy pipeline, annotate/evaluate
    audits/                     # Báo cáo audit/remediation
    history/                    # Thiết kế V3/V6/V7 cũ, không làm entry point
  research/
    training/                   # Train/eval framework cần tái lập M2
    experiments/
    notebooks/
  archive/                      # Legacy có inventory, ngoài runtime
```

Không commit dữ liệu lớn vào Git chỉ vì được gom vào repository. Dùng manifest + checksum + vị trí lưu artifact; `data/raw`, models, runs và archive cần chính sách ignore/storage riêng. Model có thể được tải/materialize theo manifest thay vì đóng hết vào wheel.

## 3. Kế hoạch di dời cụ thể

### Mã chạy chính và scripts

- `src/lbd/data`, `nlp`, `graph`, `ranking`, `evaluation`: giữ tên package, chuyển logic rải ở scripts vào đúng module; sửa import có kiểm chứng.
- `scripts/run_nlp_extraction.py`: logic sang `pipeline`/NLP stage; file cũ tạm gọi CLI `nlp` và báo deprecated.
- `scripts/run_nlp_benchmark.py`, `run_v6_end_to_end_benchmark.py`: đặt benchmark orchestration trong evaluation; tuyến V6 ghi rõ profile legacy/symbolic, không giả làm canonical V7.
- `scripts/run_nlp_v3_extraction.py`: kiểm tra có logic riêng hay chỉ alias. Alias dùng wrapper; logic nghiên cứu riêng vào `research/experiments/`.
- `scripts/run_lbd_m2_ranking.py`: launcher gọi CLI `rank --backend m2`; bỏ path lookup dựa trên vị trí source file.
- `scripts/run_original_graph_dfs_and_ranking.py`: runner tuyến `signed-path`; phần chẩn đoán snapshot chuyển `evaluation`/maintenance, không dùng như M2.
- `scripts/audit_original_graph_acceptance.py`: `scripts/maintenance/audit_graph.py` hoặc subcommand `audit graph`.
- `scripts/evaluate_gold_ground_truth.py`: nghiệp vụ vào `evaluation/nlp_gold.py`; giữ wrapper trong thời gian chuyển đổi.
- `scripts/create_curated_ground_truth_sample.py`: công cụ chuẩn bị annotation vào `evaluation/annotation.py`; không được ghi đè gold đã adjudicate.
- `scripts/inspect_artifact.py`, `validate_artifact.py`: `scripts/maintenance/`.
- `scripts/run_smoke_test.py`: kiểm tra tương thích trước, rồi chuyển thành acceptance fixture/CLI smoke dùng thư mục tạm; có thể giữ launcher ngắn.
- `scripts/demo_end_to_end_repurposing.py`: chuyển `research/experiments/demo_fixture.py`, gắn nhãn synthetic rõ; không còn là lệnh chạy khoa học mặc định.

### Config, schemas và models

- `contracts/*.json` và `src/lbd/contracts/*.json`: hiện trùng byte; hợp nhất vào `src/lbd/resources/contracts/`. Loader dùng `importlib.resources`; xóa bản sao sau khi callers và wheel check đạt.
- `configs/*.yaml` và `src/lbd/configs/*.yaml`: chuyển defaults vào `resources/defaults`; `configs/` chỉ chứa override/profile. `config.resolved.yaml` trong mỗi run ghi kết quả merge đầy đủ.
- `src/lbd/nlp/checkpoints/` và các checkpoint ngoài workspace: kiểm hash/model version, chuyển canonical model sang `assets/models/<model_version>/`; các model khác đưa nghiên cứu/lịch sử. Loader nhận ArtifactRef, không tự chọn model từ folder ngẫu nhiên.
- `artifacts/ranking_m2/`: tách model sang assets, dữ liệu feature sang run huấn luyện, reports sang run lịch sử. Giữ quan hệ manifest giữa model, columns, scale, feature matrix và snapshot; không tách bằng tên file đơn thuần.

### Corpus, gold và kết quả

- `data/processed/papers_pre2018.jsonl` và CSV ngoài `../nlp/`: nhận diện snapshot gốc, hash và lineage trước. Chọn một raw source; A và CSV là sản phẩm của collect stage cùng run, không tự coi hai file là tương đương.
- `data/processed/relations_*`: giữ bản lịch sử trong run import có nhãn `legacy_unverified_provenance`; các file v3 giống byte chỉ giữ một canonical copy + mapping tên cũ.
- `data/processed/merged_graph.json`: lưu snapshot với hash hiện tại; ghi background-only, không đặt tên gây hiểu nhầm đã merge NLP.
- `v6_contract_b_relations.jsonl` và `v6_contract_d_kg_edges.jsonl`: cùng run lịch sử V6, không tự ghép vào snapshot canonical mà bỏ qua admission.
- `data/samples/*gold*`, guide, internal key, annotation CSV: phân loại sang `data/gold/<task>/<version>/`; tách blind input, adjudicated labels, internal key và model predictions. Giữ split, provenance, quyền truy cập phù hợp.
- `evaluation_report*`, predictions và errors hiện lẫn samples: chuyển về `runs/<run>/08_evaluation/`; không để metric cạnh file gold như thể đó là nhãn.
- `artifacts/candidate_rankings.json`, `evaluation_report.json`, graph reports: import nguyên trạng vào run lịch sử, ghi rõ thiếu manifest. Không tự gán chúng cho corpus/graph hiện tại.

### Thư mục ngoài repository

- `../nlp/`: inventory theo loại; corpus → raw, model → assets, evaluation cần thiết → evaluation/research, outputs → runs lịch sử, thử nghiệm → research. Chưa bỏ đường dẫn fallback trước khi kiểm tra importer và rerun.
- `../repurposing-ranking/`: giữ nguyên package `repurank` trước; gom vào `research/training/repurposing-ranking/` theo một đợt riêng sau kiểm tra Git boundary, provenance và bản trùng với lbd. CLI production không phụ thuộc import ngầm từ nó. Model dùng chung phải có manifest chỉ về training source.
- `../dapagliflozin_pubtator3_direct_pipeline/`: kiểm tra collector còn giá trị; chỉ trích module cần dùng vào data/annotation adapter sau kiểm chứng, phần còn lại vào research hoặc archive.
- `../docs/`: tài liệu hiện hành hợp nhất vào docs của repo, bản khác giữ history; sửa link và ghi superseded.
- `../legacy_trash/`: archive riêng có inventory, không thêm vào sys.path.
- `../audit_backups/`: giữ là backup ngoài runtime; chỉ thu gọn sau nghiệm thu migration.
- `.venv`: không di chuyển môi trường đã cài; dựng lại theo dependency/lock đã kiểm thử.

## 4. Hợp đồng điều phối từng bước

Mỗi bước đăng ký `stage name → module:function`, khai báo inputs/outputs/schema/version. Danh sách dưới là interface dự kiến cần triển khai, không phải các hàm đã tồn tại đầy đủ:

1. `collect → lbd.data.collect:run`: query + cutoff hoặc raw snapshot → A JSONL + CSV + rejection report. CSV là export của A, không là nguồn chân lý thứ hai.
2. `nlp → lbd.nlp.pipeline:run`: A → events + B; annotation provider, classifier, cache mode phải ghi rõ.
3. `normalize → lbd.data.normalize_pipeline:run`: B + crosswalk → C + unresolved queue; IDs toàn cục, không verified chỉ vì có chuỗi.
4. `build-graph → lbd.graph.pipeline:run`: background + C/events + admission config → fused KG + evidence store + admission report. Thực hiện fusion xuyên corpus, không cộng dồn các posterior từng bài.
5. `subgraph → lbd.graph.subgraph:run`: drug ID + snapshot + traversal policy → graph trích và manifest. Phân biệt traversal ngược hợp lệ với hướng nhân quả; ghi hop/node limits và truncation.
6. `rank → lbd.ranking.pipeline:run`: backend + model/features + graph refs → rankings và paths. M2 cần feature của đúng snapshot/model. Subgraph phục vụ giải thích không được âm thầm thay global degree/DWPC của model đã train.
7. `export → lbd.output.export:run`: ranking + evidence refs → E; giữ score semantics, source graph, drug thực, PMID, câu nguồn và trạng thái thiếu evidence.
8. `evaluate → lbd.evaluation.pipeline:run`: prediction/run + gold version → metrics, coverage, errors, provenance. Chỉ chạy task có gold phù hợp; thiếu gold trả NOT_EVALUATED, không metric giả hoặc PASS.

Runner thực thi theo dependency, không nối bằng shell command string. Fail gate thì dừng downstream phụ thuộc. Mỗi stage ghi số input/accepted/rejected/unresolved, hashes và status. `--resume` chỉ reuse khi input/config/code/model hashes khớp; hoàn thành atomic rồi mới cập nhật manifest. Không ghi đè run khác.

`create_dataset_manifest()` hiện gắn `audited_pre_cutoff` mà chưa tự audit dữ liệu; khi chuẩn hóa phải tách manifest khai báo nguồn và kết quả gate đã thực thi, tránh giữ trạng thái chứng nhận mặc định này.

## 5. CLI mục tiêu

Các lệnh dưới là giao diện đề xuất, chưa có đủ implementation:

```powershell
# Chạy từ snapshot local đến output; evaluate khi đã cấu hình gold phù hợp
python -m lbd run --config configs/research.yaml --drug DB06292 --offline

# Lấy PubMed live phải chọn rõ chế độ thu thập; không kéo theo live LLM mặc định
python -m lbd run --config configs/research.yaml --drug DB06292 --collect-live

# Chạy một bước / tiếp tục run, đều kiểm input manifest
python -m lbd stage nlp --run-id <run_id>
python -m lbd run --resume <run_id>

# Đánh giá từng tác vụ có schema gold riêng
python -m lbd evaluate --run-id <run_id> --task nlp-relation --gold data/gold/nlp-relation/v1/manifest.json
python -m lbd evaluate --run-id <run_id> --task ranking --gold data/gold/ranking/v1/manifest.json

# UI chỉ định run cần xem
python -m lbd serve --run-id <run_id>
```

Không tự đổi sang fixture khi thiếu corpus/model/graph. Offline chỉ cấm network, không mặc nhiên bảo đảm cache là symbolic; muốn symbolic-only phải có chế độ loại cache LLM rõ ràng.

## 6. Ground truth và evaluator cần gom

Không tạo một gold chung cho mọi layer. Mỗi gold có manifest nguồn, annotation protocol, reviewer/adjudication, split, ontology version và task. Model output tuyệt đối không dùng như nhãn người đánh giá.

- `ner-linking`: span/entity ID precision/recall/F1 trên văn bản thô.
- `nlp-relation`: entity pair, predicate, direction, negation; tách supplied-gold-entities và predicted-entities tracks.
- `kg-admission`: cạnh được nhận có được evidence hỗ trợ không; kiểm partition, nguồn và temporal validity.
- `ranking`: Recall@K/MRR theo drug/time split, known-indication exclusion, gold coverage trong candidate universe. Phân biệt Hetionet link recovery với temporal repurposing.
- `evidence`: câu có trong nguồn và có hỗ trợ kết luận không, hai tiêu chí riêng.

Di chuyển evaluator hiện có trước, giữ metric baseline; sau đó thêm adapter/schema và CLI. Gold chưa có cho task nào thì đưa vào backlog annotation, không tuyên bố đã đánh giá layer đó.

## 7. Thứ tự triển khai và điều kiện nghiệm thu

### Đợt 0 — Đóng băng baseline, chưa move

Inventory file/import/caller/size/hash; ghi Git dirty state và không lẫn thay đổi có sẵn với migration. Lập `migration_inventory.json` gồm old_path, new_path, role, owner, hash, status và referenced_by. Kiểm tra repo lồng nhau trước khi move ngoài workspace. Chụp baseline tests/smoke và metrics hiện có. Định nghĩa file canonical và chính sách archive.

Hoàn tất khi mỗi file runtime/data/model/gold quan trọng có đích và dependency rõ; không cần đọc ngữ nghĩa mọi cache record để lập inventory.

### Đợt 1 — Gom tài nguyên và đường dẫn

Tạo resource loader, RunContext, ArtifactRef, config resolver. Chuyển schemas/defaults, model manifests; sửa callers và package metadata. Giữ wrapper đường dẫn cũ khi còn người dùng. Di dời từng nhóm kèm cập nhật inventory; verify hash sau move. Không thay thuật toán trong đợt này.

Nghiệm thu: import/source và wheel resource loading đạt; tests/smoke baseline không regress; không còn fallback ngầm tới folder ngoài dự án cho profile canonical.

### Đợt 2 — Chuẩn hóa dữ liệu và sửa boundary

Cutoff bắt buộc, IDs toàn cục, verified ID đúng nghĩa, admission thật, PMID dedup, novelty gate độc lập DFS. Materialize run directory, evidence store và graph provenance. Các sửa hành vi đi thành thay đổi riêng để không quy nhầm metric thay đổi do di dời.

Nghiệm thu: các repro lỗi trong audit được chặn; graph báo chính xác nguồn; mọi record xuất được validate; missing inputs không biến thành synthetic data.

### Đợt 3 — CLI và orchestration

Thực hiện stage registry và dependencies, resume/invalidation, CLI backend ranking. Migrate scripts thành wrappers. Chạy từ snapshot thật giới hạn trước, rồi toàn corpus; không yêu cầu kết quả HF/CKD phải lên top để coi là chạy thành công.

Nghiệm thu: một lệnh tạo run đến E; mỗi stage chỉ được dùng artifact cùng lineage, có logs/counts/hashes; rerun tái lập theo dung sai số học đã định.

### Đợt 4 — Evaluation và UI

Di dời gold/evaluator, thêm task manifests và error export, UI đọc E thật. Bỏ literal score/path/evidence khỏi production UI; fixture demo nằm riêng.

Nghiệm thu: evaluator nhận đúng run+gold, kiểm split/ID coverage, báo NOT_EVALUATED khi thiếu gold; UI thể hiện đúng output của run đã chọn.

### Đợt 5 — Thu gọn lịch sử

Gom outer folders theo inventory; kiểm hash, import và link docs; bỏ alias trùng và wrappers hết hạn sau khi canonical CLI đã được dùng/kiểm thử. Dọn build/cache tái tạo được sau cùng. Không xóa raw/gold/model/backup chưa có bản giữ và provenance.

Nghiệm thu: root nhìn được một repository chính, README trỏ đúng lệnh/hàm; archive không tham gia import/runtime; danh sách ngoại lệ còn lại được ghi rõ.

## 8. Phân công điều phối

- Huy: collect, cutoff, crosswalk, snapshot/provenance và phối hợp KG admission.
- Khang: NLP canonical, annotation/cache, events/fusion và gold RE/direction.
- Liam: graph subgraph policy, ranking backends, M2 model/features và evidence attribution.
- Phước: gold manifests, evaluator, Contract E/UI và acceptance workflow.
- Một người chịu trách nhiệm tích hợp chung: stage registry, config/resources, migration inventory, dependency giữa các đợt và README. Chỉ định rõ người này trước khi triển khai; không để bốn phân khu tự đổi path độc lập.

Mỗi đợt có một thay đổi review riêng; migration và sửa thuật toán không trộn thành một diff lớn. Rollback theo inventory/baseline và commit của đợt, không dùng reset/clean gây mất các sửa chưa commit.

## 9. Định nghĩa hoàn tất

Người mới chỉ cần README để cài và chạy một lệnh trên snapshot có sẵn; biết chính xác data→NLP→KG→ranking→output đi qua hàm/file nào. Kết quả mang run ID và hash nguồn, không trộn metric cũ. Evaluator có task/gold rõ. UI không giả kết quả. Folder nghiên cứu và archive có thể tồn tại nhưng không điều khiển runtime. Đó là tiêu chí quản lý được dự án; giảm số thư mục đơn thuần chưa đủ.
