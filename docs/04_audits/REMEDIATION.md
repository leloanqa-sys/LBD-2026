# Remediation log — 2026-11-01

## Completed

- Unified setup.py/requirements/pyproject runtime metadata; added previously missing
  SciPy and scikit-learn. Removed unused direct SDK dependencies from the core install.
- Removed misleading historical lock pins; compatibility file explicitly says it is
  not a lock. Generate tested platform/Python-specific locks before release.
- Packaged schemas/configs/checkpoints; schema validation fails if jsonschema is absent.
  Config loader requires a real YAML parser and rejects non-mapping configs.
- Graph builder now accepts B and D fields, preserves sign/assertion/evidence and
  background-edge provenance; flags B imports as admission_verified=False.
- Removed implicit mock NetworkX backend; NetworkX is a required dependency.
- Zoning keeps explicit section state and no longer promotes sentences by final
  position or 35% position alone. External annotation injection is explicit/offline.
- Fusion selects a deterministic assertion-priority representative instead of first
  input; output no longer claims every normalized ID has been verified.
  Full evidence-level assertion aggregation remains a follow-up.
- Empty evidence is rejected in strict mode; rejected duplicate candidates no longer
  prevent a later valid evidence record from being retained.
- HTTP failures return a structured symbolic fallback; logger is defined.
- Classifier fallback emits a warning; threshold overrides no longer mutate the
  shared default engine. This fallback remains a degraded mode, not learned inference.
- Invalid/missing dates fail the historical cutoff gate instead of becoming 1970.
- Implemented explicit local graph-snapshot loader with ID/endpoint checks and hash;
  no implicit mock or network graph. Caller must supply graph_path. It loads the whole
  supplied snapshot; its name does not promise drug-specific subgraph extraction.
- README now describes real entry points and limits, not historical accuracy as current.
- Synthetic smoke fixture retained, labelled FIXTURE MODE and guarded by an empty-graph
  Contract D integration assertion. HTTP fault mocks are restricted to offline tests.

## Verification

- 33/33 unit and acceptance cases passed; remediation_tests.xml records results.
- Smoke test passed with explicit synthetic-data label and Contract D preservation.
- Wheel built successfully. Build dependencies were installed in isolated build env;
  runtime environment was not replaced. Full clean-environment dependency resolution
  and a Python-version CI matrix remain unverified.
- Repository inventory covers 5,599 files at scan time (excluding backup/build/bytecode).
  AST scan found zero syntax errors. All 31 unresolved absolute local-module imports
  found by the scanner belong to legacy_trash. Static scan is not full dynamic coverage.

## Preserved mock/unfinished work — do not treat as scientific PASS

1. Smoke text and seed graph: deliberate synthetic fixture, retained and labelled.
2. HTTP mocks: deliberate fault injection; no live LLM correctness claim.
3. Old random-projection baseline: retained as an untrained comparison, not inference.
4. Legacy scripts: archived, 31 stale imports recorded in repository_inventory.json.
5. Domain regex/valence/coreference: still case-study heuristics. Class-to-drug
   projection, contextual valence and multiple-drug ambiguity need independent gold.
6. LLM verifier configuration: legacy nlp.yaml is labelled inactive; hardcoded model,
   cache key versioning and cache provenance need a dedicated migration.
7. Bayesian prior/thresholds, approximate LCB and representative-based assertion remain
   research policies. They are not calibrated clinical probabilities.
8. Graph metapaths and topology disagree; ranking contains neutral paths and baseline
   indications. Algorithm redesign is not replaced by fabricated graph shortcuts.
9. NER/section/edge/explanation evaluation need separate held-out gold and versioned
   runs. Old reports remain historical; do not compare changed-code metrics silently.
10. UI is not included in scientific acceptance; no live browser workflow was exercised.
11. Normalization still contains domain-specific synonym collapsing; upstream ontology
    authenticity and time-validity of the local Hetionet snapshot are not certified.

## Backups and scope

Original edited files are under outer `audit_backups/2026-11-01/`.
No source graph/corpus was rewritten, no data was fabricated and no tests were marked
xfail to hide a defect. A fixture was updated to explicit RESULTS because its purpose
is integration/admission, while separate tests cover unstructured section behavior.
The inventory is an all-file catalog plus selected static checks, not a claim to have
read every cached biomedical record or validated every notebook/PDF semantically.
