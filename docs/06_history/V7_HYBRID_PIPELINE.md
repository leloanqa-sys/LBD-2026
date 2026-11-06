> Historical design document. Current setup/status: ../README.md and REMEDIATION.md.

# V7 Hybrid Pipeline

V7 composes the fitted V3 relation classifier with V6 grounding, event reasoning,
and Bayesian evidence fusion.

## Data flow

1. V6 L1/L2 zones sentences, grounds mentions, and resolves coreference. PubTator
   mentions and local normalized concepts are retained for symbolic event building.
2. The V7 adapter passes source mentions plus V6-added entities to V3. V3 returns
   `P(relation)` and its four class probabilities for document-level entity pairs.
3. L3 uses the V3 relation threshold stored in the selected checkpoint to gate
   V6-supported candidate pairs. A missing V3 score leaves the V6 candidate path
   available for entity/process types that V3 does not model.
4. L4 retains responsibility for relation direction, polarity, assertion status,
   contraindication, and study/context rules. V3 `POSITIVE`/`NEGATIVE` correlation
   labels are recorded as model diagnostics and are not converted into `CAUSES`,
   `TREATS`, `ACTIVATES`, or `INHIBITS`.
5. L5 consumes the V3 relation-existence probability as the evidence confidence,
   deduplicates evidence by PMID, fuses support/conflict, and exports Contract B/D.

The engine is in `src/lbd/nlp/engine.py`. It accepts a fitted V3 model
object and has no hard dependency on an API key. Live LLM calls default to off.

## Independent evaluation

The evaluator intentionally stays in the outer `nlp/` directory:

```powershell
python nlp/evaluation/run_v7_hybrid_ablation.py
```

It verifies official BioRED and BC5CDR dev/test hashes, compares V3-only, V6
symbolic, and V7 hybrid relation detection, audits live PubTator spans against
benchmark entity gold, and compares V7 pre-L5 vs. V7+L5 on a cross-PMID pair proxy.
Thresholds for the L5 proxy are selected on dev. LLM calls are disabled.

The report is `nlp/evaluation/results/v7_hybrid_ablation.json`. The oracle track
supplies benchmark gold spans to L1 and isolates downstream relation layers. The
PubTator track measures realistic grounding, including mention-linking errors.
BioRED polarity is correlation polarity and is not a direct ground truth for
V6's directed clinical/mechanistic predicates. The L5 proxy does not validate KG
auto-admission. Section zoning has no independent section-zone gold in these
benchmarks, so that sublayer is not assigned an accuracy score. Same-sentence
pair co-occurrence is only an upper-bound coverage check for L1 candidate
generation; it is not relation-extraction precision or end-to-end recall.
