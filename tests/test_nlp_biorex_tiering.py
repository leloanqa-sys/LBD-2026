"""Unit & integration tests for BioREx integration, Epistemic Scope, narrowed Cohort scope,
and 3-tier Evidence Tiering (ADMITTED_KG / HYPOTHESIS_EDGE / QUARANTINE).
"""
from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import pytest

from lbd.nlp.engine import NLPEngine
from lbd.nlp.external_annotations import (
    BIOREX_RELATION_TYPES,
    _parse_pubtator,
    fetch_pubtator,
    fetch_pubtator_with_relations,
)
from lbd.nlp.polarity import PolarityEngine


ROOT = Path(__file__).resolve().parents[1]


def _load_filter_huy():
    spec = importlib.util.spec_from_file_location("filter_huy", ROOT / "scripts" / "filter_huy.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# 1. External Annotations: BioREx parsing + v3/v2 cache compatibility
# ---------------------------------------------------------------------------


def test_parse_pubtator_preserves_entities_and_all_biorex_relations():
    payload = {
        "PubTator3": [
            {
                "pmid": "30123456",
                "passages": [
                    {
                        "offset": 0,
                        "annotations": [
                            {
                                "id": "A0",
                                "text": "Dapagliflozin",
                                "infons": {"type": "Chemical", "identifier": "MESH:C529054"},
                                "locations": [{"offset": 0, "length": 13}],
                            },
                            {
                                "id": "A1",
                                "text": "heart failure",
                                "infons": {"type": "Disease", "identifier": "MESH:D006333"},
                                "locations": [{"offset": 25, "length": 13}],
                            },
                            {
                                "id": "A2",
                                "text": "SGLT2",
                                "infons": {"type": "Gene", "identifier": "6524"},
                                "locations": [{"offset": 45, "length": 5}],
                            },
                        ],
                    }
                ],
                "relations": [
                    {
                        "id": "R0",
                        "infons": {
                            "type": "Negative_Correlation",
                            "role1": {"identifier": "MESH:C529054", "type": "Chemical", "name": "Dapagliflozin"},
                            "role2": {"identifier": "MESH:D006333", "type": "Disease", "name": "heart failure"},
                            "novel": "Novel",
                            "score": "0.94",
                        },
                    },
                    {
                        "id": "R1",
                        "infons": {
                            "type": "Bind",
                            "role1": "Chemical|MESH:C529054",
                            "role2": "Gene|6524",
                            "novel": "No",
                        },
                    },
                    {
                        "id": "R2",
                        "infons": {"type": "Comparison", "novel": "No"},
                        "nodes": [{"refid": "A0", "role": "role1"}, {"refid": "A2", "role": "role2"}],
                    },
                ],
            }
        ]
    }

    parsed = _parse_pubtator(payload)
    assert "30123456" in parsed
    bundle = parsed["30123456"]
    assert len(bundle["entities"]) == 3
    assert len(bundle["relations"]) == 3

    rel_by_id = {r["id"]: r for r in bundle["relations"]}
    assert rel_by_id["R0"]["type"] == "Negative_Correlation"
    assert rel_by_id["R0"]["role1"]["id"] == "DB06292"
    assert rel_by_id["R0"]["role2"]["id"] == "DOID:6000"
    assert rel_by_id["R0"]["novel"] == "Novel"
    assert rel_by_id["R0"]["score"] == pytest.approx(0.94)

    assert rel_by_id["R1"]["type"] == "Bind"
    assert rel_by_id["R1"]["role1"]["id"] == "DB06292"
    assert rel_by_id["R1"]["role2"]["id"] == "6524"
    assert rel_by_id["R1"]["novel"] == "No"

    assert rel_by_id["R2"]["type"] == "Comparison"
    assert rel_by_id["R2"]["role1"]["id"] == "DB06292"
    assert rel_by_id["R2"]["role2"]["id"] == "6524"

    expected_types = {
        "Positive_Correlation",
        "Negative_Correlation",
        "Association",
        "Bind",
        "Cotreatment",
        "Comparison",
        "Drug_Interaction",
        "Conversion",
    }
    assert set(BIOREX_RELATION_TYPES.values()) == expected_types


def test_pubtator_cache_v3_and_v2_backward_compatibility(tmp_path: Path):
    # Pre-populate v3 cache for PMID 11111 and legacy v2 cache for PMID 22222
    v3_dir = tmp_path / "pubtator3_bioc_v3"
    v2_dir = tmp_path / "pubtator3_bioc_v2"
    v3_dir.mkdir(parents=True)
    v2_dir.mkdir(parents=True)

    (v3_dir / "11111.json").write_text(
        json.dumps(
            {
                "entities": [{"id": "DB06292", "text": "dapagliflozin", "type": "Compound"}],
                "relations": [
                    {
                        "id": "R0",
                        "type": "Negative_Correlation",
                        "role1": {"id": "DB06292", "type": "Compound"},
                        "role2": {"id": "DOID:6000", "type": "Disease"},
                        "novel": "Novel",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (v2_dir / "22222.json").write_text(
        json.dumps([{"id": "DOID:9352", "text": "type 2 diabetes", "type": "Disease"}]),
        encoding="utf-8",
    )

    entities, relations, stats = fetch_pubtator_with_relations(["11111", "22222"], tmp_path, offline=True)
    assert len(entities["11111"]) == 1
    assert len(relations["11111"]) == 1
    assert len(entities["22222"]) == 1
    assert relations["22222"] == []
    assert stats["v3_cache_hits"] == 1
    assert stats["v2_fallback_hits"] == 1
    assert stats["relations"] == 1

    # Backward-compatible wrapper still returns (entities_map, stats)
    legacy_entities, legacy_stats = fetch_pubtator(["11111", "22222"], tmp_path, offline=True)
    assert set(legacy_entities.keys()) == {"11111", "22222"}
    assert legacy_stats["cache_hits"] == 2


# ---------------------------------------------------------------------------
# 2. Polarity, Epistemic Scope, Narrowed Cohort Scope, and BioREx Refinement
# ---------------------------------------------------------------------------


def test_epistemic_scope_propagation_and_section_prior():
    pe = PolarityEngine()
    src = {"id": "DB06292", "type": "Compound", "matched_text": "dapagliflozin", "span_start": 35, "span_end": 48}
    tgt = {"id": "DOID:6000", "type": "Disease", "matched_text": "heart failure", "valence": 1, "span_start": 57, "span_end": 70}

    # (i) Outer-clause epistemic scope propagation
    s_outer = "We aimed to evaluate whether dapagliflozin reduces heart failure progression."
    res_outer = pe.resolve_pair_polarity(s_outer, src, tgt, section_zone="RESULTS")
    assert res_outer.assertion_status == "uncertain"

    # (ii) Strong empirical result blocks outer-clause propagation
    s_result = "To evaluate efficacy, we ran an RCT; dapagliflozin significantly reduced heart failure (p < 0.001)."
    res_result = pe.resolve_pair_polarity(
        s_result,
        {**src, "span_start": 37, "span_end": 50},
        {**tgt, "span_start": 73, "span_end": 86},
        section_zone="RESULTS",
    )
    assert res_result.assertion_status == "asserted"
    assert res_result.canonical_relation == "TREATS"

    # (iii) Section prior: OBJECTIVES without assertive result -> uncertain
    s_obj = "Dapagliflozin effects on heart failure outcomes were examined."
    res_obj = pe.resolve_pair_polarity(
        s_obj,
        {**src, "span_start": 0, "span_end": 13},
        {**tgt, "span_start": 25, "span_end": 38},
        section_zone="OBJECTIVES",
    )
    assert res_obj.assertion_status == "uncertain"

    # (iv) Negation still takes strict precedence
    s_neg = "Dapagliflozin did not reduce heart failure events in this trial."
    res_neg = pe.resolve_pair_polarity(
        s_neg,
        {**src, "span_start": 0, "span_end": 13},
        {**tgt, "span_start": 29, "span_end": 42},
        section_zone="RESULTS",
    )
    assert res_neg.assertion_status == "negated"


def test_narrowed_cohort_context_distinguishes_baseline_from_target():
    pe = PolarityEngine()

    # Baseline cohort criterion -> EVALUATED_IN_COHORT
    s_baseline = "Dapagliflozin lowered HbA1c in patients with type 2 diabetes."
    src = {"id": "DB06292", "type": "Compound", "matched_text": "Dapagliflozin", "span_start": 0, "span_end": 13}
    t2d = {"id": "DOID:9352", "type": "Disease", "matched_text": "type 2 diabetes", "valence": 1, "span_start": 45, "span_end": 60}
    res_base = pe.resolve_pair_polarity(s_baseline, src, t2d, section_zone="RESULTS")
    assert res_base.canonical_relation == "EVALUATED_IN_COHORT"

    # Disease is direct object of therapeutic verb -> TREATS (not EVALUATED_IN_COHORT)
    s_treated = "Dapagliflozin significantly reduced hospitalization for heart failure in patients with heart failure."
    hf = {
        "id": "DOID:6000",
        "type": "Disease",
        "matched_text": "heart failure",
        "valence": 1,
        "span_start": 56,
        "span_end": 69,
        "all_spans": [(56, 69), (87, 100)],
    }
    res_treated = pe.resolve_pair_polarity(s_treated, src, hf, section_zone="RESULTS")
    assert res_treated.canonical_relation == "TREATS"


def test_biorex_refinement_in_engine_pipeline():
    engine = NLPEngine(classifier=None)

    # 1. Negative_Correlation refines neutral co-occurrence to TREATS instead of collapsing to cohort/association
    doc_neg_corr = {
        "pmid": "990001",
        "year": 2017,
        "title": "Dapagliflozin and Heart Failure",
        "abstract": "RESULTS: Dapagliflozin and heart failure were linked in this clinical cohort.",
        "biorex_relations": [
            {
                "id": "R1",
                "type": "Negative_Correlation",
                "role1": {"id": "DB06292", "type": "Compound"},
                "role2": {"id": "DOID:6000", "type": "Disease"},
                "novel": "Novel",
            }
        ],
    }
    _, events_nc, _ = engine.run_document(doc_neg_corr, apply_relation_gate=False)
    edges_nc = [e for f in events_nc for e in f.projected_edges if e.source_id == "DB06292" and e.target_id == "DOID:6000"]
    assert edges_nc
    assert any(e.canonical_relation == "TREATS" for e in edges_nc)

    # 2. Comparison separates comparative mention from causal edges
    doc_comp = {
        "pmid": "990002",
        "year": 2017,
        "title": "Dapagliflozin versus Empagliflozin",
        "abstract": "RESULTS: Dapagliflozin was compared with empagliflozin on glycemic control.",
        "biorex_relations": [
            {
                "id": "R2",
                "type": "Comparison",
                "role1": {"id": "DB06292", "type": "Compound"},
                "role2": {"id": "DB09038", "type": "Compound"},
                "novel": "No",
            }
        ],
    }
    ext_comp = [
        {"id": "DB06292", "canonical_name": "Dapagliflozin", "text": "Dapagliflozin", "type": "Compound", "start": 44, "end": 57},
        {"id": "DB09038", "canonical_name": "Empagliflozin", "text": "empagliflozin", "type": "Compound", "start": 76, "end": 89},
    ]
    _, events_cmp, _ = engine.run_document(doc_comp, external_entities=ext_comp, apply_relation_gate=False)
    edges_cmp = [e for f in events_cmp for e in f.projected_edges if {e.source_id, e.target_id} == {"DB06292", "DB09038"}]
    assert edges_cmp
    assert all(e.canonical_relation == "ASSOCIATED_WITH" and e.hetionet_sign == 0 for e in edges_cmp)
    assert any("COMPARISON_CONTEXT" in e.clinical_polarity for e in edges_cmp)


# ---------------------------------------------------------------------------
# 3. Evidence Tiering 3-Tier Filter (ADMITTED_KG / HYPOTHESIS_EDGE / QUARANTINE)
# ---------------------------------------------------------------------------


def test_filter_huy_three_tier_evidence_tiering_and_zero_negated_leaks(tmp_path: Path):
    fh = _load_filter_huy()
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    out_dir = tmp_path / "out"

    # Build original papers JSONL (3 pre-cutoff papers, 1 post-cutoff paper)
    papers_path = tmp_path / "papers.jsonl"
    papers = [
        {
            "pmid": "1001",
            "title": "Dapagliflozin inhibits SGLT2",
            "abstract": "Dapagliflozin selectively inhibits SGLT2.",
            "publication_date": "2016-05-01",
            "print_year": "2016",
        },
        {
            "pmid": "1002",
            "title": "Dapagliflozin and SGLT2",
            "abstract": "Dapagliflozin potently inhibits SLC5A2. Dapagliflozin may ameliorate heart failure.",
            "publication_date": "2017-08-15",
            "print_year": "2017",
        },
        {
            "pmid": "1003",
            "title": "Safety and preclinical study",
            "abstract": "Dapagliflozin did not treat stroke. Dapagliflozin is contraindicated in renal failure. In mice, dapagliflozin inhibits NHE1.",
            "publication_date": "2018-03-10",
            "print_year": "2018",
        },
        {
            "pmid": "1004",
            "title": "Post-cutoff 2020 study",
            "abstract": "Dapagliflozin treats heart failure.",
            "publication_date": "2020-01-15",
            "print_year": "2020",
        },
    ]
    papers_path.write_text("\n".join(json.dumps(p) for p in papers) + "\n", encoding="utf-8")

    # Minimal run_report.json
    (bundle_dir / "run_report.json").write_text(json.dumps({"status": "ok"}), encoding="utf-8")

    # entity_mentions.csv (note: PMID 1002 uses raw text 'SLC5A2' for Gene 6524 while mention uses 'SGLT2' to test ID-first grounding)
    mentions = [
        {"pmid": "1001", "mention_text": "Dapagliflozin", "entity_id": "DB06292", "entity_type": "Compound", "offset_status": "verified_exact"},
        {"pmid": "1001", "mention_text": "SGLT2", "entity_id": "6524", "entity_type": "Gene", "offset_status": "verified_exact"},
        {"pmid": "1002", "mention_text": "Dapagliflozin", "entity_id": "DB06292", "entity_type": "Compound", "offset_status": "verified_exact"},
        {"pmid": "1002", "mention_text": "SGLT2", "entity_id": "6524", "entity_type": "Gene", "offset_status": "verified_exact"},
        {"pmid": "1002", "mention_text": "heart failure", "entity_id": "DOID:6000", "entity_type": "Disease", "offset_status": "verified_exact"},
        {"pmid": "1003", "mention_text": "Dapagliflozin", "entity_id": "DB06292", "entity_type": "Compound", "offset_status": "verified_exact"},
        {"pmid": "1003", "mention_text": "stroke", "entity_id": "DOID:6713", "entity_type": "Disease", "offset_status": "verified_exact"},
        {"pmid": "1003", "mention_text": "renal failure", "entity_id": "DOID:1074", "entity_type": "Disease", "offset_status": "verified_exact"},
        {"pmid": "1003", "mention_text": "NHE1", "entity_id": "6548", "entity_type": "Gene", "offset_status": "verified_exact"},
        {"pmid": "1004", "mention_text": "Dapagliflozin", "entity_id": "DB06292", "entity_type": "Compound", "offset_status": "verified_exact"},
        {"pmid": "1004", "mention_text": "heart failure", "entity_id": "DOID:6000", "entity_type": "Disease", "offset_status": "verified_exact"},
    ]
    with (bundle_dir / "entity_mentions.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(mentions[0].keys()))
        w.writeheader()
        w.writerows(mentions)

    # grounded_sentences.jsonl
    grounded_sents = [
        {
            "pmid": "1001",
            "sentence": "Dapagliflozin selectively inhibits SGLT2.",
            "entities": [
                {"matched_text": "Dapagliflozin", "id": "DB06292", "type": "Compound"},
                {"matched_text": "SGLT2", "id": "6524", "type": "Gene"},
            ],
        },
        {
            "pmid": "1002",
            "sentence": "Dapagliflozin potently inhibits SLC5A2.",
            "entities": [
                {"matched_text": "Dapagliflozin", "id": "DB06292", "type": "Compound"},
                {"matched_text": "SGLT2", "id": "6524", "type": "Gene"},
            ],
        },
        {
            "pmid": "1002",
            "sentence": "Dapagliflozin may ameliorate heart failure.",
            "entities": [
                {"matched_text": "Dapagliflozin", "id": "DB06292", "type": "Compound"},
                {"matched_text": "heart failure", "id": "DOID:6000", "type": "Disease"},
            ],
        },
        {
            "pmid": "1003",
            "sentence": "Dapagliflozin did not treat stroke.",
            "entities": [
                {"matched_text": "Dapagliflozin", "id": "DB06292", "type": "Compound"},
                {"matched_text": "stroke", "id": "DOID:6713", "type": "Disease"},
            ],
        },
        {
            "pmid": "1003",
            "sentence": "Dapagliflozin is contraindicated in renal failure.",
            "entities": [
                {"matched_text": "Dapagliflozin", "id": "DB06292", "type": "Compound"},
                {"matched_text": "renal failure", "id": "DOID:1074", "type": "Disease"},
            ],
        },
        {
            "pmid": "1003",
            "sentence": "In mice, dapagliflozin inhibits NHE1.",
            "entities": [
                {"matched_text": "dapagliflozin", "id": "DB06292", "type": "Compound"},
                {"matched_text": "NHE1", "id": "6548", "type": "Gene"},
            ],
        },
        {
            "pmid": "1004",
            "sentence": "Dapagliflozin treats heart failure.",
            "entities": [
                {"matched_text": "Dapagliflozin", "id": "DB06292", "type": "Compound"},
                {"matched_text": "heart failure", "id": "DOID:6000", "type": "Disease"},
            ],
        },
    ]
    (bundle_dir / "grounded_sentences.jsonl").write_text(
        "\n".join(json.dumps(s) for s in grounded_sents) + "\n", encoding="utf-8"
    )

    # Candidate relations CSV covering Tier 1, Tier 2, and Tier 3 (Quarantine)
    candidates = [
        # Tier 1 pair (2 PMIDs: 1001 + 1002, ID-first grounding on 1002)
        {
            "relation_id": "REL_1", "source_entity": "Dapagliflozin", "target_entity": "SLC5A2",
            "source_standard_id": "DB06292", "target_standard_id": "6524",
            "source_raw_text": "Dapagliflozin", "target_raw_text": "SGLT2",
            "source_type": "Compound", "target_type": "Gene",
            "relation_type": "INHIBITS", "canonical_relation": "INHIBITS",
            "assertion_status": "asserted", "confidence_score": "0.95",
            "normalization_status": "verified", "evidence": "Dapagliflozin selectively inhibits SGLT2.",
            "pmid": "1001", "sign": "-1", "metaedge": "CiG",
            "section_zone": "RESULTS", "study_design": "RCT_CLINICAL_TRIAL",
            "section_weight": "1.0", "study_weight": "1.0",
        },
        {
            "relation_id": "REL_2", "source_entity": "Dapagliflozin", "target_entity": "SLC5A2",
            "source_standard_id": "DB06292", "target_standard_id": "6524",
            "source_raw_text": "Dapagliflozin", "target_raw_text": "SLC5A2",
            "source_type": "Compound", "target_type": "Gene",
            "relation_type": "INHIBITS", "canonical_relation": "INHIBITS",
            "assertion_status": "asserted", "confidence_score": "0.92",
            "normalization_status": "verified", "evidence": "Dapagliflozin potently inhibits SLC5A2.",
            "pmid": "1002", "sign": "-1", "metaedge": "CiG",
            "section_zone": "RESULTS", "study_design": "RCT_CLINICAL_TRIAL",
            "section_weight": "1.0", "study_weight": "1.0",
        },
        # Tier 2: uncertain single-PMID therapeutic hypothesis
        {
            "relation_id": "REL_3", "source_entity": "Dapagliflozin", "target_entity": "Heart Failure",
            "source_standard_id": "DB06292", "target_standard_id": "DOID:6000",
            "source_raw_text": "Dapagliflozin", "target_raw_text": "heart failure",
            "source_type": "Compound", "target_type": "Disease",
            "relation_type": "TREATS", "canonical_relation": "TREATS",
            "assertion_status": "uncertain", "confidence_score": "0.80",
            "normalization_status": "verified", "evidence": "Dapagliflozin may ameliorate heart failure.",
            "pmid": "1002", "sign": "-1", "metaedge": "CtD",
            "section_zone": "CONCLUSIONS", "study_design": "RCT_CLINICAL_TRIAL",
            "section_weight": "0.9", "study_weight": "1.0",
        },
        # Tier 2: preclinical Compound->Gene (1 PMID)
        {
            "relation_id": "REL_4", "source_entity": "Dapagliflozin", "target_entity": "SLC9A1",
            "source_standard_id": "DB06292", "target_standard_id": "6548",
            "source_raw_text": "dapagliflozin", "target_raw_text": "NHE1",
            "source_type": "Compound", "target_type": "Gene",
            "relation_type": "INHIBITS", "canonical_relation": "INHIBITS",
            "assertion_status": "asserted", "confidence_score": "0.85",
            "normalization_status": "verified", "evidence": "In mice, dapagliflozin inhibits NHE1.",
            "pmid": "1003", "sign": "-1", "metaedge": "CiG",
            "section_zone": "RESULTS", "study_design": "PRECLINICAL_IN_VIVO",
            "section_weight": "1.0", "study_weight": "0.6",
        },
        # Tier 3 QUARANTINE: explicit negation + negated status
        {
            "relation_id": "REL_5", "source_entity": "Dapagliflozin", "target_entity": "Stroke",
            "source_standard_id": "DB06292", "target_standard_id": "DOID:6713",
            "source_raw_text": "Dapagliflozin", "target_raw_text": "stroke",
            "source_type": "Compound", "target_type": "Disease",
            "relation_type": "TREATS", "canonical_relation": "TREATS",
            "assertion_status": "negated", "confidence_score": "0.90",
            "normalization_status": "verified", "evidence": "Dapagliflozin did not treat stroke.",
            "pmid": "1003", "sign": "-1", "metaedge": "CtD",
            "section_zone": "RESULTS", "study_design": "RCT_CLINICAL_TRIAL",
            "section_weight": "1.0", "study_weight": "1.0",
        },
        # Tier 3 QUARANTINE: contraindicated
        {
            "relation_id": "REL_6", "source_entity": "Dapagliflozin", "target_entity": "Renal Failure",
            "source_standard_id": "DB06292", "target_standard_id": "DOID:1074",
            "source_raw_text": "Dapagliflozin", "target_raw_text": "renal failure",
            "source_type": "Compound", "target_type": "Disease",
            "relation_type": "TREATS", "canonical_relation": "CONTRAINDICATED_FOR",
            "assertion_status": "negated", "confidence_score": "0.99",
            "normalization_status": "verified", "evidence": "Dapagliflozin is contraindicated in renal failure.",
            "pmid": "1003", "sign": "-1", "metaedge": "CtD",
            "section_zone": "RESULTS", "study_design": "RCT_CLINICAL_TRIAL",
            "section_weight": "1.0", "study_weight": "1.0",
        },
        # Tier 3 QUARANTINE: post-cutoff (2020)
        {
            "relation_id": "REL_7", "source_entity": "Dapagliflozin", "target_entity": "Heart Failure",
            "source_standard_id": "DB06292", "target_standard_id": "DOID:6000",
            "source_raw_text": "Dapagliflozin", "target_raw_text": "heart failure",
            "source_type": "Compound", "target_type": "Disease",
            "relation_type": "TREATS", "canonical_relation": "TREATS",
            "assertion_status": "asserted", "confidence_score": "0.95",
            "normalization_status": "verified", "evidence": "Dapagliflozin treats heart failure.",
            "pmid": "1004", "sign": "-1", "metaedge": "CtD",
            "section_zone": "RESULTS", "study_design": "RCT_CLINICAL_TRIAL",
            "section_weight": "1.0", "study_weight": "1.0",
        },
    ]
    with (bundle_dir / "relations_contract_b.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(candidates[0].keys()))
        w.writeheader()
        w.writerows(candidates)

    report = fh.run(bundle_dir, out_dir, papers_path, ROOT / "configs" / "kg_filter.json")

    tier1_edges = [json.loads(line) for line in (out_dir / "kg_edges_contract_d.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    tier2_edges = [json.loads(line) for line in (out_dir / "kg_edges_hypothesis.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    tier3_edges = [json.loads(line) for line in (out_dir / "kg_edges_quarantine.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]

    # Tier 1: only DB06292 -> 6524 (INHIBITS) admitted with 2 PMIDs and strict ADMITTED_KG partition
    assert len(tier1_edges) == 1
    assert tier1_edges[0]["target"] == "6524"
    assert tier1_edges[0]["partition"] == "ADMITTED_KG"
    assert tier1_edges[0]["unique_supporting_pmids"] == 2

    # Tier 2: DOID:6000 (uncertain, 1 pre-cutoff PMID) + 6548 (preclinical Compound->Gene, 1 PMID)
    tier2_targets = {e["target"] for e in tier2_edges}
    assert tier2_targets == {"DOID:6000", "6548"}
    assert all(e["partition"] == "HYPOTHESIS_EDGE" for e in tier2_edges)
    # Post-cutoff PMID 1004 must NOT appear in Tier 2 DOID:6000 edge
    hf_edge = next(e for e in tier2_edges if e["target"] == "DOID:6000")
    assert hf_edge["pmids"] == ["1002"]

    # Tier 3 QUARANTINE: 0% negated / contraindicated / post-cutoff leaked into Tier 1 or Tier 2
    assert report["tier_summary"]["tier1_negated_leaks"] == 0
    assert report["tier_summary"]["tier2_negated_leaks"] == 0
    assert "DOID:6713" not in tier2_targets
    assert "DOID:1074" not in tier2_targets
    q_targets = {e["target"] for e in tier3_edges}
    assert {"DOID:6713", "DOID:1074", "DOID:6000"}.issubset(q_targets)
