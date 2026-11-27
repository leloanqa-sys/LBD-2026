"""Comprehensive verification suite for NLP & Ontology P0/P1 fixes.

Covers:
1. ConText/BioScope negation & uncertainty detection in polarity.py (and preservation in events.py).
2. Preservation of CLASS:SGLT2i distinct from DB06292 in zoning.py; true anaphora resolution only.
3. Non-hardcoded entity blinding and dynamic BIOMED_SLM_MODEL reporting in events.py.
4. Exact Beta 10th-percentile quantile (scipy.stats.beta.ppf(0.10, a, b)) in fusion.py.
5. Clean baseline LogisticRegression default + BaseRelationEncoder interface in classifier.py,
   and explicit degraded_mode=True metadata in relation_extraction.py.
6. Ontology/ID integrity in normalize.py, ids.py, ranking.yaml, and filter_nlp.py.
"""
from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path

import pytest
import yaml
from scipy.stats import beta as scipy_beta

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WORKSPACE_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from lbd.common.ids import ID_CROSSWALK, resolve_identifier
from lbd.data.normalize import normalize_entity_name, normalize_relation_record
from lbd.nlp.events import Layer3_BioEventAndBlindedLLMEngine
from lbd.nlp.fusion import IndependentGraphBayesEngine, Layer5_BayesianBetaBinomialFuser
from lbd.nlp.polarity import Layer4_PolarityAndGradeReasoner, UNCERTAIN_RX
from lbd.nlp.relation_extraction import extract_relations_from_text
from lbd.nlp.zoning import Layer12_ZoningAndGroundingEngine


# ---------------------------------------------------------------------------
# 1. Negation & Uncertainty (polarity.py + events.py router preservation)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "sentence",
    [
        "Dapagliflozin failed to reduce stroke incidence in this trial.",
        "Treatment did not increase the risk of diabetic ketoacidosis.",
        "There was no significant effect of dapagliflozin on bone fracture.",
        "Dapagliflozin lowered blood pressure without causing hypotension.",
        "A lack of efficacy on myocardial infarction was observed.",
        "Neither dapagliflozin nor empagliflozin increased amputation rates.",
        "Dapagliflozin does not elevate serum potassium levels.",
        "Dapagliflozin was not associated with bladder cancer.",
        "Dapagliflozin had no effect on heart rate.",
        "The compound was ineffective for treating diabetic retinopathy.",
        # Window > 25 chars between negation trigger and target entity
        "No clear dose-response relationship between dapagliflozin and genital infection was demonstrated.",
    ],
)
def test_context_negation_patterns(sentence: str):
    reasoner = Layer4_PolarityAndGradeReasoner()
    is_neg, rationale = reasoner.is_locally_negated(sentence, "dapagliflozin")
    assert is_neg is True, f"Expected negation for: {sentence}"
    assert rationale


@pytest.mark.parametrize(
    "sentence",
    [
        "Dapagliflozin may reduce progression of chronic kidney disease.",
        "SGLT2 inhibition might attenuate cardiac fibrosis.",
        "This agent could lower cardiovascular mortality.",
        "Preliminary data suggests a protective role in heart failure.",
        "Dapagliflozin appears to improve endothelial function.",
        "Potentially beneficial effects on renal outcomes were noted.",
        "We tested the hypothesis that dapagliflozin reduces inflammation.",
        "This observation warrants further investigation in clinical trials.",
        "There was a trend toward reduced hospitalization for heart failure.",
        "Possible renoprotective mechanisms remain under study.",
    ],
)
def test_bioscope_uncertainty_patterns(sentence: str):
    assert UNCERTAIN_RX.search(sentence) is not None, f"Expected uncertainty cue in: {sentence}"


def test_assertion_status_not_coerced_to_asserted_in_polarity_and_router():
    reasoner = Layer4_PolarityAndGradeReasoner()
    src = {"id": "DB06292", "canonical_name": "Dapagliflozin", "type": "Compound", "matched_text": "Dapagliflozin", "span_start": 0, "span_end": 13}
    tgt = {"id": "DOID:6000", "canonical_name": "Heart Failure", "type": "Disease", "valence": 1, "matched_text": "heart failure", "span_start": 29, "span_end": 42}

    uncertain_sent = "Dapagliflozin may reduce heart failure hospitalization."
    res_unc = reasoner.resolve_pair_polarity(uncertain_sent, src, tgt)
    assert res_unc.assertion_status == "uncertain"

    negated_sent = "Dapagliflozin did not reduce heart failure mortality."
    res_neg = reasoner.resolve_pair_polarity(negated_sent, src, tgt)
    assert res_neg.assertion_status == "negated"

    # Verify Layer 3 Router Check 3 preserves assertion_status rather than forcing "asserted"
    l3 = Layer3_BioEventAndBlindedLLMEngine(enable_live_llm=False)
    routed_unc = l3.route_and_verify(uncertain_sent, "@COMPOUND_SRC$", "Compound", "@DISEASE_TGT$", "Disease", 1, res_unc)
    assert routed_unc["assertion_status"] == "uncertain"
    routed_neg = l3.route_and_verify(negated_sent, "@COMPOUND_SRC$", "Compound", "@DISEASE_TGT$", "Disease", 1, res_neg)
    assert routed_neg["assertion_status"] == "negated"


# ---------------------------------------------------------------------------
# 2. Drug-Class Preservation (zoning.py) & Non-Hardcoded Blinding (events.py)
# ---------------------------------------------------------------------------
def test_zoning_preserves_sglt2i_class_and_resolves_true_anaphora():
    zoning = Layer12_ZoningAndGroundingEngine(use_ontology_registry=True)
    title = "Dapagliflozin in Heart Failure"
    abstract = (
        "SGLT2 inhibitors reduce cardiovascular events in clinical trials. "
        "SGLT2i therapy also improves renal outcomes. "
        "This drug significantly lowered blood pressure."
    )
    grounded = zoning.ground_and_resolve_coref(title, abstract, pmid="12345678")
    sents = grounded["sentences"]

    # Sentence 1 (SGLT2 inhibitors): must stay CLASS:SGLT2i, NOT DB06292
    s1_ids = {e["id"] for e in sents[1].entities}
    assert "CLASS:SGLT2i" in s1_ids
    assert "DB06292" not in s1_ids

    # Sentence 2 (SGLT2i): must stay CLASS:SGLT2i, NOT DB06292
    s2_ids = {e["id"] for e in sents[2].entities}
    assert "CLASS:SGLT2i" in s2_ids
    assert "DB06292" not in s2_ids

    # Sentence 3 ("This drug"): true anaphora -> resolved to DB06292
    s3_ids = {e["id"] for e in sents[3].entities}
    assert "DB06292" in s3_ids


def test_events_blinding_is_pair_specific_and_llm_mode_is_dynamic():
    l3 = Layer3_BioEventAndBlindedLLMEngine(enable_live_llm=False)
    src = {"id": "DB01394", "canonical_name": "Colchicine", "type": "Compound", "matched_text": "Colchicine"}
    tgt = {"id": "114548", "canonical_name": "NLRP3", "type": "Gene", "matched_text": "NLRP3"}
    sentence = "Colchicine inhibits NLRP3 independently of dapagliflozin or SGLT2 pathways."
    blinded, src_mask, tgt_mask = l3.blind_sentence_entities(sentence, src, tgt)

    assert src_mask in blinded
    assert tgt_mask in blinded
    # Unrelated drugs/genes must NOT be blindly overwritten as @COMPOUND_SRC$ / @GENE_TARGET$
    assert "dapagliflozin" in blinded
    assert "SGLT2" in blinded

    # Verify events.py source no longer hardcodes "live_groq_qwen"
    events_src = (PROJECT_ROOT / "src/lbd/nlp/events.py").read_text(encoding="utf-8")
    assert '"live_groq_qwen"' not in events_src
    assert 'f"live_groq:{slm_model}"' in events_src


# ---------------------------------------------------------------------------
# 3. Exact Beta Quantile in fusion.py
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("alpha,beta_param", [(1.5, 0.5), (3.0, 1.0), (0.8, 0.8), (10.0, 2.5)])
def test_fusion_exact_beta_quantile(alpha: float, beta_param: float):
    fuser = Layer5_BayesianBetaBinomialFuser()
    mean, std, lcb_90 = fuser._compute_beta_stats(alpha, beta_param)
    expected_lcb = round(max(0.01, min(0.99, float(scipy_beta.ppf(0.10, alpha, beta_param)))), 4)
    assert lcb_90 == pytest.approx(expected_lcb, abs=1e-4)

    # Also verify IndependentGraphBayesEngine uses exact Beta quantile
    post = IndependentGraphBayesEngine.compute_posterior(
        kg_prior_strength=0.5, study_tier="clinical_trial", n_studies=2, has_contradiction=False
    )
    exp_ind = round(max(0.01, min(0.99, float(scipy_beta.ppf(0.10, post["alpha_post"], post["beta_post"])))), 4)
    assert post["lcb_90"] == pytest.approx(exp_ind, abs=1e-4)


# ---------------------------------------------------------------------------
# 4. Classifier Baseline Interface & Relation Extraction Degraded Mode
# ---------------------------------------------------------------------------
def test_classifier_clean_baseline_and_base_encoder_interface():
    cls_path = PROJECT_ROOT / "src/lbd/nlp/classifier.py"
    tree = ast.parse(cls_path.read_text(encoding="utf-8"))
    class_names = {node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)}
    assert "BaseRelationEncoder" in class_names
    assert "ZonedPoolingMultiClassBioREEngineV3" in class_names

    src_text = cls_path.read_text(encoding="utf-8")
    assert "use_mlp: bool = False" in src_text
    assert "self.ensemble_weights: Tuple[float, float, float] = (0.0, 1.0, 0.0)" in src_text


def test_relation_extraction_explicit_degraded_mode_on_rule_only_fallback():
    from lbd.nlp.engine import NLPEngine

    rule_engine = NLPEngine()
    rule_engine.degraded_mode = True
    rule_engine.degraded_reason = "checkpoint_missing_test"

    records = extract_relations_from_text(
        pmid="29085817",
        title="Dapagliflozin in Heart Failure",
        abstract="Dapagliflozin reduces heart failure and inhibits SLC5A2.",
        engine=rule_engine,
    )
    assert len(records) >= 1
    assert all(r.get("degraded_mode") is True for r in records)
    assert all(r.get("degraded_reason") == "checkpoint_missing_test" for r in records)


# ---------------------------------------------------------------------------
# 5. Ontology, ID Crosswalk, Ranking Metapaths, and filter_nlp ROOT
# ---------------------------------------------------------------------------
def test_ontology_and_ids_clean():
    # Semaglutide must be independent of Liraglutide
    assert normalize_entity_name("semaglutide") == "Semaglutide"
    assert normalize_entity_name("wegovy") == "Semaglutide"
    assert normalize_entity_name("ozempic") == "Semaglutide"
    assert resolve_identifier("Semaglutide") == "DB13928"
    assert resolve_identifier("Liraglutide") == "DB06655"
    assert resolve_identifier("Semaglutide") != resolve_identifier("Liraglutide")

    # Duloxetine must be Compound::DB00476 (NOT DB00471 Montelukast)
    assert resolve_identifier("Duloxetine") == "Compound::DB00476"
    assert resolve_identifier("Montelukast") == "Compound::DB00471"
    rec = normalize_relation_record({"source_entity": "cymbalta", "target_entity": "Heart Failure"})
    assert rec["source_entity"] == "Duloxetine"
    assert rec["source_standard_id"] == "Compound::DB00476"
    assert rec["target_standard_id"] == "DOID:6000"

    # Heart Failure (DOID:6000) distinct from Dilated Cardiomyopathy (DOID:12930)
    assert ID_CROSSWALK["Heart Failure"]["doid"] == "DOID:6000"
    assert ID_CROSSWALK["Heart Failure"]["hetionet_doid"] == "Disease::DOID:6000"
    assert ID_CROSSWALK["Dilated Cardiomyopathy"]["doid"] == "DOID:12930"
    assert ID_CROSSWALK["Dilated Cardiomyopathy"]["hetionet_doid"] == "Disease::DOID:12930"


def test_ranking_yaml_metapaths_synced():
    cfg = yaml.safe_load((PROJECT_ROOT / "configs/ranking.yaml").read_text(encoding="utf-8"))
    metapaths = cfg["search"]["allowed_metapaths"]
    assert ["Compound", "Gene", "Disease"] in metapaths
    assert ["Compound", "Gene", "BiologicalProcess", "Gene", "Disease"] in metapaths
    assert ["Compound", "Gene", "Pathway", "Gene", "Disease"] in metapaths
    assert cfg["search"]["min_length"] == 2


def test_filter_nlp_root_path():
    spec = importlib.util.spec_from_file_location("filter_nlp_root", WORKSPACE_ROOT / "filter_nlp.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.ROOT == PROJECT_ROOT
    assert (mod.ROOT / "contracts").is_dir()
