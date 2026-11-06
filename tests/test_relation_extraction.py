import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

"""Unit tests for relation extraction, postprocessing, verbatim check, and schema contracts."""
from lbd.nlp.postprocess import verify_evidence_verbatim, postprocess_relations
from lbd.nlp.relation_extraction import extract_relations_from_text
from lbd.nlp.engine import NLPEngine
from lbd.common.validation import validate_contract


def test_verify_evidence_verbatim():
    abstract = "Dapagliflozin inhibits SLC5A2 and increases glucose excretion."
    evidence_good = "inhibits SLC5A2"
    evidence_bad = "reduces blood pressure completely"
    assert verify_evidence_verbatim(evidence_good, abstract) is True
    assert verify_evidence_verbatim(evidence_bad, abstract) is False


def test_extract_relations_from_text():
    pmid = "29085817"
    title = "Effects of Dapagliflozin on Heart Failure"
    abstract = "Dapagliflozin reduces heart failure in patients with type 2 diabetes. Dapagliflozin inhibits SLC5A2."

    relations = extract_relations_from_text(pmid, abstract, title=title)
    assert len(relations) >= 1

    # Verify Contract B schema compliance
    is_valid, msg = validate_contract(relations[0], "relation.schema.json")
    assert is_valid, f"Contract B validation failed: {msg}"

    # Verify key entities and standard identifiers
    entities = {r["target_entity"] for r in relations} | {r["source_entity"] for r in relations}
    assert "Dapagliflozin" in entities
    assert any(r["source_standard_id"] == "DB06292" for r in relations)


def test_nlp_engine_end_to_end():
    engine = NLPEngine()
    doc = {
        "pmid": "99999999",
        "title": "Dapagliflozin in Chronic Kidney Disease",
        "abstract": "RESULTS: Dapagliflozin attenuates chronic kidney disease progression and inhibits SLC5A2.",
    }
    grounded, events, _ = engine.run_document(doc)
    assert len(events) >= 1

    contract_b, fused, contract_d = engine.fuse_corpus(events)
    assert len(contract_b) >= 1
    assert len(contract_d) >= 1

    # Check contract D valid
    is_valid_d, msg_d = validate_contract(contract_d[0], "graph_edge.schema.json")
    assert is_valid_d, f"Contract D validation failed: {msg_d}"
