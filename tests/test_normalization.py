import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

"""Unit tests for entity name and ID normalization."""
from lbd.data.normalize import normalize_entity_name, normalize_relation_record

def test_normalize_entity_name():
    assert normalize_entity_name("dapa") == "Dapagliflozin"
    assert normalize_entity_name("farxiga") == "Dapagliflozin"
    assert normalize_entity_name("sglt2") == "SLC5A2"
    assert normalize_entity_name("ckd") == "Chronic Kidney Disease"
    assert normalize_entity_name("hf") == "Heart Failure"

def test_normalize_relation_record():
    raw_rel = {
        "relation_id": "r1",
        "source_entity": "dapa",
        "target_entity": "sglt2",
        "relation_type": "INHIBITS",
        "assertion_status": "asserted",
        "evidence": "dapa inhibits sglt2",
        "pmid": "111"
    }
    normed = normalize_relation_record(raw_rel)
    assert normed["source_entity"] == "Dapagliflozin"
    assert normed["target_entity"] == "SLC5A2"
    assert normed["source_standard_id"] == "DB06292"
    assert normed["target_standard_id"] == "6524"
    assert normed["normalization_status"] == "verified"
