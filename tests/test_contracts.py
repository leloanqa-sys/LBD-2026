import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

"""Unit test for Data Contracts A through E validation."""
import json
from pathlib import Path
from lbd.common.validation import validate_contract

def test_contract_a_document_record():
    doc = {
        "document_id": "doc_01",
        "pmid": "12345678",
        "title": "Study of Dapagliflozin in Heart Failure",
        "abstract": "Dapagliflozin reduces risk of worsening heart failure.",
        "publication_date": "2018-05-10",
        "study_type": "clinical_trial",
        "dataset_version": "v1.0"
    }
    is_valid, msg = validate_contract(doc, "paper.schema.json")
    assert is_valid, msg

def test_contract_b_relation_record():
    rel = {
        "relation_id": "rel_01",
        "source_entity": "Dapagliflozin",
        "target_entity": "SLC5A2",
        "relation_type": "INHIBITS",
        "assertion_status": "asserted",
        "evidence": "dapagliflozin inhibits SLC5A2",
        "pmid": "12345678"
    }
    is_valid, msg = validate_contract(rel, "relation.schema.json")
    assert is_valid, msg
