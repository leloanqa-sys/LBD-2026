import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

"""Unit tests for relation extraction postprocessing and verbatim check."""
from lbd.nlp.postprocess import verify_evidence_verbatim, postprocess_relations

def test_verify_evidence_verbatim():
    abstract = "Dapagliflozin inhibits SLC5A2 and increases glucose excretion."
    evidence_good = "inhibits SLC5A2"
    evidence_bad = "reduces blood pressure completely"
    assert verify_evidence_verbatim(evidence_good, abstract) is True
    assert verify_evidence_verbatim(evidence_bad, abstract) is False
