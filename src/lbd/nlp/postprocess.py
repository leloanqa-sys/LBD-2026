"""Validation and Postprocessing of Extracted Relations."""
from typing import List, Dict, Any

VALID_RELATIONS = {"INHIBITS", "ACTIVATES", "CAUSES", "INCREASES", "DECREASES", "ASSOCIATED_WITH", "TREATS", "PALLIATES"}

def verify_evidence_verbatim(evidence: str, abstract: str) -> bool:
    """Verifies that evidence sentence actually exists in the abstract."""
    return evidence.strip().lower() in abstract.strip().lower()

def postprocess_relations(relations: List[Dict[str, Any]], abstract: str) -> List[Dict[str, Any]]:
    cleaned = []
    for r in relations:
        if r.get("relation_type") not in VALID_RELATIONS:
            continue
        # Verify evidence verbatim check
        evidence = r.get("evidence", "")
        if evidence and not verify_evidence_verbatim(evidence, abstract):
            r["evidence_verbatim_status"] = "partial_or_missing"
        else:
            r["evidence_verbatim_status"] = "verified"
        cleaned.append(r)
    return cleaned
