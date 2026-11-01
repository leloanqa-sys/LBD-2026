"""Validation and Postprocessing of Extracted Relations."""
from typing import List, Dict, Any, Optional

VALID_RELATIONS = {
    "INHIBITS", "ACTIVATES", "CAUSES", "INCREASES", "DECREASES",
    "ASSOCIATED_WITH", "TREATS", "PALLIATES", "CONTRAINDICATED_FOR",
    "EVALUATED_IN_COHORT", "PREVENTS",
}


def verify_evidence_verbatim(evidence: str, abstract: str, title: str = "") -> bool:
    """Verifies that evidence sentence actually exists in the abstract or title."""
    ev = evidence.strip().lower()
    full = (title + " " + abstract).strip().lower()
    return ev in full or ev in abstract.strip().lower()


def postprocess_relations(
    relations: List[Dict[str, Any]],
    abstract: str,
    title: str = "",
    strict: bool = False,
    **kwargs: Any,
) -> List[Dict[str, Any]]:
    """Filters, deduplicates, and validates verbatim evidence for extracted relations."""
    cleaned = []
    seen = set()
    for r in relations:
        rel_type = r.get("relation_type")
        if strict and rel_type not in VALID_RELATIONS:
            continue

        # Deduplication key
        key = (
            r.get("pmid"),
            r.get("source_standard_id") or r.get("source_entity"),
            r.get("target_standard_id") or r.get("target_entity"),
            rel_type,
        )
        if key in seen:
            continue
        seen.add(key)

        evidence = r.get("evidence", "")
        if evidence and not verify_evidence_verbatim(evidence, abstract, title=title):
            r["evidence_verbatim_status"] = "partial_or_missing"
            if strict:
                continue
        else:
            r["evidence_verbatim_status"] = "verified"

        cleaned.append(r)
    return cleaned
