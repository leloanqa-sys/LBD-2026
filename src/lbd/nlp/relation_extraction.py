"""Biomedical Relation Extraction Adapter.

Provides standardized interface to extract Contract B formatted relation records
from biomedical abstracts using the canonical NLPEngine.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from .engine import NLPEngine
from .postprocess import postprocess_relations

_DEFAULT_ENGINE: Optional[NLPEngine] = None


def get_default_engine() -> NLPEngine:
    """Lazily loads and caches the canonical NLP engine."""
    global _DEFAULT_ENGINE
    if _DEFAULT_ENGINE is None:
        try:
            _DEFAULT_ENGINE = NLPEngine.load_default(dataset_mode="biored")
        except Exception:
            # Fallback to rule-based engine if checkpoints are not available
            _DEFAULT_ENGINE = NLPEngine()
    return _DEFAULT_ENGINE


def extract_relations_from_text(
    pmid: str,
    abstract: str,
    title: str = "",
    threshold: Optional[float] = None,
    engine: Optional[Any] = None,
    strict_postprocess: bool = True,
    **kwargs: Any,
) -> List[Dict[str, Any]]:
    """Extracts verified biomedical relations conforming to Contract B (relation.schema.json).

    Args:
        pmid: PubMed ID.
        abstract: Abstract text.
        title: Article title.
        threshold: Optional confidence threshold override.
        engine: Optional custom NLPEngine or classifier engine.
        strict_postprocess: Whether to strictly filter non-conforming relations.

    Returns:
        List of relation dictionaries conforming to Contract B.
    """
    if not abstract and not title:
        return []

    nlp_engine: NLPEngine
    if isinstance(engine, NLPEngine):
        nlp_engine = engine
    elif engine is not None and hasattr(engine, "predict_multiclass"):
        # Wrapped classifier passed as engine
        nlp_engine = NLPEngine(classifier=engine, relation_threshold=threshold)
    else:
        nlp_engine = get_default_engine()
        if threshold is not None:
            nlp_engine.relation_threshold = float(threshold)

    doc = {
        "pmid": str(pmid),
        "title": title or "",
        "abstract": abstract or "",
    }
    _, events, _ = nlp_engine.run_document(doc)
    contract_b_records, _, _ = nlp_engine.fuse_corpus(events)

    # Postprocess for verbatim check and deduplication
    cleaned = postprocess_relations(
        contract_b_records,
        abstract,
        title=title,
        strict=strict_postprocess,
    )
    return cleaned


# Backward-compatible alias
extract_relations_v3 = extract_relations_from_text
extract_relations = extract_relations_from_text
