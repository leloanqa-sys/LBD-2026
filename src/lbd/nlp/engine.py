"""Canonical Biomedical NLP Relation Extraction and Knowledge Graph Fusion Engine.

Unifies:
1. Zoning & 2-pass multi-ontology grounding with L2 coreference resolution (zoning.py)
2. Bio-event extraction & cross-sentence candidate pairing with counterfactual blinding (events.py)
3. Polarity algebra & contraindication resolution (polarity.py)
4. Statistical relation calibration & probability gating (classifier.py)
5. Hierarchical Bayesian Beta-Binomial evidence fusion into Knowledge Graph edges (fusion.py)
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .classifier import BiomedicalRelationClassifier, load_relation_classifier
from .events import BioEventEngine, BioEventFrame
from .fusion import BayesianFusionEngine, BayesianFusedEdge
from .zoning import ZoningEngine

CLASS_NAMES = ("NO_RELATION", "POSITIVE", "NEGATIVE", "ASSOCIATION")
ENTITY_TYPE_MAP = {
    "Compound": "Chemical",
    "Chemical": "Chemical",
    "Disease": "Disease",
    "Gene": "Gene",
    "Variant": "Variant",
}


class NLPEngine:
    """Canonical Biomedical NLP Relation Extraction and Fusion Engine."""

    def __init__(
        self,
        classifier: Optional[Any] = None,
        *,
        dataset_mode: str = "biored",
        zoning_engine: Optional[ZoningEngine] = None,
        event_engine: Optional[BioEventEngine] = None,
        fusion_engine: Optional[BayesianFusionEngine] = None,
        relation_threshold: Optional[float] = None,
    ) -> None:
        self.classifier = classifier
        if self.classifier is not None:
            if hasattr(getattr(self.classifier, "rf_clf", None), "n_jobs"):
                self.classifier.rf_clf.n_jobs = 1
            if relation_threshold is not None:
                self.relation_threshold = float(relation_threshold)
            else:
                self.relation_threshold = float(getattr(self.classifier, "relation_threshold", 0.5))
        else:
            self.relation_threshold = float(relation_threshold if relation_threshold is not None else 0.5)

        if dataset_mode not in {"biored", "bc5cdr"}:
            raise ValueError("dataset_mode must be 'biored' or 'bc5cdr'")
        self.dataset_mode = dataset_mode
        self.zoning = zoning_engine or ZoningEngine()
        self.events = event_engine or BioEventEngine(enable_live_llm=False)
        self.fusion = fusion_engine or BayesianFusionEngine()

    @classmethod
    def load_default(
        cls,
        dataset_mode: str = "biored",
        checkpoint_path: Optional[Path] = None,
        enable_live_llm: bool = False,
    ) -> NLPEngine:
        """Loads canonical engine with default pretrained calibrated classifier."""
        if checkpoint_path is None:
            ckpt_name = f"model_v3_{dataset_mode}_multiclass.pkl" if dataset_mode == "biored" else f"model_v3_{dataset_mode}.pkl"
            checkpoint_path = Path(__file__).parent / "checkpoints" / ckpt_name
        classifier = load_relation_classifier(checkpoint_path)
        return cls(
            classifier=classifier,
            dataset_mode=dataset_mode,
            event_engine=BioEventEngine(enable_live_llm=enable_live_llm),
        )

    @staticmethod
    def _prepare_classifier_document(
        pmid: str,
        title: str,
        abstract: str,
        grounded_doc: Dict[str, Any],
        external_entities: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        entities: List[Dict[str, Any]] = []
        seen_mentions = set()

        for entity in external_entities or []:
            mapped_type = ENTITY_TYPE_MAP.get(str(entity.get("type") or ""))
            entity_id = str(entity.get("id") or "")
            text = str(entity.get("text") or "")
            if not mapped_type or not entity_id or not text or entity.get("start") is None or entity.get("end") is None:
                continue
            start, end = int(entity["start"]), int(entity["end"])
            key = (entity_id, start, end, mapped_type)
            if key in seen_mentions:
                continue
            seen_mentions.add(key)
            entities.append({
                "start": start,
                "end": end,
                "text": text,
                "type": mapped_type,
                "entity_id": entity_id,
            })

        for sentence in grounded_doc["sentences"]:
            for entity in sentence.entities:
                mapped_type = ENTITY_TYPE_MAP.get(str(entity.get("type") or ""))
                entity_id = str(entity.get("id") or "")
                matched_text = str(entity.get("matched_text") or "")
                if not mapped_type or not entity_id or not matched_text:
                    continue
                start = int(sentence.char_start) + int(entity.get("span_start", 0))
                end = int(sentence.char_start) + int(entity.get("span_end", len(matched_text)))
                key = (entity_id, start, end, mapped_type)
                if key in seen_mentions:
                    continue
                seen_mentions.add(key)
                entities.append({
                    "start": start,
                    "end": end,
                    "text": matched_text,
                    "type": mapped_type,
                    "entity_id": entity_id,
                })
        return {"pmid": str(pmid), "title": title, "abstract": abstract, "entities": entities, "title_sep_len": 1}

    def score_grounded_document(
        self,
        pmid: str,
        title: str,
        abstract: str,
        grounded_doc: Dict[str, Any],
        external_entities: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[Tuple[str, str], Dict[str, Any]]:
        """Return document-level statistical relation scores indexed by canonical entity-ID pair."""
        if self.classifier is None:
            return {}
        c_doc = self._prepare_classifier_document(pmid, title, abstract, grounded_doc, external_entities)
        probabilities, _, metadata = self.classifier.predict_multiclass([c_doc], dataset_mode=self.dataset_mode)
        scores: Dict[Tuple[str, str], Dict[str, Any]] = {}
        for row, meta in zip(probabilities, metadata):
            entity_pair = tuple(sorted((str(meta["entity1_id"]), str(meta["entity2_id"]))))
            class_probs = {name: float(row[index]) for index, name in enumerate(CLASS_NAMES)}
            scores[entity_pair] = {
                "prob_has_relation": float(1.0 - row[0]),
                "class_probabilities": class_probs,
                "predicted_class": str(meta.get("pred_active_class") or ""),
                "threshold": self.relation_threshold,
                "entity1_type": meta.get("entity1_type"),
                "entity2_type": meta.get("entity2_type"),
            }
        return scores

    def run_document(
        self,
        document: Dict[str, Any],
        *,
        external_entities: Optional[List[Dict[str, Any]]] = None,
        apply_relation_gate: bool = True,
    ) -> Tuple[Dict[str, Any], List[BioEventFrame], Dict[Tuple[str, str], Dict[str, Any]]]:
        """Runs end-to-end zoning, entity grounding, candidate scoring, and bio-event extraction."""
        pmid = str(document["pmid"])
        title = str(document.get("title") or "")
        abstract = str(document.get("abstract") or "")
        grounded = self.zoning.ground_and_resolve_coref(
            title, abstract, pmid=pmid, external_entities=external_entities
        )
        source_mentions = external_entities if external_entities is not None else grounded.get("external_entities")
        scores = {}
        if self.classifier is not None and apply_relation_gate:
            scores = self.score_grounded_document(pmid, title, abstract, grounded, source_mentions)

        events = self.events.extract_document_events(
            pmid,
            int(document.get("print_year") or document.get("year") or 0),
            grounded,
            v3_relation_scores=(scores if (self.classifier and apply_relation_gate) else None),
            v3_threshold=(self.relation_threshold if (self.classifier and apply_relation_gate) else None),
        )
        return grounded, events, scores

    def fuse_corpus(
        self, events: List[BioEventFrame]
    ) -> Tuple[List[Dict[str, Any]], List[BayesianFusedEdge], List[Dict[str, Any]]]:
        """Fuses events into Contract B relation records and Contract D KG edges."""
        contract_b, fused = self.fusion.fuse_edges(events)
        contract_d = self.fusion.to_contract_d_records(fused)
        return contract_b, fused, contract_d

    def extract_relations(
        self,
        pmid: str,
        abstract: str,
        title: str = "",
        external_entities: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """Convenience method: runs document and returns Contract B relation records."""
        doc = {"pmid": pmid, "title": title, "abstract": abstract}
        _, events, _ = self.run_document(doc, external_entities=external_entities)
        contract_b, _, _ = self.fuse_corpus(events)
        return contract_b


# Canonical aliases
V7HybridEngine = NLPEngine
BiomedicalNLPEngine = NLPEngine
