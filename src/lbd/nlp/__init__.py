"""Biomedical Natural Language Processing (NLP) Package.

Unified architecture for clinical and biomedical relation extraction:
- zoning.py: Section/discourse zoning, multi-ontology grounding, L2 coreference resolution.
- events.py: Contextual bio-event frame extraction, cross-sentence candidate pairing, blinded LLM.
- polarity.py: Polarity algebra, clause-level negation, and contraindication guards.
- classifier.py: Calibrated multi-class statistical relation scorer.
- fusion.py: Hierarchical Bayesian Beta-Binomial evidence fusion into Knowledge Graph edges.
- engine.py: Canonical NLPEngine orchestrating the end-to-end pipeline.
- relation_extraction.py: Production Contract B adapter interface.
- ner.py: PubTator 3.0 API & biomedical lexicon entity annotator.
- postprocess.py: Verbatim evidence checking and deduplication.
- sentence_splitter.py: Abbreviation-aware biomedical sentence splitter.
"""

from .classifier import (
    BiomedicalRelationClassifier,
    load_relation_classifier,
    load_v3_engine,
    ZonedPoolingMultiClassBioREEngineV3,
)
from .engine import (
    NLPEngine,
    BiomedicalNLPEngine,
    V7HybridEngine,
)
from .events import (
    BioEventEngine,
    BioEventFrame,
    ProjectedEdgeCandidate,
    Layer3_BioEventAndBlindedLLMEngine,
)
from .fusion import (
    BayesianFusionEngine,
    BayesianFusedEdge,
    Layer5_BayesianBetaBinomialFuser,
)
from .ner import (
    annotate_abstract,
    annotate_abstract_entities,
    annotate_abstract_hybrid,
    annotate_abstract_pubtator,
    annotate_abstract_pubtator_live,
)
from .polarity import (
    PolarityEngine,
    PolarityResolution,
    Layer4_PolarityAndGradeReasoner,
)
from .postprocess import (
    postprocess_relations,
    verify_evidence_verbatim,
)
from .relation_extraction import (
    extract_relations,
    extract_relations_from_text,
    extract_relations_v3,
)
from .sentence_splitter import split_biomedical_sentences
from .zoning import (
    ZonedSentence,
    ZoningEngine,
    Layer12_ZoningAndGroundingEngine,
)

__all__ = [
    # Canonical Engine & Layers
    "NLPEngine",
    "BiomedicalNLPEngine",
    "ZoningEngine",
    "ZonedSentence",
    "BioEventEngine",
    "BioEventFrame",
    "ProjectedEdgeCandidate",
    "PolarityEngine",
    "PolarityResolution",
    "BayesianFusionEngine",
    "BayesianFusedEdge",
    "BiomedicalRelationClassifier",
    "load_relation_classifier",
    # Adapter Functions
    "extract_relations",
    "extract_relations_from_text",
    "annotate_abstract",
    "annotate_abstract_pubtator",
    "annotate_abstract_entities",
    "split_biomedical_sentences",
    "verify_evidence_verbatim",
    "postprocess_relations",
    # Backward Compatibility
    "V7HybridEngine",
    "Layer12_ZoningAndGroundingEngine",
    "Layer3_BioEventAndBlindedLLMEngine",
    "Layer4_PolarityAndGradeReasoner",
    "Layer5_BayesianBetaBinomialFuser",
    "load_v3_engine",
    "extract_relations_v3",
    "annotate_abstract_hybrid",
    "ZonedPoolingMultiClassBioREEngineV3",
]
