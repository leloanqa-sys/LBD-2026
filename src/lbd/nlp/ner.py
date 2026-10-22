"""Named Entity Recognition (NER) Client via PubTator 3.0 API."""
from typing import List, Dict, Any
from lbd.common.logging import get_logger

logger = get_logger("nlp_ner")

def annotate_abstract_pubtator(pmid: str, abstract_text: str) -> List[Dict[str, Any]]:
    """Mock/fallback interface for PubTator 3.0 entity extraction."""
    entities = []
    text_lower = abstract_text.lower()
    if "dapagliflozin" in text_lower or "farxiga" in text_lower:
        entities.append({"text": "Dapagliflozin", "type": "Compound", "id": "DB06292"})
    if "sglt2" in text_lower or "slc5a2" in text_lower:
        entities.append({"text": "SLC5A2", "type": "Gene", "id": "6524"})
    if "heart failure" in text_lower:
        entities.append({"text": "Heart Failure", "type": "Disease", "id": "DOID:9333"})
    if "chronic kidney disease" in text_lower or "ckd" in text_lower:
        entities.append({"text": "Chronic Kidney Disease", "type": "Disease", "id": "DOID:784"})
    if "natriuresis" in text_lower:
        entities.append({"text": "Natriuresis", "type": "BiologicalProcess", "id": "GO:0006814"})
    if "plasma volume" in text_lower:
        entities.append({"text": "Plasma Volume", "type": "BiologicalProcess", "id": "GO:0003094"})
    return entities
