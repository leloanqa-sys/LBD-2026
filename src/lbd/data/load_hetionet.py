"""Hetionet v1.0 Loader and Subgraph Extractor."""
import json
from pathlib import Path
from typing import Dict, List, Any
from lbd.common.logging import get_logger

logger = get_logger("load_hetionet")

def load_hetionet_subgraph(drug_id: str = "DB06292") -> Dict[str, List[Dict[str, Any]]]:
    """Loads a mock/sample Hetionet subgraph rooted at drug_id."""
    nodes = [
        {"id": "DB06292", "canonical_name": "Dapagliflozin", "type": "Compound"},
        {"id": "6524", "canonical_name": "SLC5A2", "type": "Gene"},
        {"id": "GO:0006814", "canonical_name": "Sodium ion transport", "type": "BiologicalProcess"},
        {"id": "GO:0003094", "canonical_name": "Glomerular filtration", "type": "BiologicalProcess"},
        {"id": "DOID:9333", "canonical_name": "Heart Failure", "type": "Disease"},
        {"id": "DOID:784", "canonical_name": "Chronic Kidney Disease", "type": "Disease"},
        {"id": "DOID:9352", "canonical_name": "Type 2 Diabetes Mellitus", "type": "Disease"}
    ]
    edges = [
        {"source": "DB06292", "target": "6524", "metaedge": "CbG", "relation_type": "INHIBITS", "sign": -1, "source_dataset": "hetionet_v1.0"},
        {"source": "6524", "target": "GO:0006814", "metaedge": "GpBP", "relation_type": "ACTIVATES", "sign": 1, "source_dataset": "hetionet_v1.0"},
        {"source": "GO:0006814", "target": "DOID:9333", "metaedge": "GpBP", "relation_type": "DECREASES", "sign": -1, "source_dataset": "hetionet_v1.0"},
        {"source": "DB06292", "target": "DOID:9352", "metaedge": "CtD", "relation_type": "TREATS", "sign": -1, "source_dataset": "hetionet_v1.0"}
    ]
    return {"nodes": nodes, "edges": edges}
