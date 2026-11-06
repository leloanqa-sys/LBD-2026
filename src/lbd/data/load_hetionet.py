"""Load an explicitly supplied local Hetionet-format snapshot; never fabricate data."""
import hashlib
import json
from pathlib import Path


def load_hetionet_subgraph(drug_id="DB06292", graph_path=None):
    if graph_path is None:
        raise ValueError("Supply graph_path to a versioned local snapshot; no mock fallback")
    path = Path(graph_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    nodes, edges = data["nodes"], data["edges"]
    ids = {n["id"] for n in nodes}
    if len(ids) != len(nodes):
        raise ValueError("Duplicate node IDs")
    if drug_id not in ids:
        raise ValueError(f"Drug {drug_id} is absent from snapshot")
    if any(e["source"] not in ids or e["target"] not in ids for e in edges):
        raise ValueError("Edge endpoint is missing from snapshot")
    return {**data, "metadata": {**data.get("metadata", {}),
            "loaded_from": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "upstream_authenticity_verified": False}}
