"""Graph adapter and temporal admission boundary. No data fabrication."""
import gzip
import hashlib
import json
from datetime import date
from pathlib import Path
import networkx as nx

TEMPORAL_FIELDS = ("source_version", "source_release_date", "evidence_date", "available_at", "retrieved_at", "valid_from", "valid_to")
LABEL_RELATIONS = {"TREATS", "PALLIATES", "CtD", "CpD"}

def node_id(value):
    return "::".join(map(str, value)) if isinstance(value, list) else str(value)

def load_graph(path, cutoff="2018-12-31", strict_temporal=False):
    path = Path(path)
    data = json.loads(gzip.decompress(path.read_bytes()) if path.suffix == ".gz" else path.read_bytes())
    graph = nx.MultiDiGraph()
    for node in data["nodes"]:
        nid = str(node["id"]) if "id" in node else node_id([node["kind"], node["identifier"]])
        if nid in graph: raise ValueError("Duplicate node: " + nid)
        graph.add_node(nid, **{**node, "type":node.get("type",node.get("kind")), "canonical_name":node.get("canonical_name",node.get("name",nid))})
    missing_time = 0
    removed = 0
    for index, edge in enumerate(data["edges"]):
        u,v = node_id(edge["source"]),node_id(edge["target"])
        if u not in graph or v not in graph: raise ValueError("Missing endpoint")
        meta = edge.get("metaedge",edge.get("kind",""))
        rel = edge.get("relation_type",meta)
        # Labels are excluded from ALL feature graph degrees and paths.
        if meta in LABEL_RELATIONS or rel in LABEL_RELATIONS or (meta in {"treats","palliates"}):
            removed += 1; continue
        verified = all(edge.get(k) for k in TEMPORAL_FIELDS if k != "valid_to")
        if verified:
            for key in TEMPORAL_FIELDS:
                if key not in {"source_version", "valid_to"}: date.fromisoformat(edge[key])
            verified = edge["available_at"] <= cutoff and edge["source_release_date"] <= cutoff and edge["evidence_date"] <= cutoff and edge["valid_from"] <= cutoff and (not edge.get("valid_to") or edge["valid_to"] >= cutoff)
        if not verified:
            missing_time += 1
            if strict_temporal: raise ValueError("Unverified/post-cutoff edge: " + str(index))
        attrs = {**edge, "metaedge":meta,"relation_type":rel,"temporal_verified":verified}
        graph.add_edge(u,v,key=str(index),**attrs)
    graph.graph.update(sha256=hashlib.sha256(path.read_bytes()).hexdigest(),cutoff=cutoff,
        temporal_unverified_edges=missing_time,label_edges_removed=removed,strict_temporal=strict_temporal)
    return graph
