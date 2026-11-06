"""Merge Hetionet and NLP Extracted Edges into NetworkX MultiDiGraph (with native fallback)."""
from typing import Dict, List, Any
from lbd.common.logging import get_logger

logger = get_logger("build_graph")

import networkx as nx

SIGN_MAP = {
    "INHIBITS": -1,
    "DECREASES": -1,
    "TREATS": -1,
    "PALLIATES": -1,
    "ACTIVATES": 1,
    "CAUSES": 1,
    "INCREASES": 1,
    "ASSOCIATED_WITH": 0
}

def create_knowledge_graph(hetio_subgraph: Dict[str, Any], nlp_relations: List[Dict[str, Any]]):
    G = nx.MultiDiGraph()
    
    # 1. Add Hetionet nodes
    for node in hetio_subgraph.get("nodes", []):
        G.add_node(node["id"], canonical_name=node["canonical_name"], type=node.get("type", "Unknown"))
        
    # Preserve source-backed edge attributes instead of resetting provenance.
    for edge in hetio_subgraph.get("edges", []):
        attrs = {k: v for k, v in edge.items() if k not in ("source", "target")}
        G.add_edge(edge["source"], edge["target"], **attrs)

    for r in nlp_relations:
        is_contract_d = "source" in r and "target" in r
        src = r.get("source") if is_contract_d else r.get("source_standard_id") or r.get("source_entity")
        tgt = r.get("target") if is_contract_d else r.get("target_standard_id") or r.get("target_entity")
        if not src or not tgt:
            raise ValueError("NLP edge is missing source/target identifiers")
        if r.get("partition") in {"REJECTED", "PROVISIONAL_QUEUE", "COHORT_CONTEXT_KG"}:
            continue
        attrs = dict(r)
        attrs.pop("source", None); attrs.pop("target", None)
        rel_type = r.get("relation_type", "ASSOCIATED_WITH")
        attrs.update(
            relation_type=rel_type,
            metaedge=r.get("metaedge", "NLP_Edge"),
            sign=r.get("sign", SIGN_MAP.get(rel_type, 0)),
            source_dataset=r.get("source_dataset", "nlp_literature"),
            evidence_ids=r.get("evidence_ids", [r["relation_id"]] if r.get("relation_id") else []),
            pmids=r.get("pmids", [r["pmid"]] if r.get("pmid") else []),
            assertion_status=r.get("assertion_status", "asserted"),
            admission_verified=is_contract_d,
        )
        G.add_edge(src, tgt, **attrs)
    logger.info(f"Knowledge Graph constructed with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    return G
