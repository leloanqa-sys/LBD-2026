"""Merge Hetionet and NLP Extracted Edges into NetworkX MultiDiGraph (with native fallback)."""
from typing import Dict, List, Any
from lbd.common.logging import get_logger

logger = get_logger("build_graph")

try:
    import networkx as nx
except ImportError:
    class SimpleMultiDiGraph:
        def __init__(self):
            self._nodes = {}
            self._adj = {}
        def add_node(self, node_id: str, **attr):
            self._nodes[node_id] = attr
            if node_id not in self._adj:
                self._adj[node_id] = {}
        def add_edge(self, u: str, v: str, key=None, **attr):
            self.add_node(u)
            self.add_node(v)
            if v not in self._adj[u]:
                self._adj[u][v] = {}
            k = key if key is not None else len(self._adj[u][v])
            self._adj[u][v][k] = attr
            return k
        def number_of_nodes(self) -> int:
            return len(self._nodes)
        def number_of_edges(self) -> int:
            return sum(len(self._adj[u][v]) for u in self._adj for v in self._adj[u])
        def neighbors(self, n: str):
            return iter(list(self._adj.get(n, {}).keys()))
        def get_edge_data(self, u: str, v: str) -> Dict[Any, Dict[str, Any]]:
            return self._adj.get(u, {}).get(v, {})
        def __contains__(self, n: str) -> bool:
            return n in self._nodes
        def subgraph(self, nodes):
            sub = SimpleMultiDiGraph()
            node_set = set(nodes)
            for n in node_set:
                if n in self._nodes:
                    sub.add_node(n, **self._nodes[n])
            for u in node_set:
                if u in self._adj:
                    for v in self._adj[u]:
                        if v in node_set:
                            for k, attr in self._adj[u][v].items():
                                sub.add_edge(u, v, key=k, **attr)
            return sub
        def copy(self):
            return self.subgraph(list(self._nodes.keys()))

    class _MockNX:
        MultiDiGraph = SimpleMultiDiGraph
    nx = _MockNX()

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
        
    # 2. Add Hetionet edges
    for e in hetio_subgraph.get("edges", []):
        G.add_edge(
            e["source"],
            e["target"],
            metaedge=e.get("metaedge", "Unknown"),
            relation_type=e.get("relation_type", "ASSOCIATED_WITH"),
            sign=e.get("sign", 0),
            source_dataset=e.get("source_dataset", "hetionet_v1.0"),
            evidence_ids=[],
            pmids=[],
            tier=1
        )
        
    # 3. Merge NLP extracted edges
    for r in nlp_relations:
        src = r.get("source_standard_id") or r.get("source_entity")
        tgt = r.get("target_standard_id") or r.get("target_entity")
        if not src or not tgt:
            continue
        rel_type = r.get("relation_type", "ASSOCIATED_WITH")
        sign = SIGN_MAP.get(rel_type, 0)
        
        G.add_edge(
            src,
            tgt,
            metaedge="NLP_Edge",
            relation_type=rel_type,
            sign=sign,
            source_dataset="nlp_literature",
            evidence_ids=[r.get("relation_id", "")],
            pmids=[r.get("pmid", "")] if r.get("pmid") else [],
            tier=1 if rel_type in ("INHIBITS", "ACTIVATES", "CAUSES") else 2
        )
    logger.info(f"Knowledge Graph constructed with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    return G
