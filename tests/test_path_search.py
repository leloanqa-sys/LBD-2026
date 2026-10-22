import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

"""Unit tests for DFS path search and historical disease filtering."""
from lbd.graph.build_graph import create_knowledge_graph
from lbd.graph.path_search import search_metapaths

def test_search_metapaths_prunes_historical():
    hetio_subgraph = {
        "nodes": [
            {"id": "DB06292", "canonical_name": "Dapagliflozin"},
            {"id": "6524", "canonical_name": "SLC5A2"},
            {"id": "GO:0006814", "canonical_name": "Sodium ion transport"},
            {"id": "DOID:9333", "canonical_name": "Heart Failure"},
            {"id": "DOID:9352", "canonical_name": "Type 2 Diabetes Mellitus"}
        ],
        "edges": [
            {"source": "DB06292", "target": "6524", "relation_type": "INHIBITS", "sign": -1},
            {"source": "6524", "target": "GO:0006814", "relation_type": "DECREASES", "sign": -1},
            {"source": "GO:0006814", "target": "DOID:9333", "relation_type": "CAUSES", "sign": 1},
            {"source": "6524", "target": "DOID:9352", "relation_type": "TREATS", "sign": -1},
            {"source": "DOID:9352", "target": "DOID:9333", "relation_type": "ASSOCIATED_WITH", "sign": 0}
        ]
    }
    
    G = create_knowledge_graph(hetio_subgraph, [])
    paths = search_metapaths(G, "DB06292", "DOID:9333", min_length=3, max_length=5, filter_historical=True)
    assert len(paths) >= 1
    # Check that DOID:9352 was not traversed as an intermediate step
    for p in paths:
        intermediate_nodes = [e["source"] for e in p[1:]]
        assert "DOID:9352" not in intermediate_nodes
