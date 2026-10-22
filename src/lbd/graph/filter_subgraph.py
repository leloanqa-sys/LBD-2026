"""Filter Subgraph within Max Hops and Specific Entity Constraints."""
from typing import Set, Any

def filter_ego_subgraph(G: Any, root_node: str, radius: int = 3) -> Any:
    sub_nodes: Set[str] = set()
    current = {root_node}
    sub_nodes.update(current)
    
    for _ in range(radius):
        next_nodes = set()
        for u in current:
            if u in G:
                for v in G.neighbors(u):
                    next_nodes.add(v)
        sub_nodes.update(next_nodes)
        current = next_nodes
        
    return G.subgraph(list(sub_nodes)).copy()
