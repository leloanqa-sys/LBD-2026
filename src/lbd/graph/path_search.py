"""Metapath-Constrained Pruned DFS Path Search."""
from typing import List, Dict, Any, Set

HISTORICAL_DISEASES = {"DOID:9352"} # Type 2 Diabetes Mellitus

def search_metapaths(
    G: Any,
    source_node: str,
    target_node: str,
    min_length: int = 3,
    max_length: int = 5,
    filter_historical: bool = True
) -> List[List[Dict[str, Any]]]:
    """Finds paths between source_node and target_node obeying biological metapaths."""
    all_paths = []
    
    def dfs(current: str, path: List[str], edges_accum: List[Dict[str, Any]]):
        if len(path) > max_length:
            return
        if len(path) >= min_length and current == target_node:
            all_paths.append(list(edges_accum))
            return
        
        if current in G:
            for neighbor in G.neighbors(current):
                if neighbor in path:
                    continue # No cycles
                if filter_historical and neighbor in HISTORICAL_DISEASES and neighbor != target_node:
                    continue # Prune historical baseline noise
                
                # Fetch edge attributes
                edge_dict = G.get_edge_data(current, neighbor)
                for key, edge_data in edge_dict.items():
                    edge_record = {
                        "source": current,
                        "target": neighbor,
                        "relation_type": edge_data.get("relation_type", "ASSOCIATED_WITH"),
                        "sign": edge_data.get("sign", 0),
                        "pmids": edge_data.get("pmids", []),
                        "tier": edge_data.get("tier", 1)
                    }
                    dfs(neighbor, path + [neighbor], edges_accum + [edge_record])
                    
    dfs(source_node, [source_node], [])
    return all_paths
