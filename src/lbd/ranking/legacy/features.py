"""Compute Edge Confidence, Hub Node Damping, and Path Quality Features."""
import math
from typing import Dict, List, Any

TIER_WEIGHTS = {1: 1.0, 2: 0.7, 3: 0.4}

def compute_edge_weight(edge: Dict[str, Any], pmid_max: int = 100) -> float:
    tier = edge.get("tier", 1)
    w_tier = TIER_WEIGHTS.get(tier, 0.4)
    w_study = 0.9 # Default for validated in vivo / clinical trial
    num_pmids = len(edge.get("pmids", []))
    
    # Saturation formula: (1 + ln(1 + |PMIDs|)) / (1 + ln(1 + PMID_max))
    sat_factor = (1.0 + math.log(1.0 + num_pmids)) / (1.0 + math.log(1.0 + pmid_max))
    w_e = w_tier * w_study * sat_factor
    return max(0.01, min(1.0, w_e))

def compute_path_quality(edges: List[Dict[str, Any]]) -> float:
    """Computes EQ(p) = (Product W(e))^(1/L)."""
    if not edges:
        return 0.0
    L = len(edges)
    product = 1.0
    for e in edges:
        product *= compute_edge_weight(e)
    return product ** (1.0 / L)

def compute_path_penalty(length: int, hub_degrees: List[int], gamma: float = 1.5, w: float = 0.4) -> float:
    """Penalty(p) = gamma^(L-1) * Product (deg(v))^w."""
    penalty = gamma ** (length - 1)
    for deg in hub_degrees:
        penalty *= (deg ** w) if deg > 0 else 1.0
    return max(1.0, penalty)
