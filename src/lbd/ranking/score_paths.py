"""[LEGACY COMPATIBILITY MODULE] Path-level Scoring with Signed Rules and Direction Fit.
Notice: This module contains legacy heuristic path formulas from Sprint 3.
For production topological ranking, use `lbd.ranking.m2_ranker.M2RepurposingRanker` or `lbd.ranking.core`.
"""
from typing import List, Dict, Any
from lbd.ranking.features import compute_path_quality, compute_path_penalty

def calculate_net_sign(edges: List[Dict[str, Any]]) -> int:
    signs = [e.get("sign", 0) for e in edges]
    if 0 in signs:
        return 0
    prod = 1
    for s in signs:
        prod *= s
    return prod

def get_direction_fit(net_sign: int) -> float:
    if net_sign == -1:
        return 1.5 # Therapeutic alignment
    elif net_sign == 0:
        return 1.0 # Suggestive link
    else:
        return 0.0 # Contradictory / adverse effect

def score_individual_path(edges: List[Dict[str, Any]], hub_degrees: List[int]) -> Dict[str, Any]:
    EQ = compute_path_quality(edges)
    net_sign = calculate_net_sign(edges)
    dfit = get_direction_fit(net_sign)
    penalty = compute_path_penalty(len(edges), hub_degrees)
    
    score = (EQ * dfit) / penalty
    return {
        "path_score": score,
        "net_sign": net_sign,
        "direction_fit": dfit,
        "eq": EQ,
        "penalty": penalty
    }
