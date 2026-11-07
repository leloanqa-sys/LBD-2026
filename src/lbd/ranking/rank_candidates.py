"""[LEGACY COMPATIBILITY MODULE] Aggregate Path Scores to Candidate Diseases (Score(d)).
Notice: This module contains legacy heuristic path formulas from Sprint 3.
For production topological ranking, use `lbd.ranking.m2_ranker.M2RepurposingRanker` or `lbd.ranking.core`.
"""
import math
from typing import List, Dict, Any

def score_disease_candidate(
    disease_id: str,
    disease_name: str,
    scored_paths: List[Dict[str, Any]],
    contradicting_pmids: List[str],
    top_m: int = 5,
    lam: float = 2.0
) -> Dict[str, Any]:
    # Sort paths descending by path_score
    sorted_paths = sorted(scored_paths, key=lambda p: p["path_score"], reverse=True)
    top_paths = sorted_paths[:top_m]
    
    # Harmonic decay aggregation: Sum (PathScore(p_k) / k)
    path_aggregate = 0.0
    for k, p in enumerate(top_paths, start=1):
        path_aggregate += (p["path_score"] / k)
        
    # Contradiction penalty: lambda * ln(1 + |U_neg|)
    penalty_contradiction = lam * math.log(1.0 + len(contradicting_pmids))
    total_score = max(0.0, path_aggregate - penalty_contradiction)
    
    return {
        "disease_id": disease_id,
        "disease_name": disease_name,
        "ranking_score": round(total_score, 4),
        "paths": top_paths,
        "contradiction_pmids": contradicting_pmids,
        "ai_explanation": f"Candidate supported by {len(top_paths)} biological paths. Net score: {round(total_score, 4)}."
    }
