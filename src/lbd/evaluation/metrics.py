"""Evaluation Metrics: Recall@K and MRR (Mean Reciprocal Rank)."""
from typing import List, Dict, Any, Set

def calculate_recall_at_k(ranked_disease_ids: List[str], ground_truth_ids: Set[str], k: int) -> float:
    top_k = set(ranked_disease_ids[:k])
    if not ground_truth_ids:
        return 0.0
    hits = len(top_k.intersection(ground_truth_ids))
    return hits / len(ground_truth_ids)

def calculate_mrr(ranked_disease_ids: List[str], ground_truth_ids: Set[str]) -> float:
    for rank, d_id in enumerate(ranked_disease_ids, start=1):
        if d_id in ground_truth_ids:
            return 1.0 / rank
    return 0.0
