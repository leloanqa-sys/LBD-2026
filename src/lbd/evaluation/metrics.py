"""Evaluation Metrics: Recall@K, Precision@K, NDCG@K, and MRR (Mean Reciprocal Rank).

Owner: Nguyễn Công Gia Phước (Evaluation, Ground Truth & Benchmarks)

Implements standard information retrieval and link-prediction ranking metrics used
across the temporal backtesting benchmarks, rare-disease evaluations, and the
4-class landmark drug repurposing ablation studies.
"""
from __future__ import annotations

import math
from typing import List, Set


def calculate_recall_at_k(ranked_disease_ids: List[str], ground_truth_ids: Set[str], k: int) -> float:
    """Computes Recall@K: fraction of ground-truth positive diseases appearing in Top-K."""
    top_k = set(ranked_disease_ids[:k])
    if not ground_truth_ids:
        return 0.0
    hits = len(top_k.intersection(ground_truth_ids))
    return hits / len(ground_truth_ids)


def calculate_precision_at_k(ranked_disease_ids: List[str], ground_truth_ids: Set[str], k: int) -> float:
    """Computes Precision@K: fraction of Top-K predictions that are ground-truth positives."""
    if k <= 0:
        return 0.0
    top_k = set(ranked_disease_ids[:k])
    hits = len(top_k.intersection(ground_truth_ids))
    return hits / float(k)


def calculate_mrr(ranked_disease_ids: List[str], ground_truth_ids: Set[str]) -> float:
    """Computes Reciprocal Rank (1 / rank of first positive hit) for a ranked disease list."""
    for rank, d_id in enumerate(ranked_disease_ids, start=1):
        if d_id in ground_truth_ids:
            return 1.0 / rank
    return 0.0


def calculate_ndcg_at_k(ranked_disease_ids: List[str], ground_truth_ids: Set[str], k: int) -> float:
    """Computes binary Normalized Discounted Cumulative Gain (NDCG@K)."""
    if k <= 0 or not ground_truth_ids:
        return 0.0
    dcg = 0.0
    for idx, d_id in enumerate(ranked_disease_ids[:k]):
        if d_id in ground_truth_ids:
            dcg += 1.0 / math.log2(idx + 2)
    ideal_hits = min(len(ground_truth_ids), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0
