"""Run Temporal Backtesting Benchmark and Multi-Drug Ablation Evaluation.

Owner: Nguyễn Công Gia Phước (Evaluation, Ground Truth & Benchmarks)

Provides evaluation runners for single-drug rankings (`evaluate_rankings`) and
multi-drug prospective benchmark suites (`evaluate_multi_drug_rankings`).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Set
from lbd.common.logging import get_logger
from lbd.evaluation.ground_truth import GROUND_TRUTH_TIERS, get_drug_ground_truth
from lbd.evaluation.metrics import (
    calculate_mrr,
    calculate_ndcg_at_k,
    calculate_precision_at_k,
    calculate_recall_at_k,
)

logger = get_logger("run_experiment")


def evaluate_rankings(
    ranked_candidates: List[Dict[str, Any]],
    ground_truth_ids: Optional[Set[str]] = None,
) -> Dict[str, Any]:
    """Evaluates a ranked list of disease candidates against ground-truth positives."""
    ranked_ids = [c["disease_id"] for c in ranked_candidates]
    hard_positives = ground_truth_ids if ground_truth_ids is not None else set(GROUND_TRUTH_TIERS["HARD_POSITIVES"].keys())

    r_5 = calculate_recall_at_k(ranked_ids, hard_positives, k=5)
    r_10 = calculate_recall_at_k(ranked_ids, hard_positives, k=10)
    p_5 = calculate_precision_at_k(ranked_ids, hard_positives, k=5)
    p_10 = calculate_precision_at_k(ranked_ids, hard_positives, k=10)
    ndcg_10 = calculate_ndcg_at_k(ranked_ids, hard_positives, k=10)
    mrr = calculate_mrr(ranked_ids, hard_positives)

    return {
        "recall@5": r_5,
        "recall@10": r_10,
        "precision@5": p_5,
        "precision@10": p_10,
        "ndcg@10": ndcg_10,
        "mrr": mrr,
        "total_ranked": len(ranked_ids),
        "hard_positives_evaluated": sorted(list(hard_positives)),
    }


def evaluate_drug_against_landmark_gt(
    drug_id: str,
    ranked_candidates: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Evaluates a specific drug's ranked candidate list against its Landmark Ground Truth."""
    gt_info = get_drug_ground_truth(drug_id)
    positives = set(gt_info.get("hard_positives", {}).keys())
    metrics = evaluate_rankings(ranked_candidates, ground_truth_ids=positives)
    metrics["drug_id"] = drug_id
    metrics["drug_name"] = gt_info.get("drug_name", drug_id)
    return metrics
