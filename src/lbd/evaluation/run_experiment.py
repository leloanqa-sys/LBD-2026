"""Run Temporal Backtesting Benchmark and Ablation Study."""
from typing import List, Dict, Any
from lbd.evaluation.ground_truth import GROUND_TRUTH_TIERS
from lbd.evaluation.metrics import calculate_recall_at_k, calculate_mrr
from lbd.common.logging import get_logger

logger = get_logger("run_experiment")

def evaluate_rankings(ranked_candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
    ranked_ids = [c["disease_id"] for c in ranked_candidates]
    hard_positives = set(GROUND_TRUTH_TIERS["HARD_POSITIVES"].keys())
    
    r_5 = calculate_recall_at_k(ranked_ids, hard_positives, k=5)
    r_10 = calculate_recall_at_k(ranked_ids, hard_positives, k=10)
    mrr = calculate_mrr(ranked_ids, hard_positives)
    
    return {
        "recall@5": r_5,
        "recall@10": r_10,
        "mrr": mrr,
        "total_ranked": len(ranked_ids),
        "hard_positives_evaluated": list(hard_positives)
    }
