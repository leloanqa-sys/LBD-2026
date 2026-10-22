import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

"""Unit tests for Candidate Disease Scoring."""
from lbd.ranking.rank_candidates import score_disease_candidate

def test_score_disease_candidate():
    scored_paths = [
        {"path_score": 0.85, "net_sign": -1},
        {"path_score": 0.60, "net_sign": -1},
        {"path_score": 0.40, "net_sign": -1}
    ]
    contradicting = []
    res = score_disease_candidate("DOID:9333", "Heart Failure", scored_paths, contradicting)
    assert res["ranking_score"] > 0
    assert res["disease_id"] == "DOID:9333"
    assert len(res["paths"]) == 3
