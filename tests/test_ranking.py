import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

"""Unit tests for Candidate Disease Scoring and Packaged M2 Ranker."""
from lbd.ranking.m2_ranker import M2RepurposingRanker
from lbd.ranking.rank_candidates import score_disease_candidate


def test_score_disease_candidate():
    scored_paths = [
        {"path_score": 0.85, "net_sign": -1},
        {"path_score": 0.60, "net_sign": -1},
        {"path_score": 0.40, "net_sign": -1},
    ]
    contradicting = []
    res = score_disease_candidate("DOID:9333", "Heart Failure", scored_paths, contradicting)
    assert res["ranking_score"] > 0
    assert res["disease_id"] == "DOID:9333"
    assert len(res["paths"]) == 3


def test_m2_repurposing_ranker_packaged():
    ranker = M2RepurposingRanker(load_graph_for_dfs=False)
    assert ranker.resolve_drug_id("Dapagliflozin") == "Compound::DB06292"
    assert ranker.resolve_drug_id("Aspirin") == "Compound::DB00945"
    assert ranker.resolve_drug_id("Colchicine") == "Compound::DB01394"

    res = ranker.rank_drug("Dapagliflozin", top_n_novel=5, include_dfs_paths=False)
    assert res["drug_id"] == "Compound::DB06292"
    assert res["total_diseases_evaluated"] == 137
    assert len(res["top_novel_candidates"]) == 5
    assert 0.0 < res["top_novel_candidates"][0]["score"] < 1.0

    contrib = ranker.compute_feature_contributions("Dapagliflozin", "Disease::DOID:784", top_k=5)
    assert contrib["disease_id"] == "Disease::DOID:784"
    assert contrib["probability_score"] > 0.70
    assert len(contrib["top_positive_features"]) > 0
