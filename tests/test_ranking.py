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


def test_repurank_core_integration():
    """Verify that pure repurank core is accessible via lbd.ranking.core and top-level repurank."""
    from lbd.ranking import core
    import repurank

    assert hasattr(core, "features")
    assert hasattr(core, "learning")
    assert hasattr(core, "signs")
    assert hasattr(core, "graph")
    assert hasattr(core, "evidence")
    assert hasattr(repurank, "features")

    # Verify signature reversal logic
    rev = core.signs.signature_reversal({"GENE1": 1.0, "GENE2": -1.0}, {"GENE1": -1.0, "GENE2": 1.0})
    assert rev["signature_available"] == 1
    assert rev["signature_reversal_cosine"] > 0.99
    assert rev["signature_opposed_fraction"] == 1.0


def test_m2_normalized_contrib_and_session():
    """Verify normalized beta sensitivity, DaG neutrality, and non-truncated session augmentation."""
    ranker = M2RepurposingRanker(load_graph_for_dfs=True)
    assert hasattr(ranker, "norm_beta_dict")
    assert len(ranker.norm_beta_dict) == len(ranker.selected_feature_names)

    # Test explain_pair_dfs returns all lit paths when requested
    n_paths, top_paths, all_lit = ranker.explain_pair_dfs(
        "Compound::DB06292", "Disease::DOID:784", top_k=2, return_all_lit=True
    )
    assert n_paths > 0
    assert len(top_paths) <= 2
    assert isinstance(all_lit, list)
    for p in top_paths:
        assert "norm_beta" in p
        assert "contrib" in p

    # Test Session live augmentation with a novel bridge edge
    session = ranker.clone_for_session()
    novel_edges = [
        {
            "source_standard_id": "Compound::DB06292",
            "target_standard_id": "Gene::6524",
            "source_type": "Compound",
            "target_type": "Gene",
            "relation_type": "INHIBITS",
            "confidence_score": 0.95,
            "quotes": [{"pmid": "99999999", "text": "Dapagliflozin strongly inhibits SLC5A2 in kidney cells."}],
        }
    ]
    added = session.augment_session_edges(novel_edges)
    assert added >= 0
    res = session.rank_drug_with_session("Dapagliflozin", top_n_novel=5, dfs_top_k=2)
    assert len(res["top_novel_candidates"]) == 5
    assert "literature_delta" in res["top_novel_candidates"][0]

