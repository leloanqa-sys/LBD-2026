import sys
from pathlib import Path
from collections import Counter, defaultdict

import pytest

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
            "partition": "ADMITTED_KG",
            "assertion_status": "asserted",
            "publication_date": "2018-12-31",
            "confidence_score": 0.95,
            "quotes": [{"pmid": "99999999", "text": "Dapagliflozin strongly inhibits SLC5A2 in kidney cells."}],
        }
    ]
    added = session.augment_session_edges(novel_edges)
    assert added >= 0
    assert ("Compound::DB06292", "CdG", "Gene::6524") in session.lit_edges
    res = session.rank_drug_with_session("Dapagliflozin", top_n_novel=5, dfs_top_k=2)
    assert len(res["top_novel_candidates"]) == 5
    assert "literature_delta" in res["top_novel_candidates"][0]


def _empty_session_ranker():
    ranker = M2RepurposingRanker.__new__(M2RepurposingRanker)
    ranker.adj = defaultdict(list)
    ranker.out_deg = Counter()
    ranker.in_deg = Counter()
    ranker.names = {}
    ranker.ctd = set()
    ranker.cpd = set()
    ranker._graph_loaded = True
    ranker.w = 0.4
    ranker.coef_dict = {}
    ranker.norm_beta_dict = {}
    return ranker


def _admitted_session_edge(**updates):
    # Synthetic algorithm input, not a claim about these entities or publications.
    edge = dict(source="DB06292", target="6524", source_type="Compound",
                target_type="Gene", relation_type="INHIBITS", sign=-1,
                partition="ADMITTED_KG", assertion_status="asserted",
                publication_date_upper_bound_min="2017-01-01",
                publication_date_upper_bound_max="2018-12-31",
                quotes=[dict(pmid="123", text="synthetic test evidence")])
    return dict(edge, **updates)


@pytest.mark.parametrize("updates", [
    dict(partition="REVIEW"),
    dict(partition="REJECT"),
    dict(partition=None),
    dict(assertion_status="negated"),
    dict(assertion_status="uncertain"),
    dict(assertion_status=None),
    dict(publication_date_upper_bound_max="2019-01-01"),
    dict(publication_date_upper_bound_max=None),
    dict(publication_date_upper_bound_max="invalid"),
    dict(publication_date="2025-01-01"),
    dict(publication_date_upper_bound_min="2019-01-01"),
])
def test_session_rejects_ineligible_edges_without_graph_mutation(updates):
    ranker = _empty_session_ranker()
    assert ranker.augment_session_edges([_admitted_session_edge(**updates)]) == 0
    assert not ranker.adj and not ranker.out_deg and not ranker.in_deg
    assert not ranker.names and not ranker.lit_edges
    assert ranker.session_augmented_count == 0


def test_session_requires_date_and_honors_filter_cutoff():
    ranker = _empty_session_ranker()
    edge = _admitted_session_edge()
    del edge["publication_date_upper_bound_min"]
    del edge["publication_date_upper_bound_max"]
    assert ranker.augment_session_edges([edge]) == 0
    edge["publication_date"] = "2018-12-31"
    assert ranker.augment_session_edges([edge], publication_cutoff="2017-12-31") == 0
    assert ranker.augment_session_edges([edge]) == 1


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("metaedge,relation", [
    ("DuG", "INCREASES"), ("DdG", "DECREASES"), ("DaG", "ASSOCIATED_WITH"),
])
def test_session_disease_gene_edges_are_reachable_in_dfs(reverse, metaedge, relation):
    ranker = _empty_session_ranker()
    compound = _admitted_session_edge()
    disease_gene = _admitted_session_edge(
        source="DOID:784" if reverse else "6524",
        target="6524" if reverse else "DOID:784",
        source_type="Disease" if reverse else "Gene",
        target_type="Gene" if reverse else "Disease",
        relation_type=relation, metaedge=metaedge,
    )
    assert ranker.augment_session_edges([compound, disease_gene]) == 2
    assert ranker.augment_session_edges([compound, disease_gene]) == 0
    assert ranker.out_deg[("Disease::DOID:784", metaedge)] == 1
    assert ranker.in_deg[("Gene::6524", metaedge)] == 1
    n_paths, paths = ranker.explain_pair_dfs("Compound::DB06292", "Disease::DOID:784")
    assert n_paths == 1
    assert paths[0]["nodes"] == ["Compound::DB06292", "Gene::6524", "Disease::DOID:784"]
    assert paths[0]["is_literature_augmented"]
    assert paths[0]["net_sign"] == {"DuG": -1, "DdG": 1, "DaG": 0}[metaedge]
    assert paths[0]["supporting_pmids"] == ["123"]


def test_session_augmentation_does_not_mutate_base_graph():
    base = _empty_session_ranker()
    session = base.clone_for_session()
    assert session.augment_session_edges([_admitted_session_edge()]) == 1
    assert not base.adj and not base.out_deg and not base.in_deg
    assert not hasattr(base, "lit_edges")

