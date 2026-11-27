"""
Unit & Integration Tests for Deterministic DOID DAG, Crosswalk, and Tier/LCA-Weighted Session KG.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

LBD_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LBD_ROOT / "src"))

from lbd.ontology import (
    STOP_ROOTS,
    DeterministicCrosswalk,
    DirectedDOIDResolver,
)
from lbd.ranking.m2_ranker import M2RepurposingRanker


@pytest.fixture(scope="module")
def resolver() -> DirectedDOIDResolver:
    return DirectedDOIDResolver()


@pytest.fixture(scope="module")
def crosswalk(resolver: DirectedDOIDResolver) -> DeterministicCrosswalk:
    return DeterministicCrosswalk(resolver=resolver)


@pytest.fixture(scope="module")
def ranker() -> M2RepurposingRanker:
    return M2RepurposingRanker(load_graph_for_dfs=True)


# ---------------------------------------------------------------------------
# 1. DirectedDOIDResolver tests
# ---------------------------------------------------------------------------
def test_doid_resolver_indexes_all_137_hetionet_diseases(resolver: DirectedDOIDResolver) -> None:
    stats = resolver.coverage_stats()
    assert stats["hetionet_diseases_total"] == 137
    assert stats["hetionet_diseases_indexed"] == 137
    assert stats["active_doid_terms"] > 10000
    assert stats["obsolete_doid_terms"] > 0

    # Every Hetionet disease must resolve to itself with exact match and lca_decay == 1.0
    for h_doid in resolver.hetionet_diseases:
        hits = resolver.resolve(h_doid)
        assert len(hits) == 1
        target, decay, dist, lca_id, mtype = hits[0]
        assert target == h_doid
        assert decay == 1.0
        assert dist == 0
        assert mtype == "exact"
        assert lca_id not in STOP_ROOTS


def test_doid_resolver_handles_alt_id_and_obsolete_replaced_by(resolver: DirectedDOIDResolver) -> None:
    # DOID:9917 is a Hetionet v1.0 disease that became an alt_id in newer doid.obo
    hits = resolver.resolve("DOID:9917")
    assert len(hits) == 1
    assert hits[0][0] == "DOID:9917"
    assert hits[0][1] == 1.0
    assert hits[0][4] == "exact"

    # Also test canonical active ID pointing back to DOID:9917
    canon_9917 = resolver.canonicalize("DOID:9917")
    assert canon_9917 is not None
    hits_canon = resolver.resolve(canon_9917)
    assert hits_canon[0][0] == "DOID:9917"
    assert hits_canon[0][4] == "exact"

    # Test an obsolete term WITHOUT replaced_by -> must return []
    obs_no_rep = next(
        (t for t in resolver.obsolete_terms if t not in resolver.replaced_by),
        None,
    )
    assert obs_no_rep is not None
    assert resolver.canonicalize(obs_no_rep) is None
    assert resolver.resolve(obs_no_rep) == []

    # Test an obsolete term WITH replaced_by -> must canonicalize to active term
    obs_with_rep = next(
        (t for t, r in resolver.replaced_by.items() if resolver.canonicalize(r) is not None),
        None,
    )
    if obs_with_rep is not None:
        assert resolver.canonicalize(obs_with_rep) in resolver.active_terms


def test_doid_resolver_blocks_stop_roots_and_cross_branch_leakage(resolver: DirectedDOIDResolver) -> None:
    # Direct queries on stop-roots must strictly return []
    for sr in STOP_ROOTS:
        assert resolver.resolve(sr, max_dist=10) == []

    # Pick a sample of 200 non-Hetionet active terms and verify NO result ever uses a stop-root as LCA
    checked = 0
    for tid in sorted(resolver.active_terms):
        if tid in resolver.canonical_to_hetionet or tid in STOP_ROOTS:
            continue
        hits = resolver.resolve(tid, max_dist=4, decay_base=0.90)
        for hetio_doid, decay, dist, lca_id, mtype in hits:
            assert lca_id not in STOP_ROOTS
            assert 1 <= dist <= 4
            assert decay == round(0.90 ** dist, 6)
            assert mtype in ("ancestor", "descendant", "lca")
        checked += 1
        if checked >= 200:
            break
    assert checked == 200


def test_doid_resolver_subtype_to_hetionet_ancestor_deterministic(resolver: DirectedDOIDResolver) -> None:
    # Find a direct child of a Hetionet disease (e.g., child of DOID:9352 or DOID:6000 or any Hetionet disease)
    child_pair = None
    for tid, ps in sorted(resolver.parents.items()):
        if tid in resolver.canonical_to_hetionet or tid in STOP_ROOTS:
            continue
        for p in ps:
            if p in resolver.canonical_to_hetionet and p not in STOP_ROOTS:
                child_pair = (tid, resolver.canonical_to_hetionet[p])
                break
        if child_pair:
            break

    assert child_pair is not None
    child_doid, parent_hetio = child_pair
    res1 = resolver.resolve(child_doid, max_dist=4, decay_base=0.90)
    res2 = resolver.resolve(child_doid, max_dist=4, decay_base=0.90)
    assert res1 == res2  # deterministic
    assert len(res1) >= 1
    top_hit = res1[0]
    assert top_hit[0] == parent_hetio
    assert top_hit[1] == 0.90
    assert top_hit[2] == 1
    assert top_hit[4] == "ancestor"


# ---------------------------------------------------------------------------
# 2. DeterministicCrosswalk tests
# ---------------------------------------------------------------------------
def test_crosswalk_mesh_umls_strict_policy_and_compounds(crosswalk: DeterministicCrosswalk) -> None:
    st = crosswalk.stats()
    assert st["hetionet_diseases_total"] == 137
    assert st["hetionet_compounds_total"] == 1552
    assert st["mesh_1to1"] > 3000
    assert st["umls_1to1"] > 6000
    assert st["hetionet_diseases_union_covered"] >= 130

    # Rejected 1:N codes must strictly return None (no false mapping)
    assert len(crosswalk.rejected_mesh) > 0
    for rej_mesh in list(crosswalk.rejected_mesh.keys())[:10]:
        assert crosswalk.mesh_to_doid(rej_mesh) is None

    assert len(crosswalk.rejected_umls) > 0
    for rej_umls in list(crosswalk.rejected_umls.keys())[:10]:
        assert crosswalk.umls_to_doid(rej_umls) is None

    # Compound resolution across 1,552 Hetionet compounds
    assert crosswalk.resolve_compound("dapagliflozin") == "Compound::DB06292"
    assert crosswalk.resolve_compound("DB00331") == "Compound::DB00331"
    assert crosswalk.resolve_compound("Compound::DB01394") == "Compound::DB01394"
    assert crosswalk.resolve_compound("nonexistent_fake_drug_xyz") is None


def test_crosswalk_external_ontology_hook(tmp_path: Path, resolver: DirectedDOIDResolver) -> None:
    ext_dir = tmp_path / "external_ontology"
    ext_dir.mkdir()
    h0, h1 = resolver.hetionet_diseases[0], resolver.hetionet_diseases[1]
    non_het = next(t for t in sorted(resolver.active_terms) if t not in resolver.canonical_to_hetionet)
    (ext_dir / "supplemental_mesh.tsv").write_text(
        (
            "mesh_id\tdoid\n"
            f"D9999901\t{h0}\n"
            f"D9999902\t{h0}\n"
            f"D9999902\t{h1}\n"
            f"D9999903\t{h0}\n"
            f"D9999903\t{non_het}\n"
        ),
        encoding="utf-8",
    )
    (ext_dir / "compound_aliases.json").write_text(
        json.dumps({"compound_aliases": {"farxiga_ext": "DB06292"}}),
        encoding="utf-8",
    )

    cw = DeterministicCrosswalk(resolver=resolver, external_dir=ext_dir)
    # 1:1 external MeSH accepted
    assert cw.mesh_to_doid("D9999901") == h0
    # 1:N external MeSH with 2 Hetionet diseases strictly rejected
    assert cw.mesh_to_doid("D9999902") is None
    # 1:N external MeSH with unique Hetionet 137 disease accepted
    assert cw.mesh_to_doid("D9999903") == h0
    # External compound alias loaded
    assert cw.resolve_compound("farxiga_ext") == "Compound::DB06292"
    assert "supplemental_mesh.tsv" in cw.external_files_loaded
    assert "compound_aliases.json" in cw.external_files_loaded


# ---------------------------------------------------------------------------
# 3. M2RepurposingRanker Tier + LCA Decay + Dedup + Calibration tests
# ---------------------------------------------------------------------------
def test_session_tier_weights_and_quarantine_skip(ranker: M2RepurposingRanker) -> None:
    drug_id = "Compound::DB06292"  # Dapagliflozin
    novel_dis = [
        d for d in ranker.diseases
        if (drug_id, d) not in ranker.ctd and (drug_id, d) not in ranker.cpd and d != "Disease::DOID:9352"
    ]
    d_t1, d_t2, d_t3 = novel_dis[0], novel_dis[1], novel_dis[2]

    # Compare Tier 1 vs Tier 2 vs Tier 3 on the exact same disease in isolated sessions
    s_t1 = ranker.clone_for_session()
    s_t2 = ranker.clone_for_session()
    s_t3 = ranker.clone_for_session()

    target_dis = d_t1
    dis_short = target_dis.split("::", 1)[1]

    added_t1 = s_t1.augment_session_edges([{
        "source_type": "Compound", "target_type": "Disease",
        "source_standard_id": "DB06292", "target_standard_id": dis_short,
        "relation_type": "TREATS", "confidence_score": 0.8,
        "admission_tier": "ADMITTED_KG", "pmid": "500001", "evidence": "Tier 1 trial.",
    }])
    added_t2 = s_t2.augment_session_edges([{
        "source_type": "Compound", "target_type": "Disease",
        "source_standard_id": "DB06292", "target_standard_id": dis_short,
        "relation_type": "TREATS", "confidence_score": 0.8,
        "admission_tier": "HYPOTHESIS_EDGE", "pmid": "500002", "evidence": "Tier 2 hypothesis.",
    }])
    added_t3 = s_t3.augment_session_edges([{
        "source_type": "Compound", "target_type": "Disease",
        "source_standard_id": "DB06292", "target_standard_id": dis_short,
        "relation_type": "TREATS", "confidence_score": 0.99,
        "admission_tier": "QUARANTINE", "pmid": "500003", "evidence": "Tier 3 quarantined.",
    }])

    assert added_t1 == 1
    assert added_t2 == 1
    assert added_t3 == 0
    assert (drug_id, "CtD", target_dis) not in s_t3.lit_edges
    assert pytest.approx(s_t1.lit_edges[(drug_id, "CtD", target_dis)]["w_ev"], abs=1e-6) == 0.8
    assert pytest.approx(s_t2.lit_edges[(drug_id, "CtD", target_dis)]["w_ev"], abs=1e-6) == 0.4

    r1 = {c["disease_id"]: c for c in s_t1.rank_drug_with_session(drug_id, top_n_novel=137, max_lit_delta=0.20)["top_novel_candidates"]}
    r2 = {c["disease_id"]: c for c in s_t2.rank_drug_with_session(drug_id, top_n_novel=137, max_lit_delta=0.20)["top_novel_candidates"]}
    r3 = {c["disease_id"]: c for c in s_t3.rank_drug_with_session(drug_id, top_n_novel=137, max_lit_delta=0.20)["top_novel_candidates"]}

    assert r3[target_dis]["literature_delta"] == 0.0
    assert r1[target_dis]["literature_delta"] > r2[target_dis]["literature_delta"] > 0.0
    # Zero Disk / Base Graph Mutation check
    assert not hasattr(ranker, "lit_edges") or len(getattr(ranker, "lit_edges", {})) == 0


def test_session_dedup_edge_key_pmid_takes_max_w_ev_and_indirect_scaling(ranker: M2RepurposingRanker) -> None:
    drug_id = "Compound::DB06292"
    cbg_genes = ranker.adj.get((drug_id, "CbG"), [])
    assert len(cbg_genes) > 0
    target_gene = cbg_genes[0]
    gene_short = target_gene.split("::", 1)[1]

    # Pick a novel disease not yet connected to target_gene
    chosen_dis = next(
        d for d in ranker.diseases
        if (drug_id, d) not in ranker.ctd
        and (drug_id, d) not in ranker.cpd
        and d != "Disease::DOID:9352"
        and target_gene not in ranker.adj.get((d, "DaG"), [])
    )
    dis_short = chosen_dis.split("::", 1)[1]

    # Test dedup (edge_key, pmid) taking max(w_ev)
    s_high = ranker.clone_for_session()
    s_high.augment_session_edges([
        {
            "source_type": "Gene", "target_type": "Disease",
            "source_standard_id": gene_short, "target_standard_id": dis_short,
            "relation_type": "ASSOCIATED_WITH", "confidence_score": 0.6,
            "admission_tier": "Tier 2", "lca_decay": 0.81, "pmid": "777001", "evidence": "Weak mention.",
        },
        {
            "source_type": "Gene", "target_type": "Disease",
            "source_standard_id": gene_short, "target_standard_id": dis_short,
            "relation_type": "ASSOCIATED_WITH", "confidence_score": 0.95,
            "admission_tier": "Tier 1", "lca_decay": 1.0, "pmid": "777001", "evidence": "Strong mention.",
        },
        {
            "source_type": "Gene", "target_type": "Disease",
            "source_standard_id": gene_short, "target_standard_id": dis_short,
            "relation_type": "ASSOCIATED_WITH", "confidence_score": 0.3,
            "admission_tier": "Tier 2", "lca_decay": 0.5, "pmid": "777001", "evidence": "Even weaker.",
        },
    ])
    ek = (chosen_dis, "DaG", target_gene)
    assert len(s_high.lit_edges[ek]["quotes"]) == 1
    assert pytest.approx(s_high.lit_edges[ek]["w_ev"], abs=1e-6) == 0.95

    # Compare indirect branch delta between high w_ev (0.95) and low w_ev (0.20)
    s_low = ranker.clone_for_session()
    s_low.augment_session_edges([{
        "source_type": "Gene", "target_type": "Disease",
        "source_standard_id": gene_short, "target_standard_id": dis_short,
        "relation_type": "ASSOCIATED_WITH", "confidence_score": 0.4,
        "admission_tier": "Tier 2", "lca_decay": 1.0, "pmid": "777002", "evidence": "Low w_ev.",
    }])
    assert pytest.approx(s_low.lit_edges[ek]["w_ev"], abs=1e-6) == 0.2

    res_high = {c["disease_id"]: c for c in s_high.rank_drug_with_session(drug_id, top_n_novel=137, max_lit_delta=0.20)["top_novel_candidates"]}
    res_low = {c["disease_id"]: c for c in s_low.rank_drug_with_session(drug_id, top_n_novel=137, max_lit_delta=0.20)["top_novel_candidates"]}
    assert res_high[chosen_dis]["literature_delta"] >= res_low[chosen_dis]["literature_delta"]
    assert res_high[chosen_dis]["score"] >= res_low[chosen_dis]["score"]


def test_session_auto_ontology_resolution_and_stop_root_rejection(
    ranker: M2RepurposingRanker, resolver: DirectedDOIDResolver
) -> None:
    drug_id = "Compound::DB06292"
    # Find a non-Hetionet child whose parent is a novel Hetionet disease for Dapagliflozin
    novel_het_short = {
        d.split("::", 1)[1] for d in ranker.diseases
        if (drug_id, d) not in ranker.ctd and (drug_id, d) not in ranker.cpd and d != "Disease::DOID:9352"
    }
    child_doid, parent_hetio = None, None
    for tid, ps in sorted(resolver.parents.items()):
        if tid in resolver.canonical_to_hetionet or tid in STOP_ROOTS:
            continue
        for p in ps:
            het = resolver.canonical_to_hetionet.get(p)
            if het and het in novel_het_short:
                child_doid, parent_hetio = tid, het
                break
        if child_doid:
            break
    assert child_doid is not None and parent_hetio is not None

    session = ranker.clone_for_session()
    added = session.augment_session_edges(
        [
            {
                "source_type": "Compound", "target_type": "Disease",
                "source_standard_id": "DB06292", "target_standard_id": child_doid,
                "relation_type": "TREATS", "confidence_score": 0.90,
                "admission_tier": "Tier 1", "pmid": "888001", "evidence": "Subtype clinical efficacy.",
            },
            {
                "source_type": "Compound", "target_type": "Disease",
                "source_standard_id": "DB06292", "target_standard_id": "DOID:1287",  # stop-root
                "relation_type": "TREATS", "confidence_score": 0.99,
                "admission_tier": "Tier 1", "pmid": "888002", "evidence": "Generic CV disease.",
            },
        ],
        resolve_ontology=True,
        ontology_resolver=resolver,
        max_ontology_dist=1,
        decay_base=0.90,
    )
    assert added >= 1
    expected_key = (drug_id, "CtD", f"Disease::{parent_hetio}")
    assert expected_key in session.lit_edges
    entry = session.lit_edges[expected_key]
    assert pytest.approx(entry["lca_decay"], abs=1e-6) == 0.90
    assert pytest.approx(entry["w_ev"], abs=1e-6) == 0.81
    # Stop-root DOID:1287 must NOT be injected
    assert (drug_id, "CtD", "Disease::DOID:1287") not in session.lit_edges

    # Calibration clamp check: |session_score - base_score| <= max_lit_delta
    res = session.rank_drug_with_session(drug_id, top_n_novel=137, max_lit_delta=0.05)
    for c in res["top_novel_candidates"]:
        assert abs(c["score"] - c["base_score"]) <= 0.05 + 1e-4
        assert c["score"] < 1.0
