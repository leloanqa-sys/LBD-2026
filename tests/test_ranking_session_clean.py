"""Unit tests for P0/P1 fixes in m2_ranker.py and fail-closed run_explorer.py."""
from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

LBD_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LBD_ROOT / "src"))

from lbd.ranking.m2_ranker import DRUG_ALIASES, M2RepurposingRanker


@pytest.fixture(scope="module")
def ranker() -> M2RepurposingRanker:
    return M2RepurposingRanker(load_graph_for_dfs=True)


def test_duloxetine_id_and_no_fake_proxies(ranker: M2RepurposingRanker) -> None:
    """Duloxetine must resolve to DB00476 (not Montelukast DB00471) and fake proxies must be removed."""
    assert ranker.resolve_drug_id("duloxetine") == "Compound::DB00476"
    assert ranker.names.get("Compound::DB00476", "").lower() == "duloxetine"
    assert ranker.names.get("Compound::DB00471", "").lower() == "montelukast"
    assert "liraglutide_proxy" not in DRUG_ALIASES
    assert "baricitinib_proxy" not in DRUG_ALIASES


def test_gene_to_disease_nlp_edge_orientation_and_dfs_visibility(ranker: M2RepurposingRanker) -> None:
    """NLP Gene -> Disease edges must be normalized to Hetionet (Disease, dm) -> Gene and visible to DFS."""
    session = ranker.clone_for_session()
    drug_id = "Compound::DB06292"  # Dapagliflozin
    cbg_genes = session.adj.get((drug_id, "CbG"), [])
    assert len(cbg_genes) > 0
    target_gene = cbg_genes[0]

    # Pick a disease not currently connected to target_gene via DaG/DuG/DdG
    chosen_disease = None
    for d in session.diseases:
        if (drug_id, d) in session.ctd or (drug_id, d) in session.cpd:
            continue
        if target_gene not in session.adj.get((d, "DaG"), []) and target_gene not in session.adj.get((d, "DuG"), []) and target_gene not in session.adj.get((d, "DdG"), []):
            chosen_disease = d
            break
    assert chosen_disease is not None

    gene_short = target_gene.split("::", 1)[1]
    dis_short = chosen_disease.split("::", 1)[1]
    added = session.augment_session_edges([
        {
            "source_type": "Gene",
            "target_type": "Disease",
            "source_standard_id": gene_short,
            "target_standard_id": dis_short,
            "relation_type": "ASSOCIATED_WITH",
            "confidence_score": 0.9,
            "pmid": "99999901",
            "evidence": "Target gene associated with disease in cohort.",
        }
    ])
    assert added == 1
    assert target_gene in session.adj[(chosen_disease, "DaG")]
    assert chosen_disease in session.adj[(target_gene, "DaG_rev")]

    _, _, lit_paths = session.explain_pair_dfs(drug_id, chosen_disease, top_k=5, return_all_lit=True)
    direct_3node = [p for p in lit_paths if p["nodes"] == [drug_id, target_gene, chosen_disease]]
    assert len(direct_3node) >= 1
    assert "99999901" in direct_3node[0]["supporting_pmids"]


def test_no_default_cpd_coercion_neutral_maps_to_cad_zero_delta(ranker: M2RepurposingRanker) -> None:
    """Neutral/unclear Compound->Disease relations must map to CaD with delta_logit = 0.0 (no CpD coercion)."""
    session = ranker.clone_for_session()
    drug_id = "Compound::DB06292"
    novel_diseases = [
        d for d in session.diseases
        if (drug_id, d) not in session.ctd and (drug_id, d) not in session.cpd and d != "Disease::DOID:9352"
    ]
    d_assoc, d_cohort, d_treat, d_pall, d_adverse = novel_diseases[:5]

    session.augment_session_edges([
        {
            "source_type": "Compound", "target_type": "Disease",
            "source_standard_id": "DB06292", "target_standard_id": d_assoc.split("::", 1)[1],
            "relation_type": "ASSOCIATED_WITH", "confidence_score": 0.95, "pmid": "100001", "evidence": "Associated.",
        },
        {
            "source_type": "Compound", "target_type": "Disease",
            "source_standard_id": "DB06292", "target_standard_id": d_cohort.split("::", 1)[1],
            "relation_type": "EVALUATED_IN_COHORT", "confidence_score": 0.95, "pmid": "100002", "evidence": "Evaluated.",
        },
        {
            "source_type": "Compound", "target_type": "Disease",
            "source_standard_id": "DB06292", "target_standard_id": d_treat.split("::", 1)[1],
            "relation_type": "TREATS", "confidence_score": 0.95, "pmid": "100003", "evidence": "Treats.",
        },
        {
            "source_type": "Compound", "target_type": "Disease",
            "source_standard_id": "DB06292", "target_standard_id": d_pall.split("::", 1)[1],
            "relation_type": "PALLIATES", "confidence_score": 0.95, "pmid": "100004", "evidence": "Relieves symptoms.",
        },
        {
            "source_type": "Compound", "target_type": "Disease",
            "source_standard_id": "DB06292", "target_standard_id": d_adverse.split("::", 1)[1],
            "relation_type": "CAUSES_ADVERSE_EFFECT", "confidence_score": 0.95, "pmid": "100005", "evidence": "Adverse toxicity.",
        },
    ])

    assert (drug_id, "CaD", d_assoc) in session.lit_edges
    assert (drug_id, "CpD", d_assoc) not in session.lit_edges
    assert (drug_id, "CaD", d_cohort) in session.lit_edges
    assert (drug_id, "CtD", d_treat) in session.lit_edges
    assert (drug_id, "CpD", d_pall) in session.lit_edges
    assert (drug_id, "CcSE", d_adverse) in session.lit_edges

    res = session.rank_drug_with_session(drug_id, top_n_novel=len(session.diseases), max_lit_delta=0.05)
    by_dis = {c["disease_id"]: c for c in res["top_novel_candidates"]}
    assert by_dis[d_assoc]["literature_delta"] == 0.0
    assert by_dis[d_cohort]["literature_delta"] == 0.0
    assert by_dis[d_treat]["literature_delta"] > 0.0
    assert by_dis[d_pall]["literature_delta"] > 0.0
    assert by_dis[d_adverse]["literature_delta"] <= 0.0


def test_no_score_saturation_dedup_and_max_lit_delta_enforced(ranker: M2RepurposingRanker) -> None:
    """Bounded tanh + UniqueLitEdgesUsed + (edge_key, pmid) dedup + max_lit_delta => 0% saturation at 1.0000."""
    session = ranker.clone_for_session()
    drug_id = "Compound::DB00331"  # Metformin

    # Inject a high-degree hub gene and 30 duplicate (edge_key, pmid) copies + direct CtD edges
    hub_gene = max(
        (g for (g, m), nbrs in session.adj.items() if m == "GpBP" and len(nbrs) > 20),
        key=lambda g: len(session.adj[(g, "GpBP")]),
    )
    hub_short = hub_gene.split("::", 1)[1]
    edges = []
    for _ in range(30):
        edges.append({
            "source_type": "Compound", "target_type": "Gene",
            "source_standard_id": "DB00331", "target_standard_id": hub_short,
            "relation_type": "INHIBITS", "confidence_score": 0.99,
            "pmid": "88888888", "evidence": "Duplicate mention of hub gene.",
        })
    for d in session.diseases[:10]:
        edges.append({
            "source_type": "Compound", "target_type": "Disease",
            "source_standard_id": "DB00331", "target_standard_id": d.split("::", 1)[1],
            "relation_type": "TREATS", "confidence_score": 0.99,
            "pmid": "77777777", "evidence": "Direct treat mention.",
        })
    session.augment_session_edges(edges)

    # Verify (edge_key, pmid) deduplication
    hub_entry = session.lit_edges[(drug_id, "CdG", hub_gene)]
    assert len(hub_entry["quotes"]) == 1

    max_delta = 0.05
    res = session.rank_drug_with_session(drug_id, top_n_novel=len(session.diseases), max_lit_delta=max_delta)
    cands = res["top_novel_candidates"]
    assert len(cands) > 0
    saturated = [c for c in cands if c["score"] >= 1.0]
    assert len(saturated) == 0, f"Found saturated scores at 1.0000: {saturated[:3]}"
    for c in cands:
        assert abs(c["score"] - c["base_score"]) <= max_delta + 1e-4


def test_two_tier_signed_kg_labels_transcriptional_vs_clinical(ranker: M2RepurposingRanker) -> None:
    """CdG/CuG + DuG/DdG paths must be labeled transcriptional_reversal/congruence, never clinical therapeutic/pathogenic."""
    session = ranker.clone_for_session()
    drug_id = "Compound::DB00331"  # Metformin (has CdG/CuG edges)
    cdg_genes = session.adj.get((drug_id, "CdG"), [])
    assert len(cdg_genes) > 0
    g = cdg_genes[0]

    # Pick a novel disease and add DuG & DdG to test both reversal and congruence
    novel_dis = next(d for d in session.diseases if (drug_id, d) not in session.ctd and (drug_id, d) not in session.cpd)
    session.augment_session_edges([
        {
            "source_type": "Gene", "target_type": "Disease",
            "source_standard_id": g.split("::", 1)[1], "target_standard_id": novel_dis.split("::", 1)[1],
            "relation_type": "UPREGULATED_IN", "pmid": "600001", "evidence": "Upregulated.",
        },
        {
            "source_type": "Compound", "target_type": "Disease",
            "source_standard_id": "DB00331", "target_standard_id": novel_dis.split("::", 1)[1],
            "relation_type": "TREATS", "pmid": "600002", "evidence": "Clinical treatment.",
        },
    ])
    _, paths, _ = session.explain_pair_dfs(drug_id, novel_dis, top_k=500, return_all_lit=True)
    tx_paths = [
        p for p in paths
        if ("CdG" in p["metapath"] or "CuG" in p["metapath"])
        and ("DuG" in p["metapath"] or "DdG" in p["metapath"])
    ]
    assert len(tx_paths) > 0
    for p in tx_paths:
        assert p["mechanism_type"] in ("transcriptional_reversal", "transcriptional_congruence")
        assert p["mechanism_type"] not in ("therapeutic", "pathogenic")

    direct_ctd = [p for p in paths if p["metapath"] == "literature:CtD"]
    assert len(direct_ctd) == 1
    assert direct_ctd[0]["mechanism_type"] == "therapeutic"


def test_run_explorer_fail_closed_no_review_bypass() -> None:
    """run_explorer.py must NEVER feed relations_review.jsonl into session_ranker.augment_session_edges()."""
    explorer_path = LBD_ROOT / "scripts" / "run_explorer.py"
    src = explorer_path.read_text(encoding="utf-8")
    tree = ast.parse(src)

    # Inspect the block around augment_session_edges
    lines_src = src.splitlines()
    aug_lines = [i for i, line in enumerate(lines_src) if "augment_session_edges" in line]
    assert len(aug_lines) == 1
    idx = aug_lines[0]
    window = "\n".join(lines_src[max(0, idx - 25) : idx + 5])
    assert "relations_review.jsonl" not in window
    assert "ADMITTED_KG" in window
    assert "literature_boost_applied" in src
    assert "admitted_kg_edges" in src
    assert isinstance(tree, ast.Module)
