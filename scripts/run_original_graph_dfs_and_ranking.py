#!/usr/bin/env python3
"""[LEGACY SPRINT 3 RUNNER] Run the historical heuristic DFS and ranking code on the 721-node snapshot.

Notice:
    This script is preserved for historical baseline audit and regression testing.
    For the official production M2 topological ranking on Hetionet v1.0, use:
    `python scripts/run_lbd_m2_ranking.py --drugs "Dapagliflozin" --top-k 5`
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import networkx as nx
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lbd.graph.path_search import search_metapaths  # noqa: E402
from lbd.ranking.rank_candidates import score_disease_candidate  # noqa: E402
from lbd.ranking.score_paths import score_individual_path  # noqa: E402

GRAPH_PATH = ROOT / "data" / "processed" / "merged_graph.json"
DFS_OUT = ROOT / "artifacts" / "original_graph_dfs_report.json"
RANK_OUT = ROOT / "artifacts" / "original_graph_ranking_report.json"
DRUG_ID = "DB06292"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def node_route(edges: list[dict[str, Any]]) -> list[str]:
    return [edges[0]["source"], *[edge["target"] for edge in edges]] if edges else []


def matching_metapath(edges: list[dict[str, Any]], G: nx.MultiDiGraph,
                      allowed: list[list[str]]) -> bool:
    route = node_route(edges)
    types = [str(G.nodes[node].get("type", "Unknown")) for node in route]
    return types in allowed


def serialize_path(edges: list[dict[str, Any]], G: nx.MultiDiGraph,
                   allowed: list[list[str]]) -> dict[str, Any]:
    route = node_route(edges)
    types = [str(G.nodes[node].get("type", "Unknown")) for node in route]
    return {
        "nodes": route,
        "node_names": [str(G.nodes[node].get("canonical_name", node)) for node in route],
        "node_types": types,
        "metapath_config_match": types in allowed,
        "edges": edges,
    }


def rank_candidates(paths_by_disease: dict[str, list[list[dict[str, Any]]]],
                    G: nx.MultiDiGraph, top_m: int, lam: float,
                    allowed: list[list[str]] | None = None) -> list[dict[str, Any]]:
    candidates = []
    for disease_id, raw_paths in paths_by_disease.items():
        selected = [p for p in raw_paths if allowed is None or matching_metapath(p, G, allowed)]
        if not selected:
            continue
        scored_paths = []
        contradicting_pmids: set[str] = set()
        signs = defaultdict(int)
        for path in selected:
            route = node_route(path)
            hub_degrees = [int(G.degree(node)) for node in route[1:-1]]
            metrics = score_individual_path(path, hub_degrees)
            signs[metrics["net_sign"]] += 1
            if metrics["net_sign"] == 1:
                contradicting_pmids.update(str(pmid) for edge in path for pmid in edge.get("pmids", []) if pmid)
            scored_paths.append({
                **serialize_path(path, G, [] if allowed is None else allowed),
                **metrics,
                "supporting_pmids": sorted({str(pmid) for edge in path for pmid in edge.get("pmids", []) if pmid}),
                "evidence_sentences": [str(sentence) for edge in path
                                       for sentence in edge.get("evidence_sentences", []) if sentence],
            })
        disease = G.nodes[disease_id]
        scored = score_disease_candidate(
            disease_id=disease_id,
            disease_name=str(disease.get("canonical_name", disease_id)),
            scored_paths=scored_paths,
            contradicting_pmids=sorted(contradicting_pmids),
            top_m=top_m,
            lam=lam,
        )
        candidates.append({
            "disease_id": disease_id,
            "disease_name": str(disease.get("canonical_name", disease_id)),
            "ranking_score": scored["ranking_score"],
            "total_paths": len(selected),
            "therapeutic_paths_net_sign_minus_1": signs[-1],
            "neutral_or_association_paths_net_sign_0": signs[0],
            "contradictory_paths_net_sign_plus_1": signs[1],
            "contradicting_pmids": sorted(contradicting_pmids),
            "top_paths": scored["paths"],
        })
    candidates.sort(key=lambda item: (-item["ranking_score"], item["disease_id"]))
    for rank, item in enumerate(candidates, start=1):
        item["rank"] = rank
    return candidates


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    started = time.perf_counter()
    config = yaml.safe_load((ROOT / "configs" / "ranking.yaml").read_text(encoding="utf-8"))
    search_cfg = config["search"]
    score_cfg = config["scoring"]
    graph_data = json.loads(GRAPH_PATH.read_text(encoding="utf-8"))
    G = nx.MultiDiGraph()
    for node in graph_data["nodes"]:
        attrs = dict(node)
        node_id = str(attrs.pop("id"))
        G.add_node(node_id, **attrs)
    for edge in graph_data["edges"]:
        attrs = dict(edge)
        source, target = str(attrs.pop("source")), str(attrs.pop("target"))
        G.add_edge(source, target, **attrs)

    if DRUG_ID not in G:
        raise ValueError(f"Drug node {DRUG_ID} is missing from {GRAPH_PATH}")
    disease_ids = sorted(n for n, attrs in G.nodes(data=True) if attrs.get("type") == "Disease")
    max_nodes = int(search_cfg["max_length"])
    reachable = nx.single_source_shortest_path_length(G, DRUG_ID, cutoff=max_nodes - 1)
    reachable_diseases = [d for d in disease_ids if d in reachable and d != DRUG_ID]
    allowed = search_cfg["allowed_metapaths"]
    by_disease: dict[str, list[list[dict[str, Any]]]] = {}
    path_rows = []
    for disease_id in reachable_diseases:
        paths = search_metapaths(
            G, DRUG_ID, disease_id,
            min_length=int(search_cfg["min_length"]),
            max_length=max_nodes,
            filter_historical=True,
        )
        if paths:
            by_disease[disease_id] = paths
            for index, path in enumerate(paths, start=1):
                row = serialize_path(path, G, allowed)
                row.update({"disease_id": disease_id,
                            "disease_name": str(G.nodes[disease_id].get("canonical_name", disease_id)),
                            "path_index_for_disease": index,
                            "edge_count": len(path),
                            "node_count": len(path) + 1})
                path_rows.append(row)

    raw_ranking = rank_candidates(
        by_disease, G, int(score_cfg["top_m_paths"]),
        float(score_cfg["contradiction_penalty_lambda"]),
    )
    configured_ranking = rank_candidates(
        by_disease, G, int(score_cfg["top_m_paths"]),
        float(score_cfg["contradiction_penalty_lambda"]), allowed,
    )
    graph_sha = sha256(GRAPH_PATH)
    runtime = round(time.perf_counter() - started, 3)
    dfs_report = {
        "report_type": "raw_metapath_dfs_on_original_graph",
        "created_epoch_seconds": time.time(),
        "input_graph": str(GRAPH_PATH),
        "input_graph_sha256": graph_sha,
        "graph_metadata": graph_data.get("metadata", {}),
        "loaded_graph_counts": {"nodes": G.number_of_nodes(), "edges": G.number_of_edges(),
                                "disease_nodes": len(disease_ids)},
        "source": {"id": DRUG_ID, "name": G.nodes[DRUG_ID].get("canonical_name"),
                   "type": G.nodes[DRUG_ID].get("type")},
        "search_parameters": {**search_cfg, "note": "The current DFS code interprets min_length/max_length as node count and does not enforce allowed_metapaths; each path is labeled with a config-match flag."},
        "targets_within_reach_cutoff": len(reachable_diseases),
        "disease_targets_with_at_least_one_path": len(by_disease),
        "total_paths": len(path_rows),
        "paths_matching_declared_metapaths": sum(bool(p["metapath_config_match"]) for p in path_rows),
        "paths_not_matching_declared_metapaths": sum(not p["metapath_config_match"] for p in path_rows),
        "runtime_seconds": runtime,
        "paths": path_rows,
    }
    DFS_OUT.write_text(json.dumps(dfs_report, indent=2, ensure_ascii=False), encoding="utf-8")

    rank_report = {
        "report_type": "ranking_on_original_graph",
        "created_epoch_seconds": dfs_report["created_epoch_seconds"],
        "input_graph": str(GRAPH_PATH),
        "input_graph_sha256": graph_sha,
        "graph_status": graph_data.get("metadata", {}).get("version"),
        "source": dfs_report["source"],
        "scoring_parameters": score_cfg,
        "runtime_seconds_including_dfs": runtime,
        "candidate_count_raw_current_dfs": len(raw_ranking),
        "top_100_raw_current_dfs": raw_ranking[:100],
        "all_raw_current_dfs": raw_ranking,
        "candidate_count_declared_metapaths_only": len(configured_ranking),
        "top_100_declared_metapaths_only": configured_ranking[:100],
        "all_declared_metapaths_only": configured_ranking,
        "interpretation": [
            "Raw ranking follows the current search_metapaths implementation and score functions.",
            "Declared-metapath ranking additionally filters paths against configs/ranking.yaml; this is a diagnostic because the production DFS currently ignores allowed_metapaths.",
            "This is a structural graph ranking, not a validated clinical prediction. Hetionet edges have no PMID/evidence sentences in this baseline.",
        ],
    }
    RANK_OUT.write_text(json.dumps(rank_report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({
        "graph": graph_data.get("metadata"),
        "dfs_path_count": len(path_rows),
        "paths_matching_configured_metapaths": dfs_report["paths_matching_declared_metapaths"],
        "candidate_count_raw": len(raw_ranking),
        "candidate_count_configured": len(configured_ranking),
        "top_10_raw": [{"rank": x["rank"], "disease": x["disease_name"], "id": x["disease_id"],
                         "score": x["ranking_score"], "paths": x["total_paths"]} for x in raw_ranking[:10]],
        "top_10_configured": [{"rank": x["rank"], "disease": x["disease_name"], "id": x["disease_id"],
                               "score": x["ranking_score"], "paths": x["total_paths"]} for x in configured_ranking[:10]],
        "dfs_report": str(DFS_OUT),
        "ranking_report": str(RANK_OUT),
        "runtime_seconds": runtime,
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
