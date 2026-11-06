#!/usr/bin/env python3
"""
Official LBD M2 DWPC Drug Repurposing Ranking & Micro-DFS CLI
-------------------------------------------------------------
Runs the packaged M2 Supervised Graph-Topological Ranker inside dapagliflozin-lbd.

Usage:
    python scripts/run_lbd_m2_ranking.py --drugs "Dapagliflozin,Colchicine,Aspirin,Ruxolitinib"
    python scripts/run_lbd_m2_ranking.py --drugs "Dapagliflozin" --top-k 5
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))

from lbd.ranking.m2_ranker import M2RepurposingRanker  # noqa: E402


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(
        description="Run M2 DWPC Repurposing Ranker & Meet-in-the-Middle Micro-DFS"
    )
    parser.add_argument(
        "--drugs",
        type=str,
        default="Dapagliflozin,Colchicine,Aspirin,Ruxolitinib",
        help="Comma-separated list of drug names or DrugBank IDs",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of top novel disease candidates to display per drug",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Optional output JSON file path",
    )
    args = parser.parse_args()

    t0 = time.perf_counter()
    print("=" * 80)
    print("  LBD M2 SUPERVISED DWPC RANKING & MICRO-DFS EXPLANATION PIPELINE")
    print("  Knowledge Graph: Hetionet v1.0 (47,031 nodes | 2,250,197 edges | 137 diseases)")
    print("=" * 80)

    ranker = M2RepurposingRanker()
    drug_queries = [d.strip() for d in args.drugs.split(",") if d.strip()]
    all_outputs = {}

    for query in drug_queries:
        res = ranker.rank_drug(query, top_n_novel=args.top_k, include_dfs_paths=True, dfs_top_k=2)
        all_outputs[res["drug_id"]] = res

        print(f"\n>>> THUỐC: {res['drug_name']} ({res['drug_id']})")
        print(f"    Chỉ định gốc đã biết trong Hetionet (CtD/CpD): {res['known_indications_count']}")
        for ki in res["known_indications"]:
            print(
                f"      - [Hạng thô #{ki['rank_raw']:>3}/137] {ki['disease_name']} "
                f"({ki['disease_id']}) | P = {ki['score']:.4f} | {ki['status']}"
            )

        print(f"    Top {args.top_k} Ứng viên Tái định vị Mới (Novel Candidates):")
        for nc in res["top_novel_candidates"]:
            print(
                f"      [Novel #{nc['rank_novel']} | Raw #{nc['rank_raw']:>2}/137] "
                f"{nc['disease_name']} ({nc['disease_id']}) | "
                f"P(M2) = {nc['score']:.4f} | DFS Paths = {nc.get('total_dfs_paths', 0):,}"
            )
            for p in nc.get("top_paths", []):
                print(
                    f"         * [Beta={p['beta']:+.4f}, Contrib={p['contrib']:+.6f}] {p['path_str']}"
                )

    elapsed = time.perf_counter() - t0
    print("\n" + "=" * 80)
    print(f"  Hoàn tất xếp hạng & giải thích DFS cho {len(drug_queries)} thuốc trong {elapsed:.2f}s.")
    print("=" * 80)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(all_outputs, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Đã lưu kết quả JSON tại: {out_path}")


if __name__ == "__main__":
    main()
