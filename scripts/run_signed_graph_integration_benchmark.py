"""Benchmark: Integrating Signed Graph into Ranking Engine:
Tests whether Signed Graph (Signature Reversal + Directional Metapaths) increases MRR.
Compares:
  1. Baseline Debiased M2 (Unsigned, beta=0.6)
  2. M3 Signed Model (Original)
  3. Debiased M3 (M3 with beta=0.6 degree-debiasing)
  4. M2 + Explicit Net Therapeutic Ratio (NTR) Signed Flux Layer
Evaluated across:
  - Suite A: 10 Post-2018 Drug Repurposing Benchmark Candidates
  - Suite B: Rare Diseases Test Suite (Low-degree diseases in Hetionet)
  - Suite C: repoDB 636 Clinical Pairs Benchmark
"""
import json
import math
import pickle
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ASSETS_DIR = ROOT / "artifacts/ranking_m2"
BENCH_DIR = ROOT / "benchmarks/repodb"
RAW_DIR = ROOT / "data/raw"
SUITE_DIR = ROOT / "data/ground_truth"

def load_ctd_cpd(raw_dir: Path) -> Tuple[set, set]:
    import gzip
    ctd, cpd = set(), set()
    edges_gz = raw_dir / "hetionet-v1.0-edges.sif.gz"
    if edges_gz.exists():
        with gzip.open(edges_gz, "rt", encoding="utf-8") as f:
            for line in f:
                s, r, t = line.strip().split("\t")
                if r == "CtD": ctd.add((s, t))
                elif r == "CpD": cpd.add((s, t))
    return ctd, cpd

def run_signed_benchmark():
    sys.stdout.reconfigure(encoding="utf-8")
    meta = json.loads((ASSETS_DIR / "data_manifest.json").read_text(encoding="utf-8"))
    z = np.load(ASSETS_DIR / "features.npz")
    X = z["X"]
    drugs = meta["drugs"]
    diseases = meta["diseases"]
    feature_names = meta["feature_names"]
    D = len(diseases)
    N = len(drugs)

    # Load M2 and M3 bundles
    m2_bundle = pickle.loads((ASSETS_DIR / "M2_model.pkl").read_bytes())
    scale_m2, scaler_m2, clf_m2 = m2_bundle["bundle"]
    cols_m2 = list(m2_bundle["columns"])

    m3_bundle = pickle.loads((ASSETS_DIR / "M3_model.pkl").read_bytes())
    scale_m3, scaler_m3, clf_m3 = m3_bundle["bundle"]
    cols_m3 = list(m3_bundle["columns"])

    # Load blocked training pairs
    ctd, cpd = load_ctd_cpd(RAW_DIR)
    blocked_pairs = ctd | cpd
    blocked_idx = set()
    for drug_idx, d_id in enumerate(drugs):
        for dis_idx, dis_id in enumerate(diseases):
            if (d_id, dis_id) in blocked_pairs:
                blocked_idx.add(drug_idx * D + dis_idx)

    # 1. Compute Base Debiased M2 Logits
    Z_m2 = scaler_m2.transform(np.arcsinh(X[:, cols_m2] / scale_m2))
    coef_m2_deb = np.copy(clf_m2.coef_)
    coef_m2_deb[0, 1] = 0.6 * clf_m2.coef_[0, 1]  # beta=0.6 debiased
    logit_m2_deb = (Z_m2 @ coef_m2_deb.T + clf_m2.intercept_).ravel()

    # 2. Compute Original M3 Logits
    Z_m3 = scaler_m3.transform(np.arcsinh(X[:, cols_m3] / scale_m3))
    logit_m3_raw = (Z_m3 @ clf_m3.coef_.T + clf_m3.intercept_).ravel()

    # 3. Compute Debiased M3 Logits (beta=0.6)
    coef_m3_deb = np.copy(clf_m3.coef_)
    coef_m3_deb[0, 1] = 0.6 * clf_m3.coef_[0, 1]  # beta=0.6 debiased
    logit_m3_deb = (Z_m3 @ coef_m3_deb.T + clf_m3.intercept_).ravel()

    # 4. Compute Explicit Net Therapeutic Signed Score (NTR) Layer
    # Identify therapeutic reversal columns (CdG..DuG, CuG..DdG) and congruent columns (CdG..DdG, CuG..DuG)
    rev_cols_m2 = [i for i, c in enumerate(cols_m2) if ("CdG" in feature_names[c] and "DuG" in feature_names[c]) or ("CuG" in feature_names[c] and "DdG" in feature_names[c])]
    cong_cols_m2 = [i for i, c in enumerate(cols_m2) if ("CdG" in feature_names[c] and "DdG" in feature_names[c]) or ("CuG" in feature_names[c] and "DuG" in feature_names[c])]

    flux_rev = np.sum(Z_m2[:, rev_cols_m2], axis=1)
    flux_cong = np.sum(Z_m2[:, cong_cols_m2], axis=1)
    sig_reversal_feat = X[:, 65]  # column 65 is signature_reversal
    sig_available_feat = X[:, 66] # column 66 is signature_available

    ntr_score = (flux_rev - 0.5 * flux_cong) + 0.8 * sig_reversal_feat * sig_available_feat
    # Scale NTR to logit dimension
    ntr_score_norm = (ntr_score - np.mean(ntr_score)) / (np.std(ntr_score) + 1e-9)

    logit_m2_ntr = logit_m2_deb + 0.35 * ntr_score_norm

    models = {
        "1. Debiased M2 (Unsigned Baseline, beta=0.6)": logit_m2_deb,
        "2. Original M3 (Signed Graph Raw)": logit_m3_raw,
        "3. Debiased M3 (Signed Graph + beta=0.6)": logit_m3_deb,
        "4. Debiased M2 + Explicit NTR Signed Flux Layer": logit_m2_ntr
    }

    # =========================================================================
    # SUITE A: 10 POST-2018 CANDIDATES
    # =========================================================================
    suite_a = json.loads((SUITE_DIR / "prospective_temporal_test.json").read_text(encoding="utf-8"))
    print("=" * 95)
    print("SUITE A: 10 POST-2018 DRUG REPURPOSING BENCHMARK (TEMPORAL HOLDOUT)")
    print("=" * 95)
    print(f"{'MODEL / INTEGRATION REGIME':<50} | {'MRR':<8} | {'P@1 (Hit@1)':<12} | {'P@3 (Hit@3)':<12} | {'AVG RANK'}")
    print("-" * 95)

    suite_a_results = {}
    suite_a_ranks = {}

    for name, logit_arr in models.items():
        prob_arr = 1.0 / (1.0 + np.exp(-logit_arr))
        ranks = []
        for item in suite_a:
            d_idx = drugs.index(item["drug_id"])
            tgt_did = item["disease_id"]
            tgt_idx = diseases.index(tgt_did)

            row_start = d_idx * D
            drug_probs = np.copy(prob_arr[row_start : row_start + D])
            for i, did in enumerate(diseases):
                if (item["drug_id"], did) in blocked_pairs:
                    drug_probs[i] = -1.0

            order = np.argsort(-drug_probs)
            rank = int(np.where(order == tgt_idx)[0][0]) + 1
            ranks.append(rank)

        suite_a_ranks[name] = ranks
        mrr = float(np.mean([1.0 / r for r in ranks]))
        p1 = float(np.mean([1.0 if r == 1 else 0.0 for r in ranks]))
        p3 = float(np.mean([1.0 if r <= 3 else 0.0 for r in ranks]))
        avg_r = float(np.mean(ranks))

        suite_a_results[name] = {"mrr": round(mrr, 4), "p@1": round(p1, 4), "p@3": round(p3, 4), "avg_rank": round(avg_r, 2)}
        print(f"{name:<50} | {mrr:<8.4f} | {p1:<12.1%} | {p3:<12.1%} | Rank {avg_r:<5.1f}")

    # Drug by drug breakdown
    print("\nDetailed drug ranks across models:")
    print(f"{'DRUG':<15} | {'TARGET':<22} | {'Debiased M2':<12} | {'Orig M3':<10} | {'Debiased M3':<12} | {'M2 + NTR Flux'}")
    print("-" * 85)
    for i, item in enumerate(suite_a):
        dname = item["drug_name"]
        tname = item["disease_name"]
        r1 = suite_a_ranks["1. Debiased M2 (Unsigned Baseline, beta=0.6)"][i]
        r2 = suite_a_ranks["2. Original M3 (Signed Graph Raw)"][i]
        r3 = suite_a_ranks["3. Debiased M3 (Signed Graph + beta=0.6)"][i]
        r4 = suite_a_ranks["4. Debiased M2 + Explicit NTR Signed Flux Layer"][i]
        print(f"{dname:<15} | {tname[:21]:<22} | Rank {r1:<10} | Rank {r2:<8} | Rank {r3:<10} | Rank {r4}")

    # =========================================================================
    # SUITE B: RARE DISEASES (DEGREE <= 30)
    # =========================================================================
    suite_b = json.loads((SUITE_DIR / "rare_disease_test.json").read_text(encoding="utf-8"))
    print("\n" + "=" * 95)
    print("SUITE B: RARE / LOW-DEGREE DISEASES BENCHMARK (11 CLINICAL PAIRS, DEGREE <= 30)")
    print("=" * 95)
    print(f"{'MODEL / INTEGRATION REGIME':<50} | {'MRR':<8} | {'HIT@5':<10} | {'HIT@10':<10} | {'AVG RANK'}")
    print("-" * 95)

    suite_b_results = {}
    for name, logit_arr in models.items():
        prob_arr = 1.0 / (1.0 + np.exp(-logit_arr))
        ranks = []
        for item in suite_b:
            d_idx = drugs.index(item["drug_id"])
            tgt_did = item["disease_id"]
            tgt_idx = diseases.index(tgt_did)

            row_start = d_idx * D
            drug_probs = np.copy(prob_arr[row_start : row_start + D])
            for i, did in enumerate(diseases):
                if (item["drug_id"], did) in blocked_pairs:
                    drug_probs[i] = -1.0

            order = np.argsort(-drug_probs)
            rank = int(np.where(order == tgt_idx)[0][0]) + 1
            ranks.append(rank)

        mrr_b = float(np.mean([1.0 / r for r in ranks]))
        h5_b = float(np.mean([1.0 if r <= 5 else 0.0 for r in ranks]))
        h10_b = float(np.mean([1.0 if r <= 10 else 0.0 for r in ranks]))
        avg_r_b = float(np.mean(ranks))

        suite_b_results[name] = {"mrr": round(mrr_b, 4), "hit@5": round(h5_b, 4), "hit@10": round(h10_b, 4), "avg_rank": round(avg_r_b, 2)}
        print(f"{name:<50} | {mrr_b:<8.4f} | {h5_b:<10.1%} | {h10_b:<10.1%} | Rank {avg_r_b:<5.1f}")

    # =========================================================================
    # SUITE C: REPODB (636 CLINICAL PAIRS)
    # =========================================================================
    print("\n" + "=" * 95)
    print("SUITE C: REPODB (636 CLINICAL PAIRS) GENERAL BENCHMARK")
    print("=" * 95)
    print(f"{'MODEL / INTEGRATION REGIME':<50} | {'MRR':<8} | {'P@5':<8} | {'P@10':<8} | {'R@10':<8} | {'NDCG@10'}")
    print("-" * 95)

    base_run = BENCH_DIR / "runs/20261103T074345Z"
    gold = json.loads((base_run / "gold_pairs.json").read_text(encoding="utf-8"))
    gold_by_drug = defaultdict(list)
    for r in gold:
        gold_by_drug[r["row"] // D].append((r["row"], r["label"]))
    all_drugs = sorted(list(set(gold_by_drug)))

    suite_c_results = {}
    for name, logit_arr in models.items():
        prob_arr = 1.0 / (1.0 + np.exp(-logit_arr))
        sc_mat = prob_arr.reshape(-1, D)

        per = []
        for drug in all_drugs:
            positives = {r % D for r, y in gold_by_drug.get(drug, []) if y and r not in blocked_idx}
            if not positives:
                continue
            candidates = [d for d in range(D) if drug * D + d not in blocked_idx]
            ordered = sorted(candidates, key=lambda d: (-sc_mat[drug, d], d))
            ranks = {d: i + 1 for i, d in enumerate(ordered)}

            mrr = 1.0 / min(ranks[d] for d in positives)
            p5 = len(set(ordered[:5]) & positives) / 5.0
            p10 = len(set(ordered[:10]) & positives) / 10.0
            r10 = len(set(ordered[:10]) & positives) / len(positives)
            dcg = sum(1.0 / np.log2(i + 2) for i, d in enumerate(ordered[:10]) if d in positives)
            ideal = sum(1.0 / np.log2(i + 2) for i in range(min(10, len(positives))))
            ndcg10 = float(dcg / ideal) if ideal > 0 else 0.0

            per.append({"mrr": mrr, "p5": p5, "p10": p10, "r10": r10, "ndcg": ndcg10})

        mrr_c = float(np.mean([p["mrr"] for p in per]))
        p5_c = float(np.mean([p["p5"] for p in per]))
        p10_c = float(np.mean([p["p10"] for p in per]))
        r10_c = float(np.mean([p["r10"] for p in per]))
        ndcg_c = float(np.mean([p["ndcg"] for p in per]))

        suite_c_results[name] = {"mrr": round(mrr_c, 4), "p5": round(p5_c, 4), "p10": round(p10_c, 4), "r10": round(r10_c, 4), "ndcg": round(ndcg_c, 4)}
        print(f"{name:<50} | {mrr_c:<8.4f} | {p5_c:<8.2%} | {p10_c:<8.2%} | {r10_c:<8.2%} | {ndcg_c:.4f}")

    # Save all results to artifacts
    out_file = ROOT / "artifacts/benchmarks/signed_graph_integration_benchmark_report.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps({
        "suite_a_post2018": suite_a_results,
        "suite_b_rare_diseases": suite_b_results,
        "suite_c_repodb_636": suite_c_results
    }, indent=2), encoding="utf-8")
    print(f"\nAll signed graph integration results saved successfully to: {out_file}")

if __name__ == "__main__":
    run_signed_benchmark()
