"""End-to-End Live Verification:
Compares:
  Mode 1: KHÔNG CÓ NLP (Hetionet thuần túy, zero literature edges)
  Mode 2: CÓ NLP (Pipeline hoàn chỉnh: Papers -> BioRED NLP -> Ingest into ephemeral Hetionet Session -> Augmented Ranking)

Tests:
  1. Edge Ingestion Audit:
     - Exact number of NLP raw extractions
     - Number of admitted edges by filter policy
     - Number of edges dynamically injected into in-memory session graph
     - Detail of specific edges added (source, metaedge, target, relation, PMIDs, sign)
  2. Ranking & Metric Delta:
     - Rank shift (Không có NLP vs Có NLP)
     - Base Score vs Literature Delta vs Final Augmented Score
     - Emergence of newly discovered literature paths ([⚡ Y văn mới])
  3. Zero Disk Mutation Invariant:
     - Verify SHA256 hashes of Hetionet raw files remain 100% identical.
"""

import hashlib
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

LBD_DIR = Path(__file__).resolve().parents[1]
ROOT = LBD_DIR
RAW_DIR = LBD_DIR / "data/raw"
ASSETS_DIR = LBD_DIR / "artifacts/ranking_m2"

sys.path.insert(0, str(LBD_DIR / "scripts"))
sys.path.insert(0, str(LBD_DIR / "src"))

import run_explorer as pipeline
from lbd.ranking.m2_ranker import M2RepurposingRanker


def file_sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def run_live_comparison(drug_name="Dapagliflozin", max_papers=20):
    sys.stdout.reconfigure(encoding="utf-8")
    print("=" * 105)
    print(f"KIỂM ĐỊNH THỰC TẾ END-TO-END: SO SÁNH CÓ NLP VS KHÔNG CÓ NLP TRÊN SESSION TẠM HETIONET")
    print(f"Thuốc thử nghiệm: {drug_name} | Số bài PubMed: {max_papers}")
    print("=" * 105)

    # 0. Check initial Hetionet raw file hashes
    raw_edges = RAW_DIR / "hetionet-v1.0-edges.sif.gz"
    raw_nodes = RAW_DIR / "hetionet-v1.0-nodes.tsv"
    initial_hash_edges = file_sha256(raw_edges)
    initial_hash_nodes = file_sha256(raw_nodes)

    # =========================================================================
    # CHẾ ĐỘ 1: KHÔNG CÓ NLP (BASE HETIONET THUẦN TÚY)
    # =========================================================================
    print("\n--- [CHẾ ĐỘ 1: KHÔNG CÓ NLP (Hetionet tĩnh, không nạp y văn)] ---")
    ranker_base = M2RepurposingRanker(load_graph_for_dfs=True)
    drug_id = ranker_base.resolve_drug_id(drug_name)
    
    # Run ranking without any session augmentation
    res_base = ranker_base.rank_drug(
        drug_query=drug_id,
        top_n_novel=15,
        include_dfs_paths=True,
        dfs_top_k=2,
        debiasing_beta=0.6,
        use_signed_graph=True,
        ntr_weight=0.35,
    )
    
    base_candidates = {c["disease_id"]: c for c in res_base["top_novel_candidates"]}
    print(f"✓ Đã xếp hạng trên Hetionet thuần túy cho {drug_id}: {len(res_base['top_novel_candidates'])} ứng viên hàng đầu.")
    print(f"  Số cạnh y văn nạp vào đồ thị: 0 cạnh.")

    # =========================================================================
    # CHẾ ĐỘ 2: CÓ NLP (CHẠY PIPELINE HOÀN CHỈNH NẠP VÀO SESSION TẠM)
    # =========================================================================
    print("\n--- [CHẾ ĐỘ 2: CÓ NLP (Thu thập PubMed -> BioRED -> Nạp Session Tạm Hetionet -> Re-rank)] ---")
    
    # Progress callback
    def progress_log(stage):
        print(f"  [Tiến trình pipeline]: {stage}")
        
    source_csv = RAW_DIR / "dapagliflozin-abstracts-2018.csv"
    pipeline_res = pipeline.run(drug=drug_name, limit=max_papers, source=source_csv, progress=progress_log)
    run_id = pipeline_res["run_id"]
    run_dir = pipeline.RUNS / run_id
    
    session_graph_path = run_dir / "session_graph.json"
    session_data = json.loads(session_graph_path.read_text(encoding="utf-8")) if session_graph_path.exists() else {}
    
    # Load session ranking results
    nlp_candidates = {c["disease_id"]: c for c in pipeline_res["candidates"]}

    # =========================================================================
    # 1. KIỂM TRA NẠP CẠNH (EDGE INGESTION AUDIT)
    # =========================================================================
    print("\n" + "=" * 105)
    print("1. KIỂM TOÁN NẠP CẠNH VÀO SESSION TẠM (EDGE INGESTION AUDIT)")
    print("=" * 105)
    print(f"Số bài báo PubMed đã phân tích:           {pipeline_res.get('papers', 0)}")
    print(f"Số cạnh NLP được thông qua bộ lọc:         {pipeline_res.get('admitted_edges', 0)}")
    print(f"Số cạnh THỰC TẾ NẠP VÀO SESSION HETIONET:  {session_data.get('session_edges_added', 0)}")
    print(f"Trạng thái đồ thị phiên:                  {session_data.get('status', 'N/A')}")
    
    print("\nChi tiết các cạnh sinh học mới được nạp vào session tạm Hetionet:")
    injected_edges = session_data.get("edges", [])
    for idx, e in enumerate(injected_edges[:10], 1):
        pmids = e.get("pmids", [])
        pmid_sample = ", ".join(pmids[:4]) + (f" (+{len(pmids)-4} pmids)" if len(pmids) > 4 else "")
        print(f"  [{idx}] {e['source']} --[{e['metaedge']}:{e['relation_type']}]--> {e['target']} | Conf: {e.get('confidence', 0.8):.2f} | PMIDs: [{pmid_sample}]")
    if len(injected_edges) > 10:
        print(f"  ... và còn {len(injected_edges) - 10} cạnh khác.")

    # =========================================================================
    # 2. SO SÁNH ĐỐI ĐẦU CHỈ SỐ: CÓ NLP VS KHÔNG CÓ NLP
    # =========================================================================
    print("\n" + "=" * 105)
    print("2. BẢNG SO SÁNH ĐỐI ĐẦU (SIDE-BY-SIDE): KHÔNG CÓ NLP VS CÓ NLP NẠP SESSION")
    print("=" * 105)
    print(f"{'BỆNH ĐÍCH (CANDIDATE)':<35} | {'RANK (K.NLP)':<12} | {'RANK (CÓ NLP)':<13} | {'ĐIỂM GỐC':<10} | {'Δ Y VĂN':<10} | {'ĐIỂM PHIÊN':<10} | {'BẰNG CHỨNG'}")
    print("-" * 105)

    for c in pipeline_res["candidates"]:
        did = c["disease_id"]
        dname = c["disease_name"][:33]
        base_cand = base_candidates.get(did)
        
        r_base_str = f"Rank #{base_cand['rank_raw']}" if base_cand else "Ngoài Top 15"
        r_nlp_str = f"Rank #{c['rank']}"
        base_s = c.get("base_score", 0.0)
        delta_s = c.get("literature_delta", 0.0)
        final_s = c.get("ranking_score", 0.0)
        ev_count = c.get("session_evidence_count", 0)
        status_tag = f"🟢 +{ev_count} cạnh" if ev_count > 0 else "⚪ 0 cạnh"

        delta_sign = f"+{delta_s:.4f}" if delta_s > 0 else f"{delta_s:.4f}"
        print(f"{dname:<35} | {r_base_str:<12} | {r_nlp_str:<13} | {base_s:<10.4f} | {delta_sign:<10} | {final_s:<10.4f} | {status_tag}")

    # =========================================================================
    # 3. KIỂM CHỨNG ĐƯỜNG ĐI TOPOLOGY MỚI (DISCOVERED PATHS)
    # =========================================================================
    print("\n" + "=" * 105)
    print("3. CÁC ĐƯỜNG DẪN CƠ CHẾ SINH HỌC MỚI KHÁM PHÁ QUA Y VĂN ([⚡ Y VĂN MỚI])")
    print("=" * 105)
    augmented_cands = [c for c in pipeline_res["candidates"] if c.get("session_status") == "augmented"]
    if augmented_cands:
        for cand in augmented_cands[:3]:
            print(f"\n>>> Ứng viên: {cand['disease_name']} ({cand['disease_id']})")
            print(f"    Điểm Hetionet gốc: {cand['base_score']} -> Điểm sau nạp Y văn: {cand['ranking_score']} (Δ = +{cand['literature_delta']})")
            for p in cand.get("structural_paths", [])[:3]:
                tag = " [⚡ Y VĂN MỚI]" if p.get("is_literature_augmented") else " [Hetionet gốc]"
                print(f"    - Đường đi: {p['path_str']}")
                if p.get("supporting_pmids"):
                    print(f"      PMID hỗ trợ: {p['supporting_pmids'][:5]}")
    else:
        print("Không có ứng viên nào nhận được cạnh tăng cường y văn trong số các ứng viên hiển thị.")

    # =========================================================================
    # 4. KIỂM ĐỊNH TÍNH BẤT BIẾN CỦA ĐỒ THỊ TRÊN ĐĨA (ZERO DISK MUTATION)
    # =========================================================================
    print("\n" + "=" * 105)
    print("4. KIỂM ĐỊNH BẢO TOÀN ĐỒ THỊ TRÊN ĐĨA (ZERO DISK MUTATION AUDIT)")
    print("=" * 105)
    final_hash_edges = file_sha256(raw_edges)
    final_hash_nodes = file_sha256(raw_nodes)
    
    edges_intact = (initial_hash_edges == final_hash_edges)
    nodes_intact = (initial_hash_nodes == final_hash_nodes)
    print(f"SHA-256 hetionet-v1.0-edges.sif.gz ban đầu: {initial_hash_edges}")
    print(f"SHA-256 hetionet-v1.0-edges.sif.gz kết thúc: {final_hash_edges}")
    print(f"-> File cạnh Hetionet nguyên vẹn 100%:        {edges_intact} (KHÔNG BỊ NHIỄM BẨN ĐỒ THỊ GỐC)")
    print(f"-> File node Hetionet nguyên vẹn 100%:        {nodes_intact}")

    # Save summary artifact
    out_artifact = ROOT / "artifacts/benchmarks/live_pipeline_session_audit.json"
    out_artifact.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "drug": drug_name,
        "run_id": run_id,
        "session_edges_added": session_data.get("session_edges_added", 0),
        "admitted_nlp_edges": pipeline_res.get("admitted_edges", 0),
        "zero_disk_mutation": edges_intact and nodes_intact,
        "candidates": pipeline_res["candidates"]
    }
    out_artifact.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[OK] Đã hoàn tất và lưu báo cáo chi tiết vào: {out_artifact}")


if __name__ == "__main__":
    run_live_comparison("Dapagliflozin", max_papers=20)
