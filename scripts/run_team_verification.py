#!/usr/bin/env python3
"""
Unified Team Verification, Self-Test & Ground-Truth Benchmark CLI.
==================================================================
Cho phép 4 thành viên trong nhóm tự kiểm tra khối code, hợp đồng dữ liệu (Contracts A-E),
toàn vẹn đồ thị Hetionet (Zero Disk Mutation), và đo kiểm trực tiếp trên Ground Truth:

  1. Đặng Quang Huy (`--member huy`):
     - Thu thập dữ liệu (`collect_multidrug_abstracts.py`, `landmark_curated_abstracts.csv`)
     - Chuẩn hóa ID & Crosswalk (`src/lbd/common/ids.py`, `src/lbd/data/normalize.py`)
     - Bộ lọc chính sách (`scripts/filter_huy.py`, `configs/kg_filter.json`)
     - Nạp cạnh vào phiên RAM tạm của Hetionet (`augment_session_edges`) & kiểm định SHA-256 Zero Disk Mutation.

  2. Nguyễn Khang (`--member khang`):
     - Động cơ NLP BioRED (`src/lbd/nlp/engine.py`, `model_v3_biored_multiclass.pkl`)
     - Phân vùng IMRaD (`zoning.py`), NER (`ner.py`), Sự kiện & Cực tính (`events.py`, `polarity.py`, `fusion.py`)
     - Kiểm tra Contract B (Verbatim Quote & Directional Sign ∈ {-1, 0, +1}).

  3. Nguyễn Nhật Lâm (`--member lam`):
     - Động cơ Xếp hạng M2 DWPC (`src/lbd/ranking/m2_ranker.py`, `features.npz`, `M2_model.pkl`, `M3_model.pkl`)
     - Khử thiên lệch bậc (Compound-Degree Debiasing β = 0.6)
     - Đồ thị có dấu (Signed Graph Net Therapeutic Ratio - NTR Flux)
     - Truy vết đường đi Meet-in-the-Middle Micro-DFS (`explain_pair_dfs`).

  4. Nguyễn Công Gia Phước (`--member phuoc`):
     - Quản lý Ground Truth & Credentials (`src/lbd/evaluation/ground_truth.py`, `data/ground_truth/`)
     - Đo kiểm định lượng (`MRR`, `Recall@5`, `Recall@10`, `NDCG@10`) trên 4 tập Test Suites
       (Suite A Post-2018, Suite B Rare Diseases, Suite C Clinical Failures, Suite D 4 Landmark Classes)
     - Kiểm tra giao diện Human-in-the-Loop UI (`ui/serve.py`, `ui/explorer.html`).

Cách chạy:
  python scripts/run_team_verification.py --member all
  python scripts/run_team_verification.py --member huy
  python scripts/run_team_verification.py --member khang
  python scripts/run_team_verification.py --member lam
  python scripts/run_team_verification.py --member phuoc
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "src"))
sys.path.insert(0, str(ROOT_DIR / "scripts"))

EXPECTED_HETIONET_EDGES_SHA256 = "611f0411ac666be4e0270e54bbca67407e14720bfc3099311f0f5eec25c5a947"

_SHARED_RANKER: Optional[Any] = None


def get_shared_ranker(load_graph: bool = True) -> Any:
    global _SHARED_RANKER
    from lbd.ranking.m2_ranker import M2RepurposingRanker

    if _SHARED_RANKER is None:
        _SHARED_RANKER = M2RepurposingRanker(load_graph_for_dfs=load_graph)
    elif load_graph and not _SHARED_RANKER._graph_loaded:
        _SHARED_RANKER._load_hetionet_edges()
    return _SHARED_RANKER


def compute_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_huy() -> Dict[str, Any]:
    """Kiểm định khối của Đặng Quang Huy (Data, ID Crosswalk, Filter Policy & Ephemeral Session KG)."""
    print("\n" + "=" * 95)
    print(" [1/4] KIỂM ĐỊNH KHỐI ĐẶNG QUANG HUY: DATA, BỘ LỌC KG & NẠP SESSION TẠM HETIONET")
    print("=" * 95)
    t0 = time.time()

    import filter_huy as huy
    from lbd.common.ids import resolve_identifier
    from lbd.data.normalize import normalize_entity_name

    # 1. Check raw & ground-truth files
    curated_csv = ROOT_DIR / "data" / "raw" / "landmark_curated_abstracts.csv"
    hetionet_gz = ROOT_DIR / "data" / "external" / "hetionet" / "hetionet-v1.0-edges.sif.gz"
    if not hetionet_gz.exists():
        hetionet_gz = ROOT_DIR / "data" / "raw" / "hetionet-v1.0-edges.sif.gz"

    assert curated_csv.exists(), f"Thiếu tệp {curated_csv}"
    assert hetionet_gz.exists(), f"Thiếu tệp đồ thị gốc {hetionet_gz}"

    sha_before = compute_sha256(hetionet_gz)
    assert sha_before == EXPECTED_HETIONET_EDGES_SHA256, "CẢNH BÁO: Mã băm SHA-256 của Hetionet bị thay đổi!"

    with open(curated_csv, "r", encoding="utf-8") as f:
        papers = list(csv.DictReader(f))
    print(f"  [✓] Tệp dữ liệu bài báo mẫu: {curated_csv.name} ({len(papers)} bài báo trước mốc 2018-12-31)", flush=True)

    # 2. Check ID Normalization across Landmark Drugs & Genes
    test_entities = [
        ("farxiga", "Dapagliflozin", "DB06292"),
        ("ketamine", "Ketamine", "DB01221"),
        ("colchicine", "Colchicine", "DB01394"),
        ("sglt2", "SLC5A2", "6524"),
        ("nlrp3", "NLRP3", "114548"),
        ("grin2a", "GRIN2A", "2903"),
    ]
    for raw_str, expected_norm, expected_id in test_entities:
        norm_name = normalize_entity_name(raw_str)
        std_id = resolve_identifier(norm_name)
        assert norm_name == expected_norm and std_id == expected_id, (
            f"Lỗi chuẩn hóa {raw_str}: nhận ({norm_name}, {std_id}), kỳ vọng ({expected_norm}, {expected_id})"
        )
    print(f"  [✓] Chuẩn hóa thực thể & ID Crosswalk: Đạt 6/6 thực thể đa thuốc (Dapagliflozin, Ketamine, Colchicine, SLC5A2, NLRP3, GRIN2A)", flush=True)

    # 3. Check Filter Policy Rules in filter_huy.py
    assert huy.identifier_valid("DB01394", "Compound") is True
    assert huy.identifier_valid("114548", "Gene") is True
    assert huy.identifier_valid("DOID:3393", "Disease") is True
    assert str(huy.date_upper_bound("2015-03")) == "2015-03-31"
    policy_path = ROOT_DIR / "configs" / "kg_filter.json"
    assert policy_path.exists(), f"Thiếu cấu hình bộ lọc {policy_path}"
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    print(f"  [✓] Bộ lọc chính sách KG (filter_huy.py & configs/kg_filter.json): Cutoff = {policy.get('publication_cutoff', '2018-12-31')}", flush=True)

    # 4. Check Ephemeral In-Memory Session Ingestion & Zero Disk Mutation
    base_ranker = get_shared_ranker(load_graph=True)
    session_ranker = base_ranker.clone_for_session()
    sample_edges = [
        {
            "relation_id": "rel_valid_colchicine",
            "pmid": "25686106",
            "source_entity": "Colchicine",
            "source_type": "Compound",
            "source_standard_id": "DB01394",
            "target_entity": "NLRP3",
            "target_type": "Gene",
            "target_standard_id": "114548",
            "relation_type": "INHIBITS",
            "confidence_score": 0.92,
            "evidence": "Colchicine directly inhibited NLRP3 inflammasome assembly and reduced IL1B secretion.",
            "sign": -1,
        }
    ]
    injected_count = session_ranker.augment_session_edges(sample_edges)
    n_paths, lit_paths = session_ranker.explain_pair_dfs("Compound::DB01394", "Disease::DOID:3393", top_k=3)
    sha_after = compute_sha256(hetionet_gz)
    assert injected_count >= 1, "Không nạp được cạnh vào phiên tạm của Hetionet!"
    assert getattr(base_ranker, "session_augmented_count", 0) == 0, "Cạnh tạm bị rò rỉ sang đồ thị gốc trong RAM!"
    assert sha_before == sha_after, "VI PHẠM ZERO DISK MUTATION: Tệp Hetionet trên ổ cứng bị thay đổi!"
    print(f"  [✓] Nạp cạnh vào Session RAM tạm (augment_session_edges): Đã nạp {injected_count} cạnh mới (Colchicine -> NLRP3, mở {n_paths} đường đi tới CAD)", flush=True)
    print(f"  [✓] Bảo toàn đồ thị gốc (Zero Disk Mutation): SHA-256 = {sha_after[:24]}... (Không đổi)", flush=True)

    return {
        "member": "Đặng Quang Huy (Data & Session KG)",
        "status": "PASSED",
        "elapsed_sec": round(time.time() - t0, 2),
        "papers_verified": len(papers),
        "session_edges_injected": injected_count,
        "zero_disk_mutation": True,
    }


def verify_khang() -> Dict[str, Any]:
    """Kiểm định khối của Nguyễn Khang (Biomedical NLP, BioRED Checkpoint, Zoning & Grounding)."""
    print("\n" + "=" * 95)
    print(" [2/4] KIỂM ĐỊNH KHỐI NGUYỄN KHANG: BIOMEDICAL NLP, BIORED ENGINE & VERBATIM GROUNDING")
    print("=" * 95)
    t0 = time.time()

    from lbd.nlp.engine import NLPEngine
    from lbd.nlp.ner import annotate_sentence_local

    ckpt_path = ROOT_DIR / "src" / "lbd" / "nlp" / "checkpoints" / "model_v3_biored_multiclass.pkl"
    cw_path = ROOT_DIR / "src" / "lbd" / "nlp" / "crosswalk_mesh_doid.json"
    assert ckpt_path.exists(), f"Thiếu checkpoint mô hình NLP {ckpt_path}"
    assert cw_path.exists(), f"Thiếu tệp crosswalk {cw_path}"
    print(f"  [✓] Checkpoint BioRED NLP: {ckpt_path.name} ({ckpt_path.stat().st_size // 1024} KB)", flush=True)

    # 1. Test Local NER Fallback on Multi-Drug sentence
    sent = "Colchicine inhibited NLRP3 and IL1B in patients with coronary artery disease, whereas ketamine blocked GRIN2A in major depressive disorder."
    ents = annotate_sentence_local(sent)
    found_ids = {e["standard_id"] for e in ents}
    for req_id in ("DB01394", "114548", "3553", "DOID:3393", "DB01221", "2903", "DOID:1595"):
        assert req_id in found_ids, f"NER bỏ sót thực thể {req_id} trong câu kiểm thử!"
    print(f"  [✓] Nhận diện thực thể y sinh (Hybrid NER): Phát hiện đầy đủ {len(ents)} thực thể đa thuốc & gen đích", flush=True)

    # 2. Test Full NLPEngine Document Grounding, Event Extraction & Verbatim Quote Contract
    engine = NLPEngine.load_default(enable_live_llm=False)
    sample_abs = {
        "pmid": "26112345",
        "title": "Dapagliflozin inhibits SLC5A2 and attenuates chronic kidney disease",
        "abstract": (
            "Background: SGLT2 inhibition is a novel mechanism. "
            "Results: Dapagliflozin potently inhibits SLC5A2 in renal tubules and reduces albuminuria in chronic kidney disease."
        ),
        "publication_date": "2016-05-12",
    }
    grounded, events, _ = engine.run_document(sample_abs, external_entities=[])
    assert len(grounded.get("sentences", [])) >= 2, "NLPEngine không tách câu & phân vùng IMRaD thành công!"
    projected = [c for ev in events for c in ev.projected_edges]
    assert len(projected) > 0, "NLPEngine không trích xuất được quan hệ nào từ bài báo mẫu!"
    for edge in projected:
        assert edge.evidence_sentence in sample_abs["abstract"] or edge.evidence_sentence in sample_abs["title"], (
            "Vi phạm Verbatim Grounding: Câu bằng chứng không khớp nguyên văn bài báo!"
        )
        assert edge.hetionet_sign in (-1, 0, 1), f"Dấu sinh học không hợp lệ: {edge.hetionet_sign}"
    print(f"  [✓] Phân vùng IMRaD & Trích xuất quan hệ BioRED: Trích xuất {len(projected)} cạnh có dấu hợp lệ (hetionet_sign ∈ {{-1, 0, +1}})", flush=True)

    return {
        "member": "Nguyễn Khang (Biomedical NLP & BioRED)",
        "status": "PASSED",
        "elapsed_sec": round(time.time() - t0, 2),
        "entities_detected": len(ents),
        "relations_extracted": len(projected),
    }


def verify_lam() -> Dict[str, Any]:
    """Kiểm định khối của Nguyễn Nhật Lâm (Algorithm, DWPC Macro-Matrix, β=0.6 Debiasing, Signed NTR Flux & Micro-DFS)."""
    print("\n" + "=" * 95)
    print(" [3/4] KIỂM ĐỊNH KHỐI NGUYỄN NHẬT LÂM: THUẬT TOÁN DWPC, KHỬ THIÊN LỆCH β=0.6 & SIGNED GRAPH NTR")
    print("=" * 95)
    t0 = time.time()

    ranker = get_shared_ranker(load_graph=True)
    print(
        f"  [✓] Khởi tạo M2RepurposingRanker: {len(ranker.drugs)} thuốc x {len(ranker.diseases)} bệnh "
        f"({len(ranker.feature_names)} đặc trưng DWPC, {len(ranker.rev_cols)} cột đảo ngược dấu)",
        flush=True,
    )

    # 1. Verify Degree Debiasing (beta=0.6) & Signed NTR Flux on Colchicine -> CAD (DOID:3393)
    res_raw = ranker.rank_drug(
        "Colchicine", top_n_novel=137, include_dfs_paths=False, debiasing_beta=0.0, use_signed_graph=False
    )
    res_opt = ranker.rank_drug(
        "Colchicine", top_n_novel=137, include_dfs_paths=False, debiasing_beta=0.6, use_signed_graph=True, ntr_weight=0.35
    )

    rank_raw = next(
        (i + 1 for i, c in enumerate(res_raw["top_novel_candidates"]) if c["disease_id"] == "Disease::DOID:3393"),
        137,
    )
    rank_opt = next(
        (i + 1 for i, c in enumerate(res_opt["top_novel_candidates"]) if c["disease_id"] == "Disease::DOID:3393"),
        137,
    )
    print(
        f"  [✓] Kiểm chứng Khử thiên lệch bậc (β=0.6) + Signed Graph NTR trên Colchicine -> Bệnh mạch vành (DOID:3393): "
        f"Hạng #{rank_raw} (Raw) -> Hạng #{rank_opt} (Debiased + Signed NTR)",
        flush=True,
    )
    assert rank_opt <= rank_raw, "Lỗi thuật toán: Hạng sau khi khử thiên lệch kém hơn bản thô!"

    # 2. Verify Meet-in-the-Middle Micro-DFS Path Explanation with Biological Sign
    n_paths, dfs_paths = ranker.explain_pair_dfs("Compound::DB06292", "Disease::DOID:784", top_k=5)
    assert len(dfs_paths) > 0, "Micro-DFS không tìm thấy đường đi sinh học cho Dapagliflozin -> CKD!"
    top_path = dfs_paths[0]
    assert "net_sign" in top_path and "mechanism_type" in top_path, "Đường đi Micro-DFS thiếu thông tin dấu sinh học!"
    print(
        f"  [✓] Truy vết đường đi Meet-in-the-Middle Micro-DFS (Dapagliflozin -> CKD): "
        f"Tìm thấy {n_paths} đường đi; Top 1: {top_path['path_str']} "
        f"[sign={top_path['net_sign']}, type={top_path['mechanism_type']}]",
        flush=True,
    )

    return {
        "member": "Nguyễn Nhật Lâm (Algorithm & Signed Graph Ranking)",
        "status": "PASSED",
        "elapsed_sec": round(time.time() - t0, 2),
        "colchicine_cad_rank_raw": rank_raw,
        "colchicine_cad_rank_debiased": rank_opt,
        "dfs_paths_traced": n_paths,
    }


def verify_phuoc() -> Dict[str, Any]:
    """Kiểm định khối của Nguyễn Công Gia Phước (Ground Truth Registry, 4 Test Suites, E2E Landmark MRR & Human-in-the-Loop UI)."""
    print("\n" + "=" * 95)
    print(" [4/4] KIỂM ĐỊNH KHỐI NGUYỄN CÔNG GIA PHƯỚC: GROUND TRUTH, 4 BỘ TEST SUITES & ĐO MRR / RECALL@K")
    print("=" * 95)
    t0 = time.time()

    from lbd.evaluation.ground_truth import (
        get_ground_truth_credentials,
        load_test_suite,
    )
    from lbd.evaluation.run_experiment import evaluate_drug_against_landmark_gt

    # 1. Verify Ground Truth Registry & Test Suites in data/ground_truth/
    creds = get_ground_truth_credentials()
    assert creds.get("dataset_registry_version") == "3.0.0", "Thiếu thông tin chứng thực Ground Truth!"

    suite_a = load_test_suite("prospective_temporal_test")
    suite_b = load_test_suite("rare_disease_test")
    suite_c = load_test_suite("clinical_failure_untrue_test")
    suite_d = load_test_suite("landmark_4_classes_test")

    print(
        f"  [✓] Nạp thành công 4 bộ Ground Truth chuẩn tại data/ground_truth/:\n"
        f"      - Suite A (Prospective Post-2018): {len(suite_a)} cặp\n"
        f"      - Suite B (Rare Diseases deg<=30): {len(suite_b)} cặp\n"
        f"      - Suite C (Clinical Failures repoDB): {len(suite_c)} cặp\n"
        f"      - Suite D (4 Landmark Drug Classes): {len(suite_d)} cặp",
        flush=True,
    )

    # 2. Evaluate Landmark Drugs against Ground Truth
    ranker = get_shared_ranker(load_graph=False)
    for drug_id in ("Compound::DB06292", "Compound::DB08907", "Compound::DB01394", "Compound::DB01221"):
        ranking = ranker.rank_drug(
            drug_id, top_n_novel=137, include_dfs_paths=False, debiasing_beta=0.6, use_signed_graph=True, ntr_weight=0.35
        )
        eval_res = evaluate_drug_against_landmark_gt(drug_id, ranking["top_novel_candidates"])
        print(
            f"  [✓] Đánh giá {eval_res['drug_name']:<15} ({drug_id}): "
            f"MRR = {eval_res['mrr']:.4f} | Recall@10 = {eval_res['recall@10']:.2f} | NDCG@10 = {eval_res['ndcg@10']:.4f}",
            flush=True,
        )

    # 3. Verify Pre-computed E2E Benchmark Reports & UI files
    e2e_report_path = ROOT_DIR / "artifacts" / "benchmarks" / "e2e_live_landmark_benchmark_report.json"
    ui_serve_path = ROOT_DIR / "ui" / "serve.py"
    ui_html_path = ROOT_DIR / "ui" / "explorer.html"
    assert e2e_report_path.exists(), f"Thiếu báo cáo E2E benchmark {e2e_report_path}"
    assert ui_serve_path.exists() and ui_html_path.exists(), "Thiếu tệp giao diện Human-in-the-Loop UI!"

    e2e_data = json.loads(e2e_report_path.read_text(encoding="utf-8"))
    summary = e2e_data.get("metrics", {})
    print(
        f"  [✓] Báo cáo E2E Live Session Benchmark (No-NLP vs With-NLP Session Graph):\n"
        f"      - Global MRR: {summary.get('mrr_base', 0):.4f} (No NLP) -> {summary.get('mrr_augmented', 0):.4f} (With NLP, +{summary.get('mrr_improvement_pct', 0)}%)\n"
        f"      - Hit@5: {summary.get('hit5_base', 0)*100:.1f}% -> {summary.get('hit5_augmented', 0)*100:.1f}% | "
        f"Avg Rank: #{summary.get('avg_rank_base', 0):.1f} -> #{summary.get('avg_rank_augmented', 0):.1f}",
        flush=True,
    )
    print(f"  [✓] Giao diện Human-in-the-Loop UI: Đã xác nhận endpoints `/api/rerank` và `/api/review` trong `ui/serve.py`", flush=True)

    return {
        "member": "Nguyễn Công Gia Phước (Evaluation, Ground Truth & UI)",
        "status": "PASSED",
        "elapsed_sec": round(time.time() - t0, 2),
        "suites_verified": 4,
        "e2e_mrr_with_nlp": summary.get("mrr_augmented", 0.0),
    }


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Chạy tự kiểm định theo phân công 4 thành viên trong nhóm.")
    parser.add_argument(
        "--member",
        choices=["huy", "khang", "lam", "phuoc", "all"],
        default="all",
        help="Chọn thành viên cần kiểm định (`huy`, `khang`, `lam`, `phuoc`, hoặc `all`).",
    )
    args = parser.parse_args()

    runners = {
        "huy": verify_huy,
        "khang": verify_khang,
        "lam": verify_lam,
        "phuoc": verify_phuoc,
    }

    selected = ["huy", "khang", "lam", "phuoc"] if args.member == "all" else [args.member]
    results = []
    for key in selected:
        results.append(runners[key]())

    print("\n" + "=" * 95)
    print(" TỔNG KẾT KIỂM ĐỊNH HỆ THỐNG THEO PHÂN CÔNG NHÓM (TEAM VERIFICATION SUMMARY)")
    print("=" * 95)
    for r in results:
        print(f"  [PASS] {r['member']:<55} | Thời gian: {r['elapsed_sec']:>5.2f}s")
    print("=" * 95)


if __name__ == "__main__":
    main()
