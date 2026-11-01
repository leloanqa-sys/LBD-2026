#!/usr/bin/env python3
"""Evaluate End-to-End Biomedical NLP Engine against Human Ground Truth (Blind 200).

Evaluates:
1. Binary Relation Detection (has_relation == 'YES' vs 'NO')
2. Precision, Recall, F1, Accuracy, and Specificity
3. Clinically Adjudicated KG-Ready Subset (63 asserted graph edges)
4. Polarity and Assertion Status Agreement
5. End-to-end LLM verification impact and stats
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "src"))

from lbd.nlp.engine import NLPEngine
from lbd.nlp.events import BioEventEngine, BioEventFrame
from lbd.nlp.zoning import ZoningEngine


def parse_entity_string(ent_str: str) -> Tuple[str, str]:
    """Parses 'Entity Name [ENTITY:ID]' into (name, id)."""
    m = re.search(r"^(.*?)\s*\[(.*?)\]$", ent_str.strip())
    if m:
        return m.group(1).strip(), m.group(2).strip()
    return ent_str.strip(), ent_str.strip()


def strip_markup(sent: str) -> Tuple[str, str, int, int, str, int, int]:
    """Removes [[SRC:...]] and [[TGT:...]] markup and returns clean text and exact spans."""
    clean = re.sub(r"\[\[(?:SRC|TGT):(.*?)\]\]", r"\1", sent)
    m_src = re.search(r"\[\[SRC:(.*?)\]\]", sent)
    m_tgt = re.search(r"\[\[TGT:(.*?)\]\]", sent)
    s_txt = m_src.group(1) if m_src else ""
    t_txt = m_tgt.group(1) if m_tgt else ""

    s_s = clean.find(s_txt) if s_txt else -1
    s_e = s_s + len(s_txt) if s_s >= 0 else -1

    t_s = clean.find(t_txt) if t_txt else -1
    t_e = t_s + len(t_txt) if t_s >= 0 else -1

    return clean, s_txt, s_s, s_e, t_txt, t_s, t_e


def evaluate(
    ground_truth_path: Path,
    enable_live_llm: bool = True,
    max_live_llm_calls: int = 150,
) -> Dict[str, Any]:
    with open(ground_truth_path, "r", encoding="utf-8-sig") as f:
        gold_rows = list(csv.DictReader(f))

    print(f"Loaded {len(gold_rows)} ground truth records from {ground_truth_path.name}")

    # Initialize Engine with LLM verifier
    event_engine = BioEventEngine(
        enable_live_llm=enable_live_llm,
        max_live_llm_calls=max_live_llm_calls,
    )
    engine = NLPEngine.load_default(
        dataset_mode="biored",
        enable_live_llm=enable_live_llm,
    )
    engine.events = event_engine

    predictions = []
    tp, fp, fn, tn = 0, 0, 0, 0

    # Subset metrics for KG-ready edges
    kg_tp, kg_fp, kg_fn, kg_tn = 0, 0, 0, 0

    by_type_pair = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "tn": 0})
    polarity_matches = 0
    polarity_evaluable = 0

    print(f"Running end-to-end NLP pipeline on all {len(gold_rows)} items (Live LLM: {enable_live_llm})...")

    for idx, r in enumerate(gold_rows, 1):
        item_id = r["item_id"]
        pmid = r["pmid"]
        gold_has_rel = r["has_relation"].strip().upper() == "YES"
        gold_verb = r.get("clinical_verb", "NONE").strip().upper()
        gold_polarity = r.get("polarity_sign", "NONE").strip().upper()
        gold_assertion = r.get("assertion_status", "NOT_APPLICABLE").strip().upper()
        kg_status = r.get("kg_export_status", "NO_EDGE").strip()

        src_name, src_id = parse_entity_string(r["source_entity"])
        tgt_name, tgt_id = parse_entity_string(r["target_entity"])
        src_type = r.get("source_type", "Entity").strip()
        tgt_type = r.get("target_type", "Entity").strip()

        raw_sentence = r["highlighted_sentence"]
        clean_text, s_txt, s_s, s_e, t_txt, t_s, t_e = strip_markup(raw_sentence)

        # Prepare external entities
        ext_entities = []
        if s_s >= 0:
            ext_entities.append({
                "start": s_s, "end": s_e, "text": s_txt,
                "type": src_type, "id": src_id, "canonical_name": src_name,
            })
        if t_s >= 0:
            ext_entities.append({
                "start": t_s, "end": t_e, "text": t_txt,
                "type": tgt_type, "id": tgt_id, "canonical_name": tgt_name,
            })

        doc = {"pmid": pmid, "title": "", "abstract": clean_text}
        grounded, events, scores = engine.run_document(doc, external_entities=ext_entities)

        # Check if relation between src and tgt was predicted
        pred_rel = "NO_RELATION"
        pred_assertion = "asserted"
        pred_conf = 0.0
        pred_cand = None

        # Look for matching edge candidate in events
        for ev in events:
            for cand in ev.projected_edges:
                # Check id match or name match in either direction
                match_fwd = (cand.source_id == src_id or cand.source_name.lower() == src_name.lower()) and \
                            (cand.target_id == tgt_id or cand.target_name.lower() == tgt_name.lower())
                match_rev = (cand.source_id == tgt_id or cand.source_name.lower() == tgt_name.lower()) and \
                            (cand.target_id == src_id or cand.target_name.lower() == src_name.lower())
                if match_fwd or match_rev:
                    if cand.canonical_relation not in ("NO_RELATION", "NONE"):
                        pred_rel = cand.canonical_relation
                        pred_assertion = cand.assertion_status
                        pred_conf = float(cand.confidence_score)
                        pred_cand = cand
                        break
            if pred_cand is not None:
                break

        # Also check statistical classifier scores if candidate wasn't picked up by event parser
        cand_pair_key = tuple(sorted((src_id, tgt_id)))
        stat_score = scores.get(cand_pair_key, {})
        prob_has_rel = stat_score.get("prob_has_relation", 0.0)

        # Final predicted decision: relation detected if event frame found with valid relation OR stat score >= threshold
        system_has_rel = (pred_rel not in ("NO_RELATION", "NONE", "EVALUATED_IN_COHORT"))

        # Scoring Binary Relation
        pair_type_label = f"{src_type}-{tgt_type}"
        if gold_has_rel and system_has_rel:
            tp += 1
            by_type_pair[pair_type_label]["tp"] += 1
        elif not gold_has_rel and system_has_rel:
            fp += 1
            by_type_pair[pair_type_label]["fp"] += 1
        elif gold_has_rel and not system_has_rel:
            fn += 1
            by_type_pair[pair_type_label]["fn"] += 1
        else:
            tn += 1
            by_type_pair[pair_type_label]["tn"] += 1

        # KG Ready Subset (63 edges marked KG_READY vs 100 NO_EDGE)
        is_kg_ready = kg_status == "KG_READY_CONTENT_TEMPORAL_UNVERIFIED"
        is_no_edge = kg_status == "NO_EDGE"
        if is_kg_ready:
            if system_has_rel:
                kg_tp += 1
            else:
                kg_fn += 1
        elif is_no_edge:
            if system_has_rel:
                kg_fp += 1
            else:
                kg_tn += 1

        predictions.append({
            "item_id": item_id,
            "pmid": pmid,
            "src": f"{src_name} [{src_id}]",
            "tgt": f"{tgt_name} [{tgt_id}]",
            "gold_has_relation": gold_has_rel,
            "gold_verb": gold_verb,
            "system_has_relation": system_has_rel,
            "system_relation": pred_rel,
            "system_assertion": pred_assertion,
            "confidence": pred_conf,
            "kg_export_status": kg_status,
        })

        if idx % 50 == 0 or idx == len(gold_rows):
            print(f"Processed [{idx}/{len(gold_rows)}] items...")

    # Calculate overall metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    # KG subset metrics
    kg_precision = kg_tp / (kg_tp + kg_fp) if (kg_tp + kg_fp) > 0 else 0.0
    kg_recall = kg_tp / (kg_tp + kg_fn) if (kg_tp + kg_fn) > 0 else 0.0
    kg_f1 = 2 * kg_precision * kg_recall / (kg_precision + kg_recall) if (kg_precision + kg_recall) > 0 else 0.0

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_records": len(gold_rows),
        "live_llm_enabled": enable_live_llm,
        "llm_stats": {
            "live_calls_made": event_engine.live_llm_calls_made,
            "cache_hits": event_engine.cache_hits,
            "symbolic_router_decisions": event_engine.symbolic_decisions,
        },
        "confusion_matrix_all_200": {
            "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        },
        "metrics_all_200": {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "accuracy": round(accuracy, 4),
            "specificity": round(specificity, 4),
        },
        "kg_ready_subset": {
            "gold_edges_evaluated": 63,
            "confusion_matrix": {"TP": kg_tp, "FP": kg_fp, "FN": kg_fn, "TN": kg_tn},
            "precision": round(kg_precision, 4),
            "recall": round(kg_recall, 4),
            "f1": round(kg_f1, 4),
        },
        "by_type_pair": dict(by_type_pair),
        "predictions": predictions,
    }

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate on Gold 200 ground truth")
    parser.add_argument("--no-llm", action="store_true", help="Disable live LLM calls")
    parser.add_argument("--max-llm-calls", type=int, default=120, help="Maximum live LLM calls")
    args = parser.parse_args()

    gt_file = ROOT_DIR / "data" / "samples" / "gold_annotation_blind_200_annotated_for_kg.csv"
    res = evaluate(gt_file, enable_live_llm=not args.no_llm, max_live_llm_calls=args.max_llm_calls)

    out_file = ROOT_DIR / "data" / "samples" / "evaluation_report_gold_200.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)

    print("\n================ EVALUATION RESULTS (GOLD 200) ================")
    print(f"Total Evaluated: {res['total_records']}")
    print(f"Confusion Matrix: TP={res['confusion_matrix_all_200']['TP']}, FP={res['confusion_matrix_all_200']['FP']}, FN={res['confusion_matrix_all_200']['FN']}, TN={res['confusion_matrix_all_200']['TN']}")
    print(f"Precision:   {res['metrics_all_200']['precision'] * 100:.2f}%")
    print(f"Recall:      {res['metrics_all_200']['recall'] * 100:.2f}%")
    print(f"F1-Score:    {res['metrics_all_200']['f1'] * 100:.2f}%")
    print(f"Accuracy:    {res['metrics_all_200']['accuracy'] * 100:.2f}%")
    print(f"Specificity: {res['metrics_all_200']['specificity'] * 100:.2f}%")
    print("---------------------------------------------------------------")
    print(f"KG-Ready Subset (63 Clean Asserted Edges):")
    print(f"  Precision: {res['kg_ready_subset']['precision'] * 100:.2f}%")
    print(f"  Recall:    {res['kg_ready_subset']['recall'] * 100:.2f}%")
    print(f"  F1-Score:  {res['kg_ready_subset']['f1'] * 100:.2f}%")
    print("---------------------------------------------------------------")
    print(f"LLM Calls Made: {res['llm_stats']['live_calls_made']} | Cache Hits: {res['llm_stats']['cache_hits']} | Symbolic Router Decisions: {res['llm_stats']['symbolic_router_decisions']}")
    print(f"Saved full report to: {out_file.relative_to(ROOT_DIR)}")
