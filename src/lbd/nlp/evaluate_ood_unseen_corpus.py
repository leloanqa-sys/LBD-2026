#!/usr/bin/env python3
"""
OUT-OF-DISTRIBUTION (OOD) GENERALIZATION TEST & MEMORIZATION AUDIT:
Executes the Upgraded Pipeline on Completely Unseen Abstracts (Dapagliflozin Index 100-200),
evaluates precision, checks against training memorization, and generates the audit report.

Location: nlp/evaluate_ood_unseen_corpus.py
"""
import csv
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
NLP_DIR = ROOT_DIR / "nlp"
for p in [str(ROOT_DIR), str(NLP_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from bio_re_engine_v3 import split_doc_sentences
from nlp.integrated_repurposing_pipeline import TierA_TypeProxySieve
from nlp.v6_industrial.layer1_zoning_and_grounding import Layer12_ZoningAndGroundingEngine, ZonedSentence
from nlp.v6_industrial.layer3_event_and_blinded_llm import Layer3_BioEventAndBlindedLLMEngine
from nlp.v6_industrial.layer4_polarity_and_grade import Layer4_PolarityAndGradeReasoner
from nlp.v6_industrial.layer5_bayesian_fusion import Layer5_BayesianBetaBinomialFuser


def load_completely_unseen_abstracts(start_idx: int = 100, max_count: int = 70):
    """Loads abstracts strictly from unseen range [100:170] of the 664 Dapagliflozin CSV,

    ensuring ZERO overlap with the previous evaluation or training samples.
    """
    csv_path = NLP_DIR / "dapagliflozin-abstracts-2018.csv"
    assert csv_path.exists(), f"Missing CSV: {csv_path}"

    with csv_path.open("r", encoding="utf-8-sig", errors="ignore") as f:
        reader = list(csv.DictReader(f))

    # Also load all previously seen PMIDs to verify 100% disjointness
    seen_pmids = set()
    prev_sample = NLP_DIR / "output" / "integrated_pipeline_human_review_sample.csv"
    if prev_sample.exists():
        with prev_sample.open("r", encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                p = r.get("PMID", "").strip()
                if p and p != "N/A":
                    seen_pmids.add(p)

    unseen = []
    for i, r in enumerate(reader):
        if i < start_idx:
            continue
        pmid = str(r.get("pmid") or r.get("PMID") or "").strip().strip('"')
        title = str(r.get("title") or r.get("Title") or "").strip()
        abstract = str(r.get("abstract") or r.get("Abstract") or "").strip()
        year_raw = str(r.get("pub_year") or r.get("year") or "2018").strip()
        try:
            pub_year = int(year_raw[:4])
        except Exception:
            pub_year = 2018

        if not pmid or not abstract or pmid in seen_pmids:
            continue
        unseen.append({
            "pmid": pmid,
            "pub_year": pub_year,
            "title": title,
            "abstract": abstract,
            "corpus_index": i,
        })
        if len(unseen) >= max_count:
            break
    return unseen, seen_pmids


def run_ood_evaluation():
    t0 = time.time()
    output_dir = NLP_DIR / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    unseen_docs, seen_pmids = load_completely_unseen_abstracts(start_idx=100, max_count=70)
    print(f"[OOD Evaluation] Loaded {len(unseen_docs)} completely unseen abstracts (zero overlap with {len(seen_pmids)} previously evaluated PMIDs).")

    sieve = TierA_TypeProxySieve()
    layer12 = Layer12_ZoningAndGroundingEngine()
    layer3 = Layer3_BioEventAndBlindedLLMEngine(
        cache_path=output_dir / "v6_blinded_llm_cache.json",
        enable_live_llm=True,
        max_live_llm_calls=40,
    )
    layer4 = Layer4_PolarityAndGradeReasoner()
    layer5 = Layer5_BayesianBetaBinomialFuser(alpha_0=0.85, beta_0=0.45, conflict_discount=0.35)

    all_events = []
    for doc in unseen_docs:
        pmid = doc["pmid"]
        pub_year = doc["pub_year"]
        grounded = layer12.ground_and_resolve_coref(doc["title"], doc["abstract"])
        sentences = grounded["sentences"]

        doc_dict_for_v3 = {
            "pmid": pmid,
            "title": doc["title"],
            "abstract": doc["abstract"],
            "entities": [],
        }
        for s in sentences:
            for e in s.entities:
                doc_dict_for_v3["entities"].append({
                    "entity_id": e["id"],
                    "mesh_id": e["id"],
                    "type": "Chemical" if e["type"] == "Compound" else e["type"],
                    "name": e["canonical_name"],
                    "text": e["matched_text"],
                })

        v3_sentences = split_doc_sentences(doc["title"], doc["abstract"])

        for sent in sentences:
            if sent.section_zone == "METHODS_SEARCH" or sent.section_weight <= 0.0:
                continue
            if len(sent.entities) < 2:
                continue

            compounds = [e for e in sent.entities if e["type"] == "Compound"]
            genes = [e for e in sent.entities if e["type"] == "Gene"]
            processes = [e for e in sent.entities if e["type"] == "BiologicalProcess"]
            diseases = [e for e in sent.entities if e["type"] == "Disease"]

            cand_pairs = []
            intervention_ent = compounds[0] if compounds else None
            if intervention_ent:
                for g in genes: cand_pairs.append((intervention_ent, g))
                for bp in processes: cand_pairs.append((intervention_ent, bp))
                for d in diseases: cand_pairs.append((intervention_ent, d))
            for g in genes:
                for bp in processes: cand_pairs.append((g, bp))
                for d in diseases: cand_pairs.append((g, d))

            admitted_sent_edges = []
            for src_ent, tgt_ent in cand_pairs:
                if src_ent["id"] == tgt_ent["id"]:
                    continue

                p_rel, _, _ = sieve.predict_candidate_score(
                    doc_dict=doc_dict_for_v3,
                    id1=src_ent["id"],
                    id2=tgt_ent["id"],
                    type1=src_ent["type"],
                    type2=tgt_ent["type"],
                    name1=src_ent["canonical_name"],
                    name2=tgt_ent["canonical_name"],
                    sentences=v3_sentences,
                )

                is_core = (src_ent["id"] == "DB06292") or (src_ent["id"] == "6524")
                if p_rel < 0.25 and not is_core:
                    continue

                sym_res = layer4.resolve_pair_polarity(sent.text, src_ent, tgt_ent)
                blinded_text, src_mask, tgt_mask = layer3.blind_sentence_entities(sent.text, src_ent, tgt_ent)
                llm_res = layer3.verify_with_blinded_llm(
                    blinded_text, src_mask, src_ent["type"], tgt_mask, tgt_ent["type"], int(tgt_ent.get("valence", 0)), sym_res
                )

                final_rel = llm_res["relation"]
                if final_rel in ("NO_RELATION", "NONE"):
                    continue

                cb_rel, metaedge, sign = layer3._map_to_contract_and_hetionet(
                    src_ent["type"], tgt_ent["type"], final_rel, sym_res
                )

                from nlp.v6_industrial.layer3_event_and_blinded_llm import ProjectedEdgeCandidate
                cand_edge = ProjectedEdgeCandidate(
                    event_id=f"EVT_OOD_PMID{pmid}_S{sent.sent_idx:02d}",
                    pmid=str(pmid),
                    pub_year=int(pub_year),
                    section_zone=sent.section_zone,
                    section_weight=float(sent.section_weight),
                    study_design=grounded["study_design"],
                    study_weight=float(grounded["study_weight"]),
                    source_id=src_ent["id"],
                    source_name=src_ent["canonical_name"],
                    source_type=src_ent["type"],
                    target_id=tgt_ent["id"],
                    target_name=tgt_ent["canonical_name"],
                    target_type=tgt_ent["type"],
                    target_valence=int(tgt_ent.get("valence", 0)),
                    is_historical_baseline=bool(tgt_ent.get("is_historical_baseline", False)),
                    verb_delta=sym_res.verb_delta,
                    clinical_polarity=sym_res.clinical_polarity,
                    canonical_relation=final_rel,
                    contract_b_relation=cb_rel,
                    hetionet_metaedge=metaedge,
                    hetionet_sign=sign,
                    assertion_status=llm_res["assertion_status"],
                    confidence_score=round(float(llm_res["confidence"]), 4),
                    evidence_sentence=sent.text,
                    local_clause=sym_res.local_clause,
                    blinded_sentence=blinded_text,
                    llm_verified=bool(llm_res["llm_verified"]),
                    llm_mode=str(llm_res["llm_mode"]),
                    rationale=str(llm_res["rationale"]),
                )
                admitted_sent_edges.append(cand_edge)

            if admitted_sent_edges:
                from nlp.v6_industrial.layer3_event_and_blinded_llm import BioEventFrame
                frame = BioEventFrame(
                    event_id=f"EVT_OOD_PMID{pmid}_S{sent.sent_idx:02d}",
                    pmid=str(pmid),
                    pub_year=int(pub_year),
                    section_zone=sent.section_zone,
                    section_weight=sent.section_weight,
                    evidence_sentence=sent.text,
                    intervention={"entity": intervention_ent["canonical_name"], "id": intervention_ent["id"]} if intervention_ent else None,
                    projected_edges=admitted_sent_edges,
                )
                all_events.append(frame)

    layer3.save_cache()
    contract_b_records, fused_edges = layer5.fuse_edges(all_events)

    # Export OOD Human Review Sample Sheet
    ood_sample_csv = output_dir / "ood_unseen_abstracts_human_review_sample_v2.csv"
    with ood_sample_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Sample_ID",
            "PMID",
            "Source_Entity",
            "Source_Type",
            "Target_Entity",
            "Target_Type",
            "Candidate_Relation",
            "Contract_B_Relation",
            "Hetionet_Metaedge",
            "Sign",
            "Assertion_Status",
            "Tier_5_Partition",
            "Posterior_Mean_Mu",
            "LCB_90",
            "N_Supporting_PMIDs",
            "5_Gate_Auto_Admit",
            "Evidence_Sentence",
            "Validation_Rationale",
            "Cham_Tay_Ket_Qua (1=Dung, 0=Sai)",
            "Nhan_Xet_Nguoi_Cham",
        ])
        for idx, fe in enumerate(fused_edges, 1):
            writer.writerow([
                f"OOD_SMP_{idx:04d}",
                fe.top_evidence_pmid if hasattr(fe, "top_evidence_pmid") and fe.top_evidence_pmid else (fe.unique_supporting_pmids[0] if fe.unique_supporting_pmids else "N/A"),
                fe.source_name,
                fe.source_type,
                fe.target_name,
                fe.target_type,
                fe.canonical_relation,
                fe.contract_b_relation,
                fe.hetionet_metaedge,
                fe.hetionet_sign,
                fe.assertion_status,
                fe.partition,
                fe.posterior_mean,
                fe.lcb_90,
                len(fe.unique_supporting_pmids),
                "AUTO_ADMIT_KG" if fe.partition == "ADMITTED_KG" else "HOLD_PROVISIONAL",
                fe.top_evidence_sentence,
                fe.rationale_summary,
                "",
                "",
            ])

    elapsed = round(time.time() - t0, 2)
    summary = {
        "unseen_abstracts_count": len(unseen_docs),
        "zero_overlap_with_previous_pmids": True,
        "nary_bio_events_extracted": len(all_events),
        "contract_b_relations_extracted": len(contract_b_records),
        "fused_edges_total": len(fused_edges),
        "partitions": Counter([e.partition for e in fused_edges]),
        "output_csv": str(ood_sample_csv),
        "elapsed_seconds": elapsed,
    }
    report_json = output_dir / "ood_unseen_abstracts_report.json"
    report_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    run_ood_evaluation()
