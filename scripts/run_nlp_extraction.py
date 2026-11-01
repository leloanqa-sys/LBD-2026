#!/usr/bin/env python3
"""Run Canonical Biomedical NLP Relation Extraction across PubMed Abstracts.

Outputs:
1. `dapagliflozin-lbd/data/processed/relations_extracted.jsonl`
2. `dapagliflozin-lbd/data/processed/relations_normalized.jsonl`
3. `dapagliflozin-lbd/data/review_sample.csv` (150 stratified samples for human precision grading)
"""
import csv
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

# Add src to path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))

from lbd.common.logging import get_logger
from lbd.common.validation import validate_contract
from lbd.data.normalize import normalize_relation_record
from lbd.nlp.engine import NLPEngine
from lbd.nlp.postprocess import postprocess_relations
from lbd.nlp.relation_extraction import extract_relations_from_text

logger = get_logger("nlp_extraction_runner")


def run(max_docs: int = None):
    csv_path = ROOT_DIR / "data" / "raw" / "dapagliflozin-abstracts-2018.csv"
    if not csv_path.exists():
        # Fallback to nlp root if available
        alt_path = ROOT_DIR.parent / "nlp" / "dapagliflozin-abstracts-2018.csv"
        if alt_path.exists():
            csv_path = alt_path
        else:
            raise FileNotFoundError(f"Missing corpus: {csv_path}")

    proc_dir = ROOT_DIR / "data" / "processed"
    proc_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Initializing Canonical NLPEngine...")
    engine = NLPEngine.load_default(dataset_mode="biored")
    logger.info(f"NLPEngine loaded successfully. Optimal threshold: {engine.relation_threshold}")

    logger.info("Reading abstracts from raw corpus...")
    docs = []
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pmid = row.get("pmid") or row.get('﻿"pmid"') or row.get('"pmid"')
            docs.append({
                "pmid": str(pmid).strip(),
                "title": row.get("title", "").strip(),
                "abstract": row.get("abstract", "").strip(),
                "year": row.get("print_year", "").strip(),
            })

    if max_docs:
        docs = docs[:max_docs]

    logger.info(f"Loaded {len(docs)} documents. Running canonical relation extraction...")

    all_extracted = []
    all_normalized = []
    rel_type_counter = Counter()

    for idx, d in enumerate(docs, 1):
        if idx % 100 == 0 or idx == len(docs):
            logger.info(
                f"Progress: [{idx}/{len(docs)}] documents processed. Extracted so far: {len(all_extracted)} relations."
            )

        pmid = d["pmid"]
        title = d["title"]
        abstract = d["abstract"]
        if not abstract:
            continue

        raw_rels = extract_relations_from_text(
            pmid, abstract, title=title, engine=engine, strict_postprocess=True
        )

        for r in raw_rels:
            all_extracted.append(r)
            norm_r = normalize_relation_record(r)
            all_normalized.append(norm_r)
            rel_type_counter[norm_r.get("relation_type", "UNKNOWN")] += 1

    logger.info(f"Extraction completed! Total extracted relations: {len(all_extracted)}")
    logger.info(f"Relation type breakdown: {dict(rel_type_counter)}")

    # 1. Export Contract B (Extracted)
    b_path = proc_dir / "relations_extracted.jsonl"
    with open(b_path, "w", encoding="utf-8") as f:
        for r in all_extracted:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    logger.info("Saved Contract B: data/processed/relations_extracted.jsonl")

    # Also save backwards-compatible relations_extracted_v3.jsonl
    b_v3_path = proc_dir / "relations_extracted_v3.jsonl"
    with open(b_v3_path, "w", encoding="utf-8") as f:
        for r in all_extracted:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # 2. Export Contract C (Normalized)
    c_path = proc_dir / "relations_normalized.jsonl"
    with open(c_path, "w", encoding="utf-8") as f:
        for r in all_normalized:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    logger.info("Saved Contract C: data/processed/relations_normalized.jsonl")

    c_v3_path = proc_dir / "relations_normalized_v3.jsonl"
    with open(c_v3_path, "w", encoding="utf-8") as f:
        for r in all_normalized:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Contract Validation
    if all_normalized:
        is_valid, msg = validate_contract(all_normalized[0], "relation.schema.json")
        assert is_valid, f"Contract validation failed: {msg}"
        logger.info("Sample contract validation passed successfully!")

    # 3. Export Stratified Human Review Sample (150 relations)
    review_csv = ROOT_DIR / "data" / "review_sample.csv"
    _export_stratified_review_sample(all_normalized, review_csv, sample_size=150, seed=42)
    logger.info("Saved stratified human review sample: data/review_sample.csv")

    review_v3_csv = ROOT_DIR / "data" / "review_sample_v3.csv"
    _export_stratified_review_sample(all_normalized, review_v3_csv, sample_size=150, seed=42)


def _export_stratified_review_sample(relations: list, out_csv: Path, sample_size: int = 150, seed: int = 42):
    rng = random.Random(seed)
    pools_by_type = {}
    for r in relations:
        t = r.get("relation_type", "UNKNOWN")
        pools_by_type.setdefault(t, []).append(r)

    sampled = []
    types = list(pools_by_type.keys())
    quota_per_type = max(10, sample_size // max(1, len(types)))

    for t, pool in pools_by_type.items():
        take = min(len(pool), quota_per_type)
        sampled.extend(rng.sample(pool, take))

    remaining_pool = [r for r in relations if r not in sampled]
    needed = sample_size - len(sampled)
    if needed > 0 and remaining_pool:
        sampled.extend(rng.sample(remaining_pool, min(needed, len(remaining_pool))))

    rng.shuffle(sampled)

    fieldnames = [
        "relation_id",
        "pmid",
        "source_entity",
        "source_id",
        "target_entity",
        "target_id",
        "relation_type",
        "confidence_score",
        "evidence",
        "human_precision_label",  # 1 = True Relation, 0 = False Relation
        "human_polarity_label",   # POS / NEG / ASSOC
        "reviewer_notes",
    ]

    with open(out_csv, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in sampled:
            writer.writerow({
                "relation_id": r.get("relation_id", ""),
                "pmid": r.get("pmid", ""),
                "source_entity": r.get("source_entity", ""),
                "source_id": r.get("source_standard_id", ""),
                "target_entity": r.get("target_entity", ""),
                "target_id": r.get("target_standard_id", ""),
                "relation_type": r.get("relation_type", ""),
                "confidence_score": round(float(r.get("confidence_score", 0.0)), 4),
                "evidence": r.get("evidence", ""),
                "human_precision_label": "",
                "human_polarity_label": "",
                "reviewer_notes": "",
            })


if __name__ == "__main__":
    run()
