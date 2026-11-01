#!/usr/bin/env python3
"""Run the real pre-2019 Dapagliflozin corpus through V6 (L1/2 -> L3/4 -> L5).

No synthetic documents or generated labels are accepted. Live LLM verification is
opt-in; API annotation failures and missing human labels remain explicit in report.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from lbd.nlp.external_annotations import (  # noqa: E402
    annotate_spans, fetch_europepmc_go, fetch_pubtator, resolve_go_aspects,
    source_manifest,
)
from lbd.nlp.zoning import Layer12_ZoningAndGroundingEngine  # noqa: E402
from lbd.nlp.events import Layer3_BioEventAndBlindedLLMEngine  # noqa: E402
from lbd.nlp.fusion import Layer5_BayesianBetaBinomialFuser  # noqa: E402


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def nonempty_gold_audit(paths: list[Path]) -> dict[str, Any]:
    audit = {}
    total_labels = 0
    for path in paths:
        if not path.exists():
            continue
        rows = read_csv(path)
        columns = list(rows[0]) if rows else []
        label_columns = [c for c in columns if "label" in c.casefold() and c not in {"relation_type"}]
        labeled = sum(bool((row.get(c) or "").strip()) for row in rows for c in label_columns)
        total_labels += labeled
        audit[str(path)] = {"rows": len(rows), "label_columns": label_columns, "nonempty_label_cells": labeled}
    return {"files": audit, "total_nonempty_label_cells": total_labels,
            "precision_recall": None if total_labels == 0 else "requires task-specific adjudication and metric implementation"}


def write_jsonl(path: Path, rows: list[Any]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False, default=str) + "\n" for row in rows), encoding="utf-8")


def schema_audit(records: list[dict[str, Any]], schema_path: Path) -> dict[str, Any]:
    try:
        import jsonschema
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        validator = jsonschema.Draft7Validator(schema)
        errors = []
        for index, record in enumerate(records):
            error = next(validator.iter_errors(record), None)
            if error is not None:
                errors.append({"index": index, "path": list(error.path), "message": error.message})
                if len(errors) >= 20:
                    break
        return {"schema": str(schema_path), "records": len(records), "valid": len(errors) == 0,
                "invalid_count_at_least": len(errors), "sample_errors": errors}
    except ImportError:
        return {"schema": str(schema_path), "records": len(records), "valid": None,
                "error": "jsonschema package not installed"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=ROOT / "nlp" / "dapagliflozin-abstracts-2018.csv")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--max-docs", type=int, default=0, help="0 means full corpus")
    parser.add_argument("--offline", action="store_true", help="Use only existing per-PMID API cache")
    parser.add_argument("--llm-live", action="store_true", help="Opt in to Groq calls; may incur usage/cost")
    parser.add_argument("--max-llm-calls", type=int, default=80)
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    rows = read_csv(args.corpus)
    if not rows or not {"pmid", "title", "abstract", "print_year"}.issubset(rows[0]):
        raise SystemExit("Corpus must have pmid,title,abstract,print_year columns and at least one row")
    if len({r["pmid"] for r in rows}) != len(rows):
        raise SystemExit("Duplicate PMIDs found; refusing ambiguous benchmark input")
    if any(not r["pmid"].isdigit() for r in rows):
        raise SystemExit("Invalid/non-numeric PMID found; refusing mock or malformed records")
    if any(int(r["print_year"]) > 2018 for r in rows):
        raise SystemExit("Post-2018 record found in historical benchmark corpus")
    if args.max_docs:
        rows = rows[:args.max_docs]

    run_id = datetime.now(timezone.utc).strftime("v6_e2e_%Y%m%dT%H%M%SZ")
    out = args.output_dir or (ROOT / "nlp" / "output" / run_id)
    out.mkdir(parents=True, exist_ok=False)
    cache = ROOT / "nlp" / "output" / "v6_e2e_cache"
    cache.mkdir(parents=True, exist_ok=True)
    pmids = [r["pmid"] for r in rows]

    pubtator, pub_stats = fetch_pubtator(pmids, cache, offline=args.offline)
    europepmc, epmc_stats = fetch_europepmc_go(pmids, cache, offline=args.offline)
    go_stats = resolve_go_aspects(europepmc, cache, offline=args.offline)

    l1 = Layer12_ZoningAndGroundingEngine()
    l3 = Layer3_BioEventAndBlindedLLMEngine(
        cache_path=out / "llm_cache.json", enable_live_llm=args.llm_live,
        max_live_llm_calls=args.max_llm_calls,
    )
    all_events = []
    doc_rows = []
    zone_counts: Counter[str] = Counter()
    entities_total = 0
    external_used = Counter()
    for row in rows:
        pmid = row["pmid"]
        ext = annotate_spans(row, pubtator.get(pmid, []) + europepmc.get(pmid, []))
        grounded = l1.ground_and_resolve_coref(row["title"], row["abstract"], pmid=pmid, external_entities=ext)
        for sent in grounded["sentences"]:
            zone_counts[sent.section_zone] += 1
            entities_total += len(sent.entities)
            for ent in sent.entities:
                external_used[ent.get("annotation_source", "local_registry_or_coref")] += 1
        events = l3.extract_document_events(pmid, int(row["print_year"]), grounded)
        all_events.extend(events)
        doc_rows.append({"pmid": pmid, "year": int(row["print_year"]), "study_design": grounded["study_design"],
                         "sentence_count": len(grounded["sentences"]), "grounded_entity_mentions": sum(len(s.entities) for s in grounded["sentences"]),
                         "event_count": len(events), "edge_candidate_count": sum(len(e.projected_edges) for e in events)})

    fuser = Layer5_BayesianBetaBinomialFuser()
    contract_b, fused = fuser.fuse_edges(all_events)
    contract_d = fuser.to_contract_d_records(fused)
    schema_results = {
        "contract_b": schema_audit(contract_b, PROJECT / "contracts" / "relation.schema.json"),
        "contract_d": schema_audit(contract_d, PROJECT / "contracts" / "graph_edge.schema.json"),
    }
    write_jsonl(out / "events.jsonl", [asdict(e) for e in all_events])
    write_jsonl(out / "contract_b_relations.jsonl", contract_b)
    write_jsonl(out / "contract_d_graph_edges.jsonl", contract_d)
    write_jsonl(out / "fused_edges_audit.jsonl", [asdict(e) for e in fused])
    with (out / "documents.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(doc_rows[0]) if doc_rows else ["pmid"])
        writer.writeheader(); writer.writerows(doc_rows)

    gold = nonempty_gold_audit([
        PROJECT / "data" / "gold_annotation_blind_200.csv",
        PROJECT / "data" / "review_sample.csv",
        PROJECT / "data" / "review_sample_v3.csv",
        ROOT / "nlp" / "feedback" / "integrated_pipeline_human_review_sample_labeled.csv",
    ])
    partitions = Counter(e.partition for e in fused)
    manifest = source_manifest(cache, pmids, pub_stats, epmc_stats, go_stats)
    (out / "external_sources.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    input_hash = hashlib.sha256(args.corpus.read_bytes()).hexdigest()
    summary = {
        "run_id": run_id, "corpus": str(args.corpus.resolve()), "corpus_sha256": input_hash,
        "documents": len(rows), "unique_pmids": len(pmids), "year_min": min(int(r["print_year"]) for r in rows),
        "year_max": max(int(r["print_year"]) for r in rows), "offline": args.offline,
        "live_llm_enabled": args.llm_live, "live_llm_calls": l3.live_llm_calls_made,
        "llm_cache_hits": l3.cache_hits, "symbolic_or_router_decisions": l3.symbolic_decisions,
        "sentences_by_zone": dict(zone_counts), "grounded_entity_mentions": entities_total,
        "entity_mentions_by_source": dict(external_used), "event_frames": len(all_events),
        "external_coverage": {
            "pubtator_annotated_pmids": pub_stats["annotated_pmids"],
            "pubtator_total_annotations": pub_stats["annotations"],
            "pubtator_unavailable_pmids": len(pub_stats["unavailable_pmids"]),
            "pubtator_failed_batches": len(pub_stats["api_failures"]),
            "europepmc_annotated_pmids": epmc_stats["annotated_pmids"],
            "europepmc_go_annotations": epmc_stats["go_annotations"],
            "europepmc_failed_batches": len(epmc_stats["api_failures"]),
            "go_aspect_counts": go_stats["aspect_counts"],
            "go_metadata_missing": len(go_stats["go_metadata_missing"]),
        },
        "contract_b_relations": len(contract_b), "contract_d_edges": len(contract_d),
        "fused_partitions": dict(partitions), "fused_edges": len(fused),
        "schema_validation": schema_results,
        "ground_truth_audit": gold,
        "precision": None, "recall": None,
        "accuracy_note": "Precision/recall are not reported unless a nonempty, adjudicated gold set is available; operational corpus counts are not accuracy metrics.",
        "output_dir": str(out.resolve()),
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
