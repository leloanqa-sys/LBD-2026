
from __future__ import annotations

import argparse
import calendar
import csv
import hashlib
import io
import json
import math
import re
import shutil
import tempfile
import zipfile
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
from typing import Any

import jsonschema
from referencing import Registry, Resource

ROOT = Path(__file__).resolve().parents[1]
SIGNS = {"INHIBITS": -1, "ACTIVATES": 1, "CAUSES": 1, "INCREASES": 1,
         "DECREASES": -1, "TREATS": -1, "PALLIATES": -1}
TYPES = {"Compound", "Gene", "Disease", "Phenotype", "BiologicalProcess", "Pathway", "Anatomy"}
PLACEHOLDERS = {"", "-", "none", "null", "nan", "unknown", "n/a"}


def norm(value: str) -> str:
    return " ".join(value.casefold().split())


def digest_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def identifier_valid(value: str, entity_type: str) -> bool:
    if not isinstance(value, str) or norm(value) in PLACEHOLDERS:
        return False
    patterns = {
        "Compound": r"(?:DB\d{5}|MESH:[CD]\d+|PUBCHEM:\d+|CHEBI:\d+)",
        "Gene": r"[1-9]\d*",
        "Disease": r"(?:DOID:\d+|MESH:D\d+)",
        "Phenotype": r"(?:HP:\d+|MESH:D\d+)",
        "BiologicalProcess": r"GO:\d{7}",
        "Pathway": r"(?:REACT:R-[A-Z]{3}-\d+|KEGG:[A-Za-z0-9]+|WP:\d+)",
        "Anatomy": r"(?:UBERON:\d+|MESH:D\d+)",
    }
    return bool(re.fullmatch(patterns.get(entity_type, r"(?!)"), value))


def date_upper_bound(value: str) -> date:
    """Conservative date bound, never invent precision absent in the source."""
    value = value.strip()
    try:
        return date.fromisoformat(value)
    except ValueError:
        pass
    if re.fullmatch(r"\d{4}-\d{2}", value):
        year, month = map(int, value.split("-"))
        return date(year, month, calendar.monthrange(year, month)[1])
    if re.fullmatch(r"\d{4}", value):
        return date(int(value), 12, 31)
    if re.fullmatch(r"\d{4}\s*[-/]\s*\d{4}", value):
        return date(max(map(int, re.findall(r"\d{4}", value))), 12, 31)
    match = re.fullmatch(r"(\d{4})\s+([A-Za-z]+)(?:-([A-Za-z]+))?(?:\s+(\d{1,2}))?", value)
    if not match:
        raise ValueError(f"Unsupported publication date: {value!r}")
    year, first, last, day = match.groups()
    if first.lower() in {"spring", "summer", "fall", "autumn", "winter"} and not last and not day:
        return date(int(year), 12, 31)
    months = {calendar.month_abbr[m].lower(): m for m in range(1, 13)}
    if first.lower()[:3] not in months or (last and last.lower()[:3] not in months):
        raise ValueError(f"Invalid month: {value!r}")
    month = months[(last or first).lower()[:3]]
    if last and month < months[first.lower()[:3]]:
        # Dec-Jan without the ending year is ambiguous; do not assume same year.
        raise ValueError(f"Ambiguous cross-year date range: {value!r}")
    return date(int(year), month, int(day) if day else calendar.monthrange(int(year), month)[1])


HOLD_CUES = {
    "negation_or_comparable_effect": r"\b(?:not|no|neither|without|unchanged|similar|comparable|independent|noninferior|non-inferior)\b|\bnon[- ]significant\b",
    "hypothesis_or_study_aim": r"\b(?:may|might|could|hypothes\w*|potential\w*|promise|promising|project\w*|simulat\w*|modelled|modeled|predicted|aim\w*|objective\w*|investigat\w*|evaluat\w*|assess\w*|ongoing|planned|will|would|whether)\b",
    "preclinical_or_species_context": r"\b(?:rats?|mice|murine|mouse|rodent\w*|rabbits?|dogs?|canine|animals?|in vitro|in vivo|cell lines?|cultured|streptozotocin)\b",
    "comparison_or_combination_context": r"\b(?:versus|vs\.?|compared|comparison|combination|combined|coadministr\w*|co-administr\w*|add-on|cotreatment|co-treatment)\b",
    "cohort_or_patient_context": r"\b(?:patients? with|subjects? with|participants? with|cohort|enrolled|recruited|randomized to|randomised to)\b",
    "catalog_or_complex_sentence": r";|\brespectively\b",
    "conditional_question_or_refuted_claim": r"\?|\b(?:if|unless|speculat\w*|disprov\w*|refut\w*|theor\w*|controvers\w*|failed to|unconfirmed|unproven)\b",
}
CLASS_CUE = re.compile(r"\b(?:sglt[- ]?2i|sglt[- ]?2 inhibitors?|sglt[- ]?2 inhibition|sodium.* inhibitors?)\b", re.I)
PREDICATES = {
    "INHIBITS": r"(?:inhibit(?:s|ed)?|block(?:s|ed)?|suppress(?:es|ed)?)",
    "ACTIVATES": r"(?:activat(?:e|es|ed)|stimulat(?:e|es|ed))",
    "CAUSES": r"(?:caus(?:e|es|ed)|induc(?:e|es|ed))",
    "INCREASES": r"(?:increas(?:e|es|ed)|rais(?:e|es|ed)|enhanc(?:e|es|ed))",
    "DECREASES": r"(?:decreas(?:e|es|ed)|reduc(?:e|es|ed)|lower(?:s|ed)?)",
    # Reducing a disease incidence is not automatically TREATS that disease.
    "TREATS": r"(?:treat(?:s|ed)?|ameliorat(?:e|es|ed))",
    "PALLIATES": r"(?:palliat(?:e|es|ed)|alleviat(?:e|es|ed))",
}
ADVERBS = r"(?:(?:potently|selectively|significantly|directly|effectively|strongly)\s+){0,3}"
NOMINAL_QUALIFIERS = r"(?:(?:a|an|the|potent|selective|specific|renal|human|competitive|reversible|and|novel|oral|orally|active|highly)\s+){0,12}"


def semantic_check(record: dict[str, Any]) -> list[str]:
    """Conservative argument/polarity gate. Never invert or repair a relation."""
    evidence = record.get("evidence", "")
    src, tgt = record.get("source_raw_text", ""), record.get("target_raw_text", "")
    relation = record.get("relation_type", "")
    reasons = []
    if re.search(r"\b(?:false|incorrect|untrue)\b", evidence, re.I):
        reasons.append("possible_refutation")
    if record.get("assertion_status") != "asserted":
        reasons.append("non_asserted_relation")
    if relation not in SIGNS or record.get("canonical_relation", relation) != relation:
        reasons.append("noncausal_or_canonical_relation_mismatch")
    if CLASS_CUE.search(src) or CLASS_CUE.search(tgt):
        reasons.append("drug_class_or_event_promoted_to_atomic_entity")
    if re.search(r"\b(?:inhibition|activation|deficiency|knockout)\b", src, re.I):
        reasons.append("event_used_as_entity")
    for code, pattern in HOLD_CUES.items():
        if re.search(pattern, evidence, re.I):
            reasons.append(code)
    # Multiple named drugs require resolving the intervention/comparator roles.
    drugs = set(re.findall(r"\b(?:dapagliflozin|canagliflozin|empagliflozin|ipragliflozin|tofogliflozin|luseogliflozin|metformin|sitagliptin|glimepiride)\b", evidence, re.I))
    if len({d.lower() for d in drugs}) > 1:
        reasons.append("multiple_drugs_argument_binding_unresolved")
    if not src or not tgt or norm(src) == norm(tgt):
        return sorted(set(reasons + ["missing_or_identical_raw_arguments"]))
    source = re.escape(src)
    target = re.escape(tgt)
    # Exact bounded arguments. No .* spanning another clause or another target.
    s = rf"(?<!\w){source}(?!\w)"
    t = rf"(?<!\w){target}(?!\w)"
    predicate = PREDICATES.get(relation, r"(?!)")
    # Do not mistake 'insulin sensitivity' for insulin as the affected gene.
    argument_end = r"(?=$|[.,;:()]|\s+(?:in|by|and|but|whereas|which|that|with|to)\b)"
    active = rf"{s}\s+{ADVERBS}{predicate}\s+(?:the\s+)?(?:human\s+)?{t}{argument_end}"
    passive = rf"{t}\s+(?:is|was|were|are)\s+{ADVERBS}{predicate}\s+by\s+{s}"
    patterns = [active, passive]
    if relation in {"INHIBITS", "ACTIVATES"}:
        noun = "inhibitor" if relation == "INHIBITS" else "activator"
        # 'S is a selective T inhibitor' or 'S is a selective inhibitor of T'.
        lead = rf"{s}(?:\s+(?:is|was|acts as|has been identified as|was identified as)|,)?\s+{NOMINAL_QUALIFIERS}"
        patterns += [rf"{lead}{t}\s+{noun}\b",
                     rf"{lead}{noun}\s+of\s+{NOMINAL_QUALIFIERS}{t}"]
    if not any(re.search(pattern, evidence, re.I) for pattern in patterns):
        reasons.append("no_direct_pair_specific_predicate")
    if src and tgt and (not re.search(s, evidence, re.I) or not re.search(t, evidence, re.I)):
        reasons.append("raw_argument_not_in_evidence")
    return sorted(set(reasons))


class Bundle:
    """Read without extracting archive paths onto disk."""
    def __init__(self, path: Path):
        self.path = path.resolve()
        self.archive = zipfile.ZipFile(path) if path.is_file() else None
        if self.archive:
            infos = [x for x in self.archive.infolist() if not x.is_dir()]
            names = [x.filename for x in infos]
            if len(names) != len(set(names)):
                raise ValueError("Duplicate ZIP entry names")
            if sum(x.file_size for x in infos) > 512 * 1024 * 1024:
                raise ValueError("ZIP exceeds 512 MiB uncompressed size limit")
            if any(".." in Path(n).parts or n.startswith(("/", "\\")) for n in names):
                raise ValueError("Unsafe ZIP member path")
            anchors = [n for n in names if n.endswith("/run_report.json") or n == "run_report.json"]
            if len(anchors) != 1:
                raise ValueError("Expected exactly one run_report.json")
            self.prefix = anchors[0][:-len("run_report.json")]
            self.names = sorted(n[len(self.prefix):] for n in names if n.startswith(self.prefix))
        else:
            if not path.is_dir():
                raise ValueError(f"Input does not exist: {path}")
            self.prefix = ""
            self.names = sorted(p.relative_to(path).as_posix() for p in path.rglob("*") if p.is_file())

    def read(self, name: str) -> bytes:
        if name not in self.names:
            raise ValueError(f"Missing required input: {name}")
        return self.archive.read(self.prefix + name) if self.archive else (self.path / name).read_bytes()

    def csv(self, name: str) -> list[dict[str, str]]:
        reader = csv.DictReader(io.StringIO(self.read(name).decode("utf-8-sig"), newline=""))
        if not reader.fieldnames or len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise ValueError(f"Missing/duplicate CSV headers: {name}")
        result = list(reader)
        if any(None in row or any(v is None for v in row.values()) for row in result):
            raise ValueError(f"Malformed CSV row width: {name}")
        return result


def validators() -> dict[str, Any]:
    registry = Registry()
    schemas = {}
    for path in (ROOT / "contracts").glob("*.schema.json"):
        schema = json.loads(path.read_text(encoding="utf-8"))
        schema.setdefault("$id", path.resolve().as_uri())
        registry = registry.with_resource(path.resolve().as_uri(), Resource.from_contents(schema))
        schemas[path.name] = schema
    return {name: jsonschema.validators.validator_for(schema)(schema, registry=registry,
                     format_checker=jsonschema.FormatChecker()) for name, schema in schemas.items()}


def schema_errors(validator: Any, record: Any) -> list[str]:
    return [f"{'/'.join(map(str, e.absolute_path)) or '$'}: {e.message}"
            for e in validator.iter_errors(record)]


def read_papers(path: Path | None) -> dict[str, dict[str, Any]]:
    if not path:
        return {}
    text = path.read_text(encoding="utf-8-sig")
    if path.suffix.lower() == ".csv":
        records = list(csv.DictReader(io.StringIO(text, newline="")))
    else:
        records = [json.loads(line) for line in text.splitlines() if line.strip()]
    papers = {}
    for row in records:
        pmid = str(row.get("pmid", "")).strip()
        if not re.fullmatch(r"[1-9]\d*", pmid) or pmid in papers:
            raise ValueError(f"Invalid or duplicate original paper PMID: {pmid!r}")
        if not isinstance(row.get("abstract"), str) or not isinstance(row.get("title"), str):
            raise ValueError(f"Original paper {pmid} is missing title/abstract")
        papers[pmid] = row
    return papers


def inventory(bundle: Bundle) -> list[dict[str, Any]]:
    results = []
    doc_schema = None
    if "extracted_relation_document.schema.json" in bundle.names:
        schema = json.loads(bundle.read("extracted_relation_document.schema.json"))
        def check_local_refs(value: Any) -> None:
            if isinstance(value, dict):
                for key, child in value.items():
                    if key in {"$ref", "$dynamicRef", "$recursiveRef"} and (not isinstance(child, str) or not child.startswith("#")):
                        raise ValueError("Bundled document schema must use only fragment-local references")
                    check_local_refs(child)
            elif isinstance(value, list):
                for child in value:
                    check_local_refs(child)
        check_local_refs(schema)
        jsonschema.Draft202012Validator.check_schema(schema)
        doc_schema = jsonschema.Draft202012Validator(schema, registry=Registry())
    for name in bundle.names:
        content = bundle.read(name)
        item = {"file": name, "bytes": len(content), "sha256": digest_bytes(content), "records": 0,
                "schema_errors": 0, "disposition": "audit_only_not_imported"}
        if name.endswith(".csv"):
            item["records"] = len(bundle.csv(name))
        elif name.endswith((".json", ".jsonl")):
            records = ([json.loads(line) for line in content.decode("utf-8-sig").splitlines() if line.strip()]
                       if name.endswith(".jsonl") else [json.loads(content)])
            item["records"] = len(records)
            if doc_schema and (name.startswith("contract_b_documents/") or name == "contract_b_documents.jsonl"):
                item["schema_errors"] = sum(len(schema_errors(doc_schema, r)) for r in records)
        results.append(item)
    return results


def numeric(value: Any) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("Nonfinite number")
    return number


QUARANTINE_SOFT_REASONS = {
    "negation_or_comparable_effect",
    "possible_refutation",
    "conditional_question_or_refuted_claim",
    "evidence_not_verbatim_in_original_source",
    "publication_after_cutoff_or_range_crosses_cutoff",
}


def is_quarantine_record(record: dict[str, Any], hard: list[str], soft: list[str]) -> list[str]:
    """Determine strict Tier 3 QUARANTINE reasons (0% negation/contraindication/leakage tolerance)."""
    q_reasons: list[str] = list(hard)
    if str(record.get("assertion_status", "")).strip().lower() == "negated":
        q_reasons.append("assertion_status_negated")
    if str(record.get("canonical_relation", "")).strip().upper() == "CONTRAINDICATED_FOR":
        q_reasons.append("contraindicated_relation")
    if str(record.get("relation_type", "")).strip().upper() == "CONTRAINDICATED_FOR":
        q_reasons.append("contraindicated_relation")
    if explicit_negation(record):
        q_reasons.append("explicit_negation_in_evidence")
    for reason in soft:
        if reason in QUARANTINE_SOFT_REASONS:
            q_reasons.append(reason)
    return sorted(set(q_reasons))


def filter_record(row: dict[str, str], policy: dict[str, Any], papers: dict[str, Any],
                  mention_index: set[tuple[str, str, str, str]],
                  sentences: set[tuple[str, str]], checks: dict[str, Any],
                  sentence_entities: dict[tuple[str, str, str, str], set[str]],
                  pmid_id_index: set[tuple[str, str, str]] | None = None,
                  sentence_id_index: dict[tuple[str, str, str], set[str]] | None = None) -> tuple[dict[str, Any], list[str], list[str]]:
    record = dict(row)
    reject, review = [], []
    for field in ["relation_id", "source_entity", "target_entity", "source_standard_id", "target_standard_id",
                  "source_raw_text", "target_raw_text", "evidence", "pmid"]:
        if not str(record.get(field, "")).strip():
            reject.append(f"missing_{field}")
    for field in ["confidence_score", "section_weight", "study_weight"]:
        try:
            record[field] = numeric(record.get(field, ""))
            if not 0 <= record[field] <= 1:
                raise ValueError("Out of range")
        except (ValueError, TypeError):
            reject.append(f"invalid_{field}")
            record[field] = None
    try:
        sign = numeric(record.get("sign", ""))
        if sign not in (-1, 0, 1):
            raise ValueError("Invalid sign")
        record["sign"] = int(sign)
    except (TypeError, ValueError):
        record["sign"] = None
        reject.append("invalid_sign")
    for role in ("source", "target"):
        entity_type, entity_id = record.get(role + "_type", ""), record.get(role + "_standard_id", "")
        id_ok = entity_type in TYPES and identifier_valid(entity_id, entity_type)
        if entity_type not in TYPES:
            reject.append(f"unsupported_{role}_type")
        elif not id_ok:
            reject.append(f"invalid_or_nonatomic_{role}_id")
        pmid_val = record.get("pmid", "")
        raw_norm = norm(record.get(role + "_raw_text", ""))
        key = (pmid_val, raw_norm, entity_id, entity_type)
        id_grounded_in_pmid = bool(pmid_id_index and (pmid_val, entity_id, entity_type) in pmid_id_index)
        if key not in mention_index and not (id_ok and id_grounded_in_pmid):
            review.append(f"{role}_mention_id_not_grounded")
        sentence_key = (pmid_val, norm(record.get("evidence", "")), raw_norm, entity_type)
        id_grounded_in_sent = bool(
            sentence_id_index
            and entity_id in sentence_id_index.get((pmid_val, norm(record.get("evidence", "")), entity_type), set())
        )
        if sentence_entities.get(sentence_key) != {entity_id} and not (id_ok and id_grounded_in_sent):
            review.append(f"{role}_id_missing_or_ambiguous_in_evidence_sentence")
        has_verified_id = id_ok and (
            id_grounded_in_sent or id_grounded_in_pmid or record.get("normalization_status") == "verified"
        )
        aliases = policy.get("drug_aliases", {}).get(entity_id)
        if aliases and raw_norm not in set(map(norm, aliases)) and not has_verified_id:
            review.append(f"{role}_drug_alias_not_verified")
        disease_aliases = policy.get("disease_aliases", {}).get(entity_id)
        if disease_aliases and raw_norm not in set(map(norm, disease_aliases)) and not has_verified_id:
            review.append(f"{role}_disease_alias_or_subtype_not_verified")
    if record.get("source_standard_id") == record.get("target_standard_id"):
        reject.append("self_loop")
    relation = record.get("relation_type")
    if relation in SIGNS and record["sign"] != SIGNS[relation]:
        reject.append("relation_sign_mismatch")
    meta_key = "|".join(record.get(k, "") for k in ("source_type", "target_type", "relation_type"))
    expected_meta = policy["metaedges"].get(meta_key)
    allowed_metaedges = policy.get("allowed_metaedges")
    if not expected_meta or (allowed_metaedges is not None and expected_meta not in allowed_metaedges):
        review.append("unsupported_relation_domain_range")
    elif record.get("metaedge") != expected_meta:
        reject.append("metaedge_domain_range_mismatch")
    if record.get("normalization_status") != "verified":
        review.append("normalization_not_verified")
    # Drug treatment requires an actual drug identifier, not a chemical/biomarker.
    if relation in {"TREATS", "PALLIATES"} and not re.fullmatch(r"DB\d{5}", record.get("source_standard_id", "")):
        review.append("treatment_source_not_verified_drug")
    if record.get("section_zone") not in policy["allowed_sections"]:
        review.append("section_not_allowed_for_kg")
    if record.get("study_design") in policy["held_study_designs"]:
        review.append("study_context_requires_review")
    elif record.get("study_design") not in policy["allowed_study_designs"]:
        review.append("unknown_study_context")
    if record["confidence_score"] is not None and record["confidence_score"] < policy["min_extraction_score"]:
        review.append("low_extraction_score")
    review.extend(semantic_check(record))
    pmid, evidence = record.get("pmid", ""), record.get("evidence", "")
    if (pmid, norm(evidence)) not in sentences:
        review.append("evidence_not_in_grounded_sentence_snapshot")
    paper = papers.get(pmid)
    record.update(evidence_origin="unverified", semantic_status="unverified")
    if paper is None:
        review.extend(["original_abstract_not_supplied", "publication_date_not_verified"])
    else:
        # Exact source quote, no lowercase/whitespace rewriting to pass this gate.
        if evidence and evidence in paper["abstract"]:
            record["evidence_origin"] = "abstract"
        elif evidence and evidence in paper["title"]:
            review.append("title_only_evidence")
            record["evidence_origin"] = "title"
        else:
            reject.append("evidence_not_verbatim_in_original_source")
        source_date = str(paper.get("publication_date") or paper.get("print_date") or "")
        try:
            upper = date_upper_bound(source_date)
            year = paper.get("print_year")
            if year and str(upper.year) != str(year).strip():
                raise ValueError("Print year conflicts with date")
            record.update(publication_date=upper.isoformat(), publication_date_raw=source_date,
                          publication_date_interpretation="conservative_upper_bound",
                          temporal_filter_status="verified_against_original_metadata")
            if upper > date.fromisoformat(policy["publication_cutoff"]):
                reject.append("publication_after_cutoff_or_range_crosses_cutoff")
        except ValueError:
            review.append("publication_date_not_verified")
    basic_errors = schema_errors(checks["relation.schema.json"], record)
    if basic_errors:
        reject.append("contract_b_schema_invalid")
        record["schema_error_details"] = basic_errors
    if not reject and not review:
        record["semantic_status"] = "direct_pattern_verified"
        strict_errors = schema_errors(checks["kg_ingestion_relation.schema.json"], record)
        if strict_errors:
            reject.append("strict_relation_schema_invalid")
            record["schema_error_details"] = strict_errors
    return record, sorted(set(reject)), sorted(set(review))


def explicit_negation(record: dict[str, Any]) -> bool:
    """Bind an explicit negated predicate to this pair, independently of votes."""
    src = re.escape(record.get("source_raw_text", ""))
    tgt = re.escape(record.get("target_raw_text", ""))
    verb = PREDICATES.get(record.get("relation_type"), r"(?!)")
    text = record.get("evidence", "")
    active = rf"(?<!\w){src}(?!\w)\s+(?:(?:does|did|do|can|could|will|would)\s+)?(?:not|never)\s+{ADVERBS}{verb}\s+(?:the\s+)?{tgt}(?!\w)"
    passive = rf"(?<!\w){tgt}(?!\w)\s+(?:is|was|were|are)\s+(?:not|never)\s+{ADVERBS}{verb}\s+by\s+{src}(?!\w)"
    return bool(src and tgt and (re.search(active, text, re.I) or re.search(passive, text, re.I)))


def json_dump(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def jsonl_dump(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")


def csv_dump(path: Path, records: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for record in records:
            writer.writerow({k: json.dumps(v, ensure_ascii=False, allow_nan=False) if isinstance(v, (list, dict)) else v
                             for k, v in record.items() if k in fields})


def run(input_path: Path, output: Path, papers_path: Path | None = None,
        policy_path: Path = ROOT / "configs" / "kg_filter.json") -> dict[str, Any]:
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output directory is not empty; choose a new --output directory")
    if input_path.is_dir() and (input_path.resolve() == output.resolve() or input_path.resolve() in output.resolve().parents):
        raise ValueError("Output must not be inside input directory")
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    if policy.get("policy_version") != "conservative_rules_v1":
        raise ValueError("Unsupported policy_version")
    if type(policy.get("min_unique_pmids")) is not int or policy["min_unique_pmids"] < 2:
        raise ValueError("min_unique_pmids must be an integer >= 2")
    if not 0 <= numeric(policy["min_extraction_score"]) <= 1:
        raise ValueError("min_extraction_score outside [0,1]")
    date.fromisoformat(policy["publication_cutoff"])
    bundle, checks, papers = Bundle(input_path), validators(), read_papers(papers_path)
    files = inventory(bundle)
    candidate_file = ("relations_contract_b_pre_postprocess.csv" if "relations_contract_b_pre_postprocess.csv" in bundle.names
                      else "relations_contract_b.csv")
    candidates = bundle.csv(candidate_file)
    relation_ids = [row.get("relation_id") for row in candidates]
    duplicate_ids = {key for key, count in Counter(relation_ids).items() if count > 1}
    required_columns = {"relation_id", "source_entity", "target_entity", "source_standard_id", "target_standard_id",
                        "source_raw_text", "target_raw_text", "source_type", "target_type", "relation_type",
                        "canonical_relation", "assertion_status", "confidence_score", "normalization_status",
                        "evidence", "pmid", "sign", "metaedge", "section_zone", "study_design", "section_weight", "study_weight"}
    if not required_columns.issubset(csv.DictReader(io.StringIO(bundle.read(candidate_file).decode("utf-8-sig"))).fieldnames or []):
        raise ValueError(f"Candidate CSV empty or missing columns: {sorted(required_columns - set(candidates[0] if candidates else {}))}")
    mentions = bundle.csv("entity_mentions.csv")
    mention_index = {(r["pmid"], norm(r["mention_text"]), r["entity_id"], r["entity_type"])
                     for r in mentions if r.get("offset_status") in {"verified_exact", "recovered_unique_quote"}
                     and identifier_valid(r.get("entity_id", ""), r.get("entity_type", ""))}
    pmid_id_index = {(r["pmid"], r["entity_id"], r["entity_type"])
                     for r in mentions if r.get("offset_status") in {"verified_exact", "recovered_unique_quote"}
                     and identifier_valid(r.get("entity_id", ""), r.get("entity_type", ""))}
    sentences = set()
    sentence_entities = defaultdict(set)
    sentence_id_index = defaultdict(set)
    for line in bundle.read("grounded_sentences.jsonl").decode("utf-8-sig").splitlines():
        if line.strip():
            sentence = json.loads(line)
            s_pmid = str(sentence["pmid"])
            s_norm = norm(sentence["sentence"])
            sentences.add((s_pmid, s_norm))
            for entity in sentence.get("entities", []):
                eid = entity.get("id", "")
                etype = entity.get("type", "")
                sentence_entities[(s_pmid, s_norm, norm(entity.get("matched_text", "")), etype)].add(eid)
                if eid and etype:
                    sentence_id_index[(s_pmid, s_norm, etype)].add(eid)
    tier_cfg = policy.get("tiering") or {}
    tier2_min_pmids = int(tier_cfg.get("tier2_min_unique_pmids", 1))
    tier2_min_score = float(tier_cfg.get("tier2_min_extraction_score", 0.45))
    tier2_allowed_status = set(tier_cfg.get("tier2_allowed_assertion_statuses", ["asserted", "uncertain"]))
    tier2_allowed_designs = set(tier_cfg.get(
        "tier2_allowed_study_designs",
        ["RCT_CLINICAL_TRIAL", "REAL_WORLD_COHORT", "REVIEW_ARTICLE", "PRECLINICAL_IN_VIVO", "IN_VITRO_CELLULAR"],
    ))
    tier2_allowed_sections = set(tier_cfg.get(
        "tier2_allowed_sections",
        ["TITLE", "BACKGROUND", "RESULTS", "CONCLUSIONS"],
    ))

    audits, filtered, rejected, review_records = [], [], [], []
    tier2_candidates, quarantine_records = [], []
    seen = set()
    all_signs = defaultdict(set)
    tier2_all_signs = defaultdict(set)
    negated_keys = set()
    quarantined_pair_rels = defaultdict(set)
    for index, row in enumerate(candidates, 1):
        record, hard, soft = filter_record(
            row, policy, papers, mention_index, sentences, checks, sentence_entities,
            pmid_id_index=pmid_id_index, sentence_id_index=sentence_id_index,
        )
        if row["relation_id"] in duplicate_ids:
            hard.append("duplicate_relation_id")
        key = tuple(record.get(k) for k in ["pmid", "source_standard_id", "target_standard_id", "relation_type", "assertion_status", "evidence"])
        if key in seen:
            hard.append("duplicate_evidence_record")
        seen.add(key)
        pair = (row["source_standard_id"], row["target_standard_id"])
        pair_rel = (*pair, row["relation_type"])
        q_reasons = is_quarantine_record(record, hard, soft)
        # Only well-formed, explicit evidence can create a polarity conflict.
        if not hard and not soft:
            all_signs[pair].add(record["sign"])
        if not q_reasons and record.get("relation_type") in SIGNS and record.get("sign") in (-1, 1):
            tier2_all_signs[pair].add(record["sign"])
        if (not hard and record.get("evidence_origin") == "abstract"
                and "publication_date_not_verified" not in soft and explicit_negation(record)):
            negated_keys.add(pair_rel)
        if q_reasons:
            quarantined_pair_rels[pair_rel].update(q_reasons)
            if any(r in q_reasons for r in ("assertion_status_negated", "contraindicated_relation", "explicit_negation_in_evidence", "negation_or_comparable_effect", "possible_refutation")):
                negated_keys.add(pair_rel)
        decision = "REJECT" if hard else "REVIEW" if soft else "PASS_RELATION"
        audit = {"row_number": index, "relation_id": row["relation_id"], "pmid": row["pmid"],
                 "source_entity": row["source_entity"], "target_entity": row["target_entity"],
                 "relation_type": row["relation_type"], "decision": decision,
                 "reject_reasons": sorted(set(hard)), "review_reasons": sorted(set(soft)),
                 "quarantine_reasons": q_reasons, "evidence": row["evidence"]}
        audits.append(audit)
        record["filter_audit"] = {k: audit[k] for k in ["decision", "reject_reasons", "review_reasons", "quarantine_reasons"]}
        if hard:
            rejected.append(record)
            quarantine_records.append(record)
        elif soft:
            review_records.append(record)
            if q_reasons:
                quarantine_records.append(record)
            else:
                # Eligible for Tier 2 HYPOTHESIS_EDGE if directional, verbatim (when papers supplied), within cutoff, non-negated
                verbatim_ok = (not papers) or (record.get("evidence_origin") in {"abstract", "title"})
                temporal_ok = (not papers) or ("publication_date_not_verified" not in soft)
                score_ok = record.get("confidence_score") is not None and record["confidence_score"] >= tier2_min_score
                meta_ok = "unsupported_relation_domain_range" not in soft
                if (
                    record.get("relation_type") in SIGNS
                    and record.get("assertion_status") in tier2_allowed_status
                    and record.get("study_design") in tier2_allowed_designs
                    and record.get("section_zone") in tier2_allowed_sections
                    and verbatim_ok and temporal_ok and score_ok and meta_ok
                ):
                    tier2_candidates.append(record)
        else:
            filtered.append(record)
            tier2_candidates.append(record)
    groups = defaultdict(list)
    for record in filtered:
        groups[tuple(record[k] for k in ["source_standard_id", "target_standard_id", "relation_type"])].append(record)
    edges, edge_audit, ready_ids = [], [], set()
    admitted_keys: set[tuple[str, str, str]] = set()
    for key, records in sorted(groups.items()):
        pmids = sorted({r["pmid"] for r in records})
        reasons = []
        if len(pmids) < policy["min_unique_pmids"]:
            reasons.append("insufficient_unique_pmids_after_filtering")
        if len(all_signs[key[:2]]) > 1:
            reasons.append("opposite_polarity_evidence_requires_review")
        if key in negated_keys:
            reasons.append("negated_pair_evidence_requires_review")
        # Negated/uncertain and other held claims never become positive votes.
        edge_id = "KGF_" + digest_bytes("|".join(key).encode())[:16]
        first = records[0]
        edge = {"edge_id": edge_id, "source": key[0], "target": key[1], "relation_type": key[2],
                "source_name": first["source_entity"], "target_name": first["target_entity"],
                "source_type": first["source_type"], "target_type": first["target_type"],
                "metaedge": first["metaedge"], "sign": SIGNS[key[2]], "assertion_status": "asserted",
                "pmids": pmids, "evidence_ids": sorted({r["relation_id"] for r in records}),
                "unique_supporting_pmids": len(pmids), "source_dataset": "nlp_literature", "tier": 2,
                "partition": "ADMITTED_KG", "validation_method": "conservative_rules_v1",
                "publication_date_upper_bound_min": min(r["publication_date"] for r in records),
                "publication_date_upper_bound_max": max(r["publication_date"] for r in records)}
        if not reasons:
            errors = schema_errors(checks["kg_ingestion_edge.schema.json"], edge)
            if errors:
                raise ValueError(f"Generated edge failed strict schema: {errors}")
            edges.append(edge)
            admitted_keys.add(key)
            ready_ids.update(edge["evidence_ids"])
        edge_audit.append({**edge, "decision": "REVIEW" if reasons else "ADMIT", "reasons": reasons})

    # Tier 2 (HYPOTHESIS_EDGE -> kg_edges_hypothesis.jsonl) & Tier 3 (QUARANTINE -> kg_edges_quarantine.jsonl)
    t2_groups = defaultdict(list)
    for record in tier2_candidates:
        gkey = tuple(record[k] for k in ["source_standard_id", "target_standard_id", "relation_type"])
        if gkey not in admitted_keys:
            t2_groups[gkey].append(record)

    hypothesis_edges: list[dict[str, Any]] = []
    quarantine_edges: list[dict[str, Any]] = []
    quarantine_seen_keys: set[tuple[str, str, str]] = set()

    for key, records in sorted(t2_groups.items()):
        pmids = sorted({r["pmid"] for r in records})
        q_block: list[str] = []
        if len(all_signs[key[:2]]) > 1 or len(tier2_all_signs[key[:2]]) > 1:
            q_block.append("opposite_polarity_evidence")
        if key in negated_keys:
            q_block.append("negated_pair_evidence")
        if any(r.get("assertion_status") == "negated" for r in records):
            q_block.append("assertion_status_negated")
        if any(r.get("canonical_relation") == "CONTRAINDICATED_FOR" for r in records):
            q_block.append("contraindicated_relation")
        first = records[0]
        edge_id = "KGH_" + digest_bytes("|".join(key).encode())[:16]
        pub_dates = [r["publication_date"] for r in records if r.get("publication_date")]
        if q_block:
            quarantine_seen_keys.add(key)
            quarantine_edges.append({
                "edge_id": "KGQ_" + digest_bytes("|".join(key).encode())[:16],
                "source": key[0], "target": key[1], "relation_type": key[2],
                "source_name": first["source_entity"], "target_name": first["target_entity"],
                "source_type": first["source_type"], "target_type": first["target_type"],
                "metaedge": first["metaedge"], "sign": SIGNS.get(key[2], 0),
                "pmids": pmids, "evidence_ids": sorted({r["relation_id"] for r in records}),
                "unique_supporting_pmids": len(pmids), "source_dataset": "nlp_literature",
                "tier": 3, "partition": "QUARANTINE", "quarantine_reasons": sorted(set(q_block)),
            })
            continue
        if len(pmids) < tier2_min_pmids:
            continue
        hyp_reasons = sorted({
            reason
            for r in records
            for reason in (r.get("filter_audit", {}).get("review_reasons") or [])
        } | ({"single_supporting_pmid"} if len(pmids) < policy["min_unique_pmids"] else set()))
        has_asserted = any(r.get("assertion_status") == "asserted" for r in records)
        hyp_edge = {
            "edge_id": edge_id, "source": key[0], "target": key[1], "relation_type": key[2],
            "source_name": first["source_entity"], "target_name": first["target_entity"],
            "source_type": first["source_type"], "target_type": first["target_type"],
            "metaedge": first["metaedge"], "sign": SIGNS[key[2]],
            "assertion_status": "asserted" if has_asserted else "uncertain",
            "pmids": pmids, "evidence_ids": sorted({r["relation_id"] for r in records}),
            "unique_supporting_pmids": len(pmids), "source_dataset": "nlp_literature",
            "tier": 2, "partition": "HYPOTHESIS_EDGE", "evidence_tier": "TIER_2_HYPOTHESIS",
            "validation_method": "tiered_conservative_rules_v2",
            "hypothesis_reasons": hyp_reasons,
            "study_designs": sorted({str(r.get("study_design") or "") for r in records if r.get("study_design")}),
            "publication_date_upper_bound_min": min(pub_dates) if pub_dates else None,
            "publication_date_upper_bound_max": max(pub_dates) if pub_dates else None,
        }
        hypothesis_edges.append(hyp_edge)

    # Also record quarantined edges from quarantine_records not already emitted
    q_groups = defaultdict(list)
    for record in quarantine_records:
        gkey = (str(record.get("source_standard_id", "")), str(record.get("target_standard_id", "")), str(record.get("relation_type", "")))
        if gkey not in quarantine_seen_keys:
            q_groups[gkey].append(record)
    for key, records in sorted(q_groups.items()):
        first = records[0]
        q_reasons = sorted({
            reason
            for r in records
            for reason in (r.get("filter_audit", {}).get("quarantine_reasons") or [])
        })
        quarantine_edges.append({
            "edge_id": "KGQ_" + digest_bytes("|".join(key).encode())[:16],
            "source": key[0], "target": key[1], "relation_type": key[2],
            "source_name": first.get("source_entity", ""), "target_name": first.get("target_entity", ""),
            "source_type": first.get("source_type", ""), "target_type": first.get("target_type", ""),
            "metaedge": first.get("metaedge", ""), "sign": first.get("sign"),
            "pmids": sorted({str(r.get("pmid", "")) for r in records if r.get("pmid")}),
            "evidence_ids": sorted({str(r.get("relation_id", "")) for r in records if r.get("relation_id")}),
            "unique_supporting_pmids": len({str(r.get("pmid", "")) for r in records if r.get("pmid")}),
            "source_dataset": "nlp_literature",
            "tier": 3, "partition": "QUARANTINE", "quarantine_reasons": q_reasons,
        })
    ready = [r for r in filtered if r["relation_id"] in ready_ids]
    ready_by_id = {r["relation_id"]: r for r in ready}
    for edge in edges:
        refs = [ready_by_id[eid] for eid in edge["evidence_ids"]]
        if set(edge["pmids"]) != {r["pmid"] for r in refs}:
            raise ValueError("Broken evidence/PMID referential integrity")
    nodes = {}
    for edge in edges:
        for endpoint in ("source", "target"):
            node = {"id": edge[endpoint], "canonical_name": edge[endpoint + "_name"], "type": edge[endpoint + "_type"]}
            if node["id"] in nodes and nodes[node["id"]]["type"] != node["type"]:
                raise ValueError("Conflicting node types for same identifier")
            if schema_errors(checks["entity.schema.json"], node):
                raise ValueError("Generated node failed entity schema")
            nodes[node["id"]] = node
    existing_audits = []
    for filename in ("kg_edges_contract_d.csv", "kg_edges_contract_d_native.csv", "kg_edges_review_queue.csv"):
        if filename not in bundle.names:
            continue
        for row in bundle.csv(filename):
            existing_audits.append({"file": filename, "edge_id": row.get("edge_id"),
                "source": row.get("source"), "target": row.get("target"), "relation_type": row.get("relation_type"),
                "disposition": "SUPERSEDED_BY_REFILTERED_EVIDENCE",
                "matching_new_edge": next((e["edge_id"] for e in edges if (e["source"], e["target"], e["relation_type"]) ==
                                            (row.get("source"), row.get("target"), row.get("relation_type"))), "")})
    counts = Counter(a["decision"] for a in audits)
    reasons = Counter(reason for a in audits for reason in a["reject_reasons"] + a["review_reasons"])
    pmid_counts = defaultdict(Counter)
    for audit in audits:
        pmid_counts[audit["pmid"]][audit["decision"]] += 1
    all_pmids = set(papers) | {key[0] for key in sentences}
    pmid_audit = [{"pmid": pmid, "original_source_available": pmid in papers,
                   "candidate_rows": sum(pmid_counts[pmid].values()),
                   "relations_passed": pmid_counts[pmid]["PASS_RELATION"],
                   "relations_held": pmid_counts[pmid]["REVIEW"],
                   "relations_rejected": pmid_counts[pmid]["REJECT"],
                   "status": "FILTERED" if pmid_counts[pmid] else "NO_EXTRACTED_RELATION"}
                  for pmid in sorted(all_pmids)]
    report = {"status": "FILTERED", "policy": policy, "input": str(input_path.resolve()),
        "filter_script_sha256": digest_bytes(Path(__file__).read_bytes()),
        "schema_sha256": {path.name: digest_bytes(path.read_bytes()) for path in (ROOT / "contracts").glob("*.schema.json")},
        "original_papers": str(papers_path.resolve()) if papers_path else None,
        "original_papers_sha256": digest_bytes(papers_path.read_bytes()) if papers_path else None,
        "candidate_file": candidate_file, "candidate_rows": len(candidates),
        "relation_decisions": dict(counts), "reason_counts_overlapping": dict(reasons.most_common()),
        "relations_passed": len(filtered), "relations_in_admitted_edges": len(ready),
        "admitted_edges": len(edges), "hypothesis_edges": len(hypothesis_edges),
        "quarantine_edges": len(quarantine_edges),
        "tier_summary": {
            "tier1_admitted_kg": len(edges),
            "tier2_hypothesis_edges": len(hypothesis_edges),
            "tier3_quarantine_edges": len(quarantine_edges),
            "tier1_negated_leaks": sum(e.get("assertion_status") == "negated" for e in edges),
            "tier2_negated_leaks": sum(e.get("assertion_status") == "negated" for e in hypothesis_edges),
        },
        "edge_groups_held": sum(a["decision"] == "REVIEW" for a in edge_audit),
        "nodes": len(nodes), "original_paper_records": len(papers), "candidate_pmids": len({r["pmid"] for r in candidates}),
        "documents_audited": len(pmid_audit),
        "documents_without_extracted_relations": sum(r["status"] == "NO_EXTRACTED_RELATION" for r in pmid_audit),
        "candidate_pmids_missing_originals": sorted({r["pmid"] for r in candidates} - papers.keys()),
        "input_files_checked": len(files), "input_document_schema_errors": sum(i["schema_errors"] for i in files),
        "limitations": ["Deterministic rules recognize a limited set of explicit predicates; REVIEW does not mean scientifically false.",
            "Extraction confidence is only an extraction score; no old posterior or global PubTator search is accepted as proof.",
            "Unique PMIDs are deduplicated publications, not guaranteed independent studies.",
            "Original CSV print dates are used as supplied; online-first dates, corrections, retractions and external ontology validity are not queried.",
            "No existing graph or original input file has been modified."],
        "schema_validation": {"relations": "PASS", "edges": "PASS", "nodes": "PASS", "evidence_references": "PASS"}}
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="kg_filter_", dir=output.parent) as temp:
        stage = Path(temp)
        json_dump(stage / "filter_report.json", report)
        json_dump(stage / "input_inventory.json", files)
        json_dump(stage / "policy_snapshot.json", policy)
        jsonl_dump(stage / "relations_filtered.jsonl", filtered)
        jsonl_dump(stage / "relations_kg_ready.jsonl", ready)
        jsonl_dump(stage / "relations_review.jsonl", review_records)
        jsonl_dump(stage / "relations_rejected.jsonl", rejected)
        jsonl_dump(stage / "kg_edges_contract_d.jsonl", edges)
        jsonl_dump(stage / "kg_edges_hypothesis.jsonl", hypothesis_edges)
        jsonl_dump(stage / "kg_edges_quarantine.jsonl", quarantine_edges)
        json_dump(stage / "kg_graph.json", {"nodes": list(nodes.values()), "edges": edges,
            "metadata": {"validation_method": "conservative_rules_v1", "publication_cutoff": policy["publication_cutoff"],
                         "source": "filtered_pubtator3_output", "requires_existing_graph_merge": True}})
        csv_dump(stage / "relation_audit.csv", audits, list(audits[0]) if audits else ["relation_id","decision","reasons"])
        csv_dump(stage / "edge_audit.csv", edge_audit,
                 ["edge_id", "source", "source_name", "target", "target_name", "relation_type", "sign", "pmids", "evidence_ids", "decision", "reasons"])
        csv_dump(stage / "existing_edges_audit.csv", existing_audits,
                 ["file", "edge_id", "source", "target", "relation_type", "disposition", "matching_new_edge"])
        csv_dump(stage / "pmid_audit.csv", pmid_audit,
                 ["pmid", "original_source_available", "candidate_rows", "relations_passed", "relations_held", "relations_rejected", "status"])
        # Publish only after every record and reference has been validated.
        output.mkdir(parents=True, exist_ok=True)
        for path in stage.iterdir():
            shutil.move(str(path), str(output / path.name))
    if bundle.archive:
        bundle.archive.close()
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "output_pubtator3.zip")
    parser.add_argument("--papers", type=Path, help="Original PMID/title/abstract CSV or JSONL with print_date/publication_date")
    parser.add_argument("--policy", type=Path, default=ROOT / "configs" / "kg_filter.json")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "pubtator3_filtered")
    args = parser.parse_args()
    try:
        report = run(args.input, args.output, args.papers, args.policy)
    except (ValueError, OSError, KeyError, csv.Error, json.JSONDecodeError, zipfile.BadZipFile) as exc:
        parser.exit(2, f"Filter stopped: {exc}\n")
    print(json.dumps({k: report[k] for k in ["status", "candidate_rows", "relation_decisions", "relations_passed",
          "relations_in_admitted_edges", "admitted_edges", "edge_groups_held", "nodes"]}, ensure_ascii=False, indent=2))
    print(f"Output: {args.output.resolve()}")


if __name__ == "__main__":
    main()
