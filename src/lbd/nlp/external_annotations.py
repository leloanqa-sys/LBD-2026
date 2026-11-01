"""Batch adapters for PubTator 3 and Europe PMC/SciLite annotations.

External annotations are inputs to V6 grounding, never relation assertions. The
runner stores a per-PMID cache and reports source coverage so a network failure
cannot masquerade as a successful annotation run.
"""
from __future__ import annotations

import hashlib
import json
import re
import random
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

PUBTATOR_URL = "https://www.ncbi.nlm.nih.gov/research/pubtator3-api/publications/export/biocjson"
EUROPE_PMC_URL = "https://www.ebi.ac.uk/europepmc/annotations_api/annotationsByArticleIds"
QUICKGO_URL = "https://www.ebi.ac.uk/QuickGO/services/ontology/go/terms/"

# Only crosswalks that are used by this focused V6 project. Unknown IDs retain
# their source namespace; they are never silently promoted to a local ID.
MESH_TO_V6 = {
    "C529054": ("DB06292", "Dapagliflozin", "Compound"),
    "D000069348": ("DB06292", "Dapagliflozin", "Compound"),
    "D003924": ("DOID:9352", "Type 2 Diabetes Mellitus", "Disease"),
    "D006333": ("DOID:6000", "Heart Failure", "Disease"),
    "D051436": ("DOID:784", "Chronic Kidney Disease", "Disease"),
}
GO_ID_RX = re.compile(r"(?:GO[:_])([0-9]{7})", re.I)


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    temp.replace(path)


def _request_json(url: str, timeout: int = 40, accept: str = "application/json") -> Any:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "dapagliflozin-lbd-v6/1.0 (research NLP)", "Accept": accept},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _request_json_with_retry(url: str, *, timeout: int = 40, accept: str = "application/json", attempts: int = 3) -> Any:
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            return _request_json(url, timeout=timeout, accept=accept)
        except Exception as exc:
            last_error = exc
            if attempt + 1 < attempts:
                time.sleep((2 ** attempt) + random.uniform(0.0, 0.35))
    assert last_error is not None
    raise last_error


def _pmid_from_doc(doc: dict[str, Any]) -> str:
    raw = doc.get("pmid") or doc.get("id") or doc.get("document_id") or ""
    match = re.search(r"([0-9]{5,})", str(raw))
    return match.group(1) if match else ""


def _normalize_pubtator_type(raw_type: str) -> str | None:
    key = (raw_type or "").casefold().replace(" ", "")
    return {
        "gene": "Gene",
        "geneor geneproduct": "Gene",
        "geneor gene product": "Gene",
        "disease": "Disease",
        "diseaseorphenotypicfeature": "Disease",
        "chemical": "Compound",
        "chemicalentity": "Compound",
        "mutation": "Variant",
        "variant": "Variant",
        "species": "Species",
        "cellline": "CellLine",
    }.get(key)


def _normalize_pubtator_entity(annotation: dict[str, Any], passage_offset: int) -> dict[str, Any] | None:
    infons = annotation.get("infons") or {}
    raw_type = str(infons.get("type") or "")
    entity_type = _normalize_pubtator_type(raw_type)
    raw_identifier = str(infons.get("identifier") or "").strip()
    text = str(annotation.get("text") or "").strip()
    if not entity_type or not raw_identifier or not text:
        return None

    identifier = raw_identifier
    canonical_name = text
    mesh_id = raw_identifier.removeprefix("MESH:")
    if mesh_id in MESH_TO_V6:
        identifier, canonical_name, mapped_type = MESH_TO_V6[mesh_id]
        entity_type = mapped_type
    elif entity_type == "Gene" and raw_identifier.isdigit():
        identifier = raw_identifier
    elif raw_identifier.startswith("MESH:"):
        identifier = raw_identifier
    elif entity_type == "Compound" and raw_identifier.startswith("CHEBI:"):
        identifier = raw_identifier

    locations = annotation.get("locations") or []
    start = end = None
    if locations:
        try:
            loc = locations[0]
            raw_start = int(loc["offset"])
            # BioC location offsets from PubTator3 are document-absolute. Older
            # variants can be passage-relative, so add passage offset only when
            # the reported location precedes the passage's document offset.
            start = raw_start if raw_start >= passage_offset else raw_start + passage_offset
            end = start + int(loc["length"])
        except (KeyError, TypeError, ValueError):
            start = end = None
    return {
        "text": text,
        "canonical_name": canonical_name,
        "type": entity_type,
        "id": identifier,
        "raw_identifier": raw_identifier,
        "source": "PubTator3",
        "start": start,
        "end": end,
    }


def _parse_pubtator(payload: Any) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    documents = payload.get("PubTator3", []) if isinstance(payload, dict) else []
    if not isinstance(documents, list):
        return result
    for doc in documents:
        if not isinstance(doc, dict):
            continue
        pmid = _pmid_from_doc(doc)
        if not pmid:
            continue
        result.setdefault(pmid, [])
        for passage in doc.get("passages", []) or []:
            try:
                passage_offset = int(passage.get("offset", 0))
            except (TypeError, ValueError):
                passage_offset = 0
            for annotation in passage.get("annotations", []) or []:
                if not isinstance(annotation, dict):
                    continue
                entity = _normalize_pubtator_entity(annotation, passage_offset)
                if entity:
                    result[pmid].append(entity)
    return result


def _cache_path(cache_root: Path, provider: str, pmid: str) -> Path:
    return cache_root / provider / f"{pmid}.json"


def _load_provider_cache(cache_root: Path, provider: str, pmids: list[str]) -> tuple[dict[str, list[dict[str, Any]]], list[str]]:
    found: dict[str, list[dict[str, Any]]] = {}
    missing: list[str] = []
    for pmid in pmids:
        value = _read_json(_cache_path(cache_root, provider, pmid))
        if isinstance(value, list):
            found[pmid] = value
        else:
            missing.append(pmid)
    return found, missing


def _save_provider_cache(cache_root: Path, provider: str, values: dict[str, list[dict[str, Any]]]) -> None:
    for pmid, annotations in values.items():
        _write_json(_cache_path(cache_root, provider, pmid), annotations)


def fetch_pubtator(pmids: list[str], cache_root: Path, *, offline: bool = False, batch_size: int = 40) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    pmids = sorted({str(p).strip() for p in pmids if str(p).strip().isdigit()})
    # v2 invalidates old caches that incorrectly added passage offsets to
    # already document-absolute BioC locations.
    values, missing = _load_provider_cache(cache_root, "pubtator3_bioc_v2", pmids)
    stats = {"provider": "PubTator3", "requested_pmids": len(pmids), "cache_hits": len(values), "api_batches": 0, "api_failures": [], "annotated_pmids": 0, "annotations": 0}
    if not offline:
        for start in range(0, len(missing), batch_size):
            batch = missing[start:start + batch_size]
            params = urllib.parse.urlencode({"pmids": ",".join(batch)})
            try:
                payload = _request_json_with_retry(f"{PUBTATOR_URL}?{params}")
                parsed = _parse_pubtator(payload)
                for pmid in batch:
                    values[pmid] = parsed.get(pmid, [])
                _save_provider_cache(cache_root, "pubtator3_bioc_v2", {p: values[p] for p in batch})
                stats["api_batches"] += 1
            except Exception as exc:  # preserve explicit failure per batch
                stats["api_failures"].append({"pmids": batch, "error": f"{type(exc).__name__}: {exc}"})
    stats["annotated_pmids"] = sum(bool(items) for items in values.values())
    stats["annotations"] = sum(len(items) for items in values.values())
    stats["unavailable_pmids"] = [p for p in pmids if p not in values]
    return values, stats


def _collect_strings(value: Any) -> list[str]:
    found: list[str] = []
    if isinstance(value, str):
        found.append(value)
    elif isinstance(value, dict):
        for key in ("id", "uri", "value", "name", "label"):
            if key in value:
                found.extend(_collect_strings(value[key]))
    elif isinstance(value, list):
        for item in value:
            found.extend(_collect_strings(item))
    return found


def _find_go_id(annotation: dict[str, Any]) -> str | None:
    for key in ("body", "tags", "target", "id", "uri"):
        for value in _collect_strings(annotation.get(key)):
            match = GO_ID_RX.search(value)
            if match:
                return f"GO:{match.group(1)}"
    return None


def _selector(annotation: dict[str, Any]) -> tuple[str, str, str]:
    target = annotation.get("target")
    if isinstance(target, list):
        target = target[0] if target else {}
    target = target if isinstance(target, dict) else {}
    selector = target.get("selector") or annotation.get("selector") or {}
    if isinstance(selector, list):
        selector = next((s for s in selector if isinstance(s, dict) and s.get("exact")), {})
    selector = selector if isinstance(selector, dict) else {}
    exact = str(selector.get("exact") or annotation.get("exact") or annotation.get("text") or "").strip()
    prefix = str(selector.get("prefix") or "")
    suffix = str(selector.get("suffix") or "")
    return exact, prefix, suffix


def _parse_europepmc_article(article: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    pmid = _pmid_from_doc(article)
    if not pmid and str(article.get("source", "")).upper() == "MED":
        pmid = str(article.get("extId") or "")
    annotations: list[dict[str, Any]] = []
    for raw in article.get("annotations", []) or []:
        if not isinstance(raw, dict):
            continue
        go_id = _find_go_id(raw)
        if not go_id:
            continue
        exact, prefix, suffix = _selector(raw)
        exact = exact or str(raw.get("exact") or "").strip()
        prefix = prefix or str(raw.get("prefix") or "")
        suffix = suffix or str(raw.get("postfix") or "")
        if not exact:
            continue
        body = raw.get("body")
        name = ""
        if isinstance(body, dict):
            name = str(body.get("label") or body.get("name") or "")
        for tag in raw.get("tags", []) or []:
            if isinstance(tag, dict) and (go_id.replace(":", "_") in str(tag.get("uri") or tag.get("id") or "")):
                name = name or str(tag.get("name") or tag.get("label") or "")
        target = raw.get("target")
        if isinstance(target, list):
            target = target[0] if target else {}
        target = target if isinstance(target, dict) else {}
        annotations.append({
            "text": exact,
            "canonical_name": name,
            "type": "GO_UNRESOLVED_ASPECT",
            "id": go_id,
            "source": "EuropePMC_SciLite",
            "provider": str(raw.get("creator") or raw.get("provider") or "EuropePMC"),
            "section": str(raw.get("section") or target.get("isPartOf") or ""),
            "prefix": prefix,
            "suffix": suffix,
        })
    return pmid, annotations


def fetch_europepmc_go(pmids: list[str], cache_root: Path, *, offline: bool = False, batch_size: int = 8) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    pmids = sorted({str(p).strip() for p in pmids if str(p).strip().isdigit()})
    values, missing = _load_provider_cache(cache_root, "europepmc", pmids)
    stats = {"provider": "EuropePMC_SciLite", "requested_pmids": len(pmids), "cache_hits": len(values), "api_batches": 0, "api_failures": [], "annotated_pmids": 0, "go_annotations": 0}
    if not offline:
        for start in range(0, len(missing), batch_size):
            batch = missing[start:start + batch_size]
            params = urllib.parse.urlencode({"articleIds": ",".join(f"MED:{p}" for p in batch), "format": "JSON"})
            try:
                payload = _request_json_with_retry(f"{EUROPE_PMC_URL}?{params}")
                articles = payload if isinstance(payload, list) else payload.get("articles", []) if isinstance(payload, dict) else []
                parsed: dict[str, list[dict[str, Any]]] = {p: [] for p in batch}
                for article in articles:
                    if not isinstance(article, dict):
                        continue
                    if isinstance(article.get("annotations"), list):
                        pmid, annotations = _parse_europepmc_article(article)
                    else:
                        pmid = str(article.get("extId") or "") if str(article.get("source", "")).upper() == "MED" else _pmid_from_doc(article)
                        annotations = []
                    if pmid and isinstance(article.get("annotations"), list):
                        nested_annotations = []
                        for raw_annotation in article["annotations"]:
                            if isinstance(raw_annotation, dict):
                                nested_annotations.append(raw_annotation)
                        annotations = []
                        for raw_annotation in nested_annotations:
                            go_id = _find_go_id(raw_annotation)
                            exact = str(raw_annotation.get("exact") or "").strip()
                            if not go_id or not exact:
                                continue
                            annotations.append({"text": exact, "canonical_name": "", "type": "GO_UNRESOLVED_ASPECT", "id": go_id,
                                "source": "EuropePMC_SciLite", "provider": str(raw_annotation.get("provider") or "EuropePMC"),
                                "section": str(raw_annotation.get("section") or ""), "prefix": str(raw_annotation.get("prefix") or ""),
                                "suffix": str(raw_annotation.get("postfix") or "")})
                    if pmid in parsed:
                        parsed[pmid].extend(annotations)
                values.update(parsed)
                _save_provider_cache(cache_root, "europepmc", parsed)
                stats["api_batches"] += 1
            except Exception as exc:
                stats["api_failures"].append({"pmids": batch, "error": f"{type(exc).__name__}: {exc}"})
    stats["annotated_pmids"] = sum(bool(items) for items in values.values())
    stats["go_annotations"] = sum(len(items) for items in values.values())
    stats["unavailable_pmids"] = [p for p in pmids if p not in values]
    return values, stats


def _aspect_key(raw: Any) -> str:
    value = str(raw or "").strip().casefold()
    if value in {"biological_process", "biological process", "p", "bp"}:
        return "biological_process"
    if value in {"molecular_function", "molecular function", "f", "mf"}:
        return "molecular_function"
    if value in {"cellular_component", "cellular component", "c", "cc"}:
        return "cellular_component"
    return "unknown"


def resolve_go_aspects(annotations_by_pmid: dict[str, list[dict[str, Any]]], cache_root: Path, *, offline: bool = False, batch_size: int = 80) -> dict[str, Any]:
    ids = sorted({a["id"] for items in annotations_by_pmid.values() for a in items if a.get("id")})
    cache_path = cache_root / "quickgo" / "go_aspects.json"
    aspects = _read_json(cache_path) or {}
    initially_missing = [go_id for go_id in ids if go_id not in aspects]
    missing = list(initially_missing)
    failures: list[str] = []
    if not offline:
        for start in range(0, len(missing), batch_size):
            batch = missing[start:start + batch_size]
            ids_param = urllib.parse.quote(",".join(batch), safe=":,")
            url = f"{QUICKGO_URL}{ids_param}?limit={len(batch)}"
            try:
                payload = _request_json_with_retry(url, timeout=40, accept="application/json")
                for term in payload.get("results", []) or []:
                    term_id = str(term.get("id") or "")
                    aspect = _aspect_key(term.get("aspect") or term.get("ontology"))
                    if term_id and aspect != "unknown":
                        aspects[term_id] = {"aspect": aspect, "name": term.get("name", ""), "obsolete": bool(term.get("isObsolete", False))}
            except Exception as exc:
                failures.append(f"{type(exc).__name__}: {exc}")
        _write_json(cache_path, aspects)
    bp_count = mf_count = cc_count = unresolved = 0
    for items in annotations_by_pmid.values():
        for annotation in items:
            meta = aspects.get(annotation.get("id"), {})
            aspect = _aspect_key(meta.get("aspect"))
            annotation["ontology_aspect"] = aspect
            annotation["canonical_name"] = meta.get("name") or annotation.get("canonical_name") or annotation["id"]
            annotation["type"] = "BiologicalProcess" if aspect == "biological_process" else "GO_OTHER"
            annotation["ontology_obsolete"] = bool(meta.get("obsolete", False))
            if aspect == "biological_process":
                bp_count += 1
            elif aspect == "molecular_function":
                mf_count += 1
            elif aspect == "cellular_component":
                cc_count += 1
            else:
                unresolved += 1
    unresolved_ids = [go_id for go_id in ids if go_id not in aspects]
    return {"unique_go_ids": len(ids), "go_metadata_cache_hits": len(ids) - len(initially_missing), "go_metadata_fetched": len(ids) - len(unresolved_ids) - (len(ids) - len(initially_missing)), "go_metadata_missing": unresolved_ids, "aspect_counts": {"biological_process": bp_count, "molecular_function": mf_count, "cellular_component": cc_count, "unknown": unresolved}, "quickgo_failures": failures}


def annotate_spans(doc: dict[str, Any], annotations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Resolve quote-only Europe PMC selectors against the supplied title/abstract."""
    title = str(doc.get("title") or "")
    abstract = str(doc.get("abstract") or "")
    text = f"{title} {abstract}" if title else abstract
    for annotation in annotations:
        if annotation.get("start") is not None and annotation.get("end") is not None:
            start, end = int(annotation["start"]), int(annotation["end"])
            if 0 <= start < end <= len(text) and text[start:end].casefold() == str(annotation.get("text") or "").casefold():
                continue
            # Recover spans from older caches or providers with passage-relative offsets.
            annotation.pop("start", None)
            annotation.pop("end", None)
        exact = str(annotation.get("text") or "")
        if not exact:
            continue
        prefix = str(annotation.get("prefix") or "")
        suffix = str(annotation.get("suffix") or "")
        start = 0
        matched = None
        while True:
            index = text.casefold().find(exact.casefold(), start)
            if index < 0:
                break
            before = text[max(0, index - len(prefix)):index] if prefix else ""
            after = text[index + len(exact):index + len(exact) + len(suffix)] if suffix else ""
            if (not prefix or before.casefold().endswith(prefix.casefold())) and (not suffix or after.casefold().startswith(suffix.casefold())):
                matched = (index, index + len(exact))
                break
            start = index + max(1, len(exact))
        if matched is not None:
            annotation["start"], annotation["end"] = matched
        annotation.pop("prefix", None)
        annotation.pop("suffix", None)
    return annotations


def source_manifest(cache_root: Path, pmids: list[str], pubtator_stats: dict[str, Any], europepmc_stats: dict[str, Any], go_stats: dict[str, Any]) -> dict[str, Any]:
    return {
        "retrieved_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "requested_pmids_sha256": hashlib.sha256("\n".join(sorted(pmids)).encode("utf-8")).hexdigest(),
        "cache_root": str(cache_root),
        "pubtator3": pubtator_stats,
        "europepmc_scilite": europepmc_stats,
        "go_aspects": go_stats,
    }
