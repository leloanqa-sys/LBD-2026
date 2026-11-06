"""Named Entity Recognition (NER) Client via NIH PubTator 3.0 API and Local Biomedical Lexicon.

Owner: Nguyễn Khang (Biomedical NLP & BioRED Engine)
Co-owner: Đặng Quang Huy (ID Crosswalk & Normalization)

Provides online PubTator 3.0 entity extraction with automatic fallback to a
high-precision local regex lexicon (`LOCAL_KEYWORD_MAP`) covering all landmark
drugs, target genes, and Disease Ontology (DOID) concepts for offline execution.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
import re
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

logger = logging.getLogger("nlp_ner")

# Crosswalk mappings from MeSH / NCBI to Hetionet / DOID / GO
MESH_TO_STANDARD: Dict[str, Dict[str, str]] = {
    # Compounds
    "C529054": {"id": "DB06292", "name": "Dapagliflozin", "type": "Compound"},
    "D000069348": {"id": "DB06292", "name": "Dapagliflozin", "type": "Compound"},
    "D000068896": {"id": "DB08907", "name": "Canagliflozin", "type": "Compound"},
    "D007649": {"id": "DB01221", "name": "Ketamine", "type": "Compound"},
    "D003078": {"id": "DB01394", "name": "Colchicine", "type": "Compound"},
    "D000068718": {"id": "DB06655", "name": "Liraglutide", "type": "Compound"},
    "D015282": {"id": "DB00104", "name": "Octreotide", "type": "Compound"},
    # Diseases
    "D006333": {"id": "DOID:6000", "name": "Heart Failure", "type": "Disease"},
    "D002311": {"id": "DOID:12930", "name": "Dilated Cardiomyopathy", "type": "Disease"},
    "D051436": {"id": "DOID:784", "name": "Chronic Kidney Disease", "type": "Disease"},
    "D003924": {"id": "DOID:9352", "name": "Type 2 Diabetes Mellitus", "type": "Disease"},
    "D003865": {"id": "DOID:1595", "name": "Major Depressive Disorder", "type": "Disease"},
    "D003324": {"id": "DOID:3393", "name": "Coronary Artery Disease", "type": "Disease"},
    "D050197": {"id": "DOID:1936", "name": "Atherosclerosis", "type": "Disease"},
    "D009765": {"id": "DOID:9970", "name": "Obesity", "type": "Disease"},
    "D000419": {"id": "HP:0012592", "name": "Albuminuria", "type": "Phenotype"},
    "D003404": {"id": "MESH:D003404", "name": "Creatinine", "type": "Compound"},
    "D009203": {"id": "DOID:5844", "name": "Myocardial Infarction", "type": "Disease"},
    "D006973": {"id": "DOID:10763", "name": "Hypertension", "type": "Disease"},
}

_CROSSWALK_FILE = Path(__file__).resolve().parent / "crosswalk_mesh_doid.json"
if _CROSSWALK_FILE.exists():
    try:
        _cw = json.loads(_CROSSWALK_FILE.read_text(encoding="utf-8"))
        for _k, _v in _cw.items():
            if _k not in MESH_TO_STANDARD:
                MESH_TO_STANDARD[_k] = {"id": _v["id"], "name": _v["name"], "type": _v["type"]}
    except Exception:
        pass

LOCAL_KEYWORD_MAP = [
    # Compounds
    (r"\b(?:dapagliflozin|farxiga|forxiga)\b", "Dapagliflozin", "Compound", "DB06292"),
    (r"\b(?:canagliflozin|invokana)\b", "Canagliflozin", "Compound", "DB08907"),
    (r"\b(?:empagliflozin|jardiance)\b", "Empagliflozin", "Compound", "DB09038"),
    (r"\b(?:ketamine|esketamine|spravato)\b", "Ketamine", "Compound", "DB01221"),
    (r"\b(?:colchicine|lodoco)\b", "Colchicine", "Compound", "DB01394"),
    (r"\b(?:liraglutide|saxenda|victoza|semaglutide|wegovy)\b", "Liraglutide", "Compound", "DB06655"),
    (r"\b(?:octreotide|sandostatin)\b", "Octreotide", "Compound", "DB00104"),
    # Genes
    (r"\b(?:sglt[\s\-]?2|slc5a2)\b", "SLC5A2", "Gene", "6524"),
    (r"\b(?:sglt[\s\-]?1|slc5a1)\b", "SLC5A1", "Gene", "6523"),
    (r"\b(?:grin1|nmdar1|glun1)\b", "GRIN1", "Gene", "2902"),
    (r"\b(?:grin2a|nmdar2a|glun2a)\b", "GRIN2A", "Gene", "2903"),
    (r"\b(?:grin2b|nmdar2b|glun2b)\b", "GRIN2B", "Gene", "2904"),
    (r"\b(?:bdnf|brain[\s\-]derived\s+neurotrophic\s+factor)\b", "BDNF", "Gene", "627"),
    (r"\b(?:nlrp3|cias1|nalp3)\b", "NLRP3", "Gene", "114548"),
    (r"\b(?:il1b|il[\s\-]?1\s*beta|interleukin[\s\-]?1\s*beta)\b", "IL1B", "Gene", "3553"),
    (r"\b(?:tubb|beta[\s\-]?tubulin)\b", "TUBB", "Gene", "203068"),
    (r"\b(?:glp[\s\-]?1r|glp1r)\b", "GLP1R", "Gene", "2740"),
    # Diseases & Phenotypes
    (r"\b(?:heart\s+failure|hf|hhf)\b", "Heart Failure", "Disease", "DOID:6000"),
    (r"\b(?:dilated\s+cardiomyopathy|dcm)\b", "Dilated Cardiomyopathy", "Disease", "DOID:12930"),
    (r"\b(?:chronic\s+kidney\s+disease|ckd|chronic\s+renal\s+failure)\b", "Chronic Kidney Disease", "Disease", "DOID:784"),
    (r"\b(?:type\s+2\s+diabetes|t2dm|t2d)\b", "Type 2 Diabetes", "Disease", "DOID:9352"),
    (r"\b(?:major\s+depressive\s+disorder|endogenous\s+depression|treatment[\s\-]resistant\s+depression)\b", "Major Depressive Disorder", "Disease", "DOID:1595"),
    (r"\b(?:coronary\s+artery\s+disease|cad)\b", "Coronary Artery Disease", "Disease", "DOID:3393"),
    (r"\b(?:atherosclerosis|atherosclerotic\s+plaque)\b", "Atherosclerosis", "Disease", "DOID:1936"),
    (r"\b(?:obesity|adiposity)\b", "Obesity", "Disease", "DOID:9970"),
    (r"\b(?:albuminuria|proteinuria)\b", "Albuminuria", "Phenotype", "HP:0012592"),
]


def annotate_abstract_pubtator_live(pmid: str, timeout: int = 8) -> List[Dict[str, Any]]:
    """Fetches real biomedical annotations from NIH PubTator 3.0 API."""
    if not pmid or not str(pmid).isdigit():
        return []

    url = f"https://www.ncbi.nlm.nih.gov/research/pubtator3-api/publications/export/biocjson?pmids={pmid}"
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "DrugRepurposingPipeline/2.0 (Biomedical NLP Research)"},
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            docs = data.get("PubTator3", [])
            if not docs:
                return []

            entities = []
            for passage in docs[0].get("passages", []):
                try:
                    passage_offset = int(passage.get("offset", 0))
                except (TypeError, ValueError):
                    passage_offset = 0
                for a in passage.get("annotations", []):
                    raw_text = a.get("text", "")
                    inf = a.get("infons", {})
                    ent_type = inf.get("type", "Unknown")
                    raw_id = inf.get("identifier", "")

                    norm_id = raw_id
                    canonical_name = raw_text

                    if ent_type == "Gene" and raw_id.isdigit():
                        norm_id = raw_id
                    elif raw_id.startswith("MESH:") or raw_id in MESH_TO_STANDARD:
                        mesh_key = raw_id.replace("MESH:", "")
                        if mesh_key in MESH_TO_STANDARD:
                            std = MESH_TO_STANDARD[mesh_key]
                            norm_id = std["id"]
                            canonical_name = std["name"]
                            ent_type = std["type"]

                    start = end = None
                    locations = a.get("locations") or []
                    if locations:
                        try:
                            location = locations[0]
                            raw_start = int(location["offset"])
                            start = raw_start if raw_start >= passage_offset else raw_start + passage_offset
                            end = start + int(location["length"])
                        except (KeyError, TypeError, ValueError):
                            start = end = None

                    if norm_id and ent_type in {"Chemical", "Disease", "Gene", "Compound", "Phenotype"}:
                        if ent_type == "Chemical":
                            ent_type = "Compound"
                        entities.append({
                            "text": raw_text,
                            "canonical_name": canonical_name,
                            "type": ent_type,
                            "standard_id": norm_id,
                            "source": "PubTator3_API",
                            "start": start,
                            "end": end,
                        })
            return entities
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, KeyError) as exc:
        logger.debug("PubTator API fallback for PMID %s: %s", pmid, exc)
        return []


def annotate_sentence_local(sentence: str) -> List[Dict[str, Any]]:
    """High-precision regex fallback for offline execution or unindexed sentences."""
    found = []
    for pattern, name, ent_type, std_id in LOCAL_KEYWORD_MAP:
        for match in re.finditer(pattern, sentence, flags=re.IGNORECASE):
            found.append({
                "text": match.group(0),
                "canonical_name": name,
                "type": ent_type,
                "standard_id": std_id,
                "source": "Local_Lexicon_Fallback",
                "start": match.start(),
                "end": match.end(),
            })
    return found


def extract_entities_for_abstract(pmid: str, full_text: str, use_live_api: bool = True) -> List[Dict[str, Any]]:
    """Hybrid NER: Queries PubTator 3.0 first, supplements with local domain lexicon."""
    entities = []
    if use_live_api:
        entities = annotate_abstract_pubtator_live(pmid)

    local_ents = annotate_sentence_local(full_text)
    seen_spans = {(e.get("start"), e.get("end"), e["standard_id"]) for e in entities}
    for le in local_ents:
        key = (le.get("start"), le.get("end"), le["standard_id"])
        if key not in seen_spans:
            entities.append(le)
            seen_spans.add(key)

    return entities


# Backward-compatible aliases exported by `lbd.nlp.__init__`
annotate_abstract_hybrid = extract_entities_for_abstract
annotate_abstract_entities = extract_entities_for_abstract
annotate_abstract = extract_entities_for_abstract
annotate_abstract_pubtator = annotate_abstract_pubtator_live

