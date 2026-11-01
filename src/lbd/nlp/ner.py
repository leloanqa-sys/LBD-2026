"""Named Entity Recognition (NER) Client via NIH PubTator 3.0 API and Local Biomedical Lexicon."""
from __future__ import annotations

import json
import logging
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
    # Diseases
    "D006333": {"id": "DOID:6000", "name": "Heart Failure", "type": "Disease"},
    "D051436": {"id": "DOID:784", "name": "Chronic Kidney Disease", "type": "Disease"},
    "D003924": {"id": "DOID:9352", "name": "Type 2 Diabetes Mellitus", "type": "Disease"},
    "D000419": {"id": "HP:0012592", "name": "Albuminuria", "type": "Phenotype"},
    "D003404": {"id": "MESH:D003404", "name": "Creatinine", "type": "Compound"},
    "D009203": {"id": "DOID:5844", "name": "Myocardial Infarction", "type": "Disease"},
    "D006973": {"id": "DOID:10763", "name": "Hypertension", "type": "Disease"},
}

LOCAL_KEYWORD_MAP = [
    (r"\b(?:dapagliflozin|farxiga|forxiga)\b", "Dapagliflozin", "Compound", "DB06292"),
    (r"\b(?:sglt[\s\-]?2|slc5a2)\b", "SLC5A2", "Gene", "6524"),
    (r"\b(?:sglt[\s\-]?1|slc5a1)\b", "SLC5A1", "Gene", "6523"),
    (r"\b(?:heart\s+failure|hf|hhf)\b", "Heart Failure", "Disease", "DOID:6000"),
    (r"\b(?:chronic\s+kidney\s+disease|ckd|chronic\s+renal\s+failure)\b", "Chronic Kidney Disease", "Disease", "DOID:784"),
    (r"\b(?:type\s+2\s+diabetes|t2dm|t2d)\b", "Type 2 Diabetes", "Disease", "DOID:9352"),
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
                    entities.append({
                        "text": raw_text,
                        "canonical_name": canonical_name,
                        "type": ent_type,
                        "id": norm_id,
                        "raw_identifier": raw_id,
                        "source": "PubTator3",
                        "start": start,
                        "end": end,
                    })
            return entities
    except Exception as e:
        logger.warning(f"PubTator 3.0 API call failed for PMID {pmid}: {e}")
        return []


def annotate_abstract_entities(pmid: str, abstract_text: str) -> List[Dict[str, Any]]:
    """Annotates abstract using PubTator 3.0 API with local dictionary fallback."""
    pubtator_ents = annotate_abstract_pubtator_live(pmid)
    if pubtator_ents:
        return pubtator_ents

    entities = []
    text_lower = abstract_text.lower()
    for pat, name, etype, eid in LOCAL_KEYWORD_MAP:
        for m in re.finditer(pat, text_lower):
            entities.append({
                "text": m.group(0),
                "canonical_name": name,
                "type": etype,
                "id": eid,
                "raw_identifier": eid,
                "source": "LocalBiomedicalLexicon",
                "start": m.start(),
                "end": m.end(),
            })
    return entities


# Aliases
annotate_abstract_pubtator = annotate_abstract_entities
annotate_abstract = annotate_abstract_entities
annotate_abstract_hybrid = annotate_abstract_entities
