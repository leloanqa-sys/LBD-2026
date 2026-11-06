"""Biomedical Entity and Term Normalization.

Owner: Đặng Quang Huy (Data Engineering & Normalization Pipeline)
Co-owner: Nguyễn Khang (NLP Entity Grounding)

Maps raw surface mentions from PubMed abstracts into canonical entity names and
resolves them to Hetionet v1.0 identifiers via `lbd.common.ids.resolve_identifier`.
"""
from typing import Any, Dict
from lbd.common.ids import resolve_identifier

SYNONYM_MAP: Dict[str, str] = {
    # SGLT2 inhibitors & targets
    "dapa": "Dapagliflozin",
    "forxiga": "Dapagliflozin",
    "farxiga": "Dapagliflozin",
    "bms-512148": "Dapagliflozin",
    "canagliflozin": "Canagliflozin",
    "invokana": "Canagliflozin",
    "empagliflozin": "Empagliflozin",
    "jardiance": "Empagliflozin",
    "sglt2": "SLC5A2",
    "sglt-2": "SLC5A2",
    "sodium-glucose cotransporter 2": "SLC5A2",
    "slc5a2": "SLC5A2",
    "sglt1": "SLC5A1",
    "slc5a1": "SLC5A1",
    # Ketamine & NMDA / BDNF axis
    "ketamine": "Ketamine",
    "esketamine": "Ketamine",
    "spravato": "Ketamine",
    "grin1": "GRIN1",
    "nmdar1": "GRIN1",
    "grin2a": "GRIN2A",
    "nmdar2a": "GRIN2A",
    "grin2b": "GRIN2B",
    "bdnf": "BDNF",
    "brain-derived neurotrophic factor": "BDNF",
    # Colchicine & NLRP3 / IL1B inflammasome axis
    "colchicine": "Colchicine",
    "lodoco": "Colchicine",
    "nlrp3": "NLRP3",
    "nlrp3 inflammasome": "NLRP3",
    "il1b": "IL1B",
    "il-1b": "IL1B",
    "il-1beta": "IL1B",
    "interleukin-1 beta": "IL1B",
    "tubulin": "TUBB",
    # GLP-1 & Obesity axis
    "liraglutide": "Liraglutide",
    "saxenda": "Liraglutide",
    "victoza": "Liraglutide",
    "semaglutide": "Liraglutide",
    "wegovy": "Liraglutide",
    "ozempic": "Liraglutide",
    "glp1r": "GLP1R",
    "glp-1r": "GLP1R",
    "glp-1 receptor": "GLP1R",
    # Diseases
    "ckd": "Chronic Kidney Disease",
    "diabetic nephropathy": "Chronic Kidney Disease",
    "renal failure": "Chronic Kidney Disease",
    "chronic kidney failure": "Chronic Kidney Disease",
    "hf": "Heart Failure",
    "cardiac failure": "Heart Failure",
    "heart failure": "Heart Failure",
    "dilated cardiomyopathy": "Dilated Cardiomyopathy",
    "t2d": "Type 2 Diabetes",
    "type 2 diabetes": "Type 2 Diabetes",
    "mdd": "Major Depressive Disorder",
    "depression": "Major Depressive Disorder",
    "major depressive disorder": "Major Depressive Disorder",
    "endogenous depression": "Major Depressive Disorder",
    "cad": "Coronary Artery Disease",
    "coronary artery disease": "Coronary Artery Disease",
    "atherosclerosis": "Atherosclerosis",
    "obesity": "Obesity",
}


def normalize_entity_name(raw_name: str) -> str:
    """Normalizes a raw entity surface string to its canonical biomedical name."""
    cleaned = raw_name.strip().lower()
    return SYNONYM_MAP.get(cleaned, raw_name.strip())


def normalize_relation_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """Normalizes both source and target entities in an extracted relation record."""
    norm_source = normalize_entity_name(record.get("source_entity", ""))
    norm_target = normalize_entity_name(record.get("target_entity", ""))

    src_id = record.get("source_standard_id") or resolve_identifier(norm_source)
    tgt_id = record.get("target_standard_id") or resolve_identifier(norm_target)

    status = "verified" if (src_id and tgt_id) else "unresolved"

    updated = dict(record)
    updated["source_entity"] = norm_source
    updated["target_entity"] = norm_target
    updated["source_standard_id"] = src_id
    updated["target_standard_id"] = tgt_id
    updated["normalization_status"] = status
    return updated
