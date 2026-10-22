"""Biomedical Entity and Term Normalization."""
from typing import Dict, Any, Optional
from lbd.common.ids import resolve_identifier

SYNONYM_MAP = {
    "dapa": "Dapagliflozin",
    "forxiga": "Dapagliflozin",
    "farxiga": "Dapagliflozin",
    "bms-512148": "Dapagliflozin",
    "sglt2": "SLC5A2",
    "sglt-2": "SLC5A2",
    "sodium-glucose cotransporter 2": "SLC5A2",
    "slc5a2": "SLC5A2",
    "ckd": "Chronic Kidney Disease",
    "diabetic nephropathy": "Chronic Kidney Disease",
    "renal failure": "Chronic Kidney Disease",
    "hf": "Heart Failure",
    "cardiac failure": "Heart Failure",
    "heart failure": "Heart Failure",
    "t2d": "Type 2 Diabetes",
    "type 2 diabetes": "Type 2 Diabetes"
}

def normalize_entity_name(raw_name: str) -> str:
    cleaned = raw_name.strip().lower()
    return SYNONYM_MAP.get(cleaned, raw_name.strip())

def normalize_relation_record(record: Dict[str, Any]) -> Dict[str, Any]:
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
