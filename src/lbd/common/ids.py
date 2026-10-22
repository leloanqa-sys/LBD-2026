"""Biomedical ID Normalization and Crosswalk Mapping."""
from typing import Dict, Optional

# Canonical identifiers for core study entities
ID_CROSSWALK: Dict[str, Dict[str, str]] = {
    "Dapagliflozin": {
        "hetionet_id": "DB06292",
        "pubchem_cid": "9887712",
        "mesh_id": "D000069348",
        "type": "Compound"
    },
    "SLC5A2": {
        "hetionet_id": "6524",
        "entrez_id": "6524",
        "symbol": "SLC5A2",
        "alias": "SGLT2",
        "type": "Gene"
    },
    "Heart Failure": {
        "doid": "DOID:9333",
        "mesh_id": "D006333",
        "canonical_name": "Heart Failure",
        "type": "Disease"
    },
    "Chronic Kidney Disease": {
        "doid": "DOID:784",
        "mesh_id": "D051436",
        "canonical_name": "Chronic Kidney Disease",
        "type": "Disease"
    },
    "Type 2 Diabetes": {
        "doid": "DOID:9352",
        "mesh_id": "D003924",
        "canonical_name": "Type 2 Diabetes Mellitus",
        "type": "Disease"
    },
    "NAFLD": {
        "doid": "DOID:0080358",
        "canonical_name": "Non-alcoholic fatty liver disease",
        "type": "Disease"
    }
}

def resolve_identifier(name: str) -> Optional[str]:
    name_clean = name.strip()
    for entity, mapping in ID_CROSSWALK.items():
        if name_clean.lower() == entity.lower():
            return mapping.get("hetionet_id") or mapping.get("doid") or mapping.get("mesh_id")
        if "alias" in mapping and name_clean.lower() == mapping["alias"].lower():
            return mapping.get("hetionet_id")
    return None
