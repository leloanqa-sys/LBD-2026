"""Biomedical ID Normalization and Crosswalk Mapping.

Owner: Đặng Quang Huy (Data Engineering, ID Crosswalk & Session KG Ingestion)
Co-owner: Nguyễn Khang (Biomedical NER Entity Grounding)

Provides canonical Hetionet v1.0 identifier resolution (DrugBank Compound IDs,
NCBI Entrez Gene IDs, and Disease Ontology DOIDs) across all landmark drug
repurposing classes and prospective test suites.
"""
from typing import Dict, Optional

# Canonical identifiers for core study entities and multi-drug benchmarks
ID_CROSSWALK: Dict[str, Dict[str, str]] = {
    # --- Compounds (DrugBank IDs in Hetionet v1.0) ---
    "Dapagliflozin": {
        "hetionet_id": "DB06292",
        "pubchem_cid": "9887712",
        "mesh_id": "D000069348",
        "type": "Compound",
    },
    "Canagliflozin": {
        "hetionet_id": "DB08907",
        "mesh_id": "D000068896",
        "type": "Compound",
    },
    "Empagliflozin": {
        "hetionet_id": "DB09038",
        "mesh_id": "C570240",
        "type": "Compound",
    },
    "Ketamine": {
        "hetionet_id": "DB01221",
        "mesh_id": "D007649",
        "alias": "Esketamine",
        "type": "Compound",
    },
    "Colchicine": {
        "hetionet_id": "DB01394",
        "mesh_id": "D003078",
        "alias": "Lodoco",
        "type": "Compound",
    },
    "Liraglutide": {
        "hetionet_id": "DB06655",
        "mesh_id": "D000068718",
        "type": "Compound",
    },
    "Semaglutide": {
        "hetionet_id": "DB13928",
        "mesh_id": "C000591009",
        "type": "Compound",
    },
    "Duloxetine": {
        "hetionet_id": "Compound::DB00476",
        "drugbank_id": "DB00476",
        "mesh_id": "D000068736",
        "type": "Compound",
    },
    "Montelukast": {
        "hetionet_id": "Compound::DB00471",
        "drugbank_id": "DB00471",
        "mesh_id": "D000068258",
        "type": "Compound",
    },
    "Octreotide": {
        "hetionet_id": "DB00104",
        "mesh_id": "D015282",
        "type": "Compound",
    },
    "Sildenafil": {
        "hetionet_id": "DB00203",
        "mesh_id": "D000068677",
        "type": "Compound",
    },
    "Metformin": {
        "hetionet_id": "DB00331",
        "mesh_id": "D008687",
        "type": "Compound",
    },
    "Thalidomide": {
        "hetionet_id": "DB01041",
        "mesh_id": "D013792",
        "type": "Compound",
    },
    "Losartan": {
        "hetionet_id": "DB00678",
        "mesh_id": "D019808",
        "type": "Compound",
    },
    # --- Genes (NCBI Entrez Gene IDs in Hetionet v1.0) ---
    "SLC5A2": {
        "hetionet_id": "6524",
        "entrez_id": "6524",
        "symbol": "SLC5A2",
        "alias": "SGLT2",
        "type": "Gene",
    },
    "SLC5A1": {
        "hetionet_id": "6523",
        "entrez_id": "6523",
        "symbol": "SLC5A1",
        "alias": "SGLT1",
        "type": "Gene",
    },
    "GRIN1": {
        "hetionet_id": "2902",
        "entrez_id": "2902",
        "symbol": "GRIN1",
        "alias": "NMDAR1",
        "type": "Gene",
    },
    "GRIN2A": {
        "hetionet_id": "2903",
        "entrez_id": "2903",
        "symbol": "GRIN2A",
        "alias": "NMDAR2A",
        "type": "Gene",
    },
    "GRIN2B": {
        "hetionet_id": "2904",
        "entrez_id": "2904",
        "symbol": "GRIN2B",
        "alias": "NMDAR2B",
        "type": "Gene",
    },
    "BDNF": {
        "hetionet_id": "627",
        "entrez_id": "627",
        "symbol": "BDNF",
        "type": "Gene",
    },
    "NLRP3": {
        "hetionet_id": "114548",
        "entrez_id": "114548",
        "symbol": "NLRP3",
        "alias": "CIAS1",
        "type": "Gene",
    },
    "IL1B": {
        "hetionet_id": "3553",
        "entrez_id": "3553",
        "symbol": "IL1B",
        "alias": "IL-1beta",
        "type": "Gene",
    },
    "TUBB": {
        "hetionet_id": "203068",
        "entrez_id": "203068",
        "symbol": "TUBB",
        "alias": "Tubulin",
        "type": "Gene",
    },
    "GLP1R": {
        "hetionet_id": "2740",
        "entrez_id": "2740",
        "symbol": "GLP1R",
        "alias": "GLP-1R",
        "type": "Gene",
    },
    # --- Diseases (Disease Ontology DOIDs) ---
    "Heart Failure": {
        "doid": "DOID:6000",
        "hetionet_doid": "Disease::DOID:6000",
        "mesh_id": "D006333",
        "canonical_name": "Heart Failure",
        "type": "Disease",
    },
    "Dilated Cardiomyopathy": {
        "doid": "DOID:12930",
        "hetionet_doid": "Disease::DOID:12930",
        "mesh_id": "D002311",
        "canonical_name": "Dilated Cardiomyopathy",
        "type": "Disease",
    },
    "Chronic Kidney Disease": {
        "doid": "DOID:784",
        "hetionet_doid": "Disease::DOID:784",
        "mesh_id": "D051436",
        "canonical_name": "Chronic Kidney Disease",
        "type": "Disease",
    },
    "Type 2 Diabetes": {
        "doid": "DOID:9352",
        "hetionet_doid": "Disease::DOID:9352",
        "mesh_id": "D003924",
        "canonical_name": "Type 2 Diabetes Mellitus",
        "type": "Disease",
    },
    "Major Depressive Disorder": {
        "doid": "DOID:1595",
        "hetionet_doid": "Disease::DOID:1595",
        "mesh_id": "D003865",
        "alias": "Depression",
        "canonical_name": "Endogenous Depression (MDD)",
        "type": "Disease",
    },
    "Coronary Artery Disease": {
        "doid": "DOID:3393",
        "hetionet_doid": "Disease::DOID:3393",
        "mesh_id": "D003324",
        "alias": "CAD",
        "canonical_name": "Coronary Artery Disease",
        "type": "Disease",
    },
    "Atherosclerosis": {
        "doid": "DOID:1936",
        "hetionet_doid": "Disease::DOID:1936",
        "mesh_id": "D050197",
        "canonical_name": "Atherosclerosis",
        "type": "Disease",
    },
    "Obesity": {
        "doid": "DOID:9970",
        "hetionet_doid": "Disease::DOID:9970",
        "mesh_id": "D009765",
        "canonical_name": "Obesity",
        "type": "Disease",
    },
    "NAFLD": {
        "doid": "DOID:0080358",
        "canonical_name": "Non-alcoholic fatty liver disease",
        "type": "Disease",
    },
}


def resolve_identifier(name: str) -> Optional[str]:
    """Resolves an entity name or alias to its canonical Hetionet / DOID / Entrez ID."""
    if not name:
        return None
    name_clean = name.strip()
    for entity, mapping in ID_CROSSWALK.items():
        if name_clean.lower() == entity.lower():
            return mapping.get("hetionet_id") or mapping.get("doid") or mapping.get("mesh_id")
        if "alias" in mapping and name_clean.lower() == mapping["alias"].lower():
            return mapping.get("hetionet_id") or mapping.get("doid")
    return None
