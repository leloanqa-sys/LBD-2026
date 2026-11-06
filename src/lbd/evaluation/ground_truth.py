"""5-Tier Ground Truth Classifications & Out-of-Sample Benchmark Registry.

Owner: Nguyễn Công Gia Phước (Evaluation, Ground Truth & Benchmarks)
Co-owners:
  - Đặng Quang Huy (Data Cutoff & Zero-Leakage Audit)
  - Nguyễn Nhật Lâm (Algorithm & Ranking Engine)

Provides:
  1. Legacy 5-Tier Ground Truth (`GROUND_TRUTH_TIERS`) for Dapagliflozin (`DB06292`).
  2. Multi-Drug Landmark Ground Truth (`LANDMARK_GROUND_TRUTH_BY_DRUG`) covering the
     4 post-2018 clinical repurposing paradigms:
       - Ketamine (`Compound::DB01221`) -> Depression (`Disease::DOID:1595`)
       - Colchicine (`Compound::DB01394`) -> CAD (`Disease::DOID:3393`) & Atherosclerosis (`Disease::DOID:1936`)
       - Dapagliflozin (`Compound::DB06292`) / Canagliflozin (`Compound::DB08907`) -> Heart Failure (`Disease::DOID:12930`) & CKD (`Disease::DOID:784`)
       - GLP-1 / Peptide GPCR Axis (`Compound::DB00104`) -> Obesity (`Disease::DOID:9970`)
  3. Loader helpers (`load_test_suite`, `get_ground_truth_credentials`) for the
     consolidated test suites in `data/ground_truth/`.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT_DIR = Path(__file__).resolve().parents[3]
GROUND_TRUTH_DIR = ROOT_DIR / "data" / "ground_truth"

GROUND_TRUTH_TIERS: Dict[str, Dict[str, str]] = {
    "HISTORICAL_BASELINE": {
        "DOID:9352": "Type 2 Diabetes Mellitus",
        "Disease::DOID:9352": "Type 2 Diabetes Mellitus",
        "Disease::DOID:10763": "Hypertension",
        "Disease::DOID:1826": "Epilepsy",
    },
    "HARD_POSITIVES": {
        "DOID:9333": "Heart Failure",
        "DOID:784": "Chronic Kidney Disease",
        "Disease::DOID:12930": "Dilated Cardiomyopathy (Heart Failure)",
        "Disease::DOID:784": "Chronic Kidney Failure (CKD)",
        "Disease::DOID:1595": "Endogenous Depression (MDD)",
        "Disease::DOID:3393": "Coronary Artery Disease",
        "Disease::DOID:1936": "Atherosclerosis",
        "Disease::DOID:9970": "Obesity",
    },
    "SOFT_POSITIVES": {
        "DOID:0080358": "Non-alcoholic fatty liver disease",
    },
    "TRUE_NEGATIVES": {
        "TRAP:TERMINATED_01": "Suspended Phase 3 Trial",
        "Disease::DOID:8893": "Psoriasis (Terminated Clinical Trial Control)",
        "Disease::DOID:5419": "Schizophrenia (Terminated Clinical Trial Control)",
    },
}

LANDMARK_GROUND_TRUTH_BY_DRUG: Dict[str, Dict[str, Any]] = {
    "Compound::DB01221": {
        "drug_name": "Ketamine",
        "historical_indications": ["Anesthesia"],
        "hard_positives": {
            "Disease::DOID:1595": {
                "disease_name": "endogenous depression (Major Depressive Disorder)",
                "approval_year": 2019,
                "authority": "FDA Approval (Spravato / Esketamine for TRD, March 2019)",
                "mechanism": "NMDA receptor antagonism (GRIN1/GRIN2A) -> BDNF synaptogenesis",
            }
        },
    },
    "Compound::DB01394": {
        "drug_name": "Colchicine",
        "historical_indications": ["Gout (DOID:13189)"],
        "hard_positives": {
            "Disease::DOID:3393": {
                "disease_name": "coronary artery disease",
                "approval_year": 2023,
                "authority": "FDA Approval (Lodoco 0.5mg, June 2023; LoDoCo2 & COLCOT trials)",
                "mechanism": "Tubulin inhibition -> NLRP3 inflammasome & IL1B innate immune suppression",
            },
            "Disease::DOID:1936": {
                "disease_name": "atherosclerosis",
                "approval_year": 2023,
                "authority": "FDA Approval (Lodoco 0.5mg, June 2023)",
                "mechanism": "NLRP3 inflammasome & IL-1beta plaque stabilization",
            },
        },
    },
    "Compound::DB06292": {
        "drug_name": "Dapagliflozin",
        "historical_indications": ["Type 2 Diabetes Mellitus (DOID:9352)"],
        "hard_positives": {
            "Disease::DOID:12930": {
                "disease_name": "dilated cardiomyopathy (Heart Failure)",
                "approval_year": 2020,
                "authority": "FDA Approval (Farxiga DAPA-HF, May 2020)",
                "mechanism": "SLC5A2 (SGLT2) inhibition -> cardiorenal preload/afterload & metabolic protection",
            },
            "Disease::DOID:784": {
                "disease_name": "chronic kidney failure (CKD)",
                "approval_year": 2021,
                "authority": "FDA Approval (Farxiga DAPA-CKD, April 2021)",
                "mechanism": "SLC5A2 inhibition -> tubuloglomerular feedback & intraglomerular pressure reduction",
            },
        },
    },
    "Compound::DB08907": {
        "drug_name": "Canagliflozin",
        "historical_indications": ["Type 2 Diabetes Mellitus (DOID:9352)"],
        "hard_positives": {
            "Disease::DOID:784": {
                "disease_name": "chronic kidney failure (CKD)",
                "approval_year": 2019,
                "authority": "FDA Approval (Invokana CREDENCE Trial, September 2019)",
                "mechanism": "SLC5A2 (SGLT2) inhibition -> renal protection",
            }
        },
    },
    "Compound::DB00104": {
        "drug_name": "Octreotide (Peptide GPCR proxy / GLP-1 axis)",
        "historical_indications": ["Acromegaly / Endocrine Tumors"],
        "hard_positives": {
            "Disease::DOID:9970": {
                "disease_name": "obesity",
                "approval_year": 2021,
                "authority": "STEP-1 / Wegovy GLP-1R Multi-Tissue Hypothalamic Axis",
                "mechanism": "Pancreatic & hypothalamic multi-tissue GPCR satiety signaling",
            }
        },
    },
}


def classify_candidate(disease_id: str) -> str:
    """Classifies a candidate disease ID into one of the Ground Truth tiers."""
    for tier, ids in GROUND_TRUTH_TIERS.items():
        if disease_id in ids:
            return tier
    return "UNKNOWN"


def get_drug_ground_truth(drug_id: str) -> Dict[str, Any]:
    """Returns structured ground truth metadata for a given DrugBank/Hetionet compound ID."""
    norm_id = drug_id if drug_id.startswith("Compound::") else f"Compound::{drug_id}"
    return LANDMARK_GROUND_TRUTH_BY_DRUG.get(norm_id, {})


def load_test_suite(suite_name: str = "prospective_temporal_test") -> List[Dict[str, Any]]:
    """Loads an adjudicated JSON test suite from `data/ground_truth/`.

    Supported `suite_name` values:
      - `prospective_temporal_test` (Suite A: 10 Post-2018 FDA/Phase 3 pairs)
      - `rare_disease_test` (Suite B: 11 Low-degree rare disease pairs)
      - `clinical_failure_untrue_test` (Suite C: 184 repoDB terminated/failed clinical trial controls)
      - `landmark_4_classes_test` (Suite D: 7 Landmark cross-graph jump cases)
    """
    filename = suite_name if suite_name.endswith(".json") else f"{suite_name}.json"
    path = GROUND_TRUTH_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Ground truth suite not found at {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def get_ground_truth_credentials() -> Dict[str, Any]:
    """Loads the dataset registry, DOI provenance, and SHA-256 checksums."""
    cred_path = GROUND_TRUTH_DIR / "ground_truth_credentials.json"
    if cred_path.exists():
        return json.loads(cred_path.read_text(encoding="utf-8"))
    return {}
