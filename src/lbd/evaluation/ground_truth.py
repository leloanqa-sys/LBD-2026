"""5-Tier Ground Truth Classifications."""
from typing import Dict, Any

GROUND_TRUTH_TIERS = {
    "HISTORICAL_BASELINE": {
        "DOID:9352": "Type 2 Diabetes Mellitus" # Filter out
    },
    "HARD_POSITIVES": {
        "DOID:9333": "Heart Failure",
        "DOID:784": "Chronic Kidney Disease"
    },
    "SOFT_POSITIVES": {
        "DOID:0080358": "Non-alcoholic fatty liver disease"
    },
    "TRUE_NEGATIVES": {
        "TRAP:TERMINATED_01": "Suspended Phase 3 Trial"
    }
}

def classify_candidate(disease_id: str) -> str:
    for tier, ids in GROUND_TRUTH_TIERS.items():
        if disease_id in ids:
            return tier
    return "UNKNOWN"
