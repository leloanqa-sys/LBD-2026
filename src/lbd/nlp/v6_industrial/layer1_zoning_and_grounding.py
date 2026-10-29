#!/usr/bin/env python3
"""
LAYER 1 & LAYER 2: Section/Discourse Zoning, Multi-Ontology Grounding with Intrinsic Valence,
and Cross-Sentence Coreference Resolution.
Location: nlp/v6_industrial/layer1_zoning_and_grounding.py
"""
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

# -------------------------------------------------------------------------
# 1. Multi-Ontology Registry with Intrinsic Biological Valence (V_entity)
# -------------------------------------------------------------------------
# Valence (V_entity):
#   +1 : Pathological state, disease, adverse endpoint, or harmful biomarker
#        (Decreasing it = Therapeutic [-1]; Increasing it = Adverse [+1])
#   -1 : Healthy physiological function, organ function, or protective process
#        (Decreasing it = Adverse [+1]; Increasing it = Therapeutic [-1])
#    0 : Neutral anatomical/molecular entity

ONTOLOGY_REGISTRY: List[Dict[str, Any]] = [
    # --- COMPOUNDS (INTERVENTIONS) ---
    {
        "id": "DB06292",
        "canonical_name": "Dapagliflozin",
        "type": "Compound",
        "valence": 0,
        "is_primary_drug": True,
        "patterns": [
            r"\bdapagliflozin\b",
            r"\bbms-512148\b",
            r"\bfarxiga\b",
            r"\bforxiga\b",
        ],
    },
    # --- MOLECULAR TARGETS (GENES / PROTEINS) ---
    {
        "id": "6524",
        "canonical_name": "SLC5A2",
        "type": "Gene",
        "valence": 0,
        "patterns": [
            r"\bslc5a2\b",
            r"\bsglt[\s\-]?2\b",
            r"\bsodium[\s\-]glucose\s+co[\-\s]?transporter(?:s)?\s+(?:type\s+)?2\b",
            r"\bsodium[\s\-]glucose\s+transporter\s+2\b",
        ],
    },
    {
        "id": "6523",
        "canonical_name": "SLC5A1",
        "type": "Gene",
        "valence": 0,
        "patterns": [
            r"\bslc5a1\b",
            r"\bsglt[\s\-]?1\b",
            r"\bsodium[\s\-]glucose\s+co[\-\s]?transporter(?:s)?\s+(?:type\s+)?1\b",
        ],
    },
    {
        "id": "3630",
        "canonical_name": "INS",
        "type": "Gene",
        "valence": -1,
        "patterns": [r"\binsulin\s+sensitivity\b", r"\bbeta[\s\-]cell\s+function\b"],
    },
    {
        "id": "5562",
        "canonical_name": "PRKAA1",
        "type": "Gene",
        "valence": -1,
        "patterns": [r"\bampk\b", r"\bamp[\s\-]activated\s+protein\s+kinase\b"],
    },
    {
        "id": "114548",
        "canonical_name": "NLRP3",
        "type": "Gene",
        "valence": 1,
        "patterns": [r"\bnlrp3(?:\s+inflammasome)?\b"],
    },
    # --- BIOLOGICAL PROCESSES & PHYSIOLOGICAL PHENOTYPES (CANONICAL HETIONET GO NODES) ---
    {
        "id": "GO:0006814",
        "canonical_name": "Sodium Ion Transport & Natriuresis",
        "type": "BiologicalProcess",
        "valence": -1,  # Increasing sodium excretion lowers blood pressure and cardiac preload
        "patterns": [
            r"\bnatriuresis\b",
            r"\bosmotic\s+diuresis\b",
            r"\burinary\s+sodium\s+excretion\b",
            r"\bsodium\s+ion\s+transport\b",
            r"\btissue\s+sodium\s+content\b",
        ],
    },
    {
        "id": "GO:0060004",
        "canonical_name": "Urinary Glucose Excretion (Glucosuria)",
        "type": "BiologicalProcess",
        "valence": -1,  # Beneficial glucuresis mechanism
        "patterns": [
            r"\burinary\s+glucose\s+excretion\b",
            r"\bglucosuria\b",
            r"\bglucuresis\b",
            r"\bglycosuria\b",
        ],
    },
    {
        "id": "GO:0006006",
        "canonical_name": "Glucose Metabolic Process & Renal Reabsorption",
        "type": "BiologicalProcess",
        "valence": 1,  # Inhibiting renal reabsorption lowers hyperglycemia
        "patterns": [
            r"\brenal\s+glucose\s+reabsorption\b",
            r"\bglucose\s+reabsorption\b",
            r"\bglucose\s+metabolic\s+process\b",
            r"\brenal\s+glucose\s+absorption\b",
        ],
    },
    {
        "id": "GO:0006954",
        "canonical_name": "Inflammatory Response & Fibrosis",
        "type": "BiologicalProcess",
        "valence": 1,  # Pathological tissue inflammation
        "patterns": [
            r"\bcardiac\s+fibrosis\b",
            r"\brenal\s+fibrosis\b",
            r"\bmyocardial\s+fibrosis\b",
            r"\binflammatory\s+response\b",
            r"\bvascular\s+inflammation\b",
        ],
    },
    {
        "id": "HP:0012592",
        "canonical_name": "Albuminuria & Proteinuria",
        "type": "BiologicalProcess",
        "valence": 1,  # Pathological urinary protein loss
        "patterns": [
            r"\balbuminuria\b",
            r"\bproteinuria\b",
            r"\bmacroalbuminuria\b",
            r"\bmicroalbuminuria\b",
            r"\buacr\b",
        ],
    },
    {
        "id": "HP:0012213",
        "canonical_name": "Glomerular Filtration Rate",
        "type": "BiologicalProcess",
        "valence": -1,  # Healthy organ function
        "patterns": [
            r"\begfr\b",
            r"\bgfr\b",
            r"\bglomerular\s+filtration\s+rate\b",
            r"\brenal\s+function\b",
            r"\bkidney\s+function\b",
        ],
    },
    # --- DISEASES & CLINICAL ENDPOINTS ---
    {
        "id": "DOID:6000",
        "canonical_name": "Heart Failure",
        "type": "Disease",
        "valence": 1,
        "is_historical_baseline": False,
        "patterns": [
            r"\bheart\s+failure\b",
            r"\bhf\b",
            r"\bhhf\b",
            r"\bhospitalization\s+for\s+heart\s+failure\b",
            r"\bheart[\s\-]failure\s+hospitalization\b",
            r"\bcongestive\s+heart\s+failure\b",
            r"\bcardiac\s+failure\b",
            r"\bleft\s+ventricular\s+dysfunction\b",
        ],
    },
    {
        "id": "DOID:0080828",
        "canonical_name": "Chronic Kidney Disease",
        "type": "Disease",
        "valence": 1,
        "is_historical_baseline": False,
        "patterns": [
            r"\bchronic\s+kidney\s+disease\b",
            r"\bckd\b",
            r"\bchronic\s+renal\s+failure\b",
            r"\bkidney\s+disease\s+progression\b",
            r"\bend[\s\-]stage\s+renal\s+disease\b",
            r"\besrd\b",
        ],
    },
    {
        "id": "DOID:0050700",
        "canonical_name": "Diabetic Nephropathy",
        "type": "Disease",
        "valence": 1,
        "is_historical_baseline": False,
        "patterns": [
            r"\bdiabetic\s+nephropathy\b",
            r"\bdiabetic\s+kidney\s+disease\b",
            r"\bdkd\b",
            r"\bnephropathy\b",
            r"\bglomerular\s+damage\b",
        ],
    },
    {
        "id": "DOID:10763",
        "canonical_name": "Cardiovascular Disease & Hypertension",
        "type": "Disease",
        "valence": 1,
        "is_historical_baseline": False,
        "patterns": [
            r"\bcardiovascular\s+disease\b",
            r"\bcardiovascular\s+death\b",
            r"\bcardiovascular\s+mortality\b",
            r"\bmajor\s+adverse\s+cardiovascular\s+events\b",
            r"\bmace\b",
            r"\bhypertension\b",
            r"\batherosclerosis\b",
        ],
    },
    {
        "id": "DOID:0080208",
        "canonical_name": "NAFLD",
        "type": "Disease",
        "valence": 1,
        "is_historical_baseline": False,
        "patterns": [
            r"\bnafld\b",
            r"\bnash\b",
            r"\bnon[\s\-]alcoholic\s+fatty\s+liver\s+disease\b",
            r"\bnonalcoholic\s+steatohepatitis\b",
            r"\bhepatic\s+steatosis\b",
            r"\bfatty\s+liver\b",
        ],
    },
    {
        "id": "DOID:11714",
        "canonical_name": "Diabetic Ketoacidosis",
        "type": "Disease",
        "valence": 1,
        "is_historical_baseline": False,
        "patterns": [
            r"\bdiabetic\s+ketoacidosis\b",
            r"\beuglycemic\s+ketoacidosis\b",
            r"\bdka\b",
        ],
    },
    {
        "id": "HP:0001919",
        "canonical_name": "Renal Impairment",
        "type": "Disease",
        "valence": 1,
        "is_historical_baseline": False,
        "patterns": [
            r"\brenal\s+impairment\b",
            r"\bacute\s+kidney\s+injury\b",
            r"\baki\b",
            r"\brenal\s+insufficiency\b",
        ],
    },
    {
        "id": "DOID:0060073",
        "canonical_name": "Urinary & Genital Tract Infection",
        "type": "Disease",
        "valence": 1,
        "is_historical_baseline": False,
        "patterns": [
            r"\burinary\s+tract\s+infection(?:s)?\b",
            r"\bgenital\s+infection(?:s)?\b",
            r"\bmycotic\s+infection(?:s)?\b",
            r"\buti\b",
        ],
    },
    {
        "id": "DOID:9352",
        "canonical_name": "Type 2 Diabetes",
        "type": "Disease",
        "valence": 1,
        "is_historical_baseline": True,  # Flagged as known baseline indication
        "patterns": [
            r"\btype\s+2\s+diabetes(?:\s+mellitus)?\b",
            r"\bt2dm\b",
            r"\bt2d\b",
            r"\bdiabetes\s+mellitus\b",
            r"\bhyperglycemia\b",
        ],
    },
]

# Precompile regexes
for _entry in ONTOLOGY_REGISTRY:
    _entry["compiled_rx"] = [re.compile(p, re.IGNORECASE) for p in _entry["patterns"]]


# -------------------------------------------------------------------------
# 2. Discourse & Section Zoning Classifier (Layer 1)
# -------------------------------------------------------------------------

_METHODS_SEARCH_RX = re.compile(
    r"\b(?:systematic\s+search|searched\s+medline|medline,\s+embase|cochrane\s+library|"
    r"cochrane\s+databases|literature\s+search|pubmed,\s+embase|inclusion\s+criteria\s+were|"
    r"electronic\s+databases|meta[\s\-]analysis\s+was\s+conducted)\b",
    re.IGNORECASE,
)

_SECTION_HEADER_RX = re.compile(
    r"^\s*(BACKGROUND|INTRODUCTION|OBJECTIVE|OBJECTIVES|AIMS|AIM|IMPORTANCE OF THE FIELD|"
    r"AREAS COVERED|METHODS|PATIENTS AND METHODS|MATERIAL AND METHODS|STUDY DESIGN|"
    r"RESULTS|FINDINGS|MAIN RESULTS|CONCLUSIONS|CONCLUSION|SUMMARY|EXPERT OPINION)\s*:\s*",
    re.IGNORECASE,
)

_RESULTS_FINDING_CUES_RX = re.compile(
    r"\b(?:significantly|p\s*[<=>]\s*0\.\d+|hazard\s+ratio|95%\s+ci|demonstrated\s+that|"
    r"showed\s+that|reduced|attenuated|improved|decreased|inhibited|ameliorated)\b",
    re.IGNORECASE,
)

_REVIEW_PUBLICATION_CUES_RX = re.compile(
    r"\b(?:this\s+review|in\s+this\s+review|areas\s+covered|expert\s+opinion|"
    r"this\s+chapter|this\s+article\s+reviews|current\s+knowledge|systematic\s+review)\b",
    re.IGNORECASE,
)

_SENT_SPLIT_RX = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\[\"])")


@dataclass
class ZonedSentence:
    sent_idx: int
    text: str
    char_start: int
    char_end: int
    section_zone: str          # BACKGROUND, METHODS, RESULTS, CONCLUSIONS, TITLE
    section_weight: float      # 0.0 for METHODS_SEARCH, 0.5 for BACKGROUND, 1.0 for RESULTS/CONCLUSIONS
    entities: List[Dict[str, Any]] = field(default_factory=list)
    has_coref_drug: bool = False


class Layer12_ZoningAndGroundingEngine:
    """Performs Section Zoning, Multi-Ontology Entity Linking with Intrinsic Valence,

    and Cross-Sentence Drug Coreference Resolution (preventing the 'Gene treats Disease' category error).
    """

    def detect_study_design(self, title: str, abstract: str) -> Tuple[str, float]:
        """Classifies study design according to GRADE hierarchy."""
        full_l = f"{title} {abstract}".lower()
        if _REVIEW_PUBLICATION_CUES_RX.search(full_l):
            return "REVIEW_ARTICLE", 0.45
        if re.search(r"\b(?:randomized|randomised|double[\s\-]blind|placebo[\s\-]controlled|phase\s+(?:iii|3|ii|2)|dapa[\s\-]hf|declare[\s\-]timi)\b", full_l):
            return "RCT_CLINICAL_TRIAL", 1.00
        if re.search(r"\b(?:cvd[\s\-]real|real[\s\-]world|propensity\s+score|cohort\s+study|retrospective|observational|registry)\b", full_l):
            return "REAL_WORLD_COHORT", 0.80
        if re.search(r"\b(?:mice|mouse|rat|rats|murine|db/db|streptozotocin|in\s+vivo)\b", full_l):
            return "PRECLINICAL_IN_VIVO", 0.60
        if re.search(r"\b(?:in\s+vitro|cell\s+line|cultured|hk[\s\-]2\s+cells|proximal\s+tubular\s+cells)\b", full_l):
            return "IN_VITRO_CELLULAR", 0.40
        return "CLINICAL_STUDY", 0.75

    def split_and_zone_document(self, title: str, abstract: str) -> List[ZonedSentence]:
        """Splits document into sentences without ever gluing unrelated sentences with [...],

        and assigns each sentence to a Discourse Section Zone.
        """
        zoned_sents: List[ZonedSentence] = []
        if title and title.strip():
            zoned_sents.append(
                ZonedSentence(
                    sent_idx=0,
                    text=title.strip(),
                    char_start=0,
                    char_end=len(title.strip()),
                    section_zone="TITLE",
                    section_weight=0.85,
                )
            )

        if not abstract or not abstract.strip():
            return zoned_sents

        offset = (len(title.strip()) + 1) if title and title.strip() else 0
        raw_parts: List[Tuple[int, int, str]] = []
        last = 0
        for m in _SENT_SPLIT_RX.finditer(abstract):
            seg = abstract[last:m.start()]
            if seg.strip():
                raw_parts.append((offset + last, offset + m.start(), seg.strip()))
            last = m.end()
        tail = abstract[last:]
        if tail.strip():
            raw_parts.append((offset + last, offset + len(abstract), tail.strip()))

        current_zone = "BACKGROUND"
        n_parts = max(1, len(raw_parts))

        for idx, (s_start, s_end, s_text) in enumerate(raw_parts):
            # 1. Check explicit header
            hdr_match = _SECTION_HEADER_RX.match(s_text)
            if hdr_match:
                hdr = hdr_match.group(1).upper()
                if hdr in ("BACKGROUND", "INTRODUCTION", "OBJECTIVE", "OBJECTIVES", "AIMS", "AIM", "IMPORTANCE OF THE FIELD", "AREAS COVERED"):
                    current_zone = "BACKGROUND"
                elif hdr in ("METHODS", "PATIENTS AND METHODS", "MATERIAL AND METHODS", "STUDY DESIGN"):
                    current_zone = "METHODS"
                elif hdr in ("RESULTS", "FINDINGS", "MAIN RESULTS"):
                    current_zone = "RESULTS"
                elif hdr in ("CONCLUSIONS", "CONCLUSION", "SUMMARY", "EXPERT OPINION"):
                    current_zone = "CONCLUSIONS"

            # 2. Check search strategy / database query noise
            if _METHODS_SEARCH_RX.search(s_text):
                assigned_zone = "METHODS_SEARCH"
                weight = 0.0  # Immediate zero weight for database search descriptions
            elif current_zone == "METHODS":
                assigned_zone = "METHODS"
                weight = 0.25
            else:
                # Positional & cue refinement when abstract is unstructured
                rel_pos = (idx + 1) / n_parts
                if not hdr_match:
                    if idx == n_parts - 1 or re.search(r"\b(?:in\s+conclusion|we\s+conclude|overall|summary)\b", s_text, re.I):
                        assigned_zone = "CONCLUSIONS"
                        weight = 1.0
                    elif _RESULTS_FINDING_CUES_RX.search(s_text) or rel_pos >= 0.35:
                        assigned_zone = "RESULTS"
                        weight = 1.0
                    else:
                        assigned_zone = "BACKGROUND"
                        weight = 0.55
                else:
                    assigned_zone = current_zone
                    weight = 1.0 if current_zone in ("RESULTS", "CONCLUSIONS") else 0.55

            zoned_sents.append(
                ZonedSentence(
                    sent_idx=idx + 1,
                    text=s_text,
                    char_start=s_start,
                    char_end=s_end,
                    section_zone=assigned_zone,
                    section_weight=weight,
                )
            )

        return zoned_sents

    def ground_and_resolve_coref(self, title: str, abstract: str) -> Dict[str, Any]:
        """Extracts ontology-grounded spans per sentence and applies Layer 2 Cross-Sentence

        Coreference Resolution so 'the drug', 'SGLT2i', 'SGLT2 inhibition' in a Dapagliflozin
        paper accurately retain the Compound anchor (DB06292) alongside the Gene target (6524).
        """
        full_text = f"{title} {abstract}"
        study_design, study_weight = self.detect_study_design(title, abstract)
        sentences = self.split_and_zone_document(title, abstract)

        # Check if Dapagliflozin is explicitly anchored in the document
        doc_has_dapagliflozin = bool(
            re.search(r"\b(?:dapagliflozin|bms-512148|farxiga|forxiga)\b", full_text, re.I)
        )

        # Coreference / Metonymy patterns referring back to Dapagliflozin in a Dapagliflozin paper
        coref_drug_rx = re.compile(
            r"\b(?:sglt[\s\-]?2\s+inhibitor(?:s)?|sglt[\s\-]?2i|sglt[\s\-]?2\s+inhibition|"
            r"this\s+drug|the\s+drug|this\s+agent|the\s+agent|dapagliflozin\s+treatment|"
            r"dapagliflozin\s+therapy)\b",
            re.IGNORECASE,
        )

        for sent in sentences:
            found_by_id: Dict[str, Dict[str, Any]] = {}
            for entry in ONTOLOGY_REGISTRY:
                for rx in entry["compiled_rx"]:
                    m = rx.search(sent.text)
                    if m:
                        eid = entry["id"]
                        if eid not in found_by_id:
                            found_by_id[eid] = {
                                "id": eid,
                                "canonical_name": entry["canonical_name"],
                                "type": entry["type"],
                                "valence": entry["valence"],
                                "is_historical_baseline": entry.get("is_historical_baseline", False),
                                "matched_text": sent.text[m.start():m.end()],
                                "span_start": m.start(),
                                "span_end": m.end(),
                                "is_coref_resolved": False,
                            }
                        break

            # Layer 2: Cross-Sentence Coreference & Pharmacological Class Metonymy Resolution
            # If the document is about Dapagliflozin and the sentence mentions 'SGLT2i' / 'SGLT2 inhibition'
            # or 'the drug', resolve Dapagliflozin (DB06292) into the sentence entities AND keep SLC5A2 (6524)!
            if doc_has_dapagliflozin and "DB06292" not in found_by_id:
                cm = coref_drug_rx.search(sent.text)
                if cm:
                    sent.has_coref_drug = True
                    found_by_id["DB06292"] = {
                        "id": "DB06292",
                        "canonical_name": "Dapagliflozin",
                        "type": "Compound",
                        "valence": 0,
                        "is_historical_baseline": False,
                        "matched_text": sent.text[cm.start():cm.end()],
                        "span_start": cm.start(),
                        "span_end": cm.end(),
                        "is_coref_resolved": True,
                    }

            sent.entities = list(found_by_id.values())

        return {
            "doc_has_dapagliflozin": doc_has_dapagliflozin,
            "study_design": study_design,
            "study_weight": study_weight,
            "sentences": sentences,
        }
