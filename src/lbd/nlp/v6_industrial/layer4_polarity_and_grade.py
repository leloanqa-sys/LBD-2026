#!/usr/bin/env python3
"""
LAYER 4: Algebra of Polarity, Clause-Scoped Negation, Contraindication Resolver,
and Domain-Range Ontology Enforcer.
Location: nlp/v6_industrial/layer4_polarity_and_grade.py
"""
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# -------------------------------------------------------------------------
# 1. Verb Direction Dictionaries (\Delta_verb \in {-1, +1})
# -------------------------------------------------------------------------
# \Delta_verb = -1 : Down-regulation, inhibition, reduction, attenuation, protection
DECREASE_VERB_PATTERNS = [
    r"\b(?:inhibit|inhibits|inhibited|inhibiting|inhibition\s+of|inhibitor(?:s)?\s+of)\b",
    r"\b(?:reduce|reduces|reduced|reducing|reduction\s+in|reduction\s+of)\b",
    r"\b(?:decrease|decreases|decreased|decreasing|decline\s+in|lowering|lower|lowers|lowered)\b",
    r"\b(?:attenuate|attenuates|attenuated|attenuating|attenuation\s+of)\b",
    r"\b(?:ameliorate|ameliorates|ameliorated|ameliorating|amelioration\s+of)\b",
    r"\b(?:prevent|prevents|prevented|preventing|prevention\s+of)\b",
    r"\b(?:suppress|suppresses|suppressed|suppressing|suppression\s+of)\b",
    r"\b(?:alleviate|alleviates|alleviated|alleviating)\b",
    r"\b(?:slow|slows|slowed|slowing\s+the\s+progression\s+of|delay|delays|delayed)\b",
    r"\b(?:protect|protects|protected|protecting\s+against|protective\s+against)\b",
    r"\b(?:improve|improves|improved|improving|improvement\s+in|beneficial\s+in|effective\s+in)\b",
    r"\b(?:treat|treats|treated|treating|treatment\s+of|treatment\s+for|therapy\s+for)\b",
]

# \Delta_verb = +1 : Up-regulation, induction, elevation, causation, risk increase
INCREASE_VERB_PATTERNS = [
    r"\b(?:increase|increases|increased|increasing|elevation\s+of|elevate|elevates|elevated)\b",
    r"\b(?:induce|induces|induced|inducing|induction\s+of)\b",
    r"\b(?:cause|causes|caused|causing)\b",
    r"\b(?:promote|promotes|promoted|promoting|stimulate|stimulates|stimulated|enhancing|enhance|enhances)\b",
    r"\b(?:trigger|triggers|triggered|precipitate|precipitates|precipitated)\b",
    r"\b(?:worsen|worsens|worsened|aggravate|aggravates|aggravated)\b",
    r"\b(?:risk\s+of|associated\s+with\s+an?\s+increased\s+risk\s+of|higher\s+incidence\s+of)\b",
]

# Contraindication / Regulatory Restriction cues (NEVER classify as TREATS or EVALUATED_IN_COHORT)
CONTRAINDICATION_RX = re.compile(
    r"\b(?:not\s+recommended\s+in|not\s+recommended\s+for|contraindicated\s+in|"
    r"should\s+not\s+be\s+used\s+in|should\s+not\s+be\s+initiated\s+in|"
    r"avoid\s+use\s+in|is\s+not\s+indicated\s+for|not\s+indicated\s+in)\b",
    re.IGNORECASE,
)

# Cohort / Eligibility Context cues (Patients enrolled with baseline disease X)
COHORT_CONTEXT_RX = re.compile(
    r"\b(?:in\s+patients\s+with|among\s+patients\s+with|in\s+adults\s+with|"
    r"in\s+subjects\s+with|patients\s+with\s+type\s+2\s+diabetes|cohort\s+of\s+patients\s+with)\b",
    re.IGNORECASE,
)

# Speculative / Hypothesis / Uncertain cues (Hedges)
UNCERTAIN_RX = re.compile(
    r"\b(?:may\s+potentially|hypothesized|hypothesis|remains\s+unclear|yet\s+to\s+be\s+determined|"
    r"requires\s+further\s+investigation|might\s+speculatively|may\s+play\s+a\s+role|"
    r"may\s+be\s+the\s+result\s+of|may\s+be\s+attributed\s+to|was\s+likely\s+associated\s+with|"
    r"could\s+theoretically\s+lead\s+to|theoretical\s+concern|theoretically\s+lead|suggestive\s+of|"
    r"may\s+help\s+control|potential\s+use|rationale\s+for\s+testing|expected\s+to\s+provide)\b",
    re.IGNORECASE,
)

# Future Research Gap cues (Studies are needed / Further work warranted)
RESEARCH_GAP_RX = re.compile(
    r"\b(?:studies\s+are\s+needed|further\s+studies\s+are\s+needed|trials\s+are\s+ongoing|"
    r"requires\s+further\s+investigation|yet\s+to\s+be\s+defined|is\s+under\s+investigation)\b",
    re.IGNORECASE,
)

_DECREASE_RX_LIST = [re.compile(p, re.IGNORECASE) for p in DECREASE_VERB_PATTERNS]
_INCREASE_RX_LIST = [re.compile(p, re.IGNORECASE) for p in INCREASE_VERB_PATTERNS]


@dataclass
class PolarityResolution:
    source_id: str
    target_id: str
    verb_delta: int                   # -1 (decrease/inhibit), +1 (increase/induce), 0 (neutral/cohort)
    target_valence: int               # +1 (pathological), -1 (healthy function), 0 (neutral)
    clinical_polarity: str            # THERAPEUTIC (-1), ADVERSE (+1), CONTRAINDICATED, COHORT_CONTEXT, MECHANISTIC, NEUTRAL
    canonical_relation: str           # Rich internal relation (TREATS, CAUSES, INHIBITS, CONTRAINDICATED_FOR, EVALUATED_IN_COHORT, etc.)
    contract_b_relation: str          # Strict Contract B enum (INHIBITS, ACTIVATES, CAUSES, INCREASES, DECREASES, ASSOCIATED_WITH, TREATS, PALLIATES)
    hetionet_metaedge: str            # CtD, CpD, CiG, CaG, CdG, CuG, DaG, etc.
    hetionet_sign: int                # -1, 0, +1
    assertion_status: str             # asserted, negated, uncertain
    local_clause: str                 # Exact local clause/window where target was governed
    rationale: str


class Layer4_PolarityAndGradeReasoner:
    r"""Implements the Algebra of Polarity:
        Clinical_Polarity = \Delta_verb * V_target
    with strict Clause-Scoped Negation and Domain-Range Ontology Constraints.
    """

    def extract_local_clause(self, sentence_text: str, target_start: int, target_end: int) -> str:
        """Splits a complex sentence at clause boundaries (', whereas', ', while', ';',
        ', without ', ', but ', ', and ') and returns ONLY the local clause containing
        the target entity span. This prevents 'without weight gain or hypoglycemia' in a
        subordinate phrase from falsely negating 'DKA' in the main clause.
        """
        # Identify prepositional 'without ...' boundaries specifically
        without_matches = list(re.finditer(r"(?:,\s*)?\bwithout\s+[^,.;]+", sentence_text, re.I))
        for wm in without_matches:
            if wm.start() <= target_start and target_end <= wm.end():
                # Target is INSIDE the 'without ...' phrase!
                return sentence_text[wm.start():wm.end()].strip()

        # If target is NOT inside a 'without ...' phrase, strip out any 'without ...' phrase
        # so its negation token cannot leak into the main clause!
        cleaned_sent = sentence_text
        if without_matches:
            # Replace 'without ...' segments with spaces of equal length to preserve offsets
            chars = list(sentence_text)
            for wm in without_matches:
                if not (wm.start() <= target_start and target_end <= wm.end()):
                    for i in range(wm.start(), wm.end()):
                        chars[i] = " "
            cleaned_sent = "".join(chars)

        # Further split by major clause delimiters (; or , whereas / , while / , however / , but)
        clause_delims = list(
            re.finditer(r"(?:;|\s*,\s*(?:whereas|while|however|but|although)\s+)", cleaned_sent, re.I)
        )
        if not clause_delims:
            return cleaned_sent.strip()

        boundaries = [0] + [m.start() for m in clause_delims] + [len(cleaned_sent)]
        for i in range(len(boundaries) - 1):
            c_start, c_end = boundaries[i], boundaries[i + 1]
            if c_start <= target_start <= c_end:
                return cleaned_sent[c_start:c_end].strip()

        return cleaned_sent.strip()

    def is_locally_negated(self, local_clause: str, target_matched_text: str) -> Tuple[bool, str]:
        """Checks if the target entity is governed by a true clinical negation inside its local clause."""
        clause_l = local_clause.lower()
        target_l = target_matched_text.lower()

        # 1. Check if inside a 'without [target]' phrase
        if re.search(r"\bwithout\b", clause_l):
            return True, "Negated locally via 'without' prepositional phrase"

        # 2. Check explicit non-occurrence / lack of effect on target
        # Note: Do NOT trigger negation if followed by contrastive 'but instead aggravated/increased'
        if re.search(r"\bdid\s+not\s+(?:alleviate|reduce|improve)[^,.;]*?\b(?:aggravated|worsened|increased)\b", clause_l):
            return False, ""

        neg_patterns = [
            r"\b(?:did\s+not|does\s+not|do\s+not)\s+(?:increase|cause|induce|affect|alter|reduce|worsen)\b",
            r"\b(?:no\s+suggestion\s+of\s+increased\s+risk|no\s+significant\s+increase\s+in|no\s+increase\s+in|no\s+effect\s+on|no\s+evidence\s+of|not\s+associated\s+with)\b",
            r"\b(?:free\s+of|absence\s+of)\b",
        ]
        for np in neg_patterns:
            m = re.search(np, clause_l)
            if m:
                # Verify negation precedes target in the local clause
                t_pos = clause_l.find(target_l)
                if t_pos == -1 or m.start() < t_pos + 25:
                    return True, f"Negated locally via '{m.group(0)}'"

        return False, ""

    def detect_verb_delta(self, local_clause: str, sentence_text: str, target_entity: Dict[str, Any]) -> Tuple[int, str]:
        r"""Determines the governing verb direction \Delta_verb \in {-1, +1, 0} for a target entity."""
        target_text = re.escape(target_entity["matched_text"])

        # Priority 0: Override "did not alleviate/reduce/prevent but instead aggravated/increased/worsened"
        if re.search(
            r"\b(?:did\s+not\s+(?:alleviate|reduce|prevent|improve|attenuate)[^,.;]*?\b(?:but\s+instead\s+)?(?:aggravated|worsened|increased))\b",
            sentence_text,
            re.I,
        ):
            return +1, "Contrastive adverse override ('did not alleviate but instead aggravated' -> +1)"
        # e.g., "increase urinary glucose excretion", "attenuates the progression of diabetic nephropathy"
        local_window_before = local_clause
        t_idx = local_clause.lower().find(target_entity["matched_text"].lower())
        if t_idx != -1:
            # Look at 65 chars before target and 40 chars after target
            w_start = max(0, t_idx - 75)
            w_end = min(len(local_clause), t_idx + len(target_entity["matched_text"]) + 45)
            local_window_before = local_clause[w_start:w_end]

        # Check specific high-precision object patterns first
        if re.search(
            r"\b(?:increase|increases|increased|increasing|enhance|enhances|enhanced|promote|promotes|inducing)\s+(?:[a-z\-]+\s+){0,4}"
            + target_text,
            local_window_before,
            re.I,
        ):
            return +1, "Direct object of INCREASE/PROMOTE verb (+1)"

        if re.search(
            r"\b(?:inhibit|inhibits|inhibited|inhibiting|inhibition\s+of|inhibitor(?:s)?\s+of|"
            r"reduce|reduces|reduced|reducing|reduction\s+in|reduction\s+of|"
            r"decrease|decreases|decreased|decreasing|"
            r"attenuate|attenuates|attenuated|attenuating|"
            r"ameliorate|ameliorates|ameliorated|"
            r"prevent|prevents|prevented|preventing|"
            r"slow|slows|slowed|slowing\s+the\s+progression\s+of|"
            r"improve|improves|improved|improving)\s+(?:[a-z\-]+\s+){0,5}"
            + target_text,
            local_window_before,
            re.I,
        ):
            return -1, "Direct object of DECREASE/INHIBIT/ATTENUATE verb (-1)"

        # Check passive or post-target verb ("diabetic nephropathy was attenuated", "SGLT2 inhibition")
        if re.search(
            target_text + r"\s+(?:[a-z\-]+\s+){0,3}(?:was\s+|were\s+|is\s+)?(?:inhibited|reduced|decreased|attenuated|ameliorated|prevented|improved|inhibition|inhibitor|inhibitors)",
            local_window_before,
            re.I,
        ):
            return -1, "Subject/modifier of DECREASE/INHIBIT verb (-1)"

        if re.search(
            target_text + r"\s+(?:[a-z\-]+\s+){0,3}(?:was\s+|were\s+|is\s+)?(?:increased|elevated|induced|caused|triggered|worsened)",
            local_window_before,
            re.I,
        ):
            return +1, "Subject/modifier of INCREASE/CAUSE verb (+1)"

        # Priority 2: Scan the narrow window around the target
        dec_hits = [rx.search(local_window_before).group(0) for rx in _DECREASE_RX_LIST if rx.search(local_window_before)]
        inc_hits = [rx.search(local_window_before).group(0) for rx in _INCREASE_RX_LIST if rx.search(local_window_before)]

        if dec_hits and not inc_hits:
            return -1, f"Window DECREASE verb '{dec_hits[0]}' (-1)"
        if inc_hits and not dec_hits:
            return +1, f"Window INCREASE verb '{inc_hits[0]}' (+1)"

        # Priority 3: Fallback to local clause scan
        dec_clause = [rx.search(local_clause).group(0) for rx in _DECREASE_RX_LIST if rx.search(local_clause)]
        inc_clause = [rx.search(local_clause).group(0) for rx in _INCREASE_RX_LIST if rx.search(local_clause)]

        if dec_clause and not inc_clause:
            return -1, f"Clause DECREASE verb '{dec_clause[0]}' (-1)"
        if inc_clause and not dec_clause:
            return +1, f"Clause INCREASE verb '{inc_clause[0]}' (+1)"
        if dec_clause and inc_clause:
            # Resolve conflict by choosing the verb closest in character distance to target_entity
            best_delta = -1
            best_dist = 9999
            best_word = ""
            for rx in _DECREASE_RX_LIST:
                for m in rx.finditer(sentence_text):
                    dist = min(abs(m.end() - target_entity["span_start"]), abs(target_entity["span_end"] - m.start()))
                    if dist < best_dist:
                        best_dist = dist
                        best_delta = -1
                        best_word = m.group(0)
            for rx in _INCREASE_RX_LIST:
                for m in rx.finditer(sentence_text):
                    dist = min(abs(m.end() - target_entity["span_start"]), abs(target_entity["span_end"] - m.start()))
                    if dist < best_dist:
                        best_dist = dist
                        best_delta = +1
                        best_word = m.group(0)
            return best_delta, f"Proximity-resolved verb '{best_word}' ({best_delta:+d}, dist={best_dist})"

        return 0, "No directional verb detected (0)"

    def resolve_pair_polarity(
        self,
        sentence_text: str,
        source_ent: Dict[str, Any],
        target_ent: Dict[str, Any],
    ) -> PolarityResolution:
        """Applies Layer 4 Algebra of Polarity + Domain-Range Constraints to a (source, target) pair."""
        src_type = source_ent["type"]
        tgt_type = target_ent["type"]
        tgt_valence = int(target_ent.get("valence", 0))

        local_clause = self.extract_local_clause(
            sentence_text, target_ent["span_start"], target_ent["span_end"]
        )

        # 1. Check Contraindication FIRST (before cohort or therapeutic!)
        # Fixes Bug #5: "Dapagliflozin is not recommended in patients with moderate or severe renal impairment"
        if CONTRAINDICATION_RX.search(sentence_text) and tgt_type == "Disease":
            return PolarityResolution(
                source_id=source_ent["id"],
                target_id=target_ent["id"],
                verb_delta=-1,
                target_valence=tgt_valence,
                clinical_polarity="CONTRAINDICATED",
                canonical_relation="CONTRAINDICATED_FOR",
                contract_b_relation="ASSOCIATED_WITH",
                hetionet_metaedge="CcD",
                hetionet_sign=-1,
                assertion_status="negated",
                local_clause=local_clause,
                rationale="Regulatory Contraindication ('not recommended / contraindicated in'); blocked from TREATS/STUDIED_IN.",
            )

        # 2. Check Cohort Eligibility Context EARLY ("in patients with X", "other causes of X")
        # before clause-wide verbs from another phrase can pollute the cohort disease!
        if tgt_type == "Disease":
            tgt_esc = re.escape(target_ent["matched_text"])
            is_cohort_phrase = bool(
                re.search(
                    rf"\b(?:in\s+patients\s+with|among\s+patients\s+with|in\s+adults\s+with|"
                    rf"in\s+subjects\s+with|patients\s+with\s+type\s+2\s+diabetes(?:\s+mellitus)?\s+and|"
                    rf"patients\s+with\s+moderate\s+or\s+severe|patients\s+with\s+moderate|"
                    rf"other\s+causes\s+of(?:\s+nondiabetic)?)\s+(?:[a-z0-9\-]+\s+){{0,3}}{tgt_esc}",
                    sentence_text,
                    re.I,
                )
            )
            if is_cohort_phrase:
                return PolarityResolution(
                    source_id=source_ent["id"],
                    target_id=target_ent["id"],
                    verb_delta=0,
                    target_valence=tgt_valence,
                    clinical_polarity="COHORT_BASELINE (0)",
                    canonical_relation="EVALUATED_IN_COHORT",
                    contract_b_relation="ASSOCIATED_WITH",
                    hetionet_metaedge="CpD",
                    hetionet_sign=0,
                    assertion_status="asserted",
                    local_clause=local_clause,
                    rationale="Patient eligibility cohort context ('in patients with X'); separated from TREATS/CAUSES.",
                )

        # 3. Check Local Clause Negation
        is_neg, neg_reason = self.is_locally_negated(local_clause, target_ent["matched_text"])
        is_uncertain = bool(UNCERTAIN_RX.search(local_clause))
        assertion_status = "negated" if is_neg else ("uncertain" if is_uncertain else "asserted")

        # 4. Compute Verb Direction \Delta_verb
        verb_delta, verb_reason = self.detect_verb_delta(local_clause, sentence_text, target_ent)

        # -----------------------------------------------------------------
        # DOMAIN-RANGE CASE A: Compound -> Gene (Pharmacological Target)
        # -----------------------------------------------------------------
        if src_type == "Compound" and tgt_type == "Gene":
            # SGLT2 (6524) or SGLT1 (6523) are pharmacologically inhibited by Dapagliflozin
            if source_ent["id"] == "DB06292" and target_ent["id"] in ("6524", "6523"):
                return PolarityResolution(
                    source_id=source_ent["id"],
                    target_id=target_ent["id"],
                    verb_delta=-1,
                    target_valence=tgt_valence,
                    clinical_polarity="MECHANISTIC_INHIBITION (-1)",
                    canonical_relation="INHIBITS",
                    contract_b_relation="INHIBITS",
                    hetionet_metaedge="CiG",
                    hetionet_sign=-1,
                    assertion_status=assertion_status,
                    local_clause=local_clause,
                    rationale=f"Primary pharmacological target inhibition ({verb_reason})",
                )
            # Check explicit activation/upregulation right near the target gene (e.g., "AMPK activation")
            tgt_esc = re.escape(target_ent["matched_text"])
            if re.search(
                rf"(?:{tgt_esc}\s+(?:[a-z\-]+\s+){{0,2}}(?:activation|activated|stimulation|upregulation|phosphorylation|pathway\s+activation)|"
                rf"(?:activates|activated|activating|stimulates|stimulated|upregulates|upregulated|activation\s+of)\s+(?:[a-z\-]+\s+){{0,3}}{tgt_esc})",
                local_clause,
                re.I,
            ) or verb_delta > 0:
                return PolarityResolution(
                    source_id=source_ent["id"],
                    target_id=target_ent["id"],
                    verb_delta=+1,
                    target_valence=tgt_valence,
                    clinical_polarity="MECHANISTIC_ACTIVATION (+1)",
                    canonical_relation="ACTIVATES",
                    contract_b_relation="ACTIVATES",
                    hetionet_metaedge="CaG",
                    hetionet_sign=1,
                    assertion_status=assertion_status,
                    local_clause=local_clause,
                    rationale=f"Compound->Gene activation ({verb_reason})",
                )
            else:
                return PolarityResolution(
                    source_id=source_ent["id"],
                    target_id=target_ent["id"],
                    verb_delta=-1,
                    target_valence=tgt_valence,
                    clinical_polarity="MECHANISTIC_INHIBITION (-1)",
                    canonical_relation="INHIBITS",
                    contract_b_relation="INHIBITS",
                    hetionet_metaedge="CiG",
                    hetionet_sign=-1,
                    assertion_status=assertion_status,
                    local_clause=local_clause,
                    rationale=f"Compound->Gene inhibition ({verb_reason})",
                )

        # -----------------------------------------------------------------
        # DOMAIN-RANGE CASE B: Compound/Gene -> BiologicalProcess (Bridging)
        # Fixes Bug #4 & Feedback Lỗ hổng 1:
        # Physiological Target Polarity Inversion:
        # - When Compound acts on BiologicalProcess: e.g. Dapagliflozin increases UGE.
        # - When Target Gene acts on BiologicalProcess: SGLT2 naturally REABSORBS glucose!
        #   Normal SGLT2 function DECREASES Urinary Glucose Excretion (Sign = -1),
        #   and INCREASES Renal Glucose Reabsorption (Sign = +1).
        #   Therefore, for Gene -> BiologicalProcess:
        #   effective_verb_delta = -1 * verb_delta_of_inhibition!
        # -----------------------------------------------------------------
        if tgt_type == "BiologicalProcess":
            eff_delta = verb_delta
            if src_type == "Gene":
                # Check known physiological role inversion for transport/metabolic genes
                if source_ent["id"] in ("6524", "6523"):
                    if target_ent["id"] in ("GO:0060004", "GO:0003091"):  # UGE, Natriuresis
                        eff_delta = -1  # SGLT2 naturally decreases excretion
                    elif target_ent["id"] in ("GO:0003092", "HP:0012592", "GO:0006954"):  # Reabsorption, Albuminuria, Fibrosis
                        eff_delta = +1  # SGLT2 naturally increases reabsorption & diabetic strain
                else:
                    if eff_delta == 0:
                        eff_delta = +1 if tgt_valence == 1 else -1
            else:
                if eff_delta == 0:
                    eff_delta = -1 if tgt_valence == 1 else +1

            product = eff_delta * tgt_valence  # -1 = Beneficial/Therapeutic mechanism, +1 = Pathological/Adverse
            clinical_pol = "THERAPEUTIC_MECHANISM (-1)" if product <= 0 else "PATHOLOGICAL_PROCESS (+1)"
            cb_rel = "INCREASES" if eff_delta > 0 else "DECREASES"
            metaedge = f"{src_type[0]}uBP" if eff_delta > 0 else f"{src_type[0]}dBP"
            return PolarityResolution(
                source_id=source_ent["id"],
                target_id=target_ent["id"],
                verb_delta=eff_delta,
                target_valence=tgt_valence,
                clinical_polarity=clinical_pol,
                canonical_relation=cb_rel,
                contract_b_relation=cb_rel,
                hetionet_metaedge=metaedge,
                hetionet_sign=eff_delta,
                assertion_status=assertion_status,
                local_clause=local_clause,
                rationale=(
                    f"Algebra of Polarity on BiologicalProcess: Delta_eff({eff_delta:+d}) * "
                    f"V_entity({tgt_valence:+d}) = {product:+d} [{clinical_pol}] ({verb_reason})"
                ),
            )

        # -----------------------------------------------------------------
        # DOMAIN-RANGE CASE C: Gene -> Disease
        # Fixes Bug #1: A Gene (SLC5A2 / SGLT2) can NEVER 'TREATS' or 'CAUSES' a Disease
        # directly when standing in for an inhibitor; Gene->Disease is strictly ASSOCIATED_WITH (DaG).
        # -----------------------------------------------------------------
        if src_type == "Gene" and tgt_type == "Disease":
            return PolarityResolution(
                source_id=source_ent["id"],
                target_id=target_ent["id"],
                verb_delta=verb_delta,
                target_valence=tgt_valence,
                clinical_polarity="GENETIC_DISEASE_ASSOCIATION (0)",
                canonical_relation="ASSOCIATED_WITH",
                contract_b_relation="ASSOCIATED_WITH",
                hetionet_metaedge="DaG",
                hetionet_sign=0,
                assertion_status=assertion_status,
                local_clause=local_clause,
                rationale="Domain-Range Constraint: Gene->Disease restricted to ASSOCIATED_WITH (prevents 'Gene TREATS Disease' category error).",
            )

        # -----------------------------------------------------------------
        # DOMAIN-RANGE CASE D: Compound -> Disease
        # Apply Algebra of Polarity: Clinical_Polarity = \Delta_verb * V_disease
        # Since V_disease = +1:
        #   - \Delta_verb = -1 => Clinical_Polarity = -1 => TREATS (CtD, sign = -1)
        #   - \Delta_verb = +1 => Clinical_Polarity = +1 => CAUSES (CcSE / CpD, sign = +1)
        # -----------------------------------------------------------------
        if src_type == "Compound" and tgt_type == "Disease":
            # Check if target is strictly inside a Cohort Eligibility phrase ("in patients with X")
            # AND not the direct object of a therapeutic/adverse verb
            cohort_match = re.search(
                r"\b(?:in\s+patients\s+with|among\s+patients\s+with|in\s+adults\s+with|in\s+subjects\s+with)\s+(?:[a-z0-9\-]+\s+){0,4}"
                + re.escape(target_ent["matched_text"]),
                sentence_text,
                re.I,
            )
            if cohort_match and verb_delta == 0:
                return PolarityResolution(
                    source_id=source_ent["id"],
                    target_id=target_ent["id"],
                    verb_delta=0,
                    target_valence=tgt_valence,
                    clinical_polarity="COHORT_BASELINE (0)",
                    canonical_relation="EVALUATED_IN_COHORT",
                    contract_b_relation="ASSOCIATED_WITH",
                    hetionet_metaedge="CpD",
                    hetionet_sign=0,
                    assertion_status=assertion_status,
                    local_clause=local_clause,
                    rationale="Patient eligibility cohort context ('in patients with X'); separated from TREATS.",
                )

            # Known adverse events (DKA, UTI/Genital infection) when discussed as risk/adverse event
            if target_ent["id"] in ("DOID:11714", "DOID:0060073") and verb_delta >= 0:
                return PolarityResolution(
                    source_id=source_ent["id"],
                    target_id=target_ent["id"],
                    verb_delta=+1,
                    target_valence=+1,
                    clinical_polarity="ADVERSE (+1)",
                    canonical_relation="CAUSES",
                    contract_b_relation="CAUSES",
                    hetionet_metaedge="CcSE",
                    hetionet_sign=1,
                    assertion_status=assertion_status,
                    local_clause=local_clause,
                    rationale=f"Adverse safety event polarity: Delta_verb(+1) * V_disease(+1) = +1 ({verb_reason})",
                )

            if verb_delta == -1:
                return PolarityResolution(
                    source_id=source_ent["id"],
                    target_id=target_ent["id"],
                    verb_delta=-1,
                    target_valence=+1,
                    clinical_polarity="THERAPEUTIC (-1)",
                    canonical_relation="TREATS",
                    contract_b_relation="TREATS",
                    hetionet_metaedge="CtD",
                    hetionet_sign=-1,
                    assertion_status=assertion_status,
                    local_clause=local_clause,
                    rationale=f"Algebra of Polarity: Delta_verb(-1) * V_disease(+1) = -1 [THERAPEUTIC] ({verb_reason})",
                )

            if verb_delta == +1:
                return PolarityResolution(
                    source_id=source_ent["id"],
                    target_id=target_ent["id"],
                    verb_delta=+1,
                    target_valence=+1,
                    clinical_polarity="ADVERSE (+1)",
                    canonical_relation="CAUSES",
                    contract_b_relation="CAUSES",
                    hetionet_metaedge="CcSE",
                    hetionet_sign=1,
                    assertion_status=assertion_status,
                    local_clause=local_clause,
                    rationale=f"Algebra of Polarity: Delta_verb(+1) * V_disease(+1) = +1 [ADVERSE] ({verb_reason})",
                )

            # Default when verb_delta == 0 in a clinical sentence
            return PolarityResolution(
                source_id=source_ent["id"],
                target_id=target_ent["id"],
                verb_delta=0,
                target_valence=+1,
                clinical_polarity="COHORT_BASELINE (0)",
                canonical_relation="EVALUATED_IN_COHORT",
                contract_b_relation="ASSOCIATED_WITH",
                hetionet_metaedge="CpD",
                hetionet_sign=0,
                assertion_status=assertion_status,
                local_clause=local_clause,
                rationale="Co-occurring clinical context without directional therapeutic/adverse verb -> EVALUATED_IN_COHORT.",
            )

        # Fallback for any other combination
        return PolarityResolution(
            source_id=source_ent["id"],
            target_id=target_ent["id"],
            verb_delta=verb_delta,
            target_valence=tgt_valence,
            clinical_polarity="NEUTRAL (0)",
            canonical_relation="ASSOCIATED_WITH",
            contract_b_relation="ASSOCIATED_WITH",
            hetionet_metaedge="CaX",
            hetionet_sign=0,
            assertion_status=assertion_status,
            local_clause=local_clause,
            rationale="General biological association.",
        )
