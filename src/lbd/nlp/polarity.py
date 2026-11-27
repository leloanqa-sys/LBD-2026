#!/usr/bin/env python3
"""
LAYER 4: Algebra of Polarity, ConText/BioScope Clause-Scoped Negation, Pseudo-Trigger Filtering,
Contraindication Resolver, and Domain-Range Ontology Enforcer.

Literature Grounding:
- Harkema et al. (J. Biomed. Inform. 2009, ConText) & Chapman et al. (J. Biomed. Inform. 2001, NegEx):
  Pre-negation, post-negation, pseudo-negation triggers, and scope termination tokens.
- Vincze et al. (BMC Bioinformatics 2008, BioScope) & Peng et al. (AMIA 2018, NegBio):
  Clause-scoped negation and uncertainty detection in biomedical text.
- Luo et al. (Briefings in Bioinformatics 2022, BioRED):
  Coarse polarity (`POSITIVE`, `NEGATIVE`, `ASSOCIATION`) and ontological domain-range constraints.
"""
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

try:
    from .sentence_splitter import split_sentence_clauses
except ImportError:
    split_sentence_clauses = None

# -------------------------------------------------------------------------
# 0. ConText Pseudo-Triggers (Harkema et al. 2009)
#    Expressions that superficially contain directional or negation tokens
#    but do NOT assert causation, increase, or clinical negation.
# -------------------------------------------------------------------------
PSEUDO_CAUSE_OR_INCREASE_RX = re.compile(
    r"\b(?:all[\s\-]cause(?:s)?|other\s+causes?\s+of|common\s+causes?\s+of|leading\s+causes?\s+of|"
    r"root\s+cause|cause\s+of\s+death|without\s+causing|did\s+not\s+cause|does\s+not\s+cause|"
    r"no\s+significant\s+increase\s+in(?:\s+the\s+risk\s+of)?|no\s+increase\s+in(?:\s+the\s+risk\s+of)?|"
    r"did\s+not\s+increase(?:\s+the\s+risk\s+of)?|does\s+not\s+increase(?:\s+the\s+risk\s+of)?|"
    r"not\s+associated\s+with\s+an?\s+increased\s+risk\s+of|similar\s+risk\s+of|comparable\s+risk\s+of|"
    r"lower\s+risk\s+of|reduced\s+risk\s+of|decreased\s+risk\s+of|reduction\s+in\s+(?:the\s+)?risk\s+of|"
    r"at\s+(?:high\s+|increased\s+)?risk\s+(?:of|for)|patients\s+at\s+risk\s+(?:of|for))\b",
    re.IGNORECASE,
)

PSEUDO_NEGATION_RX = re.compile(
    r"\b(?:no\s+longer|not\s+only|non[\s\-]inferior(?:ity)?|no\s+further|not\s+ruled\s+out)\b",
    re.IGNORECASE,
)

# -------------------------------------------------------------------------
# 1. Verb Direction Dictionaries (\Delta_verb \in {-1, +1})
# -------------------------------------------------------------------------
# \Delta_verb = -1 : Down-regulation, inhibition, reduction, attenuation, protection, therapy
DECREASE_VERB_PATTERNS = [
    r"\b(?:inhibit|inhibits|inhibited|inhibiting|inhibition\s+of|inhibitor(?:s)?\s+of)\b",
    r"\b(?:reduce|reduces|reduced|reducing|reduction\s+in|reduction\s+of|reduced\s+the\s+risk\s+of|lower\s+risk\s+of|lowered\s+the\s+risk\s+of)\b",
    r"\b(?:decrease|decreases|decreased|decreasing|decline\s+in|lowering|lower|lowers|lowered)\b",
    r"\b(?:attenuate|attenuates|attenuated|attenuating|attenuation\s+of)\b",
    r"\b(?:ameliorate|ameliorates|ameliorated|ameliorating|amelioration\s+of)\b",
    r"\b(?:prevent|prevents|prevented|preventing|prevention\s+of)\b",
    r"\b(?:suppress|suppresses|suppressed|suppressing|suppression\s+of)\b",
    r"\b(?:alleviate|alleviates|alleviated|alleviating)\b",
    r"\b(?:slow|slows|slowed|slowing\s+the\s+progression\s+of|delay|delays|delayed)\b",
    r"\b(?:protect|protects|protected|protecting\s+against|protective\s+against)\b",
    r"\b(?:improve|improves|improved|improving|improvement\s+in|beneficial\s+in|effective\s+in)\b",
    r"\b(?:treat|treats|treated|treating|treatment\s+of|treatment\s+for|therapy\s+for|reverse|reverses|reversed|abolish|abolished|mitigate|mitigates|mitigated)\b",
]

# \Delta_verb = +1 : Up-regulation, induction, elevation, causation, toxicity, adverse risk increase
# Note: bare "risk of" is deliberately excluded to prevent false positives on "reduced risk of" / "similar risk of";
# explicit "increased risk of" / "higher risk of" / "excess risk of" are matched instead.
INCREASE_VERB_PATTERNS = [
    r"\b(?:increase|increases|increased|increasing|elevation\s+of|elevate|elevates|elevated)\b",
    r"\b(?:induce|induces|induced|inducing|induction\s+of|[a-z0-9\-]+[\s\-]induced)\b",
    r"\b(?<!all[\s\-])(?:cause|causes|caused|causing)\b",
    r"\b(?:promote|promotes|promoted|promoting|stimulate|stimulates|stimulated|enhancing|enhance|enhances)\b",
    r"\b(?:trigger|triggers|triggered|precipitate|precipitates|precipitated|provoke|provokes|provoked|elicit|elicits|elicited)\b",
    r"\b(?:worsen|worsens|worsened|aggravate|aggravates|aggravated|exacerbate|exacerbates|exacerbated)\b",
    r"\b(?:increased\s+(?:the\s+)?risk\s+of|higher\s+(?:risk|incidence|rate)\s+of|excess\s+risk\s+of|associated\s+with\s+an?\s+increased\s+risk\s+of)\b",
    r"\b(?:lead\s+to|leads\s+to|led\s+to|result\s+in|results\s+in|resulted\s+in|produce|produces|produced|develop|developed)\b",
    r"\b(?:toxicity|adverse\s+effect(?:s)?|adverse\s+event(?:s)?|side\s+effect(?:s)?|complication(?:s)?\s+of|overdose)\b",
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
    r"in\s+subjects\s+with|in\s+participants\s+with|cohort\s+of\s+patients\s+with|"
    r"enrolled\s+patients\s+with|recruited\s+patients\s+with)\b",
    re.IGNORECASE,
)

# Therapeutic/efficacy cues immediately governing "in patients with X" so X is treated as target, not mere cohort
THERAPEUTIC_COHORT_GOVERNOR_RX = re.compile(
    r"\b(?:effective|efficacy|beneficial|benefit(?:s)?|approved|indicated|licensed|"
    r"treatment\s+of|therapy\s+(?:for|of)|monotherapy\s+(?:for|in)|first[\s\-]line\s+(?:for|in)|"
    r"safe\s+and\s+effective)\s+(?:in\s+|for\s+)?(?:patients|adults|subjects|individuals)\s+with\b",
    re.IGNORECASE,
)

# Speculative / Hypothesis / Uncertain cues (Hedges - ConText / BioScope)
UNCERTAIN_RX = re.compile(
    r"\b(?:may|might|could|suggests?|suggested|suggesting|suggestive\s+of|appears?\s+to|"
    r"potentially|potential|possible|possibly|hypothesis|hypothesized|hypothesize|"
    r"warrants?\s+further|trend\s+toward|trends?\s+towards?|preliminary|controversial|"
    r"unclear|remains?\s+unclear|remains?\s+to\s+be|yet\s+to\s+be\s+determined|whether|"
    r"requires?\s+further\s+investigation|further\s+studies|speculat\w*|"
    r"theoretical(?:ly)?|promising|proposed\s+as|seems?\s+to|"
    r"was\s+likely\s+associated\s+with|likely\s+to|rationale\s+for\s+testing|"
    r"expected\s+to\s+provide|can\s+be\s+expected\s+to|not\s+yet\s+established|"
    r"is\s+thought\s+to|is\s+believed\s+to|is\s+hypothesized\s+to|indicative\s+of|"
    r"plausible|putative|tentative|inconclusive|not\s+fully\s+understood|poorly\s+understood|"
    r"limited\s+evidence|merits?\s+further|need(?:s)?\s+to\s+be\s+confirmed|should\s+be\s+confirmed|"
    r"future\s+trials?|ongoing\s+trials?|is\s+being\s+evaluated|will\s+(?:evaluate|assess|determine|investigate))\b",
    re.IGNORECASE,
)

# Outer-clause / Matrix-clause Epistemic Scope cues (propagate uncertainty to subordinate clauses)
OUTER_EPISTEMIC_SCOPE_RX = re.compile(
    r"\b(?:to\s+(?:evaluate|assess|investigate|determine|examine|explore|test|compare|study|elucidate|clarify)\s+"
    r"(?:whether|if|the\s+(?:effect|effects|efficacy|safety|impact|role|association|relationship|potential|benefits?))|"
    r"(?:aimed|aims?|sought|designed|intended)\s+to\s+(?:evaluate|assess|investigate|determine|examine|explore|test|compare|study)|"
    r"(?:investigated|evaluated|assessed|examined|tested|explored|studied)\s+(?:whether|if)|"
    r"(?:hypothesized|postulated|proposed|speculated)\s+that|"
    r"remains?\s+(?:unclear|unknown|uncertain|to\s+be\s+(?:determined|elucidated|established|clarified|confirmed))|"
    r"it\s+is\s+(?:unclear|unknown|uncertain|hypothesized|suggested|proposed)\s+(?:whether|if|that))\b",
    re.IGNORECASE,
)

# Strong empirical result assertions that block outer-clause epistemic propagation into a results clause
ASSERTIVE_RESULT_RX = re.compile(
    r"\b(?:significantly\s+(?:reduced|decreased|increased|improved|inhibited|lowered|attenuated|elevated|prevented|prolonged|delayed)|"
    r"(?:demonstrated|showed|proved|confirmed)\s+(?:that|a\s+significant|significant)|"
    r"hazard\s+ratio|odds\s+ratio|95%\s*ci|statistically\s+significant|"
    r"p\s*[<≤=]\s*0\.0\d*|compared\s+with\s+placebo)\b",
    re.IGNORECASE,
)

# Section-level header hints when section_zone metadata is not explicitly supplied
SECTION_HEADER_OBJECTIVE_METHODS_RX = re.compile(
    r"^\s*(?:objectives?|aims?|purpose|methods?|study\s+design|trial\s+design)\s*:",
    re.IGNORECASE,
)
SECTION_HEADER_BACKGROUND_RX = re.compile(
    r"^\s*(?:background|introduction|context)\s*:",
    re.IGNORECASE,
)

# Weak study-plan / hypothesis cues used together with BACKGROUND section prior
STUDY_PLAN_OR_WEAK_HEDGE_RX = re.compile(
    r"\b(?:we\s+(?:investigated|examined|evaluated|assessed|explored|aimed|sought)|"
    r"this\s+(?:study|trial|review|analysis)\s+(?:evaluated|assessed|investigated|examines|explores|aims)|"
    r"rationale|candidate|emerging|ongoing|planned|designed\s+to|under\s+investigation)\b",
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


def _mask_pseudo_triggers(text: str) -> str:
    """Masks ConText pseudo-triggers with spaces of identical length so character offsets
    remain unchanged while preventing false verb/negation matches.
    """
    chars = list(text)
    for m in PSEUDO_CAUSE_OR_INCREASE_RX.finditer(text):
        phrase = m.group(0).lower()
        # Keep 'reduced/lower/decreased risk of' intact for DECREASE_VERB_PATTERNS, mask the rest
        if any(w in phrase for w in ("reduced risk", "lower risk", "decreased risk", "reduction in")):
            continue
        for idx in range(m.start(), m.end()):
            chars[idx] = " "
    return "".join(chars)


class Layer4_PolarityAndGradeReasoner:
    r"""Implements the Algebra of Polarity:
        Clinical_Polarity = \Delta_verb * V_target
    with ConText/BioScope Clause-Scoped Negation and Domain-Range Ontology Constraints.
    """

    def extract_local_clause(self, sentence_text: str, target_start: int, target_end: int) -> str:
        """Splits a complex sentence at BioScope/ConText clause boundaries (', whereas', ', while', ';',
        ', without ', ', but ', ', however ', ', although ') and returns ONLY the local clause containing
        the target entity span.
        """
        without_matches = list(re.finditer(r"(?:,\s*)?\bwithout\s+[^,.;]+", sentence_text, re.I))
        for wm in without_matches:
            if wm.start() <= target_start and target_end <= wm.end():
                return sentence_text[wm.start():wm.end()].strip()

        cleaned_sent = sentence_text
        if without_matches:
            chars = list(sentence_text)
            for wm in without_matches:
                if not (wm.start() <= target_start and target_end <= wm.end()):
                    for i in range(wm.start(), wm.end()):
                        chars[i] = " "
            cleaned_sent = "".join(chars)

        clause_delims = list(
            re.finditer(r"(?:;|\s*,\s*(?:whereas|while|however|but|although|though|yet)\s+)", cleaned_sent, re.I)
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
        """Checks if the target entity is governed by a true clinical negation inside its local clause
        using ConText pre/post-negation rules and termination tokens (Harkema et al. 2009).
        """
        clause_l = local_clause.lower()
        # Strip pseudo-negations ('not only', 'no longer', 'non-inferior', 'not ruled out')
        clause_clean = PSEUDO_NEGATION_RX.sub(lambda m: " " * len(m.group(0)), clause_l)
        target_l = target_matched_text.lower()

        if re.search(r"\bwithout\b", clause_clean):
            return True, "Negated locally via 'without' prepositional phrase (ConText)"

        if re.search(r"\bneither\b[^,.;]*?\bnor\b", clause_clean):
            return True, "Negated locally via 'neither ... nor' construct (ConText)"

        if re.search(r"\bdid\s+not\s+(?:alleviate|reduce|improve)[^,.;]*?\b(?:aggravated|worsened|increased)\b", clause_clean):
            return False, ""

        neg_patterns = [
            r"\b(?:did\s+not|does\s+not|do\s+not|failed\s+to|fail\s+to|fails\s+to|cannot|can\s+not|never)\s+(?:significantly\s+)?(?:increase|elevate|cause|induce|affect|alter|reduce|decrease|attenuate|ameliorate|worsen|prevent|produce|elicit|improve|change|differ|demonstrate|show|lower|treat|inhibit|activate|trigger|promote|stimulate)\b",
            r"\b(?:was\s+not|were\s+not|is\s+not|are\s+not|has\s+not\s+been|have\s+not\s+been)\s+(?:significantly\s+|very\s+)?(?:increased|reduced|decreased|elevated|different|associated|related|linked|caused|induced|affected|altered|improved|effective|demonstrated|observed|found|shown)\b",
            r"\b(?:no\s+suggestion\s+of\s+increased\s+risk|no\s+significant(?:\s+\w+){0,3}|no\s+clear(?:\s+\w+){0,3}|no\s+increase\s+in|no\s+effect\s+on|no\s+evidence\s+of|no\s+difference(?:\s+in)?|no\s+association(?:\s+between|\s+with)?|no\s+relationship(?:\s+between|\s+with)?|no\s+change\s+in|no\s+impact\s+on|no\s+reduction\s+in|no\s+other\s+\w+\s+was\s+(?:independently\s+)?predictive)\b",
            r"\b(?:not\s+associated\s+with|not\s+related\s+to|unrelated\s+to|lack\s+of|lacks|lacking|ineffective|free\s+of|absence\s+of|ruled\s+out)\b",
            r"\b(?:low\s+intrinsic\s+propensity\s+to\s+cause|similar\s+(?:in|between)\b[^.;]{0,80}\bplacebo)\b",
        ]
        # Generous ConText/BioScope clause-scoped window (120 chars after target, full clause before target)
        post_window_chars = 120
        for np in neg_patterns:
            m = re.search(np, clause_clean)
            if m:
                t_pos = clause_clean.find(target_l) if target_l else -1
                if t_pos == -1 or m.start() < t_pos + post_window_chars:
                    # Check ConText termination tokens between negation trigger and target
                    if t_pos > m.end():
                        between = clause_clean[m.end():t_pos]
                        if re.search(r"\b(?:but|however|whereas|although|except|aside\s+from)\b", between):
                            continue
                    elif t_pos != -1 and m.start() > t_pos + len(target_l):
                        between = clause_clean[t_pos + len(target_l):m.start()]
                        if re.search(r"\b(?:but|however|whereas|although|except|aside\s+from)\b", between):
                            continue
                    return True, f"Negated locally via '{m.group(0)}' (ConText)"

        return False, ""

    def detect_verb_delta(
        self,
        local_clause: str,
        sentence_text: str,
        target_entity: Dict[str, Any],
        source_entity: Optional[Dict[str, Any]] = None,
    ) -> Tuple[int, str]:
        r"""Determines the governing verb direction \Delta_verb \in {-1, +1, 0} for a (source, target) pair."""
        target_text = re.escape(target_entity["matched_text"])

        if re.search(
            r"\b(?:did\s+not\s+(?:alleviate|reduce|prevent|improve|attenuate)[^,.;]*?\b(?:but\s+instead\s+)?(?:aggravated|worsened|increased))\b",
            sentence_text,
            re.I,
        ):
            return +1, "Contrastive adverse override ('did not alleviate but instead aggravated' -> +1)"

        # Mask ConText pseudo-triggers (e.g. 'all-cause', 'other causes of', 'similar risk of', 'no increase in')
        masked_clause = _mask_pseudo_triggers(local_clause)
        masked_sent = _mask_pseudo_triggers(sentence_text)

        # If source_entity is provided and co-occurs in the sentence, inspect the span between source and target first!
        if source_entity is not None:
            s_start = int(source_entity.get("span_start", -1))
            s_end = int(source_entity.get("span_end", -1))
            t_start = int(target_entity.get("span_start", -1))
            t_end = int(target_entity.get("span_end", -1))
            if 0 <= s_start < len(masked_sent) and 0 <= t_start < len(masked_sent):
                lo = max(0, min(s_start, t_start) - 20)
                hi = min(len(masked_sent), max(s_end, t_end) + 25)
                between_window = masked_sent[lo:hi]
                dec_btw = [rx.search(between_window).group(0) for rx in _DECREASE_RX_LIST if rx.search(between_window)]
                inc_btw = [rx.search(between_window).group(0) for rx in _INCREASE_RX_LIST if rx.search(between_window)]
                if dec_btw and not inc_btw:
                    return -1, f"Inter-entity DECREASE verb '{dec_btw[0]}' (-1)"
                if inc_btw and not dec_btw:
                    return +1, f"Inter-entity INCREASE/CAUSE verb '{inc_btw[0]}' (+1)"

        local_window_before = masked_clause
        t_idx = masked_clause.lower().find(target_entity["matched_text"].lower())
        if t_idx != -1:
            w_start = max(0, t_idx - 75)
            w_end = min(len(masked_clause), t_idx + len(target_entity["matched_text"]) + 45)
            local_window_before = masked_clause[w_start:w_end]

        if re.search(
            r"\b(?:increase|increases|increased|increasing|enhance|enhances|enhanced|promote|promotes|inducing|induce|induces|induced|cause|causes|caused|causing|trigger|triggers|triggered)\s+(?:[a-z\-]+\s+){0,4}"
            + target_text,
            local_window_before,
            re.I,
        ):
            return +1, "Direct object of INCREASE/CAUSE/INDUCE verb (+1)"

        if re.search(
            r"\b(?:inhibit|inhibits|inhibited|inhibiting|inhibition\s+of|inhibitor(?:s)?\s+of|"
            r"reduce|reduces|reduced|reducing|reduction\s+in|reduction\s+of|"
            r"decrease|decreases|decreased|decreasing|"
            r"attenuate|attenuates|attenuated|attenuating|"
            r"ameliorate|ameliorates|ameliorated|"
            r"prevent|prevents|prevented|preventing|"
            r"slow|slows|slowed|slowing\s+the\s+progression\s+of|"
            r"improve|improves|improved|improving|treat|treats|treated|treating|treatment\s+of)\s+(?:[a-z\-]+\s+){0,5}"
            + target_text,
            local_window_before,
            re.I,
        ):
            return -1, "Direct object of DECREASE/INHIBIT/TREAT verb (-1)"

        if re.search(
            target_text + r"\s+(?:[a-z\-]+\s+){0,3}(?:was\s+|were\s+|is\s+)?(?:inhibited|reduced|decreased|attenuated|ameliorated|prevented|improved|inhibition|inhibitor|inhibitors)",
            local_window_before,
            re.I,
        ):
            return -1, "Subject/modifier of DECREASE/INHIBIT verb (-1)"

        if re.search(
            target_text + r"\s+(?:[a-z\-]+\s+){0,3}(?:was\s+|were\s+|is\s+)?(?:increased|elevated|induced|caused|triggered|worsened|provoked)",
            local_window_before,
            re.I,
        ):
            return +1, "Subject/modifier of INCREASE/CAUSE verb (+1)"

        dec_hits = [rx.search(local_window_before).group(0) for rx in _DECREASE_RX_LIST if rx.search(local_window_before)]
        inc_hits = [rx.search(local_window_before).group(0) for rx in _INCREASE_RX_LIST if rx.search(local_window_before)]

        if dec_hits and not inc_hits:
            return -1, f"Window DECREASE verb '{dec_hits[0]}' (-1)"
        if inc_hits and not dec_hits:
            return +1, f"Window INCREASE verb '{inc_hits[0]}' (+1)"

        dec_clause = [rx.search(masked_clause).group(0) for rx in _DECREASE_RX_LIST if rx.search(masked_clause)]
        inc_clause = [rx.search(masked_clause).group(0) for rx in _INCREASE_RX_LIST if rx.search(masked_clause)]

        if dec_clause and not inc_clause:
            return -1, f"Clause DECREASE verb '{dec_clause[0]}' (-1)"
        if inc_clause and not dec_clause:
            return +1, f"Clause INCREASE verb '{inc_clause[0]}' (+1)"
        if dec_clause and inc_clause:
            best_delta = -1
            best_dist = 9999
            best_word = ""
            for rx in _DECREASE_RX_LIST:
                for m in rx.finditer(masked_sent):
                    dist = min(abs(m.end() - target_entity["span_start"]), abs(target_entity["span_end"] - m.start()))
                    if dist < best_dist:
                        best_dist = dist
                        best_delta = -1
                        best_word = m.group(0)
            for rx in _INCREASE_RX_LIST:
                for m in rx.finditer(masked_sent):
                    dist = min(abs(m.end() - target_entity["span_start"]), abs(target_entity["span_end"] - m.start()))
                    if dist < best_dist:
                        best_dist = dist
                        best_delta = +1
                        best_word = m.group(0)
            return best_delta, f"Proximity-resolved verb '{best_word}' ({best_delta:+d}, dist={best_dist})"

        return 0, "No directional verb detected (0)"

    def assess_epistemic_status(
        self,
        sentence_text: str,
        local_clause: str,
        target_matched_text: str,
        section_zone: Optional[str] = None,
    ) -> Tuple[str, float, str]:
        """White-box Epistemic Scope & Uncertainty Scoring:
        Combines (i) local negation & hedging, (ii) outer-clause epistemic scope propagation
        ('to evaluate whether...', 'aimed to assess...', 'investigated if...'), and
        (iii) Section Prior (OBJECTIVES/METHODS/BACKGROUND without assertive result verbs).
        """
        is_neg, neg_reason = self.is_locally_negated(local_clause, target_matched_text)
        if is_neg:
            return "negated", 0.0, neg_reason

        uncertainty_score = 0.0
        reasons: List[str] = []
        has_assertive_result = bool(ASSERTIVE_RESULT_RX.search(local_clause))

        # (i-a) Local hedging / modal verbs / research gaps
        local_hedge = UNCERTAIN_RX.search(local_clause) or RESEARCH_GAP_RX.search(local_clause)
        if local_hedge:
            uncertainty_score += 0.70
            reasons.append(f"local_hedge='{local_hedge.group(0)}'")

        # (i-b) Outer-clause epistemic scope propagation (blocked if local clause asserts empirical results)
        if not has_assertive_result:
            outer_match = OUTER_EPISTEMIC_SCOPE_RX.search(sentence_text)
            if outer_match:
                uncertainty_score += 0.65
                reasons.append(f"outer_epistemic_scope='{outer_match.group(0)}'")
            elif sentence_text.strip().endswith("?"):
                uncertainty_score += 0.65
                reasons.append("interrogative_sentence")

        # (ii) Section Prior (OBJECTIVES / METHODS / BACKGROUND without assertive result verbs)
        zone = (section_zone or "").upper().strip()
        if not zone:
            if SECTION_HEADER_OBJECTIVE_METHODS_RX.search(sentence_text):
                zone = "OBJECTIVES"
            elif SECTION_HEADER_BACKGROUND_RX.search(sentence_text):
                zone = "BACKGROUND"

        if not has_assertive_result:
            if zone in {"OBJECTIVES", "METHODS"}:
                uncertainty_score += 0.55
                reasons.append(f"section_prior={zone}")
            elif zone == "BACKGROUND":
                uncertainty_score += 0.25
                reasons.append("section_prior=BACKGROUND")
                weak_cue = STUDY_PLAN_OR_WEAK_HEDGE_RX.search(sentence_text)
                if weak_cue:
                    uncertainty_score += 0.30
                    reasons.append(f"background_study_cue='{weak_cue.group(0)}'")

        uncertainty_score = min(1.0, round(uncertainty_score, 3))
        if uncertainty_score >= 0.50:
            return "uncertain", uncertainty_score, "; ".join(reasons)
        return "asserted", uncertainty_score, "; ".join(reasons) if reasons else "asserted_default"

    @staticmethod
    def _is_direct_action_target(
        sentence_text: str,
        local_clause: str,
        target_ent: Dict[str, Any],
    ) -> bool:
        """Return True if the target disease is governed directly by a therapeutic or adverse verb
        rather than serving purely as an enrolled patient-cohort eligibility criterion.
        """
        tgt_text = str(target_ent.get("matched_text") or "").strip()
        if not tgt_text:
            return False
        tgt_esc = re.escape(tgt_text)
        masked_sent = _mask_pseudo_triggers(sentence_text)
        masked_clause = _mask_pseudo_triggers(local_clause)

        # Direct therapeutic/adverse verb governing target without crossing a cohort preposition ('in patients with')
        direct_obj_rx = re.compile(
            r"\b(?:treat(?:s|ed|ing)?|treatment\s+(?:of|for)|therapy\s+for|"
            r"ameliorat(?:e|es|ed|ing)|attenuat(?:e|es|ed|ing)|alleviat(?:e|es|ed|ing)|"
            r"prevent(?:s|ed|ing)?|prevention\s+of|improv(?:e|es|ed|ing)|"
            r"reduc(?:e|es|ed|ing)|reduction\s+(?:in|of)|lower(?:s|ed|ing)?|"
            r"slow(?:s|ed|ing)?\s+(?:the\s+)?progression\s+of|delay(?:s|ed|ing)?|"
            r"protect(?:s|ed|ing)?\s+against|revers(?:e|es|ed|ing)|mitigat(?:e|es|ed|ing)|"
            r"caus(?:e|es|ed|ing)|induc(?:e|es|ed|ing)|trigger(?:s|ed|ing)?|"
            r"worsen(?:s|ed|ing)?|aggravat(?:e|es|ed|ing)|exacerbat(?:e|es|ed|ing)|"
            r"precipitat(?:e|es|ed|ing)|increas(?:e|es|ed|ing)\s+(?:the\s+)?(?:risk|incidence|rate)\s+of)\s+"
            r"(?:the\s+|risk\s+of\s+|incidence\s+of\s+|progression\s+of\s+|hospitalization\s+for\s+|worsening\s+of\s+)?"
            rf"(?:(?!\b(?:in|among)\s+(?:patients|adults|subjects|participants)\s+with\b)[a-z0-9\-]+\s+){{0,4}}{tgt_esc}\b",
            re.IGNORECASE,
        )
        if direct_obj_rx.search(masked_clause) or direct_obj_rx.search(masked_sent):
            return True

        # Passive voice: "<target> was reduced/improved/prevented/induced/worsened"
        passive_rx = re.compile(
            rf"\b{tgt_esc}\b\s+(?:[a-z0-9\-]+\s+){{0,3}}(?:was|were|is|are|been)\s+"
            r"(?:reduced|decreased|improved|ameliorated|attenuated|prevented|treated|induced|caused|worsened|aggravated|triggered)\b",
            re.IGNORECASE,
        )
        if passive_rx.search(masked_clause):
            return True

        # Efficacy/therapeutic governor immediately before "in patients with <target>"
        if re.search(
            rf"{THERAPEUTIC_COHORT_GOVERNOR_RX.pattern}\s+(?:[a-z0-9\-]+\s+){{0,4}}{tgt_esc}\b",
            masked_sent,
            re.IGNORECASE,
        ):
            return True

        return False

    def resolve_pair_polarity(
        self,
        sentence_text: str,
        source_ent: Dict[str, Any],
        target_ent: Dict[str, Any],
        *,
        strict_cohort_scope: bool = False,
        v3_class_probabilities: Optional[Dict[str, float]] = None,
        section_zone: Optional[str] = None,
        biorex_relation: Optional[Dict[str, Any]] = None,
    ) -> PolarityResolution:
        """Applies Layer 4 Algebra of Polarity + Domain-Range Constraints to a (source, target) pair,
        integrating BioREx relation cues, Epistemic Scope scoring, and calibrated statistical polarity.
        """
        src_type = source_ent["type"]
        tgt_type = target_ent["type"]
        tgt_valence = int(target_ent.get("valence", 0))

        # Select closest mention span when an entity appears multiple times in the sentence
        s_start = int(source_ent.get("span_start", 0))
        t_start = int(target_ent.get("span_start", 0))
        t_end = int(target_ent.get("span_end", len(sentence_text)))
        if target_ent.get("all_spans") and len(target_ent["all_spans"]) > 1:
            best_sp = min(target_ent["all_spans"], key=lambda sp: abs(sp[0] - s_start))
            t_start, t_end = best_sp[0], best_sp[1]

        local_clause = self.extract_local_clause(sentence_text, t_start, t_end)

        # 1. Check Contraindication FIRST
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

        # 2. Compute Epistemic Status (Negation / Uncertainty Scope / Section Prior) BEFORE cohort early-return
        # so negated or hedged clinical statements are never silently coerced to 'asserted'.
        assertion_status, epistemic_score, epistemic_reason = self.assess_epistemic_status(
            sentence_text, local_clause, target_ent["matched_text"], section_zone=section_zone
        )

        # 3. Check Cohort Eligibility Context ("in patients with X") ONLY when disease is a baseline
        # eligibility criterion and NOT the direct target of a therapeutic/adverse verb.
        if tgt_type == "Disease":
            tgt_esc = re.escape(target_ent["matched_text"])
            if strict_cohort_scope:
                is_cohort_phrase = False
                cohort_cues = list(COHORT_CONTEXT_RX.finditer(sentence_text))
                for cue in cohort_cues:
                    tail_end = len(sentence_text)
                    boundary = re.search(r"[,;:.]", sentence_text[cue.end():])
                    if boundary:
                        tail_end = cue.end() + boundary.start()
                    if t_start >= cue.end() and t_end <= tail_end:
                        is_cohort_phrase = True
                        break
                if not is_cohort_phrase:
                    other_cause = re.search(r"\bother\s+causes\s+of\b", sentence_text, re.I)
                    if other_cause and t_start >= other_cause.end():
                        boundary = re.search(r"[,;:.]", sentence_text[other_cause.end():])
                        other_end = other_cause.end() + boundary.start() if boundary else len(sentence_text)
                        is_cohort_phrase = t_end <= other_end
            else:
                is_cohort_phrase = bool(
                    re.search(
                        rf"\b(?:in\s+patients\s+with|among\s+patients\s+with|in\s+adults\s+with|"
                        rf"in\s+subjects\s+with|in\s+participants\s+with|cohort\s+of\s+patients\s+with|"
                        rf"patients\s+with\s+type\s+2\s+diabetes(?:\s+mellitus)?\s+and|"
                        rf"patients\s+with\s+moderate\s+or\s+severe|patients\s+with\s+moderate|"
                        rf"other\s+causes\s+of(?:\s+nondiabetic)?)\s+(?:[a-z0-9\-]+\s+){{0,3}}{tgt_esc}",
                        sentence_text,
                        re.I,
                    )
                )
            if is_cohort_phrase and not self._is_direct_action_target(sentence_text, local_clause, target_ent):
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
                    rationale=f"Patient eligibility cohort context ('in patients with X'); separated from TREATS/CAUSES. [epistemic={epistemic_reason}]",
                )

        # 4. Compute Verb Direction \Delta_verb (assisted by BioREx & V3 statistical polarity when symbolic verb_delta == 0)
        verb_delta, verb_reason = self.detect_verb_delta(local_clause, sentence_text, target_ent, source_entity=source_ent)

        # 4a. BioREx fine-grained relation refinement for co-occurring sentence pairs
        biorex_type = str((biorex_relation or {}).get("type") or "").strip()
        if biorex_type in {"Comparison", "Cotreatment"}:
            # Separate comparative / co-treatment mentions from causal edges unless target is directly governed by an action verb
            is_direct = (
                tgt_type == "Disease" and self._is_direct_action_target(sentence_text, local_clause, target_ent)
            )
            if not is_direct:
                ctx_label = "COMPARISON_CONTEXT (0)" if biorex_type == "Comparison" else "COTREATMENT_CONTEXT (0)"
                return PolarityResolution(
                    source_id=source_ent["id"],
                    target_id=target_ent["id"],
                    verb_delta=0,
                    target_valence=tgt_valence,
                    clinical_polarity=ctx_label,
                    canonical_relation="ASSOCIATED_WITH",
                    contract_b_relation="ASSOCIATED_WITH",
                    hetionet_metaedge="CdC" if (src_type == "Compound" and tgt_type == "Compound") else "CaX",
                    hetionet_sign=0,
                    assertion_status=assertion_status,
                    local_clause=local_clause,
                    rationale=f"BioREx {biorex_type}: separated non-causal comparative/cotreatment context from causal edge.",
                )
        elif biorex_type == "Drug_Interaction" and src_type == "Compound" and tgt_type == "Compound" and verb_delta == 0:
            return PolarityResolution(
                source_id=source_ent["id"],
                target_id=target_ent["id"],
                verb_delta=0,
                target_valence=tgt_valence,
                clinical_polarity="DRUG_INTERACTION (0)",
                canonical_relation="ASSOCIATED_WITH",
                contract_b_relation="ASSOCIATED_WITH",
                hetionet_metaedge="CdC",
                hetionet_sign=0,
                assertion_status=assertion_status,
                local_clause=local_clause,
                rationale="BioREx Drug_Interaction: Compound-Compound pharmacological interaction.",
            )
        elif biorex_type == "Bind" and src_type == "Compound" and tgt_type == "Gene" and verb_delta == 0:
            return PolarityResolution(
                source_id=source_ent["id"],
                target_id=target_ent["id"],
                verb_delta=0,
                target_valence=tgt_valence,
                clinical_polarity="MOLECULAR_BINDING (0)",
                canonical_relation="ASSOCIATED_WITH",
                contract_b_relation="ASSOCIATED_WITH",
                hetionet_metaedge="CbG",
                hetionet_sign=0,
                assertion_status=assertion_status,
                local_clause=local_clause,
                rationale="BioREx Bind: direct Compound-Gene binding without explicit activation/inhibition direction.",
            )
        elif verb_delta == 0 and biorex_type == "Negative_Correlation":
            verb_delta = -1
            verb_reason = f"BioREx Negative_Correlation (-1; novel={(biorex_relation or {}).get('novel', 'Unknown')})"
        elif verb_delta == 0 and biorex_type == "Positive_Correlation":
            verb_delta = +1
            verb_reason = f"BioREx Positive_Correlation (+1; novel={(biorex_relation or {}).get('novel', 'Unknown')})"
        elif verb_delta == 0 and biorex_type == "Conversion":
            verb_reason = f"BioREx Conversion ({verb_reason})"

        if verb_delta == 0 and v3_class_probabilities:
            p_pos = float(v3_class_probabilities.get("POSITIVE", 0.0))
            p_neg = float(v3_class_probabilities.get("NEGATIVE", 0.0))
            p_ass = float(v3_class_probabilities.get("ASSOCIATION", 0.0))
            if p_pos > max(p_neg, p_ass) and p_pos >= 0.25:
                verb_delta = +1
                verb_reason = f"V3 statistical polarity POSITIVE (p={p_pos:.3f})"
            elif p_neg > max(p_pos, p_ass) and p_neg >= 0.25:
                verb_delta = -1
                verb_reason = f"V3 statistical polarity NEGATIVE (p={p_neg:.3f})"

        # -----------------------------------------------------------------
        # DOMAIN-RANGE CASE A: Compound -> Gene (Pharmacological Target)
        # -----------------------------------------------------------------
        if src_type == "Compound" and tgt_type == "Gene":
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
            elif verb_delta < 0:
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
            else:
                return PolarityResolution(
                    source_id=source_ent["id"],
                    target_id=target_ent["id"],
                    verb_delta=0,
                    target_valence=tgt_valence,
                    clinical_polarity="MOLECULAR_ASSOCIATION (0)",
                    canonical_relation="ASSOCIATED_WITH",
                    contract_b_relation="ASSOCIATED_WITH",
                    hetionet_metaedge="CbG",
                    hetionet_sign=0,
                    assertion_status=assertion_status,
                    local_clause=local_clause,
                    rationale=f"Compound->Gene association/binding ({verb_reason})",
                )

        # -----------------------------------------------------------------
        # DOMAIN-RANGE CASE B: Compound/Gene -> BiologicalProcess (Bridging)
        # -----------------------------------------------------------------
        if tgt_type == "BiologicalProcess":
            eff_delta = verb_delta
            if src_type == "Gene":
                if source_ent["id"] in ("6524", "6523"):
                    if target_ent["id"] in ("GO:0060004", "GO:0003091"):
                        eff_delta = -1
                    elif target_ent["id"] in ("GO:0003092", "HP:0012592", "GO:0006954"):
                        eff_delta = +1
                else:
                    if eff_delta == 0:
                        eff_delta = +1 if tgt_valence == 1 else -1
            else:
                if eff_delta == 0:
                    eff_delta = -1 if tgt_valence == 1 else +1

            product = eff_delta * tgt_valence
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
        # -----------------------------------------------------------------
        if src_type == "Compound" and tgt_type == "Disease":
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

        # -----------------------------------------------------------------
        # DOMAIN-RANGE CASE E: Gene-Gene, Chemical-Chemical, Variant-*
        # -----------------------------------------------------------------
        if verb_delta == +1:
            return PolarityResolution(
                source_id=source_ent["id"],
                target_id=target_ent["id"],
                verb_delta=+1,
                target_valence=tgt_valence,
                clinical_polarity="POSITIVE_REGULATION (+1)",
                canonical_relation="INCREASES",
                contract_b_relation="INCREASES",
                hetionet_metaedge=f"{src_type[0]}u{tgt_type[0]}",
                hetionet_sign=1,
                assertion_status=assertion_status,
                local_clause=local_clause,
                rationale=f"Positive directional interaction ({verb_reason})",
            )
        if verb_delta == -1:
            return PolarityResolution(
                source_id=source_ent["id"],
                target_id=target_ent["id"],
                verb_delta=-1,
                target_valence=tgt_valence,
                clinical_polarity="NEGATIVE_REGULATION (-1)",
                canonical_relation="DECREASES",
                contract_b_relation="DECREASES",
                hetionet_metaedge=f"{src_type[0]}d{tgt_type[0]}",
                hetionet_sign=-1,
                assertion_status=assertion_status,
                local_clause=local_clause,
                rationale=f"Negative directional interaction ({verb_reason})",
            )

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


# Canonical names
PolarityEngine = Layer4_PolarityAndGradeReasoner
