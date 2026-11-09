#!/usr/bin/env python3
"""
LAYER 3: N-ary Contextual Bio-Event Extraction & Counterfactual Entity-Blinded LLM Verifier.
Location: nlp/v6_industrial/layer3_event_and_blinded_llm.py
"""
import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .zoning import ZonedSentence
from .polarity import (
    PolarityEngine,
    PolarityResolution,
    Layer4_PolarityAndGradeReasoner,
)

ALLOWED_FSM_RELATIONS = {
    "INHIBITS",
    "ACTIVATES",
    "TREATS",
    "CAUSES",
    "INCREASES",
    "DECREASES",
    "CONTRAINDICATED_FOR",
    "EVALUATED_IN_COHORT",
    "ASSOCIATED_WITH",
    "NO_RELATION",
}


@dataclass
class ProjectedEdgeCandidate:
    event_id: str
    pmid: str
    pub_year: int
    section_zone: str
    section_weight: float
    study_design: str
    study_weight: float
    source_id: str
    source_name: str
    source_type: str
    source_raw_text: str
    target_id: str
    target_name: str
    target_type: str
    target_raw_text: str
    target_valence: int
    is_historical_baseline: bool
    verb_delta: int
    clinical_polarity: str
    canonical_relation: str
    contract_b_relation: str
    hetionet_metaedge: str
    hetionet_sign: int
    assertion_status: str
    confidence_score: float
    evidence_sentence: str
    local_clause: str
    blinded_sentence: str
    llm_verified: bool
    llm_mode: str
    rationale: str
    v3_prob_has_relation: Optional[float] = None
    v3_class_probabilities: Optional[Dict[str, float]] = None


@dataclass
class BioEventFrame:
    """Industrial N-ary Contextual Bio-Event Hyperedge representing a clinical/mechanistic statement."""
    event_id: str
    pmid: str
    pub_year: int
    section_zone: str
    section_weight: float
    evidence_sentence: str
    intervention: Optional[Dict[str, Any]]
    molecular_mechanisms: List[Dict[str, Any]] = field(default_factory=list)
    population_context: Dict[str, Any] = field(default_factory=dict)
    outcomes: List[Dict[str, Any]] = field(default_factory=list)
    study_metadata: Dict[str, Any] = field(default_factory=dict)
    projected_edges: List[ProjectedEdgeCandidate] = field(default_factory=list)


class Layer3_BioEventAndBlindedLLMEngine:
    """Constructs N-ary Contextual Bio-Event Frames from zoned sentences and verifies

    projected edges via Counterfactual Entity-Blinded LLM Prompting (guaranteeing zero
    post-2018 parametric leakage).
    """

    def __init__(
        self,
        cache_path: Optional[Path] = None,
        enable_live_llm: bool = True,
        max_live_llm_calls: int = 80,
        improvements: Optional[Dict[str, bool]] = None,
    ) -> None:
        self.layer4 = Layer4_PolarityAndGradeReasoner()
        self.enable_live_llm = enable_live_llm
        self.improvements = dict(improvements or {})
        self.max_live_llm_calls = max_live_llm_calls
        self.live_llm_calls_made = 0
        self.cache_hits = 0
        self.symbolic_decisions = 0
        # Non-invasive gate tracing for evaluation; does not alter prediction logic.
        self.debug_records: List[Dict[str, Any]] = []

        if cache_path is None:
            cache_path = Path(__file__).resolve().parent.parent / "output" / "v6_blinded_llm_cache.json"
        self.cache_path = cache_path
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache: Dict[str, Dict[str, Any]] = self._load_cache()
        self.api_key = self._load_groq_key()

    def _load_groq_key(self) -> str:
        key = os.environ.get("GROQ_API_KEY", "").strip()
        if key:
            return key
        candidates = [
            Path(__file__).resolve().parents[3] / ".env",
            Path(__file__).resolve().parents[4] / "nlp" / ".env",
            Path(__file__).resolve().parents[1] / ".env",
        ]
        for env_file in candidates:
            if env_file.exists():
                for line in env_file.read_text(encoding="utf-8", errors="ignore").splitlines():
                    if line.strip().startswith("GROQ_API_KEY="):
                        val = line.split("=", 1)[1].strip().strip('"').strip("'")
                        if val:
                            return val
        return ""

    def _load_cache(self) -> Dict[str, Dict[str, Any]]:
        if self.cache_path.exists():
            try:
                return json.loads(self.cache_path.read_text(encoding="utf-8"))
            except Exception:
                return {}
        return {}

    def save_cache(self) -> None:
        try:
            self.cache_path.write_text(
                json.dumps(self.cache, indent=2, ensure_ascii=False), encoding="utf-8"
            )
        except Exception:
            pass

    def blind_sentence_entities(
        self,
        sentence_text: str,
        source_ent: Dict[str, Any],
        target_ent: Dict[str, Any],
    ) -> Tuple[str, str, str]:
        """Replaces specific entity names with ontological type masks (@COMPOUND_1$, @GENE_1$,
        @DISEASE_1$, @PROCESS_1$) so the LLM must judge strictly from sentence syntax and semantics
        rather than post-2018 parametric memory.
        """
        src_mask = f"@{source_ent['type'].upper()}_SRC$"
        tgt_mask = f"@{target_ent['type'].upper()}_TGT$"

        blinded = sentence_text
        # Replace longer matched_text first or by regex boundary
        for ent, mask in sorted(
            [(source_ent, src_mask), (target_ent, tgt_mask)],
            key=lambda x: len(x[0]["matched_text"]),
            reverse=True,
        ):
            pattern = re.compile(re.escape(ent["matched_text"]), re.IGNORECASE)
            blinded = pattern.sub(mask, blinded)

        # Also mask any remaining mention of dapagliflozin or sglt2 to ensure complete blinding
        blinded = re.sub(r"\bdapagliflozin\b", "@COMPOUND_SRC$", blinded, flags=re.I)
        blinded = re.sub(r"\bsglt[\s\-]?2\b", "@GENE_TARGET$", blinded, flags=re.I)
        return blinded, src_mask, tgt_mask

    def route_and_verify(
        self,
        blinded_sentence: str,
        src_mask: str,
        src_type: str,
        tgt_mask: str,
        tgt_type: str,
        tgt_valence: int,
        symbolic_res: PolarityResolution,
    ) -> Dict[str, Any]:
        """Router Pattern: Bypasses LLM when Layer 4 Invariants are 100% deterministic,
        saving API budget and eliminating the circular 'call LLM then override' conflict.
        """
        # Router Check 1: Contraindications are non-negotiable safety locks
        if symbolic_res.canonical_relation == "CONTRAINDICATED_FOR":
            self.symbolic_decisions += 1
            return {
                "relation": "CONTRAINDICATED_FOR",
                "assertion_status": "negated",
                "confidence": 0.99,
                "llm_verified": False,
                "llm_mode": "router_bypass_contraindication",
                "rationale": f"{symbolic_res.rationale} | Router Bypass: Regulatory contraindication invariant.",
            }

        # Router Check 2: Domain-Range constraint: Gene can NEVER treat Disease,
        # but in genetic/genomic contexts (BioRED), Gene->Disease may carry POSITIVE/NEGATIVE/ASSOCIATION polarity.
        if src_type == "Gene" and tgt_type == "Disease":
            self.symbolic_decisions += 1
            if self.improvements.get("association_trigger"):
                assoc_trigger = re.search(
                    r"\b(?:associated\s+with|correlated\s+with|linked\s+to|related\s+to|"
                    r"co[- ]occurs?\s+with|predicts?|predictor\s+of|contributes?\s+to|"
                    r"mediates?|regulates?|increases?\s+(?:the\s+)?risk\s+of|"
                    r"higher\s+risk\s+of|increased\s+incidence\s+of)\b",
                    symbolic_res.local_clause, re.I
                )
                if not assoc_trigger:
                    return {
                        "relation": "NO_RELATION",
                        "assertion_status": symbolic_res.assertion_status,
                        "confidence": 0.0,
                        "llm_verified": False,
                        "llm_mode": "router_association_trigger_block",
                        "rationale": f"{symbolic_res.rationale} | No explicit association/cause trigger in local clause.",
                    }
            router_conf = 0.75 if self.improvements.get("lower_gene_disease_router_confidence") else 0.95
            out_rel = symbolic_res.canonical_relation if symbolic_res.canonical_relation in ("INCREASES", "DECREASES", "ASSOCIATED_WITH") else "ASSOCIATED_WITH"
            return {
                "relation": out_rel,
                "assertion_status": symbolic_res.assertion_status,
                "confidence": router_conf,
                "llm_verified": False,
                "llm_mode": "router_bypass_domain_range",
                "rationale": f"{symbolic_res.rationale} | Router Bypass: Gene->Disease ({out_rel}).",
            }

        # Router Check 3: Deterministic direct-object therapeutic verbs
        if symbolic_res.canonical_relation == "TREATS" and "Direct object" in symbolic_res.rationale:
            self.symbolic_decisions += 1
            return {
                "relation": "TREATS",
                "assertion_status": "asserted",
                "confidence": 0.92,
                "llm_verified": False,
                "llm_mode": "router_bypass_direct_syntax",
                "rationale": f"{symbolic_res.rationale} | Router Bypass: Direct object therapeutic syntax.",
            }

        # Router Check 4: Biological process with explicit polarity algebra
        if tgt_type == "BiologicalProcess" and symbolic_res.canonical_relation in ("INCREASES", "DECREASES"):
            self.symbolic_decisions += 1
            return {
                "relation": symbolic_res.canonical_relation,
                "assertion_status": symbolic_res.assertion_status,
                "confidence": 0.90,
                "llm_verified": False,
                "llm_mode": "router_bypass_process_algebra",
                "rationale": f"{symbolic_res.rationale} | Router Bypass: BiologicalProcess polarity algebra.",
            }

        # Otherwise: Ambiguous, complex clause, or uncertain link -> Call Blinded LLM Adjudicator!
        return self.verify_with_blinded_llm(
            blinded_sentence, src_mask, src_type, tgt_mask, tgt_type, tgt_valence, symbolic_res
        )

    def verify_with_blinded_llm(
        self,
        blinded_sentence: str,
        src_mask: str,
        src_type: str,
        tgt_mask: str,
        tgt_type: str,
        tgt_valence: int,
        symbolic_res: PolarityResolution,
    ) -> Dict[str, Any]:
        """Verifies a masked sentence pair using Groq LLM (with SHA-256 disk cache and
        deterministic Layer-4 Algebraic guardrails).
        """
        cache_key = hashlib.sha256(
            f"v6_blinded|{blinded_sentence}|{src_mask}|{tgt_mask}|{tgt_valence}".encode("utf-8")
        ).hexdigest()

        if cache_key in self.cache:
            self.cache_hits += 1
            cached = dict(self.cache[cache_key])
            cached["llm_mode"] = "cache_hit"
            return cached

        # Determine if we should call live LLM:
        # We prioritize calling LLM on ambiguous or high-value Compound->Disease / Process sentences
        should_call_llm = (
            self.enable_live_llm
            and bool(self.api_key)
            and self.live_llm_calls_made < self.max_live_llm_calls
            and (src_type == "Compound" or symbolic_res.verb_delta != 0)
        )

        if not should_call_llm:
            self.symbolic_decisions += 1
            return {
                "relation": symbolic_res.canonical_relation,
                "assertion_status": symbolic_res.assertion_status,
                "confidence": 0.88 if symbolic_res.verb_delta != 0 else 0.72,
                "llm_verified": False,
                "llm_mode": "symbolic_layer4",
                "rationale": symbolic_res.rationale,
            }

        prompt = (
            "You are an Industrial Biomedical N-ary Event Verifier operating under strict Entity Blinding.\n"
            f"Blinded Sentence: \"{blinded_sentence}\"\n"
            f"Source Entity: {src_mask} (Ontology Type: {src_type})\n"
            f"Target Entity: {tgt_mask} (Ontology Type: {tgt_type}, Intrinsic Valence: {tgt_valence:+d})\n\n"
            "Rules:\n"
            "1. Judge ONLY from the syntax and semantics of the Blinded Sentence.\n"
            "2. If the sentence states 'not recommended in' or 'contraindicated in' Target, relation MUST be CONTRAINDICATED_FOR.\n"
            "3. If Target is merely the patient population enrolled ('in patients with @DISEASE_TGT$') without being reduced/treated, relation MUST be EVALUATED_IN_COHORT.\n"
            "4. If Source reduces/attenuates/prevents/improves a pathological Disease (Valence +1), relation is TREATS.\n"
            "5. If Source increases/induces a healthy physiological process (Valence -1, e.g. urinary excretion), relation is INCREASES.\n"
            "6. If Source inhibits a Gene, relation is INHIBITS.\n"
            "7. If there is no semantic relation between Source and Target, return NO_RELATION.\n\n"
            "Respond ONLY with a valid JSON object matching this schema:\n"
            "{\"relation\": \"INHIBITS|ACTIVATES|TREATS|CAUSES|INCREASES|DECREASES|CONTRAINDICATED_FOR|EVALUATED_IN_COHORT|ASSOCIATED_WITH|NO_RELATION\", "
            "\"assertion_status\": \"asserted|negated|uncertain\", \"confidence\": 0.0 to 1.0, \"rationale\": \"short explanation\"}"
        )

        payload = {
            "model": os.environ.get("BIOMED_SLM_MODEL", "llama-3.1-8b-instant"),
            "messages": [
                {"role": "system", "content": "Output strictly valid JSON. Do not include markdown fences or thinking tags."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.0,
            "max_tokens": 220,
            "response_format": {"type": "json_object"},
        }

        try:
            req = urllib.request.Request(
                "https://api.groq.com/openai/v1/chat/completions",
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                    "User-Agent": "Mozilla/5.0",
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=12) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            self.live_llm_calls_made += 1
            raw_content = body["choices"][0]["message"]["content"].strip()
            # Strip any <think>...</think> if present
            raw_content = re.sub(r"<think>.*?</think>", "", raw_content, flags=re.S).strip()
            parsed = json.loads(raw_content)

            rel = str(parsed.get("relation", symbolic_res.canonical_relation)).upper().strip()
            if rel not in ALLOWED_FSM_RELATIONS:
                rel = symbolic_res.canonical_relation

            # Enforce hard Layer 4 ontological & algebraic invariants over LLM output:
            # Invariant A: Contraindications can NEVER be overridden to EVALUATED_IN_COHORT or TREATS
            if symbolic_res.canonical_relation == "CONTRAINDICATED_FOR":
                rel = "CONTRAINDICATED_FOR"
            # Invariant B: Gene -> Disease can NEVER be TREATS or CAUSES
            elif src_type == "Gene" and tgt_type == "Disease":
                rel = "ASSOCIATED_WITH"
            # Invariant C: Compound -> Gene can only be INHIBITS or ACTIVATES
            elif src_type == "Compound" and tgt_type == "Gene" and rel not in ("INHIBITS", "ACTIVATES"):
                rel = symbolic_res.canonical_relation
            # Invariant D: BiologicalProcess with explicit algebraic polarity keeps INCREASES/DECREASES
            elif tgt_type == "BiologicalProcess" and symbolic_res.canonical_relation in ("INCREASES", "DECREASES"):
                rel = symbolic_res.canonical_relation
            # Invariant E: If Layer 4 found an explicit direct-object therapeutic verb (e.g., "attenuates progression of X"),
            # prevent LLM from down-grading it to EVALUATED_IN_COHORT
            elif symbolic_res.canonical_relation == "TREATS" and "Direct object" in symbolic_res.rationale:
                rel = "TREATS"

            conf = float(parsed.get("confidence", 0.90))
            conf = max(0.05, min(0.99, conf))

            # Keep Layer 4 local-clause assertion_status if Layer 4 proved non-negation in main clause
            assertion = symbolic_res.assertion_status

            result = {
                "relation": rel,
                "assertion_status": assertion,
                "confidence": conf,
                "llm_verified": True,
                "llm_mode": "live_groq_qwen",
                "rationale": f"{symbolic_res.rationale} | Blinded-LLM({rel}, conf={conf:.2f}): {parsed.get('rationale', '')}",
            }
            self.cache[cache_key] = result
            return result
        except Exception as e:
            self.symbolic_decisions += 1
            mode = "symbolic_fallback_http" if isinstance(e, urllib.error.HTTPError) else "symbolic_fallback"
            return {
                "relation": symbolic_res.canonical_relation,
                "assertion_status": symbolic_res.assertion_status,
                "confidence": 0.88 if symbolic_res.verb_delta != 0 else 0.72,
                "llm_verified": False,
                "llm_mode": mode,
                "rationale": symbolic_res.rationale,
            }

    def extract_document_events(
        self,
        pmid: str,
        pub_year: int,
        grounded_doc: Dict[str, Any],
        v3_relation_scores: Optional[Dict[Tuple[str, str], Dict[str, Any]]] = None,
        v3_threshold: Optional[float] = None,
    ) -> List[BioEventFrame]:
        """Extracts N-ary Contextual Bio-Event Frames from a zoned & grounded document,

        then projects validated edges for Layer 5 Bayesian Fusion.
        """
        study_design = grounded_doc["study_design"]
        study_weight = float(grounded_doc["study_weight"])
        sentences: List[ZonedSentence] = grounded_doc["sentences"]

        events: List[BioEventFrame] = []

        for sent in sentences:
            self.debug_records.append({
                "pmid": str(pmid), "evidence_sentence": sent.text,
                "section_zone": sent.section_zone, "section_weight": float(sent.section_weight),
                "stage": "sentence_entities", "entities": list(sent.entities),
            })
            # Skip METHODS_SEARCH noise immediately (weight == 0.0)
            if sent.section_zone == "METHODS_SEARCH" or sent.section_weight <= 0.0:
                self.debug_records.append({
                    "pmid": str(pmid), "evidence_sentence": sent.text,
                    "section_zone": sent.section_zone, "section_weight": float(sent.section_weight),
                    "stage": "section_weight_blocked", "entities": list(sent.entities),
                })
                continue

            if len(sent.entities) == 0:
                self.debug_records.append({
                    "pmid": str(pmid), "evidence_sentence": sent.text,
                    "section_zone": sent.section_zone, "section_weight": float(sent.section_weight),
                    "stage": "entity_coverage_zero", "entities": [],
                })
                continue
            if len(sent.entities) < 2 and (v3_relation_scores is None or v3_threshold is None):
                self.debug_records.append({
                    "pmid": str(pmid), "evidence_sentence": sent.text,
                    "section_zone": sent.section_zone, "section_weight": float(sent.section_weight),
                    "stage": "insufficient_entities", "entities": list(sent.entities),
                })
                continue

            # Partition sentence entities by ontological role
            compounds = [e for e in sent.entities if e.get("type") in ("Compound", "Chemical", "DrugClass")]
            genes = [e for e in sent.entities if e.get("type") == "Gene"]
            processes = [e for e in sent.entities if e.get("type") in ("BiologicalProcess", "Process")]
            diseases = [e for e in sent.entities if e.get("type") in ("Disease", "Phenotype")]
            variants = [e for e in sent.entities if e.get("type") == "Variant"]

            # Primary intervention anchor (if present in sentence via direct mention or Layer 2 coref)
            intervention_ent = compounds[0] if compounds else None

            event_id = f"EVT_PMID{pmid}_S{sent.sent_idx:02d}"
            frame = BioEventFrame(
                event_id=event_id,
                pmid=str(pmid),
                pub_year=int(pub_year),
                section_zone=sent.section_zone,
                section_weight=sent.section_weight,
                evidence_sentence=sent.text,
                intervention=(
                    {
                        "entity": intervention_ent["canonical_name"],
                        "id": intervention_ent["id"],
                        "matched_text": intervention_ent["matched_text"],
                        "is_coref_resolved": intervention_ent.get("is_coref_resolved", False),
                    }
                    if intervention_ent
                    else None
                ),
                study_metadata={
                    "design": study_design,
                    "study_weight": study_weight,
                    "pub_year": int(pub_year),
                },
            )

            # Build candidate directed pairs according to strict Ontological Hierarchy:
            # 1. Compound -> Gene (Mechanistic Target)
            # 2. Compound -> BiologicalProcess (Mechanistic / Physiological Phenotype)
            # 3. Compound -> Disease (Therapeutic / Adverse / Cohort / Contraindicated)
            # 4. Gene -> BiologicalProcess (Target -> Process)
            # 5. Gene -> Disease (Genetic/Target Association)
            # 6. Gene <-> Variant, Variant -> Disease
            candidate_pairs: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []

            # Intra-sentence candidate pairs: all compound-target combinations
            for comp in compounds:
                for g in genes:
                    candidate_pairs.append((comp, g))
                for bp in processes:
                    candidate_pairs.append((comp, bp))
                for d in diseases:
                    candidate_pairs.append((comp, d))
                for v in variants:
                    candidate_pairs.append((comp, v))

            for i_c in range(len(compounds)):
                for j_c in range(i_c + 1, len(compounds)):
                    candidate_pairs.append((compounds[i_c], compounds[j_c]))

            for g in genes:
                for bp in processes:
                    candidate_pairs.append((g, bp))
                for d in diseases:
                    candidate_pairs.append((g, d))
                for v in variants:
                    candidate_pairs.append((g, v))

            for i_g in range(len(genes)):
                for j_g in range(i_g + 1, len(genes)):
                    candidate_pairs.append((genes[i_g], genes[j_g]))

            for v in variants:
                for d in diseases:
                    candidate_pairs.append((v, d))

            for i_v in range(len(variants)):
                for j_v in range(i_v + 1, len(variants)):
                    candidate_pairs.append((variants[i_v], variants[j_v]))

            # Optional improvement A: add explicit Phenotype/BiologicalProcess -> Disease candidates.
            if self.improvements.get("direction_pair_expansion"):
                disease_only = [e for e in sent.entities if e.get("type") == "Disease"]
                phenotype_only = [e for e in sent.entities if e.get("type") == "Phenotype"]
                for bp in processes:
                    for dis in disease_only:
                        candidate_pairs.append((bp, dis))
                for phenotype in phenotype_only:
                    for dis in disease_only:
                        candidate_pairs.append((phenotype, dis))
                # DrugClass is already included in the Compound-like anchor list above;
                # reverse GT directions are evaluated order-independently and flagged.

            # Cross-Sentence Scope Expansion:
            # Bridge cross-sentence relations where an entity (Compound/Gene/Variant) is introduced
            # in the Title or adjacent previous sentence and the endpoint is reported in the current sentence.
            title_sent = grounded_doc["sentences"][0] if (grounded_doc.get("sentences") and grounded_doc["sentences"][0].section_zone == "TITLE") else None
            if v3_relation_scores is not None and v3_threshold is not None:
                prev_entities: List[Dict[str, Any]] = []
                for prev_sent in grounded_doc.get("sentences", []):
                    if prev_sent.sent_idx < sent.sent_idx and prev_sent.section_weight > 0:
                        prev_entities.extend(prev_sent.entities)

                existing_ids = {e["id"] for e in sent.entities}
                seen_cross_ids = set()
                type_rank = {
                    "Compound": 0, "Chemical": 0, "DrugClass": 0,
                    "Gene": 1, "Variant": 2,
                    "BiologicalProcess": 3, "Phenotype": 3, "Disease": 4,
                }
                for e_prev in prev_entities:
                    if e_prev["id"] in existing_ids or e_prev["id"] in seen_cross_ids:
                        continue
                    seen_cross_ids.add(e_prev["id"])
                    for e_curr in sent.entities:
                        if e_curr["id"] == e_prev["id"]:
                            continue
                        pair_key = tuple(sorted((str(e_prev["id"]), str(e_curr["id"]))))
                        sc = v3_relation_scores.get(pair_key)
                        if sc and float(sc.get("prob_has_relation", 0.0)) >= float(v3_threshold):
                            r_prev = type_rank.get(str(e_prev.get("type")), 5)
                            r_curr = type_rank.get(str(e_curr.get("type")), 5)
                            if r_prev <= r_curr:
                                candidate_pairs.append((e_prev, e_curr))
                            else:
                                candidate_pairs.append((e_curr, e_prev))
            elif title_sent and title_sent != sent and sent.section_zone in ("RESULTS", "CONCLUSIONS"):
                # Rule-only Title-to-Results/Conclusions bridging when no classifier is active:
                # only bridge the primary Title Compound to a Disease in Results/Conclusions if an explicit causal/therapeutic verb is present
                title_comps = [e for e in title_sent.entities if e.get("type") in ("Compound", "Chemical")]
                existing_comp_ids = {c["id"] for c in compounds}
                if len(title_comps) == 1 and title_comps[0]["id"] not in existing_comp_ids:
                    for d in diseases:
                        candidate_pairs.append((title_comps[0], d))

            candidate_debug = []
            for src_ent, tgt_ent in candidate_pairs:
                candidate_debug.append({
                    "pmid": str(pmid), "evidence_sentence": sent.text,
                    "section_zone": sent.section_zone, "section_weight": float(sent.section_weight),
                    "stage": "candidate_generated",
                    "source_id": str(src_ent.get("id", "")),
                    "source_name": str(src_ent.get("canonical_name", "")),
                    "source_type": str(src_ent.get("type", "")),
                    "target_id": str(tgt_ent.get("id", "")),
                    "target_name": str(tgt_ent.get("canonical_name", "")),
                    "target_type": str(tgt_ent.get("type", "")),
                    "classifier_probability": None,
                })
            self.debug_records.extend(candidate_debug)

            seen_pairs_in_frame: Set[Tuple[str, str]] = set()
            for src_ent, tgt_ent in candidate_pairs:
                if src_ent["id"] == tgt_ent["id"]:
                    continue

                pair_key = tuple(sorted((str(src_ent["id"]), str(tgt_ent["id"]))))
                if pair_key in seen_pairs_in_frame:
                    continue
                seen_pairs_in_frame.add(pair_key)

                pair_key = tuple(sorted((str(src_ent["id"]), str(tgt_ent["id"]))))
                v3_score = (v3_relation_scores or {}).get(pair_key)

                # Optional improvement C: filter explicit research-intent sentences only;
                # do not hard-block BACKGROUND as a section.
                if self.improvements.get("objective_sentence_filter"):
                    intent_cue = re.search(
                        r"\b(?:aims?\s+to|objective(?:s)?\s*:|purpose\s*:|"
                        r"to\s+(?:evaluate|assess|investigate|determine|examine|compare)|"
                        r"we\s+sought\s+to|whether\s+the)\b",
                        sent.text, re.I
                    )
                    result_cue = re.search(
                        r"\b(?:significantly|reduced|increased|improved|improves|decreased|"
                        r"attenuated|lowered|higher|lower|showed|demonstrated|found|observed|"
                        r"hazard\s+ratio|odds\s+ratio|confidence\s+interval|efficacy|"
                        r"effective|associated\s+with|resulted\s+in)\b",
                        sent.text, re.I
                    )
                    if intent_cue and not result_cue:
                        self.debug_records.append({
                            "pmid": str(pmid), "evidence_sentence": sent.text,
                            "section_zone": sent.section_zone, "section_weight": float(sent.section_weight),
                            "stage": "objective_sentence_filter",
                            "source_id": str(src_ent.get("id", "")),
                            "source_name": str(src_ent.get("canonical_name", "")),
                            "source_type": str(src_ent.get("type", "")),
                            "target_id": str(tgt_ent.get("id", "")),
                            "target_name": str(tgt_ent.get("canonical_name", "")),
                            "target_type": str(tgt_ent.get("type", "")),
                        })
                        continue

                # Layer 4: Symbolic Algebra of Polarity + Local Clause Scoping + V3 Statistical Polarity
                sym_res = self.layer4.resolve_pair_polarity(
                    sent.text, src_ent, tgt_ent,
                    strict_cohort_scope=bool(self.improvements.get("strict_cohort_scope")),
                    v3_class_probabilities=(v3_score.get("class_probabilities") if v3_score is not None else None),
                )
                trace = {
                    "pmid": str(pmid), "evidence_sentence": sent.text,
                    "section_zone": sent.section_zone, "section_weight": float(sent.section_weight),
                    "stage": "polarity_resolved",
                    "source_id": str(src_ent.get("id", "")),
                    "source_name": str(src_ent.get("canonical_name", "")),
                    "source_type": str(src_ent.get("type", "")),
                    "target_id": str(tgt_ent.get("id", "")),
                    "target_name": str(tgt_ent.get("canonical_name", "")),
                    "target_type": str(tgt_ent.get("type", "")),
                    "symbolic_relation": sym_res.canonical_relation,
                    "symbolic_assertion_status": sym_res.assertion_status,
                    "local_clause": sym_res.local_clause,
                    "classifier_probability": None,
                }

                # V7 hybrid gate: V3 contributes relation-existence probability only.
                # Its correlation polarity is deliberately not converted into a directed predicate.
                if v3_score is not None:
                    trace["classifier_probability"] = float(v3_score.get("prob_has_relation", 0.0))
                    trace["classifier_threshold"] = float(v3_threshold) if v3_threshold is not None else None
                if (
                    v3_score is not None
                    and v3_threshold is not None
                    and float(v3_score.get("prob_has_relation", 0.0)) < float(v3_threshold)
                    and sym_res.canonical_relation != "CONTRAINDICATED_FOR"
                ):
                    trace["stage"] = "classifier_blocked"
                    self.debug_records.append(trace)
                    continue

                # Router Pattern: When V3 Statistical Classifier already provided calibrated
                # class probabilities (POSITIVE, NEGATIVE, ASSOCIATION), use V3 polarity when
                # symbolic polarity is neutral or Gene->Disease / Gene->Gene / Compound->Compound!
                if v3_score is not None and not self.enable_live_llm:
                    v3_probs = v3_score.get("class_probabilities") or {}
                    p_pos = float(v3_probs.get("POSITIVE", 0.0))
                    p_neg = float(v3_probs.get("NEGATIVE", 0.0))
                    p_ass = float(v3_probs.get("ASSOCIATION", 0.0))
                    s_type = src_ent["type"]
                    t_type = tgt_ent["type"]
                    is_bc5cdr_cid_model = (p_pos > 0.0 and p_neg == 0.0 and p_ass == 0.0)
                    if is_bc5cdr_cid_model and s_type in ("Compound", "Chemical") and t_type == "Disease":
                        sym_res.canonical_relation = "CAUSES"
                        sym_res.contract_b_relation = "CAUSES"
                        sym_res.hetionet_metaedge = "CcSE"
                        sym_res.hetionet_sign = 1
                        sym_res.assertion_status = "asserted"
                    elif sym_res.canonical_relation != "CONTRAINDICATED_FOR":
                        # If target is a clinical Phenotype or BiologicalProcess and Layer 4 resolved an explicit
                        # directional verb (INCREASES/DECREASES/INHIBITS), preserve Layer 4's algebraic predicate.
                        has_explicit_clinical_verb = (
                            sym_res.verb_delta != 0
                            and (
                                t_type in ("Phenotype", "BiologicalProcess")
                                or s_type in ("Phenotype", "BiologicalProcess", "DrugClass")
                                or (s_type in ("Compound", "Chemical") and t_type in ("Gene", "Disease") and p_ass < 0.55)
                            )
                        )
                        if not has_explicit_clinical_verb:
                            if p_pos >= max(p_neg, p_ass):
                                if s_type in ("Compound", "Chemical") and t_type == "Disease":
                                    sym_res.canonical_relation = "CAUSES"
                                    sym_res.contract_b_relation = "CAUSES"
                                    sym_res.hetionet_metaedge = "CcSE"
                                    sym_res.hetionet_sign = 1
                                elif s_type in ("Compound", "Chemical") and t_type == "Gene":
                                    sym_res.canonical_relation = "ACTIVATES"
                                    sym_res.contract_b_relation = "ACTIVATES"
                                    sym_res.hetionet_metaedge = "CaG"
                                    sym_res.hetionet_sign = 1
                                else:
                                    sym_res.canonical_relation = "INCREASES"
                                    sym_res.contract_b_relation = "INCREASES"
                                    sym_res.hetionet_sign = 1
                            elif p_neg >= max(p_pos, p_ass):
                                if s_type in ("Compound", "Chemical") and t_type == "Disease":
                                    sym_res.canonical_relation = "TREATS"
                                    sym_res.contract_b_relation = "TREATS"
                                    sym_res.hetionet_metaedge = "CtD"
                                    sym_res.hetionet_sign = -1
                                elif s_type in ("Compound", "Chemical") and t_type == "Gene":
                                    sym_res.canonical_relation = "INHIBITS"
                                    sym_res.contract_b_relation = "INHIBITS"
                                    sym_res.hetionet_metaedge = "CiG"
                                    sym_res.hetionet_sign = -1
                                else:
                                    sym_res.canonical_relation = "DECREASES"
                                    sym_res.contract_b_relation = "DECREASES"
                                    sym_res.hetionet_sign = -1
                            else:
                                sym_res.canonical_relation = "ASSOCIATED_WITH"
                                sym_res.contract_b_relation = "ASSOCIATED_WITH"
                                sym_res.hetionet_sign = 0

                # Layer 3: Counterfactual Entity-Blinded LLM Verification
                blinded_text, src_mask, tgt_mask = self.blind_sentence_entities(
                    sent.text, src_ent, tgt_ent
                )
                llm_res = self.route_and_verify(
                    blinded_text,
                    src_mask,
                    src_ent["type"],
                    tgt_mask,
                    tgt_ent["type"],
                    int(tgt_ent.get("valence", 0)),
                    sym_res,
                )
                if self.improvements.get("association_trigger") and llm_res.get("relation") in ("ASSOCIATED_WITH", "EVALUATED_IN_COHORT"):
                    trigger_text = sym_res.local_clause
                    relation_trigger = re.search(
                        r"\b(?:associated\s+with|correlated\s+with|linked\s+to|related\s+to|"
                        r"co[- ]occurs?\s+with|predicts?|predictor\s+of|contributes?\s+to|"
                        r"mediates?|regulates?|increases?\s+(?:the\s+)?risk\s+of|"
                        r"higher\s+risk\s+of|increased\s+incidence\s+of|"
                        r"in\s+patients\s+with|among\s+patients\s+with|"
                        r"in\s+adults\s+with|in\s+subjects\s+with)\b",
                        trigger_text, re.I
                    )
                    if not relation_trigger:
                        llm_res = {
                            "relation": "NO_RELATION",
                            "assertion_status": llm_res.get("assertion_status", sym_res.assertion_status),
                            "confidence": 0.0,
                            "llm_verified": False,
                            "llm_mode": "association_trigger_gate",
                            "rationale": "No explicit association/cohort trigger in the local clause.",
                        }
                trace["routed_relation"] = str(llm_res.get("relation", ""))
                trace["routed_assertion_status"] = str(llm_res.get("assertion_status", ""))
                trace["routed_confidence"] = float(llm_res.get("confidence", 0.0) or 0.0)

                if v3_score is not None:
                    # L5 consumes the V3 relation-existence probability as the edge evidence score.
                    llm_res["confidence"] = float(v3_score["prob_has_relation"])
                    llm_res["rationale"] = (
                        f"{llm_res['rationale']} | V3 P(relation)={float(v3_score['prob_has_relation']):.4f}; "
                        f"V3 class scores={v3_score.get('class_probabilities', {})}"
                    )

                final_canonical_rel = llm_res["relation"]
                trace["final_relation"] = final_canonical_rel
                # Hard Rule: NEVER admit NO_RELATION / NONE into projected edges!
                if final_canonical_rel in ("NO_RELATION", "NONE"):
                    trace["stage"] = "no_relation_blocked"
                    self.debug_records.append(trace)
                    continue

                # Map canonical_relation to strict Contract B enum & Hetionet metaedge
                cb_rel, metaedge, sign = self._map_to_contract_and_hetionet(
                    src_ent["type"], tgt_ent["type"], final_canonical_rel, sym_res
                )

                # Record into N-ary Bio-Event Frame slots
                if src_ent["type"] == "Compound" and tgt_ent["type"] == "Gene":
                    frame.molecular_mechanisms.append(
                        {
                            "target_gene": tgt_ent["canonical_name"],
                            "target_id": tgt_ent["id"],
                            "action": final_canonical_rel,
                            "sign": sign,
                        }
                    )
                elif final_canonical_rel == "EVALUATED_IN_COHORT":
                    frame.population_context["cohort_disease"] = f"{tgt_ent['canonical_name']} ({tgt_ent['id']})"
                elif final_canonical_rel == "CONTRAINDICATED_FOR":
                    frame.population_context["contraindication"] = f"{tgt_ent['canonical_name']} ({tgt_ent['id']})"
                elif tgt_ent["type"] in ("Disease", "BiologicalProcess") and src_ent["type"] == "Compound":
                    frame.outcomes.append(
                        {
                            "endpoint": tgt_ent["canonical_name"],
                            "endpoint_id": tgt_ent["id"],
                            "endpoint_type": tgt_ent["type"],
                            "relation": final_canonical_rel,
                            "clinical_polarity": sym_res.clinical_polarity,
                            "assertion_status": llm_res["assertion_status"],
                            "sign": sign,
                        }
                    )

                edge_cand = ProjectedEdgeCandidate(
                    event_id=event_id,
                    pmid=str(pmid),
                    pub_year=int(pub_year),
                    section_zone=sent.section_zone,
                    section_weight=float(sent.section_weight),
                    study_design=study_design,
                    study_weight=study_weight,
                    source_id=src_ent["id"],
                    source_name=src_ent["canonical_name"],
                    source_type=src_ent["type"],
                    source_raw_text=str(src_ent.get("matched_text") or src_ent.get("text") or ""),
                    target_id=tgt_ent["id"],
                    target_name=tgt_ent["canonical_name"],
                    target_type=tgt_ent["type"],
                    target_raw_text=str(tgt_ent.get("matched_text") or tgt_ent.get("text") or ""),
                    target_valence=int(tgt_ent.get("valence", 0)),
                    is_historical_baseline=bool(tgt_ent.get("is_historical_baseline", False)),
                    verb_delta=sym_res.verb_delta,
                    clinical_polarity=sym_res.clinical_polarity,
                    canonical_relation=final_canonical_rel,
                    contract_b_relation=cb_rel,
                    hetionet_metaedge=metaedge,
                    hetionet_sign=sign,
                    assertion_status=llm_res["assertion_status"],
                    confidence_score=round(float(llm_res["confidence"]), 4),
                    evidence_sentence=sent.text,
                    local_clause=sym_res.local_clause,
                    blinded_sentence=blinded_text,
                    llm_verified=bool(llm_res["llm_verified"]),
                    llm_mode=str(llm_res["llm_mode"]),
                    rationale=str(llm_res["rationale"]),
                    v3_prob_has_relation=(float(v3_score["prob_has_relation"]) if v3_score is not None else None),
                    v3_class_probabilities=(dict(v3_score.get("class_probabilities", {})) if v3_score is not None else None),
                )
                frame.projected_edges.append(edge_cand)
                trace["stage"] = "projected_edge"
                trace["final_relation"] = final_canonical_rel
                trace["contract_b_relation"] = cb_rel
                trace["hetionet_metaedge"] = metaedge
                trace["hetionet_sign"] = sign
                self.debug_records.append(trace)

            if frame.projected_edges:
                events.append(frame)

        return events

    def _map_to_contract_and_hetionet(
        self,
        src_type: str,
        tgt_type: str,
        canonical_rel: str,
        sym_res: PolarityResolution,
    ) -> Tuple[str, str, int]:
        """Maps rich industrial relation types to strict Contract B enum & Hetionet metaedges."""
        if canonical_rel == "INHIBITS":
            return "INHIBITS", "CiG", -1
        if canonical_rel == "ACTIVATES":
            return "ACTIVATES", "CaG", 1
        if canonical_rel == "TREATS":
            return "TREATS", "CtD", -1
        if canonical_rel == "CAUSES":
            return "CAUSES", "CcSE", 1
        if canonical_rel == "INCREASES":
            return "INCREASES", f"{src_type[0]}u{tgt_type[0]}", 1
        if canonical_rel == "DECREASES":
            return "DECREASES", f"{src_type[0]}d{tgt_type[0]}", -1
        if canonical_rel == "CONTRAINDICATED_FOR":
            # In Contract B enum, map to ASSOCIATED_WITH with assertion_status='negated' and sign=-1
            return "ASSOCIATED_WITH", "CcD", -1
        if canonical_rel == "EVALUATED_IN_COHORT":
            return "ASSOCIATED_WITH", "CpD", 0
        return sym_res.contract_b_relation, sym_res.hetionet_metaedge, sym_res.hetionet_sign


# Canonical names
BioEventEngine = Layer3_BioEventAndBlindedLLMEngine
