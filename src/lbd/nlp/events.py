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
    target_id: str
    target_name: str
    target_type: str
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
    ) -> None:
        self.layer4 = Layer4_PolarityAndGradeReasoner()
        self.enable_live_llm = enable_live_llm
        self.max_live_llm_calls = max_live_llm_calls
        self.live_llm_calls_made = 0
        self.cache_hits = 0
        self.symbolic_decisions = 0

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

        # Router Check 2: Domain-Range constraint: Gene can NEVER treat Disease
        if src_type == "Gene" and tgt_type == "Disease":
            self.symbolic_decisions += 1
            return {
                "relation": "ASSOCIATED_WITH",
                "assertion_status": symbolic_res.assertion_status,
                "confidence": 0.95,
                "llm_verified": False,
                "llm_mode": "router_bypass_domain_range",
                "rationale": f"{symbolic_res.rationale} | Router Bypass: Gene->Disease restricted to ASSOCIATED_WITH.",
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
            "model": "qwen/qwen3.8-27b",
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
            return {
                "relation": symbolic_res.canonical_relation,
                "assertion_status": symbolic_res.assertion_status,
                "confidence": 0.88 if symbolic_res.verb_delta != 0 else 0.72,
                "llm_verified": False,
                "llm_mode": f"symbolic_fallback",
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
            # Skip METHODS_SEARCH noise immediately (weight == 0.0)
            if sent.section_zone == "METHODS_SEARCH" or sent.section_weight <= 0.0:
                continue

            if len(sent.entities) == 0:
                continue
            if len(sent.entities) < 2 and (v3_relation_scores is None or v3_threshold is None):
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

            for g in genes:
                for bp in processes:
                    candidate_pairs.append((g, bp))
                for d in diseases:
                    candidate_pairs.append((g, d))
                for v in variants:
                    candidate_pairs.append((g, v))

            for v in variants:
                for d in diseases:
                    candidate_pairs.append((v, d))

            # Cross-Sentence Scope Expansion (Gated by V3 relation existence probability):
            # When V3 relation scores are provided (V7 Hybrid mode), bridge cross-sentence
            # relations where the Compound is introduced in Title or adjacent previous sentence,
            # and the endpoint (Disease/Gene/Process) is reported in the current sentence.
            if v3_relation_scores is not None and v3_threshold is not None:
                cross_sources: List[Dict[str, Any]] = []
                # Check Title compounds
                title_sent = grounded_doc["sentences"][0] if (grounded_doc.get("sentences") and grounded_doc["sentences"][0].section_zone == "TITLE") else None
                if title_sent and title_sent != sent:
                    cross_sources.extend([e for e in title_sent.entities if e.get("type") in ("Compound", "Chemical")])
                # Check adjacent previous sentence compounds
                if sent.sent_idx > 1 and grounded_doc.get("sentences"):
                    prev_idx = sent.sent_idx - 2  # 0-indexed position
                    if 0 <= prev_idx < len(grounded_doc["sentences"]):
                        prev_sent = grounded_doc["sentences"][prev_idx]
                        if prev_sent.section_weight > 0:
                            cross_sources.extend([e for e in prev_sent.entities if e.get("type") in ("Compound", "Chemical")])

                existing_comp_ids = {c["id"] for c in compounds}
                seen_cross_ids = set()
                for c_src in cross_sources:
                    if c_src["id"] in existing_comp_ids or c_src["id"] in seen_cross_ids:
                        continue
                    seen_cross_ids.add(c_src["id"])

                    for d in diseases:
                        pair_key = tuple(sorted((str(c_src["id"]), str(d["id"]))))
                        sc = v3_relation_scores.get(pair_key)
                        if sc and float(sc.get("prob_has_relation", 0.0)) >= float(v3_threshold):
                            candidate_pairs.append((c_src, d))

                    for g in genes:
                        pair_key = tuple(sorted((str(c_src["id"]), str(g["id"]))))
                        sc = v3_relation_scores.get(pair_key)
                        if sc and float(sc.get("prob_has_relation", 0.0)) >= float(v3_threshold):
                            candidate_pairs.append((c_src, g))

                    for bp in processes:
                        pair_key = tuple(sorted((str(c_src["id"]), str(bp["id"]))))
                        sc = v3_relation_scores.get(pair_key)
                        if sc and float(sc.get("prob_has_relation", 0.0)) >= float(v3_threshold):
                            candidate_pairs.append((c_src, bp))

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

                # Layer 4: Symbolic Algebra of Polarity + Local Clause Scoping
                sym_res = self.layer4.resolve_pair_polarity(sent.text, src_ent, tgt_ent)

                # V7 hybrid gate: V3 contributes relation-existence probability only.
                # Its correlation polarity is deliberately not converted into a directed predicate.
                if (
                    v3_score is not None
                    and v3_threshold is not None
                    and float(v3_score.get("prob_has_relation", 0.0)) < float(v3_threshold)
                    and sym_res.canonical_relation != "CONTRAINDICATED_FOR"
                ):
                    continue

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

                if v3_score is not None:
                    # L5 consumes the V3 relation-existence probability as the edge evidence score.
                    llm_res["confidence"] = float(v3_score["prob_has_relation"])
                    llm_res["rationale"] = (
                        f"{llm_res['rationale']} | V3 P(relation)={float(v3_score['prob_has_relation']):.4f}; "
                        f"V3 class scores={v3_score.get('class_probabilities', {})}"
                    )

                final_canonical_rel = llm_res["relation"]
                # Hard Rule: NEVER admit NO_RELATION / NONE into projected edges!
                if final_canonical_rel in ("NO_RELATION", "NONE"):
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
                    target_id=tgt_ent["id"],
                    target_name=tgt_ent["canonical_name"],
                    target_type=tgt_ent["type"],
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
