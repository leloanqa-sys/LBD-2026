#!/usr/bin/env python3
"""
LAYER 5: Hierarchical Bayesian Beta-Binomial Evidence Fusion & Contract B/D Exporter.
Location: nlp/v6_industrial/layer5_bayesian_fusion.py
"""
import json
import math
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .events import (
    BioEventFrame,
    ProjectedEdgeCandidate,
)


@dataclass
class BayesianFusedEdge:
    edge_id: str
    source_id: str
    source_name: str
    source_type: str
    target_id: str
    target_name: str
    target_type: str
    canonical_relation: str
    contract_b_relation: str
    hetionet_metaedge: str
    hetionet_sign: int
    assertion_status: str
    is_historical_baseline: bool
    alpha_post: float
    beta_post: float
    posterior_mean: float
    posterior_std: float
    lcb_90: float
    unique_supporting_pmids: List[str]
    unique_contradicting_pmids: List[str]
    evidence_relation_ids: List[str]
    partition: str                 # ADMITTED_KG, HISTORICAL_BASELINE_KG, CONTRAINDICATED_LOCK, PROVISIONAL_QUEUE, REJECTED
    tier: int                      # 1 (High), 2 (Moderate/Baseline), 3 (Provisional)
    top_evidence_sentence: str
    top_evidence_pmid: str
    rationale_summary: str


class Layer5_BayesianBetaBinomialFuser:
    """Aggregates sentence-level projected edges across the literature corpus using a

    Hierarchical Bayesian Beta-Binomial model at the deduplicated PMID level:
        alpha = alpha_0 + sum_{supporting PMIDs} (c_k * W_study * W_section)
        beta  = beta_0  + sum_{contradicting PMIDs} (c_m * W_study * W_section)
    Computes the Posterior Mean mu = alpha / (alpha + beta) and the 90% Lower Confidence
    Bound (LCB_90) so single-paper claims are neither crushed to 0.05 (fixing Bug #2 on
    diabetic nephropathy) nor trivially saturated to 1.00.
    """

    def __init__(self, alpha_0: float = 0.5, beta_0: float = 0.5, conflict_discount: float = 1.0) -> None:
        # Jeffreys Non-informative Prior Beta(0.5, 0.5) (Jeffreys, Proc. Royal Soc. A 1946)
        # combined with Noisy-OR evidence pooling (Pearl, 1988; Himmelstein et al., eLife 2017):
        # Eliminates arbitrary magic priors (0.85 / 0.45 / 0.35).
        self.alpha_0 = alpha_0
        self.beta_0 = beta_0
        self.conflict_discount = conflict_discount

    def _compute_beta_stats(self, alpha: float, beta: float) -> Tuple[float, float, float]:
        """Computes (mean, std, lcb_90) of Beta(alpha, beta)."""
        total = alpha + beta
        mean = alpha / total
        var = (alpha * beta) / ((total ** 2) * (total + 1.0))
        std = math.sqrt(max(1e-9, var))
        # Normal-approximate 90% Lower Confidence Bound (z_0.90 = 1.2816), bounded in [0.01, 0.99]
        lcb_90 = max(0.01, min(0.99, mean - 1.28155 * std))
        return round(mean, 4), round(std, 4), round(lcb_90, 4)

    def fuse_edges(
        self, all_events: List[BioEventFrame]
    ) -> Tuple[List[Dict[str, Any]], List[BayesianFusedEdge]]:
        """Projects all Bio-Event Frames into Contract B relation records and fuses them

        into Contract D Knowledge Graph edges via Bayesian Beta-Binomial updating.
        """
        contract_b_records: List[Dict[str, Any]] = []
        all_candidates: List[Tuple[str, ProjectedEdgeCandidate]] = []

        rel_counter = 1
        for ev in all_events:
            for cand in ev.projected_edges:
                # Hard Gate: Never admit NO_RELATION or NONE
                if cand.canonical_relation in ("NO_RELATION", "NONE"):
                    continue
                if cand.section_weight <= 0.0:
                    continue

                rel_id = f"REL_{rel_counter:05d}"
                rel_counter += 1

                # Strictly compliant with dapagliflozin-lbd/contracts/relation.schema.json
                cb_record = {
                    "relation_id": rel_id,
                    "source_entity": cand.source_name,
                    "target_entity": cand.target_name,
                    "source_standard_id": cand.source_id,
                    "target_standard_id": cand.target_id,
                    "relation_type": cand.contract_b_relation,
                    "assertion_status": cand.assertion_status,
                    "evidence": cand.evidence_sentence,
                    "pmid": str(cand.pmid),
                    "confidence_score": round(float(cand.confidence_score), 4),
                    "normalization_status": "unresolved",
                }
                contract_b_records.append(cb_record)
                all_candidates.append((rel_id, cand))

        # Group by (source_id, target_id) to detect cross-polarity contradictions
        # (e.g. if one paper says Compound TREATS Disease X and another says Compound CAUSES Disease X)
        pair_groups: Dict[Tuple[str, str], List[Tuple[str, ProjectedEdgeCandidate]]] = defaultdict(list)
        for rel_id, cand in all_candidates:
            pair_groups[(cand.source_id, cand.target_id)].append((rel_id, cand))

        fused_edges: List[BayesianFusedEdge] = []
        edge_counter = 1

        for (src_id, tgt_id), items in pair_groups.items():
            # Group further by canonical_relation within (src_id, tgt_id)
            by_rel: Dict[str, List[Tuple[str, ProjectedEdgeCandidate]]] = defaultdict(list)
            for rel_id, cand in items:
                by_rel[cand.canonical_relation].append((rel_id, cand))

            for can_rel, rel_items in by_rel.items():
                rel_items = sorted(rel_items, key=lambda item: (
                    {"asserted": 0, "uncertain": 1, "negated": 2}.get(item[1].assertion_status, 3),
                    -item[1].confidence_score, str(item[1].pmid), item[1].evidence_sentence))
                exemplar = rel_items[0][1]

                # Deduplicate supporting evidence at the PMID level
                pmid_support_weight: Dict[str, float] = {}
                pmid_to_sentence: Dict[str, str] = {}
                rel_ids: List[str] = []
                best_ev_sent = exemplar.evidence_sentence
                best_ev_pmid = exemplar.pmid
                best_ev_score = -1.0

                for rel_id, cand in rel_items:
                    rel_ids.append(rel_id)
                    # Effective evidence weight = confidence * W_study * W_section
                    eff_w = cand.confidence_score * cand.study_weight * cand.section_weight
                    if cand.assertion_status == "uncertain":
                        eff_w *= 0.5
                    elif cand.assertion_status == "negated" and can_rel != "CONTRAINDICATED_FOR":
                        eff_w = 0.0

                    prev = pmid_support_weight.get(cand.pmid, 0.0)
                    if eff_w > prev:
                        pmid_support_weight[cand.pmid] = eff_w
                        pmid_to_sentence[cand.pmid] = cand.evidence_sentence
                    if eff_w > best_ev_score:
                        best_ev_score = eff_w
                        best_ev_sent = cand.evidence_sentence
                        best_ev_pmid = cand.pmid

                # Check contradicting PMIDs (opposite sign on the same (src_id, tgt_id) pair)
                pmid_contra_weight: Dict[str, float] = {}
                for other_rel, other_items in by_rel.items():
                    if other_rel == can_rel:
                        continue
                    # TREATS (-1) vs CAUSES (+1) or INCREASES (+1) vs DECREASES (-1)
                    if (can_rel == "TREATS" and other_rel in ("CAUSES", "CONTRAINDICATED_FOR")) or (
                        can_rel == "CAUSES" and other_rel == "TREATS"
                    ) or (
                        can_rel == "INCREASES" and other_rel == "DECREASES"
                    ) or (
                        can_rel == "DECREASES" and other_rel == "INCREASES"
                    ):
                        for _, o_cand in other_items:
                            if o_cand.assertion_status == "asserted":
                                cw = o_cand.confidence_score * o_cand.study_weight * o_cand.section_weight
                                pmid_contra_weight[o_cand.pmid] = max(
                                    pmid_contra_weight.get(o_cand.pmid, 0.0), cw
                                )

                sum_delta_pos = sum(pmid_support_weight.values())
                sum_delta_neg = sum(pmid_contra_weight.values()) * self.conflict_discount

                # Noisy-OR probability across independent supporting PMIDs (Himmelstein et al., eLife 2017):
                # P_noisy_or = 1 - prod_{p in PMIDs} (1 - w_p)
                noisy_or_prob = 1.0
                for w_p in pmid_support_weight.values():
                    noisy_or_prob *= (1.0 - max(0.0, min(0.99, w_p)))
                p_noisy_or = 1.0 - noisy_or_prob

                alpha_post = round(self.alpha_0 + sum_delta_pos, 4)
                beta_post = round(self.beta_0 + sum_delta_neg, 4)
                beta_mean, std, lcb_90 = self._compute_beta_stats(alpha_post, beta_post)
                # Combine Jeffreys posterior mean with Noisy-OR probability when uncontradicted
                mean = round(max(beta_mean, p_noisy_or * (alpha_post / (alpha_post + sum_delta_neg))), 4)
                lcb_90 = round(max(lcb_90, mean - 1.28155 * std), 4)

                supporting_pmids = sorted([p for p, w in pmid_support_weight.items() if w > 0.0])
                contra_pmids = sorted(list(pmid_contra_weight.keys()))

                # Partitioning Logic (Layer 5 Decision Boundary under Jeffreys Prior + Noisy-OR)
                auto_admit_allowed = (
                    can_rel != "CONTRAINDICATED_FOR"
                    and exemplar.assertion_status == "asserted"
                    and not exemplar.is_historical_baseline
                    and can_rel not in ("EVALUATED_IN_COHORT", "ASSOCIATED_WITH")
                    and lcb_90 >= 0.30
                    and mean >= 0.60
                    and len(supporting_pmids) >= 1
                )

                if can_rel == "CONTRAINDICATED_FOR":
                    partition = "CONTRAINDICATED_LOCK"
                    tier = 1
                elif exemplar.assertion_status == "negated" or sum_delta_pos <= 0.05:
                    partition = "REJECTED"
                    tier = 3
                elif exemplar.assertion_status == "uncertain":
                    partition = "PROVISIONAL_QUEUE"
                    tier = 3
                elif exemplar.is_historical_baseline and can_rel in ("TREATS", "EVALUATED_IN_COHORT", "ASSOCIATED_WITH"):
                    # Separate known baseline indication (Type 2 Diabetes DOID:9352) from Novel Repurposing
                    partition = "HISTORICAL_BASELINE_KG"
                    tier = 2
                elif can_rel in ("EVALUATED_IN_COHORT", "ASSOCIATED_WITH") and exemplar.source_type == "Gene":
                    partition = "PROVISIONAL_QUEUE" if lcb_90 < 0.55 else "ADMITTED_KG"
                    tier = 2
                elif can_rel == "EVALUATED_IN_COHORT":
                    partition = "COHORT_CONTEXT_KG"
                    tier = 2
                elif auto_admit_allowed:
                    partition = "ADMITTED_KG"
                    tier = 1 if lcb_90 >= 0.50 else 2
                elif mean >= 0.50:
                    partition = "PROVISIONAL_QUEUE"
                    tier = 3
                else:
                    partition = "REJECTED"
                    tier = 3

                edge_id = f"EDGE_{edge_counter:04d}"
                edge_counter += 1

                fused_edges.append(
                    BayesianFusedEdge(
                        edge_id=edge_id,
                        source_id=src_id,
                        source_name=exemplar.source_name,
                        source_type=exemplar.source_type,
                        target_id=tgt_id,
                        target_name=exemplar.target_name,
                        target_type=exemplar.target_type,
                        canonical_relation=can_rel,
                        contract_b_relation=exemplar.contract_b_relation,
                        hetionet_metaedge=exemplar.hetionet_metaedge,
                        hetionet_sign=exemplar.hetionet_sign,
                        assertion_status=exemplar.assertion_status,
                        is_historical_baseline=exemplar.is_historical_baseline,
                        alpha_post=alpha_post,
                        beta_post=beta_post,
                        posterior_mean=mean,
                        posterior_std=std,
                        lcb_90=lcb_90,
                        unique_supporting_pmids=supporting_pmids,
                        unique_contradicting_pmids=contra_pmids,
                        evidence_relation_ids=rel_ids,
                        partition=partition,
                        tier=tier,
                        top_evidence_sentence=best_ev_sent,
                        top_evidence_pmid=best_ev_pmid,
                        rationale_summary=exemplar.rationale,
                    )
                )

        # Sort fused edges: ADMITTED_KG first, then by LCB_90 descending
        priority_map = {
            "ADMITTED_KG": 0,
            "CONTRAINDICATED_LOCK": 1,
            "HISTORICAL_BASELINE_KG": 2,
            "COHORT_CONTEXT_KG": 3,
            "PROVISIONAL_QUEUE": 4,
            "REJECTED": 5,
        }
        fused_edges.sort(key=lambda e: (priority_map.get(e.partition, 9), -e.lcb_90, -e.posterior_mean))

        return contract_b_records, fused_edges

    def to_contract_d_records(self, fused_edges: List[BayesianFusedEdge]) -> List[Dict[str, Any]]:
        """Converts admitted/active BayesianFusedEdges to strict Contract D JSON records
        compliant with `dapagliflozin-lbd/contracts/graph_edge.schema.json`.
        Strict Industrial Gate: Only ADMITTED_KG, HISTORICAL_BASELINE_KG, and CONTRAINDICATED_LOCK
        are allowed into the Knowledge Graph. PROVISIONAL_QUEUE and REJECTED are strictly excluded.
        """
        ALLOWED_PARTITIONS = {"ADMITTED_KG", "HISTORICAL_BASELINE_KG", "CONTRAINDICATED_LOCK"}
        contract_d: List[Dict[str, Any]] = []
        for fe in fused_edges:
            if fe.partition not in ALLOWED_PARTITIONS:
                continue
            record = {
                "edge_id": fe.edge_id,
                "source": fe.source_id,
                "target": fe.target_id,
                "metaedge": fe.hetionet_metaedge,
                "relation_type": fe.contract_b_relation,
                "assertion_status": fe.assertion_status,
                "sign": int(fe.hetionet_sign),
                "evidence_ids": fe.evidence_relation_ids,
                "pmids": fe.unique_supporting_pmids,
                "source_dataset": "nlp_literature",
                "tier": int(fe.tier),
            }
            contract_d.append(record)
        return contract_d


# Canonical names
BayesianFusionEngine = Layer5_BayesianBetaBinomialFuser


# -----------------------------------------------------------------------------
# Independent Graph-Topological Empirical Bayes Engine
# -----------------------------------------------------------------------------
class IndependentGraphBayesEngine:
    """Independent Bayesian Fuser: Learns priors from Hetionet topology and updates via objective study tiers."""

    TIER_WEIGHTS = {
        "clinical_trial": 1.20,
        "clinical_cohort": 0.90,
        "preclinical_invivo": 0.60,
        "invitro": 0.30,
        "case_report": 0.40,
    }

    @staticmethod
    def derive_topological_prior(kg_prior_strength: float) -> Tuple[float, float]:
        """Dynamically learns alpha_0, beta_0 from Hetionet graph proximity."""
        alpha_0 = 0.50 + 1.50 * kg_prior_strength
        beta_0 = 1.20 - 0.90 * kg_prior_strength
        return round(alpha_0, 4), round(beta_0, 4)

    @classmethod
    def compute_posterior(
        cls,
        kg_prior_strength: float,
        study_tier: str,
        n_studies: int = 1,
        has_contradiction: bool = False,
    ) -> Dict[str, Any]:
        alpha_0, beta_0 = cls.derive_topological_prior(kg_prior_strength)

        unit_w = cls.TIER_WEIGHTS.get(study_tier, 0.60)
        delta_pos = n_studies * unit_w
        delta_neg = 0.80 if has_contradiction else 0.0

        alpha_post = alpha_0 + delta_pos
        beta_post = beta_0 + delta_neg

        total = alpha_post + beta_post
        mean = alpha_post / total
        var = (alpha_post * beta_post) / ((total ** 2) * (total + 1.0))
        std = math.sqrt(max(1e-9, var))
        lcb_90 = max(0.01, min(0.99, mean - 1.28155 * std))

        req_lcb = 0.38 if study_tier == "clinical_trial" else 0.45
        admitted = (lcb_90 >= req_lcb and mean >= 0.55)

        return {
            "alpha_0": alpha_0,
            "beta_0": beta_0,
            "alpha_post": round(alpha_post, 4),
            "beta_post": round(beta_post, 4),
            "posterior_mean": round(mean, 4),
            "posterior_std": round(std, 4),
            "lcb_90": round(lcb_90, 4),
            "admitted": admitted,
            "effective_evidence_weight": round(mean * delta_pos, 4) if admitted else 0.0,
        }
