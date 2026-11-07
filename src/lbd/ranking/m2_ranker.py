"""
M2 Supervised Graph-Topological Ranking, Signed Graph NTR Flux & Meet-in-the-Middle Micro-DFS Engine.
-----------------------------------------------------------------------------------------------------
Owner: Nguyễn Nhật Lâm (Algorithm, Topological DWPC, Degree Debiasing & Signed Graph NTR)
Co-owner: Đặng Quang Huy (In-Memory Ephemeral Session KG Ingestion `augment_session_edges`)

Architecture & Training Data Specification (`artifacts/ranking_m2/`):
1. Training & CV Tensor (`features.npz`, `split_indices.npz`, `gold_labels_from_hetionet.csv`):
   - Matrix X shape: `(212624, 68)` representing 1,552 Compounds x 137 Diseases in Hetionet v1.0.
   - Columns 0..64 (`M2_model.pkl`): 2 degree priors + 63 topological DWPC (Degree-Weighted Path Count,
     damping exponent w=0.4) transformed via learned `arcsinh(x / mean)` and z-score standardization.
   - Columns 65..67 (`M3_model.pkl`): Signed transcriptional reversal & directional metapath features.
   - Cross-Validation: 5-fold outer / 3-fold inner Nested CV grouped by `CrC` (Compound-resembles-Compound)
     connected components (192,947 dev pairs / 669 CtD positives; 19,013 test pairs / 86 CtD positives).
2. Compound-Specific Degree Debiasing (beta = 0.6):
   - Suppresses high-degree disease hub bias (Cancer, Hypertension) via disease prior attenuation:
     `Score_debiased(c, d) = P_M2(c, d) / (P_prior(d) ^ beta)`.
3. Signed Graph Net Therapeutic Ratio (NTR Flux Layer):
   - Evaluates directional biological sign multiplication (`+1`, `-1`, `0`):
     * Therapeutic Reversal (`net_sign = -1`): `CdG..DuG`, `CuG..DdG`, and L1000 signature reversal.
     * Pathogenic Exacerbation (`net_sign = +1`): `CdG..DdG`, `CuG..DuG`.
     * `NTR(c, d) = (Flux_rev - Flux_cong) / max(Flux_rev + Flux_cong, 1e-6) + 0.5 * Sig_term` (clipped to `[-1, +1]`).
4. Ephemeral In-Memory Session Graph Augmentation (`augment_session_edges`):
   - Injects NLP-extracted edges (`Compound->Gene`, `Gene->Disease`) into an isolated RAM session copy
     of the Hetionet graph (Zero Disk Mutation) and computes live DWPC + NTR deltas.
5. Meet-in-the-Middle Micro-DFS (`explain_pair_dfs`):
   - Traces concrete 3-node (C-G-D) and 5-node (C-G-X-G-D) biological pathways and ranks them by
     exact learned log-odds contribution (`beta_metapath * dwpc_path_weight`) with biological sign tags.
"""
from __future__ import annotations

import copy
import csv
import gzip
import json
import math
import pickle
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


DEFAULT_RANKING_DIR = Path(__file__).resolve().parents[3] / "artifacts" / "ranking_m2"
DEFAULT_RAW_DIR = Path(__file__).resolve().parents[3] / "data" / "raw"

# Known aliases for convenient drug lookup across Landmark & Prospective Test Suites
DRUG_ALIASES = {
    "aspirin": "Compound::DB00945",
    "acetylsalicylic acid": "Compound::DB00945",
    "colchicine": "Compound::DB01394",
    "lodoco": "Compound::DB01394",
    "dapagliflozin": "Compound::DB06292",
    "farxiga": "Compound::DB06292",
    "forxiga": "Compound::DB06292",
    "canagliflozin": "Compound::DB08907",
    "invokana": "Compound::DB08907",
    "empagliflozin": "Compound::DB09038",
    "jardiance": "Compound::DB09038",
    "ketamine": "Compound::DB01221",
    "esketamine": "Compound::DB01221",
    "spravato": "Compound::DB01221",
    "sildenafil": "Compound::DB00203",
    "metformin": "Compound::DB00331",
    "duloxetine": "Compound::DB00471",
    "thalidomide": "Compound::DB01041",
    "celecoxib": "Compound::DB00482",
    "disulfiram": "Compound::DB00822",
    "spironolactone": "Compound::DB00421",
    "losartan": "Compound::DB00678",
    "octreotide": "Compound::DB00104",
    "liraglutide_proxy": "Compound::DB00104",
    "ruxolitinib": "Compound::DB08877",
    "baricitinib_proxy": "Compound::DB08877",
}


class M2RepurposingRanker:
    """End-to-end M2 DWPC Ranker and Mechanistic Micro-DFS Explainer for LBD."""

    def __init__(
        self,
        artifact_dir: Optional[Path] = None,
        raw_dir: Optional[Path] = None,
        damping_exponent: float = 0.4,
        load_graph_for_dfs: bool = True,
    ) -> None:
        self.artifact_dir = Path(artifact_dir) if artifact_dir else DEFAULT_RANKING_DIR
        self.raw_dir = Path(raw_dir) if raw_dir else DEFAULT_RAW_DIR
        self.w = damping_exponent

        # 1. Load manifest and feature matrices
        manifest_path = self.artifact_dir / "data_manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Missing M2 manifest at {manifest_path}")
        self.meta = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.drugs: List[str] = self.meta["drugs"]
        self.diseases: List[str] = self.meta["diseases"]
        self.feature_names: List[str] = self.meta["feature_names"]
        self.num_diseases = len(self.diseases)

        features_npz = np.load(self.artifact_dir / "features.npz")
        self.X: np.ndarray = features_npz["X"]

        # 2. Load trained M2 model bundle
        model_bundle = pickle.loads((self.artifact_dir / "M2_model.pkl").read_bytes())
        self.scale, self.scaler, self.clf = model_bundle["bundle"]
        self.cols: List[int] = list(model_bundle["columns"])
        self.selected_feature_names = [self.feature_names[c] for c in self.cols]
        self.coef_dict: Dict[str, float] = dict(
            zip(self.selected_feature_names, [float(x) for x in self.clf.coef_[0]])
        )
        self.intercept: float = float(self.clf.intercept_[0])

        # Normalized linear sensitivity in raw feature space: beta_j / (scale_j * std_j)
        std_vals = getattr(self.scaler, "scale_", np.ones_like(self.scale))
        self.norm_beta_dict: Dict[str, float] = {}
        for i, fname in enumerate(self.selected_feature_names):
            s = float(self.scale[i]) if i < len(self.scale) else 1.0
            sigma = float(std_vals[i]) if i < len(std_vals) else 1.0
            eff_scale = s * sigma if (s * sigma) > 1e-12 else 1.0
            self.norm_beta_dict[fname] = float(self.clf.coef_[0][i]) / eff_scale

        # Detect Signed Graph Metapaths & Signature Reversal Features
        self.rev_cols = [
            i for i, c in enumerate(self.cols)
            if ("CdG" in self.feature_names[c] and "DuG" in self.feature_names[c])
            or ("CuG" in self.feature_names[c] and "DdG" in self.feature_names[c])
        ]
        self.cong_cols = [
            i for i, c in enumerate(self.cols)
            if ("CdG" in self.feature_names[c] and "DdG" in self.feature_names[c])
            or ("CuG" in self.feature_names[c] and "DuG" in self.feature_names[c])
        ]
        self.sig_reversal_col = 65 if self.X.shape[1] > 65 else None
        self.sig_avail_col = 66 if self.X.shape[1] > 66 else None

        # 3. Load node names and build lookup index
        self.names: Dict[str, str] = dict(self.meta.get("names", {}))
        nodes_tsv = self.raw_dir / "hetionet-v1.0-nodes.tsv"
        if nodes_tsv.exists():
            with nodes_tsv.open("r", encoding="utf-8") as f:
                for row in csv.DictReader(f, delimiter="\t"):
                    self.names[row["id"]] = row["name"]

        self.name_to_drug_id: Dict[str, str] = dict(DRUG_ALIASES)
        for d_id in self.drugs:
            d_name = self.names.get(d_id, "").lower().strip()
            if d_name:
                self.name_to_drug_id[d_name] = d_id
            short_id = d_id.replace("Compound::", "").lower()
            self.name_to_drug_id[short_id] = d_id
            self.name_to_drug_id[d_id.lower()] = d_id

        # 4. Optionally load Hetionet edges for CtD/CpD filtering and Micro-DFS
        self.ctd: set[Tuple[str, str]] = set()
        self.cpd: set[Tuple[str, str]] = set()
        self.out_deg: Counter = Counter()
        self.in_deg: Counter = Counter()
        self.adj: Dict[Tuple[str, str], List[str]] = defaultdict(list)
        self._graph_loaded = False

        if load_graph_for_dfs:
            self._load_hetionet_edges()

        # Canonical baseline approved indications (pre-2018 knowledge)
        # Prevents primary approved disease indications from leaking into "novel" candidate lists
        self.canonical_baseline_indications: Dict[str, Set[str]] = {
            "Compound::DB06292": {"Disease::DOID:9352"},  # Dapagliflozin -> Type 2 Diabetes
            "Compound::DB08907": {"Disease::DOID:9352"},  # Canagliflozin -> Type 2 Diabetes
            "Compound::DB09038": {"Disease::DOID:9352"},  # Empagliflozin -> Type 2 Diabetes
            "Compound::DB01394": {"Disease::DOID:13189"}, # Colchicine -> Gout
            "Compound::DB00331": {"Disease::DOID:9352"},  # Metformin -> Type 2 Diabetes
            "Compound::DB00203": {"Disease::DOID:10763"}, # Sildenafil -> Erectile/PAH
            "Compound::DB06655": {"Disease::DOID:9352"},  # Liraglutide -> Type 2 Diabetes
            "Compound::DB00104": {"Disease::DOID:10534"}, # Octreotide -> Acromegaly
        }

    def _load_hetionet_edges(self) -> None:
        edges_gz = self.raw_dir / "hetionet-v1.0-edges.sif.gz"
        if not edges_gz.exists():
            return
        with gzip.open(edges_gz, "rt", encoding="utf-8") as stream:
            for e in csv.DictReader(stream, delimiter="\t"):
                u, m, v = e["source"], e["metaedge"], e["target"]
                if m == "CtD":
                    self.ctd.add((u, v))
                    continue
                if m == "CpD":
                    self.cpd.add((u, v))
                    continue
                self.out_deg[(u, m)] += 1
                self.in_deg[(v, m)] += 1
                self.adj[(u, m)].append(v)
                if m in ("GiG", "GcG", "GpBP", "GpPW", "GpMF", "GpCC", "DaG", "DuG", "DdG"):
                    self.adj[(v, m + "_rev")].append(u)
        self._graph_loaded = True

    def resolve_drug_id(self, query: str) -> str:
        """Resolves a drug name, DrugBank ID, or Hetionet ID to canonical Compound::DBxxxxx."""
        q = query.strip().lower()
        if q in self.name_to_drug_id:
            return self.name_to_drug_id[q]
        if query in self.drugs:
            return query
        if f"Compound::{query}" in self.drugs:
            return f"Compound::{query}"
        raise KeyError(
            f"Drug '{query}' not found in Hetionet v1.0 (1,552 compounds). "
            f"For post-2017 JAK1/2 inhibitors like Baricitinib, use 'Ruxolitinib' (DB08877)."
        )

    def edge_weight(self, u: str, metaedge: str, v: str) -> float:
        """Computes DWPC degree-damped weight for a single directed step."""
        d_out = max(self.out_deg[(u, metaedge)], 1)
        d_in = max(self.in_deg[(v, metaedge)], 1)
        return float((d_out * d_in) ** (-self.w))

    def explain_pair_dfs(
        self, drug_id: str, disease_id: str, top_k: int = 3, return_all_lit: bool = False
    ) -> Tuple[Any, ...]:
        """
        Meet-in-the-Middle Micro-DFS: Finds all 3-node (C-G-D) and 5-node (C-G-X-G-D)
        simple paths and returns total path count plus top_k paths sorted by M2 contribution.
        """
        if not self._graph_loaded:
            self._load_hetionet_edges()

        paths: List[Dict[str, Any]] = []

        # Step 1: Collect target disease genes
        dis_genes: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
        for dm in ("DaG", "DuG", "DdG"):
            for g in self.adj[(disease_id, dm)]:
                dis_genes[g].append((dm, self.edge_weight(disease_id, dm, g)))

        # Step 2: Reverse lookup via shared biological entities (PW, BP, MF, CC)
        b_map: Dict[str, Dict[str, List[Tuple[str, str, float, float]]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for g2, dm_list in dis_genes.items():
            for dm, w_d in dm_list:
                for bm in ("GpPW", "GpBP", "GpMF", "GpCC"):
                    for b in self.adj[(g2, bm)]:
                        w_b = self.edge_weight(g2, bm, b)
                        b_map[bm][b].append((g2, dm, w_b, w_d))

        # Step 3: Forward traversal from drug genes
        drug_name = self.names.get(drug_id, drug_id)
        dis_name = self.names.get(disease_id, disease_id)

        # Check for direct literature edges (Compound -> Disease)
        if hasattr(self, "lit_edges"):
            for dm in ("CtD", "CpD"):
                e_dir = (drug_id, dm, disease_id)
                if e_dir in self.lit_edges:
                    info = self.lit_edges[e_dir]
                    pmids = [q["pmid"] for q in info.get("quotes", []) if q.get("pmid")]
                    rel_up = str(info.get("relation_type", dm)).upper()
                    if dm == "CtD" or any(w in rel_up for w in ("TREAT", "ALLEVIAT", "PREVENT")):
                        lit_sign, lit_tag, lit_mech = -1, " [🟢 Trị liệu]", "therapeutic"
                    elif dm == "CcSE" or any(w in rel_up for w in ("CAUSE", "TOXIC", "ADVERSE", "SIDE_EFFECT")):
                        lit_sign, lit_tag, lit_mech = 1, " [🔴 Cảnh báo độc tính]", "pathogenic"
                    else:
                        lit_sign, lit_tag, lit_mech = 0, " [⚪ Trung tính]", "neutral"

                    paths.append({
                        "metapath": f"literature:{dm}",
                        "nodes": [drug_id, disease_id],
                        "dwpc_weight": 1.0,
                        "beta": 1.5,
                        "contrib": 1.5,
                        "net_sign": int(lit_sign),
                        "mechanism_type": lit_mech,
                        "is_literature_augmented": True,
                        "supporting_pmids": pmids,
                        "quotes": info.get("quotes", []),
                        "path_str": f"{drug_name} --[Y văn: {info.get('relation_type', dm)}]--> {dis_name}{lit_tag} [⚡ Y văn trực tiếp]",
                    })

        for cm in ("CbG", "CuG", "CdG"):
            for g1 in self.adj[(drug_id, cm)]:
                w1 = self.edge_weight(drug_id, cm, g1)
                g1_name = self.names.get(g1, g1)

                # Direct 3-node path: C -> G1 <- D
                if g1 in dis_genes:
                    for dm, w2 in dis_genes[g1]:
                        mp_name = f"dwpc:{cm}|direct|{dm}"
                        weight = w1 * w2
                        beta = self.coef_dict.get(mp_name, 0.0)
                        norm_beta = self.norm_beta_dict.get(mp_name, 0.0)
                        contrib = norm_beta * weight

                        lit_info = []
                        if hasattr(self, "lit_edges"):
                            for edge_cand in ((drug_id, cm, g1), (disease_id, dm, g1), (g1, dm, disease_id)):
                                if edge_cand in self.lit_edges:
                                    lit_info.append(self.lit_edges[edge_cand])
                        is_lit = len(lit_info) > 0
                        pmids = [q["pmid"] for item in lit_info for q in item.get("quotes", []) if q.get("pmid")]
                        tag = " [⚡ Y văn mới]" if is_lit else ""

                        # Biological sign tracking (DuG: +1, DdG: -1, DaG: 0 associative)
                        cm_s = -1 if cm == "CdG" else (1 if cm == "CuG" else 0)
                        dm_s = 1 if dm == "DuG" else (-1 if dm == "DdG" else 0)
                        p_sign = 0 if (cm == "CbG" or dm == "DaG") else (cm_s * dm_s)
                        if p_sign == -1:
                            sign_tag = " [🟢 Trị liệu]"
                            mech_type = "therapeutic"
                        elif p_sign == 1:
                            sign_tag = " [🔴 Cảnh báo độc tính]"
                            mech_type = "pathogenic"
                        elif dm == "DaG":
                            sign_tag = " [⚪ Tương quan]"
                            mech_type = "suggestive_association"
                        else:
                            sign_tag = " [⚪ Trung tính]"
                            mech_type = "neutral"

                        paths.append({
                            "metapath": mp_name,
                            "nodes": [drug_id, g1, disease_id],
                            "dwpc_weight": round(float(weight), 6),
                            "beta": round(float(beta), 4),
                            "norm_beta": round(float(norm_beta), 4),
                            "contrib": round(float(contrib), 6),
                            "net_sign": int(p_sign),
                            "mechanism_type": mech_type,
                            "is_literature_augmented": is_lit,
                            "supporting_pmids": list(dict.fromkeys(pmids)),
                            "quotes": [q for item in lit_info for q in item.get("quotes", [])],
                            "path_str": f"{drug_name} --[{cm}]--> {g1_name} <--[{dm}]-- {dis_name}{sign_tag}{tag}",
                        })

                # 5-node path: C -> G1 -> B <- G2 <- D
                for bm in ("GpPW", "GpBP", "GpMF", "GpCC"):
                    if bm not in b_map:
                        continue
                    cur_b_map = b_map[bm]
                    for b in self.adj[(g1, bm)]:
                        if b in cur_b_map:
                            w2 = self.edge_weight(g1, bm, b)
                            b_name = self.names.get(b, b)
                            for g2, dm, w3, w4 in cur_b_map[b]:
                                if g2 == g1:
                                    continue
                                mp_name = f"dwpc:{cm}|{bm}|{dm}"
                                weight = w1 * w2 * w3 * w4
                                beta = self.coef_dict.get(mp_name, 0.0)
                                norm_beta = self.norm_beta_dict.get(mp_name, 0.0)
                                contrib = norm_beta * weight
                                g2_name = self.names.get(g2, g2)

                                lit_info = []
                                if hasattr(self, "lit_edges"):
                                    for edge_cand in ((drug_id, cm, g1), (g1, bm, b), (g2, bm, b), (disease_id, dm, g2)):
                                        if edge_cand in self.lit_edges:
                                            lit_info.append(self.lit_edges[edge_cand])
                                is_lit = len(lit_info) > 0
                                pmids = [q["pmid"] for item in lit_info for q in item.get("quotes", []) if q.get("pmid")]
                                tag = " [⚡ Y văn mới]" if is_lit else ""

                                # Biological sign tracking (DuG: +1, DdG: -1, DaG: 0 associative)
                                cm_s = -1 if cm == "CdG" else (1 if cm == "CuG" else 0)
                                dm_s = 1 if dm == "DuG" else (-1 if dm == "DdG" else 0)
                                p_sign = 0 if (cm == "CbG" or dm == "DaG") else (cm_s * dm_s)
                                if p_sign == -1:
                                    sign_tag = " [🟢 Trị liệu]"
                                    mech_type = "therapeutic"
                                elif p_sign == 1:
                                    sign_tag = " [🔴 Cảnh báo độc tính]"
                                    mech_type = "pathogenic"
                                elif dm == "DaG":
                                    sign_tag = " [⚪ Tương quan]"
                                    mech_type = "suggestive_association"
                                else:
                                    sign_tag = " [⚪ Trung tính]"
                                    mech_type = "neutral"

                                paths.append({
                                    "metapath": mp_name,
                                    "nodes": [drug_id, g1, b, g2, disease_id],
                                    "dwpc_weight": round(float(weight), 6),
                                    "beta": round(float(beta), 4),
                                    "norm_beta": round(float(norm_beta), 4),
                                    "contrib": round(float(contrib), 6),
                                    "net_sign": int(p_sign),
                                    "mechanism_type": mech_type,
                                    "is_literature_augmented": is_lit,
                                    "supporting_pmids": list(dict.fromkeys(pmids)),
                                    "quotes": [q for item in lit_info for q in item.get("quotes", [])],
                                    "path_str": (
                                        f"{drug_name} --[{cm}]--> {g1_name} "
                                        f"--[{bm}]--> {b_name} <--[{bm}]-- "
                                        f"{g2_name} <--[{dm}]-- {dis_name}{sign_tag}{tag}"
                                    ),
                                })

        paths.sort(key=lambda x: x["contrib"], reverse=True)
        all_lit_paths = [p for p in paths if p.get("is_literature_augmented")]
        if return_all_lit:
            return len(paths), paths[:top_k], all_lit_paths
        return len(paths), paths[:top_k]

    def compute_feature_contributions(
        self, drug_query: str, disease_id: str, top_k: int = 10
    ) -> Dict[str, Any]:
        """Decomposes the exact linear logit and probability for a (drug, disease) pair."""
        drug_id = self.resolve_drug_id(drug_query)
        di = self.drugs.index(drug_id)
        dis_idx = self.diseases.index(disease_id)
        row_idx = di * self.num_diseases + dis_idx

        x_raw = self.X[row_idx, self.cols]
        x_trans = self.scaler.transform(np.arcsinh(x_raw.reshape(1, -1) / self.scale))[0]
        prob = float(self.clf.predict_proba(x_trans.reshape(1, -1))[0, 1])

        contribs = []
        for idx, fname in enumerate(self.selected_feature_names):
            raw_val = float(x_raw[idx])
            std_val = float(x_trans[idx])
            beta = float(self.clf.coef_[0][idx])
            logit_add = beta * std_val
            if raw_val > 0 or abs(logit_add) > 1e-4:
                contribs.append({
                    "feature": fname,
                    "raw_value": round(raw_val, 6),
                    "standardized_value": round(std_val, 4),
                    "beta": round(beta, 4),
                    "logit_contribution": round(logit_add, 4),
                })

        contribs.sort(key=lambda c: c["logit_contribution"], reverse=True)
        return {
            "drug_id": drug_id,
            "drug_name": self.names.get(drug_id, drug_id),
            "disease_id": disease_id,
            "disease_name": self.names.get(disease_id, disease_id),
            "probability_score": round(prob, 4),
            "intercept": round(self.intercept, 4),
            "top_positive_features": contribs[:top_k],
        }

    def rank_drug(
        self,
        drug_query: str,
        top_n_novel: int = 10,
        include_dfs_paths: bool = True,
        dfs_top_k: int = 2,
        debiasing_beta: float = 0.6,
        use_signed_graph: bool = True,
        ntr_weight: float = 0.35,
    ) -> Dict[str, Any]:
        """
        Executes the packaged M2 + Signed Graph NTR ranking pipeline for a drug across all 137 diseases:
        1. Macro DWPC scoring with Compound-Degree debiasing (beta=0.6).
        2. Signed Graph Net Therapeutic Ratio (NTR) Flux computation:
           NTR = (Phi_therapeutic - Phi_pathogenic) / max(Phi_therapeutic + Phi_pathogenic, 1e-6) + 0.5 * Cosine_reversal (clipped to [-1.0, 1.0])
        3. Separation of known indications (CtD/CpD) vs novel candidates.
        4. Micro-DFS path explanation with biological sign tracking.
        """
        drug_id = self.resolve_drug_id(drug_query)
        di = self.drugs.index(drug_id)
        row_start = di * self.num_diseases
        row_end = (di + 1) * self.num_diseases
        X_drug = self.X[row_start : row_end][:, self.cols]
        xx = self.scaler.transform(np.arcsinh(X_drug / self.scale))

        # 1. Base logit with degree debiasing
        coef = np.copy(self.clf.coef_)
        if debiasing_beta is not None and debiasing_beta != 1.0:
            coef[0, 1] = debiasing_beta * self.clf.coef_[0, 1]
        base_logits = (xx @ coef.T + self.clf.intercept_).ravel()
        base_probs = 1.0 / (1.0 + np.exp(-base_logits))

        # 2. Signed Graph NTR Flux computation
        # Compute directional biological flux from actual topological features (avoiding Z-score zero-inflation)
        raw_rev = np.sum(X_drug[:, self.rev_cols], axis=1) if self.rev_cols else np.zeros(self.num_diseases)
        raw_cong = np.sum(X_drug[:, self.cong_cols], axis=1) if self.cong_cols else np.zeros(self.num_diseases)

        if self.sig_reversal_col is not None and self.sig_avail_col is not None:
            sig_rev = self.X[row_start : row_end, self.sig_reversal_col]
            sig_avail = self.X[row_start : row_end, self.sig_avail_col]
            sig_term = sig_rev * sig_avail
        else:
            sig_term = np.zeros(self.num_diseases)

        # Net Therapeutic Ratio in raw scale [-1.0, +1.0]:
        # Positive (> 0): Net therapeutic reversal of disease state
        # Negative (< 0): Pathogenic exacerbation or congruent adverse risk
        # Zero (= 0): Neutral or uncharacterized transcriptional direction
        denom = raw_rev + raw_cong
        safe_denom = np.where(denom > 1e-6, denom, 1.0)
        raw_ntr = np.where(denom > 1e-6, (raw_rev - raw_cong) / safe_denom, 0.0) + 0.5 * sig_term
        raw_ntr = np.clip(raw_ntr, -1.0, 1.0)
        norm_ntr = raw_ntr

        if use_signed_graph and ntr_weight > 0.0:
            final_logits = base_logits + ntr_weight * norm_ntr
        else:
            final_logits = base_logits

        scores = 1.0 / (1.0 + np.exp(-final_logits))
        order = np.argsort(-scores)

        known_indications: List[Dict[str, Any]] = []
        novel_candidates: List[Dict[str, Any]] = []

        baseline_dis_set = self.canonical_baseline_indications.get(drug_id, set())

        for rank_raw, idx in enumerate(order, 1):
            dis_id = self.diseases[idx]
            dis_name = self.names.get(dis_id, dis_id)
            score_val = float(scores[idx])
            is_treat = (drug_id, dis_id) in self.ctd or dis_id in baseline_dis_set
            is_pall = (drug_id, dis_id) in self.cpd

            ntr_val = float(raw_ntr[idx])
            mech_class = (
                "therapeutic_reversal" if ntr_val > 0.05
                else ("pathogenic_risk" if ntr_val < -0.05 else "neutral")
            )

            item: Dict[str, Any] = {
                "rank_raw": rank_raw,
                "disease_id": dis_id,
                "disease_name": dis_name,
                "score": round(score_val, 4),
                "base_score": round(float(base_probs[idx]), 4),
                "signed_mechanism": {
                    "therapeutic_flux": round(float(raw_rev[idx]), 4),
                    "pathogenic_flux": round(float(raw_cong[idx]), 4),
                    "net_flux": round(float(raw_rev[idx] - raw_cong[idx]), 4),
                    "signature_reversal": round(float(sig_term[idx]), 4),
                    "ntr_score": round(ntr_val, 4),
                    "normalized_ntr": round(float(norm_ntr[idx]), 4),
                    "mechanism_classification": mech_class,
                },
                "status": (
                    "KNOWN_TREATS"
                    if is_treat
                    else "KNOWN_PALLIATES"
                    if is_pall
                    else "NOVEL_CANDIDATE"
                ),
            }

            if is_treat or is_pall:
                known_indications.append(item)
            else:
                novel_candidates.append(item)

        top_novel: List[Dict[str, Any]] = []
        for novel_rank, cand in enumerate(novel_candidates[:top_n_novel], 1):
            cand_copy = dict(cand)
            cand_copy["rank_novel"] = novel_rank
            if include_dfs_paths and self._graph_loaded:
                n_paths, top_paths = self.explain_pair_dfs(
                    drug_id, cand_copy["disease_id"], top_k=dfs_top_k
                )
                cand_copy["total_dfs_paths"] = n_paths
                cand_copy["top_paths"] = top_paths
            top_novel.append(cand_copy)

        return {
            "drug_id": drug_id,
            "drug_name": self.names.get(drug_id, drug_id),
            "total_diseases_evaluated": self.num_diseases,
            "known_indications_count": len(known_indications),
            "known_indications": known_indications,
            "top_novel_candidates": top_novel,
            "all_novel_candidates_count": len(novel_candidates),
        }

    def clone_for_session(self) -> "M2RepurposingRanker":
        """
        Creates an isolated in-memory session replica of the ranker.
        The underlying frozen matrices (X, clf, scaler) are shared read-only,
        while graph topologies (adj, degrees, lit_edges) are independently mutable
        and will be discarded when the session ends.
        """
        if not self._graph_loaded:
            self._load_hetionet_edges()
        session = copy.copy(self)
        session.adj = defaultdict(list, {k: list(v) for k, v in self.adj.items()})
        session.out_deg = Counter(self.out_deg)
        session.in_deg = Counter(self.in_deg)
        session.names = dict(self.names)
        session.ctd = set(self.ctd)
        session.cpd = set(self.cpd)
        session.lit_edges = {}
        session.session_augmented_count = 0
        return session

    def augment_session_edges(self, edges: List[Dict[str, Any]]) -> int:
        """
        Temporarily injects literature-derived edges into this session's in-memory graph.
        Does NOT touch any disk files.
        """
        if not hasattr(self, "lit_edges"):
            self.lit_edges = {}
            self.session_augmented_count = 0

        added = 0
        for edge in edges:
            s_type = edge.get("source_type") or "Compound"
            t_type = edge.get("target_type") or "Gene"
            s_raw = str(edge.get("source_standard_id") or edge.get("source") or edge.get("source_entity") or "")
            t_raw = str(edge.get("target_standard_id") or edge.get("target") or edge.get("target_entity") or "")
            if not s_raw or not t_raw:
                continue

            u = s_raw if "::" in s_raw else f"{s_type}::{s_raw}"
            v = t_raw if "::" in t_raw else f"{t_type}::{t_raw}"

            if edge.get("source_entity") and u not in self.names:
                self.names[u] = edge["source_entity"]
            if edge.get("target_entity") and v not in self.names:
                self.names[v] = edge["target_entity"]

            # Map relation and metaedge to canonical Hetionet schema
            metaedge = edge.get("metaedge")
            rel = (edge.get("relation_type") or edge.get("canonical_relation") or "").upper()
            
            # Map Compound -> Gene
            if s_type == "Compound" and t_type == "Gene":
                if metaedge in ("CiG", "CdG") or any(w in rel for w in ("INHIBIT", "DOWN", "DECREAS")):
                    metaedge = "CdG"
                elif metaedge in ("CaG", "CuG") or any(w in rel for w in ("ACTIVAT", "UP", "INCREAS", "STIMULAT")):
                    metaedge = "CuG"
                else:
                    metaedge = "CbG"
            # Map Gene -> Disease
            elif s_type == "Gene" and t_type == "Disease":
                if any(w in rel for w in ("UP", "OVEREXPRESS")):
                    metaedge = "DuG"
                elif any(w in rel for w in ("DOWN", "UNDEREXPRESS")):
                    metaedge = "DdG"
                else:
                    metaedge = "DaG"
            # Map Compound -> Disease
            elif s_type == "Compound" and t_type == "Disease":
                if metaedge == "CcSE" or any(w in rel for w in ("CAUSE", "INDUC", "TOXIC", "ADVERSE", "SIDE_EFFECT")):
                    metaedge = "CcSE"
                elif any(w in rel for w in ("PALLIAT", "SYMPTOM", "RELIEV")):
                    metaedge = "CpD"
                elif any(w in rel for w in ("TREAT", "ALLEVIAT", "PREVENT", "IMPROV")):
                    metaedge = "CtD"
                else:
                    metaedge = "CpD"
            elif s_type == "Gene" and t_type == "Gene":
                metaedge = "GiG"
            else:
                metaedge = metaedge or "CbG"

            # Check if edge already exists in this session
            existing = self.adj.get((u, metaedge), [])
            is_new = v not in existing
            if is_new:
                self.adj[(u, metaedge)].append(v)
                self.out_deg[(u, metaedge)] += 1
                self.in_deg[(v, metaedge)] += 1
                if metaedge in ("GiG", "GcG", "GpBP", "GpPW", "GpMF", "GpCC", "DaG", "DuG", "DdG"):
                    self.adj[(v, metaedge + "_rev")].append(u)
                added += 1

            edge_key = (u, metaedge, v)
            quotes = edge.get("quotes") or []
            if not quotes and edge.get("evidence"):
                quotes = [{"pmid": edge.get("pmid", ""), "text": edge.get("evidence", "")}]

            if edge_key not in self.lit_edges:
                self.lit_edges[edge_key] = {
                    "source": u,
                    "target": v,
                    "metaedge": metaedge,
                    "relation_type": edge.get("relation_type", metaedge),
                    "confidence": float(edge.get("confidence_score") or 0.8),
                    "sign": int(edge.get("sign") or 0),
                    "quotes": list(quotes),
                    "is_novel_edge": is_new,
                }
            else:
                self.lit_edges[edge_key]["quotes"].extend(quotes)

        self.session_augmented_count += added
        return added

    def rank_drug_with_session(
        self,
        drug_query: str,
        top_n_novel: int = 10,
        include_dfs_paths: bool = True,
        dfs_top_k: int = 3,
        max_lit_delta: float = 0.05,
        debiasing_beta: float = 0.6,
        use_signed_graph: bool = True,
        ntr_weight: float = 0.35,
    ) -> Dict[str, Any]:
        """
        Executes M2 ranking with dynamic session-scoped literature augmentation:
        1. Computes baseline M2 + Signed Graph NTR score from Hetionet topology & signatures.
        2. Evaluates newly ingested literature bridge paths and direct assertions.
        3. Computes statistical session delta score: Score_session = Score_base + Delta_lit.
        4. Dynamically re-ranks candidates based on augmented session scores.
        """
        base_result = self.rank_drug(
            drug_query=drug_query,
            top_n_novel=self.num_diseases,
            include_dfs_paths=False,
            debiasing_beta=debiasing_beta,
            use_signed_graph=use_signed_graph,
            ntr_weight=ntr_weight,
        )
        drug_id = base_result["drug_id"]
        novel_candidates = base_result["top_novel_candidates"]

        augmented_candidates = []
        for cand in novel_candidates:
            dis_id = cand["disease_id"]
            base_score = cand["score"]

            n_paths, top_paths, all_lit_paths = 0, [], []
            if self._graph_loaded:
                n_paths, top_paths, all_lit_paths = self.explain_pair_dfs(
                    drug_id, dis_id, top_k=dfs_top_k, return_all_lit=True
                )

            direct_lit = [
                e for (u, m, v), e in getattr(self, "lit_edges", {}).items()
                if (u == drug_id and v == dis_id) or (u == dis_id and v == drug_id)
            ]

            evidence_weight = 0.0
            adverse_count = 0
            for dl in direct_lit:
                m_edge = dl.get("metaedge")
                rel_type = (dl.get("relation_type") or "").upper()
                conf = dl.get("confidence", 0.8)
                if m_edge == "CtD" or rel_type in ("TREATS", "ALLEVIATES", "PREVENTS"):
                    evidence_weight += 2.0 * conf
                elif m_edge == "CpD" or rel_type in ("PALLIATES", "ASSOCIATED_WITH"):
                    evidence_weight += 0.5 * conf
                elif m_edge == "CcSE" or rel_type in ("CAUSES", "SIDE_EFFECT", "TOXICITY"):
                    adverse_count += 1
                    evidence_weight -= 1.5 * conf

            for lp in all_lit_paths:
                evidence_weight += 1.2 * max(lp.get("dwpc_weight", 0.1), 0.05)

            # Logit-space shift (preserves distribution & avoids probability ties at 0.9999)
            clamped_base = min(max(base_score, 0.001), 0.999)
            base_logit = math.log(clamped_base / (1.0 - clamped_base))
            delta_logit = 0.35 * evidence_weight
            session_prob = 1.0 / (1.0 + math.exp(-(base_logit + delta_logit)))
            session_score = round(session_prob, 4)
            delta = round(session_score - base_score, 4)
            cand_entry = dict(cand)
            cand_entry["base_score"] = base_score
            cand_entry["literature_delta"] = round(delta, 4)
            cand_entry["score"] = session_score
            cand_entry["session_evidence_count"] = len(direct_lit) + len(all_lit_paths)
            cand_entry["session_status"] = "augmented" if delta > 0 else "baseline"
            cand_entry["total_dfs_paths"] = n_paths
            cand_entry["top_paths"] = top_paths
            augmented_candidates.append(cand_entry)

        augmented_candidates.sort(key=lambda x: x["score"], reverse=True)
        for novel_rank, c in enumerate(augmented_candidates[:top_n_novel], 1):
            c["rank_novel"] = novel_rank

        return {
            "drug_id": drug_id,
            "drug_name": base_result["drug_name"],
            "total_diseases_evaluated": base_result["total_diseases_evaluated"],
            "known_indications_count": base_result["known_indications_count"],
            "known_indications": base_result["known_indications"],
            "top_novel_candidates": augmented_candidates[:top_n_novel],
            "all_novel_candidates_count": len(augmented_candidates),
            "session_augmented_edges_count": getattr(self, "session_augmented_count", 0),
        }
