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
    "duloxetine": "Compound::DB00476",
    "thalidomide": "Compound::DB01041",
    "celecoxib": "Compound::DB00482",
    "disulfiram": "Compound::DB00822",
    "spironolactone": "Compound::DB00421",
    "losartan": "Compound::DB00678",
    "octreotide": "Compound::DB00104",
    "ruxolitinib": "Compound::DB08877",
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
            for dm in ("CtD", "CpD", "CcSE", "CaD"):
                e_dir = (drug_id, dm, disease_id)
                if e_dir in self.lit_edges:
                    info = self.lit_edges[e_dir]
                    pmids = list(dict.fromkeys(q["pmid"] for q in info.get("quotes", []) if q.get("pmid")))
                    if dm == "CtD":
                        lit_sign, lit_tag, lit_mech = -1, " [🟢 Trị liệu]", "therapeutic"
                        lit_beta, lit_contrib = 1.5, 1.5
                    elif dm == "CpD":
                        lit_sign, lit_tag, lit_mech = -1, " [🟢 Giảm triệu chứng]", "palliative"
                        lit_beta, lit_contrib = 0.5, 0.5
                    elif dm == "CcSE":
                        lit_sign, lit_tag, lit_mech = 1, " [🔴 Cảnh báo độc tính]", "pathogenic"
                        lit_beta, lit_contrib = -1.5, -1.5
                    else:
                        lit_sign, lit_tag, lit_mech = 0, " [⚪ Tương quan trung tính]", "neutral"
                        lit_beta, lit_contrib = 0.0, 0.0

                    paths.append({
                        "metapath": f"literature:{dm}",
                        "nodes": [drug_id, disease_id],
                        "dwpc_weight": 1.0,
                        "beta": lit_beta,
                        "contrib": lit_contrib,
                        "net_sign": int(lit_sign),
                        "mechanism_type": lit_mech,
                        "is_literature_augmented": True,
                        "lit_edge_keys": [e_dir],
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
                        lit_keys = []
                        if hasattr(self, "lit_edges"):
                            for edge_cand in ((drug_id, cm, g1), (disease_id, dm, g1), (g1, dm, disease_id)):
                                if edge_cand in self.lit_edges:
                                    lit_info.append(self.lit_edges[edge_cand])
                                    lit_keys.append(edge_cand)
                        is_lit = len(lit_info) > 0
                        pmids = [q["pmid"] for item in lit_info for q in item.get("quotes", []) if q.get("pmid")]
                        tag = " [⚡ Y văn mới]" if is_lit else ""

                        # 2-tier Signed KG tracking: CdG/CuG + DuG/DdG are transcriptional only
                        cm_s = -1 if cm == "CdG" else (1 if cm == "CuG" else 0)
                        dm_s = 1 if dm == "DuG" else (-1 if dm == "DdG" else 0)
                        p_sign = 0 if (cm == "CbG" or dm == "DaG") else (cm_s * dm_s)
                        if p_sign == -1:
                            sign_tag = " [🟢 Đảo chiều phiên mã]"
                            mech_type = "transcriptional_reversal"
                        elif p_sign == 1:
                            sign_tag = " [🔴 Đồng chiều phiên mã]"
                            mech_type = "transcriptional_congruence"
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
                            "lit_edge_keys": lit_keys,
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
                                lit_keys = []
                                if hasattr(self, "lit_edges"):
                                    for edge_cand in ((drug_id, cm, g1), (g1, bm, b), (g2, bm, b), (disease_id, dm, g2)):
                                        if edge_cand in self.lit_edges:
                                            lit_info.append(self.lit_edges[edge_cand])
                                            lit_keys.append(edge_cand)
                                is_lit = len(lit_info) > 0
                                pmids = [q["pmid"] for item in lit_info for q in item.get("quotes", []) if q.get("pmid")]
                                tag = " [⚡ Y văn mới]" if is_lit else ""

                                # 2-tier Signed KG tracking: CdG/CuG + DuG/DdG are transcriptional only
                                cm_s = -1 if cm == "CdG" else (1 if cm == "CuG" else 0)
                                dm_s = 1 if dm == "DuG" else (-1 if dm == "DdG" else 0)
                                p_sign = 0 if (cm == "CbG" or dm == "DaG") else (cm_s * dm_s)
                                if p_sign == -1:
                                    sign_tag = " [🟢 Đảo chiều phiên mã]"
                                    mech_type = "transcriptional_reversal"
                                elif p_sign == 1:
                                    sign_tag = " [🔴 Đồng chiều phiên mã]"
                                    mech_type = "transcriptional_congruence"
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
                                    "lit_edge_keys": lit_keys,
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

    @staticmethod
    def _resolve_tier_weight(admission_tier: Any) -> Optional[float]:
        """
        Maps an edge admission tier to its evidence multiplier:
        - Tier 1 / ADMITTED_KG -> 1.0
        - Tier 2 / HYPOTHESIS_EDGE -> 0.5
        - Tier 3 / QUARANTINE (or unknown) -> None (fail-closed skip)
        """
        if admission_tier is None:
            return 1.0
        t = str(admission_tier).strip().upper()
        if not t or t in ("TIER 1", "TIER_1", "TIER1", "1", "ADMITTED_KG", "ADMITTED"):
            return 1.0
        if t in ("TIER 2", "TIER_2", "TIER2", "2", "HYPOTHESIS_EDGE", "HYPOTHESIS"):
            return 0.5
        if t in ("TIER 3", "TIER_3", "TIER3", "3", "QUARANTINE", "REJECTED", "REVIEW_ONLY"):
            return None
        return None

    def _get_ontology_resolver(self, ontology_resolver: Optional[Any] = None) -> Any:
        """Returns the provided DirectedDOIDResolver or lazily initializes a cached instance."""
        if ontology_resolver is not None:
            return ontology_resolver
        if getattr(self, "_ontology_resolver", None) is None:
            from lbd.ontology.doid_dag import DirectedDOIDResolver

            self._ontology_resolver = DirectedDOIDResolver(
                hetionet_diseases=self.diseases,
            )
        return self._ontology_resolver

    def augment_session_edges(
        self,
        edges: List[Dict[str, Any]],
        resolve_ontology: bool = False,
        ontology_resolver: Optional[Any] = None,
        max_ontology_dist: int = 4,
        decay_base: float = 0.90,
    ) -> int:
        """
        Temporarily injects literature-derived edges into this session's in-memory graph.
        Does NOT touch any disk files.

        Supports backward-compatible tier weighting (`admission_tier`), LCA ontology decay
        (`lca_decay` or automatic `DirectedDOIDResolver` when `resolve_ontology=True`),
        effective evidence weight `w_ev = clip(conf * w_tier * lca_decay, 0, 1)`,
        and deduplication by `(edge_key, pmid)` taking `max(w_ev)`.
        """
        if not hasattr(self, "lit_edges"):
            self.lit_edges = {}
            self.session_augmented_count = 0

        resolver = self._get_ontology_resolver(ontology_resolver) if resolve_ontology else None

        added = 0
        for edge in edges:
            # 1. Admission tier gate (Tier 3 / QUARANTINE skipped fail-closed)
            tier_raw = edge.get("admission_tier") if "admission_tier" in edge else edge.get("tier")
            w_tier = self._resolve_tier_weight(tier_raw)
            if w_tier is None:
                continue

            s_type = edge.get("source_type") or "Compound"
            t_type = edge.get("target_type") or "Gene"
            s_raw = str(edge.get("source_standard_id") or edge.get("source") or edge.get("source_entity") or "")
            t_raw = str(edge.get("target_standard_id") or edge.get("target") or edge.get("target_entity") or "")
            if not s_raw or not t_raw:
                continue

            u_base = s_raw if "::" in s_raw else f"{s_type}::{s_raw}"
            v_base = t_raw if "::" in t_raw else f"{t_type}::{t_raw}"

            raw_lca_decay = edge.get("lca_decay")
            base_lca_decay = float(raw_lca_decay) if raw_lca_decay is not None else 1.0
            base_lca_decay = min(max(base_lca_decay, 0.0), 1.0)

            # 2. Optional DOID DAG ontology resolution for Disease endpoints
            resolved_pairs: List[Tuple[str, str, float, Optional[Dict[str, Any]]]] = []
            if resolve_ontology and resolver is not None and (s_type == "Disease" or t_type == "Disease"):
                if t_type == "Disease" and s_type != "Disease":
                    hits = resolver.resolve(t_raw, max_dist=max_ontology_dist, decay_base=decay_base)
                    for hetio_doid, res_decay, tree_dist, lca_id, match_type in hits:
                        eff_decay = min(max(base_lca_decay * float(res_decay), 0.0), 1.0)
                        resolved_pairs.append((
                            u_base,
                            f"Disease::{hetio_doid}",
                            eff_decay,
                            {
                                "original_disease_id": v_base,
                                "lca_id": lca_id,
                                "tree_dist": tree_dist,
                                "match_type": match_type,
                            },
                        ))
                elif s_type == "Disease" and t_type != "Disease":
                    hits = resolver.resolve(s_raw, max_dist=max_ontology_dist, decay_base=decay_base)
                    for hetio_doid, res_decay, tree_dist, lca_id, match_type in hits:
                        eff_decay = min(max(base_lca_decay * float(res_decay), 0.0), 1.0)
                        resolved_pairs.append((
                            f"Disease::{hetio_doid}",
                            v_base,
                            eff_decay,
                            {
                                "original_disease_id": u_base,
                                "lca_id": lca_id,
                                "tree_dist": tree_dist,
                                "match_type": match_type,
                            },
                        ))
                else:
                    resolved_pairs.append((u_base, v_base, base_lca_decay, None))
                if not resolved_pairs:
                    continue
            else:
                resolved_pairs.append((u_base, v_base, base_lca_decay, None))

            # Map relation and metaedge to canonical Hetionet schema
            metaedge_raw = edge.get("metaedge")
            rel = (edge.get("relation_type") or edge.get("canonical_relation") or "").upper()
            conf = float(edge.get("confidence_score") if edge.get("confidence_score") is not None else 0.8)

            raw_quotes = edge.get("quotes") or []
            if not raw_quotes and edge.get("evidence"):
                raw_quotes = [{"pmid": edge.get("pmid", ""), "text": edge.get("evidence", "")}]
            elif not raw_quotes and edge.get("pmid"):
                raw_quotes = [{"pmid": edge.get("pmid", ""), "text": ""}]

            for u, v, lca_decay, onto_meta in resolved_pairs:
                if edge.get("source_entity") and u not in self.names:
                    self.names[u] = edge["source_entity"]
                if edge.get("target_entity") and v not in self.names:
                    self.names[v] = edge["target_entity"]

                metaedge = metaedge_raw
                edge_u, edge_v = u, v

                # Map Compound -> Gene
                if s_type == "Compound" and t_type == "Gene":
                    if metaedge in ("CiG", "CdG") or any(w in rel for w in ("INHIBIT", "DOWN", "DECREAS")):
                        metaedge = "CdG"
                    elif metaedge in ("CaG", "CuG") or any(w in rel for w in ("ACTIVAT", "UP", "INCREAS", "STIMULAT")):
                        metaedge = "CuG"
                    else:
                        metaedge = "CbG"
                # Map Gene -> Disease (normalize to Hetionet convention: Disease -> Gene)
                elif s_type == "Gene" and t_type == "Disease":
                    if metaedge == "DuG" or any(w in rel for w in ("UP", "OVEREXPRESS")):
                        metaedge = "DuG"
                    elif metaedge == "DdG" or any(w in rel for w in ("DOWN", "UNDEREXPRESS")):
                        metaedge = "DdG"
                    else:
                        metaedge = "DaG"
                    edge_u, edge_v = v, u
                # Map Disease -> Gene (already Hetionet convention: Disease -> Gene)
                elif s_type == "Disease" and t_type == "Gene":
                    if metaedge == "DuG" or any(w in rel for w in ("UP", "OVEREXPRESS")):
                        metaedge = "DuG"
                    elif metaedge == "DdG" or any(w in rel for w in ("DOWN", "UNDEREXPRESS")):
                        metaedge = "DdG"
                    else:
                        metaedge = "DaG"
                # Map Compound -> Disease (no default CpD coercion; neutral/unclear -> CaD)
                elif s_type == "Compound" and t_type == "Disease":
                    if metaedge == "CcSE" or any(w in rel for w in ("CAUSE", "INDUC", "TOXIC", "ADVERSE", "SIDE_EFFECT", "CONTRAINDICAT")):
                        metaedge = "CcSE"
                    elif any(w in rel for w in ("PALLIAT", "SYMPTOM", "RELIEV")):
                        metaedge = "CpD"
                    elif metaedge == "CtD" or any(w in rel for w in ("TREAT", "ALLEVIAT", "PREVENT", "IMPROV")):
                        metaedge = "CtD"
                    else:
                        metaedge = "CaD"
                elif s_type == "Gene" and t_type == "Gene":
                    metaedge = "GiG"
                else:
                    metaedge = metaedge or "CbG"

                w_ev = min(max(conf * w_tier * lca_decay, 0.0), 1.0)

                # Check if edge already exists in this session
                existing = self.adj.get((edge_u, metaedge), [])
                is_new = edge_v not in existing
                if is_new:
                    self.adj[(edge_u, metaedge)].append(edge_v)
                    self.out_deg[(edge_u, metaedge)] += 1
                    self.in_deg[(edge_v, metaedge)] += 1
                    if metaedge in ("GiG", "GcG", "GpBP", "GpPW", "GpMF", "GpCC", "DaG", "DuG", "DdG"):
                        self.adj[(edge_v, metaedge + "_rev")].append(edge_u)
                    added += 1

                edge_key = (edge_u, metaedge, edge_v)
                quote_keys: List[Tuple[str, str]] = []
                for q in raw_quotes:
                    pmid_k = str(q.get("pmid") or "").strip()
                    quote_keys.append(("pmid", pmid_k) if pmid_k else ("text", str(q.get("text") or "").strip()))
                if not quote_keys:
                    quote_keys = [("anon", "")]

                if edge_key not in self.lit_edges:
                    dedup_quotes = []
                    seen_keys: set[Tuple[str, str]] = set()
                    pmid_weights: Dict[Tuple[str, str], float] = {}
                    for q, dedup_k in zip(raw_quotes, quote_keys):
                        if dedup_k not in seen_keys:
                            seen_keys.add(dedup_k)
                            dedup_quotes.append(q)
                        pmid_weights[dedup_k] = max(pmid_weights.get(dedup_k, 0.0), w_ev)
                    if not raw_quotes:
                        pmid_weights[("anon", "")] = w_ev

                    eff_w_ev = max(pmid_weights.values()) if pmid_weights else w_ev
                    entry_dict: Dict[str, Any] = {
                        "source": edge_u,
                        "target": edge_v,
                        "metaedge": metaedge,
                        "relation_type": edge.get("relation_type", metaedge),
                        "confidence": conf,
                        "w_tier": w_tier,
                        "lca_decay": lca_decay,
                        "w_ev": eff_w_ev,
                        "admission_tier": tier_raw if tier_raw is not None else "Tier 1",
                        "sign": int(edge.get("sign") or 0),
                        "quotes": dedup_quotes,
                        "_seen_quote_keys": seen_keys,
                        "_pmid_weights": pmid_weights,
                        "is_novel_edge": is_new,
                    }
                    if onto_meta is not None:
                        entry_dict["ontology_resolution"] = onto_meta
                    self.lit_edges[edge_key] = entry_dict
                else:
                    entry = self.lit_edges[edge_key]
                    seen_keys = entry.setdefault("_seen_quote_keys", set())
                    pmid_weights = entry.setdefault("_pmid_weights", {})
                    for q, dedup_k in zip(raw_quotes, quote_keys):
                        if dedup_k not in seen_keys:
                            seen_keys.add(dedup_k)
                            entry["quotes"].append(q)
                        pmid_weights[dedup_k] = max(pmid_weights.get(dedup_k, 0.0), w_ev)
                    if not raw_quotes:
                        pmid_weights[("anon", "")] = max(pmid_weights.get(("anon", ""), 0.0), w_ev)

                    entry["confidence"] = max(entry["confidence"], conf)
                    entry["w_tier"] = max(float(entry.get("w_tier", 1.0)), w_tier)
                    entry["lca_decay"] = max(float(entry.get("lca_decay", 1.0)), lca_decay)
                    entry["w_ev"] = max(pmid_weights.values()) if pmid_weights else max(float(entry.get("w_ev", 0.0)), w_ev)

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
        max_lit_logit: float = 1.5,
        lit_tau: float = 2.0,
    ) -> Dict[str, Any]:
        """
        Executes M2 ranking with dynamic session-scoped literature augmentation:
        1. Computes baseline M2 + Signed Graph NTR score from Hetionet topology & signatures.
        2. Evaluates newly ingested literature bridge paths and direct assertions, deduplicated by
           unique literature edges (UniqueLitEdgesUsed) and (edge_key, pmid).
        3. Applies bounded logit-space tanh shift and enforces |session_score - base_score| <= max_lit_delta.
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

        lit_edges_map = getattr(self, "lit_edges", {})
        augmented_candidates = []
        for cand in novel_candidates:
            dis_id = cand["disease_id"]
            base_score = cand["score"]

            n_paths, top_paths, all_lit_paths = 0, [], []
            if self._graph_loaded:
                n_paths, top_paths, all_lit_paths = self.explain_pair_dfs(
                    drug_id, dis_id, top_k=dfs_top_k, return_all_lit=True
                )

            # 1. Direct Compound <-> Disease literature edges (deduplicated by edge_key & PMID)
            evidence_weight = 0.0
            adverse_count = 0
            unique_direct_keys = set()
            unique_direct_scored_keys = set()
            for edge_key, dl in lit_edges_map.items():
                u_k, m_edge, v_k = edge_key
                if not ((u_k == drug_id and v_k == dis_id) or (u_k == dis_id and v_k == drug_id)):
                    continue
                unique_direct_keys.add(edge_key)
                w_ev = float(dl.get("w_ev", dl.get("confidence", 0.8)))
                # Neutral/unclear association (CaD) carries zero logit shift
                if m_edge == "CaD":
                    continue
                elif m_edge == "CtD":
                    unique_direct_scored_keys.add(edge_key)
                    evidence_weight += 2.0 * w_ev
                elif m_edge == "CpD":
                    unique_direct_scored_keys.add(edge_key)
                    evidence_weight += 0.5 * w_ev
                elif m_edge == "CcSE":
                    unique_direct_scored_keys.add(edge_key)
                    adverse_count += 1
                    evidence_weight -= 1.5 * w_ev

            # 2. Indirect path-level literature edges aggregated by UniqueLitEdgesUsed (max dwpc per edge)
            unique_lit_edge_max_dwpc: Dict[Tuple[str, str, str], float] = {}
            for lp in all_lit_paths:
                if str(lp.get("metapath", "")).startswith("literature:"):
                    continue
                lp_w = max(float(lp.get("dwpc_weight", 0.0)), 0.0)
                for ek in lp.get("lit_edge_keys", []):
                    if ek[1] == "CaD":
                        continue
                    prev = unique_lit_edge_max_dwpc.get(ek, 0.0)
                    if lp_w > prev:
                        unique_lit_edge_max_dwpc[ek] = lp_w

            for ek, max_w in unique_lit_edge_max_dwpc.items():
                dl_ind = lit_edges_map.get(ek, {})
                w_ev_ind = float(dl_ind.get("w_ev", dl_ind.get("confidence", 1.0)))
                evidence_weight += 1.2 * max(max_w, 0.05) * w_ev_ind

            # 3. Bounded logit-space shift via tanh + strict max_lit_delta clamp (0% saturation at 1.0000)
            clamped_base = min(max(base_score, 0.001), 0.999)
            base_logit = math.log(clamped_base / (1.0 - clamped_base))
            tau_safe = max(float(lit_tau), 1e-6)
            delta_logit = float(max_lit_logit) * math.tanh(evidence_weight / tau_safe)
            raw_session_prob = 1.0 / (1.0 + math.exp(-(base_logit + delta_logit)))

            if max_lit_delta is not None and max_lit_delta > 0:
                lo = max(0.0, base_score - max_lit_delta)
                hi = min(0.9999, base_score + max_lit_delta)
                session_prob = min(max(raw_session_prob, lo), hi)
            else:
                session_prob = min(max(raw_session_prob, 0.0001), 0.9999)

            session_score = min(round(session_prob, 4), 0.9999)
            delta = round(session_score - base_score, 4)
            cand_entry = dict(cand)
            cand_entry["base_score"] = base_score
            cand_entry["literature_delta"] = delta
            cand_entry["score"] = session_score
            cand_entry["unique_lit_edges_used"] = len(unique_lit_edge_max_dwpc) + len(unique_direct_scored_keys)
            cand_entry["session_evidence_count"] = len(unique_lit_edge_max_dwpc) + len(unique_direct_keys)
            cand_entry["session_status"] = "augmented" if delta > 0 else ("penalized" if delta < 0 else "baseline")
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
