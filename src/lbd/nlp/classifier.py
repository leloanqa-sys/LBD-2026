#!/usr/bin/env python3
"""
Reformed Multi-Entity, Multi-Class Biomedical Relation Extraction Engine (V3).
Location: nlp/bio_re_engine_v3.py (Strictly synced with dapagliflozin-lbd/src/lbd/nlp/classifier.py)

Literature Grounding:
1. Piecewise Zoned Representation (`Left`, `Middle`, `Right` around target entity pairs):
   - Zeng et al. (EMNLP 2015): Distant Supervision for Relation Extraction via Piecewise Convolutional Neural Networks.
2. Document-Level Mention Pooling & Entity Centrality / Discourse Dominance:
   - Gu et al. (Database 2016) & Verga et al. (NAACL 2018, BRAN): Chemical-induced disease extraction across
     intra-sentence and cross-sentence mentions using document centrality and title-to-abstract bridging.
3. Multi-Entity Biomedical Coverage & 4-Class Polarity (`NO_RELATION`, `POSITIVE`, `NEGATIVE`, `ASSOCIATION`):
   - Luo et al. (Briefings in Bioinformatics 2022, BioRED).
4. Exact-Offset Sentence Segmentation (`_mask_preserving_length`):
   - Griffis et al. (BMC Bioinformatics 2016): Zero offset drift across biomedical abbreviations and numeric literals.
"""
import math
import pickle
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

sys.modules["bio_re_engine_v3"] = sys.modules[__name__]
sys.modules["lbd.nlp.bio_re_engine_v3"] = sys.modules[__name__]

import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

try:
    from .sentence_splitter import split_biomedical_sentences
except ImportError:
    split_biomedical_sentences = None

_SENT_SPLIT_RX = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\[\"])")
# Case-insensitive placeholder & word tokenizer so lowercased @chemical1$ is always matched in ZonedEmbeddingPooler
_WORD_TOKEN_RX = re.compile(r"@[A-Za-z0-9_]+\$|[a-z0-9]+(?:-[a-z0-9]+)*|[(),.;:%<>+\-/]", re.IGNORECASE)
_IS_WORD_OR_TAG_RX = re.compile(r"^(?:@[a-z0-9_]+\$|[a-z0-9]+(?:-[a-z0-9]+)*)$", re.IGNORECASE)

CLASS_TO_INT = {
    "NO_RELATION": 0,
    "POSITIVE": 1,     # +1: Positive_Correlation (CAUSES / INCREASES / ACTIVATES / INDUCES)
    "NEGATIVE": 2,     # -1: Negative_Correlation (TREATS / INHIBITS / DECREASES / PREVENTS)
    "ASSOCIATION": 3,  #  0: Association / Bind / Cotreatment / Drug_Interaction
}
INT_TO_CLASS = {v: k for k, v in CLASS_TO_INT.items()}

VALID_BIORED_PAIR_TYPES = {
    ("Chemical", "Disease"),
    ("Chemical", "Gene"),
    ("Disease", "Gene"),
    ("Gene", "Gene"),
    ("Chemical", "Chemical"),
    ("Disease", "Variant"),
    ("Chemical", "Variant"),
    ("Gene", "Variant"),
    ("Variant", "Variant"),
}


def canon_pair(id1: str, id2: str) -> Tuple[str, str]:
    return (id1, id2) if id1 <= id2 else (id2, id1)


def split_doc_sentences(title: str, abstract: str, title_sep_len: int = 1) -> List[Dict[str, Any]]:
    sentences: List[Dict[str, Any]] = []
    if title:
        sentences.append({
            "sent_idx": 0,
            "is_title": True,
            "start": 0,
            "end": len(title),
            "text": title,
        })
    offset = len(title) + title_sep_len if title else 0
    if not abstract:
        return sentences

    if split_biomedical_sentences is not None:
        spans = split_biomedical_sentences(abstract, start_offset=offset)
        for s_idx, (c_start, c_end, s_text) in enumerate(spans, start=1):
            sentences.append({
                "sent_idx": s_idx,
                "is_title": False,
                "start": c_start,
                "end": c_end,
                "text": s_text,
            })
        return sentences

    last = 0
    s_idx = 1
    for m in _SENT_SPLIT_RX.finditer(abstract):
        seg = abstract[last:m.start()]
        if seg.strip():
            sentences.append({
                "sent_idx": s_idx,
                "is_title": False,
                "start": offset + last,
                "end": offset + m.start(),
                "text": seg,
            })
            s_idx += 1
        last = m.end()

    tail = abstract[last:]
    if tail.strip():
        sentences.append({
            "sent_idx": s_idx,
            "is_title": False,
            "start": offset + last,
            "end": offset + len(abstract),
            "text": tail,
        })
    return sentences


def blind_multi_entity_sentence(
    sent: Dict[str, Any],
    sent_entities: List[Dict[str, Any]],
    e1_id: str,
    e2_id: str,
    e1_type: str,
    e2_type: str,
) -> Tuple[str, List[int], List[int], int]:
    """Replaces surface entity mentions with abstract type placeholders (`@CHEMICAL1$`, `@GENE2$`, `@OTHER_DISEASE$`)."""
    s_start = sent["start"]
    text = sent["text"]
    s_len = len(text)

    spans = []
    for e in sent_entities:
        eid = e.get("entity_id") or e.get("mesh_id") or ""
        etype = e.get("type", "Entity").upper()
        rel_s = max(0, e["start"] - s_start)
        rel_e = min(s_len, e["end"] - s_start)
        if rel_e <= rel_s:
            continue
        is_target = 1 if eid in (e1_id, e2_id) else 0
        spans.append((rel_s, rel_e, is_target, eid, etype))

    spans.sort(key=lambda x: (x[0], -x[2], -(x[1] - x[0])))
    filtered = []
    last_end = -1
    for sp in spans:
        if sp[0] >= last_end:
            filtered.append(sp)
            last_end = sp[1]

    pieces: List[str] = []
    cursor = 0
    cur_len = 0
    e1_pos: List[int] = []
    e2_pos: List[int] = []
    other_count = 0

    tag1 = f"@{e1_type.upper()}1$"
    tag2 = f"@{e2_type.upper()}2$"

    for rel_s, rel_e, _, eid, etype in filtered:
        prefix = text[cursor:rel_s]
        pieces.append(prefix)
        cur_len += len(prefix)

        if eid == e1_id:
            tok = f" {tag1} "
            e1_pos.append(cur_len + 1)
        elif eid == e2_id:
            tok = f" {tag2} "
            e2_pos.append(cur_len + 1)
        else:
            tok = f" @OTHER_{etype}$ "
            other_count += 1

        pieces.append(tok)
        cur_len += len(tok)
        cursor = rel_e

    pieces.append(text[cursor:])
    return "".join(pieces), e1_pos, e2_pos, other_count


def _build_doc_stats(doc: Dict[str, Any], sentences: List[Dict[str, Any]]) -> Dict[str, Any]:
    entities = doc["entities"]
    sent_to_ents: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    id_to_sents: Dict[str, Set[int]] = defaultdict(set)
    idx_to_sent = {s["sent_idx"]: s for s in sentences}
    type_counts: Counter = Counter()
    id_counts: Counter = Counter()
    id_types: Dict[str, str] = {}

    for e in entities:
        eid = e.get("entity_id") or e.get("mesh_id")
        etype = e.get("type", "")
        if not eid:
            continue
        type_counts[etype] += 1
        id_counts[eid] += 1
        id_types[eid] = etype

    max_type_mentions: Dict[str, int] = defaultdict(int)
    for eid, cnt in id_counts.items():
        et = id_types[eid]
        if cnt > max_type_mentions[et]:
            max_type_mentions[et] = cnt

    for s in sentences:
        si = s["sent_idx"]
        for e in entities:
            if e["start"] >= s["start"] and e["end"] <= s["end"]:
                sent_to_ents[si].append(e)
                eid = e.get("entity_id") or e.get("mesh_id")
                if eid:
                    id_to_sents[eid].add(si)

    return {
        "sent_to_ents": sent_to_ents,
        "id_to_sents": id_to_sents,
        "idx_to_sent": idx_to_sent,
        "type_counts": type_counts,
        "max_type_mentions": max_type_mentions,
    }


def extract_zoned_representation(
    doc: Dict[str, Any],
    e1_id: str,
    e2_id: str,
    e1_type: str,
    e2_type: str,
    sentences: List[Dict[str, Any]],
    return_evidence: bool = False,
):
    """Extracts structural/topological discourse features and 3 syntactic text zones (`left_zone`, `mid_zone`, `right_zone`)
    for Piecewise Zoned Embedding Pooling (Zeng et al., 2015; Gu et al., 2016).
    """
    entities = doc["entities"]
    e1_mentions = [e for e in entities if (e.get("entity_id") or e.get("mesh_id")) == e1_id]
    e2_mentions = [e for e in entities if (e.get("entity_id") or e.get("mesh_id")) == e2_id]

    total_sents = max(1, len(sentences))
    doc_len = max(1, len(doc.get("title", "")) + len(doc.get("abstract", "")) + 1)

    doc_stats = _build_doc_stats(doc, sentences)
    sent_to_ents = doc_stats["sent_to_ents"]
    e1_sents = doc_stats["id_to_sents"][e1_id]
    e2_sents = doc_stats["id_to_sents"][e2_id]

    co_sents = sorted(e1_sents & e2_sents)
    min_sent_dist = 99
    if co_sents:
        min_sent_dist = 0
    elif e1_sents and e2_sents:
        min_sent_dist = min(abs(a - b) for a in e1_sents for b in e2_sents)

    e1_in_title = 1.0 if 0 in e1_sents else 0.0
    e2_in_title = 1.0 if 0 in e2_sents else 0.0
    both_in_title = 1.0 if (0 in co_sents) else 0.0
    last_sent_idx = sentences[-1]["sent_idx"] if sentences else 0
    in_last_sent = 1.0 if (last_sent_idx in co_sents) else 0.0

    first_e1_pos = min((e["start"] for e in e1_mentions), default=doc_len) / doc_len
    first_e2_pos = min((e["start"] for e in e2_mentions), default=doc_len) / doc_len
    last_e1_pos = max((e["end"] for e in e1_mentions), default=0) / doc_len
    last_e2_pos = max((e["end"] for e in e2_mentions), default=0) / doc_len

    total_type1_mentions = max(1, doc_stats["type_counts"][e1_type])
    total_type2_mentions = max(1, doc_stats["type_counts"][e2_type])
    e1_dominance = len(e1_mentions) / total_type1_mentions
    e2_dominance = len(e2_mentions) / total_type2_mentions
    is_most_freq_e1 = 1.0 if len(e1_mentions) == doc_stats["max_type_mentions"][e1_type] else 0.0
    is_most_freq_e2 = 1.0 if len(e2_mentions) == doc_stats["max_type_mentions"][e2_type] else 0.0

    union_sents = len(e1_sents | e2_sents)
    sent_jaccard = len(co_sents) / union_sents if union_sents > 0 else 0.0
    p_e2_given_e1 = len(co_sents) / len(e1_sents) if e1_sents else 0.0
    p_e1_given_e2 = len(co_sents) / len(e2_sents) if e2_sents else 0.0

    left_zones: List[str] = []
    mid_zones: List[str] = []
    right_zones: List[str] = []
    zoned_ngram_tokens: List[str] = []

    min_tok_dist = 100.0
    e1_before_e2_count = 0
    e2_before_e1_count = 0
    min_other_between = 10.0
    same_clause_count = 0
    closest_in_sent_count = 0
    best_evidence = ""

    tag1 = f"@{e1_type.upper()}1$"
    tag2 = f"@{e2_type.upper()}2$"
    pair_type_tag = f"PAIR_{min(e1_type, e2_type).upper()}_{max(e1_type, e2_type).upper()}"
    zoned_ngram_tokens.append(pair_type_tag)

    for si in co_sents:
        s_obj = doc_stats["idx_to_sent"][si]
        if not best_evidence or (not s_obj["is_title"] and best_evidence == sentences[0]["text"]):
            best_evidence = s_obj["text"]

        blinded, _, _, _ = blind_multi_entity_sentence(
            s_obj, sent_to_ents[si], e1_id, e2_id, e1_type, e2_type
        )
        toks = [t.lower() for t in _WORD_TOKEN_RX.findall(blinded)]
        idx1 = [i for i, t in enumerate(toks) if t == tag1.lower()]
        idx2 = [i for i, t in enumerate(toks) if t == tag2.lower()]
        if not idx1 or not idx2:
            continue

        best_pair = min(
            ((i, j) for i in idx1 for j in idx2),
            key=lambda p: abs(p[0] - p[1]),
        )
        i1, i2 = best_pair
        dist = abs(i1 - i2)
        if dist < min_tok_dist:
            min_tok_dist = float(dist)

        if i1 < i2:
            e1_before_e2_count += 1
            lo, hi = i1, i2
            dir_tag = "DIR_1TO2"
        else:
            e2_before_e1_count += 1
            lo, hi = i2, i1
            dir_tag = "DIR_2TO1"

        left_raw = toks[max(0, lo - 6):lo]
        mid_raw = toks[lo + 1:min(hi, lo + 20)]
        right_raw = toks[hi + 1:min(len(toks), hi + 7)]

        if not any(t in (";", "whereas", "while", "however", "but", "although") for t in mid_raw):
            same_clause_count += 1

        # Filter bare punctuation tokens so prefixing never produces degenerate 'l_', 'm_', 'r_' features
        left_slice = [t for t in left_raw if _IS_WORD_OR_TAG_RX.match(t)]
        mid_slice = [t for t in mid_raw if _IS_WORD_OR_TAG_RX.match(t)]
        right_slice = [t for t in right_raw if _IS_WORD_OR_TAG_RX.match(t)]

        other_in_mid = sum(1 for t in mid_slice if t.startswith("@other_"))
        if other_in_mid < min_other_between:
            min_other_between = float(other_in_mid)
        if other_in_mid == 0:
            closest_in_sent_count += 1

        left_zones.append(" ".join(left_slice))
        mid_zones.append(" ".join(mid_slice))
        right_zones.append(" ".join(right_slice))

        zoned_ngram_tokens.append(dir_tag)
        zoned_ngram_tokens.extend([f"L_{t}" for t in left_slice])
        zoned_ngram_tokens.extend([f"M_{t}" for t in mid_slice])
        zoned_ngram_tokens.extend([f"R_{t}" for t in right_slice])
        for k in range(len(mid_slice) - 1):
            zoned_ngram_tokens.append(f"MB_{mid_slice[k]}_{mid_slice[k+1]}")

    if not co_sents and e1_sents and e2_sents:
        s1_idx, s2_idx = min(
            ((a, b) for a in e1_sents for b in e2_sents),
            key=lambda p: abs(p[0] - p[1]),
        )
        s1_obj = doc_stats["idx_to_sent"][s1_idx]
        s2_obj = doc_stats["idx_to_sent"][s2_idx]
        # Never concatenate sentences with '[...]' so verbatim evidence verification remains 100% valid
        best_evidence = s2_obj["text"] if not s2_obj["is_title"] else s1_obj["text"]

        b1, _, _, _ = blind_multi_entity_sentence(s1_obj, sent_to_ents[s1_idx], e1_id, e2_id, e1_type, e2_type)
        b2, _, _, _ = blind_multi_entity_sentence(s2_obj, sent_to_ents[s2_idx], e1_id, e2_id, e1_type, e2_type)
        toks1 = [t.lower() for t in _WORD_TOKEN_RX.findall(b1) if _IS_WORD_OR_TAG_RX.match(t)]
        toks2 = [t.lower() for t in _WORD_TOKEN_RX.findall(b2) if _IS_WORD_OR_TAG_RX.match(t)]

        pos1 = [i for i, t in enumerate(toks1) if t == tag1.lower()]
        pos2 = [i for i, t in enumerate(toks2) if t == tag2.lower()]
        c1 = pos1[0] if pos1 else 0
        c2 = pos2[0] if pos2 else 0
        w1 = toks1[max(0, c1 - 6):min(len(toks1), c1 + 7)]
        w2 = toks2[max(0, c2 - 6):min(len(toks2), c2 + 7)]

        left_zones.append(" ".join(w1))
        right_zones.append(" ".join(w2))
        zoned_ngram_tokens.extend([f"X1_{t}" for t in w1])
        zoned_ngram_tokens.extend([f"X2_{t}" for t in w2])

    p_sorted = tuple(sorted([e1_type, e2_type]))
    feats: Dict[str, float] = {
        "pair_is_chem_dis": 1.0 if p_sorted == ("Chemical", "Disease") else 0.0,
        "pair_is_chem_gene": 1.0 if p_sorted == ("Chemical", "Gene") else 0.0,
        "pair_is_dis_gene": 1.0 if p_sorted == ("Disease", "Gene") else 0.0,
        "pair_is_gene_gene": 1.0 if p_sorted == ("Gene", "Gene") else 0.0,
        "pair_is_chem_chem": 1.0 if p_sorted == ("Chemical", "Chemical") else 0.0,
        "pair_is_dis_var": 1.0 if p_sorted == ("Disease", "Variant") else 0.0,
        "co_sent_count": float(len(co_sents)),
        "has_co_sent": 1.0 if co_sents else 0.0,
        "same_clause_count": float(same_clause_count),
        "closest_in_sent_count": float(closest_in_sent_count),
        "in_last_sent": in_last_sent,
        "min_sent_dist": float(min(min_sent_dist, 12)),
        "inv_sent_dist": 1.0 / (1.0 + float(min_sent_dist)),
        "min_tok_dist": float(min(min_tok_dist, 50.0)),
        "inv_tok_dist": 1.0 / (1.0 + float(min_tok_dist)) if co_sents else 0.0,
        "e1_mentions": float(len(e1_mentions)),
        "e2_mentions": float(len(e2_mentions)),
        "e1_dominance": e1_dominance,
        "e2_dominance": e2_dominance,
        "is_most_freq_e1": is_most_freq_e1,
        "is_most_freq_e2": is_most_freq_e2,
        "sent_jaccard": sent_jaccard,
        "p_e2_given_e1": p_e2_given_e1,
        "p_e1_given_e2": p_e1_given_e2,
        "e1_sent_frac": float(len(e1_sents)) / total_sents,
        "e2_sent_frac": float(len(e2_sents)) / total_sents,
        "e1_in_title": e1_in_title,
        "e2_in_title": e2_in_title,
        "both_in_title": both_in_title,
        "title_to_body_bridge": 1.0 if ((e1_in_title or e2_in_title) and min_sent_dist <= 3) else 0.0,
        "first_e1_pos": first_e1_pos,
        "first_e2_pos": first_e2_pos,
        "last_e1_pos": last_e1_pos,
        "last_e2_pos": last_e2_pos,
        "e1_before_e2": float(e1_before_e2_count),
        "e2_before_e1": float(e2_before_e1_count),
        "min_other_between": min_other_between if co_sents else 5.0,
    }

    if return_evidence:
        return (
            feats,
            " ".join(left_zones),
            " ".join(mid_zones),
            " ".join(right_zones),
            " ".join(zoned_ngram_tokens),
            best_evidence,
        )
    return (
        feats,
        " ".join(left_zones),
        " ".join(mid_zones),
        " ".join(right_zones),
        " ".join(zoned_ngram_tokens),
    )


class ZonedEmbeddingPooler:
    """Learns dense biomedical word embeddings (`d = 48`) on entity-blinded text and performs
    Piecewise Zoned Mean & Max Pooling across `[Left_Zone, Middle_Zone, Right_Zone]`.
    """

    def __init__(self, embed_dim: int = 48, max_vocab: int = 8000, random_state: int = 42):
        self.embed_dim = embed_dim
        self.max_vocab = max_vocab
        self.random_state = random_state
        self.word_tfidf = TfidfVectorizer(
            ngram_range=(1, 1),
            min_df=2,
            max_features=max_vocab,
            token_pattern=r"@[a-z0-9_]+\$|[a-z0-9]+(?:-[a-z0-9]+)*",
        )
        self.svd = TruncatedSVD(n_components=embed_dim, random_state=random_state)
        self.word_to_vec: Dict[str, np.ndarray] = {}
        self.actual_dim: int = embed_dim

    def fit(self, zone_texts: List[str]) -> "ZonedEmbeddingPooler":
        tfidf_mat = self.word_tfidf.fit_transform(zone_texts)
        n_comp = min(self.embed_dim, tfidf_mat.shape[1] - 1, tfidf_mat.shape[0] - 1)
        self.actual_dim = max(8, n_comp)
        self.svd = TruncatedSVD(n_components=self.actual_dim, random_state=self.random_state)
        self.svd.fit(tfidf_mat)

        components = self.svd.components_.T
        vocab = self.word_tfidf.vocabulary_
        self.word_to_vec = {
            word: components[idx].astype(np.float32)
            for word, idx in vocab.items()
        }
        return self

    def _pool_single_zone(self, text: str) -> np.ndarray:
        d = self.actual_dim
        if not text:
            return np.zeros(2 * d, dtype=np.float32)
        toks = _WORD_TOKEN_RX.findall(text.lower())
        vecs = [self.word_to_vec[t] for t in toks if t in self.word_to_vec]
        if not vecs:
            return np.zeros(2 * d, dtype=np.float32)
        mat = np.vstack(vecs)
        mean_pool = np.mean(mat, axis=0)
        max_pool = np.max(mat, axis=0)
        return np.concatenate([mean_pool, max_pool])

    def transform_zones(
        self,
        left_texts: List[str],
        mid_texts: List[str],
        right_texts: List[str],
    ) -> np.ndarray:
        n = len(left_texts)
        out = np.zeros((n, 6 * self.actual_dim), dtype=np.float32)
        for i in range(n):
            p_l = self._pool_single_zone(left_texts[i])
            p_m = self._pool_single_zone(mid_texts[i])
            p_r = self._pool_single_zone(right_texts[i])
            out[i] = np.concatenate([p_l, p_m, p_r])
        return out


class ZonedPoolingMultiClassBioREEngineV3:
    """Reformed V3 Engine: Entity-Blinded Zoned Embedding Pooling + Neural MLP + Multi-Class Softmax."""

    def __init__(self, embed_dim: int = 48, random_state: int = 42):
        self.embed_dim = embed_dim
        self.random_state = random_state
        self.dict_vec = DictVectorizer(sparse=False)
        self.struct_scaler = StandardScaler()
        self.pooler = ZonedEmbeddingPooler(embed_dim=embed_dim, random_state=random_state)
        self.dense_scaler = StandardScaler()

        self.zoned_tfidf = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=2,
            max_features=12000,
            sublinear_tf=True,
            token_pattern=r"[A-Za-z0-9_@$+\-]+",
        )

        self.mlp_clf = MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            solver="adam",
            alpha=1e-3,
            batch_size=128,
            learning_rate_init=1e-3,
            max_iter=35,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=5,
            random_state=random_state,
        )

        self.softmax_linear_clf = LogisticRegression(
            C=0.65,
            class_weight="balanced",
            solver="lbfgs",
            max_iter=800,
            random_state=random_state,
        )

        self.rf_clf = RandomForestClassifier(
            n_estimators=200,
            max_depth=14,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=random_state,
            n_jobs=1,
        )

        self.ensemble_weights: Tuple[float, float, float] = (0.15, 0.45, 0.40)
        self.relation_threshold: float = 0.50
        self.high_recall_threshold: float = 0.40
        self.is_fitted: bool = False
        self.classes_: List[int] = [0, 1, 2, 3]

    def _extract_candidates_from_docs(
        self, docs: List[Dict[str, Any]], dataset_mode: str = "biored"
    ) -> Tuple[List[Dict[str, float]], List[str], List[str], List[str], List[str], np.ndarray, List[Dict[str, Any]]]:
        feat_dicts: List[Dict[str, float]] = []
        left_list: List[str] = []
        mid_list: List[str] = []
        right_list: List[str] = []
        zoned_list: List[str] = []
        labels: List[int] = []
        meta_list: List[Dict[str, Any]] = []

        for doc in docs:
            pmid = str(doc["pmid"])
            sep_len = int(doc.get("title_sep_len", 1))
            sentences = split_doc_sentences(doc.get("title", ""), doc.get("abstract", ""), title_sep_len=sep_len)

            id_to_type: Dict[str, str] = {}
            id_to_name: Dict[str, str] = {}
            for e in doc["entities"]:
                eid = e.get("entity_id") or e.get("mesh_id") or ""
                etype = e.get("type", "")
                if not eid or etype in ("Species", "CellLine"):
                    continue
                id_to_type[eid] = etype
                if eid not in id_to_name:
                    id_to_name[eid] = e.get("text") or e.get("name") or eid

            gold_map: Dict[Tuple[str, str], str] = {}
            if dataset_mode == "biored":
                for r in doc.get("relations", []):
                    pair = canon_pair(r["entity1_id"], r["entity2_id"])
                    pol = r.get("polarity_class", "ASSOCIATION")
                    if pair not in gold_map or pol in ("POSITIVE", "NEGATIVE"):
                        gold_map[pair] = pol
            else:
                for c_id, d_id in doc.get("cid_relations", []):
                    gold_map[(c_id, d_id)] = "POSITIVE"

            unique_ids = sorted(id_to_type.keys())
            if dataset_mode == "bc5cdr":
                chem_ids = [eid for eid in unique_ids if id_to_type[eid] == "Chemical"]
                dis_ids = [eid for eid in unique_ids if id_to_type[eid] == "Disease"]
                candidate_pairs = [(c, d) for c in chem_ids for d in dis_ids]
            else:
                candidate_pairs = []
                for i in range(len(unique_ids)):
                    for j in range(i + 1, len(unique_ids)):
                        id1, id2 = unique_ids[i], unique_ids[j]
                        t_pair = tuple(sorted([id_to_type[id1], id_to_type[id2]]))
                        if t_pair in VALID_BIORED_PAIR_TYPES:
                            candidate_pairs.append((id1, id2))

            for id1, id2 in candidate_pairs:
                t1, t2 = id_to_type[id1], id_to_type[id2]
                f_dict, l_str, m_str, r_str, z_str, best_ev = extract_zoned_representation(
                    doc, id1, id2, t1, t2, sentences, return_evidence=True
                )
                key = (id1, id2) if dataset_mode == "bc5cdr" else canon_pair(id1, id2)
                gold_pol = gold_map.get(key, "NO_RELATION")
                y = CLASS_TO_INT[gold_pol]

                feat_dicts.append(f_dict)
                left_list.append(l_str)
                mid_list.append(m_str)
                right_list.append(r_str)
                zoned_list.append(z_str)
                labels.append(y)
                meta_list.append({
                    "pmid": pmid,
                    "entity1_id": id1,
                    "entity2_id": id2,
                    "entity1_type": t1,
                    "entity2_type": t2,
                    "entity1_name": id_to_name[id1],
                    "entity2_name": id_to_name[id2],
                    "pair_type": f"{min(t1, t2)}-{max(t1, t2)}",
                    "evidence": best_ev,
                    "gold_class": gold_pol,
                    "gold_int": y,
                })

        return feat_dicts, left_list, mid_list, right_list, zoned_list, np.array(labels, dtype=np.int32), meta_list

    def predict_untrained(
        self, docs: List[Dict[str, Any]], dataset_mode: str = "biored", seed: int = 42
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """Evaluates the untrained V3 architecture using a fixed random projection matrix W (row-order invariant)."""
        feat_dicts, _, _, _, _, _, meta_list = self._extract_candidates_from_docs(docs, dataset_mode=dataset_mode)
        n = len(feat_dicts)
        if n == 0:
            return np.zeros((0, 4), dtype=np.float64), meta_list
        sorted_keys = sorted(feat_dicts[0].keys())
        rng = np.random.RandomState(seed)
        W = rng.normal(0.0, 0.05, size=(len(sorted_keys), 4))
        bias = np.array([1.2, -0.8, -0.9, -0.5])
        probs = np.zeros((n, 4), dtype=np.float64)
        for i, f in enumerate(feat_dicts):
            vals = np.array([f.get(k, 0.0) for k in sorted_keys], dtype=np.float64)
            logits = np.dot(vals, W) + bias
            logits -= np.max(logits)
            exp_l = np.exp(logits)
            probs[i] = exp_l / np.sum(exp_l)
        return probs, meta_list

    def fit_and_calibrate(
        self,
        train_docs: List[Dict[str, Any]],
        dev_docs: List[Dict[str, Any]],
        dataset_mode: str = "biored",
    ) -> Dict[str, Any]:
        f_tr, l_tr, m_tr, r_tr, z_tr, y_tr, _ = self._extract_candidates_from_docs(train_docs, dataset_mode=dataset_mode)

        all_zones_tr = [f"{a} {b} {c}".strip() for a, b, c in zip(l_tr, m_tr, r_tr)]
        self.pooler.fit(all_zones_tr)
        X_pooled_tr = self.pooler.transform_zones(l_tr, m_tr, r_tr)
        X_struct_tr = self.dict_vec.fit_transform(f_tr)

        X_dense_raw_tr = np.hstack([X_struct_tr, X_pooled_tr])
        X_dense_tr = self.dense_scaler.fit_transform(X_dense_raw_tr)
        X_tfidf_tr = self.zoned_tfidf.fit_transform(z_tr)
        X_full_tr = hstack([csr_matrix(X_dense_tr), X_tfidf_tr])

        self.mlp_clf.fit(X_dense_tr, y_tr)
        self.softmax_linear_clf.fit(X_full_tr, y_tr)
        self.rf_clf.fit(X_dense_raw_tr, y_tr)
        self.classes_ = list(self.softmax_linear_clf.classes_)
        self.is_fitted = True

        f_dev, l_dev, m_dev, r_dev, z_dev, y_dev, _ = self._extract_candidates_from_docs(dev_docs, dataset_mode=dataset_mode)
        X_pooled_dev = self.pooler.transform_zones(l_dev, m_dev, r_dev)
        X_struct_dev = self.dict_vec.transform(f_dev)
        X_dense_raw_dev = np.hstack([X_struct_dev, X_pooled_dev])
        X_dense_dev = self.dense_scaler.transform(X_dense_raw_dev)
        X_tfidf_dev = self.zoned_tfidf.transform(z_dev)
        X_full_dev = hstack([csr_matrix(X_dense_dev), X_tfidf_dev])

        p_mlp_dev = self._align_proba_4class(self.mlp_clf.predict_proba(X_dense_dev), getattr(self.mlp_clf, "classes_", self.classes_))
        p_lin_dev = self._align_proba_4class(self.softmax_linear_clf.predict_proba(X_full_dev), getattr(self.softmax_linear_clf, "classes_", self.classes_))
        p_rf_dev = self._align_proba_4class(self.rf_clf.predict_proba(X_dense_raw_dev), getattr(self.rf_clf, "classes_", self.classes_))

        # Simplex candidate grid selected strictly on official Dev split (zero Test exposure)
        weight_grid = [
            (0.00, 0.35, 0.65),
            (0.00, 0.50, 0.50),
            (0.10, 0.40, 0.50),
            (0.15, 0.45, 0.40),
            (0.20, 0.50, 0.30),
            (0.25, 0.50, 0.25),
            (0.30, 0.45, 0.25),
        ]

        best_f1 = -1.0
        best_w = (0.15, 0.45, 0.40)
        best_th = 0.50
        best_p, best_r = 0.0, 0.0

        y_bin_dev = (y_dev > 0).astype(np.int32)
        for w_mlp, w_lin, w_rf in weight_grid:
            p_blend = w_mlp * p_mlp_dev + w_lin * p_lin_dev + w_rf * p_rf_dev
            p_rel_dev = 1.0 - p_blend[:, 0]
            for th in np.linspace(0.35, 0.75, 41):
                pred_bin = (p_rel_dev >= th).astype(np.int32)
                tp = int(np.sum((pred_bin == 1) & (y_bin_dev == 1)))
                fp = int(np.sum((pred_bin == 1) & (y_bin_dev == 0)))
                fn = int(np.sum((pred_bin == 0) & (y_bin_dev == 1)))
                p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
                if f1 > best_f1:
                    best_f1 = f1
                    best_w = (w_mlp, w_lin, w_rf)
                    best_th = float(th)
                    best_p, best_r = p, r

        self.ensemble_weights = best_w
        self.relation_threshold = round(best_th, 4)

        w_mlp, w_lin, w_rf = self.ensemble_weights
        p_blend = w_mlp * p_mlp_dev + w_lin * p_lin_dev + w_rf * p_rf_dev
        p_rel_dev = 1.0 - p_blend[:, 0]
        hr_th = 0.40
        for th in np.linspace(0.25, 0.60, 36):
            pred_bin = (p_rel_dev >= th).astype(np.int32)
            tp = int(np.sum((pred_bin == 1) & (y_bin_dev == 1)))
            fn = int(np.sum((pred_bin == 0) & (y_bin_dev == 1)))
            r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            if r >= 0.70:
                hr_th = float(th)
        self.high_recall_threshold = round(hr_th, 4)

        return {
            "train_candidate_pairs": len(y_tr),
            "dev_candidate_pairs": len(y_dev),
            "structural_features": X_struct_tr.shape[1],
            "zoned_pooled_embedding_dim": X_pooled_tr.shape[1],
            "zoned_tfidf_features": X_tfidf_tr.shape[1],
            "ensemble_weights_mlp_lin_rf": self.ensemble_weights,
            "optimal_threshold_on_dev": self.relation_threshold,
            "high_recall_threshold_on_dev": self.high_recall_threshold,
            "dev_binary_precision": round(best_p, 4),
            "dev_binary_recall": round(best_r, 4),
            "dev_binary_f1": round(best_f1, 4),
        }

    @staticmethod
    def _align_proba_4class(proba: np.ndarray, classes: Any) -> np.ndarray:
        if proba.shape[1] == 4 and list(classes) == [0, 1, 2, 3]:
            return proba
        out = np.zeros((proba.shape[0], 4), dtype=np.float64)
        for col_idx, cls_val in enumerate(classes):
            c_int = int(cls_val)
            if 0 <= c_int < 4:
                out[:, c_int] = proba[:, col_idx]
        return out

    def predict_multiclass(
        self, docs: List[Dict[str, Any]], dataset_mode: str = "biored"
    ) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        if not self.is_fitted:
            raise RuntimeError("V3 Engine must be fitted before calling predict_multiclass.")

        f_d, l_d, m_d, r_d, z_d, _, meta_list = self._extract_candidates_from_docs(docs, dataset_mode=dataset_mode)
        if not f_d:
            return np.zeros((0, 4), dtype=np.float64), np.zeros((0, 4), dtype=np.float64), []
        X_pooled = self.pooler.transform_zones(l_d, m_d, r_d)
        X_struct = self.dict_vec.transform(f_d)
        X_dense_raw = np.hstack([X_struct, X_pooled])
        X_dense = self.dense_scaler.transform(X_dense_raw)
        X_tfidf = self.zoned_tfidf.transform(z_d)
        X_full = hstack([csr_matrix(X_dense), X_tfidf])

        p_mlp = self._align_proba_4class(self.mlp_clf.predict_proba(X_dense), getattr(self.mlp_clf, "classes_", self.classes_))
        p_lin = self._align_proba_4class(self.softmax_linear_clf.predict_proba(X_full), getattr(self.softmax_linear_clf, "classes_", self.classes_))
        p_rf = self._align_proba_4class(self.rf_clf.predict_proba(X_dense_raw), getattr(self.rf_clf, "classes_", self.classes_))

        w_mlp, w_lin, w_rf = self.ensemble_weights
        p_ensemble = w_mlp * p_mlp + w_lin * p_lin + w_rf * p_rf

        for idx, m in enumerate(meta_list):
            p_none, p_pos, p_neg, p_assoc = [float(x) for x in p_ensemble[idx]]
            p_rel = max(1e-6, p_pos + p_neg + p_assoc)
            true_soft_sign = [
                round(p_pos / p_rel, 4),
                round(p_neg / p_rel, 4),
                round(p_assoc / p_rel, 4),
            ]
            active_cls_idx = int(np.argmax(p_ensemble[idx, 1:])) + 1
            m["prob_has_relation"] = round(1.0 - p_none, 4)
            m["prob_positive"] = round(p_pos, 4)
            m["prob_negative"] = round(p_neg, 4)
            m["prob_association"] = round(p_assoc, 4)
            m["true_softmax_sign"] = true_soft_sign
            m["pred_active_class"] = INT_TO_CLASS[active_cls_idx]

        return p_ensemble, p_mlp, meta_list

    def save_checkpoint(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)


_GLOBAL_V3_ENGINE: Optional[ZonedPoolingMultiClassBioREEngineV3] = None


class _V3Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module in ("bio_re_engine_v3", "lbd.nlp.bio_re_engine_v3", "lbd.nlp.classifier", "biomed_nlp.classifier"):
            return globals()[name]
        return super().find_class(module, name)


def load_v3_engine(checkpoint_path: Optional[Path] = None) -> ZonedPoolingMultiClassBioREEngineV3:
    global _GLOBAL_V3_ENGINE
    if checkpoint_path is None:
        checkpoint_path = Path(__file__).parent / "checkpoints" / "model_v3_biored_multiclass.pkl"
    checkpoint_path = Path(checkpoint_path).resolve()
    if _GLOBAL_V3_ENGINE is not None and getattr(_GLOBAL_V3_ENGINE, "_ckpt_path", None) == str(checkpoint_path):
        return _GLOBAL_V3_ENGINE
    with open(checkpoint_path, "rb") as f:
        try:
            engine = pickle.load(f)
        except (ModuleNotFoundError, AttributeError):
            f.seek(0)
            engine = _V3Unpickler(f).load()
    engine._ckpt_path = str(checkpoint_path)
    _GLOBAL_V3_ENGINE = engine
    return engine


# Canonical names
BiomedicalRelationClassifier = ZonedPoolingMultiClassBioREEngineV3
load_relation_classifier = load_v3_engine
