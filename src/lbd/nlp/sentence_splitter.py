"""
Biomedical Sentence & Syntactic Clause Splitter (Exact-Offset Preserving).

Grounded in:
- Griffis et al. (BMC Bioinformatics 2016): Empirical evaluation of biomedical sentence boundary detection.
- Harkema et al. (J. Biomed. Inform. 2009, ConText) & Vincze et al. (BMC Bioinformatics 2008, BioScope):
  Clause boundary segmentation and scope termination tokens.

Guarantees 100% character-offset invariance (`len(masked_text) == len(text)`), eliminating
offset drift when grounding character-level PubTator3 / BioRED / BC5CDR spans.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Tuple

_SCISPACY_NLP = None
try:
    import spacy
    try:
        _SCISPACY_NLP = spacy.load("en_core_sci_sm")
    except Exception:
        try:
            _SCISPACY_NLP = spacy.blank("en")
            _SCISPACY_NLP.add_pipe("sentencizer")
        except Exception:
            _SCISPACY_NLP = None
except Exception:
    _SCISPACY_NLP = None

# Biomedical abbreviations, statistical expressions, and numeric literals whose internal
# periods must NEVER trigger sentence boundary splits.
PROTECTED_ABBRS = [
    r"\b(?:vs|i\.e|e\.g|et\s+al|fig|figs|table|eq|dr|prof|inc|ltd|no|vol|pp|ca|approx|wt|ref|refs|min|hr|hrs|sec|wk|wks|yr|yrs|st|ave|dept|univ|jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec|cf)\.",
    r"\b[pP]\s*[<=>≤≥]\s*0?\.\d+",
    r"\b\d+\.\d+\b",
    r"\b[A-Z]\.\s*[A-Z]\.",
    r"\b(?:[A-Za-z]\.){2,}",
]
PROTECTED_PATTERN = re.compile("|".join(PROTECTED_ABBRS), re.IGNORECASE)

# Sentence boundary regex: sentence-ending punctuation followed by whitespace and a capital letter, digit, or opening bracket/quote
SENT_BOUNDARY_RX = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\[\"])")

# Clause boundary delimiters (BioScope / ConText discourse coordination & subordination markers)
CLAUSE_DELIMITER_RX = re.compile(
    r"(?:;|\s*,\s*(?:whereas|while|however|but|although|though|yet|except|without|nor)\b|\b(?:whereas|while|however|although)\b)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class SyntacticClauseSpan:
    clause_idx: int
    char_start: int
    char_end: int
    text: str
    connector: str = ""


def _mask_preserving_length(text: str) -> str:
    """Replaces sentence-ending punctuation inside protected biomedical abbreviations/numbers
    with same-length '_' characters so character offsets remain 100% identical to `text`.
    """
    chars = list(text)
    for m in PROTECTED_PATTERN.finditer(text):
        for idx in range(m.start(), m.end()):
            if chars[idx] in ".!?":
                chars[idx] = "_"
    # Also protect parenthetical citations/statistics up to 60 chars that contain periods
    for m in re.finditer(r"\([^()\n]{1,60}\)", text):
        inner = m.group(0)
        if re.search(r"(?:[pP]\s*[<=>]|et\s+al\.|vs\.|fig\.|\d+\.\d+)", inner, re.I):
            for idx in range(m.start(), m.end()):
                if chars[idx] in ".!?":
                    chars[idx] = "_"
    return "".join(chars)


def split_biomedical_sentences(text: str, start_offset: int = 0) -> List[Tuple[int, int, str]]:
    """Splits biomedical text into `(char_start, char_end, sentence_text)` tuples with
    strictly exact character offsets into `text` (offset by `start_offset`).
    """
    if not text or not text.strip():
        return []

    if _SCISPACY_NLP is not None:
        try:
            doc = _SCISPACY_NLP(text)
            spans: List[Tuple[int, int, str]] = []
            for s in doc.sents:
                raw_sub = text[s.start_char:s.end_char]
                l_strip = len(raw_sub) - len(raw_sub.lstrip())
                r_strip = len(raw_sub) - len(raw_sub.rstrip())
                c_start = s.start_char + l_strip
                c_end = s.end_char - r_strip
                if c_end > c_start:
                    spans.append((start_offset + c_start, start_offset + c_end, text[c_start:c_end]))
            if spans:
                return spans
        except Exception:
            pass

    masked_text = _mask_preserving_length(text)
    results: List[Tuple[int, int, str]] = []
    last = 0

    for m in SENT_BOUNDARY_RX.finditer(masked_text):
        raw_seg = text[last:m.start()]
        if raw_seg.strip():
            l_strip = len(raw_seg) - len(raw_seg.lstrip())
            r_strip = len(raw_seg) - len(raw_seg.rstrip())
            c_start = last + l_strip
            c_end = m.start() - r_strip
            if c_end > c_start:
                results.append((start_offset + c_start, start_offset + c_end, text[c_start:c_end]))
        last = m.end()

    tail = text[last:]
    if tail.strip():
        l_strip = len(tail) - len(tail.lstrip())
        r_strip = len(tail) - len(tail.rstrip())
        c_start = last + l_strip
        c_end = len(text) - r_strip
        if c_end > c_start:
            results.append((start_offset + c_start, start_offset + c_end, text[c_start:c_end]))

    return results


def split_sentence_clauses(sentence_text: str, sent_start_offset: int = 0) -> List[SyntacticClauseSpan]:
    """Segments a biomedical sentence into syntactic clauses using BioScope/ConText
    coordination and subordination boundaries, preserving exact offsets.
    """
    if not sentence_text:
        return []

    clauses: List[SyntacticClauseSpan] = []
    last = 0
    last_conn = ""
    c_idx = 0

    for m in CLAUSE_DELIMITER_RX.finditer(sentence_text):
        seg = sentence_text[last:m.start()]
        if seg.strip():
            l_strip = len(seg) - len(seg.lstrip())
            r_strip = len(seg) - len(seg.rstrip())
            c_start = last + l_strip
            c_end = m.start() - r_strip
            clauses.append(
                SyntacticClauseSpan(
                    clause_idx=c_idx,
                    char_start=sent_start_offset + c_start,
                    char_end=sent_start_offset + c_end,
                    text=sentence_text[c_start:c_end],
                    connector=last_conn,
                )
            )
            c_idx += 1
        last = m.start()
        last_conn = m.group(0).strip(" ,;")

    tail = sentence_text[last:]
    if tail.strip():
        l_strip = len(tail) - len(tail.lstrip())
        r_strip = len(tail) - len(tail.rstrip())
        c_start = last + l_strip
        c_end = len(sentence_text) - r_strip
        clauses.append(
            SyntacticClauseSpan(
                clause_idx=c_idx,
                char_start=sent_start_offset + c_start,
                char_end=sent_start_offset + c_end,
                text=sentence_text[c_start:c_end],
                connector=last_conn,
            )
        )

    return clauses
