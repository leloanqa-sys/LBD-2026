"""
Biomedical Sentence Splitter using ScispaCy (if installed) with Rule-based Abbreviation-aware Fallback.
Prevents erroneous splitting at 'p < 0.05', 'vs.', 'et al.', 'i.e.', 'e.g.', 'wt.'.
"""
import re
from typing import List, Tuple

# Try loading spacy / scispacy
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

# Comprehensive biomedical abbreviation protected list
PROTECTED_ABBRS = [
    r"vs\.", r"i\.e\.", r"e\.g\.", r"et\s+al\.", r"fig\.", r"ref\.", r"no\.",
    r"wt\.", r"approx\.", r"ca\.", r"vol\.", r"min\.", r"hr\.", r"sec\.",
    r"p\s*<\s*0\.\d+", r"p\s*=\s*0\.\d+", r"p\s*>\s*0\.\d+",
    r"\d+\.\d+",  # Floating numbers like 10.5 mg
]
PROTECTED_PATTERN = re.compile("|".join(PROTECTED_ABBRS), re.IGNORECASE)

def split_biomedical_sentences(text: str, start_offset: int = 0) -> List[Tuple[int, int, str]]:
    """Splits biomedical text into sentences preserving spans and avoiding abbreviation pitfalls."""
    if not text or not text.strip():
        return []

    # 1. Preferred Path: SciSpacy Engine
    if _SCISPACY_NLP is not None:
        try:
            doc = _SCISPACY_NLP(text)
            spans = []
            for s in doc.sents:
                s_str = s.text.strip()
                if s_str:
                    spans.append((start_offset + s.start_char, start_offset + s.end_char, s_str))
            if spans:
                return spans
        except Exception:
            pass

    # 2. Robust Rule-based Fallback: Mask protected abbreviations with placeholders
    masked_text = text
    masks = []
    for m in PROTECTED_PATTERN.finditer(text):
        ph = f"__PROTECTED_{len(masks)}__"
        masks.append((ph, m.group(0), m.start(), m.end()))

    # Apply masks in reverse order of appearance
    for ph, orig, start, end in reversed(masks):
        masked_text = masked_text[:start] + ph + masked_text[end:]

    # Sentence boundary regex: punctuation followed by space and capital letter or digit
    sent_rx = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\[\"])")
    raw_spans = []
    last = 0
    for m in sent_rx.finditer(masked_text):
        seg = masked_text[last:m.start()]
        if seg.strip():
            raw_spans.append((last, m.start(), seg.strip()))
        last = m.end()
    tail = masked_text[last:]
    if tail.strip():
        raw_spans.append((last, len(masked_text), tail.strip()))

    # Restore original text from masks
    results = []
    for seg_start, seg_end, seg_text in raw_spans:
        restored = seg_text
        for ph, orig, _, _ in masks:
            restored = restored.replace(ph, orig)
        results.append((start_offset + seg_start, start_offset + seg_end, restored.strip()))

    return results
