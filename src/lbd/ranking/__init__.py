"""Official Ranking Package for LBD Drug Repurposing.

Production Engine:
    - M2RepurposingRanker: Supervised DWPC graph ranker with degree debiasing,
      NTR signed graph flux, and Meet-in-the-Middle Micro-DFS.

Subpackages:
    - core: Pure mathematical graph topological extraction & ML pipeline (repurank).
    - legacy: Historical heuristic DFS path scoring modules (deprecated).
"""
from lbd.ranking.m2_ranker import M2RepurposingRanker
from lbd.ranking.rank_candidates import score_disease_candidate
from lbd.ranking import core

__all__ = ["M2RepurposingRanker", "score_disease_candidate", "core"]
