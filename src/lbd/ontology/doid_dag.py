"""
Directed DOID Ontology DAG Resolver for Hetionet v1.0 (137 Diseases).
--------------------------------------------------------------------
Parses local `doid.obo` deterministically:
- Excludes obsolete terms (`is_obsolete: true`) from the active DAG.
- Resolves `alt_id` and unambiguous `replaced_by` pointers to canonical active DOIDs.
- Precomputes upward ancestor maps for all 137 Hetionet v1.0 diseases in `__init__`.
- Resolves any DOID to Hetionet v1.0 disease targets with LCA decay (`decay_base ** tree_dist`)
  while strictly blocking high-level stop-roots to prevent cross-branch semantic leakage.
"""
from __future__ import annotations

import csv
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple


DEFAULT_DOID_OBO_PATH = (
    Path(__file__).resolve().parents[3] / "benchmarks" / "repodb" / "raw" / "doid.obo"
)
DEFAULT_HETIONET_NODES_PATH = (
    Path(__file__).resolve().parents[3] / "data" / "raw" / "hetionet-v1.0-nodes.tsv"
)

# High-level generic ontology roots that MUST NEVER serve as a valid LCA or match target.
STOP_ROOTS: frozenset[str] = frozenset({
    "DOID:4",        # disease
    "DOID:7",        # disease of anatomical entity
    "DOID:0060118",  # thoracic disease
    "DOID:14566",    # disease of cellular proliferation
    "DOID:1287",     # cardiovascular system disease
})

_MATCH_TYPE_RANK: Dict[str, int] = {
    "exact": 0,
    "ancestor": 1,
    "descendant": 2,
    "lca": 3,
}


def normalize_doid(doid: str) -> str:
    """Normalizes a DOID string (strips `Disease::` prefix and standardizes `DOID:` case)."""
    s = str(doid or "").strip()
    if not s:
        return ""
    if s.startswith("Disease::"):
        s = s[len("Disease::"):].strip()
    if s.lower().startswith("doid:"):
        s = "DOID:" + s.split(":", 1)[1].strip()
    return s


def load_hetionet_diseases(nodes_path: Optional[Path] = None) -> List[str]:
    """Loads the 137 canonical Hetionet v1.0 disease DOIDs from `hetionet-v1.0-nodes.tsv`."""
    path = Path(nodes_path) if nodes_path else DEFAULT_HETIONET_NODES_PATH
    if not path.exists():
        raise FileNotFoundError(f"Missing Hetionet nodes file: {path}")
    diseases: List[str] = []
    with path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if row.get("kind") == "Disease":
                doid = normalize_doid(row["id"])
                if doid:
                    diseases.append(doid)
    return sorted(set(diseases))


def parse_doid_obo(obo_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Single-pass deterministic parser for `doid.obo`.

    Returns a dictionary containing:
    - active_terms: set[str]
    - obsolete_terms: set[str]
    - names: dict[str, str]
    - parents: dict[str, list[str]] (active term -> sorted unique `is_a` parents)
    - alt_to_canonical: dict[str, str]
    - replaced_by: dict[str, str] (only unambiguous 1:1 replacements for obsolete terms)
    - mesh_to_doids: dict[str, set[str]] (active terms only)
    - umls_to_doids: dict[str, set[str]] (active terms only)
    - doid_to_mesh: dict[str, set[str]]
    - doid_to_umls: dict[str, set[str]]
    """
    path = Path(obo_path) if obo_path else DEFAULT_DOID_OBO_PATH
    if not path.exists():
        raise FileNotFoundError(f"Missing DOID OBO file: {path}")

    active_terms: Set[str] = set()
    obsolete_terms: Set[str] = set()
    names: Dict[str, str] = {}
    parents_raw: Dict[str, Set[str]] = defaultdict(set)
    alt_to_canonical: Dict[str, str] = {}
    replaced_by: Dict[str, str] = {}
    mesh_to_doids: Dict[str, Set[str]] = defaultdict(set)
    umls_to_doids: Dict[str, Set[str]] = defaultdict(set)
    doid_to_mesh: Dict[str, Set[str]] = defaultdict(set)
    doid_to_umls: Dict[str, Set[str]] = defaultdict(set)

    cur: Optional[Dict[str, Any]] = None

    def _flush(stanza: Optional[Dict[str, Any]]) -> None:
        if not stanza:
            return
        tid = normalize_doid(stanza.get("id", ""))
        if not tid.startswith("DOID:"):
            return
        if stanza.get("is_obsolete"):
            obsolete_terms.add(tid)
            reps = [normalize_doid(r) for r in stanza.get("replaced_by", []) if normalize_doid(r).startswith("DOID:")]
            # Only accept unambiguous single replacement to avoid arbitrary split mapping
            unique_reps = sorted(set(reps))
            if len(unique_reps) == 1:
                replaced_by[tid] = unique_reps[0]
            return

        active_terms.add(tid)
        if stanza.get("name"):
            names[tid] = stanza["name"]
        for p in stanza.get("is_a", []):
            p_norm = normalize_doid(p)
            if p_norm.startswith("DOID:") and p_norm != tid:
                parents_raw[tid].add(p_norm)
        for alt in stanza.get("alt_id", []):
            alt_norm = normalize_doid(alt)
            if alt_norm.startswith("DOID:") and alt_norm != tid:
                alt_to_canonical[alt_norm] = tid
        for x in stanza.get("xref", []):
            if x.startswith("MESH:"):
                mesh_id = x.split(":", 1)[1].strip()
                if mesh_id:
                    mesh_to_doids[mesh_id].add(tid)
                    doid_to_mesh[tid].add(mesh_id)
            elif x.startswith("UMLS_CUI:"):
                cui = x.split(":", 1)[1].strip()
                if cui:
                    umls_to_doids[cui].add(tid)
                    doid_to_umls[tid].add(cui)

    with path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            if line == "[Term]":
                _flush(cur)
                cur = {
                    "id": "",
                    "name": "",
                    "is_obsolete": False,
                    "is_a": [],
                    "alt_id": [],
                    "replaced_by": [],
                    "xref": [],
                }
            elif line.startswith("[") and line.endswith("]"):
                _flush(cur)
                cur = None
            elif cur is not None:
                if line.startswith("id: "):
                    cur["id"] = line[4:].strip()
                elif line.startswith("name: "):
                    cur["name"] = line[6:].strip()
                elif line.startswith("is_obsolete: true"):
                    cur["is_obsolete"] = True
                elif line.startswith("is_a: "):
                    cur["is_a"].append(line[6:].split("!", 1)[0].strip())
                elif line.startswith("alt_id: "):
                    cur["alt_id"].append(line[8:].split("!", 1)[0].strip())
                elif line.startswith("replaced_by: "):
                    cur["replaced_by"].append(line[13:].split("!", 1)[0].strip())
                elif line.startswith("xref: "):
                    cur["xref"].append(line[6:].split()[0].strip())

    _flush(cur)

    # Filter parent pointers to active terms only
    parents: Dict[str, List[str]] = {
        tid: sorted(p for p in ps if p in active_terms)
        for tid, ps in parents_raw.items()
        if tid in active_terms
    }

    return {
        "active_terms": active_terms,
        "obsolete_terms": obsolete_terms,
        "names": names,
        "parents": parents,
        "alt_to_canonical": alt_to_canonical,
        "replaced_by": replaced_by,
        "mesh_to_doids": dict(mesh_to_doids),
        "umls_to_doids": dict(umls_to_doids),
        "doid_to_mesh": dict(doid_to_mesh),
        "doid_to_umls": dict(doid_to_umls),
    }


class DirectedDOIDResolver:
    """
    Deterministic DOID DAG resolver mapping any DOID to Hetionet v1.0's 137 diseases
    with shortest-path LCA decay and strict stop-root isolation.
    """

    def __init__(
        self,
        obo_path: Optional[Path] = None,
        hetionet_nodes_path: Optional[Path] = None,
        hetionet_diseases: Optional[Iterable[str]] = None,
        stop_roots: Optional[Iterable[str]] = None,
        parsed_obo: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.obo_path = Path(obo_path) if obo_path else DEFAULT_DOID_OBO_PATH
        self.hetionet_nodes_path = (
            Path(hetionet_nodes_path) if hetionet_nodes_path else DEFAULT_HETIONET_NODES_PATH
        )
        self.stop_roots: frozenset[str] = (
            frozenset(normalize_doid(r) for r in stop_roots)
            if stop_roots is not None
            else STOP_ROOTS
        )

        obo = parsed_obo if parsed_obo is not None else parse_doid_obo(self.obo_path)
        self._parsed_obo = obo
        self.active_terms: Set[str] = obo["active_terms"]
        self.obsolete_terms: Set[str] = obo["obsolete_terms"]
        self.names: Dict[str, str] = obo["names"]
        self.parents: Dict[str, List[str]] = obo["parents"]
        self.alt_to_canonical: Dict[str, str] = obo["alt_to_canonical"]
        self.replaced_by: Dict[str, str] = obo["replaced_by"]

        if hetionet_diseases is not None:
            raw_diseases = sorted({normalize_doid(d) for d in hetionet_diseases if normalize_doid(d)})
        else:
            raw_diseases = load_hetionet_diseases(self.hetionet_nodes_path)

        self.hetionet_diseases: Tuple[str, ...] = tuple(raw_diseases)
        self.hetionet_to_canonical: Dict[str, str] = {}
        self.canonical_to_hetionet: Dict[str, str] = {}
        self._hetio_ancestors: Dict[str, Dict[str, int]] = {}

        # Precompute canonical IDs and upward ancestor distance maps for all 137 Hetionet diseases
        for h_doid in self.hetionet_diseases:
            h_canon = self.canonicalize(h_doid)
            if h_canon is None:
                continue
            self.hetionet_to_canonical[h_doid] = h_canon
            # Keep deterministic first mapping if multiple Hetionet IDs ever share a canonical term
            if h_canon not in self.canonical_to_hetionet:
                self.canonical_to_hetionet[h_canon] = h_doid
            self._hetio_ancestors[h_doid] = self.get_ancestors(h_canon, include_stop_roots=False)

    def canonicalize(self, doid: str) -> Optional[str]:
        """
        Resolves `alt_id` and `replaced_by` chains to an active canonical DOID.
        Returns `None` if unknown or obsolete without an active replacement.
        """
        cur = normalize_doid(doid)
        if not cur:
            return None
        visited: Set[str] = set()
        for _ in range(16):
            if cur in visited:
                return None
            visited.add(cur)
            if cur in self.active_terms:
                return cur
            if cur in self.alt_to_canonical:
                cur = self.alt_to_canonical[cur]
                continue
            if cur in self.replaced_by:
                cur = self.replaced_by[cur]
                continue
            return None
        return None

    def get_ancestors(self, canon_doid: str, include_stop_roots: bool = False) -> Dict[str, int]:
        """
        BFS upward along `is_a` edges from `canon_doid`.
        Returns `{ancestor_doid: min_hop_distance}` including `{canon_doid: 0}`.
        When `include_stop_roots=False`, traversal stops at stop-roots and excludes them.
        """
        if not canon_doid or canon_doid not in self.active_terms:
            return {}
        if not include_stop_roots and canon_doid in self.stop_roots:
            return {}

        dist_map: Dict[str, int] = {canon_doid: 0}
        queue: deque[Tuple[str, int]] = deque([(canon_doid, 0)])

        while queue:
            node, d = queue.popleft()
            for parent in self.parents.get(node, ()):
                if not include_stop_roots and parent in self.stop_roots:
                    continue
                if parent not in dist_map:
                    dist_map[parent] = d + 1
                    queue.append((parent, d + 1))
        return dist_map

    def resolve(
        self,
        doid: str,
        max_dist: int = 4,
        decay_base: float = 0.90,
    ) -> List[Tuple[str, float, int, str, str]]:
        """
        Resolves an input DOID to Hetionet v1.0 disease targets.

        Returns a deterministic list of tuples:
            `[(hetio_doid, lca_decay, tree_dist, lca_id, match_type)]`
        where:
        - `hetio_doid`: canonical Hetionet v1.0 DOID (e.g. `"DOID:9352"`)
        - `lca_decay`: `decay_base ** tree_dist` (`1.0` for exact match)
        - `tree_dist`: shortest DAG distance via valid non-stop-root LCA
        - `lca_id`: DOID of the Lowest Common Ancestor (never in `STOP_ROOTS`)
        - `match_type`: `"exact" | "ancestor" | "descendant" | "lca"`
        """
        if max_dist < 0:
            return []
        canon = self.canonicalize(doid)
        if canon is None or canon in self.stop_roots:
            return []

        # 1. Exact match to a Hetionet v1.0 disease
        if canon in self.canonical_to_hetionet:
            hetio_doid = self.canonical_to_hetionet[canon]
            return [(hetio_doid, 1.0, 0, canon, "exact")]

        if max_dist == 0:
            return []

        q_anc = self.get_ancestors(canon, include_stop_roots=False)
        if not q_anc:
            return []

        results: List[Tuple[str, float, int, str, str]] = []
        for h_doid in self.hetionet_diseases:
            h_canon = self.hetionet_to_canonical.get(h_doid)
            if not h_canon or h_canon in self.stop_roots:
                continue
            h_anc = self._hetio_ancestors.get(h_doid, {})
            if not h_anc:
                continue

            # Case A: Hetionet disease is an ancestor of query (query is_a* h_doid)
            if h_canon in q_anc:
                dist = q_anc[h_canon]
                if 1 <= dist <= max_dist:
                    decay = round(float(decay_base ** dist), 6)
                    results.append((h_doid, decay, dist, h_canon, "ancestor"))
                continue

            # Case B: Query is an ancestor of Hetionet disease (h_doid is_a* query)
            if canon in h_anc:
                dist = h_anc[canon]
                if 1 <= dist <= max_dist:
                    decay = round(float(decay_base ** dist), 6)
                    results.append((h_doid, decay, dist, canon, "descendant"))
                continue

            # Case C: Shared non-stop-root LCA
            common = (q_anc.keys() & h_anc.keys()) - self.stop_roots
            if not common:
                continue
            best_lca = min(common, key=lambda a: (q_anc[a] + h_anc[a], a))
            dist = q_anc[best_lca] + h_anc[best_lca]
            if dist <= max_dist:
                decay = round(float(decay_base ** dist), 6)
                results.append((h_doid, decay, dist, best_lca, "lca"))

        results.sort(key=lambda item: (item[2], _MATCH_TYPE_RANK.get(item[4], 99), item[0]))
        return results

    def coverage_stats(self) -> Dict[str, Any]:
        """Returns diagnostic statistics for the loaded DOID DAG and Hetionet 137 diseases."""
        return {
            "active_doid_terms": len(self.active_terms),
            "obsolete_doid_terms": len(self.obsolete_terms),
            "alt_id_mappings": len(self.alt_to_canonical),
            "replaced_by_mappings": len(self.replaced_by),
            "hetionet_diseases_total": len(self.hetionet_diseases),
            "hetionet_diseases_indexed": len(self._hetio_ancestors),
            "stop_roots": sorted(self.stop_roots),
        }
