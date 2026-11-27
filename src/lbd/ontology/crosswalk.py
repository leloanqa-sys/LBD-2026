"""
Deterministic Crosswalk for Disease (MeSH / UMLS_CUI <-> DOID) and Compound (Hetionet 1,552).
---------------------------------------------------------------------------------------------
Strict anti-false-mapping policy:
- Automatically extracts `MeSH <-> DOID` and `UMLS_CUI <-> DOID` from local `doid.obo` (active terms only).
- Accepts `1:1` mappings unconditionally.
- For `1:N` mappings, ONLY accepts if there is a UNIQUE DOID belonging to Hetionet v1.0's 137 diseases;
  all other `1:N` mappings are strictly REJECTED (never guessed).
- Loads all 1,552 Hetionet v1.0 compounds from `hetionet-v1.0-nodes.tsv`.
- Provides a clean, optional hook to ingest additional crosswalk tables from `data/external/ontology/`
  when present, enforcing the same deterministic 1:1 policy with Zero Disk Mutation.
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from lbd.ontology.doid_dag import (
    DEFAULT_DOID_OBO_PATH,
    DEFAULT_HETIONET_NODES_PATH,
    DirectedDOIDResolver,
    load_hetionet_diseases,
    normalize_doid,
    parse_doid_obo,
)


DEFAULT_EXTERNAL_ONTOLOGY_DIR = (
    Path(__file__).resolve().parents[3] / "data" / "external" / "ontology"
)


def _normalize_mesh(mesh_id: str) -> str:
    s = str(mesh_id or "").strip()
    if s.upper().startswith("MESH:"):
        s = s.split(":", 1)[1].strip()
    return s.upper()


def _normalize_umls(cui: str) -> str:
    s = str(cui or "").strip()
    upper = s.upper()
    if upper.startswith("UMLS_CUI:"):
        s = s.split(":", 1)[1].strip()
    elif upper.startswith("UMLS:"):
        s = s.split(":", 1)[1].strip()
    return s.upper()


class DeterministicCrosswalk:
    """
    Deterministic MeSH/UMLS <-> DOID and Compound lookup table backed by local `doid.obo`
    and `hetionet-v1.0-nodes.tsv`, with optional external ontology directory hook.
    """

    def __init__(
        self,
        obo_path: Optional[Path] = None,
        hetionet_nodes_path: Optional[Path] = None,
        external_dir: Optional[Path] = None,
        resolver: Optional[DirectedDOIDResolver] = None,
    ) -> None:
        self.obo_path = Path(obo_path) if obo_path else DEFAULT_DOID_OBO_PATH
        self.hetionet_nodes_path = (
            Path(hetionet_nodes_path) if hetionet_nodes_path else DEFAULT_HETIONET_NODES_PATH
        )
        self.external_dir = (
            Path(external_dir) if external_dir is not None else DEFAULT_EXTERNAL_ONTOLOGY_DIR
        )

        if resolver is not None:
            self.resolver = resolver
            obo = resolver._parsed_obo
        else:
            obo = parse_doid_obo(self.obo_path)
            self.resolver = DirectedDOIDResolver(
                obo_path=self.obo_path,
                hetionet_nodes_path=self.hetionet_nodes_path,
                parsed_obo=obo,
            )

        self.hetionet_diseases: Set[str] = set(self.resolver.hetionet_diseases)
        self._hetionet_canon_set: Set[str] = set(self.resolver.canonical_to_hetionet.keys())

        # Build strict MeSH <-> DOID and UMLS_CUI <-> DOID maps
        self._mesh_to_doid: Dict[str, str] = {}
        self._mesh_mapping_mode: Dict[str, str] = {}
        self.rejected_mesh: Dict[str, List[str]] = {}
        self._doid_to_mesh: Dict[str, List[str]] = defaultdict(list)

        self._umls_to_doid: Dict[str, str] = {}
        self._umls_mapping_mode: Dict[str, str] = {}
        self.rejected_umls: Dict[str, List[str]] = {}
        self._doid_to_umls: Dict[str, List[str]] = defaultdict(list)

        self._build_xref_map(
            raw_map=obo["mesh_to_doids"],
            target_map=self._mesh_to_doid,
            mode_map=self._mesh_mapping_mode,
            rejected_map=self.rejected_mesh,
            reverse_map=self._doid_to_mesh,
            normalizer=_normalize_mesh,
        )
        self._build_xref_map(
            raw_map=obo["umls_to_doids"],
            target_map=self._umls_to_doid,
            mode_map=self._umls_mapping_mode,
            rejected_map=self.rejected_umls,
            reverse_map=self._doid_to_umls,
            normalizer=_normalize_umls,
        )

        # Load 1,552 Hetionet v1.0 compounds
        self.compounds: Dict[str, str] = {}  # Compound::DBxxxxx -> Name
        self._compound_lookup: Dict[str, str] = {}  # lowercased key -> Compound::DBxxxxx
        self._ambiguous_compound_keys: Set[str] = set()
        self._load_hetionet_compounds()

        # Optional external ontology hook (e.g., data/external/ontology/*.tsv|*.json)
        self.external_files_loaded: List[str] = []
        self._load_external_ontology_dir()

    def _to_hetionet_or_canonical(self, canon_doid: str) -> str:
        """Maps a canonical DOID back to its Hetionet v1.0 DOID if it is one of the 137 diseases."""
        return self.resolver.canonical_to_hetionet.get(canon_doid, canon_doid)

    def _build_xref_map(
        self,
        raw_map: Dict[str, Set[str]],
        target_map: Dict[str, str],
        mode_map: Dict[str, str],
        rejected_map: Dict[str, List[str]],
        reverse_map: Dict[str, List[str]],
        normalizer: Any,
    ) -> None:
        for raw_key, raw_doids in sorted(raw_map.items()):
            key = normalizer(raw_key)
            if not key:
                continue
            canon_doids = sorted({
                c for d in raw_doids
                for c in (self.resolver.canonicalize(d),)
                if c is not None
            })
            if not canon_doids:
                continue
            if len(canon_doids) == 1:
                chosen = self._to_hetionet_or_canonical(canon_doids[0])
                target_map[key] = chosen
                mode_map[key] = "1:1"
                reverse_map[chosen].append(key)
            else:
                het_hits = [c for c in canon_doids if c in self._hetionet_canon_set]
                if len(het_hits) == 1:
                    chosen = self._to_hetionet_or_canonical(het_hits[0])
                    target_map[key] = chosen
                    mode_map[key] = "1:N_hetionet_unique"
                    reverse_map[chosen].append(key)
                else:
                    rejected_map[key] = canon_doids

        for k in list(reverse_map.keys()):
            reverse_map[k] = sorted(set(reverse_map[k]))

    def _register_compound_alias(self, alias: str, compound_id: str) -> None:
        key = str(alias or "").strip().lower()
        if not key or key in self._ambiguous_compound_keys:
            return
        existing = self._compound_lookup.get(key)
        if existing is None:
            self._compound_lookup[key] = compound_id
        elif existing != compound_id:
            # Ambiguous alias across multiple compounds -> strictly reject
            del self._compound_lookup[key]
            self._ambiguous_compound_keys.add(key)

    def _load_hetionet_compounds(self) -> None:
        if not self.hetionet_nodes_path.exists():
            raise FileNotFoundError(f"Missing Hetionet nodes file: {self.hetionet_nodes_path}")
        with self.hetionet_nodes_path.open("r", encoding="utf-8") as f:
            for row in csv.DictReader(f, delimiter="\t"):
                if row.get("kind") != "Compound":
                    continue
                cid = row["id"].strip()
                cname = row["name"].strip()
                self.compounds[cid] = cname
                self._register_compound_alias(cid, cid)
                if cid.startswith("Compound::"):
                    self._register_compound_alias(cid.split("::", 1)[1], cid)
                if cname:
                    self._register_compound_alias(cname, cid)

    def _load_external_ontology_dir(self) -> None:
        """
        Hook to load optional supplemental crosswalk tables from `data/external/ontology/`
        if the directory exists. Supports TSV files with columns:
        - `(mesh_id, doid)` or `(umls_cui, doid)`: applied only when not already mapped & unambiguous.
        - `(alias, drugbank_id)` or `(external_id, compound_id)`: registered via strict 1:1 compound lookup.
        """
        if not self.external_dir.exists() or not self.external_dir.is_dir():
            return
        for path in sorted(self.external_dir.iterdir()):
            if not path.is_file():
                continue
            suffix = path.suffix.lower()
            if suffix == ".tsv":
                self._ingest_external_tsv(path)
            elif suffix == ".json":
                self._ingest_external_json(path)

    def _ingest_external_tsv(self, path: Path) -> None:
        with path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            if not reader.fieldnames:
                return
            fields = {c.strip().lower(): c for c in reader.fieldnames}
            rows = list(reader)

        loaded_any = False
        # Disease MeSH -> DOID supplemental
        if "mesh_id" in fields and "doid" in fields:
            grouped: Dict[str, Set[str]] = defaultdict(set)
            for r in rows:
                mk = _normalize_mesh(r.get(fields["mesh_id"], ""))
                dk = self.resolver.canonicalize(r.get(fields["doid"], ""))
                if mk and dk:
                    grouped[mk].add(dk)
            for mk, dset in sorted(grouped.items()):
                if mk in self._mesh_to_doid or mk in self.rejected_mesh:
                    continue
                if len(dset) == 1:
                    chosen = self._to_hetionet_or_canonical(next(iter(dset)))
                    self._mesh_to_doid[mk] = chosen
                    self._mesh_mapping_mode[mk] = "external_1:1"
                    self._doid_to_mesh[chosen] = sorted(set(self._doid_to_mesh[chosen]) | {mk})
                    loaded_any = True
                else:
                    het_hits = [c for c in dset if c in self._hetionet_canon_set]
                    if len(het_hits) == 1:
                        chosen = self._to_hetionet_or_canonical(het_hits[0])
                        self._mesh_to_doid[mk] = chosen
                        self._mesh_mapping_mode[mk] = "external_1:N_hetionet_unique"
                        self._doid_to_mesh[chosen] = sorted(set(self._doid_to_mesh[chosen]) | {mk})
                        loaded_any = True

        # Disease UMLS -> DOID supplemental
        if "umls_cui" in fields and "doid" in fields:
            grouped_u: Dict[str, Set[str]] = defaultdict(set)
            for r in rows:
                uk = _normalize_umls(r.get(fields["umls_cui"], ""))
                dk = self.resolver.canonicalize(r.get(fields["doid"], ""))
                if uk and dk:
                    grouped_u[uk].add(dk)
            for uk, dset in sorted(grouped_u.items()):
                if uk in self._umls_to_doid or uk in self.rejected_umls:
                    continue
                if len(dset) == 1:
                    chosen = self._to_hetionet_or_canonical(next(iter(dset)))
                    self._umls_to_doid[uk] = chosen
                    self._umls_mapping_mode[uk] = "external_1:1"
                    self._doid_to_umls[chosen] = sorted(set(self._doid_to_umls[chosen]) | {uk})
                    loaded_any = True
                else:
                    het_hits = [c for c in dset if c in self._hetionet_canon_set]
                    if len(het_hits) == 1:
                        chosen = self._to_hetionet_or_canonical(het_hits[0])
                        self._umls_to_doid[uk] = chosen
                        self._umls_mapping_mode[uk] = "external_1:N_hetionet_unique"
                        self._doid_to_umls[chosen] = sorted(set(self._doid_to_umls[chosen]) | {uk})
                        loaded_any = True

        # Compound alias/external_id -> Hetionet Compound::DBxxxxx
        id_col = fields.get("compound_id") or fields.get("drugbank_id")
        alias_col = fields.get("alias") or fields.get("external_id") or fields.get("synonym")
        if id_col and alias_col:
            for r in rows:
                raw_cid = str(r.get(id_col, "")).strip()
                cid = raw_cid if raw_cid.startswith("Compound::") else f"Compound::{raw_cid}"
                if cid in self.compounds:
                    self._register_compound_alias(r.get(alias_col, ""), cid)
                    loaded_any = True

        if loaded_any:
            self.external_files_loaded.append(path.name)

    def _ingest_external_json(self, path: Path) -> None:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return
        if not isinstance(payload, dict):
            return
        loaded_any = False
        comp_map = payload.get("compound_aliases")
        if isinstance(comp_map, dict):
            for alias, raw_cid in comp_map.items():
                cid = raw_cid if str(raw_cid).startswith("Compound::") else f"Compound::{raw_cid}"
                if cid in self.compounds:
                    self._register_compound_alias(alias, cid)
                    loaded_any = True
        if loaded_any:
            self.external_files_loaded.append(path.name)

    # ------------------------------------------------------------------
    # Public deterministic lookup methods
    # ------------------------------------------------------------------
    def mesh_to_doid(self, mesh_id: str) -> Optional[str]:
        """Maps a MeSH descriptor ID to a single deterministic DOID, or `None` if unknown/rejected."""
        return self._mesh_to_doid.get(_normalize_mesh(mesh_id))

    def umls_to_doid(self, cui: str) -> Optional[str]:
        """Maps a UMLS CUI to a single deterministic DOID, or `None` if unknown/rejected."""
        return self._umls_to_doid.get(_normalize_umls(cui))

    def doid_to_mesh(self, doid: str) -> List[str]:
        """Returns deterministic accepted MeSH IDs associated with a DOID."""
        norm = normalize_doid(doid)
        if norm in self._doid_to_mesh:
            return list(self._doid_to_mesh[norm])
        canon = self.resolver.canonicalize(norm)
        if canon:
            mapped = self._to_hetionet_or_canonical(canon)
            return list(self._doid_to_mesh.get(mapped, []))
        return []

    def doid_to_umls(self, doid: str) -> List[str]:
        """Returns deterministic accepted UMLS CUIs associated with a DOID."""
        norm = normalize_doid(doid)
        if norm in self._doid_to_umls:
            return list(self._doid_to_umls[norm])
        canon = self.resolver.canonicalize(norm)
        if canon:
            mapped = self._to_hetionet_or_canonical(canon)
            return list(self._doid_to_umls.get(mapped, []))
        return []

    def resolve_compound(self, query: str) -> Optional[str]:
        """Resolves a drug name, DrugBank ID, or Hetionet Compound ID to `Compound::DBxxxxx`."""
        q = str(query or "").strip().lower()
        if not q:
            return None
        return self._compound_lookup.get(q)

    def resolve_disease_to_hetionet(
        self,
        identifier: str,
        max_dist: int = 4,
        decay_base: float = 0.90,
    ) -> List[Tuple[str, float, int, str, str]]:
        """
        End-to-end deterministic disease identifier resolution to Hetionet v1.0's 137 diseases.
        Accepts `DOID:...`, `Disease::DOID:...`, `MESH:...`, `UMLS_CUI:...`, `UMLS:...`,
        or bare MeSH (`D...`) / UMLS (`C...`) codes.
        """
        raw = str(identifier or "").strip()
        if not raw:
            return []
        upper = raw.upper()
        doid: Optional[str] = None
        if upper.startswith("DISEASE::DOID:") or upper.startswith("DOID:"):
            doid = normalize_doid(raw)
        elif upper.startswith("MESH:"):
            doid = self.mesh_to_doid(raw)
        elif upper.startswith("UMLS_CUI:") or upper.startswith("UMLS:"):
            doid = self.umls_to_doid(raw)
        elif upper.startswith("D") and len(upper) >= 7 and upper[1:].isdigit():
            doid = self.mesh_to_doid(upper)
        elif upper.startswith("C") and len(upper) >= 8 and upper[1:].isdigit():
            doid = self.umls_to_doid(upper)

        if not doid:
            return []
        return self.resolver.resolve(doid, max_dist=max_dist, decay_base=decay_base)

    def stats(self) -> Dict[str, Any]:
        """Returns comprehensive deterministic mapping & coverage metrics."""
        mesh_1to1 = sum(1 for m in self._mesh_mapping_mode.values() if m == "1:1")
        mesh_1ton = sum(1 for m in self._mesh_mapping_mode.values() if m == "1:N_hetionet_unique")
        umls_1to1 = sum(1 for m in self._umls_mapping_mode.values() if m == "1:1")
        umls_1ton = sum(1 for m in self._umls_mapping_mode.values() if m == "1:N_hetionet_unique")

        het_mesh_covered = {d for d in self.hetionet_diseases if self._doid_to_mesh.get(d)}
        het_umls_covered = {d for d in self.hetionet_diseases if self._doid_to_umls.get(d)}
        het_union = het_mesh_covered | het_umls_covered

        return {
            "mesh_accepted_total": len(self._mesh_to_doid),
            "mesh_1to1": mesh_1to1,
            "mesh_1ton_hetionet_unique": mesh_1ton,
            "mesh_rejected": len(self.rejected_mesh),
            "umls_accepted_total": len(self._umls_to_doid),
            "umls_1to1": umls_1to1,
            "umls_1ton_hetionet_unique": umls_1ton,
            "umls_rejected": len(self.rejected_umls),
            "hetionet_diseases_total": len(self.hetionet_diseases),
            "hetionet_diseases_mesh_covered": len(het_mesh_covered),
            "hetionet_diseases_umls_covered": len(het_umls_covered),
            "hetionet_diseases_union_covered": len(het_union),
            "hetionet_compounds_total": len(self.compounds),
            "external_files_loaded": list(self.external_files_loaded),
        }
