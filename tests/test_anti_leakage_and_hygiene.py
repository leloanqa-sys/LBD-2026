"""Anti-Leakage, Ground-Truth Integrity, and Code Hygiene Gate Tests.

Enforces Chief Engineer QA standards:
1. All 27 pairs in `prospective_temporal_test.json` (15 post-2017 FDA positives +
   12 post-2017 clinical failure hard negatives) exist in Hetionet v1.0 nodes.
2. Zero pairs overlap with Hetionet v1.0 `CtD` or `CpD` training edges.
3. Quarantined legacy scripts (`run_system_mrr_precision_ablation.py` and
   `run_landmark_ablation_study.py` C6) raise RuntimeError when invoked.
4. `test_nlp_connection.py` contains no fake `ONTOLOGY_ALIASES` mapping
   (`Disease::DOID:10763` -> `Disease::DOID:3393` / `Disease::DOID:1936`) and
   does not perform unfiltered reads of `relations_contract_b.csv`.
5. `verify_report_sync.py` runs deterministically and audits metric sync.
"""

from __future__ import annotations

import csv
import gzip
import importlib.util
import json
from pathlib import Path
from typing import Dict, Set, Tuple

import pytest

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
LBD_ROOT = WORKSPACE_ROOT / "dapagliflozin-lbd"
HETIONET_NODES = LBD_ROOT / "data" / "raw" / "hetionet-v1.0-nodes.tsv"
HETIONET_EDGES = LBD_ROOT / "data" / "raw" / "hetionet-v1.0-edges.sif.gz"
TEST_SUITE_PATH = (
    WORKSPACE_ROOT
    / "repurposing-ranking"
    / "data"
    / "test_suites"
    / "prospective_temporal_test.json"
)
MRR_ABLATION_SCRIPT = (
    WORKSPACE_ROOT
    / "repurposing-ranking"
    / "scripts"
    / "run_system_mrr_precision_ablation.py"
)
LANDMARK_ABLATION_SCRIPT = (
    WORKSPACE_ROOT
    / "repurposing-ranking"
    / "scripts"
    / "run_landmark_ablation_study.py"
)
NLP_CONN_SCRIPT = (
    WORKSPACE_ROOT / "experiments_clean_ablation" / "test_nlp_connection.py"
)
VERIFY_SYNC_SCRIPT = (
    WORKSPACE_ROOT / "evaluation_metrics_summary" / "verify_report_sync.py"
)


def _load_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_hetionet_nodes() -> Dict[str, Tuple[str, str]]:
    nodes: Dict[str, Tuple[str, str]] = {}
    with HETIONET_NODES.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            nodes[row["id"]] = (row["name"], row["kind"])
    return nodes


def _load_hetionet_ctd_cpd() -> Tuple[Set[Tuple[str, str]], Set[Tuple[str, str]]]:
    ctd: Set[Tuple[str, str]] = set()
    cpd: Set[Tuple[str, str]] = set()
    with gzip.open(HETIONET_EDGES, "rt", encoding="utf-8") as f:
        header = next(f, None)
        assert header is not None
        for line in f:
            s, r, t = line.strip().split("\t")
            if r == "CtD":
                ctd.add((s, t))
            elif r == "CpD":
                cpd.add((s, t))
    return ctd, cpd


def test_prospective_temporal_suite_schema_and_counts():
    assert TEST_SUITE_PATH.exists(), f"Missing test suite: {TEST_SUITE_PATH}"
    data = json.loads(TEST_SUITE_PATH.read_text(encoding="utf-8"))
    assert isinstance(data, list)
    assert len(data) == 27, f"Expected 27 pairs, found {len(data)}"

    positives = [r for r in data if r["label"] == 1]
    hard_negatives = [r for r in data if r["label"] == 0]
    assert len(positives) == 15, f"Expected 15 positives, found {len(positives)}"
    assert len(hard_negatives) == 12, f"Expected 12 hard negatives, found {len(hard_negatives)}"

    required_fields = {
        "pair_id",
        "drug_name",
        "drug_id",
        "disease_name",
        "disease_id",
        "label",
        "split_group",
        "temporal_year",
        "evidence_source",
        "physiological_mechanism",
    }
    seen_ids = set()
    seen_pairs = set()
    for row in data:
        missing = required_fields - set(row.keys())
        assert not missing, f"Row {row.get('pair_id')} missing fields: {missing}"
        assert row["pair_id"] not in seen_ids, f"Duplicate pair_id: {row['pair_id']}"
        seen_ids.add(row["pair_id"])
        pair_key = (row["drug_id"], row["disease_id"])
        assert pair_key not in seen_pairs, f"Duplicate (drug_id, disease_id): {pair_key}"
        seen_pairs.add(pair_key)
        assert int(row["temporal_year"]) >= 2018, (
            f"Pair {row['pair_id']} is not post-2017: year={row['temporal_year']}"
        )


def test_prospective_temporal_suite_all_nodes_exist_in_hetionet():
    assert HETIONET_NODES.exists(), f"Missing Hetionet nodes file: {HETIONET_NODES}"
    nodes = _load_hetionet_nodes()
    data = json.loads(TEST_SUITE_PATH.read_text(encoding="utf-8"))

    missing_drugs = []
    missing_diseases = []
    for row in data:
        d_id = row["drug_id"]
        dis_id = row["disease_id"]
        if d_id not in nodes or nodes[d_id][1] != "Compound":
            missing_drugs.append((row["pair_id"], d_id))
        if dis_id not in nodes or nodes[dis_id][1] != "Disease":
            missing_diseases.append((row["pair_id"], dis_id))

    assert not missing_drugs, f"Drug IDs missing from Hetionet v1.0: {missing_drugs}"
    assert not missing_diseases, f"Disease IDs missing from Hetionet v1.0: {missing_diseases}"


def test_prospective_temporal_suite_zero_ctd_and_cpd_leakage():
    assert HETIONET_EDGES.exists(), f"Missing Hetionet edges file: {HETIONET_EDGES}"
    ctd, cpd = _load_hetionet_ctd_cpd()
    data = json.loads(TEST_SUITE_PATH.read_text(encoding="utf-8"))

    ctd_leaks = []
    cpd_leaks = []
    for row in data:
        pair = (row["drug_id"], row["disease_id"])
        if pair in ctd:
            ctd_leaks.append((row["pair_id"], pair))
        if pair in cpd:
            cpd_leaks.append((row["pair_id"], pair))

    assert len(ctd_leaks) == 0, f"CtD training edge leakage detected: {ctd_leaks}"
    assert len(cpd_leaks) == 0, f"CpD training edge leakage detected: {cpd_leaks}"


def test_quarantined_mrr_precision_ablation_raises_runtime_error():
    mod = _load_module(MRR_ABLATION_SCRIPT, "quarantined_mrr_ablation")
    with pytest.raises(RuntimeError, match="QUARANTINED"):
        mod.run_ablation()


def test_quarantined_landmark_ablation_c6_raises_runtime_error():
    mod = _load_module(LANDMARK_ABLATION_SCRIPT, "quarantined_landmark_ablation")
    with pytest.raises(RuntimeError, match="QUARANTINED"):
        mod.run_ablation()


def test_nlp_connection_no_fake_ontology_aliases_and_no_unfiltered_csv_bypass():
    src = NLP_CONN_SCRIPT.read_text(encoding="utf-8")
    assert "ONTOLOGY_ALIASES = {" not in src, "Hardcoded ONTOLOGY_ALIASES dict still present!"
    assert '"Disease::DOID:10763"' not in src, "Fake DOID:10763 ontology alias still present!"
    assert "open(b_csv" not in src, "Unfiltered direct open() of relations_contract_b.csv still present!"
    assert "csv.DictReader" not in src, "Unfiltered csv.DictReader ingestion still present!"


def test_verify_report_sync_gate_executes_and_reports_status():
    assert VERIFY_SYNC_SCRIPT.exists(), f"Missing {VERIFY_SYNC_SCRIPT}"
    mod = _load_module(VERIFY_SYNC_SCRIPT, "verify_report_sync_mod")
    report = mod.verify_report_sync(workspace_root=WORKSPACE_ROOT)
    assert report["overall_status"] in {"PASS", "FAIL"}
    assert report["total_checks"] >= 21
    assert report["tolerance"] == pytest.approx(1e-4)
