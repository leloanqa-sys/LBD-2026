#!/usr/bin/env python3
"""End-to-End System Integration Smoke Test.

Exercises the real pipeline across all components:
1. Canonical Biomedical NLP extraction & evidence postprocessing
2. Data Contract B and Contract D schema validation
3. Entity & relation normalization
4. Knowledge Graph construction
5. Metapath search & biological sign consistency checking
"""
import sys
from pathlib import Path

# Add src to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))

from lbd.common.logging import get_logger
from lbd.common.validation import validate_contract
from lbd.data.normalize import normalize_relation_record
from lbd.graph.build_graph import create_knowledge_graph
from lbd.graph.path_search import search_metapaths
from lbd.nlp.engine import NLPEngine
from lbd.nlp.relation_extraction import extract_relations_from_text
from lbd.ranking.score_paths import calculate_net_sign, get_direction_fit

logger = get_logger("smoke_test")


def main() -> int:
    logger.info("=== Starting LBD System Integration Smoke Test ===")

    # Step 1: Real NLP Relation Extraction
    sample_pmid = "29085817"
    sample_title = "Dapagliflozin in patients with heart failure and type 2 diabetes"
    sample_abstract = (
        "Dapagliflozin reduces worsening heart failure and cardiovascular death in patients "
        "with type 2 diabetes. Dapagliflozin inhibits SLC5A2, lowering blood glucose. "
        "Inhibition of SLC5A2 attenuates chronic kidney disease progression."
    )

    logger.info("Step 1: Running canonical NLPEngine relation extraction...")
    relations = extract_relations_from_text(
        sample_pmid, sample_abstract, title=sample_title, strict_postprocess=True
    )
    logger.info(f"Extracted {len(relations)} relations from sample abstract.")
    if not relations:
        logger.error("FAIL: No relations extracted from sample abstract.")
        return 1

    # Step 2: Contract B Validation
    logger.info("Step 2: Validating Contract B records against JSON Schema...")
    for rel in relations:
        is_valid, msg = validate_contract(rel, "relation.schema.json")
        if not is_valid:
            logger.error(f"FAIL: Contract B validation failed for {rel.get('relation_id')}: {msg}")
            return 1
    logger.info(f"PASS: All {len(relations)} relations conform strictly to Contract B.")

    # Step 3: Normalization
    logger.info("Step 3: Normalizing relations...")
    normed_relations = [normalize_relation_record(r) for r in relations]
    logger.info(f"PASS: Normalized {len(normed_relations)} relation records.")

    # Step 4: Knowledge Graph Construction & Contract D Validation
    logger.info("Step 4: Fusing into Knowledge Graph and validating Contract D edges...")
    engine = NLPEngine()
    doc = {"pmid": sample_pmid, "title": sample_title, "abstract": sample_abstract}
    _, events, _ = engine.run_document(doc)
    contract_b, fused, contract_d = engine.fuse_corpus(events)
    if not contract_d:
        logger.error("FAIL: No fused edges produced for Contract D.")
        return 1

    for edge in contract_d:
        is_valid, msg = validate_contract(edge, "graph_edge.schema.json")
        if not is_valid:
            logger.error(f"FAIL: Contract D validation failed for {edge.get('edge_id')}: {msg}")
            return 1
    logger.info(f"PASS: Produced and validated {len(contract_d)} Contract D graph edges.")

    # Step 5: Graph Building & Path Search
    logger.info("Step 5: Testing Knowledge Graph traversal & metapath search...")
    hetio_subgraph = {
        "nodes": [
            {"id": "DB06292", "canonical_name": "Dapagliflozin"},
            {"id": "6524", "canonical_name": "SLC5A2"},
            {"id": "DOID:6000", "canonical_name": "Heart Failure"},
            {"id": "DOID:9352", "canonical_name": "Type 2 Diabetes Mellitus"},
        ],
        "edges": [
            {"source": "DB06292", "target": "6524", "relation_type": "INHIBITS", "sign": -1},
            {"source": "6524", "target": "DOID:6000", "relation_type": "CAUSES", "sign": 1},
            {"source": "6524", "target": "DOID:9352", "relation_type": "TREATS", "sign": -1},
        ],
    }
    graph = create_knowledge_graph(hetio_subgraph, contract_d)
    paths = search_metapaths(graph, "DB06292", "DOID:6000", min_length=2, max_length=4)
    logger.info(f"Found {len(paths)} metapath(s) connecting Dapagliflozin to Heart Failure.")
    if not paths:
        logger.error("FAIL: Expected at least one metapath to Heart Failure.")
        return 1

    # Step 6: Biological Sign Consistency
    logger.info("Step 6: Testing biological sign algebra...")
    golden_path_edges = [{"sign": -1}, {"sign": 1}]  # INHIBITS (-1) * CAUSES (+1) = -1 (Therapeutic)
    net_sign = calculate_net_sign(golden_path_edges)
    fit = get_direction_fit(net_sign)
    if net_sign != -1 or fit != 1.5:
        logger.error(f"FAIL: Expected net_sign=-1 and fit=1.5, got sign={net_sign}, fit={fit}")
        return 1
    logger.info("PASS: Biological sign algebra matches theoretical expectation.")

    logger.info("=== ALL SMOKE TESTS PASSED CLEANLY (0 ERRORS) ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
