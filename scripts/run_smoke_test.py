"""End-to-end Smoke Test for Dapagliflozin LBD Pipeline."""
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from lbd.data.fetch_pubmed import create_document_record
from lbd.data.normalize import normalize_relation_record
from lbd.data.load_hetionet import load_hetionet_subgraph
from lbd.nlp.ner import annotate_abstract_pubtator
from lbd.nlp.relation_extraction import extract_relations_from_text
from lbd.nlp.postprocess import postprocess_relations
from lbd.graph.build_graph import create_knowledge_graph
from lbd.graph.path_search import search_metapaths
from lbd.ranking.score_paths import score_individual_path
from lbd.ranking.rank_candidates import score_disease_candidate
from lbd.evaluation.run_experiment import evaluate_rankings
from lbd.common.logging import get_logger

logger = get_logger("smoke_test")

def run():
    logger.info("=== Starting Smoke Test for Dapagliflozin LBD ===")
    
    # 1. Test Ingestion & Document Record
    abstract = "Dapagliflozin inhibits SLC5A2 in kidney tubules. Natriuresis decreases plasma volume. Plasma volume causes heart failure burden."
    doc = create_document_record("doc_test", "12345", "Test Title", abstract, "2018-01-01", "clinical_trial")
    assert doc["temporal_filter_status"] == "passed"
    logger.info("1. Document record temporal check passed.")
    
    # 2. Test NER & Relation Extraction
    ents = annotate_abstract_pubtator("12345", abstract)
    raw_rels = extract_relations_from_text("12345", abstract)
    cleaned_rels = postprocess_relations(raw_rels, abstract)
    normed_rels = [normalize_relation_record(r) for r in cleaned_rels]
    logger.info(f"2. Extracted and normalized {len(normed_rels)} relations.")
    
    # 3. Test Graph Build & Metapath Search
    hetio = load_hetionet_subgraph("DB06292")
    G = create_knowledge_graph(hetio, normed_rels)
    paths = search_metapaths(G, "DB06292", "DOID:9333", min_length=3, max_length=5)
    logger.info(f"3. Found {len(paths)} metapaths to Heart Failure (DOID:9333).")
    
    # 4. Score Paths & Rank Candidate
    scored_paths = [score_individual_path(p, hub_degrees=[10]) for p in paths]
    candidate = score_disease_candidate("DOID:9333", "Heart Failure", scored_paths, contradicting_pmids=[])
    logger.info(f"4. Candidate Heart Failure score: {candidate['ranking_score']}")
    
    # 5. Evaluate Rankings
    eval_res = evaluate_rankings([candidate])
    logger.info(f"5. Evaluation results: Recall@5={eval_res['recall@5']}, MRR={eval_res['mrr']}")
    
    logger.info("=== Smoke Test Completed Successfully! ===")

if __name__ == "__main__":
    run()
