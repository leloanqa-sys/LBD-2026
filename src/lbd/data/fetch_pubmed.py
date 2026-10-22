"""PubMed Abstract Fetcher with Temporal Cutoff Guard (<= 2018-12-31)."""
from datetime import datetime
from typing import Dict, List, Any
from lbd.common.logging import get_logger

logger = get_logger("fetch_pubmed")
DEFAULT_CUTOFF = "2018-12-31"

def parse_date(date_str: str) -> datetime:
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y-%m", "%Y"):
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    return datetime(1970, 1, 1)

def filter_by_cutoff(record: Dict[str, Any], cutoff: str = DEFAULT_CUTOFF) -> bool:
    pub_date = parse_date(record.get("publication_date", ""))
    cutoff_dt = parse_date(cutoff)
    return pub_date <= cutoff_dt

def create_document_record(
    document_id: str,
    pmid: str,
    title: str,
    abstract: str,
    publication_date: str,
    study_type: str = "unknown",
    dataset_version: str = "v1.0-pre2018"
) -> Dict[str, Any]:
    record = {
        "document_id": document_id,
        "pmid": pmid,
        "title": title,
        "abstract": abstract,
        "publication_date": publication_date,
        "study_type": study_type,
        "dataset_version": dataset_version,
        "retrieved_at": datetime.utcnow().isoformat() + "Z",
        "temporal_filter_status": "passed" if filter_by_cutoff({"publication_date": publication_date}) else "rejected"
    }
    return record
