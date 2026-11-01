"""Hetionet loader boundary. This project has no production Hetionet file loader yet."""
from typing import Any, Dict, List


def load_hetionet_subgraph(drug_id: str = "DB06292") -> Dict[str, List[Dict[str, Any]]]:
    """Fail closed instead of returning the former hardcoded demo graph.

    Supply a real Hetionet-backed implementation before using this pipeline for
    inference. The previous seven-node sample graph was only a smoke-test mock
    and must not be treated as scientific evidence.
    """
    raise RuntimeError(
        "Hetionet production loader is not configured; refusing to substitute mock data. "
        "Load a versioned Hetionet graph and implement the source-backed loader first."
    )
