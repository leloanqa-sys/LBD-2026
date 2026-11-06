"""Offline boundary tests. HTTP mocks simulate faults, not biomedical evidence."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from urllib.error import HTTPError

import pytest

from lbd.data.fetch_pubmed import filter_by_cutoff
from lbd.data.load_hetionet import load_hetionet_subgraph
from lbd.nlp.events import BioEventEngine


@pytest.mark.parametrize('date', ['', 'garbage', '2020-01-01'])
def test_invalid_or_post_cutoff_date_rejected(date):
    assert not filter_by_cutoff({'publication_date': date})


@pytest.mark.parametrize('status', [401, 429, 500])
def test_mock_http_failure_returns_explicit_symbolic_fallback(tmp_path, status):
    engine = BioEventEngine(cache_path=tmp_path/'cache.json', enable_live_llm=True)
    engine.api_key = 'offline-test-key'
    symbolic = SimpleNamespace(canonical_relation='ASSOCIATED_WITH', assertion_status='asserted', verb_delta=0, rationale='test')
    with patch('urllib.request.urlopen', side_effect=HTTPError('https://offline.invalid', status, 'mock', {}, None)):
        result = engine.verify_with_blinded_llm('mock sentence', '@SRC$', 'Compound', '@TGT$', 'Disease', 1, symbolic)
    assert result['llm_mode'] == 'symbolic_fallback_http'
    assert result['llm_verified'] is False
    assert result['relation'] == symbolic.canonical_relation


def test_snapshot_loader_requires_explicit_source():
    with pytest.raises(ValueError, match='graph_path'):
        load_hetionet_subgraph()


def test_packaged_schemas_match_repository_sources():
    root = Path(__file__).resolve().parents[1]
    for schema in (root/'contracts').glob('*.json'):
        assert schema.read_bytes() == (root/'src/lbd/contracts'/schema.name).read_bytes()
