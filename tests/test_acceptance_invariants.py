"""Acceptance requirements: failures expose defects, never xfail known defects.

Fixtures are algorithm inputs, not biomedical evidence. No API calls are made.
"""
from copy import deepcopy
from types import SimpleNamespace

import networkx as nx
import pytest

from lbd.graph.build_graph import create_knowledge_graph
from lbd.graph.path_search import search_metapaths
from lbd.nlp.engine import NLPEngine
from lbd.nlp.fusion import BayesianFusionEngine
from lbd.nlp.postprocess import postprocess_relations
from lbd.nlp.zoning import ZoningEngine
from lbd.ranking.score_paths import calculate_net_sign, get_direction_fit


def test_contract_d_reaches_graph_without_seed_edges():
    record = dict(edge_id='E1', source='A', target='B', metaedge='CtD',
                  relation_type='TREATS', assertion_status='asserted', sign=-1,
                  evidence_ids=['R1'], pmids=['123'], source_dataset='nlp_literature', tier=1)
    graph = create_knowledge_graph({'nodes': [], 'edges': []}, [record])
    assert graph.number_of_edges() == 1, 'Contract D edge silently lost'
    edge = next(iter(graph.get_edge_data('A', 'B').values()))
    for field in ('sign', 'evidence_ids', 'pmids', 'assertion_status'):
        assert edge[field] == record[field]


def test_fusion_partition_is_permutation_invariant():
    engine = NLPEngine()
    _, events, _ = engine.run_document(dict(pmid='123', title='', abstract='Dapagliflozin reduces heart failure.'))
    asserted = deepcopy(events[0].projected_edges[0])
    uncertain = deepcopy(asserted)
    uncertain.assertion_status = 'uncertain'
    uncertain.pmid = '456'
    fuser = BayesianFusionEngine()
    def result(items):
        _, edges = fuser.fuse_edges([SimpleNamespace(projected_edges=items)])
        return [(e.canonical_relation, e.assertion_status, e.partition, e.posterior_mean) for e in edges]
    assert result([asserted, uncertain]) == result([uncertain, asserted])


@pytest.mark.parametrize('section', ['BACKGROUND', 'RESULTS', 'CONCLUSIONS', 'METHODS'])
def test_structured_section_persists_until_next_header(section):
    sentences = ZoningEngine().split_and_zone_document('', f'{section}: First sentence. Further research is needed. RESULTS: Outcomes improved.')
    assert sentences[1].section_zone == section


def test_objective_alone_is_not_conclusion():
    sentence = ZoningEngine().split_and_zone_document('', 'We aim to investigate dapagliflozin in heart failure.')[0]
    assert sentence.section_zone != 'CONCLUSIONS'


def test_empty_evidence_is_not_verified_in_strict_mode():
    record = dict(pmid='123', source_entity='A', target_entity='B', relation_type='TREATS', evidence='')
    assert postprocess_relations([record], 'Some unrelated text.', strict=True) == []


def test_same_pmid_does_not_double_fusion_evidence():
    _, events, _ = NLPEngine().run_document(dict(pmid='123', title='', abstract='Dapagliflozin reduces heart failure.'))
    fuser = BayesianFusionEngine()
    def posterior(items):
        _, edges = fuser.fuse_edges(items)
        return [(e.alpha_post, e.beta_post, e.posterior_mean) for e in edges]
    assert posterior(events) == posterior(events + deepcopy(events))


def test_dfs_equals_independent_networkx_oracle_with_cycle_and_parallel_edges():
    graph = nx.MultiDiGraph()
    for u, v, key in [('A','B','ab1'), ('A','B','ab2'), ('B','C','bc'), ('C','B','cycle'), ('C','D','cd'), ('A','D','direct')]:
        graph.add_edge(u,v,key=key,edge_id=key)
    actual = search_metapaths(graph, 'A','D',min_length=3,max_length=4,filter_historical=False)
    expected = [path for path in nx.all_simple_edge_paths(graph,'A','D',cutoff=3) if len(path)>=2]
    assert sorted(tuple(e['edge_id'] for e in p) for p in actual) == sorted(tuple(k for _,_,k in p) for p in expected)


@pytest.mark.parametrize('signs,expected', [([-1,1],-1),([-1,-1],1),([-1,0],0)])
def test_sign_algebra_and_direction(signs, expected):
    sign = calculate_net_sign([dict(sign=s) for s in signs])
    assert sign == expected
    assert get_direction_fit(sign) == {-1:1.5,0:1.0,1:0.0}[expected]
