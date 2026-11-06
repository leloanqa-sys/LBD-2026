import importlib.util
from pathlib import Path
import numpy as np

spec=importlib.util.spec_from_file_location('repodb_benchmark',Path(__file__).with_name('run_benchmark.py'))
bench=importlib.util.module_from_spec(spec);spec.loader.exec_module(bench)

def test_ontology_mapping_uses_exact_xref_not_names(tmp_path):
    p=tmp_path/'do.obo'
    p.write_text('[Term]\nid: DOID:1\nname: heart failure\nxref: UMLS_CUI:C0001\n\n[Term]\nid: DOID:2\nname: heart failure\nxref: UMLS_CUI:C0002\n',encoding='utf-8')
    result=bench.read_ontology(p,['Disease::DOID:1'])
    assert result=={'C0001':{'Disease::DOID:1'}}

def test_one_to_many_xref_is_preserved_for_rejection(tmp_path):
    p=tmp_path/'do.obo'
    p.write_text('[Term]\nid: DOID:1\nxref: UMLS_CUI:C0001\n\n[Term]\nid: DOID:2\nxref: UMLS_CUI:C0001\n',encoding='utf-8')
    assert len(bench.read_ontology(p,['Disease::DOID:1','Disease::DOID:2'])['C0001'])==2

def test_ranking_keeps_unknown_candidates_and_filters_known_gold():
    scores=np.arange(12,dtype=float).reshape(1,12)
    gold={0:[(0,1),(11,1),(8,0)]}
    all_result=bench.ranking_metrics(scores,gold,[0],12,set(),np.zeros(12,dtype=int))
    assert all_result['recall@5']==.5
    novel=bench.ranking_metrics(scores,gold,[0],12,{11},np.zeros(12,dtype=int))
    assert novel['positive_pairs']==1 and novel['recall@5']==0
    assert novel['per_drug'][0]['candidates']==11

def test_no_positive_is_not_reported_as_success():
    result=bench.ranking_metrics(np.zeros((1,3)),{0:[(1,0)]},[0],3,set(),np.zeros(3,dtype=int))
    assert result['status']=='NOT_EVALUATED'
