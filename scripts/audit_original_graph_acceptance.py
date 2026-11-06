"""Offline acceptance audit; graph is read-only. Independent NetworkX DFS oracle."""
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import networkx as nx
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from lbd.graph.path_search import search_metapaths
from lbd.ranking.score_paths import calculate_net_sign


def main():
    source = ROOT / 'data/processed/merged_graph.json'
    before = hashlib.sha256(source.read_bytes()).hexdigest()
    data = json.loads(source.read_text(encoding='utf-8'))
    cfg = yaml.safe_load((ROOT / 'configs/ranking.yaml').read_text(encoding='utf-8'))['search']
    graph = nx.MultiDiGraph()
    for node in data['nodes']:
        graph.add_node(node['id'], **{k:v for k,v in node.items() if k!='id'})
    for i, edge in enumerate(data['edges']):
        graph.add_edge(edge['source'],edge['target'],key=i,**{**edge,'edge_id':str(i)})
    duplicates = Counter(n['id'] for n in data['nodes'])
    declared = set(duplicates)
    missing_endpoints = sorted({e[k] for e in data['edges'] for k in ('source','target')} - declared)
    comparisons, routes, signs = [], Counter(), Counter()
    total = 0
    matches = 0
    for disease in sorted(n for n,a in graph.nodes(data=True) if a.get('type')=='Disease'):
        actual = search_metapaths(graph,'DB06292',disease,cfg['min_length'],cfg['max_length'],True)
        oracle = []
        for path in nx.all_simple_edge_paths(graph,'DB06292',disease,cutoff=cfg['max_length']-1):
            nodes = ['DB06292'] + [v for u,v,k in path]
            if len(nodes) < cfg['min_length'] or 'DOID:9352' in nodes[1:-1]:
                continue
            oracle.append(tuple(str(k) for u,v,k in path))
        observed = [tuple(e['edge_id'] for e in p) for p in actual]
        comparisons.append(dict(target=disease, actual=len(observed), oracle=len(oracle), equal=Counter(observed)==Counter(oracle)))
        for path in actual:
            nodes = [path[0]['source']] + [e['target'] for e in path]
            types = [graph.nodes[n].get('type') for n in nodes]
            routes[' -> '.join(types)] += 1
            signs[str(calculate_net_sign(path))] += 1
            matches += types in cfg['allowed_metapaths']
            total += 1
    result = dict(
        scope='Local baseline snapshot; upstream Hetionet authenticity and clinical predictions not certified',
        graph_sha256=before, graph_unchanged=before==hashlib.sha256(source.read_bytes()).hexdigest(),
        nodes=graph.number_of_nodes(), edges=graph.number_of_edges(),
        duplicate_node_ids=[k for k,v in duplicates.items() if v>1], missing_endpoints=missing_endpoints,
        node_types=dict(Counter(a.get('type') for n,a in graph.nodes(data=True))),
        paths=total, configured_metapath_matches=matches, path_type_sequences=dict(routes), path_net_signs=dict(signs),
        edges_with_pmids=sum(bool(e.get('pmids')) for e in data['edges']),
        edges_with_evidence_sentences=sum(bool(e.get('evidence_sentences')) for e in data['edges']),
        independent_dfs_oracle=dict(all_targets_equal=all(r['equal'] for r in comparisons),targets=len(comparisons),details=comparisons),
        target_presence={x: x in graph for x in ['DOID:6000','DOID:9333','DOID:784']},
        acceptance=dict(dfs_enumeration='PASS' if all(r['equal'] for r in comparisons) else 'FAIL',
                        configured_metapaths='PASS' if matches else 'FAIL',
                        therapeutic_evidence='NOT_ESTABLISHED', production_end_to_end='FAIL'),
    )
    out = ROOT / 'artifacts/original_graph_acceptance.json'
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='independent_dfs_oracle'},ensure_ascii=True,indent=2))
    print('oracle_all_targets_equal:',result['independent_dfs_oracle']['all_targets_equal'])


if __name__=='__main__':
    main()
