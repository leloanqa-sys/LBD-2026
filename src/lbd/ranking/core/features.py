"""Reference simple-path features; sparse WALK counts are explicitly separate.

Reference weighting uses per-metarelation source/target degrees on each edge.
This is not asserted to be an exact reproduction of Rephetio normalization.
"""
from collections import defaultdict
import math
from scipy import sparse
from .signs import map_sign

BRIDGES = ("BiologicalProcess", "Pathway", "MolecularFunction", "CellularComponent")
METAPATHS = [("Compound","Gene",k,"Gene","Disease") for k in BRIDGES] + [("Compound","Gene","Gene","Disease"),("Compound","Gene","Disease")]

def relation_degrees(graph):
    out,inc = defaultdict(int),defaultdict(int)
    for u,v,e in graph.edges(data=True):
        out[u,e["metaedge"]]+=1; inc[v,e["metaedge"]]+=1
    return out,inc

def iter_paths(graph, drug, disease, max_nodes=5, max_degree=None, max_paths=100000):
    if drug not in graph or disease not in graph: raise ValueError("Query endpoint absent")
    emitted=0
    def visit(node,nodes,edges):
        nonlocal emitted
        if node == disease:
            if tuple(graph.nodes[n]["type"] for n in nodes) in METAPATHS:
                emitted+=1
                if emitted>max_paths: raise RuntimeError("Path budget exceeded; incomplete features must not be trained")
                yield list(edges)
            return
        if len(nodes)>=max_nodes: return
        for _,v,key,e in graph.out_edges(node,keys=True,data=True):
            if v in nodes: continue
            if v != disease and max_degree is not None and graph.degree(v)>max_degree: continue
            yield from visit(v,nodes+[v],edges+[(node,v,key,e)])
    yield from visit(drug,[drug],[])

def pair_features(graph,drug,disease,w=0.4,max_paths=100000):
    if not 0 <= w <= 1: raise ValueError("degree exponent outside [0,1]")
    out,inc=relation_degrees(graph)
    result=defaultdict(float)
    result.update(log_drug_degree=math.log1p(graph.degree(drug)),log_disease_degree=math.log1p(graph.degree(disease)))
    targets=set(); node_chains=set()
    for path in iter_paths(graph,drug,disease,max_paths=max_paths):
        # Key includes edge relation sequence, not only node type sequence.
        name="|".join(e["metaedge"] for _,_,_,e in path)
        weight=math.prod((out[u,e["metaedge"]]*inc[v,e["metaedge"]])**(-w) for u,v,k,e in path)
        result["dwpc:"+name]+=weight
        codes=[map_sign(e) for _,_,_,e in path]
        state="unknown" if any(e["code"]=="unknown" or e["negated"] for e in codes) else "odd" if sum(e["code"]=="negative" for e in codes)%2 else "even"
        result["sign:"+state+":"+name]+=weight
        targets.add(path[0][1]);node_chains.add(tuple([drug]+[v for _,v,_,_ in path]))
    result["distinct_targets"]=len(targets); result["distinct_node_chains"]=len(node_chains)
    return dict(result)

def sparse_walk_counts(graph, relation_sequence):
    """Unweighted metarelation walk count for validation/scaling work ONLY.
    Repeated nodes are permitted. Do not rename this result DWPC.
    """
    nodes=sorted(graph.nodes);index={n:i for i,n in enumerate(nodes)}
    value=sparse.eye(len(nodes),format="csr")
    for relation in relation_sequence:
        rows=[];cols=[]
        for u,v,e in graph.edges(data=True):
            if e["metaedge"]==relation: rows.append(index[u]);cols.append(index[v])
        matrix=sparse.csr_matrix(([1.0]*len(rows),(rows,cols)),shape=(len(nodes),len(nodes)))
        value=value@matrix
    return nodes,value
