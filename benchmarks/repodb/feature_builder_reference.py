"""Feature-only extraction from repurposing-ranking/scripts/run_hetionet_real.py.
Original SHA256: 677763e724dd2bbb57b0e07074aba5b16e7bb0167fcb80a4a089364cd1080756.
Caller must set RAW and OUT; no model fitting or labels used by benchmark.
"""
import os

import csv,gzip,json,hashlib,time,pickle,sys,warnings

from pathlib import Path

from collections import Counter,defaultdict

import numpy as np

import networkx as nx

from scipy import sparse

from sklearn.model_selection import GroupShuffleSplit,StratifiedGroupKFold

from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import average_precision_score,roc_auc_score

from sklearn.exceptions import ConvergenceWarning

from sklearn import __version__ as skversion

def dump(name,x): (OUT/name).write_text(json.dumps(x,indent=2,ensure_ascii=False),encoding='utf-8')

def weighted(a,w=.4):
    a=a.astype(np.float64).tocsr();a.sum_duplicates();a.data[:]=1
    rows=np.asarray(a.sum(axis=1)).ravel();cols=np.asarray(a.sum(axis=0)).ravel()
    return sparse.diags(np.maximum(rows,1)**(-w))@a@sparse.diags(np.maximum(cols,1)**(-w))

def features():
    nodes=list(csv.DictReader((RAW/'hetionet-v1.0-nodes.tsv').open(encoding='utf-8'),delimiter='\t'))
    kinds=defaultdict(list)
    for n in nodes:kinds[n['kind'].replace(' ','')].append(n['id'])
    for k in kinds:kinds[k].sort()
    indices={k:{n:i for i,n in enumerate(v)} for k,v in kinds.items()}
    selected=['CbG','CuG','CdG','DaG','DuG','DdG','GpBP','GpPW','GpMF','GpCC','GiG','GcG']
    edges={k:[[],[]] for k in selected};counts=Counter();degree=Counter();pos=[];pall=[]
    cluster=nx.Graph();cluster.add_nodes_from(kinds['Compound'])
    with gzip.open(RAW/'hetionet-v1.0-edges.sif.gz','rt') as f:
        for e in csv.DictReader(f,delimiter='\t'):
            u,m,v=e['source'],e['metaedge'],e['target'];counts[m]+=1
            if m=='CtD':pos.append((u,v));continue
            if m=='CpD':pall.append((u,v));continue
            degree[u]+=1;degree[v]+=1
            if m=='CrC':cluster.add_edge(u,v)
            if m in edges:
                edges[m][0].append(indices[u.split('::')[0].replace(' ','')][u]);edges[m][1].append(indices[v.split('::')[0].replace(' ','')][v])
    shape={'CbG':('Compound','Gene'),'CuG':('Compound','Gene'),'CdG':('Compound','Gene'),'DaG':('Disease','Gene'),'DuG':('Disease','Gene'),'DdG':('Disease','Gene'),'GiG':('Gene','Gene'),'GcG':('Gene','Gene')}
    shape.update({m:('Gene',k) for m,k in zip(['GpBP','GpPW','GpMF','GpCC'],['BiologicalProcess','Pathway','MolecularFunction','CellularComponent'])})
    matrices={}
    for m,(rows,cols) in edges.items():
        a=sparse.csr_matrix((np.ones(len(rows)),(rows,cols)),shape=tuple(len(kinds[k]) for k in shape[m]))
        if m in ('GiG','GcG'):a=a.maximum(a.T);a.setdiag(0);a.eliminate_zeros()
        matrices[m]=a
    del edges
    W={m:weighted(a) for m,a in matrices.items()}
    C,D=len(kinds['Compound']),len(kinds['Disease']);columns=[];names=[]
    for cm in ('CbG','CuG','CdG'):
        for bridge in ('direct','GiG','GcG','GpBP','GpPW','GpMF','GpCC'):
            per_d={dm:np.zeros((C,D),dtype=np.float32) for dm in ('DaG','DuG','DdG')}
            a=W[cm]
            for start in range(0,C,16):
                aa=a[start:start+16]
                if bridge=='direct':mid=aa
                elif bridge in ('GiG','GcG'):mid=aa@W[bridge]
                else:
                    b=W[bridge]
                    diagonal=np.asarray(b.multiply(b).sum(axis=1)).ravel()
                    # Exact exclusion of repeated Gene in C-G-X-G-D.
                    mid=(aa@b)@b.T-aa@sparse.diags(diagonal)
                for dm in per_d:
                    values=(mid@W[dm].T).toarray();values[np.abs(values)<1e-12]=0
                    if np.min(values)<-1e-8:raise ValueError('Negative path counts')
                    per_d[dm][start:start+16]=np.maximum(values,0)
            for dm,values in per_d.items():columns.append(values.ravel());names.append('dwpc:'+cm+'|'+bridge+'|'+dm)
            print('feature',cm,bridge,flush=True)
    X=np.column_stack([np.repeat([np.log1p(degree[n]) for n in kinds['Compound']],D),np.tile([np.log1p(degree[n]) for n in kinds['Disease']],C)]+columns).astype(np.float32)
    names=['log_drug_degree','log_disease_degree']+names
    sc=matrices['CuG']-matrices['CdG'];sd=matrices['DuG']-matrices['DdG']
    denominator=np.sqrt(np.asarray(sc.multiply(sc).sum(axis=1)))@np.sqrt(np.asarray(sd.multiply(sd).sum(axis=1))).T
    reversal=np.divide(-(sc@sd.T).toarray(),denominator,out=np.zeros((C,D)),where=denominator>0)
    X=np.column_stack([X,reversal.ravel(),(denominator>0).ravel()]).astype(np.float32);names+=['signature_reversal','signature_available']
    y=np.zeros(C*D,dtype=np.int8);eligible=np.ones(C*D,dtype=bool)
    for u,v in pos:y[indices['Compound'][u]*D+indices['Disease'][v]]=1
    for u,v in pall:eligible[indices['Compound'][u]*D+indices['Disease'][v]]=False
    comps=sorted([sorted(c) for c in nx.connected_components(cluster)],key=lambda c:c[0]);groups={n:i for i,c in enumerate(comps) for n in c}
    sglt=[n['id'] for n in nodes if n['kind']=='Compound' and 'gliflozin' in n['name'].lower()]
    reserved={groups[n] for n in sglt};ids=np.arange(C*D);g=np.repeat([groups[n] for n in kinds['Compound']],D)
    valid=eligible & ~np.isin(g,list(reserved))
    np.savez_compressed(OUT/'features.npz',X=X,y=y,eligible=eligible,groups=g,valid=valid)
    meta=dict(nodes=len(nodes),edges=sum(counts.values()),node_counts={k:len(v) for k,v in kinds.items()},edge_counts=dict(counts),drugs=kinds['Compound'],diseases=kinds['Disease'],names={n['id']:n['name'] for n in nodes if n['kind'] in ('Compound','Disease')},feature_names=names,cluster_sizes=sorted(map(len,comps),reverse=True),sglt2_reserved=sglt,reserved_clusters=sorted(reserved),hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in RAW.glob('hetionet*')},temporal_status='2017 release-level provenance; edge-level dates unavailable in TSV; not strict temporal validation',w=.4)
    dump('data_manifest.json',meta);return X,y,eligible,g,valid,meta
