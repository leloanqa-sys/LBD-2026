"""Independent repoDB benchmark on real Hetionet. Does not import or edit lbd."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import argparse, csv, gzip, hashlib, importlib.util, json, pickle, platform, sys, time, warnings
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.exceptions import ConvergenceWarning
import sklearn

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def dump(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')

def read_ontology(path, diseases):
    """Exact UMLS CUI xrefs only, no fuzzy strings or ancestor expansion."""
    mapping = defaultdict(set)
    records = path.read_text(encoding='utf-8').split('[Term]')[1:]
    for record in records:
        lines = record.splitlines()
        ids = [x[4:] for x in lines if x.startswith('id: DOID:')]
        ids += [x[8:] for x in lines if x.startswith('alt_id: DOID:')]
        targets = {f'Disease::{x.strip()}' for x in ids} & set(diseases)
        if not targets:
            continue
        for line in lines:
            if line.startswith('xref: UMLS_CUI:'):
                cui = line.split('UMLS_CUI:', 1)[1].split()[0]
                mapping[cui].update(targets)
    return dict(mapping)

def prepare_gold(raw, meta, out):
    import rdata
    import pandas as pd
    objects = rdata.read_rda(raw / 'repodb_original.RData', default_encoding='utf-8')
    frames = [v for v in objects.values() if isinstance(v, pd.DataFrame) and {'drug_id','ind_id','status'} <= set(v.columns)]
    if len(frames) != 1:
        raise ValueError(f'Expected one repoDB table, got {len(frames)}')
    frame = frames[0]
    cols = [c for c in ['drug_name','drug_id','ind_name','ind_id','NCT','status','phase','DetailedStatus'] if c in frame.columns]
    frame[cols].to_csv(out/'repodb_source.csv', index=False)
    cross = read_ontology(raw/'doid.obo', meta['diseases'])
    dump(out/'umls_to_hetionet.json', {k:sorted(v) for k,v in cross.items()})
    drugs, diseases = meta['drugs'], meta['diseases']
    di, dj = {v:i for i,v in enumerate(drugs)}, {v:i for i,v in enumerate(diseases)}
    buckets = defaultdict(list); counts = Counter(); reasons = Counter(); audit = []
    failures = {'Terminated','Withdrawn','Suspended'}
    for idx, r in enumerate(frame.to_dict('records')):
        status = str(r['status']).strip(); counts[status] += 1
        drug = 'Compound::' + str(r['drug_id']).strip()
        cui = str(r['ind_id']).strip().removeprefix('UMLS:')
        targets = cross.get(cui, set())
        if drug not in di: reason = 'drug_not_in_kg'
        elif not targets: reason = 'no_exact_cui_in_kg'
        elif len(targets) != 1: reason = 'ambiguous_cui_to_multiple_diseases'
        elif status not in failures | {'Approved'}: reason = 'unsupported_status'
        else:
            disease = next(iter(targets)); row = di[drug]*len(diseases)+dj[disease]
            buckets[row].append({'status':status,'cui':cui,'source_row':idx})
            reason = 'mapped'
        reasons[reason] += 1
        audit.append({'source_row':idx,'drug':drug,'cui':cui,'status':status,'mapping_status':reason,'targets':sorted(targets)})
    gold=[];conflicts=[]
    for row, evidence in sorted(buckets.items()):
        statuses = sorted({e['status'] for e in evidence})
        if 'Approved' in statuses and set(statuses) & failures:
            conflicts.append({'row':row,'statuses':statuses,'evidence':evidence});continue
        gold.append({'row':row,'drug_id':drugs[row//len(diseases)],'disease_id':diseases[row%len(diseases)],'label':int('Approved' in statuses),'statuses':statuses,'source_rows':[e['source_row'] for e in evidence]})
    dump(out/'mapping_audit.json',audit); dump(out/'conflicting_pairs.json',conflicts);dump(out/'gold_pairs.json',gold)
    return gold, {'raw_rows':len(frame),'status_counts':dict(counts),'mapping_rows':dict(reasons),'mapped_pairs_before_conflict':len(buckets),'conflicting_pairs_excluded':len(conflicts),'gold_pairs':len(gold),'positive_pairs':sum(r['label'] for r in gold),'failure_status_pairs':sum(1-r['label'] for r in gold),'mapped_drugs':len({r['drug_id'] for r in gold}),'mapped_diseases':len({r['disease_id'] for r in gold})}

def known_pairs(raw, meta):
    d, s = {x:i for i,x in enumerate(meta['drugs'])}, {x:i for i,x in enumerate(meta['diseases'])}
    blocked=set(); edge_counts=Counter()
    with gzip.open(raw/'hetionet-v1.0-edges.sif.gz','rt',encoding='utf-8') as f:
        for row in csv.DictReader(f,delimiter='\t'):
            edge_counts[row['metaedge']]+=1
            if row['metaedge'] in {'CtD','CpD'}:
                blocked.add(d[row['source']]*len(s)+s[row['target']])
    return blocked,dict(edge_counts)

def ranking_metrics(scores, gold_by_drug, drugs, D, blocked, groups, seed=42):
    per=[]
    for drug in drugs:
        positives={r%D for r,y in gold_by_drug.get(drug,[]) if y and r not in blocked}
        if not positives: continue
        candidates=[d for d in range(D) if drug*D+d not in blocked]
        ordered=sorted(candidates,key=lambda d:(-scores[drug,d],d))
        ranks={d:i+1 for i,d in enumerate(ordered)}
        item={'drug_index':int(drug),'group':int(groups[drug*D]),'positives':len(positives),'candidates':len(candidates),'mrr':1/min(ranks[d] for d in positives)}
        for k in [5,10,20]:
            hits=len(set(ordered[:k])&positives)
            dcg=sum(1/np.log2(i+2) for i,d in enumerate(ordered[:k]) if d in positives)
            ideal=sum(1/np.log2(i+2) for i in range(min(k,len(positives))))
            item[f'recall@{k}']=hits/len(positives);item[f'ndcg@{k}']=float(dcg/ideal)
        per.append(item)
    if not per:return {'status':'NOT_EVALUATED','positive_drugs':0}
    keys=['mrr']+[f'{m}@{k}' for m in ['recall','ndcg'] for k in [5,10,20]]
    result={k:float(np.mean([r[k] for r in per])) for k in keys}
    result.update(positive_drugs=len(per),positive_pairs=sum(r['positives'] for r in per),per_drug=per)
    # Resample drug clusters, retaining all drugs within selected clusters.
    gvals=sorted({r['group'] for r in per});rng=np.random.default_rng(seed)
    values=[]
    for _ in range(1000):
        sampled=rng.choice(gvals,len(gvals),replace=True)
        v=[r['recall@10'] for g in sampled for r in per if r['group']==g]
        values.append(float(np.mean(v)))
    result['recall@10_cluster_bootstrap_CI95']=[float(x) for x in np.quantile(values,[.025,.975])]
    result['positive_clusters']=len(gvals)
    return result

def fit_score(X,y,rows,columns):
    scale=np.mean(np.abs(X[rows][:,columns]),axis=0);scale[scale==0]=1
    scaler=StandardScaler().fit(np.arcsinh(X[rows][:,columns]/scale))
    clf=LogisticRegression(C=1.0,solver='lbfgs',class_weight='balanced',max_iter=2000,tol=1e-6,random_state=42)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always',ConvergenceWarning)
        clf.fit(scaler.transform(np.arcsinh(X[rows][:,columns]/scale)),y)
    pred=clf.predict_proba(scaler.transform(np.arcsinh(X[:,columns]/scale)))[:,1]
    return pred,{'bundle':(scale,scaler,clf),'columns':columns},[str(w.message) for w in caught if issubclass(w.category,ConvergenceWarning)]

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--reuse-features',type=Path)
    args=parser.parse_args();started=time.perf_counter()
    out=HERE/'runs'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ');out.mkdir(parents=True)
    raw=ROOT/'data/raw'
    frozen={'seed':42,'test_fraction_groups':.2,'model':'L2 logistic C=1 fixed before evaluation; no hyperparameter search','train_labels':'repoDB Approved=1, failed-status=0; unknown excluded; conflicting mapped pairs excluded','disease_mapping':'exact UMLS_CUI xref in pinned Disease Ontology; no names/ancestors','features':'degree + 63 unsigned DWPC, w=.4; CtD/CpD removed before all degrees and feature building','split':'CrC connected components from KG; disjoint train/test groups','temporal':False,'negative_semantics':'trial status proxy, not confirmed biological ineffectiveness'}
    dump(out/'protocol.json',frozen)
    if args.reuse_features:
        feature_dir=args.reuse_features.resolve()
        meta=json.loads((feature_dir/'data_manifest.json').read_text(encoding='utf-8'))
    else:
        spec=importlib.util.spec_from_file_location('repodb_features',HERE/'feature_builder_reference.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        module.RAW=raw;module.OUT=out/'features';module.OUT.mkdir()
        module.features(); feature_dir=module.OUT
        meta=json.loads((feature_dir/'data_manifest.json').read_text(encoding='utf-8'))
    for name,digest in meta['hashes'].items():
        if sha(raw/name)!=digest:raise ValueError(f'Graph hash mismatch: {name}')
    z=np.load(feature_dir/'features.npz');X=z['X'];groups=z['groups'];D=len(meta['diseases'])
    # Original Hetionet y/valid/eligible arrays are deliberately NOT read.
    gold,coverage=prepare_gold(HERE/'raw',meta,out)
    rows=np.array([r['row'] for r in gold],dtype=int);y=np.array([r['label'] for r in gold],dtype=int)
    if len(set(y))<2:raise ValueError('Need both label classes')
    train,test=next(GroupShuffleSplit(n_splits=1,test_size=.2,random_state=42).split(rows,y,groups[rows]))
    assert not set(groups[rows[train]])&set(groups[rows[test]])
    assert not set(rows[train]//D)&set(rows[test]//D)
    for split in [train,test]:
        if len(set(y[split]))<2:raise ValueError('Fixed split missing label class; no adaptive seed changes allowed')
    blocked,edge_counts=known_pairs(raw,meta)
    gold_by_drug=defaultdict(list)
    for r in gold:gold_by_drug[r['row']//D].append((r['row'],r['label']))
    test_drugs=sorted(set(rows[test]//D));predictions={};results={}
    random=np.random.default_rng(42).random(X.shape[0]);predictions['Random']=random
    for name,columns in [('M0_degree',[0,1]),('M2_DWPC',list(range(65)))]:
        pred,bundle,errs=fit_score(X,y[train],rows[train],columns);predictions[name]=pred
        with (out/(name+'.pkl')).open('wb') as f:pickle.dump(bundle,f)
        results[name]={'convergence_warnings':errs}
    # Negative control uses exactly the same split/features, shuffled TRAIN labels only.
    shuffled=np.random.default_rng(123).permutation(y[train])
    pred,_,errs=fit_score(X,shuffled,rows[train],list(range(65)));predictions['M2_train_label_shuffle']=pred
    results['M2_train_label_shuffle']={'convergence_warnings':errs}
    for name,pred in predictions.items():
        scores=pred.reshape(-1,D)
        results.setdefault(name,{})
        for track,exclude in [('all_indications',set()),('novel_relative_to_KG',blocked)]:
            eligible=np.array([i for i in test if int(rows[i]) not in exclude],dtype=int)
            m=ranking_metrics(scores,gold_by_drug,test_drugs,D,exclude,groups)
            m['labeled_test_pairs']=len(eligible)
            if len(eligible) and len(set(y[eligible]))==2:
                m['labeled_AP']=float(average_precision_score(y[eligible],pred[rows[eligible]]))
                m['labeled_AUROC']=float(roc_auc_score(y[eligible],pred[rows[eligible]]))
                m['labeled_prevalence']=float(y[eligible].mean())
            results[name][track]=m
    with (out/'heldout_rankings.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=['drug_id','drug_name','disease_id','disease_name','gold_label','known_in_KG',*predictions]);writer.writeheader()
        lookup={r['row']:r['label'] for r in gold}
        for drug in test_drugs:
            for dis in range(D):
                row=drug*D+dis;u,v=meta['drugs'][drug],meta['diseases'][dis]
                writer.writerow({'drug_id':u,'drug_name':meta['names'].get(u,u),'disease_id':v,'disease_name':meta['names'].get(v,v),'gold_label':lookup.get(row,''),'known_in_KG':row in blocked,**{k:float(s[row]) for k,s in predictions.items()}})
    split_report={name:{'pairs':len(ix),'positives':int(y[ix].sum()),'failure_status_pairs':int(len(ix)-y[ix].sum()),'drugs':len(set(rows[ix]//D)),'groups':len(set(groups[rows[ix]]))} for name,ix in [('train',train),('test',test)]}
    dump(out/'split.json',{'summary':split_report,'train_pair_rows':rows[train].tolist(),'test_pair_rows':rows[test].tolist(),'group_overlap':0,'drug_overlap':0})
    report={'status':'COMPLETED','scope':'External repoDB benchmark: newly trained ranking on real KG; not full NLP E2E or temporal validation','run_dir':str(out),'coverage':coverage,'graph':{'nodes':meta['nodes'],'edges':sum(edge_counts.values()),'drugs':len(meta['drugs']),'diseases':D,'edge_counts':edge_counts,'features_recomputed':not bool(args.reuse_features)},'split':split_report,'results':results,'runtime_seconds':time.perf_counter()-started,'limitations':['repoDB original repository snapshot, not latest hosted release','Current DO ontology maps historical KG IDs; not temporal benchmark','Failed status is a noisy proxy, not biological true negative','Only exact, unambiguous CUI mappings; unmapped rows excluded and reported','KG has 137 diseases, narrower than repoDB; ranking evaluated conditional on mapping','Novel means absent CtD/CpD, not confirmed future approved indication','Hyperparameters fixed; one grouped split, not nested CV','NLP and evidence explanation are not evaluated']}
    dump(out/'report.json',report)
    files=[HERE/'raw/repodb_original.RData',HERE/'raw/doid.obo',HERE/'raw/sources.json',HERE/'run_benchmark.py',HERE/'feature_builder_reference.py',feature_dir/'features.npz',feature_dir/'data_manifest.json',raw/'hetionet-v1.0-nodes.tsv',raw/'hetionet-v1.0-edges.sif.gz']
    dump(out/'manifest.json',{'created_utc':datetime.now(timezone.utc).isoformat(),'sources':json.loads((HERE/'raw/sources.json').read_text(encoding='utf-8-sig')),'sha256':{str(p.resolve()):sha(p) for p in files},'python':platform.python_version(),'sklearn':sklearn.__version__,'numpy':np.__version__})
    (HERE/'LATEST_RUN.txt').write_text(str(out),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='results'},ensure_ascii=True),flush=True)
    for name,r in results.items():
        print(name,json.dumps({t:{k:v for k,v in m.items() if k!='per_drug'} for t,m in r.items() if isinstance(m,dict)}),flush=True)

if __name__=='__main__':
    if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf-8')
    main()
