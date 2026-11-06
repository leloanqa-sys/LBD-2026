"""Frozen original LBD M2 inference on repoDB; no fit, tuning or model writes."""
import csv,json,pickle,sys,time
from pathlib import Path
from datetime import datetime,timezone
from collections import defaultdict,Counter
import numpy as np
from sklearn.metrics import average_precision_score,roc_auc_score
import run_benchmark as b

def infer(path,X):
    bundle=pickle.loads(path.read_bytes())
    scale,scaler,clf=bundle['bundle'];columns=list(bundle['columns'])
    return clf.predict_proba(scaler.transform(np.arcsinh(X[:,columns]/scale)))[:,1]

def main():
    start=time.perf_counter();base=Path((b.HERE/'LATEST_RUN.txt').read_text(encoding='utf-8'))
    out=b.HERE/'original_checkpoint_runs'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ');out.mkdir(parents=True)
    assets=b.ROOT/'artifacts/ranking_m2';raw=b.ROOT/'data/raw'
    model=assets/'M2_model.pkl';before=b.sha(model)
    meta=json.loads((assets/'data_manifest.json').read_text(encoding='utf-8'))
    z=np.load(assets/'features.npz');X=z['X'];groups=z['groups'];original_y=z['y'];D=len(meta['diseases'])
    for filename,digest in meta['hashes'].items():assert b.sha(raw/filename)==digest,filename
    # Ensure repoDB row IDs refer to exactly the same drug/disease ordering.
    source_manifest=json.loads((base/'manifest.json').read_text(encoding='utf-8'))
    old_feature=next(Path(p) for p in source_manifest['sha256'] if p.endswith('data_manifest.json'))
    old_meta=json.loads(old_feature.read_text(encoding='utf-8'))
    assert meta['drugs']==old_meta['drugs'] and meta['diseases']==old_meta['diseases']
    assert meta['feature_names']==old_meta['feature_names']
    gold=json.loads((base/'gold_pairs.json').read_text(encoding='utf-8'))
    split=np.load(assets/'split_indices.npz');dev=set(map(int,split['development']));locked=set(map(int,split['locked_test']))
    devgroups=set(groups[list(dev)]);testgroups=set(groups[list(locked)])
    assert not devgroups&testgroups
    devdrugs={r//D for r in dev};testdrugs={r//D for r in locked}
    blocked,edge_counts=b.known_pairs(raw,meta)
    gold_by_drug=defaultdict(list)
    for r in gold:
        assert r['drug_id']==meta['drugs'][r['row']//D] and r['disease_id']==meta['diseases'][r['row']%D]
        gold_by_drug[r['row']//D].append((r['row'],r['label']))
    all_drugs=set(gold_by_drug)
    subsets={'all_mapped':all_drugs,'original_development_drugs':all_drugs&devdrugs,'original_locked_test_drugs':all_drugs&testdrugs,'all_groups_unseen_in_fit':{d for d in all_drugs if groups[d*D] not in devgroups}}
    predictions={'Original_M2':infer(model,X),'Original_M0_degree':infer(assets/'M0_model.pkl',X),'Random':np.random.default_rng(42).random(X.shape[0])}
    assert all(np.all(np.isfinite(x)) for x in predictions.values())
    results={};subset_summary={}
    for subset,drugs in subsets.items():
        pairs=[r for r in gold if r['row']//D in drugs]
        subset_summary[subset]={'drugs':len(drugs),'groups':len({int(groups[d*D]) for d in drugs}),'pairs':len(pairs),'positive_pairs':sum(r['label'] for r in pairs),'failure_status_pairs':sum(1-r['label'] for r in pairs),'overlap_fit_groups':len({groups[d*D] for d in drugs}&devgroups)}
        results[subset]={}
        for name,pred in predictions.items():
            results[subset][name]={}
            for track,exclude in [('all_indications',set()),('novel_relative_to_KG',blocked)]:
                m=b.ranking_metrics(pred.reshape(-1,D),gold_by_drug,sorted(drugs),D,exclude,groups)
                labeled=[r for r in pairs if r['row'] not in exclude]
                yy=np.array([r['label'] for r in labeled]);ss=np.array([pred[r['row']] for r in labeled])
                m['labeled_pairs']=len(labeled)
                if len(set(yy))==2:
                    m.update(labeled_AP=float(average_precision_score(yy,ss)),labeled_AUROC=float(roc_auc_score(yy,ss)),labeled_prevalence=float(yy.mean()))
                results[subset][name][track]=m
    exposure=Counter();conflicts=[]
    for r in gold:
        row=r['row'];state='FIT_PAIR' if row in dev else 'LOCKED_TEST_PAIR' if row in locked else 'EXCLUDED_OR_RESERVED_PAIR'
        exposure[state]+=1
        if row in dev and int(original_y[row])!=r['label']:
            conflicts.append({**r,'original_training_label':int(original_y[row])})
    b.dump(out/'training_label_disagreements.json',conflicts)
    with (out/'repoDB_predictions.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['drug_id','disease_id','label','known_in_KG','original_pair_exposure','original_training_label',*predictions]);writer.writeheader()
        for r in gold:
            row=r['row'];writer.writerow({'drug_id':r['drug_id'],'disease_id':r['disease_id'],'label':r['label'],'known_in_KG':row in blocked,'original_pair_exposure':'FIT_PAIR' if row in dev else 'LOCKED_TEST_PAIR' if row in locked else 'EXCLUDED_OR_RESERVED_PAIR','original_training_label':int(original_y[row]) if row in dev else '',**{name:float(s[row]) for name,s in predictions.items()}})
    with (out/'rankings_all_mapped_drugs.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['drug_id','disease_id','gold_label','known_in_KG',*predictions]);writer.writeheader()
        labels={r['row']:r['label'] for r in gold}
        for drug in sorted(all_drugs):
            for disease in range(D):
                row=drug*D+disease;writer.writerow({'drug_id':meta['drugs'][drug],'disease_id':meta['diseases'][disease],'gold_label':labels.get(row,''),'known_in_KG':row in blocked,**{name:float(s[row]) for name,s in predictions.items()}})
    # Compare scores with the actual production ranker API; graph filtering is
    # performed independently above to avoid conflating DFS loading with novelty.
    sys.path.insert(0,str(b.ROOT/'src'))
    from lbd.ranking.m2_ranker import M2RepurposingRanker
    ranker=M2RepurposingRanker(load_graph_for_dfs=False)
    checks=[]
    for query in ['Dapagliflozin','Aspirin','Colchicine']:
        response=ranker.rank_drug(query,top_n_novel=D,include_dfs_paths=False)
        drug=meta['drugs'].index(response['drug_id']);lookup={x['disease_id']:x for x in response['top_novel_candidates']}
        delta=max(abs(lookup[d]['score']-predictions['Original_M2'][drug*D+i]) for i,d in enumerate(meta['diseases']))
        assert delta<=.00005001
        checks.append({'drug':query,'diseases_compared':D,'maximum_difference_due_to_API_rounding':float(delta)})
    dapa=meta['drugs'].index('Compound::DB06292');case={}
    for track,excluded in [('all_indications',set()),('novel_relative_to_KG',blocked)]:
        order=sorted([d for d in range(D) if dapa*D+d not in excluded],key=lambda d:(-predictions['Original_M2'][dapa*D+d],d))
        case[track]=[{'rank':i+1,'disease_id':meta['diseases'][d],'disease_name':meta['names'][meta['diseases'][d]],'score':float(predictions['Original_M2'][dapa*D+d])} for i,d in enumerate(order)]
    assert b.sha(model)==before
    report={'status':'COMPLETED','scope':'Frozen user checkpoint inference; no retraining, fitting, threshold or hyperparameter changes','model_sha256_before':before,'model_sha256_after':b.sha(model),'repoDB_source_run':str(base),'graph':{'nodes':meta['nodes'],'edges':sum(edge_counts.values()),'diseases':D},'subsets':subset_summary,'pair_exposure':dict(exposure),'label_disagreements_on_fit_pairs':len(conflicts),'results':results,'production_API_checks':checks,'dapagliflozin':case,'runtime_seconds':time.perf_counter()-start,'limitations':['All-mapped score includes drugs and labels used to fit original checkpoint; not independent validation','Original locked-test groups not used for fitting, but original experiment already inspected their test results','repoDB and original CtD labels may share upstream sources','Only exact mapped subset, 636 pairs; failed trial statuses are imperfect negative proxies','Not temporal or full NLP E2E validation','Classifier scores are not calibrated clinical probabilities']}
    b.dump(out/'report.json',report)
    files=[model,assets/'M0_model.pkl',assets/'features.npz',assets/'data_manifest.json',assets/'split_indices.npz',base/'gold_pairs.json',base/'manifest.json',raw/'hetionet-v1.0-nodes.tsv',raw/'hetionet-v1.0-edges.sif.gz',Path(__file__),b.HERE/'run_benchmark.py',b.ROOT/'src/lbd/ranking/m2_ranker.py']
    b.dump(out/'manifest.json',{'sha256':{str(p.resolve()):b.sha(p) for p in files},'fit_called':False,'source_model_modified':False,'created_utc':datetime.now(timezone.utc).isoformat()})
    (b.HERE/'LATEST_ORIGINAL_CHECKPOINT_RUN.txt').write_text(str(out),encoding='utf-8')
    print(json.dumps({'run':str(out),'subsets':subset_summary,'pair_exposure':dict(exposure),'training_label_disagreements':len(conflicts),'API_checks':checks,'runtime':report['runtime_seconds']}),flush=True)
    for subset,models in results.items():
        print(subset,json.dumps({k:{track:{metric:s.get(metric) for metric in ['recall@5','recall@10','recall@20','mrr','labeled_AP','labeled_AUROC','positive_drugs','positive_pairs','recall@10_cluster_bootstrap_CI95']} for track,s in v.items()} for k,v in models.items()}),flush=True)

if __name__=='__main__':main()
