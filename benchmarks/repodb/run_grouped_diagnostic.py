"""Exploratory stability check; retains the original fixed-split result unchanged."""
import json, time
from pathlib import Path
from collections import defaultdict
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.metrics import average_precision_score, roc_auc_score
import run_benchmark as b

def main():
    start=time.perf_counter()
    out=Path((b.HERE/'LATEST_RUN.txt').read_text(encoding='utf-8'))
    manifest=json.loads((out/'manifest.json').read_text(encoding='utf-8'))
    feature_path=next(Path(p) for p in manifest['sha256'] if p.endswith('features.npz'))
    meta=json.loads(feature_path.with_name('data_manifest.json').read_text(encoding='utf-8'))
    data=np.load(feature_path);X=data['X'];groups=data['groups'];D=len(meta['diseases'])
    gold=json.loads((out/'gold_pairs.json').read_text(encoding='utf-8'))
    rows=np.array([r['row'] for r in gold]);y=np.array([r['label'] for r in gold])
    bydrug=defaultdict(list)
    for r in gold:bydrug[r['row']//D].append((r['row'],r['label']))
    blocked,_=b.known_pairs(b.ROOT/'data/raw',meta)
    names=['M0_degree','M2_DWPC','M2_train_label_shuffle']
    scores={k:np.full(X.shape[0],np.nan) for k in names};folds=[]
    b.dump(out/'diagnostic_protocol.json',{'scope':'Exploratory 5-fold grouped OOF stability check after observing fixed-split results; not independent confirmation','split':'GroupKFold 5; KG CrC components','C':1,'columns':'same as fixed run','shuffle_seed':123})
    for fold,(tr,te) in enumerate(GroupKFold(5).split(rows,y,groups[rows])):
        assert not set(groups[rows[tr]])&set(groups[rows[te]])
        test_drugs=sorted(set(rows[te]//D));target=np.concatenate([np.arange(d*D,(d+1)*D) for d in test_drugs])
        info={'fold':fold,'train_pairs':len(tr),'test_pairs':len(te),'test_drugs':len(test_drugs),'test_positive_pairs':int(y[te].sum()),'warnings':{}}
        for name in names:
            cols=[0,1] if name=='M0_degree' else list(range(65))
            labels=np.random.default_rng(123).permutation(y[tr]) if name.endswith('shuffle') else y[tr]
            pred,_,warn=b.fit_score(X,labels,rows[tr],cols)
            scores[name][target]=pred[target];info['warnings'][name]=warn
        folds.append(info);print('diagnostic fold',fold,info['test_pairs'],flush=True)
    results={}
    for name,pred in scores.items():
        assert np.all(np.isfinite(pred[rows]))
        results[name]={}
        for track,excluded in [('all_indications',set()),('novel_relative_to_KG',blocked)]:
            m=b.ranking_metrics(pred.reshape(-1,D),bydrug,sorted(bydrug),D,excluded,groups)
            ix=np.array([i for i,r in enumerate(rows) if int(r) not in excluded])
            m.update(labeled_pairs=len(ix),labeled_AP=float(average_precision_score(y[ix],pred[rows[ix]])),labeled_AUROC=float(roc_auc_score(y[ix],pred[rows[ix]])),labeled_prevalence=float(y[ix].mean()))
            results[name][track]=m
    # Paired cluster bootstrap on the same OOF drugs, not independent CIs subtraction.
    paired={}
    for track in ['all_indications','novel_relative_to_KG']:
        a={r['drug_index']:r for r in results['M2_DWPC'][track]['per_drug']}
        for other in ['M0_degree','M2_train_label_shuffle']:
            c={r['drug_index']:r for r in results[other][track]['per_drug']}
            gvals=sorted({r['group'] for r in a.values()});rng=np.random.default_rng(42);vals=[]
            for _ in range(2000):
                selected=rng.choice(gvals,len(gvals),replace=True)
                delta=[r['recall@10']-c[d]['recall@10'] for g in selected for d,r in a.items() if r['group']==g]
                vals.append(float(np.mean(delta)))
            paired[f'{track}: M2 minus {other}']={'difference':results['M2_DWPC'][track]['recall@10']-results[other][track]['recall@10'],'CI95':[float(x) for x in np.quantile(vals,[.025,.975])]}
    b.dump(out/'grouped_diagnostic.json',{'scope':'Exploratory, after fixed-split result; original test unchanged','folds':folds,'results':results,'paired_recall10':paired,'runtime_seconds':time.perf_counter()-start,'script_sha256':b.sha(Path(__file__))})
    print(json.dumps({'paired':paired,'runtime':time.perf_counter()-start}),flush=True)

if __name__=='__main__':main()
