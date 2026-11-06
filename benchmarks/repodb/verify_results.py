"""Recalculate held-out Recall@10 from exported CSV, independently of model fit."""
import csv,json,math
from pathlib import Path
import numpy as np

def main():
    here=Path(__file__).resolve().parent
    out=Path((here/'LATEST_RUN.txt').read_text(encoding='utf-8'))
    report=json.loads((out/'report.json').read_text(encoding='utf-8'))
    split=json.loads((out/'split.json').read_text(encoding='utf-8'))
    assert not set(split['train_pair_rows'])&set(split['test_pair_rows'])
    grouped={}
    with (out/'heldout_rankings.csv').open(encoding='utf-8',newline='') as f:
        for r in csv.DictReader(f):grouped.setdefault(r['drug_id'],[]).append(r)
    checks=[]
    for model in report['results']:
        for track in ['all_indications','novel_relative_to_KG']:
            recalls=[]
            for drug,rows in grouped.items():
                candidates=[r for r in rows if track=='all_indications' or r['known_in_KG']=='False']
                assert all(math.isfinite(float(r[model])) for r in candidates)
                positives={r['disease_id'] for r in candidates if r['gold_label']=='1'}
                if not positives:continue
                order=sorted(candidates,key=lambda r:(-float(r[model]),r['disease_id']))
                recalls.append(len({r['disease_id'] for r in order[:10]}&positives)/len(positives))
            actual=float(np.mean(recalls));expected=report['results'][model][track]['recall@10']
            assert abs(actual-expected)<1e-12,(model,track,actual,expected)
            checks.append({'model':model,'track':track,'recall10_recalculated':actual,'positive_drugs':len(recalls)})
    result={'status':'PASS','scope':'artifact/metric integrity only, not scientific acceptance','heldout_drugs':len(grouped),'train_test_pair_overlap':0,'checks':checks}
    (out/'verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result))

if __name__=='__main__':main()
