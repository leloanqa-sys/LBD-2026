"""Offline feature-audit CLI; never trains from the single-drug snapshot."""
import argparse,json
from pathlib import Path
from .graph import load_graph
from .features import pair_features

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--graph",required=True);p.add_argument("--drug",required=True)
    p.add_argument("--output",required=True);p.add_argument("--cutoff",default="2018-12-31")
    p.add_argument("--strict-temporal",action="store_true");p.add_argument("--w",type=float,default=0.4)
    args=p.parse_args();g=load_graph(args.graph,args.cutoff,args.strict_temporal)
    if args.drug not in g: raise ValueError("Drug missing; no node injection")
    records=[]
    for disease,a in g.nodes(data=True):
        if a["type"]!="Disease":continue
        records.append({"drug_id":args.drug,"disease_id":disease,"features":pair_features(g,args.drug,disease,args.w),
          "feature_source":"background_graph","temporal_verified":g.graph["temporal_unverified_edges"]==0})
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps({"mode":"reference_feature_audit_not_predictions","graph_manifest":g.graph,"pairs":records},indent=2),encoding="utf-8")
    print(json.dumps({"pairs":len(records),"output":str(out),"temporal_unverified_edges":g.graph["temporal_unverified_edges"]}))
if __name__=="__main__":main()
