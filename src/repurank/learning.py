"""Grouped inner-CV estimator frame. No bundled labels or fitted model."""
import numpy as np
from sklearn import __version__ as sklearn_version
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

class TrainAsinh(BaseEstimator, TransformerMixin):
    def fit(self,X,y=None):
        self.scale_=np.mean(np.abs(X),axis=0);self.scale_[self.scale_==0]=1
        return self
    def transform(self,X): return np.arcsinh(X/self.scale_)

def select_features(row,model):
    prefixes={"M0":("log_",),"M1":("log_","s_orig"),"M2":("log_","dwpc:"),"M3":("log_","dwpc:","sign:","signature_"),"M4":("log_","dwpc:","sign:","signature_")}
    if model not in prefixes: raise ValueError("Unknown model")
    selected={k:v for k,v in row.items() if k.startswith(prefixes[model])}
    if model == "M1" and "s_orig" not in selected: raise ValueError("M1 requires externally audited s_orig; do not substitute zero")
    return selected

def assert_split_integrity(rows):
    pairs=set();groups={};drug_groups={}
    for row in rows:
        if row["drug_id"] in drug_groups and drug_groups[row["drug_id"]] != row["drug_cluster"]:
            raise ValueError("Drug assigned to multiple clusters")
        drug_groups[row["drug_id"]] = row["drug_cluster"]
        pair=(row["drug_id"],row["disease_id"])
        if pair in pairs: raise ValueError("Duplicate pair")
        pairs.add(pair)
        if row["split"] not in {"train","dev","test","case_study"}: raise ValueError("Invalid split")
        if row["drug_cluster"] in groups and groups[row["drug_cluster"]]!=row["split"]: raise ValueError("Cluster leakage")
        groups[row["drug_cluster"]]=row["split"]
        if (row.get("reserved_sglt2i") or row["drug_cluster"] == "SGLT2i") and row["split"] not in {"test","case_study"}: raise ValueError("SGLT2 holdout leakage")

def fit_ranker(rows,model="M2",folds=3):
    assert_split_integrity(rows)
    train=[r for r in rows if r["split"]=="train" and r["label"] in (0,1)]
    if not train: raise ValueError("No training labels")
    if any(not r.get("temporal_verified") or r.get("feature_source")!="background_graph" for r in train):
        raise ValueError("Training requires time-verified background features, no NLP")
    X=[select_features(r["features"],model) for r in train]
    y=np.array([r["label"] for r in train]);groups=np.array([r["drug_cluster"] for r in train])
    cv=StratifiedGroupKFold(n_splits=folds,shuffle=True,random_state=42)
    splits=list(cv.split(X,y,groups))
    if any(len(set(y[a]))<2 or len(set(y[b]))<2 for a,b in splits): raise ValueError("Too few positive/negative drug groups for CV")
    penalty_options = {"penalty":"elasticnet"} if tuple(map(int, sklearn_version.split(".")[:2])) < (1,8) else {}
    clf=HistGradientBoostingClassifier(random_state=42,class_weight="balanced") if model=="M4" else LogisticRegression(solver="saga",**penalty_options,class_weight="balanced",max_iter=3000,random_state=42)
    pipeline=Pipeline([("vectorize",DictVectorizer(sparse=False)),("asinh",TrainAsinh()),("standardize",StandardScaler()),("model",clf)])
    grid={"model__max_leaf_nodes":[7,15]} if model=="M4" else {"model__C":[0.01,0.1,1.0],"model__l1_ratio":[0.1,0.5,0.9]}
    search=GridSearchCV(pipeline,grid,cv=splits,scoring="average_precision",error_score="raise",n_jobs=1)
    search.fit(X,y)
    return search

def metrics(y,scores):
    return {"average_precision":float(average_precision_score(y,scores)),"auroc":float(roc_auc_score(y,scores)) if len(set(y))>1 else None,
            "positive_prevalence":float(np.mean(y)),"calibrated_probability":False}
