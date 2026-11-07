"""Expression signs describe perturbation/response, never gene->disease causality."""
POSITIVE = {"ACTIVATES", "INCREASES", "CuG", "DuG"}
NEGATIVE = {"INHIBITS", "DECREASES", "CdG", "DdG"}

def map_sign(edge):
    relation = edge.get("metaedge", "")
    if relation not in POSITIVE | NEGATIVE:
        relation = edge.get("relation_type", "")
    return {"code": "positive" if relation in POSITIVE else "negative" if relation in NEGATIVE else "unknown",
            "negated":edge.get("assertion_status") == "negated",
            "causal":False if relation in {"DuG","DdG","GuD","GdD"} else None}

def signature_reversal(drug, disease):
    import numpy as np
    genes = sorted(set(drug) & set(disease))
    if not genes: return {"signature_reversal_cosine":0.0,"signature_available":0,"signature_opposed_fraction":0.0,"signature_same_fraction":0.0}
    a,b = np.array([drug[g] for g in genes]),np.array([disease[g] for g in genes])
    norm = np.linalg.norm(a)*np.linalg.norm(b)
    return {"signature_reversal_cosine":float(-a.dot(b)/norm) if norm else 0.0,"signature_available":int(norm>0),
            "signature_opposed_fraction":float(np.mean(a*b<0)),"signature_same_fraction":float(np.mean(a*b>0))}
