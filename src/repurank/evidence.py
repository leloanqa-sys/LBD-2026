"""NLP evidence is attached after ranking, never used by fit_ranker."""
from datetime import date

def attach_evidence(candidate,records,cutoff):
    support=[];conflict=[];gaps=[]
    for record in records:
        if (record.get("source_standard_id"),record.get("target_standard_id")) != (candidate["drug_id"],candidate["disease_id"]): continue
        available=record.get("available_at")
        if not available:
            gaps.append("evidence_time_unverified");continue
        date.fromisoformat(available)
        if available>cutoff: continue
        if not record.get("pmid") or not record.get("evidence") or record["evidence"] not in record.get("source_abstract",""):
            gaps.append("missing_or_nonverbatim_evidence");continue
        item={k:record.get(k) for k in ("relation_id","pmid","evidence","assertion_status","relation_type")}
        if record.get("assertion_status")=="negated" or record.get("relation_type") in {"CAUSES","CONTRAINDICATED_FOR"}: conflict.append(item)
        elif record.get("assertion_status")=="asserted" and record.get("relation_type") in {"TREATS","PALLIATES","PREVENTS"}: support.append(item)
        else: gaps.append("uncertain_or_indirect_evidence")
    return {**candidate,"supporting_evidence":support,"contradicting_evidence":conflict,"evidence_gaps":sorted(set(gaps)),"ai_explanation":None}
