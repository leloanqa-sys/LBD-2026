"""Run-keyed, offline-capable literature evidence + frozen M2 ranking pipeline."""
import argparse
import csv
import hashlib
import math
import json
import os
import re
import sys
import uuid
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import networkx as nx

import collect_multidrug_abstracts as collector
import filter_huy as huy

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs" / "explorer"

for base in (ROOT, ROOT.parent, ROOT.parent / "nlp"):
    env_file = base / ".env"
    if env_file.is_file():
        try:
            for line in env_file.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    k = k.strip()
                    v = v.strip().strip('"').strip("'")
                    if k and k not in os.environ:
                        os.environ[k] = v
        except Exception:
            pass
sys.path.insert(0, str(ROOT / "src"))
from lbd.nlp.engine import NLPEngine
from lbd.ranking.m2_ranker import M2RepurposingRanker


def save(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def lines(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


FIELDS = ("relation_id pmid source_entity target_entity source_standard_id target_standard_id "
          "source_type target_type source_raw_text target_raw_text canonical_relation relation_type "
          "confidence_score sign assertion_status normalization_status evidence metaedge section_zone "
          "study_design section_weight study_weight").split()


def write_bundle(folder, papers, nodes, drug_id=None, drug_term=None):
    if drug_id and drug_term:
        from lbd.nlp import zoning
        clean_id = drug_id.split("::")[-1]
        existing = [x for x in zoning.ONTOLOGY_REGISTRY if x.get("id") == clean_id]
        if not existing:
            patterns = [rf"\b{re.escape(drug_term.lower())}\b"]
            entry = {
                "id": clean_id,
                "canonical_name": drug_term,
                "type": "Compound",
                "valence": 0,
                "is_primary_drug": True,
                "patterns": patterns,
                "compiled_rx": [re.compile(p, re.IGNORECASE) for p in patterns]
            }
            zoning.ONTOLOGY_REGISTRY.insert(0, entry)

    from lbd.nlp.external_annotations import fetch_pubtator
    pmids = [p["pmid"] for p in papers if p.get("pmid")]
    cache_root = ROOT / "data" / "cache"
    pubtator_map = {}
    try:
        pubtator_map, _ = fetch_pubtator(pmids, cache_root, offline=False)
    except Exception:
        pass
    engine = NLPEngine.load_default(enable_live_llm=False)
    engine.events.cache = {}
    candidates, mentions, sentences = [], [], []
    for paper in papers:
        pmid = paper["pmid"]
        ext_entities = pubtator_map.get(pmid, [])
        grounded, events, _ = engine.run_document(paper, external_entities=ext_entities)
        for sent in grounded["sentences"]:
            entities = []
            for ent in sent.entities:
                start, end = ent["span_start"], ent["span_end"]
                raw = sent.text[start:end]
                if raw != ent["matched_text"] or not raw:
                    continue
                entities.append(ent)
                mentions.append(dict(pmid=pmid, mention_text=raw, entity_id=ent["id"], entity_type=ent["type"],
                                     offset_status="verified_exact", sentence_index=sent.sent_idx, start=start, end=end))
            sentences.append(dict(pmid=pmid,sentence=sent.text,entities=entities))
        evidence_lookup = {s.text:s for s in grounded["sentences"]}
        for event in events:
            for i,c in enumerate(event.projected_edges):
                sentence = evidence_lookup.get(c.evidence_sentence)
                def argument(identifier):
                    matches = {v["matched_text"] for v in sentence.entities if v["id"] == identifier} if sentence else set()
                    return next(iter(matches)) if len(matches)==1 else ""
                src, tgt = argument(c.source_id), argument(c.target_id)
                def verified(kind, identifier, raw):
                    return bool(raw and nodes.get(f"{kind}::{identifier}", "").casefold() == raw.casefold())
                candidates.append(dict(relation_id=f"{event.event_id}_{i}",pmid=pmid,source_entity=c.source_name,
                    target_entity=c.target_name,source_standard_id=c.source_id,target_standard_id=c.target_id,
                    source_type=c.source_type,target_type=c.target_type,source_raw_text=src,target_raw_text=tgt,
                    canonical_relation=c.canonical_relation,relation_type=c.contract_b_relation,
                    confidence_score=c.confidence_score,sign=c.hetionet_sign,assertion_status=c.assertion_status,
                    normalization_status="verified" if verified(c.source_type,c.source_id,src) and verified(c.target_type,c.target_id,tgt) else "unresolved",
                    evidence=c.evidence_sentence,metaedge=c.hetionet_metaedge,section_zone=c.section_zone,
                    study_design=c.study_design,section_weight=c.section_weight,study_weight=c.study_weight))
    huy.csv_dump(folder/"relations_contract_b.csv",candidates,FIELDS)
    huy.csv_dump(folder/"entity_mentions.csv",mentions,"pmid mention_text entity_id entity_type offset_status sentence_index start end".split())
    huy.jsonl_dump(folder/"grounded_sentences.jsonl",sentences)
    save(folder/"run_report.json",dict(documents=len(papers),candidates=len(candidates)))


def run(drug, limit=20, source=None, progress=print):
    drug = drug.strip()
    if not re.fullmatch(r"[\w ()/-]{1,90}",drug) or not 1 <= limit <= 1000:
        raise ValueError("Tên thuốc hoặc giới hạn bài không hợp lệ")
    raw = ROOT/"data/raw"
    assets = ROOT/"artifacts/ranking_m2"
    with (raw/"hetionet-v1.0-nodes.tsv").open(encoding="utf-8") as f:
        nodes = {r["id"]:r["name"] for r in csv.DictReader(f,delimiter="\t")}
    matches = [k for k,v in nodes.items() if k.startswith("Compound::") and drug.casefold() in (k.casefold(),k.split("::")[1].casefold(),v.casefold())]
    if len(matches)>1:
        raise ValueError("Tên chưa duy nhất. Dùng DrugBank ID: "+", ".join(matches))
    drug_id = matches[0] if matches else None
    term = nodes.get(drug_id,drug)
    folder = RUNS/uuid.uuid4().hex
    folder.mkdir(parents=True)
    progress("Thu thập abstract")
    if source is None:
        source = folder/"collection"
        source.mkdir()
        collector.collect(SimpleNamespace(drugs=[term], diseases=[],mode="broad",start=date(2000,1,1),
            cutoff=date(2018,12,31),pool=max(limit,40),per_group=limit,seed=42),source)
        source = source/"abstracts.csv"
    rejected=[]
    papers=[]
    seen=set()
    with Path(source).open(encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            try:
                dt=huy.date_upper_bound(row["print_date"])
            except (ValueError, KeyError):
                rejected.append(dict(pmid=row.get("pmid"),reason="invalid_date"))
                continue
            if dt>date(2018,12,31) or not row["abstract"].strip() or row["pmid"] in seen:
                rejected.append(dict(pmid=row["pmid"],reason="cutoff_empty_or_duplicate"))
                continue
            if not row["pmid"].isdigit(): raise ValueError("Invalid PMID")
            seen.add(row["pmid"])
            papers.append(dict(row, document_id="PMID:"+row["pmid"],publication_date=dt.isoformat(),
                               study_type="unknown",dataset_version=folder.name))
            if len(papers)>=limit:break
    save(folder/"collection_rejected.json",rejected)
    if not papers: raise ValueError("Không có abstract đạt điều kiện.")
    checks=huy.validators()
    for p in papers:checks["paper.schema.json"].validate(p)
    huy.jsonl_dump(folder/"papers.jsonl",papers)
    progress("NLP và đối chiếu định danh")
    bundle=folder/"bundle";bundle.mkdir()
    write_bundle(bundle, papers, nodes, drug_id=drug_id, drug_term=term)
    progress("Lọc bằng chứng với chính sách Huy pilot")
    filtered=folder/"filtered"
    audit=huy.run(bundle,filtered,folder/"papers.jsonl",ROOT/"configs/kg_filter.json")
    graph=read(filtered/"kg_graph.json")
    graph["metadata"]["source"]="canonical_nlp_filtered"
    save(filtered/"kg_graph.json",graph)
    warnings_by_pair={}
    for record in lines(filtered/"relations_review.jsonl"):
        if record.get("canonical_relation")=="CONTRAINDICATED_FOR" and record.get("evidence_origin")=="abstract":
            pair=record["source_standard_id"],record["target_standard_id"]
            warnings_by_pair.setdefault(pair,[]).append(dict(pmid=record["pmid"],text=record["evidence"],
                status="auto_detected_requires_review"))
    evidence = {}
    for f in (filtered / "relations_kg_ready.jsonl", filtered / "relations_review.jsonl"):
        if f.exists():
            for r in lines(f):
                evidence[r["relation_id"]] = r
    progress("Xếp hạng M2 và trích đường đi")
    literature = nx.MultiDiGraph()
    for edge in graph["edges"]:
        if edge.get("partition") != "ADMITTED_KG" or edge.get("assertion_status") != "asserted":
            raise ValueError("Ineligible edge in active literature graph")
        a=edge["source_type"]+"::"+edge["source"]
        b=edge["target_type"]+"::"+edge["target"]
        literature.add_edge(a,b,key=edge["edge_id"],record=edge)
    ranking=[]
    session_edges_count = 0
    literature_boost_applied = False
    if drug_id:
        manifest=read(assets/"data_manifest.json")
        for file,expected in manifest["hashes"].items():
            if digest(raw/file)!=expected:raise ValueError("Graph hash mismatch: "+file)
        ranker=M2RepurposingRanker()
        if not ranker._graph_loaded:raise ValueError("Không tải được KG cho M2")

        # In-Memory Session Graph Augmentation (Zero Disk Mutation, Fail-Closed: ADMITTED_KG only)
        session_ranker = ranker.clone_for_session()
        admitted_kg_edges = [
            e for e in graph.get("edges", [])
            if e.get("partition") == "ADMITTED_KG" and e.get("assertion_status") == "asserted"
        ]
        session_edges = [
            dict(
                e,
                quotes=[
                    dict(pmid=evidence[i]["pmid"], text=evidence[i]["evidence"])
                    for i in e.get("evidence_ids", [])
                    if i in evidence
                ],
            )
            for e in admitted_kg_edges
        ]
        if session_edges:
            session_edges_count = session_ranker.augment_session_edges(session_edges)
        else:
            session_edges_count = 0
        literature_boost_applied = bool(session_edges_count > 0)

        result = session_ranker.rank_drug_with_session(drug_id, top_n_novel=25, dfs_top_k=3)
        known_indications_list = []
        for k in result.get("known_indications", []):
            known_indications_list.append({
                "disease_id": k["disease_id"],
                "disease_name": k["disease_name"],
                "score": k["score"],
                "status": k.get("status", "KNOWN_TREATS")
            })

        for c in result["top_novel_candidates"]:
            target=c["disease_id"];paths=c.get("top_paths",[])
            selected={drug_id,target}|{n for path in paths for n in path["nodes"]}
            signed_paths=[]
            if drug_id in literature and target in literature:
                for route in nx.all_simple_edge_paths(literature,drug_id,target,cutoff=4):
                    if len(signed_paths)>=20:break
                    steps=[literature.edges[item]["record"] for item in route]
                    ids=[drug_id]+[item[1] for item in route]
                    selected.update(ids)
                    signed_paths.append(dict(nodes=ids,relations=[x["relation_type"] for x in steps],
                        net_sign=math.prod(x["sign"] for x in steps),
                        edge_ids=[x["edge_id"] for x in steps]))
            edges = []
            seen_edges = set()
            for edge in graph.get("edges", []):
                a=edge["source_type"]+"::"+edge["source"];b=edge["target_type"]+"::"+edge["target"]
                if a in selected and b in selected:
                    seen_edges.add((a, b, edge.get("relation_type")))
                    edges.append(dict(edge, quotes=[dict(pmid=evidence[i]["pmid"], text=evidence[i]["evidence"]) for i in edge["evidence_ids"] if i in evidence]))
            review_file = filtered / "relations_review.jsonl"
            if review_file.exists():
                dis_name = c["disease_name"].casefold()
                for r in lines(review_file):
                    s_type = r.get("source_type", "Compound")
                    t_type = r.get("target_type", "Disease")
                    s_id = f"{s_type}::{r.get('source_standard_id') or r.get('source_entity')}"
                    t_id = f"{t_type}::{r.get('target_standard_id') or r.get('target_entity')}"
                    s_name = (r.get("source_entity") or "").casefold()
                    t_name = (r.get("target_entity") or "").casefold()
                    # Exact pathway-specific or direct indication matching
                    rel_match = (
                        (s_id == drug_id and t_id == target) or
                        (s_id == target and t_id == drug_id) or
                        (s_id == drug_id and t_id in selected and t_id != target) or
                        (t_id == target and s_id in selected and s_id != drug_id)
                    )
                    if rel_match:
                        edge_key = (s_id, t_id, r.get("relation_type"))
                        if edge_key not in seen_edges and len(edges) < 15:
                            seen_edges.add(edge_key)
                            edges.append({
                                "source": r.get("source_standard_id") or r.get("source_entity"),
                                "source_name": r.get("source_entity"),
                                "source_type": s_type,
                                "target": r.get("target_standard_id") or r.get("target_entity"),
                                "target_name": r.get("target_entity"),
                                "target_type": t_type,
                                "relation_type": r.get("relation_type", "ASSOCIATED_WITH"),
                                "sign": r.get("sign", 0),
                                "partition": "REVIEW",
                                "quotes": [{"pmid": r["pmid"], "text": r["evidence"]}]
                            })

            candidate_paths = []
            for p in paths:
                pmids = [str(x) for x in p.get("supporting_pmids", []) if x]
                quotes = [q["text"] for q in p.get("quotes", []) if isinstance(q, dict) and "text" in q]
                candidate_paths.append({
                    "nodes": p["nodes"],
                    "relations": [p["metapath"]],
                    "path_score": float(p.get("contrib", 1.0)),
                    "net_sign": int(p.get("net_sign", 0)),
                    "mechanism_type": p.get("mechanism_type", "neutral"),
                    "supporting_pmids": pmids,
                    "evidence_sentences": quotes
                })

            record=dict(rank=c["rank_novel"],drug_id=drug_id,disease_id=target,disease_name=c["disease_name"],
                ranking_score=c["score"],base_score=c.get("base_score", c["score"]),
                literature_delta=c.get("literature_delta", 0.0),
                session_evidence_count=c.get("session_evidence_count", 0),
                session_status=c.get("session_status", "baseline"),
                signed_mechanism=c.get("signed_mechanism", {}),
                paths=candidate_paths,signed_paths=signed_paths,structural_paths=paths,evidence_edges=edges,
                warnings=warnings_by_pair.get((drug_id.split("::",1)[1],target.split("::",1)[1]),[]),
                ai_explanation="Chưa tạo diễn giải LLM",features=ranker.compute_feature_contributions(drug_id,target)["top_positive_features"])
            checks["candidate.schema.json"].validate(record)
            ranking.append(record)

        # Snapshot session graph before in-memory cleanup
        session_graph_data = {
            "session_id": folder.name,
            "drug": term,
            "drug_id": drug_id,
            "papers_analyzed": len(papers),
            "session_edges_added": session_edges_count,
            "admitted_kg_edges": len(graph["edges"]),
            "admitted_nlp_edges": len(graph["edges"]),
            "literature_boost_applied": literature_boost_applied,
            "status": "ephemeral_session_only_discarded_after_run",
            "edges": [
                {
                    "source": k[0],
                    "metaedge": k[1],
                    "target": k[2],
                    "relation_type": v["relation_type"],
                    "confidence": v["confidence"],
                    "origin": "session_augmented",
                    "pmids": [q["pmid"] for q in v.get("quotes", []) if q.get("pmid")]
                }
                for k, v in getattr(session_ranker, "lit_edges", {}).items()
            ]
        }
        save(folder / "session_graph.json", session_graph_data)
        del session_ranker

    output=dict(run_id=folder.name,drug=term,drug_id=drug_id,papers=len(papers),filter_counts=audit["relation_decisions"],
        admitted_edges=len(graph["edges"]),admitted_kg_edges=len(graph["edges"]),
        session_edges_count=session_edges_count,literature_boost_applied=literature_boost_applied,
        known_indications=known_indications_list if drug_id else [],
        candidates=ranking,
        rank_status="ranked" if drug_id else "outside_model",
        source_csv_sha256=digest(Path(source)),policy_sha256=digest(ROOT/"configs/kg_filter.json"),
        model_sha256=digest(assets/"M2_model.pkl") if drug_id else None)
    files=[ROOT/"configs/kg_filter.json",ROOT/"scripts/filter_huy.py",Path(__file__),
           ROOT/"src/lbd/nlp/checkpoints/model_v3_biored_multiclass.pkl",folder/"papers.jsonl",
           *sorted((ROOT/"contracts").glob("*.schema.json")),*sorted(bundle.iterdir()),filtered/"kg_graph.json"]
    if drug_id:files.extend([assets/"features.npz",assets/"M2_model.pkl",assets/"data_manifest.json"])
    save(folder/"manifest.json",dict(run_id=folder.name,source=str(source),source_sha256=output["source_csv_sha256"],
         cutoff="2018-12-31",ranking_graph="Hetionet v1.0",literature_graph="filtered/kg_graph.json",
         hashes={str(f):digest(f) for f in files}))
    save(folder/"result.json",output)
    return output


if __name__=="__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    parser=argparse.ArgumentParser()
    parser.add_argument("--drug",required=True)
    parser.add_argument("--max-papers",type=int,default=20)
    parser.add_argument("--papers",type=Path)
    a=parser.parse_args()
    print(run(a.drug,a.max_papers,a.papers)["run_id"])
