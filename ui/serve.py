"""Local LBD Explorer & Human-in-the-Loop Repurposing Decision Platform.
Run with: python -X utf8 ui/serve.py
"""
import argparse
import copy
import json
import math
import os
import re
import sys
import threading
import uuid
from datetime import date
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.request import Request, urlopen

UI = Path(__file__).resolve().parent

def load_env():
    for base in (UI.parent, UI.parent.parent, UI.parent.parent / "nlp"):
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

load_env()
sys.path.insert(0, str(UI.parent / "scripts"))
import run_explorer as pipeline

jobs = {}
slot = threading.Semaphore(1)
_shared_ranker = None


def get_ranker():
    global _shared_ranker
    if _shared_ranker is None:
        _shared_ranker = pipeline.M2RepurposingRanker(load_graph_for_dfs=True)
    return _shared_ranker


def execute(job, drug, limit):
    try:
        def update(stage):
            jobs[job] = {"stage": stage}
        result = pipeline.run(drug, limit, progress=update)
        jobs[job] = {"run_id": result["run_id"]}
    except Exception as exc:
        jobs[job] = {"error": str(exc)}
    finally:
        slot.release()


def result_for(key):
    if not re.fullmatch(r"[a-f0-9]{32}", key):
        raise ValueError("Run ID không hợp lệ")
    return pipeline.read(pipeline.RUNS / key / "result.json")


def get_reviews(run_id):
    if not re.fullmatch(r"[a-f0-9]{32}", run_id):
        raise ValueError("Run ID không hợp lệ")
    review_path = pipeline.RUNS / run_id / "human_reviews.json"
    if review_path.is_file():
        try:
            return json.loads(review_path.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def save_review(data):
    run_id = data.get("run_id", "")
    if not re.fullmatch(r"[a-f0-9]{32}", run_id):
        raise ValueError("Run ID không hợp lệ")
    review_path = pipeline.RUNS / run_id / "human_reviews.json"
    reviews = get_reviews(run_id)

    dis_id = data["disease_id"]
    reviews[dis_id] = {
        "disease_name": data.get("disease_name", ""),
        "status": data.get("status", "APPROVED"),
        "notes": data.get("notes", "").strip(),
        "selected_paths": data.get("selected_paths", []),
        "timestamp": data.get("timestamp") or str(date.today()),
        "reviewer": data.get("reviewer", "Chuyên gia y khoa")
    }
    review_path.write_text(json.dumps(reviews, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"status": "ok", "reviews": reviews}


def rerank(data):
    run_id = data.get("run_id", "")
    result = result_for(run_id)
    drug_id = result.get("drug_id")
    if not drug_id:
        raise ValueError("Run này không có drug_id hợp lệ trong Hetionet")

    model_mode = data.get("model_mode", "signed_flux")
    beta = float(data.get("debiasing_beta", 0.6))
    ntr_w = float(data.get("ntr_weight", 0.35))
    top_n = int(data.get("top_n", 20))

    if model_mode == "raw_m2":
        beta = 1.0
        use_signed = False
        ntr_w = 0.0
    elif model_mode == "debiased_m2":
        use_signed = False
        ntr_w = 0.0
    else:  # signed_flux
        use_signed = True

    rk = get_ranker()
    ranked = rk.rank_drug(
        drug_query=drug_id,
        top_n_novel=rk.num_diseases,
        include_dfs_paths=True,
        dfs_top_k=3,
        debiasing_beta=beta,
        use_signed_graph=use_signed,
        ntr_weight=ntr_w,
    )

    orig_cand_map = {c["disease_id"]: c for c in result.get("candidates", [])}
    reviews = get_reviews(run_id)

    updated_candidates = []
    for cand in ranked["top_novel_candidates"]:
        dis_id = cand["disease_id"]
        orig = orig_cand_map.get(dis_id, {})
        lit_delta = orig.get("literature_delta", 0.0)
        session_ev = orig.get("session_evidence_count", 0)
        session_st = orig.get("session_status", "baseline")
        ev_edges = orig.get("evidence_edges", [])
        warnings = orig.get("warnings", [])

        base_s = cand["score"]
        if lit_delta:
            clamped = min(max(base_s, 0.001), 0.999)
            base_logit = math.log(clamped / (1.0 - clamped))
            session_prob = 1.0 / (1.0 + math.exp(-(base_logit + 0.35 * lit_delta * 10.0)))
            final_score = round(session_prob, 4)
        else:
            final_score = base_s

        cand_paths = []
        for p in cand.get("top_paths", []):
            pmids = [str(x) for x in p.get("supporting_pmids", []) if x]
            cand_paths.append({
                "nodes": p["nodes"],
                "relations": [p["metapath"]],
                "path_score": float(p.get("contrib", 1.0)),
                "net_sign": int(p.get("net_sign", 0)),
                "mechanism_type": p.get("mechanism_type", "neutral"),
                "supporting_pmids": pmids,
                "evidence_sentences": [q["text"] for q in p.get("quotes", []) if isinstance(q, dict) and "text" in q]
            })

        record = {
            "rank": len(updated_candidates) + 1,
            "drug_id": drug_id,
            "disease_id": dis_id,
            "disease_name": cand["disease_name"],
            "ranking_score": final_score,
            "base_score": base_s,
            "literature_delta": lit_delta,
            "session_evidence_count": session_ev,
            "session_status": session_st,
            "signed_mechanism": cand.get("signed_mechanism", {}),
            "paths": cand_paths,
            "signed_paths": orig.get("signed_paths", []),
            "structural_paths": cand.get("top_paths", []),
            "evidence_edges": ev_edges,
            "warnings": warnings,
            "human_review": reviews.get(dis_id),
            "ai_explanation": orig.get("ai_explanation", "Chưa tạo diễn giải LLM"),
            "features": rk.compute_feature_contributions(drug_id, dis_id)["top_positive_features"]
        }
        updated_candidates.append(record)
        if len(updated_candidates) >= top_n:
            break

    return {
        "run_id": run_id,
        "model_mode": model_mode,
        "debiasing_beta": beta,
        "ntr_weight": ntr_w,
        "candidates": updated_candidates
    }


def explain(data):
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        raise ValueError("Cần cấu hình GROQ_API_KEY cho server để tạo diễn giải.")
    result = result_for(data["run_id"])
    candidate = next((x for x in result["candidates"] if x["disease_id"] == data["disease"]), None)
    if not candidate:
        raise ValueError("Ứng viên không thuộc run")

    def selection(items, indexes):
        if not isinstance(indexes, list) or any(type(i) != int or i < 0 or i >= len(items) for i in indexes):
            raise ValueError("Lựa chọn không hợp lệ")
        return [items[i] for i in dict.fromkeys(indexes)]

    paths = selection(candidate["structural_paths"], data.get("paths", []))
    quotes = [dict(predicate=edge["relation_type"], source=edge["source"], target=edge["target"], **quote)
              for edge in candidate["evidence_edges"] for quote in edge["quotes"]]
    evidence = selection(quotes, data.get("quotes", []))
    if not paths and not evidence:
        paths = candidate.get("structural_paths", [])[:3]
    if not paths and not evidence and not candidate.get("features"):
        raise ValueError("Không có thông tin đường đi hoặc đặc trưng để diễn giải")

    reviews = get_reviews(data["run_id"])
    rev_info = reviews.get(data["disease"])

    payload = dict(
        drug=result["drug"],
        disease=candidate["disease_name"],
        score=candidate["ranking_score"],
        base_score=candidate.get("base_score"),
        signed_mechanism=candidate.get("signed_mechanism"),
        literature_delta=candidate.get("literature_delta"),
        human_review=rev_info,
        warnings=candidate.get("warnings", []),
        features=candidate["features"],
        selected_paths=paths,
        selected_quotes=evidence
    )
    prompt = (
        "Bạn là chuyên gia dược lý học tính toán và tái định vị thuốc. "
        "Hãy viết báo cáo thẩm định tiếng Việt theo 4 phần mạch lạc: "
        "1. Đánh giá xếp hạng & Dòng chảy sinh học dấu (Signed Flux): phân tích Dòng chảy trị liệu đảo ngược (Therapeutic Reversal) vs Dòng chảy độc tính (Pathogenic Flux) và Chữ ký gen L1000. "
        "2. Cơ chế đường topology sinh học: phân tích vai trò các gene trung gian (đích thụ thể, enzyme chuyển hóa) từ Hetionet và y văn mới. "
        "3. Phân tích bằng chứng y văn PubMed và Cảnh báo an toàn (warnings). "
        "4. Đối chiếu Thẩm định Chuyên gia (Human-in-the-Loop review nếu có) và khuyến nghị bước tiếp theo (in vitro / in vivo). "
        "Trích dẫn đúng PMID từ selected_quotes. Không tư vấn điều trị thương mại."
    )
    body = dict(
        model=os.getenv("LBD_LLM_MODEL") or os.getenv("GROQ_MODEL") or "qwen/qwen3.8-27b",
        temperature=0,
        messages=[
            dict(role="system", content=prompt),
            dict(role="user", content=json.dumps(payload, ensure_ascii=False))
        ]
    )
    req = Request(
        "https://api.groq.com/openai/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={
            "Authorization": "Bearer " + key,
            "Content-Type": "application/json",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
        }
    )
    with urlopen(req, timeout=45) as res:
        return {"text": json.load(res)["choices"][0]["message"]["content"]}


class Handler(BaseHTTPRequestHandler):
    def send(self, status, value, html=False):
        content = value if html else json.dumps(value, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", ("text/html" if html else "application/json") + "; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(content)

    def do_GET(self):
        try:
            if self.path in ("/", "/index.html"):
                return self.send(200, (UI / "explorer.html").read_bytes(), True)
            if self.path == "/api/latest":
                files = sorted(pipeline.RUNS.glob("*/result.json"), key=lambda f: f.stat().st_mtime)
                if not files:
                    return self.send(200, {})
                res = pipeline.read(files[-1])
                revs = get_reviews(res.get("run_id", ""))
                res["human_reviews"] = revs
                for c in res.get("candidates", []):
                    c["human_review"] = revs.get(c["disease_id"])
                return self.send(200, res)
            if self.path.startswith("/api/jobs/") or self.path.startswith("/api/poll/"):
                job_id = self.path.split("/")[-1]
                return self.send(200, jobs.get(job_id, {"error": "Job không tồn tại"}))
            if self.path.startswith("/api/reviews/"):
                run_id = self.path.split("/")[-1]
                return self.send(200, get_reviews(run_id))
            if self.path.startswith("/api/runs/") or self.path.startswith("/api/result/"):
                run_id = self.path.split("/")[-1]
                res = result_for(run_id)
                revs = get_reviews(run_id)
                res["human_reviews"] = revs
                for c in res.get("candidates", []):
                    c["human_review"] = revs.get(c["disease_id"])
                return self.send(200, res)
            self.send(404, {"error": "Unknown endpoint"})
        except (OSError, ValueError, KeyError) as e:
            self.send(404, {"error": f"Lỗi truy vấn: {str(e)}"})

    def do_POST(self):
        try:
            client_ip = self.client_address[0]
            if not (client_ip in ("127.0.0.1", "::1", "localhost") or client_ip.startswith("127.")):
                return self.send(403, {"error": "Local clients only"})

            size = int(self.headers.get("Content-Length", 0))
            if not 0 < size <= 65536 or self.headers.get_content_type() != "application/json":
                return self.send(400, {"error": "Expected JSON payload"})
            data = json.loads(self.rfile.read(size))

            if self.path == "/api/explain":
                return self.send(200, explain(data))
            if self.path == "/api/rerank":
                return self.send(200, rerank(data))
            if self.path == "/api/review":
                return self.send(200, save_review(data))
            if self.path != "/api/run":
                return self.send(404, {"error": "Unknown endpoint"})

            drug = data["drug"].strip()
            limit = int(data.get("limit", 20))
            if not re.fullmatch(r"[\w ()/-]{1,90}", drug) or not 1 <= limit <= 200:
                raise ValueError("Thông số không hợp lệ")
            if not slot.acquire(blocking=False):
                return self.send(409, {"error": "Đang có một lượt chạy"})
            job = uuid.uuid4().hex
            jobs[job] = {"stage": "Chuẩn bị"}
            threading.Thread(target=execute, args=(job, drug, limit), daemon=True).start()
            self.send(202, {"job": job, "job_id": job})
        except Exception as exc:
            self.send(400, {"error": str(exc)})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    port = parser.parse_args().port
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"LBD Explorer (Human-in-the-Loop): http://127.0.0.1:{port}", flush=True)
    server.serve_forever()
