"""Collect MED/PubMed abstracts via Europe PMC. Run with --help for sampling options."""
import argparse
import csv
import hashlib
import html
import json
import random
import re
import tempfile
import time
from datetime import date
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

API = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
FIELDS = "pmid print_year print_date title journal doi link abstract".split()


def terms(value):
    items = list(dict.fromkeys(s.strip() for s in value.split(",") if s.strip()))
    if not items or any(not re.fullmatch(r"[\w ()/-]+", s) for s in items):
        raise argparse.ArgumentTypeError("Use comma-separated drug or disease names.")
    return items


def fetch(query, size):
    url = API + "?" + urlencode(dict(query=query, resultType="core", format="json", pageSize=size))
    for attempt in range(3):
        try:
            with urlopen(Request(url, headers={"User-Agent": "LBD-abstract-collector/1.0"}), timeout=35) as response:
                return json.load(response)
        except (URLError, TimeoutError) as error:
            if isinstance(error, HTTPError) and error.code < 500 and error.code != 429:
                raise
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)


def clean(text):
    return " ".join(html.unescape(re.sub(r"<[^>]+>", " ", text or "")).split())


def collect(args, folder):
    records = {}
    membership = {}
    groups = []
    for drug in args.drugs:
        diseases = ([None] if args.mode != "targeted" else []) + (
            args.diseases if args.mode != "broad" else [])
        for disease in diseases:
            key = drug + (" / " + disease if disease else " / broad")
            q = (f'SRC:MED AND HAS_ABSTRACT:Y AND TITLE_ABS:"{drug}" '
                 f'AND FIRST_PDATE:[{args.start} TO {args.cutoff}]')
            if disease:
                q += f' AND TITLE_ABS:"{disease}"'
            result = fetch(q, args.pool)
            candidates = {}
            for item in result["resultList"]["result"]:
                pmid = str(item.get("pmid") or "")
                dt = item.get("firstPublicationDate", "")
                abstract = clean(item.get("abstractText"))
                if not (pmid.isdigit() and abstract and dt and args.start.isoformat() <= dt <= args.cutoff.isoformat()):
                    continue
                candidates[pmid] = dict(pmid=pmid, print_year=dt[:4], print_date=dt,
                    title=clean(item.get("title")), journal=item.get("journalInfo", {}).get("journal", {}).get("title",""),
                    doi=item.get("doi",""), link=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/", abstract=abstract)
            ids = sorted(candidates)
            selected = random.Random(f"{args.seed}|{q}").sample(ids, min(len(ids), args.per_group))
            for pmid in selected:
                records[pmid] = candidates[pmid]
                membership.setdefault(pmid, []).append(key)
            groups.append(dict(group=key, drug=drug, disease=disease, query=q, hits=result["hitCount"],
                               eligible_in_first_page=len(ids), selected_pmids=selected))
            print(f"{key}: {len(selected)} selected", flush=True)
            time.sleep(0.2)
    if not records:
        raise ValueError("No eligible abstracts matched.")
    path = folder / "abstracts.csv"
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(records[k] for k in sorted(records, key=int))
    manifest = dict(api=API, source="MED", sampling="Seeded sample from the first relevance-ranked API page per query",
                    date_field="firstPublicationDate (exported as print_date for CSV compatibility)",
                    run_date=date.today().isoformat(), seed=args.seed, pool=args.pool, quota=args.per_group,
                    unique_pmids=len(records), csv_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    groups=groups, membership=membership)
    (folder / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--drugs", type=terms, default="dapagliflozin,metformin,aspirin,methotrexate")
    p.add_argument("--diseases", type=terms, default="type 2 diabetes,heart failure,rheumatoid arthritis,colorectal cancer")
    p.add_argument("--mode", choices=["broad","targeted","both"], default="both")
    p.add_argument("--start", type=date.fromisoformat, default=date(2000,1,1))
    p.add_argument("--cutoff", type=date.fromisoformat, default=date(2018,12,31))
    p.add_argument("--per-group", type=int, default=5)
    p.add_argument("--pool", type=int, default=40)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=Path, required=True, help="New output directory; existing directories are protected")
    a = p.parse_args()
    if a.start > a.cutoff or not 1 <= a.per_group <= a.pool <= 1000 or a.out.exists():
        p.error("Require start <= cutoff, 1 <= quota <= pool <= 1000, and a new output directory.")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=a.out.parent) as tmp:
        folder = Path(tmp) / "result"
        folder.mkdir()
        collect(a, folder)
        folder.rename(a.out)
    print(a.out / "abstracts.csv")


if __name__ == "__main__":
    main()
