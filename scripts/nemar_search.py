#!/usr/bin/env python3
"""Search NEMAR/OpenNeuro datasets WITHOUT an NSG job.

NEMAR is a daily mirror of OpenNeuro, so anything OpenNeuro lists is on Expanse at
$NEMARPATH/<ds-id>. That means discovery can run locally against OpenNeuro's public GraphQL API in
seconds, instead of burning an NSG job that waits hours in the Expanse queue.

    python scripts/nemar_search.py --kw "motor imagery" --kw ASSR --modality eeg
    python scripts/nemar_search.py --task-kw "assr|steady|click" --min-subjects 20
    python scripts/nemar_search.py --kw schizophrenia --modality meg --json out.json

Matching is case-insensitive regex over the dataset Name, and (with --task-kw) over BIDS task
labels, which catches paradigms whose title never mentions them.
"""
import argparse
import json
import re
import sys
import urllib.request

URL = "https://openneuro.org/crn/graphql"
Q = ('{ datasets(first: 100%s) { pageInfo { hasNextPage endCursor } edges { node { id '
     'latestSnapshot { description { Name } summary { modalities tasks subjects } } } } } }')


def _post(query, timeout=90):
    req = urllib.request.Request(URL, data=json.dumps({"query": query}).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=timeout))


def fetch_all(max_pages=40, verbose=True):
    rows, after = [], None
    for page in range(max_pages):
        r = _post(Q % (f', after: "{after}"' if after else ""))
        d = (r.get("data") or {}).get("datasets")
        if not d:
            break
        for e in d["edges"]:
            n = (e or {}).get("node") or {}
            if not n.get("id"):
                continue
            snap = n.get("latestSnapshot") or {}
            desc, s = snap.get("description") or {}, snap.get("summary") or {}
            rows.append({"id": n["id"], "name": desc.get("Name") or "",
                         "modalities": s.get("modalities") or [],
                         "tasks": s.get("tasks") or [],
                         "n_subjects": len(s.get("subjects") or [])})
        if not d["pageInfo"]["hasNextPage"]:
            break
        after = d["pageInfo"]["endCursor"]
        if verbose:
            print(f"  ...{len(rows)} scanned", end="\r", file=sys.stderr)
    return rows


def search(rows, kw=(), task_kw=None, modality=None, min_subjects=0):
    out = []
    name_re = re.compile("|".join(kw), re.I) if kw else None
    task_re = re.compile(task_kw, re.I) if task_kw else None
    for r in rows:
        if modality and not any(m.lower() == modality.lower() for m in r["modalities"]):
            continue
        if r["n_subjects"] < min_subjects:
            continue
        hit_name = bool(name_re and name_re.search(r["name"]))
        hit_task = bool(task_re and any(task_re.search(t) for t in r["tasks"]))
        if (name_re or task_re) and not (hit_name or hit_task):
            continue
        r = dict(r, matched=("name" if hit_name else "") + ("+task" if hit_task else ""))
        out.append(r)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kw", action="append", default=[], help="keyword in dataset Name (repeatable)")
    ap.add_argument("--task-kw", help="regex over BIDS task labels")
    ap.add_argument("--modality", help="eeg | meg | ieeg | mri ...")
    ap.add_argument("--min-subjects", type=int, default=0)
    ap.add_argument("--json", help="write full results here")
    a = ap.parse_args()

    rows = fetch_all()
    print(f"scanned {len(rows)} OpenNeuro datasets (all mirrored on NEMAR)", file=sys.stderr)
    hits = search(rows, a.kw, a.task_kw, a.modality, a.min_subjects)
    print(f"\n{len(hits)} match\n")
    for h in sorted(hits, key=lambda x: -x["n_subjects"]):
        mods = ",".join(h["modalities"]) or "-"
        tasks = ",".join(h["tasks"][:4]) + ("..." if len(h["tasks"]) > 4 else "")
        print(f"  {h['id']:11} n={h['n_subjects']:<4} [{mods:14}] {h['name'][:64]}")
        if tasks:
            print(f"              tasks: {tasks}")
    if a.json:
        json.dump(hits, open(a.json, "w"), indent=2)
        print(f"\nwrote {a.json}")
    print("\nRead on NSG with:  os.path.join(os.environ['NEMARPATH'], '<ds-id>')", file=sys.stderr)


if __name__ == "__main__":
    main()
