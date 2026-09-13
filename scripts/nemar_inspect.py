#!/usr/bin/env python3
"""Inspect one NEMAR/OpenNeuro dataset before committing an NSG job to it.

    python scripts/nemar_inspect.py ds002718
    python scripts/nemar_inspect.py ds002718 --files      # list file tree from the snapshot

WHY THIS MATTERS: a dataset's title tells you almost nothing about whether it holds the data you
need. In this project we burned real time discovering that an 8-mouse "graded anesthesia" dataset
was spontaneous-only (every file `stim-SPN`, the Stim event channel present but empty) -- so the
perturbational half of the analysis was impossible. Check tasks, modalities and event structure
BEFORE writing the job.
"""
import argparse
import json
import sys
import urllib.request

URL = "https://openneuro.org/crn/graphql"


def _post(q, timeout=90):
    req = urllib.request.Request(URL, data=json.dumps({"query": q}).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=timeout))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", help="e.g. ds002718")
    ap.add_argument("--files", action="store_true", help="list the snapshot file tree")
    a = ap.parse_args()

    q = ('{ dataset(id: "%s") { id created latestSnapshot { tag readme '
         'description { Name Authors License } '
         'summary { modalities tasks subjects sessions totalFiles size dataProcessed } } } }'
         % a.dataset)
    d = (_post(q).get("data") or {}).get("dataset")
    if not d:
        sys.exit(f"{a.dataset}: not found on OpenNeuro (so not on NEMAR either)")
    snap = d.get("latestSnapshot") or {}
    desc, s = snap.get("description") or {}, snap.get("summary") or {}

    print(f"\n{a.dataset}  ({snap.get('tag','?')})  created {str(d.get('created'))[:10]}")
    print(f"  name       : {desc.get('Name')}")
    print(f"  license    : {desc.get('License')}")
    auth = desc.get("Authors") or []
    print(f"  authors    : {', '.join(auth[:4])}{' ...' if len(auth) > 4 else ''}")
    print(f"  modalities : {s.get('modalities')}")
    print(f"  subjects   : {len(s.get('subjects') or [])}   sessions: {len(s.get('sessions') or [])}")
    print(f"  files      : {s.get('totalFiles')}   size: {(s.get('size') or 0)/1e9:.2f} GB")
    print(f"  tasks      : {s.get('tasks')}")

    print(f"\n  NSG path   : os.path.join(os.environ['NEMARPATH'], '{a.dataset}')")

    print("\n  ---- BEFORE writing a job, confirm ----")
    print("  * do the task labels match the paradigm you need (evoked vs resting vs spontaneous)?")
    print("  * is there an events.tsv with the trial_type / conditions you plan to contrast?")
    print("  * are the recordings long enough, and the channel count what you expect?")
    print("  A title saying 'anesthesia' or 'stimulation' does NOT guarantee evoked responses.")

    rd = (snap.get("readme") or "").strip()
    if rd:
        print(f"\n  ---- README (first 900 chars) ----\n{rd[:900]}")

    if a.files:
        q2 = ('{ snapshot(datasetId: "%s", tag: "%s") { files { filename size directory } } }'
              % (a.dataset, snap.get("tag")))
        try:
            fs = (_post(q2).get("data") or {}).get("snapshot", {}).get("files") or []
            print(f"\n  ---- top level ({len(fs)} entries) ----")
            for f in fs[:40]:
                kind = "dir " if f.get("directory") else "file"
                print(f"    {kind} {(f.get('size') or 0):>12,}  {f['filename']}")
        except Exception as e:
            print(f"  (file listing unavailable: {e!r})")


if __name__ == "__main__":
    main()
