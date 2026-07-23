#!/usr/bin/env python3
"""Search NEMAR (all 547 OpenNeuro datasets on the node) for BCI / motor-imagery EEG.

Answers: "are the BCI Competition datasets already in NEMAR?" — stdlib only, fast (reads each
dataset_description.json Name + top of README + task labels). Reports every matching ds so we can
read data straight from $NEMARPATH instead of downloading via MOABB at runtime.
"""
import glob
import json
import os
import re
import time

RESULT = {"schema": "nsg-agent-kit/nemar-search/v1", "started": time.strftime("%Y-%m-%dT%H:%M:%S")}
KEYWORDS = ["bci competition", "motor imagery", "motor-imagery", "bnci", "imagined movement",
            "left hand", "right hand", "sensorimotor", " mi ", "brain-computer", "brain computer"]


def dataset_text(ds):
    parts = []
    dd = os.path.join(ds, "dataset_description.json")
    if os.path.isfile(dd):
        try:
            j = json.load(open(dd))
            parts.append(str(j.get("Name", "")))
        except Exception:
            pass
    for rn in ("README", "README.md", "README.txt"):
        rp = os.path.join(ds, rn)
        if os.path.isfile(rp):
            try:
                parts.append(open(rp, errors="ignore").read(3000))
            except Exception:
                pass
            break
    return " ".join(parts)


def task_labels(ds):
    tasks = set()
    # shallow: look at sub-01 (or first sub) eeg filenames for task-XXX
    subs = sorted(glob.glob(os.path.join(ds, "sub-*")))
    for sub in subs[:1]:
        for f in glob.glob(os.path.join(sub, "**", "*task-*"), recursive=True):
            m = re.search(r"task-([A-Za-z0-9]+)", os.path.basename(f))
            if m:
                tasks.add(m.group(1))
    return sorted(tasks)


def main():
    root = os.environ.get("NEMARPATH")
    if not root or not os.path.isdir(root):
        RESULT["error"] = "NEMARPATH unset/missing"; return _write()
    ds_ids = sorted(e for e in os.listdir(root) if e.startswith("ds"))
    RESULT["n_scanned"] = len(ds_ids)

    hits = []
    for ds_id in ds_ids:
        ds = os.path.join(root, ds_id)
        if not os.path.isdir(ds):
            continue
        text = dataset_text(ds).lower()
        matched = [k for k in KEYWORDS if k in text]
        # also match task label containing 'imag' or 'motor' or 'mi'
        tasks = task_labels(ds)
        task_match = [t for t in tasks if re.search(r"imag|motor|mi|bci", t, re.I)]
        if matched or task_match:
            name = ""
            dd = os.path.join(ds, "dataset_description.json")
            if os.path.isfile(dd):
                try:
                    name = json.load(open(dd)).get("Name", "")
                except Exception:
                    pass
            hits.append({"ds": ds_id, "name": name[:120],
                         "matched_keywords": matched, "tasks": tasks,
                         "task_match": task_match,
                         "n_subjects": len(glob.glob(os.path.join(ds, "sub-*")))})
    RESULT["n_hits"] = len(hits)
    RESULT["hits"] = hits
    _write()


def _write():
    RESULT["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    with open("metrics.json", "w") as f:
        json.dump(RESULT, f, indent=2)
    print(json.dumps(RESULT, indent=2))


if __name__ == "__main__":
    main()
