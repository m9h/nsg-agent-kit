#!/usr/bin/env python3
"""Read a NEMAR (OpenNeuro) dataset straight from Expanse's filesystem — no download.

Datasets live at $NEMARPATH/<ds-id> in BIDS layout. This walks a dataset, lists EEG recordings,
and writes a small summary — a minimal proof that NEMAR data is reachable from your job. Point
DS_ID at any id from https://nemar.org.
"""
import json
import os
import time

DS_ID = os.environ.get("DS_ID", "ds002718")  # override per dataset


def main():
    root = os.environ.get("NEMARPATH")
    summary = {"schema": "nsg-agent-kit/nemar/v1", "ds": DS_ID,
               "nemarpath": root, "finished": None}

    if not root or not os.path.isdir(root):
        summary["error"] = "NEMARPATH not set or missing in this tool env — see docs/nemar-data.md"
        _write(summary); return

    ds = os.path.join(root, DS_ID)
    if not os.path.isdir(ds):
        summary["error"] = f"{DS_ID} not found under NEMARPATH"
        _write(summary); return

    subs, eeg_files = [], []
    for entry in sorted(os.listdir(ds)):
        if entry.startswith("sub-"):
            subs.append(entry)
    for dirpath, _dirs, files in os.walk(ds):
        for fn in files:
            if fn.endswith((".set", ".edf", ".vhdr", ".bdf", ".fif")):
                eeg_files.append(os.path.relpath(os.path.join(dirpath, fn), ds))

    summary.update(
        n_subjects=len(subs),
        subjects=subs[:10],
        n_eeg_files=len(eeg_files),
        example_eeg=eeg_files[:5],
        has_dataset_description=os.path.isfile(os.path.join(ds, "dataset_description.json")),
    )
    _write(summary)


def _write(summary):
    summary["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    with open("metrics.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
