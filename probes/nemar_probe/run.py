#!/usr/bin/env python3
"""NEMAR discovery + one-dataset summary — a real 'examples with NEMAR' step.

Read-only, stdlib-only. Proves that OpenNeuro data is reachable from an NSG job by:
  1. listing what's actually mounted under $NEMARPATH (count + a sample of ds ids),
  2. summarizing one dataset (subjects, modalities, data files, description name).

Set DS_ID via cmdlineopts to target a specific dataset; otherwise it auto-picks the
first dataset that contains EEG. Writes metrics.json to the working dir (NSG returns it).
"""
import json
import os
import sys
import time

RESULT = {"schema": "nsg-agent-kit/nemar/v1", "started": time.strftime("%Y-%m-%dT%H:%M:%S")}
DATA_EXTS = (".set", ".edf", ".vhdr", ".bdf", ".fif", ".nwb")


def modalities_in(ds):
    mods = set()
    for _dp, dirs, _files in os.walk(ds):
        for d in ("eeg", "meg", "ieeg"):
            if d in dirs:
                mods.add(d)
    return sorted(mods)


def summarize(ds):
    subs = [e for e in sorted(os.listdir(ds)) if e.startswith("sub-")]
    dfiles = []
    for dp, _dirs, files in os.walk(ds):
        for fn in files:
            if fn.endswith(DATA_EXTS):
                dfiles.append(os.path.relpath(os.path.join(dp, fn), ds))
    name = None
    dd = os.path.join(ds, "dataset_description.json")
    if os.path.isfile(dd):
        try:
            name = json.load(open(dd)).get("Name")
        except Exception:
            pass
    return {"n_subjects": len(subs), "subjects_sample": subs[:8],
            "modalities": modalities_in(ds), "n_data_files": len(dfiles),
            "data_files_sample": dfiles[:5], "description_name": name}


def main():
    root = os.environ.get("NEMARPATH")
    RESULT["nemarpath"] = root
    if not root or not os.path.isdir(root):
        RESULT["error"] = "NEMARPATH not set / missing in this tool env"
        return _write()

    entries = sorted(e for e in os.listdir(root) if e.startswith("ds"))
    RESULT["n_datasets_total"] = len(entries)
    RESULT["datasets_sample"] = entries[:30]

    target = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("DS_ID")
    if not target:
        # auto-pick the first dataset that has an eeg/ modality
        for ds_id in entries:
            ds = os.path.join(root, ds_id)
            if os.path.isdir(ds) and "eeg" in modalities_in(ds):
                target = ds_id
                break
    RESULT["target_ds"] = target

    if target:
        ds = os.path.join(root, target)
        if os.path.isdir(ds):
            RESULT["summary"] = summarize(ds)
        else:
            RESULT["error"] = f"{target} not found under NEMARPATH"
    _write()


def _write():
    RESULT["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    with open("metrics.json", "w") as f:
        json.dump(RESULT, f, indent=2)
    print(json.dumps(RESULT, indent=2))


if __name__ == "__main__":
    main()
