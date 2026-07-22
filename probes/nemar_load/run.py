#!/usr/bin/env python3
"""Load a real NEMAR EEG recording with MNE — and prove runtime pip works.

Two things at once:
  1. M1 evidence: `pip install mne` on the compute node (egress was true in M0) — does a real
     third-party package install + import succeed at runtime? Records pip_ok + timing.
  2. NEMAR example: find the first EEG recording in a dataset under $NEMARPATH (following the
     git-annex symlinks) and read its header with MNE — n_channels, sfreq, duration, ch types.

Writes metrics.json to the working dir. Target dataset via argv[1] or DS_ID (default ds001784).
"""
import glob
import json
import os
import subprocess
import sys
import time

RESULT = {"schema": "nsg-agent-kit/nemar-load/v1", "started": time.strftime("%Y-%m-%dT%H:%M:%S")}
EXTS = ("_eeg.fif", "_eeg.set", "_eeg.edf", "_eeg.bdf", "_eeg.vhdr")


def pip_install(pkgs):
    t0 = time.time()
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--quiet", *pkgs],
            stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT, timeout=600,
        )
        return True, round(time.time() - t0, 1), None
    except Exception as e:
        return False, round(time.time() - t0, 1), repr(e)


def find_first_eeg(ds):
    # follow symlinks into git-annex; prefer BIDS-named recordings under sub-*/**/eeg/
    for ext in EXTS:
        hits = sorted(glob.glob(os.path.join(ds, "sub-*", "**", f"*{ext}"), recursive=True))
        if hits:
            return hits[0]
    return None


def main():
    root = os.environ.get("NEMARPATH")
    ds_id = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("DS_ID", "ds001784")
    RESULT["nemarpath"] = root
    RESULT["ds"] = ds_id
    if not root:
        RESULT["error"] = "NEMARPATH not set"; return _write()
    ds = os.path.join(root, ds_id)

    # 1) runtime pip
    ok, secs, err = pip_install(["mne"])
    RESULT["pip_ok"] = ok
    RESULT["pip_seconds"] = secs
    if err:
        RESULT["pip_error"] = err
    if not ok:
        return _write()

    import mne
    mne.set_log_level("ERROR")
    RESULT["mne_version"] = mne.__version__

    # 2) load a real recording header
    rec = find_first_eeg(ds)
    RESULT["recording"] = os.path.relpath(rec, ds) if rec else None
    if not rec:
        RESULT["error"] = f"no EEG recording found under {ds_id}"; return _write()
    try:
        raw = mne.io.read_raw(rec, preload=False, verbose="ERROR")
        RESULT["load_ok"] = True
        RESULT["n_channels"] = raw.info["nchan"]
        RESULT["sfreq_hz"] = float(raw.info["sfreq"])
        RESULT["duration_s"] = round(raw.n_times / raw.info["sfreq"], 1)
        types = {}
        for t in raw.get_channel_types():
            types[t] = types.get(t, 0) + 1
        RESULT["channel_types"] = types
    except Exception as e:
        RESULT["load_ok"] = False
        RESULT["load_error"] = repr(e)
    _write()


def _write():
    RESULT["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    with open("metrics.json", "w") as f:
        json.dump(RESULT, f, indent=2)
    print(json.dumps(RESULT, indent=2))


if __name__ == "__main__":
    main()
