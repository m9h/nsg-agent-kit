#!/usr/bin/env python3
"""M2: one real OpenEEGBench-style cell on NEMAR data — end to end on the V100.

A genuine EEG decoding cell using ONLY pre-installed libs (mne 1.6.1 + torch 2.0.1, no pip):
  data (real NEMAR recording) -> events -> 2-class epochs -> inline EEGNet on GPU -> test accuracy.

Self-configuring: scans a shortlist of NEMAR datasets, picks the first whose events.tsv yields two
event types with enough trials, decodes that contrast within one subject, and reports balanced
accuracy vs chance. Writes metrics.json (the cell descriptor + score) to the working dir.
"""
import csv
import glob
import json
import os
import time

RESULT = {"schema": "nsg-agent-kit/cell/v1", "started": time.strftime("%Y-%m-%dT%H:%M:%S")}
CANDIDATES = os.environ.get("DS_LIST", "ds002718,ds002691,ds001785,ds002034,ds002094,ds001784").split(",")
MIN_TRIALS = 40


def eeg_files(ds):
    for ext in ("_eeg.fif", "_eeg.set", "_eeg.edf", "_eeg.vhdr", "_eeg.bdf"):
        hits = sorted(glob.glob(os.path.join(ds, "sub-*", "**", f"*{ext}"), recursive=True))
        if hits:
            return hits
    return []


def events_tsv_for(rawpath):
    for ext in ("_eeg.fif", "_eeg.set", "_eeg.edf", "_eeg.vhdr", "_eeg.bdf"):
        if rawpath.endswith(ext):
            cand = rawpath[: -len(ext)] + "_events.tsv"
            return cand if os.path.isfile(cand) else None
    return None


def read_trial_types(tsv):
    rows = []
    with open(tsv) as f:
        for r in csv.DictReader(f, delimiter="\t"):
            tt = r.get("trial_type") or r.get("value") or r.get("stim_type")
            onset = r.get("onset")
            if tt and onset not in (None, "n/a"):
                try:
                    rows.append((float(onset), tt))
                except ValueError:
                    pass
    return rows


def read_raw(path):
    import mne
    if path.endswith(".fif"):
        return mne.io.read_raw_fif(path, preload=True, verbose="ERROR")
    if path.endswith(".set"):
        return mne.io.read_raw_eeglab(path, preload=True, verbose="ERROR")
    if path.endswith(".edf"):
        return mne.io.read_raw_edf(path, preload=True, verbose="ERROR")
    if path.endswith(".bdf"):
        return mne.io.read_raw_bdf(path, preload=True, verbose="ERROR")
    if path.endswith(".vhdr"):
        return mne.io.read_raw_brainvision(path, preload=True, verbose="ERROR")
    raise ValueError(path)


def pick_cell():
    root = os.environ.get("NEMARPATH")
    for ds_id in CANDIDATES:
        ds = os.path.join(root, ds_id)
        files = eeg_files(ds)
        if not files:
            continue
        for rawf in files[:1]:  # first subject/run
            tsv = events_tsv_for(rawf)
            if not tsv:
                continue
            rows = read_trial_types(tsv)
            counts = {}
            for _o, tt in rows:
                counts[tt] = counts.get(tt, 0) + 1
            top = sorted((c for c in counts.items()), key=lambda kv: -kv[1])
            usable = [t for t in top if t[1] >= MIN_TRIALS]
            if len(usable) >= 2:
                return ds_id, rawf, tsv, rows, [usable[0][0], usable[1][0]], counts
    return None


def main():
    if not os.environ.get("NEMARPATH"):
        RESULT["error"] = "NEMARPATH unset"; return _write()
    pick = pick_cell()
    if not pick:
        RESULT["error"] = f"no decodable 2-class contrast found in {CANDIDATES}"; return _write()
    ds_id, rawf, tsv, rows, classes, counts = pick
    RESULT.update(ds=ds_id, recording=os.path.basename(rawf), classes=classes,
                  class_counts={c: counts[c] for c in classes})

    import numpy as np
    import mne
    mne.set_log_level("ERROR")

    raw = read_raw(rawf)
    raw.pick("eeg")
    sf = raw.info["sfreq"]
    code = {classes[0]: 1, classes[1]: 2}
    ev = np.array([[int(o * sf), 0, code[tt]] for o, tt in rows if tt in code], dtype=int)
    ev = ev[(ev[:, 0] > 0) & (ev[:, 0] < raw.n_times)]

    epochs = mne.Epochs(raw, ev, {classes[0]: 1, classes[1]: 2}, tmin=-0.2, tmax=0.8,
                        baseline=(None, 0), picks="eeg", preload=True, verbose="ERROR")
    epochs.resample(128)
    X = epochs.get_data().astype("float32")           # (n, C, T)
    y = (epochs.events[:, 2] == 2).astype("int64")    # 0/1
    RESULT.update(n_trials=int(X.shape[0]), n_channels=int(X.shape[1]),
                  n_times=int(X.shape[2]), sfreq_hz=float(sf))
    if X.shape[0] < 2 * MIN_TRIALS:
        RESULT["error"] = f"too few epochs after epoching ({X.shape[0]})"; return _write()

    # per-channel z-score within epoch
    X = (X - X.mean(axis=2, keepdims=True)) / (X.std(axis=2, keepdims=True) + 1e-7)

    import torch, torch.nn as nn
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    RESULT["device"] = dev
    RESULT["gpu_name"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None

    # stratified 75/25 split (fixed seed)
    rng = np.random.RandomState(0)
    idx = {c: rng.permutation(np.where(y == c)[0]) for c in (0, 1)}
    tr, te = [], []
    for c in (0, 1):
        n = len(idx[c]); k = max(1, int(0.75 * n))
        tr += list(idx[c][:k]); te += list(idx[c][k:])
    tr, te = np.array(tr), np.array(te)

    C, T = X.shape[1], X.shape[2]

    class EEGNet(nn.Module):
        def __init__(self, C, T, F1=8, D=2, F2=16, kern=64):
            super().__init__()
            self.conv1 = nn.Conv2d(1, F1, (1, kern), padding=(0, kern // 2), bias=False)
            self.bn1 = nn.BatchNorm2d(F1)
            self.depth = nn.Conv2d(F1, F1 * D, (C, 1), groups=F1, bias=False)
            self.bn2 = nn.BatchNorm2d(F1 * D)
            self.pool1 = nn.AvgPool2d((1, 4))
            self.sep = nn.Conv2d(F1 * D, F2, (1, 16), padding=(0, 8), bias=False)
            self.bn3 = nn.BatchNorm2d(F2)
            self.pool2 = nn.AvgPool2d((1, 8))
            self.drop = nn.Dropout(0.5)
            self.act = nn.ELU()
            with torch.no_grad():
                d = self._feat(torch.zeros(1, 1, C, T)).shape[1]
            self.fc = nn.Linear(d, 2)

        def _feat(self, x):
            x = self.bn1(self.conv1(x))
            x = self.pool1(self.act(self.bn2(self.depth(x))))
            x = self.pool2(self.act(self.bn3(self.sep(x))))
            return self.drop(x).flatten(1)

        def forward(self, x):
            return self.fc(self._feat(x))

    Xt = torch.tensor(X).unsqueeze(1).to(dev)
    yt = torch.tensor(y).to(dev)
    model = EEGNet(C, T).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    lossf = nn.CrossEntropyLoss()
    model.train()
    for ep in range(60):
        perm = torch.tensor(rng.permutation(tr)).to(dev)
        for i in range(0, len(perm), 32):
            b = perm[i:i + 32]
            opt.zero_grad(); loss = lossf(model(Xt[b]), yt[b]); loss.backward(); opt.step()

    model.eval()
    with torch.no_grad():
        pred = model(Xt[torch.tensor(te).to(dev)]).argmax(1).cpu().numpy()
    yte = y[te]
    # balanced accuracy = mean per-class recall
    recalls = []
    for c in (0, 1):
        m = yte == c
        if m.sum():
            recalls.append(float((pred[m] == c).mean()))
    RESULT["test_balanced_accuracy"] = round(sum(recalls) / len(recalls), 4)
    RESULT["test_accuracy"] = round(float((pred == yte).mean()), 4)
    RESULT["chance"] = round(float(max(np.mean(y == 0), np.mean(y == 1))), 4)
    RESULT["n_test"] = int(len(te))
    RESULT["cell_ok"] = RESULT["test_balanced_accuracy"] > RESULT["chance"] - 0.05
    _write()


def _write():
    RESULT["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    with open("metrics.json", "w") as f:
        json.dump(RESULT, f, indent=2)
    print(json.dumps(RESULT, indent=2))


if __name__ == "__main__":
    main()
