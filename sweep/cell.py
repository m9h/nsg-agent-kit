#!/usr/bin/env python3
"""One sweep cell: (source, dataset, subject, model) -> a scored result on NSG's V100.

Config comes from cell.json in the same dir (or env fallbacks). Supports:
  source: "moabb"  -> download a MOABB dataset at runtime (egress works)   e.g. BNCI2014_001
          "nemar"  -> read an OpenNeuro ds from $NEMARPATH (offline)        e.g. ds007221
  model:  "eegnet" -> inline EEGNet, supervised, session-holdout
          "reve"   -> frozen REVE-base encoder (hf:brain-bzh/reve-base) + linear probe

Preprocessing is shared: motor-imagery epochs, band-pass 4-38 Hz, resampled to 200 Hz (REVE's
requirement; fine for EEGNet). Writes metrics.json (cell descriptor + accuracy) to the working dir.

Deps install to node-local scratch (system venv is read-only; see docs/dependencies.md).
"""
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
CFG = {}
for p in (os.path.join(HERE, "cell.json"), "cell.json"):
    if os.path.isfile(p):
        CFG = json.load(open(p)); break
SOURCE = CFG.get("source", os.environ.get("SOURCE", "moabb"))
DATASET = CFG.get("dataset", os.environ.get("DATASET", "BNCI2014_001"))
SUBJECT = int(CFG.get("subject", os.environ.get("SUBJECT", 1)))
MODEL = CFG.get("model", os.environ.get("MODEL", "eegnet"))
REVE = CFG.get("reve", "brain-bzh/reve-base")

RESULT = {"schema": "nsg-agent-kit/sweep-cell/v1", "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
          "source": SOURCE, "dataset": DATASET, "subject": SUBJECT, "model": MODEL}
LIBS = os.path.join(os.environ.get("TMPDIR", "/tmp"), "nsgkit-pylibs")


def pip(pkgs):
    t0 = time.time()
    r = subprocess.run([sys.executable, "-m", "pip", "install", "--target", LIBS, *pkgs],
                       capture_output=True, text=True, timeout=1200)
    return r.returncode == 0, round(time.time() - t0, 1), r.stderr[-600:]


def load_data():
    """Return X (n,C,T) float32 @200Hz, y (int), ch_names, sessions (array), class_names."""
    import numpy as np
    scratch = os.environ.get("TMPDIR", "/tmp")
    data_dir = os.path.join(scratch, "mne_data")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(os.path.join(scratch, "moabb_res"), exist_ok=True)
    os.environ["MNE_DATA"] = data_dir
    os.environ["MOABB_RESULTS"] = os.path.join(scratch, "moabb_res")
    if SOURCE == "moabb":
        import moabb
        from moabb.paradigms import MotorImagery
        moabb.set_log_level("ERROR")
        ds_cls = getattr(__import__("moabb.datasets", fromlist=[DATASET]), DATASET)
        ds = ds_cls()
        paradigm = MotorImagery(fmin=4, fmax=38, resample=200)
        ep, y, meta = paradigm.get_data(dataset=ds, subjects=[SUBJECT], return_epochs=True)
        X = ep.get_data().astype("float32")
        ch = ep.ch_names
        sessions = meta["session"].to_numpy()
        classes = sorted(set(y))
        yi = np.array([classes.index(v) for v in y], dtype="int64")
        return X, yi, ch, sessions, classes
    raise ValueError(f"source {SOURCE} not implemented in this validation build")


def split_sessions(sessions):
    import numpy as np
    uniq = sorted(set(sessions))
    if len(uniq) >= 2:
        tr = sessions == uniq[0]
        te = ~tr
    else:  # single session -> 75/25 by order
        n = len(sessions); k = int(0.75 * n)
        tr = np.zeros(n, bool); tr[:k] = True; te = ~tr
    return np.where(tr)[0], np.where(te)[0]


def run_eegnet(X, y, tr, te, n_classes):
    import numpy as np, torch, torch.nn as nn
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    RESULT["device"] = dev
    Xz = (X - X.mean(2, keepdims=True)) / (X.std(2, keepdims=True) + 1e-7)
    C, T = X.shape[1], X.shape[2]

    class EEGNet(nn.Module):
        def __init__(self, C, T, n_cls, F1=8, D=2, F2=16, k=64):
            super().__init__()
            self.c1 = nn.Conv2d(1, F1, (1, k), padding=(0, k // 2), bias=False); self.b1 = nn.BatchNorm2d(F1)
            self.dw = nn.Conv2d(F1, F1 * D, (C, 1), groups=F1, bias=False); self.b2 = nn.BatchNorm2d(F1 * D)
            self.p1 = nn.AvgPool2d((1, 4))
            self.sp = nn.Conv2d(F1 * D, F2, (1, 16), padding=(0, 8), bias=False); self.b3 = nn.BatchNorm2d(F2)
            self.p2 = nn.AvgPool2d((1, 8)); self.dp = nn.Dropout(0.5); self.a = nn.ELU()
            with torch.no_grad():
                d = self._f(torch.zeros(1, 1, C, T)).shape[1]
            self.fc = nn.Linear(d, n_cls)

        def _f(self, x):
            x = self.b1(self.c1(x)); x = self.p1(self.a(self.b2(self.dw(x))))
            x = self.p2(self.a(self.b3(self.sp(x)))); return self.dp(x).flatten(1)

        def forward(self, x): return self.fc(self._f(x))

    Xt = torch.tensor(Xz).unsqueeze(1).to(dev); yt = torch.tensor(y).to(dev)
    m = EEGNet(C, T, n_classes).to(dev)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-4); lf = nn.CrossEntropyLoss()
    rng = np.random.RandomState(0); m.train()
    for _ in range(80):
        perm = torch.tensor(rng.permutation(tr)).to(dev)
        for i in range(0, len(perm), 32):
            b = perm[i:i + 32]; opt.zero_grad(); lf(m(Xt[b]), yt[b]).backward(); opt.step()
    m.eval()
    with torch.no_grad():
        pred = m(Xt[torch.tensor(te).to(dev)]).argmax(1).cpu().numpy()
    return pred


def run_reve(X, y, tr, te, ch_names, n_classes):
    import numpy as np, torch
    from transformers import AutoModel
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    RESULT["device"] = dev
    pos_bank = AutoModel.from_pretrained("brain-bzh/reve-positions", trust_remote_code=True)
    reve = AutoModel.from_pretrained(REVE, trust_remote_code=True).to(dev).eval()
    positions = pos_bank(ch_names)                      # (C, 3)
    if isinstance(positions, torch.Tensor):
        positions = positions.to(dev)

    def embed(batch):
        xt = torch.tensor(batch).to(dev)
        pos = positions.expand(xt.size(0), -1, -1)
        with torch.no_grad():
            out = reve(xt, pos)
        t = getattr(out, "last_hidden_state", out)
        if isinstance(t, (tuple, list)):
            t = t[0]
        if t.ndim == 3:
            t = t.mean(1)
        return t.float().cpu().numpy()

    Xz = ((X - X.mean(2, keepdims=True)) / (X.std(2, keepdims=True) + 1e-7)).astype("float32")
    embs = np.concatenate([embed(Xz[i:i + 64]) for i in range(0, len(Xz), 64)], 0)
    RESULT["embedding_dim"] = int(embs.shape[1])
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
    clf.fit(embs[tr], y[tr])
    return clf.predict(embs[te])


def main():
    # Pin numpy<2: the image's torch 2.0.1 was built against numpy 1.x, and moabb/transformers
    # otherwise pull numpy 2.x into the target dir, shadowing the image numpy and breaking torch
    # ("Could not infer dtype of numpy.float32"; transformers then can't detect torch).
    deps = ["numpy<2", "moabb", "scikit-learn"]
    if MODEL == "reve":
        deps += ["transformers", "huggingface_hub", "safetensors", "einops"]
    ok, secs, err = pip(deps)
    RESULT["pip_ok"] = ok; RESULT["pip_seconds"] = secs
    if not ok:
        RESULT["pip_err"] = err; return _write()
    sys.path.insert(0, LIBS)

    import numpy as np
    try:
        X, y, ch, sessions, classes = load_data()
    except Exception as e:
        RESULT["data_error"] = repr(e); return _write()
    RESULT.update(n_trials=int(len(y)), n_channels=int(X.shape[1]), n_times=int(X.shape[2]),
                  n_classes=len(classes), classes=classes, ch_names=ch[:30],
                  sessions=sorted(set(map(str, sessions))))
    tr, te = split_sessions(sessions)
    RESULT["n_train"], RESULT["n_test"] = int(len(tr)), int(len(te))

    try:
        if MODEL == "eegnet":
            pred = run_eegnet(X, y, tr, te, len(classes))
        elif MODEL == "reve":
            pred = run_reve(X, y, tr, te, ch, len(classes))
        else:
            RESULT["error"] = f"unknown model {MODEL}"; return _write()
    except Exception as e:
        import traceback
        RESULT["model_error"] = repr(e); RESULT["trace"] = traceback.format_exc()[-1200:]
        return _write()

    yte = y[te]
    acc = float((pred == yte).mean())
    recs = [float((pred[yte == c] == c).mean()) for c in range(len(classes)) if (yte == c).sum()]
    RESULT["test_accuracy"] = round(acc, 4)
    RESULT["test_balanced_accuracy"] = round(sum(recs) / len(recs), 4)
    RESULT["chance"] = round(1.0 / len(classes), 4)
    RESULT["cell_ok"] = RESULT["test_balanced_accuracy"] > RESULT["chance"]
    _write()


def _write():
    RESULT["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    with open("metrics.json", "w") as f:
        json.dump(RESULT, f, indent=2)
    print(json.dumps(RESULT, indent=2))


if __name__ == "__main__":
    main()
