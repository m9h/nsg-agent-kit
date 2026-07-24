#!/usr/bin/env python3
"""Paper-faithful REVE reproduction cell on BCI IV 2a — one subject.

Reproduces the REVE paper's (arXiv 2510.21585, Table 2) BCI-IV-2a evaluation as closely as the
gateway allows, so the NSG number is comparable to their **REVE-Base 0.6396 ± 0.0095** (bal. acc.):

  - preprocessing: band-pass **0.5–99.5 Hz** (broadband, as pretrained) + resample 200 Hz;
    **across-session z-score** (per-channel stats from the training session, applied to both).
  - protocol: **two-step fine-tune** — (A) linear-probe warmup with the encoder frozen, then
    (B) **LoRA** (r=32/alpha=64) on REVE's attention (to_qkv/to_out) + FFN (net.1/net.3) plus the head.
  - split: within-subject, session-1 train / session-2 test (the CBraMod/LaBraM/BIOT convention).
  - metric: balanced accuracy on the test session. Aggregate over 9 subjects downstream.

Weights are vendored (HERE/reve-base, HERE/reve-positions). torch pinned via cell.json (2.4.1: has
torch.nn.attention, keeps V100/sm_70). Deps to node-local scratch. Writes metrics.json.
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
DATASET = CFG.get("dataset", "BNCI2014_001")
SUBJECT = int(CFG.get("subject", 1))
TORCH_OVERRIDE = CFG.get("torch", "2.4.1")
PAPER_TARGET = 0.6396  # REVE-Base BCI-IV-2a balanced accuracy (paper Table 2)

RESULT = {"schema": "nsg-agent-kit/repro-reve/v1", "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
          "dataset": DATASET, "subject": SUBJECT, "protocol": "linear-probe+LoRA",
          "preproc": "bp0.5-99.5,resample200,across-session-zscore", "paper_target": PAPER_TARGET}
LIBS = os.path.join(os.environ.get("TMPDIR", "/tmp"), "nsgkit-pylibs")


def pip(pkgs):
    t0 = time.time()
    r = subprocess.run([sys.executable, "-m", "pip", "install", "--target", LIBS, *pkgs],
                       capture_output=True, text=True, timeout=1800)
    return r.returncode == 0, round(time.time() - t0, 1), r.stderr[-800:]


def capture_env():
    env = {"python": sys.version.split()[0]}
    try:
        smi = subprocess.run(["nvidia-smi", "--query-gpu=driver_version,name", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=15)
        env["nvidia_smi"] = smi.stdout.strip().splitlines()[0]
    except Exception as e:
        env["smi_error"] = repr(e)
    RESULT["env"] = env


def pool_tokens(out):
    t = getattr(out, "last_hidden_state", out)
    if isinstance(t, (tuple, list)):
        t = t[0]
    if t.ndim > 2:
        t = t.mean(dim=tuple(range(1, t.ndim - 1)))
    return t


def main():
    capture_env()
    deps = [f"torch=={TORCH_OVERRIDE.split('+')[0]}", "numpy<2", "moabb", "scikit-learn",
            "transformers==4.44.2", "peft==0.10.0", "huggingface_hub", "safetensors", "einops"]
    ok, secs, err = pip(deps)
    RESULT["pip_ok"], RESULT["pip_seconds"] = ok, secs
    if not ok:
        RESULT["pip_err"] = err; return _write()
    sys.path.insert(0, LIBS)
    os.environ["USE_TORCH"] = "1"

    import numpy as np
    import torch, torch.nn as nn
    RESULT["env"]["torch"] = torch.__version__
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    RESULT["device"] = dev

    # --- data: broadband, 200 Hz ---
    scratch = os.environ.get("TMPDIR", "/tmp")
    for d in ("mne_data", "moabb_res"):
        os.makedirs(os.path.join(scratch, d), exist_ok=True)
    os.environ["MNE_DATA"] = os.path.join(scratch, "mne_data")
    try:
        import moabb
        from moabb.paradigms import MotorImagery
        moabb.set_log_level("ERROR")
        ds_cls = getattr(__import__("moabb.datasets", fromlist=[DATASET]), DATASET)
        paradigm = MotorImagery(fmin=0.5, fmax=99.5, resample=200)
        ep, y, meta = paradigm.get_data(dataset=ds_cls(), subjects=[SUBJECT], return_epochs=True)
        X = ep.get_data().astype("float32")
        ch = ep.ch_names
        sessions = meta["session"].to_numpy()
    except Exception as e:
        RESULT["data_error"] = repr(e); return _write()
    classes = sorted(set(y))
    yi = np.array([classes.index(v) for v in y], dtype="int64")
    uniq = sorted(set(sessions))
    tr = np.where(sessions == uniq[0])[0]
    te = np.where(sessions != uniq[0])[0] if len(uniq) > 1 else tr
    RESULT.update(n_trials=int(len(yi)), n_channels=int(X.shape[1]), n_times=int(X.shape[2]),
                  n_classes=len(classes), classes=classes, n_train=int(len(tr)), n_test=int(len(te)))

    # --- across-session z-score: per-channel stats from the training session ---
    mu = X[tr].mean(axis=(0, 2), keepdims=True)
    sd = X[tr].std(axis=(0, 2), keepdims=True) + 1e-7
    X = ((X - mu) / sd).astype("float32")

    try:
        acc, bacc = train_reve_lora(X, yi, tr, te, ch, len(classes), dev, RESULT)
    except Exception as e:
        import traceback
        RESULT["model_error"] = repr(e); RESULT["trace"] = traceback.format_exc()[-1500:]
        return _write()
    RESULT["test_accuracy"] = round(acc, 4)
    RESULT["test_balanced_accuracy"] = round(bacc, 4)
    RESULT["chance"] = round(1.0 / len(classes), 4)
    RESULT["delta_vs_paper"] = round(bacc - PAPER_TARGET, 4)
    _write()


def train_reve_lora(X, y, tr, te, ch_names, n_classes, dev, RES):
    import numpy as np, torch, torch.nn as nn
    from transformers import AutoModel
    from peft import LoraConfig, get_peft_model

    pos_bank = AutoModel.from_pretrained(os.path.join(HERE, "reve-positions"),
                                         trust_remote_code=True, local_files_only=True)
    reve = AutoModel.from_pretrained(os.path.join(HERE, "reve-base"),
                                     trust_remote_code=True, local_files_only=True).to(dev)
    RES["reve_source"] = "vendored"
    positions = pos_bank(ch_names)
    if isinstance(positions, torch.Tensor):
        positions = positions.to(dev)

    Xt = torch.tensor(X).to(dev)
    yt = torch.tensor(y).to(dev)
    rng = np.random.RandomState(0)

    def fwd(idx):
        pos = positions.expand(len(idx), -1, -1)
        return pool_tokens(reve(Xt[idx], pos))

    # infer hidden dim
    with torch.no_grad():
        hid = fwd(torch.tensor(tr[:2]).to(dev)).shape[1]
    RES["embedding_dim"] = int(hid)
    head = nn.Linear(hid, n_classes).to(dev)
    lossf = nn.CrossEntropyLoss()

    # --- Phase A: linear-probe warmup (encoder frozen) ---
    for p in reve.parameters():
        p.requires_grad_(False)
    optA = torch.optim.AdamW(head.parameters(), lr=1e-3)
    for _ in range(int(os.environ.get("PROBE_EPOCHS", 20))):
        perm = rng.permutation(tr)
        for i in range(0, len(perm), 32):
            b = torch.tensor(perm[i:i + 32]).to(dev)
            with torch.no_grad():
                h = fwd(b)
            optA.zero_grad(); lossf(head(h), yt[b]).backward(); optA.step()

    # --- Phase B: LoRA fine-tune (attention + FFN) + head ---
    lora_cfg = LoraConfig(r=32, lora_alpha=64, lora_dropout=0.05,
                          target_modules=["to_qkv", "to_out", "net.1", "net.3"], bias="none")
    reve = get_peft_model(reve, lora_cfg)
    tr_p = sum(p.numel() for p in reve.parameters() if p.requires_grad)
    RES["lora_trainable_params"] = int(tr_p)
    params = [p for p in reve.parameters() if p.requires_grad] + list(head.parameters())
    n_epochs = int(os.environ.get("LORA_EPOCHS", 30))
    opt = torch.optim.AdamW(params, lr=5e-4, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)
    reve.train(); head.train()
    for _ep in range(n_epochs):
        perm = rng.permutation(tr)
        for i in range(0, len(perm), 16):
            b = torch.tensor(perm[i:i + 16]).to(dev)
            opt.zero_grad(); loss = lossf(head(fwd(b)), yt[b]); loss.backward(); opt.step()
        sched.step()
    RES["final_train_loss"] = float(loss.item())

    reve.eval(); head.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(te), 32):
            b = torch.tensor(te[i:i + 32]).to(dev)
            preds.append(head(fwd(b)).argmax(1).cpu().numpy())
    pred = np.concatenate(preds)
    yte = y[te]
    acc = float((pred == yte).mean())
    recs = [float((pred[yte == c] == c).mean()) for c in range(n_classes) if (yte == c).sum()]
    return acc, sum(recs) / len(recs)


def _write():
    RESULT["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    with open("metrics.json", "w") as f:
        json.dump(RESULT, f, indent=2)
    print(json.dumps(RESULT, indent=2))


if __name__ == "__main__":
    main()
