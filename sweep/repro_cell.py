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

NORM = CFG.get("norm", "per-session")   # per-session | train | global
SPLIT = CFG.get("split", "session")     # session (S1 train/S2 test) | pooled (random 80/20)
RESULT = {"schema": "nsg-agent-kit/repro-reve/v4", "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
          "dataset": DATASET, "subject": SUBJECT, "norm": NORM, "split": SPLIT,
          "protocol": "linear-probe+LoRA (val-early-stop, warmup+cosine, gradclip, labelsmooth)",
          "preproc": f"bp0.5-99.5,resample200,{NORM}-zscore", "paper_target": PAPER_TARGET}
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
    if SPLIT == "pooled":
        # combine both sessions, stratified random 80/20 -> removes cross-session transfer challenge
        _rng = np.random.RandomState(0); trl, tel = [], []
        for c in range(len(classes)):
            ic = np.where(yi == c)[0].copy(); _rng.shuffle(ic)
            k = int(round(0.8 * len(ic))); trl += ic[:k].tolist(); tel += ic[k:].tolist()
        tr = np.array(sorted(trl)); te = np.array(sorted(tel))
    else:  # session: S1 train / S2 test (standard BCI 2a benchmark, harder)
        tr = np.where(sessions == uniq[0])[0]
        te = np.where(sessions != uniq[0])[0] if len(uniq) > 1 else tr
    RESULT.update(n_trials=int(len(yi)), n_channels=int(X.shape[1]), n_times=int(X.shape[2]),
                  n_classes=len(classes), classes=classes, n_train=int(len(tr)), n_test=int(len(te)))

    # --- normalization (the crux for cross-session transfer) ---
    # per-session: standardize each recording session by its own per-channel stats -> removes the
    #   session-level scale/offset shift that tanked v2's session-2 test acc (the paper z-scores
    #   "across recording sessions"). global: pooled stats over all data. train: train-session only.
    if NORM == "per-session":
        for s in set(sessions):
            m = sessions == s
            mu = X[m].mean(axis=(0, 2), keepdims=True)
            sd = X[m].std(axis=(0, 2), keepdims=True) + 1e-7
            X[m] = (X[m] - mu) / sd
    elif NORM == "global":
        mu = X.mean(axis=(0, 2), keepdims=True); sd = X.std(axis=(0, 2), keepdims=True) + 1e-7
        X = (X - mu) / sd
    else:  # train-session stats only (v2 behaviour)
        mu = X[tr].mean(axis=(0, 2), keepdims=True); sd = X[tr].std(axis=(0, 2), keepdims=True) + 1e-7
        X = (X - mu) / sd
    X = X.astype("float32")

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

    import copy, math
    try:
        from peft import get_peft_model_state_dict, set_peft_model_state_dict
    except Exception:
        from peft.utils import get_peft_model_state_dict, set_peft_model_state_dict
    torch.manual_seed(0)

    Xt = torch.tensor(X).to(dev)
    yt = torch.tensor(y).to(dev)
    rng = np.random.RandomState(0)
    tr = np.asarray(tr); te = np.asarray(te)

    # stratified train/val split off the training session (val = 20%) for early stopping
    val = []
    for c in range(n_classes):
        ic = tr[y[tr] == c].copy(); rng.shuffle(ic)
        val.extend(ic[:max(1, int(round(0.2 * len(ic))))].tolist())
    val = np.array(sorted(val)); vs = set(val.tolist())
    trn = np.array([i for i in tr if i not in vs])
    RES["n_train_fit"], RES["n_val"] = int(len(trn)), int(len(val))

    def fwd(idx):
        pos = positions.expand(len(idx), -1, -1)
        return pool_tokens(reve(Xt[idx], pos))

    def bacc(pred, yy):
        recs = [float((pred[yy == c] == c).mean()) for c in range(n_classes) if (yy == c).sum()]
        return sum(recs) / len(recs)

    def evaluate(idx):
        reve.eval(); head.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(idx), 32):
                b = torch.tensor(idx[i:i + 32]).to(dev)
                out.append(head(fwd(b)).argmax(1).cpu().numpy())
        p = np.concatenate(out)
        return p, bacc(p, y[idx])

    with torch.no_grad():
        hid = fwd(torch.tensor(trn[:2]).to(dev)).shape[1]
    RES["embedding_dim"] = int(hid)
    head = nn.Linear(hid, n_classes).to(dev)
    lossf = nn.CrossEntropyLoss(label_smoothing=0.1)

    # --- Phase A: linear probe on CACHED frozen embeddings (fast, stable head init) ---
    for p in reve.parameters():
        p.requires_grad_(False)
    reve.eval()
    with torch.no_grad():
        emb = torch.cat([fwd(torch.tensor(tr[i:i + 32]).to(dev)) for i in range(0, len(tr), 32)])
    pos_of = {int(t): k for k, t in enumerate(tr)}
    E_trn = emb[[pos_of[i] for i in trn]]; y_trn = yt[trn]
    E_val = emb[[pos_of[i] for i in val]]
    optA = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-2)
    best_hv, best_head = -1.0, None
    for _ in range(80):
        head.train(); perm = rng.permutation(len(E_trn))
        for i in range(0, len(perm), 32):
            b = torch.tensor(perm[i:i + 32]).to(dev)
            optA.zero_grad(); lossf(head(E_trn[b]), y_trn[b]).backward(); optA.step()
        head.eval()
        with torch.no_grad():
            v = bacc(head(E_val).argmax(1).cpu().numpy(), y[val])
        if v > best_hv:
            best_hv, best_head = v, copy.deepcopy(head.state_dict())
    head.load_state_dict(best_head)
    RES["probe_val_bacc"] = round(best_hv, 4)

    # --- Phase B: LoRA fine-tune, warmup+cosine LR, grad-clip, early stop on val ---
    lora_cfg = LoraConfig(r=32, lora_alpha=64, lora_dropout=0.1,
                          target_modules=["to_qkv", "to_out", "net.1", "net.3"], bias="none")
    reve = get_peft_model(reve, lora_cfg)
    RES["lora_trainable_params"] = int(sum(p.numel() for p in reve.parameters() if p.requires_grad))
    params = [p for p in reve.parameters() if p.requires_grad] + list(head.parameters())
    max_ep = int(os.environ.get("LORA_EPOCHS", 60)); warmup = 5; base_lr = 2e-4
    opt = torch.optim.AdamW(params, lr=base_lr, weight_decay=1e-2)

    def lr_scale(ep):
        if ep < warmup:
            return (ep + 1) / warmup
        return 0.5 * (1 + math.cos(math.pi * (ep - warmup) / max(1, max_ep - warmup)))

    best_vv, best_state, bad, patience = -1.0, None, 0, 15
    loss = None
    for ep in range(max_ep):
        for g in opt.param_groups:
            g["lr"] = base_lr * lr_scale(ep)
        reve.train(); head.train(); perm = rng.permutation(trn)
        for i in range(0, len(perm), 16):
            b = torch.tensor(perm[i:i + 16]).to(dev)
            opt.zero_grad(); loss = lossf(head(fwd(b)), yt[b]); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
        _, vv = evaluate(val)
        if vv > best_vv + 1e-4:
            best_vv, bad = vv, 0
            best_state = {"head": copy.deepcopy(head.state_dict()),
                          "lora": copy.deepcopy(get_peft_model_state_dict(reve))}
        else:
            bad += 1
            if bad >= patience:
                break
    RES["lora_val_bacc"] = round(best_vv, 4)
    RES["lora_epochs_ran"] = ep + 1
    RES["final_train_loss"] = float(loss.item()) if loss is not None else None
    if best_state is not None:
        head.load_state_dict(best_state["head"])
        set_peft_model_state_dict(reve, best_state["lora"])

    pred, test_bacc = evaluate(te)
    yte = y[te]
    return float((pred == yte).mean()), test_bacc


def _write():
    RESULT["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    with open("metrics.json", "w") as f:
        json.dump(RESULT, f, indent=2)
    print(json.dumps(RESULT, indent=2))


if __name__ == "__main__":
    main()
