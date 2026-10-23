#!/usr/bin/env python3
"""One NSG job = one OpenEEGBench *shard*: a list of (backbone, dataset, strategy, head, seed)
experiments run sequentially with upstream `open_eeg_bench` (not a re-implementation) on the
node, cached into ./results in exca's own layout so the shards merge losslessly afterwards.

Why shards, not one-job-per-experiment: every NSG job pays ~2-4 min of pip + the HF dataset
download (0.1-13 GB per dataset). Grouping all backbones x strategies for one dataset x seed into
a job pays the download once. Why exca's cache and not our own metrics.json: upstream's
`collect_completed_results()` then reads the merged caches unchanged (verified relocatable: the
cache holds no absolute paths), so our numbers are produced by the benchmark's code path, not a
port of it.

Config comes from shard.json next to this file:
  {"backbones": ["biot"], "datasets": ["arithmetic_zyma2019"], "strategies": ["frozen"],
   "heads": ["linear_head"], "seeds": [0], "device": "cuda",
   "torch": "2.4.1", "oeb": "0.6.0",
   "checkpoints": {"reve": "reve_weights/reve-base/model.safetensors"},   # optional, for gated repos
   "training_overrides": {"eegpt": {"batch_size": 16}}}                   # optional, flagged deviation

Deps install to node-local scratch (system venv is read-only; see docs/dependencies.md). The
heavy work runs in a FRESH interpreter that inherits PYTHONPATH, so the target's numpy/torch win
over the image's (docs/LIMITATIONS.md §3).
"""
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SCRATCH = os.environ.get("TMPDIR", "/tmp")
LIBS = os.path.join(SCRATCH, "nsgkit-oeblibs")
HF_HOME = os.path.join(SCRATCH, "hf")          # dataset + weight cache: NOT returned in output.tar.gz
RESULTS = os.path.join(HERE, "results")        # exca cache: returned in output.tar.gz
SUMMARY = os.path.join(HERE, "shard_summary.json")


def load_cfg():
    for p in (os.path.join(HERE, "shard.json"), "shard.json"):
        if os.path.isfile(p):
            return json.load(open(p))
    return {}


def install(cfg, summary):
    """torch FIRST and pinned: peft would otherwise pull a cu130 torch that dropped Volta/V100
    (docs/FINDINGS.md §6a). numpy<2: the image torch is numpy-1 ABI. --ignore-installed makes the
    target self-contained so it shadows the image's torch 2.0.1 cleanly."""
    torch_pin = cfg.get("torch", "2.4.1")
    oeb_pin = cfg.get("oeb", "0.6.0")
    # torchaudio must match torch exactly: braindecode requires it unpinned and an unmatched build
    # fails at import (undefined symbol aoti_torch_abi_version) [measured locally 2026-10-22].
    # transformers 5.x is broken on torch 2.4.1 (NameError: nn in integrations/accelerate.py) ->
    # pin 4.46.3. peft must be RECENT (>=0.17: upstream finetuning.py passes target_parameters,
    # 'all-linear' on plain nn.Modules, new OFT semantics) -- peft 0.14.0 failed 33/59 bench1 cells
    # on NSG 2026-10-23; 0.21.2 imports cleanly next to transformers 4.46.3 [measured locally].
    # pysqlite3-binary: the tool's Python 3.11.4 has NO _sqlite3 extension (measured on exp-9-57,
    # 2026-10-22, job ...F6E9A7AD) and exca imports sqlite3 -> shimmed in run_shard().
    pkgs = [f"torch=={torch_pin}", f"torchaudio=={torch_pin}", "numpy<2",
            cfg.get("transformers", "transformers==4.46.3"), cfg.get("peft", "peft==0.21.2"),
            f"open-eeg-bench=={oeb_pin}", "pysqlite3-binary"] + cfg.get("extra_pip", [])
    t0 = time.time()
    r = subprocess.run([sys.executable, "-m", "pip", "install", "--target", LIBS,
                        "--ignore-installed", "-q", *pkgs], capture_output=True, text=True, timeout=2400)
    summary["pip_ok"] = r.returncode == 0
    summary["pip_seconds"] = round(time.time() - t0, 1)
    summary["pip_pkgs"] = pkgs
    if r.returncode != 0:
        summary["pip_err"] = r.stderr[-1500:]
    return r.returncode == 0


def build_experiments(backbone_dict, datasets, heads, strategies, seeds, device, infra, training=None):
    """Upstream Experiment configs for one backbone, built the way benchmark() builds them but from
    the FULL backbone dump (benchmark() has no training_required_modules/parameters kwargs, so
    BIOT/LaBraM fail at build() through it in 0.6.0 -- measured 2026-10-22). exca UIDs are
    content-addressed over the config, so oeb_plan.py collect calls this same function to find the
    cached results. Returns (experiments, skipped) where skipped are the (strategy) combos upstream's
    validator rejects for this backbone."""
    from open_eeg_bench.backbone import PretrainedBackbone
    from open_eeg_bench.default_configs.experiments import make_all_experiments
    backbone = PretrainedBackbone.model_validate(backbone_dict)
    # `training` = per-backbone deviations from upstream's Training defaults (batch_size=64, 50 epochs):
    # EEGPT with gradients through the backbone needs >32 GB at batch 64 (FINDINGS §8b), so a V100
    # shard can set {"eegpt": {"batch_size": 16}}. It changes the exca uid, so such a cell never
    # masquerades as the default-protocol cell in a pooled cache, and the deviation is in the row.
    overrides = {"backbone": backbone, "training": {"device": device, **(training or {})}, "infra": infra}
    experiments, skipped = [], []
    for strategy in strategies:
        base = make_all_experiments(datasets=datasets, heads=heads, finetuning_strategies=[strategy],
                                    n_seeds=max(seeds) + 1)
        try:
            exps = [e.infra.clone_obj(overrides) for e in base]
        except ValueError as err:
            msg = [ln.strip() for ln in str(err).splitlines() if ln.strip().startswith("Value error")]
            skipped.append({"finetuning": strategy, "reason": (msg[0] if msg else str(err))[:200]})
            continue
        experiments += [e for e in exps if e.seed in seeds]
    return experiments, skipped


def shim_sqlite3(summary):
    """NSG's PyTorch-tool Python (3.11.4) ships without the _sqlite3 extension, so `import sqlite3`
    raises ModuleNotFoundError and exca (upstream's runner) cannot be imported. pysqlite3-binary is
    a drop-in: alias it as sqlite3 BEFORE anything imports exca. [measured 2026-10-22]"""
    try:
        import sqlite3  # noqa: F401
        summary["sqlite3"] = "stdlib"
    except ModuleNotFoundError:
        import pysqlite3
        sys.modules["sqlite3"] = pysqlite3
        sys.modules["sqlite3.dbapi2"] = pysqlite3.dbapi2
        summary["sqlite3"] = f"pysqlite3 {pysqlite3.version} (no _sqlite3 in the image)"


def _deviation(training):
    """'' when `training` equals upstream's protocol (Training defaults), else 'key=value,...'
    for every field that differs (device/num_workers excluded: not part of the uid either)."""
    default = type(training)().model_dump()
    diff = [f"{k}={v}" for k, v in training.model_dump().items()
            if k not in ("device", "num_workers") and v != default[k]]
    return ",".join(diff)


def run_shard(cfg, summary):
    """Runs in the fresh interpreter. Builds upstream Experiment configs and runs them in-process
    (cluster=None) with the exca cache under ./results."""
    shim_sqlite3(summary)
    import open_eeg_bench as oeb
    from open_eeg_bench.default_configs import backbones as BB
    from open_eeg_bench.default_configs.experiments import make_all_experiments
    from open_eeg_bench.experiment import run_many
    import torch

    summary["oeb_version"] = oeb.__version__ if hasattr(oeb, "__version__") else "?"
    summary["torch"] = torch.__version__
    summary["cuda"] = torch.cuda.is_available()
    summary["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None

    device = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    infra = {"folder": RESULTS, "cluster": None, "mode": cfg.get("mode", "cached")}
    checkpoints = cfg.get("checkpoints", {})

    experiments, skipped, bb_cfgs = [], [], {}
    for bb_name in cfg.get("backbones", ["biot"]):
        bb_dict = BB.ALL_BACKBONES[bb_name]().model_dump()
        if bb_name in checkpoints:                      # gated hub repo (e.g. REVE) -> vendored file
            bb_dict.update(hub_repo=None, checkpoint_path=os.path.join(HERE, checkpoints[bb_name]))
        bb_cfgs[bb_name] = bb_dict
        exps, skip = build_experiments(bb_dict, cfg.get("datasets"), cfg.get("heads"),
                                       cfg.get("strategies", ["frozen"]), cfg.get("seeds", [0]), device, infra,
                                       training=cfg.get("training_overrides", {}).get(bb_name))
        experiments += exps
        skipped += [dict(backbone=bb_name, **k) for k in skip]
    summary["backbone_configs"] = bb_cfgs
    summary["training_overrides"] = cfg.get("training_overrides", {})
    summary["n_experiments"] = len(experiments)
    summary["skipped_invalid"] = skipped

    t0 = time.time()
    # One run_many() call per experiment, with a GPU cleanup in between: the cells share one
    # process, and bench1 (2026-10-23, exp-10-60) showed EEGPT full/two_stages OOM on a 32 GB V100
    # with 28.5 GB "allocated by PyTorch" after 40 earlier cells -- the per-cell peak below tells
    # whether that was the cell or the accumulation. Also gives per-cell wall time independent of
    # upstream's fit_time. Failures stay per-cell (collect_all=True) and never abort the shard.
    import gc
    rows, cells = [], []
    for e in experiments:
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        tc = time.time()
        df = run_many([e], collect_all=True)
        wall = round(time.time() - tc, 1)
        peak = round(torch.cuda.max_memory_allocated() / 2**30, 2) if torch.cuda.is_available() else None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        held = round(torch.cuda.memory_allocated() / 2**30, 2) if torch.cuda.is_available() else None
        for r in (json.loads(df.to_json(orient="records")) if len(df) else []):
            r["cell_wall_seconds"] = wall
            r["cell_peak_gib"] = peak
            r["gib_still_allocated_after"] = held     # growing => something leaks across cells
            # upstream's row only keeps training.kind ("sgd"); a training override (batch_size) is in
            # the exca uid but invisible in the leaderboard, so stamp it on the row explicitly.
            r["batch_size"] = e.training.batch_size
            r["protocol_deviation"] = _deviation(e.training)
            rows.append(r)
        cells.append({"uid": e.infra.uid()[:60], "wall": wall, "peak_gib": peak, "held_gib": held})
        json.dump(dict(summary, rows=rows, cells=cells, partial=True), open(SUMMARY, "w"), indent=2, default=str)
    summary["run_seconds"] = round(time.time() - t0, 1)
    summary["rows"] = rows
    summary["cells"] = cells
    summary.pop("partial", None)
    summary["n_completed"] = sum(1 for r in rows if r.get("status") == "completed")
    summary["n_failed"] = sum(1 for r in rows if r.get("status") == "failed")


def main():
    cfg = load_cfg()
    if os.environ.get("NSGKIT_OEB_CHILD") == "1":
        summary = json.load(open(SUMMARY))
        for stale in ("shard_error", "trace", "rows", "cells", "partial"):   # a re-run must not inherit a prior failure
            summary.pop(stale, None)
        try:
            run_shard(cfg, summary)
        except Exception as e:                       # noqa: BLE001 - we want the trace in the artifact
            import traceback
            summary["shard_error"] = repr(e)
            summary["trace"] = traceback.format_exc()[-3000:]
        summary["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        json.dump(summary, open(SUMMARY, "w"), indent=2, default=str)
        return

    summary = {"schema": "nsg-agent-kit/oeb-shard/v1", "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
               "shard": cfg, "node": os.uname().nodename, "python": sys.version.split()[0]}
    os.makedirs(HF_HOME, exist_ok=True)
    json.dump(summary, open(SUMMARY, "w"), indent=2)
    if not install(cfg, summary):
        json.dump(summary, open(SUMMARY, "w"), indent=2)
        sys.exit(1)
    json.dump(summary, open(SUMMARY, "w"), indent=2)
    env = dict(os.environ, PYTHONPATH=LIBS + os.pathsep + os.environ.get("PYTHONPATH", ""),
               HF_HOME=HF_HOME, NSGKIT_OEB_CHILD="1", WANDB_MODE="disabled")
    if cfg.get("hf_token"):
        env["HF_TOKEN"] = cfg["hf_token"]
    r = subprocess.run([sys.executable, os.path.abspath(__file__)], env=env)
    sys.exit(r.returncode)


if __name__ == "__main__":
    main()
