#!/usr/bin/env python3
"""Plan, package, and merge an OpenEEGBench sweep on NSG.

  plan   -> write one shard dir + zip per (dataset, seed) under sweep/oeb/<name>/shards/
  merge  -> untar fetched NSG outputs, pool their exca caches, write leaderboard.csv from the
            shard summaries (stdlib only)
  collect-> re-derive the configs with upstream benchmark() and read the pooled cache in
            mode=read-only (never recomputes) -> leaderboard_upstream.csv

    python sweep/oeb_plan.py plan  --name smoke --backbones biot --datasets arithmetic_zyma2019 \
        --strategies frozen ridge_probe --seeds 0
    for z in sweep/oeb/smoke/shards/*.zip; do ./nsgr/nsgr.sh submit PYTORCH_PY_EXPANSE "$z"; done
    # ... fetch each job's output.tar.gz into sweep/oeb/smoke/fetched/<shard>/ ...
    python sweep/oeb_plan.py merge --name smoke
    python sweep/oeb_plan.py collect --name smoke      # needs open-eeg-bench importable

Runs locally with only the stdlib; `merge` needs open-eeg-bench importable (pip install
open-eeg-bench) because it hands the pooled cache back to upstream's collector.
"""
import argparse
import glob
import json
import os
import shutil
import tarfile
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
PUBLIC_DATASETS = ["arithmetic_zyma2019", "bcic2a", "bcic2020_3", "physionet",
                   "chbmit", "faced", "isruc_sleep", "mdd_mumtaz2016"]
# seed_v, seed_vig, tuab, tuev are PRIVATE on HF (licence; upstream issue #46) -> arena-only.
BACKBONES = ["biot", "labram", "bendr", "cbramod", "signal_jepa", "reve", "eegpt"]
STRATEGIES = ["frozen", "ridge_probe", "lora", "ia3", "adalora", "dora", "oft",
              "full_finetune", "two_stages"]


def plan(a):
    root = os.path.join(HERE, "oeb", a.name)
    shards_dir = os.path.join(root, "shards")
    os.makedirs(shards_dir, exist_ok=True)
    manifest = []
    for ds in a.datasets:
        for seed in a.seeds:
            sid = f"{ds}_s{seed}"
            d = os.path.join(shards_dir, sid)
            os.makedirs(d, exist_ok=True)
            shutil.copy(os.path.join(HERE, "oeb_shard.py"), os.path.join(d, "run.py"))
            cfg = {"backbones": a.backbones, "datasets": [ds], "strategies": a.strategies,
                   "heads": a.heads, "seeds": [seed], "device": a.device,
                   "torch": a.torch, "oeb": a.oeb}
            if "reve" in a.backbones and os.path.isfile(a.reve_weights):
                os.makedirs(os.path.join(d, "reve_weights", "reve-base"), exist_ok=True)
                shutil.copy(a.reve_weights, os.path.join(d, "reve_weights", "reve-base", "model.safetensors"))
                cfg["checkpoints"] = {"reve": "reve_weights/reve-base/model.safetensors"}
            json.dump(cfg, open(os.path.join(d, "shard.json"), "w"), indent=2)
            # NSG input: one zip, exactly one top-level dir, entry = <dir>/run.py (tool-params.md)
            zpath = os.path.join(shards_dir, sid + ".zip")
            with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
                for base, _, files in os.walk(d):
                    for f in files:
                        p = os.path.join(base, f)
                        z.write(p, os.path.join(sid, os.path.relpath(p, d)))
            manifest.append({"shard": sid, "zip": os.path.relpath(zpath, HERE),
                             "n_cells": len(a.backbones) * len(a.strategies) * len(a.heads),
                             "zip_bytes": os.path.getsize(zpath)})
    json.dump({"name": a.name, "shards": manifest, "vparams_hint": {
        "runtime": a.runtime_hours, "number_gpus": 1, "number_gbmemorypernode": 96}},
        open(os.path.join(root, "manifest.json"), "w"), indent=2)
    tot = sum(m["n_cells"] for m in manifest)
    print(f"{len(manifest)} shards, {tot} experiments -> {root}")
    for m in manifest:
        print(f"  {m['shard']:28s} {m['n_cells']:4d} cells  {m['zip_bytes']/1e6:7.1f} MB")


def merge(a):
    root = os.path.join(HERE, "oeb", a.name)
    pooled = os.path.join(root, "pooled_results")
    os.makedirs(pooled, exist_ok=True)
    summaries = []
    for tgz in glob.glob(os.path.join(root, "fetched", "*", "output.tar.gz")):
        shard = os.path.basename(os.path.dirname(tgz))
        ex = os.path.join(root, "fetched", shard, "extracted")
        if not os.path.isdir(ex):
            with tarfile.open(tgz) as t:
                t.extractall(ex, filter="data")
        for cache in glob.glob(os.path.join(ex, "**", "results"), recursive=True):
            # exca layout: results/<Class>.run,<version>/<uid>/{config.yaml,job.pkl,...}; uid is
            # content-addressed so pooling across shards cannot collide.
            shutil.copytree(cache, pooled, dirs_exist_ok=True)
        for s in glob.glob(os.path.join(ex, "**", "shard_summary.json"), recursive=True):
            summaries.append(json.load(open(s)))
    print(f"pooled {len(summaries)} shard summaries into {pooled}")
    rows = [r for s in summaries for r in s.get("rows", [])]
    failed = [r for r in rows if r.get("status") != "completed"]
    if rows:
        import csv
        keys = sorted({k for r in rows for k in r})
        with open(os.path.join(root, "leaderboard.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
        print(f"{len(rows)} rows ({len(failed)} not completed) -> {root}/leaderboard.csv")
    json.dump({"name": a.name, "n_rows": len(rows), "n_failed": len(failed),
               "pooled_cache": os.path.relpath(pooled, HERE), "shards": [s.get("shard") for s in summaries]},
              open(os.path.join(root, "merge_summary.json"), "w"), indent=2)
    print("Upstream can now read the pooled cache directly:\n"
          "  oeb.benchmark(..., infra={'folder': '%s', 'mode': 'read-only'})" % pooled)


def collect(a):
    """Rebuild the shard configs exactly as the shards did (oeb_shard.build_experiments on the stored
    backbone dumps) and read the pooled exca cache in read-only mode -- nothing is recomputed."""
    import sys
    sys.path.insert(0, HERE)
    from oeb_shard import build_experiments
    from open_eeg_bench.experiment import collect_completed_results
    root = os.path.join(HERE, "oeb", a.name)
    pooled = os.path.join(root, "pooled_results")
    infra = {"folder": pooled, "cluster": None, "mode": "read-only"}
    experiments = []
    for s in glob.glob(os.path.join(root, "fetched", "*", "extracted", "**", "shard_summary.json"), recursive=True):
        sm = json.load(open(s)); sh = sm["shard"]
        for bb_name, bb_dict in sm.get("backbone_configs", {}).items():
            exps, _ = build_experiments(bb_dict, sh.get("datasets"), sh.get("heads"), sh.get("strategies", ["frozen"]),
                                        sh.get("seeds", [0]), sh.get("device", "cuda"), infra)
            experiments += exps
    missing = [e.infra.uid()[:80] for e in experiments if e.infra.status() == "not submitted"]
    for m in missing:
        print(f"  MISSING in pooled cache: {m}...")
    df = collect_completed_results(experiments, collect_all=True)
    df.to_csv(os.path.join(root, "leaderboard_upstream.csv"), index=False)
    cols = [c for c in ("backbone", "dataset", "finetuning", "head", "seed", "status", "test_balanced_accuracy", "test_r2") if c in df]
    print(df[cols].to_string() if len(df) else "no rows")
    print(f"-> {root}/leaderboard_upstream.csv  ({len(df)} rows via upstream collect_completed_results, read-only)")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    pp = sub.add_parser("plan")
    pp.add_argument("--name", required=True)
    pp.add_argument("--backbones", nargs="+", default=["biot"], choices=BACKBONES)
    pp.add_argument("--datasets", nargs="+", default=["arithmetic_zyma2019"], choices=PUBLIC_DATASETS)
    pp.add_argument("--strategies", nargs="+", default=["frozen"], choices=STRATEGIES)
    pp.add_argument("--heads", nargs="+", default=["linear_head"])
    pp.add_argument("--seeds", nargs="+", type=int, default=[0])
    pp.add_argument("--device", default="cuda")
    pp.add_argument("--torch", default="2.4.1")
    pp.add_argument("--oeb", default="0.6.0")
    pp.add_argument("--runtime-hours", type=float, default=4.0)
    pp.add_argument("--reve-weights", default=os.path.join(HERE, "reve_weights", "reve-base", "model.safetensors"))
    pp.set_defaults(fn=plan)
    pm = sub.add_parser("merge")
    pm.add_argument("--name", required=True)
    pm.set_defaults(fn=merge)
    pc = sub.add_parser("collect", help="read the pooled cache through upstream benchmark(mode=read-only)")
    pc.add_argument("--name", required=True)
    pc.set_defaults(fn=collect)
    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
