#!/usr/bin/env python3
"""Plan, package, and merge an OpenEEGBench sweep on NSG.

  plan   -> write one shard dir + zip per (dataset, seed) under sweep/oeb/<name>/shards/
  submit -> submit every shard zip to NSG through nsgr/nsgr.sh with the right vparams
            (filename_/subdirname_ per shard) -> sweep/oeb/<name>/jobs.json
  poll   -> ask NSG-R for each job's stage; fetch output.tar.gz of finished jobs into fetched/<shard>/
  merge  -> untar fetched NSG outputs, pool their exca caches, write leaderboard.csv from the
            shard summaries (stdlib only)
  collect-> re-derive the configs with upstream benchmark() and read the pooled cache in
            mode=read-only (never recomputes) -> leaderboard_upstream.csv

    python sweep/oeb_plan.py plan  --name smoke --backbones biot --datasets arithmetic_zyma2019 \
        --strategies frozen ridge_probe --seeds 0
    set -a; source nsgr/config.env; read -s NSG_PASSWORD; set +a   # creds in env, never on disk
    python sweep/oeb_plan.py submit --name smoke          # -> jobs.json (one NSG job per shard)
    python sweep/oeb_plan.py poll --name smoke            # repeat until every job is fetched
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
import subprocess
import tarfile
import time
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
NSGR = os.path.join(os.path.dirname(HERE), "nsgr", "nsgr.sh")
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
    # (dataset, seed) is the shard unit; the big datasets (OPENEEGBENCH.md §4: chbmit/faced 4.7 GB,
    # isruc_sleep 13 GB) are split further by backbone so one shard stays well inside 48 h.
    units = []
    for ds in a.datasets:
        for seed in a.seeds:
            if ds in (a.split_by_backbone or []):
                units += [(f"{ds}_{bb}_s{seed}", ds, seed, [bb], a.split_runtime_hours) for bb in a.backbones]
            else:
                units.append((f"{ds}_s{seed}", ds, seed, a.backbones, a.runtime_hours))
    for sid, ds, seed, backbones, runtime in units:
        if True:
            d = os.path.join(shards_dir, sid)
            os.makedirs(d, exist_ok=True)
            shutil.copy(os.path.join(HERE, "oeb_shard.py"), os.path.join(d, "run.py"))
            cfg = {"backbones": backbones, "datasets": [ds], "strategies": a.strategies,
                   "heads": a.heads, "seeds": [seed], "device": a.device,
                   "torch": a.torch, "oeb": a.oeb, "peft": a.peft, "transformers": a.transformers}
            if a.training_override:
                tov = {}
                for spec in a.training_override:            # backbone:key=value, e.g. eegpt:batch_size=16
                    bb, kv = spec.split(":", 1); k, v = kv.split("=", 1)
                    tov.setdefault(bb, {})[k] = int(v) if v.isdigit() else float(v) if v.replace(".", "", 1).isdigit() else v
                cfg["training_overrides"] = tov
            # brain-bzh/reve-base downloads anonymously since at least 2026-10-23 (277 MB, HTTP 200);
            # vendoring is only needed if HF gates it again -> --vendor-reve
            if a.vendor_reve and "reve" in backbones and os.path.isfile(a.reve_weights):
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
                             "n_cells": len(backbones) * len(a.strategies) * len(a.heads),
                             "runtime": runtime, "zip_bytes": os.path.getsize(zpath)})
    json.dump({"name": a.name, "shards": manifest, "vparams_hint": {
        "runtime": a.runtime_hours, "number_gpus": 1, "number_gbmemorypernode": 96}},
        open(os.path.join(root, "manifest.json"), "w"), indent=2)
    tot = sum(m["n_cells"] for m in manifest)
    print(f"{len(manifest)} shards, {tot} experiments -> {root}")
    for m in manifest:
        print(f"  {m['shard']:34s} {m['n_cells']:4d} cells  {m['runtime']:4.0f} h  {m['zip_bytes']/1e6:7.1f} MB")


def _shard_summary(root, sid):
    """shard_summary.json out of fetched/<sid>/output.tar.gz, or None if the shard came back empty."""
    tgz = os.path.join(root, "fetched", sid, "output.tar.gz")
    if not os.path.isfile(tgz):
        return None
    with tarfile.open(tgz) as t:
        m = [x for x in t.getmembers() if x.name.endswith("shard_summary.json")]
        return json.load(t.extractfile(m[0])) if m else None


def lost_shards(root, jobs):
    """Fetched shards with nothing usable: no tarball (NSG disk full, FINDINGS §9), no summary, or
    zero completed cells (e.g. the HF rate limit took the dataset out). These are re-run, not results."""
    out = []
    for sid, j in jobs.items():
        if not j.get("fetched"):
            continue
        s = _shard_summary(root, sid)
        if s is None or not s.get("n_completed"):
            out.append(sid)
    return out


def _zip_with_token(zpath, sid, token):
    """The committed shard zip never holds a secret. With --hf-token-file, a copy is made under
    $TMPDIR (or the scratchpad) with `<sid>/hf_token` added; run.py reads and deletes it."""
    if not token:
        return zpath
    tmp = os.path.join(os.environ.get("TMPDIR", "/tmp"), "nsgkit-submit")
    os.makedirs(tmp, mode=0o700, exist_ok=True)
    out = os.path.join(tmp, os.path.basename(zpath))
    shutil.copy(zpath, out)
    with zipfile.ZipFile(out, "a") as z:
        z.writestr(f"{sid}/hf_token", token + "\n")
    return out


def _nsgr(args, extra=""):
    """nsgr.sh needs NSG_URL/USER/APPKEY/PASSWORD in the env (source nsgr/config.env; read -s
    NSG_PASSWORD). NSG_EXTRA is overridden per call so a stale one in config.env can't point a
    shard at the wrong subdirname."""
    for k in ("NSG_URL", "NSG_USER", "NSG_APPKEY", "NSG_PASSWORD"):
        if not os.environ.get(k):
            raise SystemExit(f"{k} not set: `set -a; source nsgr/config.env; read -s NSG_PASSWORD; set +a`")
    env = dict(os.environ, NSG_EXTRA=extra)
    r = subprocess.run([NSGR, *args], env=env, capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"nsgr.sh {' '.join(args)} failed:\n{r.stderr[-2000:]}")
    return r.stdout


def _tag(xml, name):
    import re
    m = re.search(rf"<{name}>([^<]*)</{name}>", xml)
    return m.group(1).strip() if m else None


def submit(a):
    root = os.path.join(HERE, "oeb", a.name)
    man = json.load(open(os.path.join(root, "manifest.json")))
    hint = man.get("vparams_hint", {})
    jobs_path = os.path.join(root, "jobs.json")
    jobs = json.load(open(jobs_path)) if os.path.isfile(jobs_path) else {}
    tool = os.environ.get("NSG_TOOL", "PYTORCH_PY_EXPANSE")
    # fair use (LIMITATIONS.md §1): never fan the whole matrix out at once. --max-inflight N submits
    # only while fewer than N of this plan's jobs are unfetched; a `poll; submit` loop tops it up.
    inflight = sum(1 for j in jobs.values() if not j.get("fetched"))
    token = open(os.path.expanduser(a.hf_token_file)).read().strip() if a.hf_token_file else None
    redo = set(lost_shards(root, jobs)) if a.redo_lost else set()
    for m in man["shards"]:
        sid = m["shard"]
        if a.only and sid not in a.only:
            continue
        if sid in jobs and not a.resubmit and sid not in redo:
            continue
        if a.max_inflight and inflight >= a.max_inflight:
            left = sum(1 for x in man["shards"] if x["shard"] not in jobs) + len(redo)
            print(f"  {inflight} in flight >= --max-inflight {a.max_inflight}; ~{left} shard(s) wait")
            break
        if sid in jobs:                                   # re-run: keep the old attempt's files aside
            old = os.path.join(root, "fetched", sid)
            if os.path.isdir(old):
                os.makedirs(os.path.join(root, "fetched_lost"), exist_ok=True)
                shutil.move(old, os.path.join(root, "fetched_lost", f"{sid}_{jobs[sid]['handle'][-8:]}"))
            prev = jobs[sid].get("previous", []) + [jobs[sid]["handle"]]
        else:
            prev = []
        vp = {"filename": "run.py", "subdirname": sid,
              "runtime": a.runtime_hours or m.get("runtime") or hint.get("runtime", 4.0),
              "number_gpus": hint.get("number_gpus", 1),
              "number_gbmemorypernode": a.gb_memory or hint.get("number_gbmemorypernode", 96)}
        extra = " ".join(f"-F vparam.{k}_={v}" for k, v in vp.items())
        xml = _nsgr(["submit", tool, _zip_with_token(os.path.join(HERE, m["zip"]), sid, token)], extra)
        handle = _tag(xml, "jobHandle")
        if not handle:
            if "error creating tmp file for input parameter" in xml:
                # NSG-R upload staging is down server-side (seen 2026-07-26 and 2026-10-22; GET/auth
                # fine, every tool affected, a July zip that ran fine fails identically). Not our bug:
                # retry later. Exit 75 = EX_TEMPFAIL so a wrapper loop can tell it from a real error.
                print(f"  {sid}: NSG-R upload staging outage ('error creating tmp file'); "
                      f"nothing was submitted. Retry in a few minutes; see docs/LIMITATIONS.md §NSG-R outages.")
                raise SystemExit(75)
            raise SystemExit(f"{sid}: no jobHandle in NSG response:\n{xml[-2000:]}")
        jobs[sid] = {"handle": handle, "tool": tool, "vparams": vp, "submitted": time.strftime("%Y-%m-%dT%H:%M:%S"),
                     "stage": _tag(xml, "jobStage"), "hf_token": bool(token)}
        if prev:
            jobs[sid]["previous"] = prev
        print(f"  {sid}: {handle}  ({jobs[sid]['stage']})")
        json.dump(jobs, open(jobs_path, "w"), indent=2)
        inflight += 1
    print(f"-> {jobs_path}  ({len(jobs)}/{len(man['shards'])} submitted)")


def poll(a):
    """One pass over jobs.json: refresh stage, fetch finished jobs once. Exit code 0 only when every
    job has been fetched (so `until oeb_plan.py poll --name X; do sleep 120; done` is a waiter)."""
    root = os.path.join(HERE, "oeb", a.name)
    jobs_path = os.path.join(root, "jobs.json")
    jobs = json.load(open(jobs_path))
    pending = 0
    for sid, j in jobs.items():
        if j.get("fetched"):
            print(f"  {sid}: fetched -> {j['fetched']}")
            continue
        xml = _nsgr(["status", j["handle"]])
        j["stage"] = _tag(xml, "jobStage"); j["failed"] = _tag(xml, "failed") == "true"
        terminal = _tag(xml, "terminalStage") == "true"
        msgs = [m for m in xml.split("<message>")[1:]]
        j["last_message"] = msgs[-1].split("</message>")[0][:200] if msgs else None
        if terminal:
            dest = os.path.join(root, "fetched", sid)
            _nsgr(["fetch", j["handle"], dest])
            j["fetched"] = os.path.relpath(dest, HERE)
            got = sorted(os.listdir(dest))
            print(f"  {sid}: {j['stage']} failed={j['failed']} -> fetched {got}")
            if not any(f.endswith(".tar.gz") for f in got):
                print(f"    !! no output.tar.gz for {sid}: read STDERR/STDOUT in {dest}")
        else:
            pending += 1
            print(f"  {sid}: {j['stage']}  {j['last_message'] or ''}")
    json.dump(jobs, open(jobs_path, "w"), indent=2)
    done = sum(1 for j in jobs.values() if j.get("fetched"))
    lost = lost_shards(root, jobs)
    if lost:
        print(f"  !! {len(lost)} fetched shard(s) with no usable result (submit --redo-lost): {' '.join(lost)}")
    if pending:
        raise SystemExit(f"{done} fetched ({len(lost)} lost), {pending} job(s) still running; run poll again")
    print("all jobs fetched -> `oeb_plan.py merge --name %s`" % a.name)


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
                                        sh.get("seeds", [0]), sh.get("device", "cuda"), infra,
                                        training=sh.get("training_overrides", {}).get(bb_name))
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
    # peft 0.14.0 lacks target_parameters / all-linear on plain nn.Modules / new OFT semantics that
    # upstream's finetuning.py uses -> 33 of 59 bench1 cells failed (2026-10-23). 0.21.2 imports
    # cleanly next to transformers 4.46.3 + torch 2.4.1 [measured locally].
    pp.add_argument("--peft", default="peft==0.21.2")
    pp.add_argument("--transformers", default="transformers==4.46.3")
    pp.add_argument("--training-override", nargs="*", default=[], metavar="BACKBONE:KEY=VALUE",
                    help="deviate from upstream Training defaults for one backbone (e.g. eegpt:batch_size=16 "
                         "for the V100); changes the cell's uid and is recorded in the row")
    pp.add_argument("--runtime-hours", type=float, default=4.0)
    pp.add_argument("--split-by-backbone", nargs="*", default=[], choices=PUBLIC_DATASETS, metavar="DATASET",
                    help="make one shard per backbone for these (big) datasets instead of one per dataset")
    pp.add_argument("--split-runtime-hours", type=float, default=48.0, help="runtime for the split shards")
    pp.add_argument("--reve-weights", default=os.path.join(HERE, "reve_weights", "reve-base", "model.safetensors"))
    pp.add_argument("--vendor-reve", action="store_true", help="ship local REVE weights in the zip (only if HF gates the repo)")
    pp.set_defaults(fn=plan)
    ps = sub.add_parser("submit", help="submit every shard zip to NSG (needs NSG_* in env)")
    ps.add_argument("--name", required=True)
    ps.add_argument("--runtime-hours", type=float, default=None, help="override manifest vparams_hint")
    ps.add_argument("--gb-memory", type=int, default=None)
    ps.add_argument("--resubmit", action="store_true")
    ps.add_argument("--only", nargs="*", default=None, metavar="SHARD", help="restrict to these shard ids")
    ps.add_argument("--redo-lost", action="store_true",
                    help="also resubmit fetched shards that came back with no tarball / no completed cell")
    ps.add_argument("--hf-token-file", default=None, metavar="PATH",
                    help="read a Hugging Face token (e.g. ~/.cache/huggingface/token) and ship it in a submit-time "
                         "copy of the zip as <shard>/hf_token; lifts HF's per-IP anonymous rate limit (LIMITATIONS §6)")
    ps.add_argument("--max-inflight", type=int, default=None,
                    help="submit only while fewer than N jobs of this plan are unfetched (fair use)")
    ps.set_defaults(fn=submit)
    po = sub.add_parser("poll", help="refresh job stages; fetch finished jobs into fetched/<shard>/")
    po.add_argument("--name", required=True)
    po.set_defaults(fn=poll)
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
