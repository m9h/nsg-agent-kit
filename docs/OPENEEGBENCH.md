# OpenEEGBench on NSG — upstream status vs. what this kit has (2026-10-22)

The target in [`ROADMAP.md`](ROADMAP.md) is "run OpenEEGBench at scale on NSG". This page is the
periodic reality check: what upstream *is* now, what of it NSG can run, and which pieces of this
kit already cover each step. Facts tagged **[measured]** were checked on the date above.

## 1. Upstream, as of 2026-10-22 [measured]

| | |
|---|---|
| Package | `open-eeg-bench` **0.6.0** on PyPI (2026-06-02); last push to `main` 2026-07-09 (unreleased fix: regression MSE broadcast bug, #42) |
| Entry point | `oeb.benchmark(model_cls, hub_repo \| checkpoint_url \| checkpoint_path, datasets, finetuning_strategies, heads, n_seeds, device, infra)` → `pd.DataFrame` |
| Backbones (7) | `biot`, `labram`, `bendr`, `cbramod`, `signal_jepa`, `reve`, `eegpt` (added 0.6.0) — all `braindecode.models.*`, weights on HF (`braindecode/*`, REVE at gated `brain-bzh/reve-base`) |
| Strategies (9) | `frozen`, `ridge_probe` (closed-form, 0.3.0), `lora`, `ia3`, `adalora`, `dora`, `oft`, `full_finetune`, `two_stages` |
| Heads (3) | `linear_head`, `mlp_head`, `original_head` |
| Datasets (12) | pre-windowed, 100 Hz, hosted at `braindecode/*` on HF, **pinned to commit SHAs** since 0.5.0 |
| Runner | `exca.TaskInfra` → `submitit`: `cluster=None` (in-process) / `"local"` / `"slurm"` job array; results cached per content-addressed UID in `infra.folder` |
| Leaderboard | HF Space **OpenEEG-Bench Arena** (`braindecode/OpenEEGBench`, docker, last modified 2026-06-15): users submit a model, a worker runs `oeb.benchmark` and appends to `braindecode/contents`; **5 submissions / user / month**; evaluates 10 datasets (not `chbmit`, `seed_vig`) |
| Paper | Guetschel et al., EUSIPCO 2026, "Toward OpenEEG-Bench" |

What changed since this kit's July picture (`ROADMAP.md` "The matrix"): the dataset list is **not**
"braindecode/MOABB + EEGDash corpora" — it is a fixed set of 12 HF datasets; CBraMod and EEGPT are
in, the supervised baselines are only reachable via `ScratchBackbone`+`full_finetune`; and the
protocol list grew from 3 to 9.

### Dataset availability — the hard fact [measured]

| dataset | HF size | public? | upstream key |
|---|---|---|---|
| Arithmetic (Zyma 2019) | 0.12 GB | ✅ | `arithmetic_zyma2019` |
| BCI IV 2a | 0.34 GB | ✅ | `bcic2a` |
| BCI 2020-3 (imagined speech) | 0.85 GB | ✅ | `bcic2020_3` |
| MDD Mumtaz 2016 | 1.04 GB | ✅ | `mdd_mumtaz2016` |
| PhysioNet MI | 1.38 GB | ✅ | `physionet` |
| CHB-MIT | 4.69 GB | ✅ | `chbmit` |
| FACED | 4.70 GB | ✅ | `faced` |
| ISRUC-Sleep | 13.1 GB | ✅ | `isruc_sleep` |
| SEED-V, SEED-VIG, TUAB, TUEV | — | **❌ private (HTTP 401)** | licence forbids redistribution — upstream issue #46, closed 2026-09-11 |

So **8 of 12 datasets (~26 GB) are runnable by anyone; 4 are arena-only.** Any local/NSG
reproduction is a 8-dataset benchmark by construction; the 4 private ones can only be scored by
submitting the model to the arena. None of the 8 are on NEMAR (`nemar-data.md` — still true).

## 2. Can NSG run it? Step by step

| step | what upstream needs | NSG reality | kit coverage |
|---|---|---|---|
| Python | ≥3.11 | 3.11.4 on `PYTORCH_PY_EXPANSE` [measured 2026-09-22] | — |
| torch | `>=2.0`; lockfile is 2.11 | image 2.0.1+cu117; **`torch==2.4.1` (cu121) installs at runtime and keeps Volta** [measured, FINDINGS §6a] | `sweep/oeb_shard.py` pins it first |
| full dep stack | braindecode[hub] ≥1.6.1, peft ≥0.14, exca, skorch, zarr, hf-hub, wandb… | **resolves for cp311/manylinux with `torch==2.4.1 numpy<2`** → braindecode 1.8.1, peft 0.21.2, exca 0.6.0, transformers 5.20.2, 120 pkgs [measured dry-run 2026-10-22]. Must go to `$TMPDIR` (read-only venv). **Two more pins are mandatory**: `transformers==4.46.3`+`peft==0.14.0` (5.x transformers is broken on torch 2.4.1, see §5) and **`torchaudio` must be pinned to the torch version**: braindecode requires it unpinned; `uv` resolved 2.11.0 next to torch 2.4.1 and braindecode failed at import (`undefined symbol: aoti_torch_abi_version`) [measured locally 2026-10-22]; pip's resolver happened to pick 2.4.1 — don't rely on it | `oeb_shard.py` `install()` pins torch, torchaudio, numpy first |
| datasets | `BaseConcatDataset.pull_from_hub` → `snapshot_download` of a zarr repo at a pinned revision | runtime egress works [measured]; **no persistence → re-downloaded every job** (0.1–13 GB) → `HF_HOME=$TMPDIR/hf` so it is not returned | `oeb_shard.py` |
| weights | `hf_hub_download(hub_repo, "model.safetensors")` | public for 6 backbones; **REVE is gated** → pass `checkpoint_path` to the 264 MB file we already vendor (`sweep/reve_weights/`, gitignored) | `oeb_plan.py --backbones reve` copies it into the zip |
| scheduler | `cluster="slurm"` → submitit `sbatch` | **not available** — NSG jobs cannot call `sbatch`; the gateway *is* the scheduler | run with `cluster=None` inside each NSG job; fan-out is ours (`oeb_plan.py plan`) |
| result cache | exca folder `results/<Class>.run,<ver>/<uid>/{config.yaml,job.pkl}` | returned in `output.tar.gz`; **cache is relocatable (no absolute paths) and UIDs are content-addressed** [measured 2026-10-22 with exca 0.6.0] → caches from many jobs pool by copy | `oeb_plan.py merge` |
| collection | `collect_completed_results()` / `benchmark(..., infra={"mode":"read-only"})` | runs locally on the pooled folder | `oeb_plan.py merge` prints the call |
| wall-clock | one experiment = ≤30 epochs, early stop | 48 h/job; a (dataset × seed) shard of 7 backbones × 9 strategies = 63 cells fits for the small datasets, **not** for `isruc_sleep`/`faced` → split those by backbone | `oeb_plan.py` shards per (dataset, seed); add `--backbones` per shard for big sets |
| wandb | optional dependency, imported by `Training` | no account on the node | `WANDB_MODE=disabled` in `oeb_shard.py` |

**Verdict:** NSG can run the public 8-dataset benchmark with **upstream code unmodified**, on the
gateway, without Apptainer. The only adaptation is *where the fan-out happens*: upstream fans out
via SLURM arrays; here each NSG job runs one in-process shard and the exca caches are pooled
afterwards. Nothing in the scoring path is ours.

## 3. What this kit already had, and what was added

Already in place before this check (all [measured] on Expanse in July 2026):
- platform facts + pins (`FINDINGS.md`, `LIMITATIONS.md`, `env-snapshots/`): read-only venv,
  `--target $TMPDIR`, `numpy<2`, `torch==2.4.1` for peft, egress works, no persistence;
- `sweep/cell.py`: a *re-implementation* of one cell (EEGNet / frozen REVE / REVE+LoRA on MOABB
  `BNCI2014_001` or a NEMAR `ds`) — useful as the platform probe it was, but its numbers are not
  OpenEEGBench numbers (different windows, splits, preprocessing);
- `sweep/repro_*`: the BCI-IV-2a EEGNet-vs-REVE reproduction (`REPRODUCTION.md`);
- vendored REVE weights (the gated-repo workaround);
- `nsgr/nsgr.sh` submit/status/fetch.

Added 2026-10-22:
- `sweep/oeb_shard.py` — the per-job driver that runs **upstream** `open_eeg_bench` experiments
  with `cluster=None` into `./results` (exca cache) and writes `shard_summary.json`;
- `sweep/oeb_plan.py` — `plan` (one zip per dataset × seed), `submit` (every shard zip through
  `nsgr/nsgr.sh` with per-shard `filename_`/`subdirname_` vparams → `jobs.json`; exits 75 on the
  NSG-R staging outage, `LIMITATIONS.md` §5a), `poll` (refresh stages, fetch finished jobs into
  `fetched/<shard>/`, exit 0 only when all are fetched), `merge` (pool caches + leaderboard from the
  summaries, stdlib) and `collect` (re-derive configs → upstream `collect_completed_results` on the
  pooled cache in read-only mode — the numbers come out of upstream's collector);
- `probes/tests/test_oeb_shard_contract.py` — RED until a shard has been fetched; GREEN asserts
  **NSG provenance** (node `exp-*`, V100 — a laptop run copied into `fetched/` stays RED unless
  `OEB_ALLOW_LOCAL=1`), the pinned stack, every cell completed, every classification cell above chance.

The student path, start to finish:

```bash
python sweep/oeb_plan.py plan --name smoke --backbones biot --datasets arithmetic_zyma2019 --strategies frozen ridge_probe
set -a; source nsgr/config.env; read -s NSG_PASSWORD; set +a
until python sweep/oeb_plan.py submit --name smoke; do sleep 300; done   # waits out a staging outage
until python sweep/oeb_plan.py poll   --name smoke; do sleep 120; done   # queue + run, ~10 min for smoke
python sweep/oeb_plan.py merge --name smoke && python sweep/oeb_plan.py collect --name smoke
pytest probes/tests/test_oeb_shard_contract.py -q
```

## 3a. Local smoke run of the shard driver [measured 2026-10-22, Legion RTX 2080, Python 3.11]

Same pins the NSG job installs (`torch==2.4.1 torchaudio==2.4.1 numpy<2 transformers==4.46.3
peft==0.14.0 open-eeg-bench==0.6.0`), same driver, `shard.json` = BIOT × `arithmetic_zyma2019` ×
{frozen, ridge_probe} × seed 0:

| | |
|---|---|
| dataset pull | `braindecode/arithmetic_zyma2019@8359206` → 379 MB in `HF_HOME` (zarr + weights) |
| data | 1,343 train / 172 val / 192 test windows, (19 ch × 1000) @ 200 Hz |
| BIOT frozen + linear head | 5,378 / 3,187,458 trainable (0.2 %); fit 46 s; **test balanced acc 0.576** (chance 0.5) |
| BIOT ridge_probe | **rejected by upstream's validator** (`training_required_modules=['encoder.channel_tokens']` needs SGD) — driver records it under `skipped_invalid`, shard continues |
| whole shard | 54 s; exca cache 16 KB; `oeb_plan.py merge` pools it; `oeb_plan.py collect` re-derives the config and upstream's `collect_completed_results` returns the **same row (0.576389) from the pooled cache without recomputing**; re-running the shard is deterministic (0.5764 twice) |
| contract | `test_oeb_shard_contract.py` **GREEN** on the local artifact |

Three things this bought that the dry-run resolve could not: the two companion pins above
(torchaudio, transformers/peft), and the fact that `ridge_probe` is not universal — upstream's own
rules exclude it for BIOT (and any backbone with `training_required_*`), so the 9-strategy count
in §4 is an upper bound. **Still unmeasured: the same shard on NSG itself** (V100, read-only venv,
`$TMPDIR` install) — that is the remaining step, and the contract is written for it.

## 4. Sizing the public benchmark on NSG

7 backbones × 8 datasets × 9 strategies × 1 head × 3 seeds = **1,512 experiments**; as
(dataset × seed) shards that is **24 NSG jobs** of 63 cells each (fewer, bigger downloads per job),
or 168 jobs if also split by backbone. Per-job overhead: ~3 min pip + the dataset download
(≈1 min/GB observed for PyPI-scale egress; unmeasured for HF). The 13 GB ISRUC download repeated
per job is the single largest waste — it argues for `--backbones` shards of *all* strategies for
ISRUC so the download is paid 7×3 times, not 63×3. Fair-use note (`LIMITATIONS.md` §1): submit in
batches of ≤10 and back off on 429.

## 5. Open items

- [ ] **Run the smoke shard on NSG** (`sweep/oeb/smoke/shards/arithmetic_zyma2019_s0.zip`) — it is
      GREEN locally (§3a); the NSG run is what makes the V100/`$TMPDIR` path a measured fact.
      **2026-10-22: submission attempted and refused by NSG-R** (`error creating tmp file for input
      parameter input.infile_`, the staging outage of `LIMITATIONS.md` §5a; the July probe zip fails
      identically, so it is the service). A retry loop is submitting every 5 min; result goes in §3b.
- [ ] Measure HF download throughput on the node (sets the shard shape for ISRUC/FACED/CHB-MIT).
- [x] `transformers 5.20` in the resolve **is** fatal on torch 2.4.1: `peft 0.21` imports it eagerly
      and it dies with `NameError: name 'nn' is not defined` (`transformers/integrations/accelerate.py`).
      Pinned **`transformers==4.46.3` + `peft==0.14.0`** (the oldest peft open-eeg-bench accepts);
      imports clean [measured locally 2026-10-22, Python 3.11 venv].
- [ ] **Upstream bug to report:** `oeb.benchmark()` has no `training_required_modules` /
      `training_required_parameters` kwargs, so the pattern in `examples/run_on_slurm.py` (factory
      fields → `benchmark()`) fails at `build()` for BIOT (`encoder.channel_tokens` absent from the
      checkpoint → "declare in training_required_modules") and presumably LaBraM (`temporal_embedding`)
      [measured locally 2026-10-22]. The arena worker also goes through `benchmark()`. Our driver
      builds `Experiment`s from the full `PretrainedBackbone` dump instead (`oeb_shard.build_experiments`).
- [ ] Decide whether arena submission (5/month, needs a public `hub_repo` or `checkpoint_url`) is
      the route for the 4 private datasets — it is the *only* route.
- [ ] Upstream's unreleased #42 fix matters only for `seed_vig` (private) — irrelevant for us until
      a 0.6.1 release.
