# Target: run OpenEEGBench at scale on NSG

## North star

Make **NSG (free SDSC Expanse GPUs) the compute backend for the entire OpenEEGBench
evaluation matrix** — every EEG foundation model × every dataset × every probing protocol —
run reproducibly, driven by an agent, with results fetched back and validated by TDD contracts.
The LoRA-REVE fine-tune sweep is the first-class research payload that rides the same rails.

Why NSG for this: OpenEEGBench is *embarrassingly parallel* (each model×dataset×protocol cell is
independent) and NSG gives free GPU time on Expanse (up to 4× V100/node, 72 nodes, 48h/job).
One fan-out of the matrix into NSG-R jobs turns a benchmark that would saturate a lab GPU for weeks
into an overnight batch — at zero marginal cost.

## The matrix (pinned from upstream 0.6.0 — see [`OPENEEGBENCH.md`](OPENEEGBENCH.md))

- **Backbones (7):** BIOT, LaBraM, BENDR, CBraMod, Signal-JEPA, **REVE**, EEGPT — all
  `braindecode.models.*` with HF weights (REVE gated → vendored). Supervised baselines only via
  `ScratchBackbone` + `full_finetune`.
- **Datasets (12, fixed, HF `braindecode/*`, pinned revisions):** **8 public** (arithmetic,
  bcic2a, bcic2020-3, physionet, chbmit, faced, isruc-sleep, mdd_mumtaz2016; ~26 GB) and **4
  private** (SEED-V, SEED-VIG, TUAB, TUEV — licence; arena-only).
- **Strategies (9):** frozen, ridge_probe, lora, ia3, adalora, dora, oft, full_finetune, two_stages.

Public matrix: 7 × 8 × 9 × 3 seeds = **1,512 experiments = 24 NSG jobs** as (dataset × seed)
shards — see `OPENEEGBENCH.md` §4.

## Milestones (each with a red→green gate, in the kit's TDD style)

| # | Milestone | Green gate | Status |
|---|---|---|---|
| **M0** | **Platform probe** — env/GPU/network/NEMARPATH facts | `probes/reve_frozen_probe` returns valid `metrics.json`; contract test GREEN | **in flight** (job `NGBW-JOB-PYTORCH_PY_EXPANSE-…`) |
| **M1** | **Dependency reality** — do `peft`/`braindecode`/`transformers` vendor against torch 2.0.1+cu117, or do we need Apptainer? | `templates/pytorch-gpu` job imports all real deps on the node; decide Strategy B vs C (`docs/dependencies.md`) | **DONE — quick win.** Runtime `pip install mne` worked in 11 s (`nemar_load`). Deps install at runtime; only torch≥2.2 needs Apptainer. |
| **M2** | **One real cell** — a single (model, dataset, protocol) OpenEEGBench cell end-to-end on NSG | a job produces a valid per-cell result (accuracy/AUROC) fetched back + schema-validated | **DONE (pipeline)** — `m2_cell`: EEGNet on ds002718 FaceRecognition (74ch, 650 trials) → test bal-acc 0.975 vs chance 0.769 on the V100. **Caveat:** the auto-picked 2-class contrast (most-frequent `trial_type`s) is not a curated scientific task — real cells need proper task defs. |
| **M3** | **Data staging** — get benchmark datasets onto Expanse without runtime internet | datasets resolvable from `$NEMARPATH` (OpenNeuro `ds`) or a one-time staged cache; job reads them offline | **largely done** — 552 OpenNeuro `ds` live at `/expanse/projects/nemar/openneuro/` (git-annex, materialized). Remaining: map the oeb dataset list onto available `ds` ids; HF-only sets still need staging |
| **M4** | **Sweep harness** — fan the full matrix out as many NSG-R jobs, track/fetch/aggregate | `sweep/oeb_plan.py plan` → N shard zips; `merge` pools exca caches + leaderboard; `test_oeb_shard_contract.py` asserts every cell GREEN | **smoke shard GREEN on NSG 2026-10-22** (`sweep/oeb_shard.py` runs *upstream* `open_eeg_bench` in-process per job on exp-9-57/V100; `submit`/`poll`/`merge`/`collect` round-trip; BIOT frozen arithmetic 0.576389 = local; `OPENEEGBENCH.md` §3b). bench1/1b/1c (FINDINGS §8a–c) settled the pins (peft 0.21.2) and the EEGPT/V100 ceiling; **`public1` = 8 public datasets × seed 0 × 7 × 9 as 26 jobs, in flight 2026-10-23** (`OPENEEGBENCH.md` §4) |
| **M5** | **LoRA-REVE sweep** — the research payload: frozen REVE + LoRA across datasets/hyperparams | LoRA results table reproduced on NSG; matches (± tol) a local reference cell | **in flight** — LoRA adapter builds correctly (r=32, 7.68% trainable, right target modules) on BNCI2014_001/subject 1. Earlier "no kernel image" error was **NOT a node upgrade** (§6a, measured): `peft` pulled a fresh cu130 torch into the job that dropped Volta/V100 support; the image torch is unchanged (2.0.1+cu117). Fix: pin `torch==2.4.1` (cu121, has `torch.nn.attention`, keeps sm_70) into the target via `cell.json`. See docs/FINDINGS.md §6/§6a. |

## Execution substrate: NSG gateway vs direct Expanse (a strategic fork)

Everything above assumes the **NSG gateway** (zip upload, free, no allocation paperwork, fixed
Singularity tool envs, agent-drivable via NSG-R). Its ceiling is torch 2.0.1/cu117 unless we ship an
Apptainer image. An alternative substrate is a **direct SDSC Expanse ACCESS allocation** (SSH, Lmod
modules, SDSC's Spack, persistent installs, any CUDA) — full control at the cost of an allocation
request and losing the push-button gateway flow. Decision rule: stay on the gateway while runtime
pip + optional Apptainer cover the matrix (they currently do); move to a direct allocation only if we
need a toolchain the container route can't reach, or interactive/iterative builds. See
`docs/dependencies.md` Strategy C/D for the Apptainer + Spack bridge that keeps us on the gateway
longer.

## Success metrics

- **Coverage:** fraction of matrix cells completed & TDD-green (target 100%, gaps logged, never silently dropped).
- **Cost:** GPU-hours consumed (free, but tracked) and wall-clock — the win is *parallel* wall-clock ≪ serial.
- **Reproducibility:** every reported number regenerable by re-running its job zip; contracts encode the schema.
- **Portability:** the same zip runs on NSG *and* locally (Legion/Spark) unchanged — NSG is a backend, not a fork.

## Design principles carried from the kit

1. **Offline-first.** No runtime internet — vendor wheels + checkpoints, or Apptainer, or stage data to Expanse. (`docs/dependencies.md`, `docs/nemar-data.md`)
2. **One cell = one job = one contract.** Every unit of the sweep is an independently submittable zip whose output a `pytest` contract validates — the M0 probe is the template.
3. **Agent-drivable end to end.** `nsgr/nsgr.sh` (+ a coming `sweep.py`) means an agent submits, polls, fetches, and scores with no human in the loop after credentials are set.
4. **No silent caps.** If the sweep bounds coverage (subset of datasets, seeds), it is logged, not hidden.

## Progress log (live probes on Expanse, 2026-07-22)

- **M0 platform** ✅ Py 3.11.4, V100-SXM2-32GB, driver 580.82.07, torch 2.0.1+cu117, egress=true.
- **NEMAR** ✅ 547 (now 552) OpenNeuro `ds` at `/expanse/projects/nemar/openneuro/`; real EEG loads via MNE.
- **M1 deps** ✅ quick win with a caveat: system venv is READ-ONLY → `pip install --target … --ignore-installed`
  into node-local scratch (not the returned workdir); mne/torch/numpy pre-installed.
- **JAX** ✅ `jax 0.10.2` GPU-native (`cuda:0`, backend=gpu) via the same pattern → **neurojax can run on NSG**.
- **M2 cell** ✅ (pipeline) EEGNet on real NEMAR EEG (ds002718, 650 trials) → bal-acc 0.975 vs chance 0.769 on the V100. Contrast auto-picked, not yet curated — see caveat in the table.

## Immediate next steps

1. **M2 — one real scored cell:** attach a model to the NEMAR EEG already loading in `nemar_load`
   (e.g. a braindecode classifier on a `ds` task) → produce accuracy/AUROC, fetch, schema-validate.
2. **Port neurojax:** a job that `--target`-installs neurojax and runs its CLI on a NEMAR recording.
3. **M4 — sweep harness (`nsgr/sweep.py`):** fan the matrix out as many NSG-R jobs, poll/fetch/aggregate.
