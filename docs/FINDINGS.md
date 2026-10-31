# NSG / Expanse platform findings

> **Re-verified 2026-09-22 — no drift.** torch 2.0.1, Python 3.11.4, driver 580.82.07, V100-32GB,
> read-only venv and `numpy<2` all still hold on both tools. New: NEMAR is now **552** datasets
> (was 547); **GPU nodes have 40 cores, not 128**; `nrnivmodl` is absent from the GPU tool;
> `netpyne 1.0.4.2` is pre-installed on the CPU tool. Full snapshot:
> [`env-snapshots/2026-09-22.md`](env-snapshots/2026-09-22.md).

*Captured 2026-07-21 for the LoRA-REVE / OpenEEGBench porting decision. Confidence levels are
explicit so a downstream agent knows what to trust vs. what to probe.*

Legend: **[LIVE]** read from the portal this session · **[DOC]** from official NSG/SDSC docs ·
**[INFER]** reasoned from HPC norms, not yet confirmed · **[VERIFY]** open question, resolve
with a probe or an authenticated portal page.

---

## 1. What NSG is

- NSG = a **science gateway** that submits jobs to NSF HPC. Backend today is **SDSC Expanse**.
  Two front doors:
  - **Web portal** — `https://nsgprod.sdsc.edu:8443/portal2/` (login → Data → Tasks). **[LIVE]**
  - **NSG-R REST** — `https://nsgr.sdsc.edu:8443/cipresrest/v1` (curl, app key + basic auth). **[DOC]**
    NSG is built on the CIPRES/Cipres Workbench framework — hence `cipresrest` in the URL and the
    `cipres-appkey` header.
- Job model: upload **one `.zip`** whose **top level is a single directory** containing your code;
  pick a **tool** (fixed software env); set runtime params; NSG stages it to Expanse, runs Slurm,
  zips your working dir back. **[DOC]**
- **Limits:** 48 h wallclock max; nodes described as 64-core / 128 GB, up to 243 GB RAM; docs
  recommend 32–64 cores to avoid OOM. **[DOC]**
- **Cost:** free to academic users; NSG holds the ACCESS allocation. Human-subject data must be
  de-identified (portal login banner). **[LIVE]**

## 2. Live tool catalog (2026-07-21)

Full dated snapshot in [`tool-catalog.md`](tool-catalog.md). The ones that matter for a
PyTorch/LoRA workload:

| Tool (portal name) | Version | Kind | Notes |
|---|---|---|---|
| **PyTorch Python on Expanse** | **2.0.1+cu117** | **GPU** | the target tool for a LoRA/REVE sweep |
| Python on Expanse | 3.11.4 | CPU | general Python (BMTK, Brian2, DEAP, HNN-Core, NetPyNE, PyNN, NEST) |
| Python on Expanse GPU | 3.8.10 | GPU | TensorFlow-oriented |
| TensorFlow Python on Expanse | 2.7.9 | GPU | |
| EEGLAB on Expanse | Latest | — | MATLAB/EEGLAB, the NEMAR default path |
| FREESURFER on Expanse | 7.1.1 | — | |
| MRTrix (NeuroDesk) on Expanse | 3.0.3 | — | NeuroDesk container route exists on NSG |
| SpikeInterface on Expanse | 0.101.0 | — | |
| AMICA on Expanse | 17 | — | ICA, common in EEG |

**Reading of the catalog for our port:**
- Torch is **2.0.1 / CUDA 11.7** — mid-2023. `peft`, `transformers`, `braindecode` all *can* run on
  torch 2.0.1, but check each pin. Anything needing torch ≥2.2 or a CUDA-12 `transformer_engine`
  build will **not** work in the stock env → Apptainer route. **[INFER]**
- The PyTorch tool's **bundled Python version** is not shown in the catalog. **[VERIFY]** (expected
  3.9/3.10 — the Expanse `pytorch/2.0.1` singularity). The probe prints `sys.version`.
- MRTrix is shipped "(NeuroDesk)" — evidence NSG **can run NeuroDesk/Apptainer containers**, which is
  the escape hatch for custom deps. **[LIVE catalog + INFER]**

## 3. The three cruxes (what the orchestrator asked)

### 3a. Runtime internet on compute nodes → **EGRESS WORKS** **[LIVE — probe, 2026-07-22]**
- **Surprise, and good news:** the probe on GPU node `exp-2-58` reported **`network_egress: true`** —
  the compute node reached the public internet (pypi.org:443 / 8.8.8.8:53). This **contradicts** the
  usual "HPC compute nodes are air-gapped" assumption; NSG's PyTorch tool node has outbound access.
- **Consequence:** runtime `pip install <x>` from PyPI is **likely viable**, which makes the LoRA
  port materially easier (Strategy A/B both open, maybe without even vendoring). **Still confirm with a
  real install** (DNS + resolver + no proxy quirks) — that's the M1 probe: `pip install` then import
  `peft`/`transformers`/`braindecode` on the node.
- **[LIVE — with an important caveat] runtime pip works, but the tool's system venv is READ-ONLY.**
  - `pip install mne` returned success in 11 s (`nemar_load`) — but that was because **mne is
    pre-installed** in the image; pip had nothing to write.
  - Installing a genuinely new package to the default location **fails**: the `jax_probe` v1 job hit
    `OSError [Errno 30] Read-only file system: /usr/local/python/venv/.../site-packages/jax_plugins`.
  - **Fix (the pattern to use):** install into a writable dir in the job cwd and add it to the path —
    `pip install --target ./pylibs <pkgs>` then `sys.path.insert(0, "./pylibs")` (or `PYTHONPATH`).
    `--user` is **not** an option (the env is a venv, which disallows `--user`).
  - So the LoRA port is still a **quick win**, but the entry script must `--target`-install
    `peft transformers braindecode` (not plain `pip install`). Vendoring stays the reproducible option.
  - Driver is **`580.82.07`** (very recent, CUDA-12 capable) — so JAX's CUDA-12 wheels are viable
    *once installed to a writable target*; that's what `jax_probe` v2 tests.
- Vendoring wheels (`--no-index --find-links vendor/`) remains the robust fallback and the
  reproducibility-preferred path (pinned, no network flakiness) — see [`dependencies.md`](dependencies.md).

### 3b. GPU tool/queue specifics → **RESOLVED** **[LIVE — REST param spec]**
- Tool id = **`PYTORCH_PY_EXPANSE`**. GPU is chosen by picking this tool (distinct from the CPU
  `PY_EXPANSE`). The tool's Pise param spec (public, no auth) exposes:
  `number_gpus` (default 1, **max 4 V100/node**), `number_nodes` (max 72),
  `number_gbmemorypernode` (max **243 GB**), `runtime` (default 0.5h, max 48h).
- Entry point: param **`filename`** (default `input.py`) + **`subdirname`** (the top-level dir in
  your zip that holds the entry file). Full table in [`tool-params.md`](tool-params.md).
- All tools run **in Singularity** (per REST tool names) — containerized execution confirmed.
- **[LIVE — probe] Confirmed on `exp-2-58`:** GPU = **`Tesla V100-SXM2-32GB`**, `gpu_count=1`
  (as requested), **Python `3.11.4`**, `torch 2.0.1+cu117`, `cuda_available=true`. The bundled
  Python is modern (3.11) — no ancient-interpreter tax for wheel compatibility.

### 3c. NEMAR data vs OpenEEGBench data → **no overlap** **[ANALYSIS]**
- **NEMAR** = SDSC mirror of **OpenNeuro** EEG/MEG/iEEG, addressed as `ds######`, synced daily via
  DataLad, mounted on Expanse at **`$NEMARPATH`**. A job reads `os.environ["NEMARPATH"] + "/ds00XXXX"`
  directly — **no download, no internet**. **[DOC]**
- **[LIVE — probe] `NEMARPATH` IS exported inside the PyTorch GPU tool**, pointing at
  **`/expanse/projects/nemar/openneuro/`** and it exists on the node. So NEMAR/OpenNeuro data sits
  right next to the V100 in `PYTORCH_PY_EXPANSE` — the "petabytes next to the GPU" win is real here,
  not just in the EEGLAB/NEMAR tools.
- **OpenEEGBench** ships its datasets via **HuggingFace `braindecode/*`**, *not* OpenNeuro `ds` ids.
  Those are **not** on NEMARPATH.
- **Therefore for the oeb sweep you must either:** (a) pre-download the HF dataset(s) into the upload
  zip (fine for small probe splits, bounded by upload-size limits), or (b) re-target the sweep to an
  equivalent OpenNeuro `ds######` that *is* on NEMARPATH (best when you want NEMAR's scale for free).
  Do not assume the braindecode splits appear under `$NEMARPATH`. See [`nemar-data.md`](nemar-data.md).

## 4. Go / no-go for porting the LoRA-REVE / oeb sweep

**Quick win, *if*** the real deps (`peft`, `braindecode`, `transformers`, and your
`transformer_engine` patch) install cleanly against **torch 2.0.1 / cu117** *and* vendor as offline
wheels. Then: vendor wheels → zip → submit to `PyTorch Python on Expanse` → done.

**Real rework, *if*** any dep needs torch ≥2.2 / CUDA 12 (likely for a recent `transformer_engine`).
Then the path is **Apptainer**: build an image locally with the exact env, push/stage it, run via a
container-capable tool. Heavier, but removes the version ceiling entirely.

**Decision gate:** run the frozen probe (§3, torch-only, ~2 min GPU). It returns `network_egress`,
`gpu_name`, `python_version`, `torch_version` — the four facts that turn every **[VERIFY]** above
into a **[LIVE]** and pick the branch above with certainty.

## 5. Open must-verify list

Resolved from the public REST API (2026-07-22):
- [x] Tool id = `PYTORCH_PY_EXPANSE`; GPU count (max 4), nodes (max 72), mem (max 243 GB),
  runtime (max 48h), entry via `filename`+`subdirname` — see `tool-params.md`.
- [x] Execution is containerized (Singularity) — Apptainer route is first-class.

Resolved by the M0 probe on GPU node `exp-2-58` (2026-07-22, Expanse Slurm job 52376208):
- [x] **`network_egress = true`** — compute node reaches the internet; runtime pip likely viable.
- [x] **Python 3.11.4**, GPU **Tesla V100-SXM2-32GB** ×1, torch 2.0.1+cu117, CUDA available.
- [x] **`NEMARPATH` is live in `PYTORCH_PY_EXPANSE`** = `/expanse/projects/nemar/openneuro/` (exists).
- [x] GPU compute runs end to end (probe_accuracy 0.988, `probe_ok`).

Still open:
- [ ] Real `pip install peft transformers braindecode` on the node (egress works — does a full install + import succeed?) — **M1 probe**
- [ ] Which OpenNeuro `ds######` are actually present under `$NEMARPATH` — **NEMAR discovery job (next)**
- [ ] Upload-size ceiling for the input zip — **portal / nsghelp@sdsc.edu**
- [ ] Whether a container-capable tool accepts an *arbitrary user* Apptainer image — **nsghelp@sdsc.edu**

## Sources

- NSG portal (live): <https://nsgprod.sdsc.edu:8443/portal2/tools.action>
- NSG-R user guide: <https://www.nsgportal.org/guide.html>
- NSG quick start: <http://www.nsgportal.org/qs.html> · FAQ: <https://www.nsgportal.org/faq.html>
- NEMAR ↔ NSG: <https://nemar.org/nsg_for_nemar> · NEMAR paper: <https://academic.oup.com/database/article/doi/10.1093/database/baac096/6823529>
- Expanse user guide: <https://www.sdsc.edu/systems/expanse/user_guide.html>

## 6. M5 attempt: LoRA-REVE cell on NSG (2026-07-22/23)

Three real jobs submitted via `sweep/cell.py` (model=lora, BNCI2014_001 subject 1, MOABB source),
each fixing one real blocker found by the previous run — no blocker was guessed, all three were
diagnosed from the actual job's `metrics.json`/traceback:

1. **Job `...1C5E01DA...`**: `model_error` = gated-repo 401 on `brain-bzh/reve-base` — no HF token
   was passed into the job. Fixed by threading `hf_token` through `cell.json` (sourced from the
   local `~/.cache/huggingface/token`, never logged).
2. **Job `...3A1B5258...`**: HF auth now works (model/data load past that point); new
   `model_error` = `AttributeError("module 'torch.distributed' has no attribute 'tensor'")` inside
   `peft==0.18.1`'s `_get_in_out_features` (a DTensor/FSDP2 probe not populated in this torch
   build). Fixed by pinning `peft==0.10.0`, predating that probe (also sidesteps the unrelated
   `transformer_engine` import issue seen on the DGX 26.06 image — a different peft/torch pairing
   entirely, same genre of "peft version probes a torch internal that isn't there").
3. **Job `...210AFC61...`**: peft 0.10.0's LoRA adapter builds cleanly — `trainable_params:
   5,758,016 / total_params: 74,947,648` (7.68%, consistent with r=32 LoRA on REVE's attention+FFN,
   *not* a full-finetune-sized count). Training then fails at the first forward pass:
   **`torch.AcceleratorError: CUDA error: no kernel image is available for execution on the
   device`** — a torch-build/GPU-architecture mismatch, not a dependency-pin issue. This is
   evidence the tool's actual environment is **not what the roadmap/tool-catalog docs describe**:
   `diag_torch_version` reports **`2.13.0+cu130`** in all three jobs above, not the previously
   `[LIVE]`-verified `2.0.1+cu117` from the M0 probe (2026-07-21/22) — **the NSG `PYTORCH_PY_EXPANSE`
   tool's underlying image appears to have been upgraded since M0/M1 were verified**, and the new
   torch build's compiled CUDA kernels likely don't include the V100's compute capability (a common
   failure mode when a wheel is built targeting newer datacenter GPUs only).

**This is a genuine, currently-unresolved platform blocker, not a config mistake we made.** Likely
next step: confirm the GPU architecture actually assigned (`torch.cuda.get_device_capability()`,
not captured in this run — add it to the next probe) and either (a) find a torch/peft/transformers
combination whose kernels cover that architecture, or (b) escalate to nsghelp@sdsc.edu — the
tool's live env no longer matches its own documented spec, worth reporting upstream regardless of
our own workaround.

**Progress against M5's actual gate**: not yet green (no successful forward pass on real data), but
two of three real blockers on the path are now cleared and documented, and the LoRA adapter
construction itself is confirmed working (right parameter count, right target modules) — the
remaining gap is a torch/CUDA-kernel compatibility question, not our LoRA code.

Artifacts: `sweep/cell.py` (`run_lora`), `probes/tests/test_lora_cell_contract.py` (RED before this
work, still RED — the CUDA-kernel error means neither test passes yet),
`sweep/results/lora_bnci2014_001/metrics.json` (the real, failed-at-forward-pass result).

### 6a. CORRECTION (2026-07-23, measured): the node was NOT upgraded — peft pulled a new torch

The §6 conclusion that "the `PYTORCH_PY_EXPANSE` tool image appears to have been upgraded to
`2.13.0+cu130`" is **wrong**, and the fix is different (and easier) than escalating to SDSC. Measured
directly on **three fresh jobs (2026-07-23)** that record the node env via `capture_env()`
(`nvidia-smi` + `torch.__version__`):

- **eegnet** and **reve** cells report **`torch 2.0.1+cu117`**, `python 3.11.4`, driver `580.82.07`,
  `Tesla V100-SXM2-32GB` — **identical to M0**. The **image torch is unchanged.**
- The jax re-probe still gets `jax 0.10.2`, `gpu_visible=True`.

Why the LoRA jobs saw `2.13.0+cu130`: **`peft` hard-requires `torch`.** When `pip install --target`
installs peft and torch is not already in the target dir, pip pulls the **latest** torch from PyPI
(`2.13.0+cu130`) into `--target`, which **shadows** the image's 2.0.1. That cu130/CUDA-13 wheel's
kernels **dropped Volta (sm_70)** → `no kernel image is available for execution on the device`. The
reve/eegnet cells don't install peft (transformers lists torch as *optional*, moabb doesn't need it),
so they use the image torch and run fine. **The node never changed; the job installed the wrong torch.**

**Fix (applied):** pin a **V100-compatible torch into the target** via a `torch` key in `cell.json`
(handled in `main()`): **`torch==2.4.1`** — PyPI's default is a **cu121** build that *has*
`torch.nn.attention` (REVE's model code needs it; torch 2.0.1 lacks it, torch ≥2.1 has it) *and* still
ships sm_70 kernels for the V100. Put it first in the dep list so peft sees torch already satisfied and
doesn't pull cu130. This keeps REVE/LoRA on the gateway — **no Apptainer needed, no SDSC escalation.**
(Do not use cu130/nightly torch on the V100.)

So the real ceiling is narrower than "torch 2.0.1": REVE/LoRA need **torch ≥2.1**, satisfied by
installing **torch 2.4.1 (cu121)** into the target. Only a dep that needs newer **CUDA** than the
driver, or torch that has dropped Volta, would force Apptainer.

## 7. M5 GREEN: LoRA-REVE cell succeeds on NSG (2026-07-24)

Coordinator correction (superseding this doc's §6 "platform gap" conclusion): the CUDA-kernel
error was **our own unconstrained pip install shadowing the base image's torch with a
Volta-incompatible wheel** (`2.13.0+cu130`), not NSG silently upgrading the tool image. Fix:
pin **torch==2.4.1+cu121** explicitly (still ships sm_70/Volta kernels, has `torch.nn.attention`,
matches the node's 580.x driver) via `cell.json`'s `torch` override key, installed as part of the
same `pip install --target LIBS` call so later installs (transformers/peft/etc.) see it already
satisfied. **Verified directly, not assumed**: `gpu_capability: [7, 0]` (V100/Volta, correct),
`cuda_kernel_smoketest_ok: true` (an actual `x @ x` on `cuda:0` before proceeding).

One more real blocker after the torch fix: a pandas C-extension/numpy-ABI mismatch
(`ImportError: C extension: None not built`) — fixed by explicitly listing `pandas` alongside
`numpy<2` in the same install call, rather than letting moabb's transitive resolution pick a
mismatched one.

**Note on process**: this fix was developed in parallel by two independent efforts (this session's
DGX-Spark-coordinated work, and a separate concurrent session that pushed `72d32f3`/`d4669a9` with
its own `torch_override` mechanism, vendored REVE weights avoiding HF gating, and a `_pool_tokens`
fix for REVE's 4-D output) — both independently converged on **torch==2.4.1+cu121** as the fix,
which is itself a second, independent confirmation the diagnosis was right. Reconciled by rebasing
onto the concurrent work and adding only the one still-missing piece (the pandas pin) rather than
redoing already-solved work.

**Real, complete M5 result** (job `NGBW-JOB-PYTORCH_PY_EXPANSE-D14CA47B343C4733A5B4A3F38CC46D70`,
BNCI2014_001 subject 1, session-holdout 288 train/288 test):

| | value |
|---|---|
| trainable / total params | 5,758,016 / 74,947,648 (7.68%) |
| final train loss | 0.000686 |
| test_balanced_accuracy | **0.5104** |
| chance | 0.25 |
| cell_ok | true |

**Caveat before comparing to the local DGX reference**: this is a **single-subject, session-holdout**
split (one subject's train session vs. their own test session), not the DGX/`open-eeg-bench`
harness's full-cohort, subject-disjoint split used for the `bcic2a` OpenEEGBench submission numbers
(`docs/oeb_lora_reve_submission.md` in `emeg-fm`, DGX raw test_balanced_accuracy 0.5926 for the
r=16/attention-only config on the full multi-subject dataset). Different eval unit, not a strict
apples-to-apples comparison — but both show the same qualitative result: LoRA-REVE (same core
method: attention+FFN LoRA targets, AdamW) clearly beats chance by a wide margin on the same
underlying raw dataset (BCI Competition IV 2a), on two independent hardware/software stacks (DGX
26.06/CUDA13 vs NSG Expanse V100/CUDA12.1).

M5 gate (`docs/ROADMAP.md`): **substantively met** for one cell — a real LoRA-REVE result,
reproduced end-to-end on NSG's free GPU, cross-hardware-validating the core method (not a strict
numeric match to the DGX cell, which used a different split protocol).

## 8. OpenEEGBench smoke shard on NSG (2026-10-22) — the upstream-code path, measured

Shard `arithmetic_zyma2019_s0` (BIOT, frozen + linear head, seed 0; `ridge_probe` rejected by upstream's
validator for BIOT and recorded as skipped), tool `PYTORCH_PY_EXPANSE`, 1 V100, 96 GB, 1 h limit.
Driver = `sweep/oeb_shard.py`, submitted with `sweep/oeb_plan.py submit`, fetched with `poll`.

| attempt | job | node | result |
|---|---|---|---|
| 0 (10:55–11:05 PDT, every 5 min) | — | — | **NSG-R refused the upload**: `Internal Cipres Error: error creating tmp file for input parameter input.infile_`. The July control zip failed identically → service outage (LIMITATIONS §5a). Cleared by 17:21. |
| 1 (17:21) `…F6E9A7AD` | exp-9-57 | pip of the pinned stack **OK in 139 s** (torch 2.4.1, torchaudio 2.4.1, numpy<2, transformers 4.46.3, peft 0.14.0, open-eeg-bench 0.6.0 into `$TMPDIR`) — then `import open_eeg_bench` → exca → `import sqlite3` → **`ModuleNotFoundError: No module named '_sqlite3'`**. The image's Python 3.11.4 lacks the sqlite3 C extension (LIMITATIONS §3). Fixed in the driver with `pysqlite3-binary` aliased as `sqlite3` (`shim_sqlite3`), verified locally against a blocked `_sqlite3`. |
| 2 (17:55) `…DC769080` | exp-9-57 | **GREEN.** Queue 7 min (SUBMITTED 17:55 → running 18:02), pip 149 s, shard 53 s, done 18:06. `Tesla V100-SXM2-32GB`, torch `2.4.1+cu121`, `sqlite3 = pysqlite3 2.6.0`. 1/1 experiment completed (`ridge_probe` skipped by upstream's validator, as locally). **`test_balanced_accuracy = 0.576389` — bit-identical to the local RTX 2080 run.** Returned tarball 16.6 kB (exca cache + summary; HF cache and libs stayed in `$TMPDIR`). `oeb_plan.py merge` + `collect` → `leaderboard_upstream.csv` via upstream `collect_completed_results` in read-only mode; `test_oeb_shard_contract.py` **4 passed**. |

Measured facts this establishes: (a) the whole upstream OpenEEGBench stack runs on the V100 tool from a
pip install into `$TMPDIR` in ~2.5 min, no container needed; (b) HF dataset download + exca cache work
on the node; (c) results are deterministic across GPUs for this cell, so NSG numbers are comparable
with local ones; (d) 15 min wall-clock per smoke job including queue — the budget per shard is the
dataset download + 7 backbones × 9 strategies × ~1 min, not NSG overhead.

### 8a. bench1: one dataset × seed, all 7 backbones × 9 strategies (2026-10-23, job `…C464E692`, exp-10-60)

`oeb_plan.py plan --name bench1 --backbones <all 7> --strategies <all 9> --heads linear_head`,
59 valid cells (upstream's validator drops BIOT×ridge_probe and LaBraM×{ridge_probe, ia3, oft}).
Queue **1 h 52 min** (12:18 → 14:10 PDT), pip 131 s, **59 cells in 1,544 s** (25.7 min; sum of
upstream `fit_time` 1,349 s), returned tarball 82 kB. **24 completed, 35 failed** — every failure a
stack problem, none a benchmark result:

| n | exception | cause → fix |
|---|---|---|
| 21 | `LoraConfig`/`AdaLoraConfig.__init__() got an unexpected keyword argument 'target_parameters'` | upstream `finetuning.py` passes peft ≥0.17 kwargs; our `peft==0.14.0` pin (chosen in §OPENEEGBENCH to dodge transformers 5.x) is too old → **`peft==0.21.2`** imports cleanly next to `transformers==4.46.3` + torch 2.4.1 [measured locally]; validated in bench1b |
| 6 | `Only instances of PreTrainedModel support target_modules='all-linear'` (ia3) | same peft-0.14 limitation |
| 6 | OFT: `You can only specify either r (8) or oft_block_size (8)` | same — OFT config semantics changed in peft 0.15 |
| 2 | `CUDA out of memory` — EEGPT full_finetune and two_stages, 28.5 GB "allocated by PyTorch" of 31.7 | ran after 40 earlier cells in the same process; either EEGPT full FT genuinely needs >32 GB at upstream's batch size or memory accumulates across cells → driver now runs one cell at a time with `gc` + `empty_cache` and logs per-cell peak / still-held GiB |

The 24 completed cells (balanced accuracy, arithmetic_zyma2019, seed 0, linear head unless `flatten`
= ridge_probe):

| backbone | frozen | ridge | full | two_stages | fit s (frozen / full) |
|---|---|---|---|---|---|
| BIOT | 0.576 | — | **0.681** | 0.608 | 22 / 34 |
| CBraMod | 0.510 | 0.542 | 0.524 | **0.674** | 6 / 19 |
| BENDR | 0.500 | 0.476 | 0.500 | 0.500 | 10 / 24 |
| EEGPT | 0.580 | 0.597 | OOM | OOM | 162 / — |
| LaBraM | 0.559 | — | 0.545 | **0.649** | 187 / 245 |
| Signal-JEPA | 0.500 | 0.493 | 0.500 | 0.500 | 16 / 64 |
| REVE | 0.545 | **0.632** | 0.618 | 0.594 | 26 / 99 |

BENDR and Signal-JEPA sit at chance on this 2-class, 1,343-window dataset under every strategy —
a benchmark finding (the 0.6.0 paper's arithmetic column is similarly weak for them), so the
contract's "beats chance" assertion now applies only to the BIOT reference cell.

**Sizing update.** Cells cost 1–245 s here; LaBraM and EEGPT dominate. For this smallest dataset a
full (backbone × strategy) shard is ~30 min of V100 once the adapters work, i.e. **queue time, not
compute, is the budget** at this scale; ISRUC (13 GB, 10× the windows) is where the 48 h limit will bite.

### 8b. bench1b: the peft fix validated, the EEGPT OOM isolated (2026-10-23, job `…0FCDB03E`, exp-13-59)

BIOT + EEGPT × {frozen, lora, ia3, adalora, dora, oft, full_finetune, two_stages}, `peft==0.21.2`,
driver running one cell at a time with `gc` + `empty_cache` and per-cell peak/held GiB. Queue 19 min,
pip 136 s, 16 cells in 584 s.

| backbone | strategy | status | bal. acc | wall s | peak GiB | held after |
|---|---|---|---|---|---|---|
| BIOT | frozen / ia3 / lora / adalora / dora / oft / full / two_stages | **8/8 completed** | 0.576 / 0.580 / 0.615 / 0.632 / 0.625 / 0.608 / 0.681 / 0.608 | 25–75 | 0.6–1.9 | 0.06 |
| EEGPT | frozen | completed | 0.580 | 153 | 3.2 | 0.06 |
| EEGPT | ia3 / lora / adalora / dora / oft / full | **OOM** | — | **2–2.5** | **30.0–31.0** | 0.06 |
| EEGPT | two_stages | OOM (after its frozen stage) | — | 69 | 30.5 | 0.06 |

- **`peft==0.21.2` closes all 33 adapter failures of bench1** — every PEFT strategy runs on the V100
  next to `transformers==4.46.3` + torch 2.4.1.
- **EEGPT with gradients through the backbone needs > 32 GB at upstream's config.** Each such cell
  OOMs in ~2 s (first backward), while `held after` stays at 0.06 GiB across all 16 cells — so the
  bench1 OOMs were EEGPT itself, not accumulation. Frozen EEGPT peaks at 3.2 GiB. This is a
  **V100-32GB limit**, not an NSG limit: the 5 trainable EEGPT strategies need a bigger GPU or a
  smaller batch than upstream's default (see §8c for what upstream's config allows).

### 8c. bench1c: EEGPT at batch 16 fits, but the numbers are not the protocol's (2026-10-23, job `…66FCA037`, exp-1-58)

EEGPT × {lora, full_finetune}, linear head, arithmetic seed 0, `--training-override eegpt:batch_size=16`
(everything else at upstream's `Training` defaults: 50 epochs, lr 5e-4, fp32). Queue 55 min, pip 139 s,
2 cells in 1,692 s.

| strategy | status | bal. acc | wall s | fit s | peak GiB | held after | trainable params |
|---|---|---|---|---|---|---|---|
| EEGPT lora | completed | 0.521 | 704 | 659 | **10.1** | 0.06 | 1.18 M (4.4 %) |
| EEGPT full_finetune | completed | 0.510 | 987 | 984 | **8.8** | 0.06 | 25.4 M (100 %) |

- **Batch 16 fits with room**: peak 8.8–10.1 GiB against > 30 GiB at batch 64 (§8b), i.e. the
  first-backward footprint scales ~linearly with batch and batch 32 would likely fit too (~20 GiB,
  not measured). So every EEGPT strategy *can* be run on a V100-32GB — at a deviated batch.
- **The deviated cells are slow and land at chance**: 0.52 / 0.51 against 0.58 for frozen EEGPT
  (§8a/§8b) and 0.68 for BIOT full fine-tune. Four times the optimizer steps per epoch at the same
  lr/warmup is a different training regime, not the protocol with a smaller batch, so these rows
  must not be compared with protocol rows. ~12–16 min per cell against 2.5 min for frozen EEGPT.
- **Consequence for the sweep**: option 1 of `OPENEEGBENCH.md` §4a (report the 7 trainable EEGPT
  cells as *not runnable at protocol on V100-32GB*) is the default; the batch-16 path exists for a
  study of the batch-size effect, not as a substitute. The driver now stamps every row with
  `batch_size` and `protocol_deviation` (e.g. `batch_size=16`) so a pooled leaderboard can filter
  them; bench1c's own rows pre-date that field (its deviation is only in the exca uid and in
  `shard.json`).
- NSG-R cost of this shard: 55 min queue for 28 min of GPU — the queue, not the GPU, sets the
  wall-clock of a sweep (consistent with 7 min / 1 h 52 / 19 min on the earlier shards).

## 9. public1, first batch (2026-10-24/25): three ways a shard can come back empty

The first public sweep (`OPENEEGBENCH.md` §4: 26 jobs, seed 0) went in at 23:04 PDT on 2026-10-24,
the minute a 25 h NSG-R staging outage cleared. Of the first 10 fetched shards **none produced a
usable cell**, for three independent reasons:

| shards | what NSG said | what actually happened | evidence |
|---|---|---|---|
| arithmetic, bcic2a, bcic2020_3 | COMPLETED, failed=false | ran 4 min, then the NSG workspace disk was full: STDOUT truncated, no `output.tar.gz` | `scheduler_stderr.txt`: `No space left on device` on `done.txt` and `tar` |
| physionet, mdd_mumtaz2016 | COMPLETED, failed=false | started in the same full-disk window (STDOUT = 174 B header only); physionet's 4,584-file HF pull throttled; **killed at the 16 h limit** | `slurmstepd: CANCELLED … DUE TO TIME LIMIT`, STDERR stuck at `Fetching 4584 files: 18%` |
| chbmit × {bendr, cbramod, labram, signal_jepa, reve} | COMPLETED, tarball present | ran 17:13–17:17 on 2026-10-25, 0/9 cells: HF rate-limited the node's shared IP, snapshot incomplete, every cell `No group found in store` | `exception` in `shard_summary.json` rows: `We had to rate limit your IP (198.202.103.196)` |

What was right: pip 143–150 s per job (same as before), the driver's per-cell error capture
(0.1 s per failed cell after the first), tarballs 27–35 kB. What was wrong is all upstream of our
code — NSG's disk, HF's anonymous quota — and both are now in `LIMITATIONS.md` (§5a, §6).

Consequences: (1) `submit --hf-token-file` ships a Hugging Face token in a submit-time copy of the
zip (`run.py` reads and deletes it; `summary["hf_token"]` records that it was used); (2) `submit
--only SHARD…` resubmits just the lost shards; (3) arithmetic's first cell reproduced 0.5764 again
before the disk filled — the fourth identical number on V100. The 16 h / 48 h limits are untouched:
with a token, physionet's download should take minutes, not hours (HF throughput on the node is still
the open measurement).

The first batch also answered the per-user concurrency question: NSG passed 3 jobs to Slurm at a
time and held the others in `SUBMITTING` for 2–20 h, so `--max-inflight 6` was already generous.

### 9a. public1 after the restart with a token (2026-10-26/27)

| shard | cells | GPU time | notes |
|---|---|---|---|
| arithmetic, 7 backbones | 47/59 | 45 min | BIOT identical to bench1; best REVE lora 0.785, CBraMod lora 0.701; BENDR + Signal-JEPA at chance |
| bcic2020_3, 7 backbones | 53/59 | 5.2 h | 5 classes (chance 0.20); best REVE frozen 0.283 |
| physionet, 7 backbones | 47/59 | 6.6 h | 5 cells lost to HF 429 even with the token |
| faced, BIOT | 8/8 | 88 min | 9 classes (chance 0.11); 0.155–0.191 |

Three findings, two of them new problems:

- **HF 429 with a token**: upstream re-runs `snapshot_download()` per cell → `prefetch()` once per
  shard + `HF_HUB_OFFLINE=1` for the cells (`LIMITATIONS.md` §6). Verified locally: arithmetic
  prefetch 61 s for 510 files, then an offline dataset + weight load in 4 s.
- **A second NSG disk-full** (2026-10-26, two shards, no outage before it): recurring, not a one-off.
- **New stack failures, the same on every dataset so far** (the contract's
  `test_no_cell_failed_for_a_software_reason` will be RED on them): LaBraM × {lora, adalora, dora}
  `Targeting an nn.Parameter on the top-level module is not supported (parameter 'temporal_embedding')`
  (peft 0.21.2 vs LaBraM's `training_required_parameters`), and CBraMod / REVE × one adapter each
  `shape '[N, …]' is invalid for input of size 0`. Not yet diagnosed.
- Also fixed: a fetch interrupted after the download left files that `curl -OJ` refuses to
  overwrite, so the poll failed on physionet for 4 h; `poll` now fetches into an empty directory.

### 9b. public1 with prefetch + offline cells (2026-10-27 → 31): 20 of 26 shards in, both stack failures diagnosed

The loop ran to its 120 h deadline with all 26 shards submitted: 20 fetched with none lost, 6 still in
Slurm (chbmit/faced/isruc × LaBraM, isruc × Signal-JEPA/REVE/EEGPT), plus physionet resubmitted on
the prefetch zip (below). `merge` over the 20: **427 rows, 359 completed.**

**The prefetch fix worked.** Every shard that ran with it has `prefetch_ok: true` (2–56 min per shard,
isruc_sleep the slowest) and **no cell in those shards failed on HF.** bcic2a, which hit the 16 h limit
twice on the old zip, finished in 2.2 h. The only HF failures left are 6 BIOT physionet cells from the
pre-prefetch run → physionet resubmitted with `submit --only physionet_s0 --resubmit` (the 47-cell
attempt is kept in `fetched_lost/`). No shard needed the new `deadline_stopped` guard.

Best balanced accuracy per backbone (best strategy for each dataset, seed 0, isruc and LaBraM partial):

| backbone | arithmetic | bcic2020-3 | bcic2a | chbmit | faced | isruc-sleep | mdd | physionet |
|---|---|---|---|---|---|---|---|---|
| chance | 0.50 | 0.20 | 0.25 | 0.50 | 0.11 | 0.20 | 0.50 | 0.25 |
| BENDR | 0.500 | 0.213 | 0.278 | 0.553 | 0.130 | 0.382 | 0.731 | 0.298 |
| BIOT | 0.681 | 0.271 | 0.304 | 0.818 | 0.191 | **0.774** | 0.888 | 0.340 |
| CBraMod | 0.701 | 0.269 | 0.481 | **0.961** | **0.333** | 0.764 | **0.890** | 0.591 |
| EEGPT (frozen/ridge only) | 0.597 | 0.247 | 0.333 | 0.728 | 0.169 | — | 0.874 | 0.476 |
| LaBraM | 0.649 | 0.208 | 0.367 | — | — | — | 0.863 | 0.535 |
| REVE | **0.785** | **0.283** | **0.563** | 0.859 | 0.320 | — | 0.879 | **0.630** |
| Signal-JEPA | 0.500 | 0.209 | 0.351 | 0.624 | 0.227 | — | 0.718 | 0.481 |

REVE and CBraMod lead; BENDR and Signal-JEPA are near chance on most tasks. BIOT frozen on chbmit
is exactly 0.500 (the majority class on an imbalanced seizure task) — a result, so the contract's
reference-cell check is now arithmetic-only, as its docstring always said.

**The two stack failures are upstream / peft bugs, not our pins** (reproduced locally on CPU with
random weights, peft 0.21.2 and 0.18.1; repro in the session scratchpad):

- **LaBraM × {lora, dora, adalora}** (15 cells). open-eeg-bench passes LaBraM's
  `temporal_embedding` — an `nn.Parameter` on the *top-level* module — to peft as LoRA-family
  `target_parameters`. peft 0.21.2 (latest) rejects top-level parameters; 0.17.1/0.18.1 accept them
  but raise `lora.ParamWrapper does not work with lora_dropout != 0` against upstream's
  `dropout=0.1`. **No released peft runs this configuration.** Upstream's own comment says the
  embedding must be *retrained from scratch*, which a LoRA delta on it would not do anyway.
- **CBraMod × oft, REVE × oft** (15 cells). peft's OFT rotation unfolds every 4-D input as if it fed a
  `Conv2d` (`if len(orig_shape) == 4: x = self._unfold(x)`). CBraMod's
  `patch_embedding.spectral_proj` and REVE's `to_patch_embedding` are `Linear` layers fed
  `(batch, chan, patch, feat)` → the unfold with `kernel=(0, 0)` is empty → `shape '[N, r, b]' is
  invalid for input of size 0`. Same code in 0.18.1.

Both are recorded in the contract as `KNOWN_UPSTREAM_BROKEN`, matched on the exception text, so a
different failure in those cells is still RED. We do not patch around them: that would change the
protocol being benchmarked. Drafts for an open-eeg-bench issue and a peft issue are with the user.

New OOM: **REVE × DoRA on FACED** (1 cell, the only non-EEGPT OOM in 427 rows) → `LIMITATIONS.md` §2
and `KNOWN_V100_OOM_CELLS`.

Two harness bugs found and fixed on the way (`5f41213`): `poll` aborted the whole pass when one
terminal job listed no outputs, so nothing was saved and nothing resubmitted for ~12 h; and a Slurm
TIME LIMIT kill returns no tarball at all, so the driver now stops starting cells once the longest
cell so far + 30 min no longer fits in `runtime_hours`.
