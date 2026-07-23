# NSG / Expanse platform findings

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
