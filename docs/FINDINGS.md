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

### 3a. Runtime internet on compute nodes → **assume NO** **[INFER / VERIFY]**
- Expanse compute nodes have **no general outbound internet** (standard SDSC posture; only login /
  data-mover nodes reach the outside). NSG runs jobs on compute nodes.
- NSG's public docs do **not** document a pip-at-runtime mechanism; the FAQ only says "commonly used
  python modules are made available" — i.e. the tool env is **fixed**.
- **Consequence:** do **not** rely on `pip install <x>` reaching PyPI mid-job. Options, in order of
  preference:
  1. Use only packages already in the tool env.
  2. **Vendor wheels** into the zip and `pip install --no-index --find-links vendor/`
     (works offline; see [`dependencies.md`](dependencies.md)).
  3. Ship an **Apptainer** image with the full env baked in.
- **Probe answer:** `probes/reve_frozen_probe/run.py` tries a guarded outbound request and records
  `network_egress: true|false` in `metrics.json`. That single field closes this crux empirically.

### 3b. GPU tool/queue specifics → **RESOLVED** **[LIVE — REST param spec]**
- Tool id = **`PYTORCH_PY_EXPANSE`**. GPU is chosen by picking this tool (distinct from the CPU
  `PY_EXPANSE`). The tool's Pise param spec (public, no auth) exposes:
  `number_gpus` (default 1, **max 4 V100/node**), `number_nodes` (max 72),
  `number_gbmemorypernode` (max **243 GB**), `runtime` (default 0.5h, max 48h).
- Entry point: param **`filename`** (default `input.py`) + **`subdirname`** (the top-level dir in
  your zip that holds the entry file). Full table in [`tool-params.md`](tool-params.md).
- All tools run **in Singularity** (per REST tool names) — containerized execution confirmed.
- **Still probe-only:** exact GPU *model string* and the bundled Python version — `run.py` records
  `torch.cuda.get_device_name(0)` / `device_count()` / `sys.version`.

### 3c. NEMAR data vs OpenEEGBench data → **no overlap** **[ANALYSIS]**
- **NEMAR** = SDSC mirror of **OpenNeuro** EEG/MEG/iEEG, addressed as `ds######`, synced daily via
  DataLad, mounted on Expanse at **`$NEMARPATH`**. A job reads `os.environ["NEMARPATH"] + "/ds00XXXX"`
  directly — **no download, no internet**. **[DOC]**
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

Still open (need the live probe or SDSC):
- [ ] `network_egress` on a compute node (pip-at-runtime feasibility) — **probe**
- [ ] Bundled Python version + exact GPU model string of the PyTorch tool — **probe**
- [ ] Whether `NEMARPATH` is exported inside `PYTORCH_PY_EXPANSE` (vs only `NEMAR_EXPANSE`/EEGLAB) — **probe**
- [ ] Whether `peft` / `transformers` / `braindecode` wheels resolve against torch 2.0.1+cu117 — **local `pip download` dry run**, see `dependencies.md`
- [ ] Upload-size ceiling for the input zip (limits the "vendor everything" strategy) — **portal / nsghelp@sdsc.edu**
- [ ] Whether a container-capable tool accepts an *arbitrary user* Apptainer image — **nsghelp@sdsc.edu**

## Sources

- NSG portal (live): <https://nsgprod.sdsc.edu:8443/portal2/tools.action>
- NSG-R user guide: <https://www.nsgportal.org/guide.html>
- NSG quick start: <http://www.nsgportal.org/qs.html> · FAQ: <https://www.nsgportal.org/faq.html>
- NEMAR ↔ NSG: <https://nemar.org/nsg_for_nemar> · NEMAR paper: <https://academic.oup.com/database/article/doi/10.1093/database/baac096/6823529>
- Expanse user guide: <https://www.sdsc.edu/systems/expanse/user_guide.html>
