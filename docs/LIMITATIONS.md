# Limitations & constraints of running on NSG

The rules and boundaries you must design around when using NSG/Expanse. Each entry is tagged
**[policy]** (NSG/SDSC rule), **[measured]** (observed on real jobs this project ran), or
**[verify]** (a real limit whose exact value NSG doesn't publish — confirm with `nsghelp@sdsc.edu`
before you rely on a specific number).

---

## 1. Fair-use & allocation — this is a shared, free, finite resource

- **[policy]** NSG provides a **"fair share… not unlimited"** amount of compute from an annual NSF
  (ACCESS) allocation split across all users. There is no published per-user core-hour cap, but heavy
  use is throttled by fair-share scheduling. **Abide by:** don't treat it as private infinite compute.
  Large sweeps (hundreds–thousands of jobs) must be genuinely justified and **batched**, not blasted.
- **[measured]** **NSG-R rate-limits** submissions per token/tool (HTTP 429 → `error_kind:rate_limited`).
  **Abide by:** submit in controlled batches with back-off; don't fan out a whole matrix at once.
- **[verify]** Per-user **concurrent/queued job limit** and total **allocation (SU)** — not published.
  Assume a modest concurrency cap; confirm before scheduling a large sweep.

## 2. Hard job-shape limits

- **[policy/measured]** **48 h max wallclock** per job. Jobs exceeding it are killed. **Abide by:**
  checkpoint into the working dir and design for resume; prefer sub-30-min jobs (they schedule sooner).
- **[measured]** Nodes are **64-core AMD EPYC, ≤243 GB RAM**. GPU nodes carry **4× NVIDIA
  V100-SXM2-32 GB**. Params cap at `number_nodes ≤ 72`, `number_gpus ≤ 4`/node,
  `number_gbmemorypernode ≤ 243`. **Abide by:** one V100 (32 GB) is the default unit; size models/batches
  to 32 GB.
- **[policy]** Input must be **one `.zip` with exactly one top-level directory**. **[verify]** the
  **upload size ceiling** is not documented — keep zips small; don't ship large data/checkpoints if avoidable.

## 3. Software environment is fixed and locked down

- **[measured]** The `PYTORCH_PY_EXPANSE` tool is **Python 3.11.4 + torch 2.0.1+cu117**, driver 580.82.07.
  **Torch is pinned** — anything needing torch ≥2.2 / CUDA 12 (e.g. a recent `transformer_engine`)
  cannot use the stock env and needs a custom **Apptainer** image.
- **[measured]** The tool's system venv (`/usr/local/python/venv`) is **READ-ONLY**. You cannot
  `pip install` into it; `--user` is disallowed (it's a venv). **Abide by:** install into node-local
  scratch — `pip install --target "$TMPDIR/libs"` + `PYTHONPATH` (see `dependencies.md`, `entry.sh`).
- **[measured]** **numpy compatibility is a tripwire.** The image torch 2.0.1 is built against numpy 1.x;
  a dep that pulls **numpy 2.x** breaks torch (`Could not infer dtype…`). **Pin `numpy<2` for torch
  jobs.** JAX is the opposite (wants numpy ≥1.25) — keep torch and JAX jobs separate.
- **[measured]** Fresh-interpreter rule: import heavy libs in a new `python` that inherits `PYTHONPATH`,
  or an already-imported image module (e.g. numpy) shadows your target install.

## 4. No persistence between jobs — every job starts from zero

- **[measured]** Each job runs in a **fresh container**; there is **no state carried between jobs** and
  no way to amortize an install. **Every job re-pays** its dependency-install cost (~40 s–3 min) *and*
  the queue+startup latency. **Abide by:** for repeated/sweep jobs, either accept the per-job install
  tax, **vendor wheels** into the zip, or bake an **Apptainer** image; don't assume a warm cache.
- **[measured]** NSG returns your **entire working directory** as `output.tar.gz`. Installing deps into
  `./` balloons the download (a `jax[cuda12]` install in `./` = **3.3 GB**). **Abide by:** install to
  `$TMPDIR` (node-local, not returned); write only results/checkpoints into the working dir.
- **[verify]** Whether any path (`$HOME`) persists across jobs for caching — unconfirmed; don't rely on it.

## 5. Latency — this is batch, not interactive

- **[measured]** Turnaround is **~8–10 min minimum** even for a trivial job (queue + stage + run + return),
  and it's queue-dominated and variable. **Abide by:** NSG is **not** for interactive work or tight
  debug loops. Debug logic locally / on your own GPU first; use NSG for the actual heavy/batch run.
  Every code fix costs a full round-trip.

## 6. Data constraints

- **[measured]** NEMAR/OpenNeuro data at `$NEMARPATH` is **read-only** (547 datasets, git-annex). Your
  own data must ride in the upload zip (subject to the size ceiling) or be an on-NEMAR `ds`.
- **[policy]** **Human-subject data must be de-identified** — only de-identified data may be transferred
  to or run on NSG.
- **[measured]** Not all benchmark data is on NEMAR (e.g. **BCI IV 2a is not** — Graz/BNCI-hosted).
  Such data is fetched at runtime via MOABB/HF (runtime internet works) or must be staged.
- **[verify]** **Runtime internet egress works today** but is **not documented/guaranteed** — it could be
  restricted. For reproducibility, prefer vendored wheels/data over relying on it.

## 7. Compliance & attribution

- **[policy]** Agree to the NSG **usage policy**; academic use.
- **[policy]** **Cite NSG (and Expanse/ACCESS)** in any resulting publication — they track enabled
  publications and their allocation depends on it.

## 8. What NSG is therefore *not* good for

- Interactive development / tight iterative debugging (latency).
- Single jobs needing **>48 h** (must checkpoint & resume).
- Workloads that must **return large data** (whole-workdir output).
- Toolchains needing **newer torch/CUDA** without building an Apptainer image.
- **Very large job counts** without fair-use justification and batched submission.

## 9. Must-verify list (before relying on a specific number)

Ask `nsghelp@sdsc.edu` / check the authenticated Task page:
- [ ] Input `.zip` upload size ceiling
- [ ] Per-user concurrent / queued job limit
- [ ] Total allocation (SU) and whether GPU-hours are metered
- [ ] Output size ceiling and result **retention period**
- [ ] Whether any path persists across jobs (for caching)
- [ ] Whether a **custom Apptainer image** can be supplied (vs the fixed containerized tools)
- [ ] Whether runtime internet egress is a supported guarantee or incidental

---

**Bottom line for this project:** NSG is excellent as a **free, batched GPU backend** for
independent, ≤48 h, ≤32 GB-GPU, self-contained jobs against colocated NEMAR data — which is exactly
the shape of an OpenEEGBench-style sweep. It is **not** an interactive cluster, not a place for a warm
cache, and not a home for a bleeding-edge CUDA toolchain without a container. Design every job to be
self-contained, offline-capable, small-output, and resumable.
