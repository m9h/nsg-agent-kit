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
- **[measured]** **There is no budget/quota API, and no usage figure anywhere in NSG-R.** Probed
  2026-09-22: `/usage`, `/account`, `/allocation`, `/quota` all **404**; `/user/<you>` returns only
  profile fields plus `role`, `active` and **`canSubmit`** — no SUs, no core-hours, no charge. Job
  records carry timestamps and stages but **no resource-usage fields**. So:
  - **`canSubmit: true` on `/user/<you>` is the only machine-readable "am I still allowed to run"
    signal** — worth checking if submissions start failing.
  - To ask about an actual allocation, use the portal (**My Profile** shows an *XSEDE Allocation*
    field) or email **nsghelp@sdsc.edu**. NSG holds a pooled community allocation and does not
    meter per-user SUs back to you.
  - Practically: NSG is fair-share, not a metered budget. There is nothing to "check" before a run —
    but that is exactly why the fair-use obligation below is on you rather than enforced by a quota.

## 2. Hard job-shape limits

- **[policy/measured]** **48 h max wallclock** per job. Jobs exceeding it are killed. **Abide by:**
  checkpoint into the working dir and design for resume; prefer sub-30-min jobs (they schedule sooner).
- **[measured]** Nodes are **64-core AMD EPYC, ≤243 GB RAM**. GPU nodes carry **4× NVIDIA
  V100-SXM2-32 GB**. Params cap at `number_nodes ≤ 72`, `number_gpus ≤ 4`/node,
  `number_gbmemorypernode ≤ 243`. **Abide by:** one V100 (32 GB) is the default unit; size models/batches
  to 32 GB.
- **[policy]** Input must be **one `.zip` with exactly one top-level directory**. **[verify]** the
  **upload size ceiling** is not documented — keep zips small; don't ship large data/checkpoints if avoidable.

- **[measured 2026-10-23]** **32 GB of GPU memory is the ceiling per model.** The V100-SXM2-32GB is the only
  GPU; `number_gpus` buys parallel devices, not a bigger one. An fp32 fine-tune that needs more (e.g.
  OpenEEGBench's EEGPT at batch 64) cannot run at its published protocol here — see `OPENEEGBENCH.md` §4a.

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
- **[measured]** **Pin companions to torch.** Anything that depends on `torchaudio` (braindecode ≥1.6 does)
  must pin `torchaudio==<torch version>`; an unmatched build imports torch fine and then dies with
  `undefined symbol: aoti_torch_abi_version`. Likewise `transformers` 5.x is broken on torch 2.4.1
  (`NameError: nn`) — pin `transformers==4.46.3`; `peft` itself must be recent (`peft==0.21.2`: upstream's adapter configs need ≥0.17, and 0.21.2 imports fine next to transformers 4.46.3).
  Same genre as the peft→cu130 trap (FINDINGS §6a): the Volta-compatible torch is old enough that
  **every** torch-adjacent package must be pinned to its era.
- **[measured 2026-10-22, exp-9-57]** **The tool's Python 3.11.4 has no `_sqlite3` extension.** `import sqlite3`
  raises `ModuleNotFoundError: No module named '_sqlite3'`, which takes down anything that uses the
  stdlib module at import time — exca (OpenEEGBench's runner) does, so `import open_eeg_bench` fails
  after a clean install. **Abide by:** add `pysqlite3-binary` to the install and alias it before the
  first import (`sys.modules["sqlite3"] = pysqlite3; sys.modules["sqlite3.dbapi2"] = pysqlite3.dbapi2`;
  see `oeb_shard.py::shim_sqlite3`). Expect the same for other stdlib C extensions (`_bz2`, `_lzma`,
  `_tkinter`…) — the image's Python was built with few optional modules; test `python -c "import X"`
  in a probe before relying on one.
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

## 5a. NSG-R itself has outages — tell them apart from your own bugs

Observed three times (2026-07-26; 2026-10-22 10:55–~17:00 PDT; again 2026-10-23 from ~22:00 PDT): every
`POST /job/<user>` with an `input.infile_` upload returns

```
<displayMessage>Internal Cipres Error</displayMessage>
<message>Internal Error: java.lang.RuntimeException: error creating tmp file for input parameter input.infile_</message>
```

while `GET` (auth, `/user`, job list, `/tool`) keeps working and the portal shows no notice. It is the
gateway's upload-staging disk, not your zip: the same July probe zip that ran to COMPLETED on
2026-09-22 fails identically. Both times it cleared without action on our side. What to do:

- Re-submit a zip you *know* ran before (the control). Same error ⇒ outage, stop debugging your job.
- Retry on a timer rather than hammering: `sweep/oeb_plan.py submit` exits **75** (`EX_TEMPFAIL`) on
  exactly this message so a loop can wait for it (`until python sweep/oeb_plan.py submit --name X; do sleep 300; done`).
- If it lasts more than a day, mail nsghelp@sdsc.edu with the message, your username, and the time.
- Nothing was submitted, so nothing was charged against the shared allocation.

## 6. Data constraints

- **[measured]** NEMAR/OpenNeuro data at `$NEMARPATH` is **read-only** (552 datasets as of 2026-09-22, git-annex). Your
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
