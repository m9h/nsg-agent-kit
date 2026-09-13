---
name: nemar-job
description: Write, submit, poll and fetch an NSG/Expanse job that analyses a NEMAR dataset from $NEMARPATH. Encodes the measured platform gotchas (read-only venv, pip --target, numpy/transformers pins, output-tarball bloat, tool choice). Use when turning a chosen dataset into an actual run.
---

# Skill: run a NEMAR analysis on NSG

Prereq: pick the dataset with `nemar-search`, verify it with `nemar-inspect`. Credentials and the
submit/poll/fetch wrapper are in `nsgr/` (see `docs/NSG_ACCESS.md`).

## Job shape

One `.zip` whose **top level is exactly one directory** holding the entry script:

```
myjob/
  run.py          # entry; reads os.environ["NEMARPATH"]
  cell.json       # optional config the script reads
```

Submit (parameters documented in `docs/tool-params.md`):

```bash
source nsgr/config.env; source ~/.nsg_secret.env
./nsgr/nsgr.sh submit PY_EXPANSE myjob.zip     # CPU; PYTORCH_PY_EXPANSE for GPU
./nsgr/nsgr.sh list
./nsgr/nsgr.sh fetch <JOBHANDLE> results/
```
Key vparams: `filename` (entry script), `subdirname` (the top dir), `runtime` (h, ≤48),
`number_cores` (≤128 on the node we measured), `number_gpus` (≤4 V100), `number_gbmemorypernode` (≤243).

## Reading NEMAR data inside the job

```python
import os
ds = os.path.join(os.environ["NEMARPATH"], "ds002718")   # /expanse/projects/nemar/openneuro/...
```
BIDS paths are **git-annex symlinks** with materialized content — follow symlinks
(`glob(..., recursive=True)` and mne/mne-bids handle this; raw `os.walk` needs `followlinks=True`).

## Gotchas that WILL bite (all measured, not guessed)

| problem | fix |
|---|---|
| System venv `/usr/local/python/venv` is **read-only** | `pip install --target "$TMPDIR/libs"` + `PYTHONPATH`; `--user` is disallowed in a venv |
| Deps skip already-present packages and then break | add `--ignore-installed` so the target is self-contained |
| Old numpy shadows your install | import in a **fresh interpreter** with `PYTHONPATH` set — importing numpy first pins the image's copy |
| Image torch is **2.0.1**; newer libs silently drop it | pin `numpy<2` and `transformers==4.44.2`, or install `torch==2.4.1` (cu121: has `torch.nn.attention`, keeps V100/sm_70 — **not** cu130, which dropped Volta) |
| NSG returns the **whole working dir** | install deps to `$TMPDIR`, not `./` — a `jax[cuda12]` install in `./` made a 3.3 GB download |
| `hnn_core` on `PY_EXPANSE` is **0.3**, not 0.6.x | do not use it where version matters; check `__version__` and report it in your results |
| Queue latency is hours and variable | one cheap probe job first; never iterate interactively |

## Always emit provenance

Write a `metrics.json` into the working dir recording: dataset id, n recordings analysed vs
attempted, library versions (`hnn_core`, `mne`, `torch`, `numpy`), node (`nvidia-smi`, `os.cpu_count()`),
and every parameter. Results whose provenance is unknown are not reusable — and version drift
between backends is real (see `hnn-jax`: hnn_core 0.3 vs 0.6.1 gave waveform correlation 0.62 on
identical inputs).

## Templates

- `probes/nemar_probe/` — inventory what is on `$NEMARPATH`
- `probes/nemar_load/` — read a real recording with MNE (follows the annex symlinks)
- `templates/nemar-eeg/` — minimal dataset-reading job
- `templates/pytorch-gpu/entry.sh` — the `--target` + `$TMPDIR` + `PYTHONPATH` pattern, correct
