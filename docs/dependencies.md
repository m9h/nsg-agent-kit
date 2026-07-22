# Getting your dependencies onto a node with no internet

Working assumption (see FINDINGS §3a): **the compute node cannot reach PyPI/HuggingFace during the
job.** Design for offline. Three strategies, cheapest first.

> **Read-only system venv (measured).** The tool's Python lives at `/usr/local/python/venv` and is
> **read-only** — a plain `pip install <new-pkg>` fails with `OSError [Errno 30] Read-only file
> system`. `--user` is disallowed (it's a venv). **Always install into a writable target and add it to
> the path:** `pip install --target ./pylibs <pkgs>` + `export PYTHONPATH=$PWD/pylibs:$PYTHONPATH`
> (see `templates/pytorch-gpu/entry.sh`). Packages already baked into the image (torch, mne, numpy…)
> import with no install. Runtime egress to PyPI works, so `--target` installs succeed at job start.

> **Output-tarball bloat (measured).** NSG returns your **entire working dir** as `output.tar.gz`.
> A `jax[cuda12]` install into `./pylibs` made the returned tarball **3.3 GB** (bundled NVIDIA CUDA
> libs). **Install deps into node-local scratch outside the returned dir** (`$TMPDIR/nsgkit-pylibs`),
> or delete the target before the script exits. `entry.sh` does the former.

## JAX on NSG — confirmed working (jax 0.10.2, GPU)

The PyTorch tool is a fine base for JAX too (driver 580.82.07, V100). Recipe that works:
```bash
pip install --target "$TMPDIR/nsgkit-pylibs" --ignore-installed "jax[cuda12]"   # ~166 s
PYTHONPATH="$TMPDIR/nsgkit-pylibs" python your_jax_script.py                     # fresh interpreter
```
Result: `jax.devices() -> [cuda:0]`, `default_backend='gpu'`, numpy 2.4.6 in the target. This is the
path for **neurojax** and any JAX tool. Two non-obvious requirements:
1. `--ignore-installed` so the target carries a modern numpy (the image pins numpy 1.24.3, which
   modern JAX rejects with `numpy has no attribute 'dtypes'`).
2. import JAX in a **fresh** `python` that inherits `PYTHONPATH` — don't `import numpy/torch` in the
   same process first, or the old numpy gets pinned in `sys.modules` and shadows the target.

## Strategy A — use only what the tool env ships

The `PyTorch Python on Expanse (2.0.1+cu117)` env already has torch + CUDA, **mne**, and the usual
scientific stack (confirmed: `mne` imported with no install). If your code needs nothing else, you're
done. The frozen probe deliberately lives here (torch + stdlib only) so it can run before you've
solved deps.

## Strategy B — vendor wheels into the upload zip (recommended for the LoRA sweep)

Download every dependency (and transitive deps) as wheels **for the target platform** on a machine
with internet, ship them inside the zip, and install offline in the job's entry script.

```bash
# On a machine with internet — match the tool: cp39/cp310, manylinux, cu117.
# (Confirm the tool's exact Python tag from the probe's python_version first.)
pip download \
  --only-binary=:all: \
  --python-version 3.10 --platform manylinux2014_x86_64 --implementation cp \
  --dest vendor/ \
  peft==0.11.1 transformers==4.41.2 braindecode==0.8.1 safetensors accelerate
# torch is already in the tool env — do NOT vendor a second torch, it'll clash with cu117.
```

Entry script (runs on the node, offline):

```bash
python -m pip install --no-index --find-links vendor/ -r requirements.txt
python run.py
```

Caveats:
- **Match the interpreter + platform tags** or `--no-index` install fails. Get the exact tags from the
  probe (`python_version`, and `pip debug --verbose` in a scratch job).
- **Do not re-vendor torch** — the env's cu117 build must win. Pin your other deps to versions whose
  wheels don't drag in a conflicting torch.
- Watch the **input-zip size ceiling** ([VERIFY] with nsghelp@sdsc.edu). `transformers` + `peft`
  wheels are tens of MB; a model checkpoint (e.g. REVE weights) can be far larger — vendor checkpoints
  too, since HuggingFace is also unreachable at runtime.

## Strategy C — Apptainer/Singularity image (removes the torch-2.0.1 ceiling)

Expanse supports Singularity/Apptainer, and NSG already runs containerized tools (MRTrix NeuroDesk).
If a dep needs **torch ≥2.2 or CUDA 12** (a recent `transformer_engine` will), bake the whole env into
an image:

```dockerfile
# build locally, convert to .sif, stage per NSG container instructions
FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime
RUN pip install peft transformers braindecode accelerate safetensors
# + your transformer_engine patch
COPY run.py /opt/run.py
```

Then run under a container-capable NSG tool. **[VERIFY]** the exact mechanism for supplying a *custom*
image (vs. the fixed NeuroDesk ones) with nsghelp@sdsc.edu — this is the one path that isn't fully
self-serve from the public docs.

## Strategy D — Spack (considered; niche fit for NSG, strong for the image build)

[Spack](https://computing.llnl.gov/projects/spack-hpc-package-manager) is the LLNL HPC package
manager (from-source builds, exact reproducible specs). Honest assessment against **how NSG runs
jobs** — each tool is a *fixed Singularity container*, you upload a zip, it runs `python run.py`;
there is **no persistent, interactive build step** and no `spack install` you can amortize across
runs:

- **Poor fit for per-job use on the NSG gateway.** You can't stand up a Spack environment inside a
  0.5–48 h batch job without recompiling the world each run — wasteful, and there's no shell to
  `spack load` into. Runtime pip already works here (M1: mne in 11 s), so Spack buys us nothing for
  the base case.
- **Strong fit for building the Apptainer image (Strategy C).** `spack containerize` generates a
  reproducible Apptainer/Singularity recipe from a concretized spec — more reproducible than a
  hand-written Dockerfile, and the right tool if we need a compiled `transformer_engine` against
  torch≥2.2/CUDA-12. Build once locally, ship the `.sif`.
- **Native fit *if* we leave the gateway for a direct Expanse ACCESS allocation** (SSH + Lmod +
  SDSC's Spack). That's a different execution substrate — see ROADMAP's "gateway vs direct Expanse"
  note. It trades the gateway's zero-setup/free-batch model for full control.

**Verdict:** not needed now (runtime pip = quick win). Keep Spack in reserve for the reproducible
Apptainer image (Strategy C) and revisit it seriously only if we move to a direct Expanse allocation.

## Decision

- Deps resolve against **torch 2.0.1 / cu117** → **Strategy A/B**, quick win (runtime pip confirmed).
- Any dep forces **newer torch/CUDA** → **Strategy C** (Apptainer), optionally built with **Spack (D)**.
- Move to a **direct Expanse allocation** → Spack/Lmod become the native stack manager.

Run this locally to decide B-vs-C in ~1 minute, before touching NSG:

```bash
pip download --only-binary=:all: --dest /tmp/vendortest \
  --python-version 3.10 --platform manylinux2014_x86_64 --implementation cp \
  peft transformers braindecode 2>&1 | tail -20
# if it resolves without demanding torch>=2.2 → Strategy B is viable.
```
