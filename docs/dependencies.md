# Getting your dependencies onto a node with no internet

Working assumption (see FINDINGS §3a): **the compute node cannot reach PyPI/HuggingFace during the
job.** Design for offline. Three strategies, cheapest first.

## Strategy A — use only what the tool env ships

The `PyTorch Python on Expanse (2.0.1+cu117)` env already has torch + CUDA and the usual scientific
stack. If your code needs nothing else, you're done. The frozen probe deliberately lives here
(torch + stdlib only) so it can run before you've solved deps.

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

## Decision

- Deps resolve against **torch 2.0.1 / cu117** → **Strategy B**, quick win.
- Any dep forces **newer torch/CUDA** → **Strategy C**, real rework.

Run this locally to decide B-vs-C in ~1 minute, before touching NSG:

```bash
pip download --only-binary=:all: --dest /tmp/vendortest \
  --python-version 3.10 --platform manylinux2014_x86_64 --implementation cp \
  peft transformers braindecode 2>&1 | tail -20
# if it resolves without demanding torch>=2.2 → Strategy B is viable.
```
