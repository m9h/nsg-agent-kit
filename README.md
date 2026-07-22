# nsg-agent-kit

**A practical guide + tooling for running computational-neuroscience jobs on the Neuroscience
Gateway (NSG → SDSC Expanse) — for people *and* AI agents.**

NSG ([nsgportal.org](https://www.nsgportal.org)) gives neuroscientists **free** access to NSF
supercomputers (SDSC **Expanse**, incl. V100 GPUs) — no HPC allocation, no SSH. You upload a zip of
your code, pick a software environment, and NSG runs it on Expanse and hands back the output. Two
front doors: a **web portal** (click-to-submit) and **NSG-R**, a REST API you drive with `curl`.
Through **NEMAR**, **547 OpenNeuro EEG/MEG/iEEG datasets** already sit on Expanse's disk, free to read
from your job.

### 👉 New to NSG? Start with **[docs/GETTING_STARTED.md](docs/GETTING_STARTED.md)**
Zero to a real job on a real V100 — and its results back on your laptop — in ~30 minutes.

This repo is the missing manual + tooling for getting an arbitrary Python/PyTorch/JAX pipeline (a JAX
EMEG toolkit, a LoRA fine-tune sweep, an EEG benchmark) running on NSG against NEMAR data. It's
written so a **newcomer can follow the guide and run their first job**, and so an **agent can read
`docs/FINDINGS.md`, run one probe, and know whether porting a real workload is a quick win or real
rework** — without re-discovering the platform from scratch. Everything here is **verified live on
Expanse** (7 real jobs, 2026-07-22) — the facts and gotchas are measured, not guessed.

> **Target ([`docs/ROADMAP.md`](docs/ROADMAP.md)):** make NSG the free-GPU backend for the entire
> **OpenEEGBench** matrix — every EEG foundation model (BENDR, EEGPT, LaBraM, BIOT, REVE, …) ×
> dataset × probing protocol — run as a fanned-out batch of NSG jobs, with the **LoRA-REVE** sweep
> as the first research payload. Milestones M0–M5, each with a red→green gate.

## TL;DR for a porting decision

| Question | Answer | Confidence |
|---|---|---|
| What runs the job? | SDSC Expanse, Slurm, **48 h max**, 64-core / 128–243 GB nodes | doc-confirmed |
| PyTorch available? | **`PyTorch Python on Expanse (2.0.1+cu117)`** (GPU tool) | live catalog |
| General Python? | **`Python on Expanse (3.11.4)`** (CPU) | live catalog |
| Can the job `pip install` at runtime? | **Yes, likely** — M0 probe found `network_egress=true` on the GPU node (unusual for HPC). Confirm a real install in M1. Vendoring stays the reproducible fallback. | **live probe** |
| Where's the data? | NEMAR/OpenNeuro `ds######` on disk at **`$NEMARPATH/<ds-id>`** = `/expanse/projects/nemar/openneuro/`, **confirmed live in the PyTorch tool** | **live probe** |
| Does NEMAR overlap OpenEEGBench data? | **No.** OpenEEGBench uses HuggingFace `braindecode/*`; NEMAR is OpenNeuro `ds######`. Two different corpora. | analysis |
| GPU type/count per PyTorch task? | **1× `Tesla V100-SXM2-32GB`** (confirmed on node `exp-2-58`), Python **3.11.4** | **live probe** |

See **[`docs/FINDINGS.md`](docs/FINDINGS.md)** for the full writeup and every "must-verify".

## The three things that will bite a naive port

1. **No runtime internet.** `pip install peft braindecode transformers` inside the job will
   likely fail. You must either (a) pre-download wheels into the upload zip and
   `pip install --no-index --find-links vendor/`, or (b) ship an Apptainer/Singularity image
   (Expanse supports it). See [`docs/dependencies.md`](docs/dependencies.md).
2. **Your data may not be on NEMAR.** NEMARPATH holds OpenNeuro `ds######`. HuggingFace-hosted
   benchmark data (OpenEEGBench / `braindecode/*`) is *not* there — pre-package it or map to an
   equivalent `ds######`. See [`docs/nemar-data.md`](docs/nemar-data.md).
3. **Torch is pinned at 2.0.1 / CUDA 11.7.** Anything needing newer torch (or `transformer_engine`
   built against a newer CUDA) needs the Apptainer route, not the stock tool env.

## Quickstart (red → green)

```bash
# 1. Build the cheapest end-to-end probe: a frozen-encoder + linear-probe on synthetic EEG,
#    torch-only (no extra deps) so it tests env+GPU+I/O in isolation.
cd probes/reve_frozen_probe && ./make_zip.sh        # -> reve_frozen_probe.zip

# 2. RED: the contract test fails because no results have come back yet.
cd ../.. && pytest probes/tests -q                  # RED

# 3. Submit the zip to NSG (portal GUI or NSG-R). Two paths, both documented:
#    - Portal:  docs/submit-portal.md   (tool: "PyTorch Python on Expanse")
#    - REST:    ./nsgr/nsgr.sh submit PYTORCH_PY_EXPANSE probes/reve_frozen_probe/reve_frozen_probe.zip
#      (entry/GPU/runtime params come from NSG_EXTRA in config.env — see docs/tool-params.md)

# 4. Fetch outputs into probes/reve_frozen_probe/results/ when the job finishes:
#    ./nsgr/nsgr.sh fetch <jobhandle> probes/reve_frozen_probe/results/

# 5. GREEN: the same test now validates the returned metrics.json.
pytest probes/tests -q                              # GREEN
```

## Layout

```
docs/GETTING_STARTED.md # ⭐ newcomer on-ramp: what NSG is, first job, key gotchas
docs/ROADMAP.md         # the target: OpenEEGBench at scale on NSG (milestones M0–M5)
docs/FINDINGS.md        # platform intel + confidence levels + must-verify list
docs/tool-catalog.md    # dated snapshot of the live NSG tool list (exact versions)
docs/nemar-data.md      # $NEMARPATH, ds access, NEMAR-vs-braindecode overlap
docs/dependencies.md    # no-internet strategy: vendored wheels vs Apptainer
docs/submit-portal.md   # click-by-click portal submission
docs/submit-rest.md     # NSG-R REST submission
nsgr/nsgr.sh            # curl wrapper: submit / status / fetch (env-var creds, never prints secrets)
templates/pytorch-gpu/  # zip template for the PyTorch 2.0.1 GPU tool (+ offline dep vendoring)
templates/python-cpu/   # zip template for the Python 3.11.4 CPU tool
templates/nemar-eeg/    # example reading a dataset from $NEMARPATH
probes/reve_frozen_probe/  # the cheapest real end-to-end probe + TDD contract test
```

## Status

Tool catalog captured **live** from the portal on 2026-07-21. Live GPU-config details and the
runtime-internet question are marked **must-verify** because the automation session could not
reach authenticated portal pages (password entry is out of scope for the agent). Running the
probe in `probes/reve_frozen_probe/` answers them empirically. PRs welcome from any agent that
closes a must-verify.

MIT licensed.
