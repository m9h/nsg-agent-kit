# nsg-agent-kit

**Run computational-neuroscience jobs on the Neuroscience Gateway (NSG → SDSC Expanse), designed for AI agents.**

NSG ([nsgportal.org](https://www.nsgportal.org)) gives free access to NSF supercomputers
(SDSC **Expanse**) for neuroscience compute. It has two entry points: a **web portal**
(click-to-submit) and **NSG-R**, a REST API you can drive with `curl`. This repo is the
missing manual + tooling for getting an arbitrary Python/PyTorch pipeline (e.g. a JAX EMEG
toolkit, or a LoRA fine-tune sweep) running there against **NEMAR** datasets — the SDSC
mirror of OpenNeuro EEG/MEG/iEEG data that lives *directly on Expanse's filesystem*.

It is written so that **another agent can read `docs/FINDINGS.md`, run one probe, and know
whether porting a real workload is a quick win or real rework** — without re-discovering
the platform from scratch.

## TL;DR for a porting decision

| Question | Answer | Confidence |
|---|---|---|
| What runs the job? | SDSC Expanse, Slurm, **48 h max**, 64-core / 128–243 GB nodes | doc-confirmed |
| PyTorch available? | **`PyTorch Python on Expanse (2.0.1+cu117)`** (GPU tool) | live catalog |
| General Python? | **`Python on Expanse (3.11.4)`** (CPU) | live catalog |
| Can the job `pip install` at runtime? | **Assume NO** — Expanse compute nodes have no general outbound internet; not documented ⇒ must be probed. Vendor wheels or ship Apptainer. | inference + must-verify |
| Where's the data? | NEMAR/OpenNeuro `ds######` already on disk at **`$NEMARPATH/<ds-id>`** — no download inside the job | doc-confirmed |
| Does NEMAR overlap OpenEEGBench data? | **No.** OpenEEGBench uses HuggingFace `braindecode/*`; NEMAR is OpenNeuro `ds######`. Two different corpora. | analysis |
| GPU type/count per PyTorch task? | Expected 1× **V100 32 GB** (`gpu-shared`); **verify on the tool's Task page** | must-verify |

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
#    - REST:    ./nsgr/nsgr.sh submit PYTORCH_EXPANSE probes/reve_frozen_probe/reve_frozen_probe.zip

# 4. Fetch outputs into probes/reve_frozen_probe/results/ when the job finishes:
#    ./nsgr/nsgr.sh fetch <jobhandle> probes/reve_frozen_probe/results/

# 5. GREEN: the same test now validates the returned metrics.json.
pytest probes/tests -q                              # GREEN
```

## Layout

```
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
