# Getting started with NSG

New to the Neuroscience Gateway? This gets you from zero to a real job running on an NSF
supercomputer — and its results back on your laptop — in about 30 minutes. No HPC experience,
no allocation paperwork, no SSH required.

---

## What NSG actually is (plain version)

The **Neuroscience Gateway (NSG)** gives neuroscientists **free** access to NSF supercomputers
(currently **SDSC Expanse**, incl. NVIDIA V100 GPUs). You don't manage a cluster or hold your own
allocation. Instead:

> **You upload a zip of your code, pick a software environment ("tool"), and NSG runs it on Expanse
> and hands you back the output.**

Two ways to drive it:
- **Web portal** — point-and-click. Best for your first job and one-offs.
- **NSG-R (REST API)** — scriptable with `curl`. Best for automation, sweeps, and agents.

**Bonus — free data next to the GPU:** through **NEMAR**, **547 OpenNeuro EEG/MEG/iEEG datasets**
are already sitting on Expanse's filesystem (at `$NEMARPATH`). Your job reads them directly — no
download, no S3, no cost.

## The mental model

```
   your code (zip)          a "tool"                params                 Expanse (Slurm)
 ┌────────────────┐   ┌──────────────────┐   ┌──────────────────┐   ┌──────────────────────┐
 │ myjob/         │   │ PyTorch GPU      │   │ entry = run.py   │   │  runs your job,      │
 │   run.py       │──▶│ (fixed env:      │──▶│ gpus = 1         │──▶│  ≤ 48 h,             │
 │   (+ data/…)   │   │  torch 2.0.1,    │   │ hours = 0.5      │   │  then zips your      │
 └────────────────┘   │  Python 3.11)    │   └──────────────────┘   │  whole workdir back  │
    one top dir       └──────────────────┘                          └──────────┬───────────┘
                                                                                │
                                                              output.tar.gz  ◀──┘  (metrics, logs…)
```

Rules that matter:
- The zip's **top level is exactly one directory** containing your code.
- You tell the tool which file to run (`filename`) and which subdir it's in (`subdirname`).
- **GPU is chosen by picking a GPU tool** (e.g. `PyTorch Python on Expanse`) — there's no separate
  GPU checkbox.
- Jobs run **≤ 48 h**. NSG returns your **entire working directory** as `output.tar.gz`.

## Step 1 — Get an account

Go to the [NSG portal](https://nsgprod.sdsc.edu:8443/portal2/), click the new-user form, and register
with an academic email. Approval is usually quick. (Read the usage policy; de-identify human-subject
data.)

## Step 2 — Run your first job

The repo ships a tiny, self-contained example — a frozen-encoder + linear-probe on synthetic EEG
(torch only, ~2 min). It's the "hello world" of NSG and prints the platform facts.

```bash
git clone https://github.com/m9h/nsg-agent-kit && cd nsg-agent-kit
cd probes/reve_frozen_probe && ./make_zip.sh      # -> reve_frozen_probe.zip
```

### Path A — Web portal (easiest first time)
1. **Data → Upload Data:** upload `reve_frozen_probe.zip`.
2. **Tasks → Create New Task:** select that data; select tool **PyTorch Python on Expanse**.
3. **Set Parameters:** entry file `run.py`, sub-directory `reve_frozen_probe`, max time `0.5` h,
   1 GPU. Save & Run.
4. When it finishes, **download the output**, open `reve_frozen_probe/metrics.json`.

### Path B — NSG-R (REST, scriptable)
One-time: **Developer → Application Management → Create New Application** (type **DIRECT**) to get an
**Application ID**. Then:
```bash
cp nsgr/config.example.env nsgr/config.env      # fill NSG_USER + NSG_APPKEY (NSG_TOOL is preset)
# stash your password so it never lands on disk in the repo:
umask 077; read -rsp "NSG password: " P; printf 'export NSG_PASSWORD=%q\n' "$P" > ~/.nsg_secret.env; unset P

source nsgr/config.env; source ~/.nsg_secret.env
./nsgr/nsgr.sh submit PYTORCH_PY_EXPANSE probes/reve_frozen_probe/reve_frozen_probe.zip
./nsgr/nsgr.sh list                              # find your job handle
./nsgr/nsgr.sh fetch <JOBHANDLE> probes/reve_frozen_probe/results/
pytest probes/tests -q                           # GREEN once results are back
```

Either way, you'll get something like:
```json
{ "python_version": "3.11.4", "gpu_name": "Tesla V100-SXM2-32GB",
  "network_egress": true, "nemarpath": "/expanse/projects/nemar/openneuro/",
  "probe_accuracy": 0.988, "probe_ok": true }
```
That's a real job, on a real V100, done.

## Step 3 — Use NEMAR data

547 OpenNeuro datasets live at `$NEMARPATH` on the node. Reading one is just:
```python
import os
root = os.environ["NEMARPATH"]              # /expanse/projects/nemar/openneuro/
ds   = os.path.join(root, "ds001784")       # a BIDS dataset — pick ids at https://nemar.org
```
See `probes/nemar_probe` (lists what's available) and `probes/nemar_load` (reads real EEG with MNE).

## Five things that will save you hours

These are measured behaviors of the NSG environment — know them before you port real code:

1. **Pick the GPU by picking the GPU tool.** `PYTORCH_PY_EXPANSE` = 1–4× V100. There's no checkbox.
2. **The tool's Python venv is READ-ONLY.** To add packages, install into a writable dir:
   `pip install --target "$TMPDIR/libs" --ignore-installed <pkgs>` and set `PYTHONPATH`.
   (`--user` doesn't work — it's a venv.)
3. **`--ignore-installed` matters.** Otherwise the image's older numpy shadows your install and
   breaks modern JAX/other libs.
4. **NSG returns your whole working dir.** Install deps to node-local `$TMPDIR`, *not* `./`, or your
   download balloons (a `jax[cuda12]` install in `./` = 3.3 GB).
5. **48-hour cap; shorter jobs schedule sooner.** Checkpoint into your working dir so partial results
   come back.

## Where to go next

| I want to… | Read |
|---|---|
| Understand the platform in depth | [`FINDINGS.md`](FINDINGS.md) |
| See every tool id + submit parameter | [`tool-params.md`](tool-params.md) |
| Add real dependencies (pip / wheels / Apptainer / Spack) | [`dependencies.md`](dependencies.md) |
| Work with NEMAR / OpenNeuro data | [`nemar-data.md`](nemar-data.md) |
| Submit by GUI / by REST | [`submit-portal.md`](submit-portal.md) · [`submit-rest.md`](submit-rest.md) |
| See the big-picture goal (OpenEEGBench at scale) | [`ROADMAP.md`](ROADMAP.md) |

## Mini-glossary

- **NSG** — the gateway (web portal + REST) that submits your jobs to a supercomputer.
- **NSG-R** — NSG's REST API (`curl`-drivable). Built on the CIPRES framework (hence `cipresrest`).
- **Expanse** — the SDSC supercomputer that actually runs the jobs (Slurm scheduler).
- **NEMAR** — SDSC's mirror of OpenNeuro EEG/MEG/iEEG data, on Expanse's disk at `$NEMARPATH`.
- **Tool** — a fixed software environment (e.g. `PYTORCH_PY_EXPANSE`) you run your code inside.
- **Task / job** — one submission: your zip + a tool + parameters.
- **Application ID (app key)** — your REST credential, created in the portal's Developer menu.
