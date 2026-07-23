# nsg-agent-kit

**Get free GPU compute for computational neuroscience on the Neuroscience Gateway (NSG → SDSC
Expanse) — for people *and* AI agents.**

NSG gives you **free** access to NSF supercomputers (SDSC **Expanse**, V100 GPUs) — no HPC
allocation, no SSH. You upload a zip of your code, pick a software environment, and NSG runs it and
hands back the output. Through **NEMAR**, 547 OpenNeuro EEG/MEG/iEEG datasets already sit on Expanse's
disk, free to read from your job.

---

# Get access (≈10 minutes)

## 1. Create an NSG account
Go to **<https://nsgprod.sdsc.edu:8443/portal2/>** → use the **new-user registration form** on the
login page → register with an **academic email** → accept the usage policy. Approval is usually quick.
(De-identified human-subject data only.)

That's all you need to submit through the **web portal**. For scripted/agent access via the REST API,
do steps 2–4.

## 2. Create an application key (for REST/agent access)
Log in, then: **Developer** menu → **Application Management** → **Create New Application**.

| Field | Enter |
|---|---|
| **Name** | short, letters/numbers/underscores — e.g. `my_nsg_app` |
| **Long Name** | any label — e.g. `My NSG App` |
| **Authentication Type** | **DIRECT** (leave as-is; it's for scripts/CLI/agents) |
| Web Site / Contacts | optional — leave blank |

Click **Create**. Copy the **Application ID** it shows (e.g. `my_nsg_app-B142E82F…`) — that's your
app key.

## 3. Store your credentials (never commit secrets)
```bash
git clone https://github.com/m9h/nsg-agent-kit && cd nsg-agent-kit
cp nsgr/config.example.env nsgr/config.env        # gitignored — then edit:
#   NSG_USER=<your username>   NSG_APPKEY=<the Application ID>   NSG_TOOL=PYTORCH_PY_EXPANSE
# Stash your password in a 0600 file (never in the repo):
umask 077; read -rsp "NSG password: " P; printf 'export NSG_PASSWORD=%q\n' "$P" > ~/.nsg_secret.env; unset P
```

## 4. Verify access
```bash
source nsgr/config.env; source ~/.nsg_secret.env
curl -sS -u "$NSG_USER:$NSG_PASSWORD" -H "cipres-appkey:$NSG_APPKEY" "$NSG_URL/job/$NSG_USER"
# HTTP 200 + <joblist> = you're in.
```

**Full walkthrough with screenshots-worth of detail:** [`docs/NSG_ACCESS.md`](docs/NSG_ACCESS.md).

---

# Run your first job

```bash
# Build the cheapest end-to-end probe (torch-only frozen encoder on synthetic EEG, ~2 min on a V100):
cd probes/reve_frozen_probe && ./make_zip.sh && cd ../..
pytest probes/tests -q                                                   # RED (no result yet)

# Submit, wait, fetch:
./nsgr/nsgr.sh submit PYTORCH_PY_EXPANSE probes/reve_frozen_probe/reve_frozen_probe.zip
./nsgr/nsgr.sh list                                                      # get the job handle
./nsgr/nsgr.sh fetch <jobhandle> probes/reve_frozen_probe/results/       # when it finishes
pytest probes/tests -q                                                   # GREEN
```
Prefer clicking? Upload the same zip via the portal — see [`docs/submit-portal.md`](docs/submit-portal.md).

New to all this? The gentle version is [`docs/GETTING_STARTED.md`](docs/GETTING_STARTED.md).

---

# What you get (and must live with)

- **Hardware:** 1–4× `Tesla V100-SXM2-32GB` per job, Python 3.11.4, torch 2.0.1+cu117, 48 h max.
- **Data:** NEMAR/OpenNeuro `ds######` at `$NEMARPATH` (`/expanse/projects/nemar/openneuro/`) — 547
  datasets, read-only, no download.
- **Key constraints:** the tool's Python venv is **read-only** (install deps with
  `pip install --target "$TMPDIR/libs"`), **torch is pinned at 2.0.1** (pin `numpy<2`, and
  `transformers<5`; newer torch needs a custom Apptainer image), and each job is **stateless** with
  ~8–10 min batch latency — it's not interactive. Read **[`docs/LIMITATIONS.md`](docs/LIMITATIONS.md)**
  before porting real work.

---

# Docs

| File | What |
|---|---|
| [`docs/NSG_ACCESS.md`](docs/NSG_ACCESS.md) | **Get access** — account + app key, step by step |
| [`docs/GETTING_STARTED.md`](docs/GETTING_STARTED.md) | Newcomer on-ramp: what NSG is, first job, key gotchas |
| [`docs/LIMITATIONS.md`](docs/LIMITATIONS.md) | The constraints you must abide by |
| [`docs/dependencies.md`](docs/dependencies.md) | Installing deps (read-only venv, `--target`, numpy/transformers pins, Apptainer/Spack) |
| [`docs/nemar-data.md`](docs/nemar-data.md) | Reading NEMAR/OpenNeuro data from `$NEMARPATH` |
| [`docs/tool-params.md`](docs/tool-params.md) | Every tool id + submission parameter (from the public REST API) |
| [`docs/submit-rest.md`](docs/submit-rest.md) · [`docs/submit-portal.md`](docs/submit-portal.md) | Submitting by REST / by GUI |
| [`docs/FINDINGS.md`](docs/FINDINGS.md) · [`docs/ROADMAP.md`](docs/ROADMAP.md) | Platform intel · the OpenEEGBench-at-scale target |

`nsgr/nsgr.sh` — curl wrapper (submit/status/fetch, env-var creds). `templates/` — zip templates for
PyTorch-GPU / Python-CPU / NEMAR-EEG jobs. `probes/` — working end-to-end examples with TDD contracts.

Everything here is **verified live on Expanse** (2026-07). MIT licensed.
