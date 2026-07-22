# NSG-R tool interface — verified from the REST API (no auth needed)

The `/tool` and `/tool/<ID>/doc/pise` endpoints are **public**. This is the authoritative,
machine-readable interface — captured 2026-07-22.

## Tool IDs (the string for `-F tool=...`)

| Portal name | REST `toolId` | Notes |
|---|---|---|
| PyTorch Python on Expanse | **`PYTORCH_PY_EXPANSE`** | our GPU target |
| Python on Expanse (3.11.4) | `PY_EXPANSE` | CPU, "in Singularity" |
| Python on Expanse GPU (3.8.10, TF) | `GPU_PY_EXPANSE` | |
| TensorFlow Python on Expanse | `TENSORFLOW_PY_EXPANSE` | |
| TensorFlow Python on NSG OSG | `TENSORFLOW_PY_NSGOSG` | |
| EEGLAB on Expanse | `EEGLAB_EXPANSE` | |
| **NEMAR on Expanse** | `NEMAR_EXPANSE` | dedicated NEMAR tool |
| MRTrix on Expanse | `MRTRIX_EXPANSE` | |
| SpikeInterface | `SPIKEINTERFACE_EXPANSE` | |
| AMICA | `AMICA_EXPANSE` | |
| FreeSurfer | (via portal) | |
| NEURON / CoreNEURON | `NEURON_EXPANSE` / `CORENEURON_EXPANSE` | |
| BluePyOpt | `BLUEPYOPT_EXPANSE`, `BLUEPYOPT_EXPANSE1143` | |
| HNN / HNN GUI | `HNN_EXPANSE` / `HNN_GUI_EXPANSE` | |
| Open Science Brain | `OSBv2_EXPANSE_0_7_3` | |
| PGENESIS | `SINGULARITY_PGENESIS24_EXPANSE` | |
| HiAER-Spike FPGA Python | `PY_CRI` | |

Nearly every tool name is "… **in Singularity** on EXPANSE" → NSG runs your job inside a
container. This is why a **custom Apptainer image** is the clean escape hatch for newer deps.

## `PYTORCH_PY_EXPANSE` parameters (from the Pise XML)

Submit as `-F vparam.<name>_=<value>` (note the trailing underscore), input zip as
`-F input.infile_=@job.zip`.

| Param | Default | Meaning / limit |
|---|---|---|
| `filename` | `input.py` | **name of the entry Python file** to run |
| `subdirname` | — | top-level directory in the zip that contains the entry file |
| `runtime` | `0.5` | max wallclock **hours** (≤48). Short jobs (<0.5h) schedule sooner |
| `number_nodes` | `1` | nodes (max 72) |
| `number_gpus` | `1` | **GPUs per node (max 4 = V100)** |
| `number_gbmemorypernode` | `1` | GB RAM per node (max **243**) |
| `cmdlineopts` | — | args passed to your script (escape `\"` for doubles; no single quotes) |
| `nrnivmodl_o` | `1` | compile NEURON mod files — irrelevant unless you use NEURON |

`command_name`, `infile`, `outputfile` are framework-managed (input zip / stdout capture).

## Canonical submit for a zip whose top dir is `myjob/` with entry `myjob/run.py`

```bash
curl -u "$NSG_USER:$NSG_PASSWORD" -H "cipres-appkey:$NSG_APPKEY" \
  "$NSG_URL/job/$NSG_USER" \
  -F tool=PYTORCH_PY_EXPANSE \
  -F input.infile_=@myjob.zip \
  -F vparam.filename_=run.py \
  -F vparam.subdirname_=myjob \
  -F vparam.runtime_=0.5 \
  -F vparam.number_gpus_=1 \
  -F vparam.number_gbmemorypernode_=16 \
  -F metadata.statusEmail=true
```

For the frozen probe: zip top dir is `reve_frozen_probe/`, entry `run.py` → set
`vparam.subdirname_=reve_frozen_probe` and `vparam.filename_=run.py`.
