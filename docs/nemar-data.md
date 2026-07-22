# NEMAR data on Expanse (and why OpenEEGBench data isn't there)

## Confirmed live (M0/NEMAR probe, 2026-07-22)

- `NEMARPATH` = **`/expanse/projects/nemar/openneuro/`**, present in the `PYTORCH_PY_EXPANSE` tool.
- **547 OpenNeuro `ds######` datasets** are mounted there — reachable with zero download.
- The mirror is a **DataLad / git-annex** checkout: actual file content lives in
  `.git/annex/objects/…` and is **materialized on Expanse** (real sizes, e.g. 11–316 MB `.fif`).
  BIDS paths (`sub-01/eeg/sub-01_..._eeg.fif`) are **symlinks** into the annex store.
  **Implication for loaders:** follow symlinks (mne-bids / `os.walk(followlinks=True)`), and don't
  assume a plain file — resolve the link. Content is already local, so reads are fast and offline.
- Example summarized: `ds001784` — 14 subjects, EEG, "Effects of ON/OFF deep brain stimulation on
  cognitive control in treatment-resistant depression (EEG)".

## How NEMAR data reaches your job

- NEMAR is the SDSC mirror of **OpenNeuro** EEG/MEG/iEEG. Uploads to OpenNeuro are cloned to SDSC via
  **DataLad** and re-synced **daily**, then indexed by their BIDS id `ds######`.
- On Expanse those datasets are mounted read-only and reachable through the **`NEMARPATH`** environment
  variable. Your job builds a path and reads it directly — **no download, no internet, no S3 auth**:

  ```python
  import os
  root = os.environ["NEMARPATH"]            # set by NSG for NEMAR-enabled tools
  ds   = os.path.join(root, "ds002718")     # a BIDS dataset dir
  # ... walk BIDS: sub-*/eeg/*_eeg.set etc.
  ```

  MATLAB equivalent: `filePath = [getenv('NEMARPATH') bidsName];`

- This is the big win of NSG-for-NEMAR: **petabytes of curated EEG/MEG sit next to the GPU**, so a
  sweep over many subjects/datasets pays no data-transfer cost.

> **Verify:** `NEMARPATH` is populated for the EEGLAB tool for certain. Whether it is exported inside
> the **PyTorch** tool env is a **[VERIFY]** — the probe prints `os.environ.get("NEMARPATH")`. If it is
> empty there, either (a) request NEMARPATH be enabled for the PyTorch tool, or (b) stage the specific
> `ds######` yourself, or (c) run EEG extraction under EEGLAB and hand arrays to the PyTorch job.

## The overlap question: NEMAR vs OpenEEGBench / braindecode

**They do not overlap.**

| | Addressing | Host | On `$NEMARPATH`? |
|---|---|---|---|
| NEMAR | OpenNeuro `ds######` (BIDS) | SDSC Expanse Lustre | **yes** |
| OpenEEGBench | HuggingFace `braindecode/*` datasets | huggingface.co | **no** |

OpenEEGBench distributes preprocessed splits through HuggingFace under the `braindecode` org. Those
are **not** OpenNeuro `ds` ids and are **not** on the Expanse filesystem. Two ways forward:

1. **Bring the benchmark data with you.** `huggingface-cli download braindecode/<name>` locally, drop
   the files under `data/` in your upload zip, and read them from the job's own working dir. Bounded by
   the input-zip size limit — good for probe-scale splits, painful for the full benchmark.
2. **Re-target to a NEMAR `ds######`.** If the sweep only needs "some real EEG at scale," pick an
   OpenNeuro dataset already on NEMARPATH (browse at <https://nemar.org>) and adapt the loader. This is
   the way to exploit NEMAR's free colocated storage.

For the **frozen-probe** decision job, neither is needed — the probe uses synthetic EEG so it isolates
env+GPU+I/O from any data question. Bring real data only once the platform facts are green.

## Finding a dataset id

- Browse/search: <https://nemar.org> (data explorer) — each dataset detail page shows its `ds######`.
- Anything visible on OpenNeuro will appear on NEMAR within ~a day of publication.
