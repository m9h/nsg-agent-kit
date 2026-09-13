---
name: nemar-search
description: Find NEMAR/OpenNeuro datasets (EEG/MEG/iEEG) by keyword, BIDS task label, modality or subject count — locally, in seconds, without spending an NSG job. Use whenever you need to pick a dataset for an NSG/Expanse analysis.
---

# Skill: search NEMAR for a dataset

## The key fact

**NEMAR is a daily mirror of OpenNeuro.** Every `ds######` OpenNeuro lists is already on Expanse at
`$NEMARPATH/<ds-id>` (`/expanse/projects/nemar/openneuro/`, ~547 datasets, git-annex, materialized).

So **discovery does not need an NSG job.** Query OpenNeuro's public GraphQL API locally and get the
answer in seconds. Doing this as an NSG job instead means waiting hours in the Expanse queue for
information that is free — a mistake made once in this project already.

## Use

```bash
python scripts/nemar_search.py --kw "motor imagery" --modality eeg --min-subjects 10
python scripts/nemar_search.py --task-kw "assr|steady|click|entrain" --modality eeg
python scripts/nemar_search.py --kw schizophrenia --kw psychosis --json hits.json
```

- `--kw` matches the dataset **Name** (repeatable, case-insensitive, OR'd)
- `--task-kw` is a regex over **BIDS task labels** — search this too, because paradigms are often
  invisible in the title. Searching names alone found 15 candidates in this project; adding task
  labels surfaced datasets the titles never mentioned.
- `--modality` = `eeg` | `meg` | `ieeg` | `mri`

## Then

1. **Always** run `nemar-inspect` on a candidate before writing a job — a title is not evidence that
   the data contains the paradigm you need.
2. Read it inside a job with:
   ```python
   ds = os.path.join(os.environ["NEMARPATH"], "ds002718")
   ```
   No download, no S3, no credentials — it is on local disk next to the compute.

## Gotchas

- The mirror is **git-annex**: BIDS paths like `sub-01/eeg/*_eeg.fif` are **symlinks** into
  `.git/annex/objects/`. Content is materialized, so reads are fast, but a loader must follow
  symlinks (mne-bids does; `os.walk` needs `followlinks=True`).
- Not everything is on NEMAR. HuggingFace-hosted benchmark data (e.g. `braindecode/*`) and
  lab-hosted sets (BCI Competition IV 2a is Graz/BNCI, not OpenNeuro) are **not** mirrored — those
  must be downloaded at runtime or vendored into the job zip.
