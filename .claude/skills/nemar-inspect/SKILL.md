---
name: nemar-inspect
description: Inspect a specific NEMAR/OpenNeuro dataset (subjects, sessions, BIDS tasks, modalities, size, README, license) before committing an NSG job to it. Use after nemar-search and BEFORE writing any analysis code.
---

# Skill: inspect a NEMAR dataset before using it

## Why this step is not optional

A dataset's **title tells you almost nothing** about whether it holds the data you need. The most
expensive lesson in this project: an 8-mouse "graded anesthesia" dataset with both spontaneous and
evoked activity described in the paper turned out, in the public release, to be **spontaneous only** —
every file `stim-SPN`, the metadata defining that as *"Spontanious stimulation"*, and the recordings
carrying a `Stim` event channel with **zero events**. Half the intended analysis was impossible, and
that was discoverable in minutes rather than after writing the pipeline.

## Use

```bash
python scripts/nemar_inspect.py ds002718
python scripts/nemar_inspect.py ds002718 --files      # snapshot file tree
```

Reports name, license, authors, modalities, subject/session counts, file count and size, **BIDS task
labels**, the README, and the exact `$NEMARPATH` expression to use in a job.

## Checklist before writing the job

- **Task labels** — do they match the paradigm (evoked / resting / spontaneous)? Task names are the
  single most informative field.
- **`events.tsv`** — does it carry the `trial_type` values you intend to contrast? Confirm on the
  node; do not assume. If you auto-pick "the two most frequent labels" you may be contrasting a
  *response* label against a *stimulus* label and get a meaninglessly easy decode.
- **Channel count and duration** — matches expectations?
- **Modality** — `eeg` and `meg` datasets can share an id; check which files actually exist.
- **License** — CC0 vs CC BY-NC-SA changes what you may publish.

## Verifying on the node

For anything the metadata cannot answer (actual event counts, sampling rate, symlink resolution),
run a small `PY_EXPANSE` job that walks `$NEMARPATH/<ds>` and prints a summary — see
`probes/nemar_probe/` (dataset inventory) and `probes/nemar_load/` (read a real recording with MNE).
Prefer one cheap inspection job over a large analysis job built on an assumption.
