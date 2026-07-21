# Template: PyTorch GPU job with vendored deps (the LoRA-REVE sweep pattern)

For the `PyTorch Python on Expanse (2.0.1+cu117)` tool when you need packages beyond the stock env
(`peft`, `transformers`, `braindecode`, a `transformer_engine` patch, …) **and** the node has no
runtime internet.

## Files
- `run.py`         — your entry script (set as the tool's runfile).
- `requirements.txt` — the extra deps (NOT torch — the env already has cu117 torch).
- `package_deps.sh` — on a machine with internet, downloads wheels into `vendor/`.
- `entry.sh`       — runs on the node: offline-installs `vendor/` then runs `run.py`.
- `make_zip.sh`    — assembles the single-top-dir zip.

## Workflow
```bash
# 1. Confirm the tool's Python tag first (from the frozen probe's python_version), edit package_deps.sh.
./package_deps.sh            # internet host -> fills vendor/ with wheels + any model checkpoints
./make_zip.sh                # -> pytorch-gpu-job.zip  (contains code + vendor/)
# 2. Submit to the PyTorch tool; set runfile = entry.sh (or run.py if you install elsewhere).
../../nsgr/nsgr.sh submit "$NSG_TOOL" pytorch-gpu-job.zip
```

## Gotchas (see docs/dependencies.md)
- Wheel platform/interpreter tags MUST match the tool (cp3x, manylinux2014, cu117). Mismatch → offline
  install fails.
- Do **not** vendor torch. Pin deps so they don't pull a conflicting torch>=2.2.
- HuggingFace is unreachable at runtime → vendor model checkpoints into `vendor/` too and load them
  from disk (`AutoModel.from_pretrained('vendor/reve-...', local_files_only=True)`).
- If any dep demands newer torch/CUDA, switch to the Apptainer route (dependencies.md, Strategy C).
