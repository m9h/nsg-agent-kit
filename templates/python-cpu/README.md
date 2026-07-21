# Template: CPU Python job

For the `Python on Expanse (3.11.4)` tool (BMTK, Brian 2, DEAP, HNN-Core, NetPyNE, PyNN, NEST +
the scientific stack). Same rules as the GPU template:

- Zip's top level is exactly one directory; entry script (e.g. `run.py`) inside it.
- No runtime internet — vendor wheels (`docs/dependencies.md`) or rely on the stock env.
- Recommended ≤32–64 cores; ≤48 h wallclock; up to ~243 GB RAM.
- Read NEMAR data from `$NEMARPATH/<ds-id>` (see `../nemar-eeg/run.py`).

Reuse the GPU template's `package_deps.sh` / `entry.sh` / `make_zip.sh`, dropping the torch/CUDA
specifics. This directory is intentionally a pointer, not a copy, to avoid drift.
