#!/usr/bin/env bash
# Runs ON the Expanse compute node.
#  - the tool's system venv is READ-ONLY, so install deps into a writable dir and add to PYTHONPATH
#  - install into NODE-LOCAL SCRATCH (outside the job working dir), because NSG tars the ENTIRE
#    working dir back as output.tar.gz — a jax[cuda12] install in ./ makes that ~3.3 GB. Keep libs
#    out of the returned dir.
#  - run.py runs as a FRESH interpreter that inherits PYTHONPATH, so the target's numpy wins over the
#    image's older numpy (importing numpy in this shell first would pin the old one).
set -euo pipefail
cd "$(dirname "$0")"

TARGET="${TMPDIR:-/tmp}/nsgkit-pylibs"      # node-local scratch, NOT returned in output.tar.gz
mkdir -p "$TARGET"

if [ -f requirements.txt ]; then
  if [ -d vendor ]; then
    echo "installing deps from vendored wheels into $TARGET ..."
    python -m pip install --no-index --find-links vendor/ --target "$TARGET" --ignore-installed -r requirements.txt
  else
    echo "installing deps from PyPI into $TARGET (system venv is read-only) ..."
    # --ignore-installed makes the target self-contained (pulls a modern numpy etc. that the
    # image's pinned numpy would otherwise shadow).
    python -m pip install --target "$TARGET" --ignore-installed -r requirements.txt
  fi
fi

export PYTHONPATH="$TARGET:${PYTHONPATH:-}"
echo "python: $(python --version 2>&1)"
python run.py "$@"
