#!/usr/bin/env bash
# Runs ON the Expanse compute node. The tool's system venv is READ-ONLY, so we install deps into a
# writable target dir in the job cwd and put it on PYTHONPATH. Works whether deps come from PyPI
# (runtime pip — egress is available) or from vendored wheels.
set -euo pipefail
cd "$(dirname "$0")"

TARGET="$PWD/pylibs"
mkdir -p "$TARGET"

if [ -f requirements.txt ]; then
  if [ -d vendor ]; then
    echo "installing deps from vendored wheels into $TARGET ..."
    python -m pip install --no-index --find-links vendor/ --target "$TARGET" -r requirements.txt
  else
    echo "installing deps from PyPI into $TARGET (system venv is read-only) ..."
    python -m pip install --target "$TARGET" -r requirements.txt
  fi
fi

export PYTHONPATH="$TARGET:${PYTHONPATH:-}"
echo "python: $(python --version 2>&1)"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
python run.py "$@"
