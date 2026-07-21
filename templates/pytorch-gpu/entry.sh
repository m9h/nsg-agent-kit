#!/usr/bin/env bash
# Runs ON the Expanse compute node (no internet assumed). Offline-install vendored wheels, then run.
set -euo pipefail
cd "$(dirname "$0")"

if [ -d vendor ] && [ -f requirements.txt ]; then
  echo "installing vendored deps offline ..."
  python -m pip install --no-index --find-links vendor/ -r requirements.txt
fi

echo "python: $(python --version 2>&1)"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
python run.py "$@"
