#!/usr/bin/env bash
# Run on a machine WITH internet. Downloads wheels for the extra deps into vendor/ so they can be
# installed offline on the Expanse node. Match --python-version / --platform to the tool env
# (confirm the Python tag from the frozen probe's reported python_version).
set -euo pipefail
cd "$(dirname "$0")"

PYVER="${PYVER:-3.10}"                          # <-- set to the tool's actual Python
PLAT="${PLAT:-manylinux2014_x86_64}"

mkdir -p vendor
pip download \
  --only-binary=:all: \
  --python-version "$PYVER" --platform "$PLAT" --implementation cp \
  --dest vendor/ \
  -r requirements.txt

echo "vendored $(ls vendor | wc -l) files into vendor/"
echo "NOTE: also place any model checkpoints (e.g. REVE weights) under vendor/ — HF is unreachable at runtime."
