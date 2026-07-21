#!/usr/bin/env bash
# Assemble the single-top-dir upload zip: code + vendored wheels.
set -euo pipefail
cd "$(dirname "$0")"
stage="pytorch-gpu-job"
rm -rf "$stage" "$stage.zip"
mkdir -p "$stage"
cp run.py entry.sh requirements.txt "$stage/"
[ -d vendor ] && cp -r vendor "$stage/vendor"
[ -d data ]   && cp -r data   "$stage/data"
chmod +x "$stage/entry.sh"
zip -r -q "$stage.zip" "$stage"
rm -rf "$stage"
echo "built $(pwd)/$stage.zip"; unzip -l "$stage.zip" | tail -5
