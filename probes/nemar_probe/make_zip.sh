#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
stage="nemar_probe"
rm -rf "$stage" "$stage.zip"
mkdir -p "$stage"; cp run.py "$stage/"
zip -r -q "$stage.zip" "$stage"; rm -rf "$stage"
echo "built $(pwd)/$stage.zip"; unzip -l "$stage.zip"
