#!/usr/bin/env bash
# Build the NSG upload zip for the frozen probe.
# NSG requires: top level of the zip is exactly ONE directory containing the code.
set -euo pipefail
cd "$(dirname "$0")"

stage="reve_frozen_probe"          # the single top-level dir inside the zip
rm -rf "$stage" "$stage.zip"
mkdir -p "$stage"
cp run.py "$stage/"                 # entry script (set as the tool's runfile = run.py)

zip -r -q "$stage.zip" "$stage"
rm -rf "$stage"
echo "built $(pwd)/$stage.zip"
unzip -l "$stage.zip"
