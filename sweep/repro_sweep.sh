#!/usr/bin/env bash
# Fan out the REVE paper-reproduction cell across BCI IV 2a subjects, poll, fetch, aggregate.
# Usage:  ./repro_sweep.sh "1 2 3 4 5 6 7 8 9"     (default: all 9)
# Requires: nsgr/config.env + ~/.nsg_secret.env sourced (creds), sweep/reve_weights vendored.
set -euo pipefail
cd "$(dirname "$0")/.."
source nsgr/config.env; source ~/.nsg_secret.env
SUBJECTS="${1:-1 2 3 4 5 6 7 8 9}"
OUT=sweep/results/repro; mkdir -p "$OUT"
HANDLES="$OUT/handles.tsv"; : > "$HANDLES"

echo "== submit =="
for s in $SUBJECTS; do
  stage="repro_s$s"; rm -rf "/tmp/$stage" "/tmp/$stage.zip"; mkdir -p "/tmp/$stage"
  cp sweep/repro_cell.py "/tmp/$stage/run.py"
  echo "{\"dataset\":\"BNCI2014_001\",\"subject\":$s,\"torch\":\"2.4.1\"}" > "/tmp/$stage/cell.json"
  cp -r sweep/reve_weights/reve-base sweep/reve_weights/reve-positions "/tmp/$stage/"
  ( cd /tmp && zip -qr "$stage.zip" "$stage" )
  h=$(curl -sS -m 300 -u "$NSG_USER:$NSG_PASSWORD" -H "cipres-appkey:$NSG_APPKEY" \
      "$NSG_URL/job/$NSG_USER" -F tool=PYTORCH_PY_EXPANSE -F input.infile_=@/tmp/$stage.zip \
      -F vparam.filename_=run.py -F vparam.subdirname_=$stage \
      -F vparam.runtime_=1.5 -F vparam.number_gpus_=1 -F vparam.number_gbmemorypernode_=32 \
      -F metadata.statusEmail=false | grep -oE '<jobHandle>[^<]+' | sed 's/<jobHandle>//')
  printf '%s\t%s\n' "$s" "$h" >> "$HANDLES"
  echo "  subj $s -> $h"
  sleep 3   # be gentle with NSG-R rate limits
done

echo "== poll until all terminal =="
while :; do
  pending=0
  while IFS=$'\t' read -r s h; do
    t=$(curl -sS -m 30 -u "$NSG_USER:$NSG_PASSWORD" -H "cipres-appkey:$NSG_APPKEY" \
        "$NSG_URL/job/$NSG_USER/$h" 2>/dev/null | grep -oE '<terminalStage>[^<]+' | sed 's/<[^>]*>//g')
    [ "$t" = "true" ] || pending=$((pending+1))
  done < "$HANDLES"
  echo "  $(date +%H:%M:%S) pending=$pending"
  [ "$pending" = "0" ] && break
  sleep 90
done

echo "== fetch =="
while IFS=$'\t' read -r s h; do
  curl -sS -m 30 -u "$NSG_USER:$NSG_PASSWORD" -H "cipres-appkey:$NSG_APPKEY" \
    "$NSG_URL/job/$NSG_USER/$h/output" -o "/tmp/ol_s$s.xml" 2>/dev/null
  sid=$(python3 -c "import re;x=open('/tmp/ol_s$s.xml').read();ids=[re.search(r'outputDocumentId>(\d+)',j).group(1) for j in re.findall(r'<jobfile>.*?</jobfile>',x,re.S) if 'STDOUT' in j];print(ids[0] if ids else '')")
  curl -sS -m 30 -u "$NSG_USER:$NSG_PASSWORD" -H "cipres-appkey:$NSG_APPKEY" \
    "$NSG_URL/job/$NSG_USER/$h/output/$sid" 2>/dev/null | \
    python3 -c "import sys,re,json;m=re.search(r'\{.*\}',sys.stdin.read(),re.S);open('$OUT/subject_$s.json','w').write(m.group(0) if m else '{}')"
  echo "  saved $OUT/subject_$s.json"
done < "$HANDLES"

echo "== aggregate =="
python3 sweep/aggregate_repro.py "$OUT"
