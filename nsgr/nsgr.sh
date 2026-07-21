#!/usr/bin/env bash
# nsgr.sh — minimal NSG-R (CIPRES REST) client for submitting jobs to NSG/Expanse.
#
# Credentials come from the environment (never hard-code a password):
#   NSG_URL      e.g. https://nsgr.sdsc.edu:8443/cipresrest/v1
#   NSG_USER     your NSG username
#   NSG_APPKEY   your Application ID (cipres-appkey header)
#   NSG_PASSWORD exported in your shell only (read -s NSG_PASSWORD; export NSG_PASSWORD)
#   NSG_EXTRA    optional extra "-F vparam.x_=y" args, space-separated (tool-specific)
#
# The password is passed to curl via -u and is never echoed by this script.
#
# Usage:
#   ./nsgr.sh submit <TOOL_ID> <job.zip>
#   ./nsgr.sh list
#   ./nsgr.sh status <JOBHANDLE>
#   ./nsgr.sh fetch  <JOBHANDLE> <dest_dir>
set -euo pipefail

: "${NSG_URL:?set NSG_URL (source nsgr/config.env)}"
: "${NSG_USER:?set NSG_USER}"
: "${NSG_APPKEY:?set NSG_APPKEY}"
: "${NSG_PASSWORD:?export NSG_PASSWORD in your shell (do not store it on disk)}"

auth=(-u "${NSG_USER}:${NSG_PASSWORD}" -H "cipres-appkey:${NSG_APPKEY}")
base="${NSG_URL}/job/${NSG_USER}"

cmd="${1:-help}"; shift || true

case "$cmd" in
  submit)
    tool="${1:?tool id, e.g. PYTORCH_EXPANSE}"; zip="${2:?path to job.zip}"
    [ -f "$zip" ] || { echo "no such zip: $zip" >&2; exit 1; }
    # shellcheck disable=SC2206
    extra=(${NSG_EXTRA:-})
    echo "submitting $zip to tool=$tool ..." >&2
    curl -sS "${auth[@]}" "$base" \
      -F tool="$tool" \
      -F input.infile_=@"$zip" \
      -F metadata.statusEmail=true \
      "${extra[@]}"
    echo
    ;;
  list)
    curl -sS "${auth[@]}" "$base"
    echo
    ;;
  status)
    handle="${1:?job handle, e.g. NGBW-JOB-...}"
    curl -sS "${auth[@]}" "$base/$handle"
    echo
    ;;
  fetch)
    handle="${1:?job handle}"; dest="${2:?dest dir}"
    mkdir -p "$dest"
    echo "listing outputs for $handle ..." >&2
    listing="$(curl -sS "${auth[@]}" "$base/$handle/output")"
    # extract every downloadUri; works with or without xmllint
    urls="$(printf '%s' "$listing" | grep -oE 'https://[^<]*/output/[0-9]+' || true)"
    if [ -z "$urls" ]; then
      echo "no output urls found — is the job terminalStage=true? raw listing:" >&2
      printf '%s\n' "$listing" >&2
      exit 1
    fi
    while IFS= read -r u; do
      [ -z "$u" ] && continue
      echo "  downloading $u" >&2
      # -OJ honours the server-provided filename; land files in $dest
      ( cd "$dest" && curl -sS -OJ "${auth[@]}" "$u" )
    done <<< "$urls"
    echo "done -> $dest" >&2
    ;;
  *)
    sed -n '1,26p' "$0"
    ;;
esac
