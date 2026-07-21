# Submitting via NSG-R (REST API) — the agent-friendly path

NSG-R lets you submit/poll/fetch with plain `curl`, no browser. This is the recommended path for
autonomous agents. Wrapper: [`../nsgr/nsgr.sh`](../nsgr/nsgr.sh).

## One-time setup
1. Have an NSG portal account (see `submit-portal.md`).
2. In the portal: **Developer → Application Management → Create New Application**. Note the
   **Application ID** — this is your `cipres-appkey` (`KEY`).
3. Copy `nsgr/config.example.env` → `nsgr/config.env` and fill in:
   - `NSG_URL=https://nsgr.sdsc.edu:8443/cipresrest/v1`
   - `NSG_USER=<your username>`
   - `NSG_APPKEY=<Application ID>`
   - `NSG_TOOL=<tool id>`  ← the authenticated Task page shows this (e.g. `PYTORCH_EXPANSE`)
   - `NSG_PASSWORD` — **do NOT put your password in the file.** Export it in your shell only:
     `read -s NSG_PASSWORD; export NSG_PASSWORD`  (so it never lands on disk or in logs).

> Credential hygiene: `nsgr.sh` reads `NSG_PASSWORD` from the environment and passes it to `curl`
> via `-u`. It never echoes it. The agent running this never needs to *see* the secret — the human
> exports it once per shell session.

## Submit / poll / fetch

```bash
source nsgr/config.env

# submit (tool id + zip). Prints the job handle (JOBURL/…).
./nsgr/nsgr.sh submit "$NSG_TOOL" probes/reve_frozen_probe/reve_frozen_probe.zip

# poll one handle (or 'list' for all)
./nsgr/nsgr.sh status  NGBW-JOB-XXXX-YYYY
./nsgr/nsgr.sh list

# when terminalStage=true, download every output file into a dir
./nsgr/nsgr.sh fetch   NGBW-JOB-XXXX-YYYY probes/reve_frozen_probe/results/
```

## Raw curl (what the wrapper does)

```bash
# submit
curl -u "$NSG_USER:$NSG_PASSWORD" -H "cipres-appkey:$NSG_APPKEY" \
  "$NSG_URL/job/$NSG_USER" \
  -F tool="$NSG_TOOL" \
  -F input.infile_=@job.zip \
  -F metadata.statusEmail=true

# list jobs
curl -u "$NSG_USER:$NSG_PASSWORD" -H "cipres-appkey:$NSG_APPKEY" "$NSG_URL/job/$NSG_USER"

# a job's status doc (XML) → look for <terminalStage>true</terminalStage> and the results url
curl -u "$NSG_USER:$NSG_PASSWORD" -H "cipres-appkey:$NSG_APPKEY" \
  "$NSG_URL/job/$NSG_USER/NGBW-JOB-XXXX-YYYY"

# list result files, then download each
curl -u "$NSG_USER:$NSG_PASSWORD" -H "cipres-appkey:$NSG_APPKEY" \
  "$NSG_URL/job/$NSG_USER/NGBW-JOB-XXXX-YYYY/output"
```

Notes:
- Some tools require extra `-F vparam.<name>_=<value>` parameters (e.g. runtime hours, runfile name).
  The set is tool-specific; the authenticated Task page (or `.../tool/<TOOLID>`) lists them. Add them
  to `NSG_EXTRA` in `config.env` and they'll be appended by the wrapper.
- Responses are XML. `nsgr.sh` greps the fields it needs; for scripting, pipe through `xmllint`.
- Reference: NSG-R guide <https://www.nsgportal.org/guide.html>.
