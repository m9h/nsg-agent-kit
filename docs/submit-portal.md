# Submitting via the NSG web portal (GUI)

For a human (or an agent driving a browser that shares the login session). Portal:
`https://nsgprod.sdsc.edu:8443/portal2/`.

## One-time
1. **Get an account.** New users: "New users who are interested in getting an account should fill out
   the form" on the login page. Approval is usually quick for academic email.
2. Read the usage policy (linked on the login page). De-identify any human-subject data.

## Per job
1. **Log in.**
2. **Data → Upload Data.** Upload your `.zip`. Rules that actually matter:
   - The zip's **top level must be exactly one directory**; your code lives inside it.
   - The **entry file** is named per the tool (Python tools: you set the "main" filename in a task
     parameter — commonly `run.py`/`input.py`; MATLAB: `input.m`). Keep it at the top of that directory.
3. **Tasks → Create New Task.**
   - **Select Input Data:** the zip you uploaded.
   - **Select Tool:** `PyTorch Python on Expanse` (GPU) for a torch job; `Python on Expanse` (3.11.4)
     for CPU. **Picking the GPU tool is how you get a GPU** — there's no separate GPU checkbox.
   - **Set Parameters:** on this page you'll see the real, authenticated values — **record them into
     the repo**: max wallclock hours (≤48), node/core count, memory, and the GPU count field. Set the
     "main input file" / runfile parameter to your entry script name.
   - **Save & Run Task.**
4. **Monitor:** Tasks list shows status; enable email notification. Jobs run on Expanse via Slurm.
5. **Retrieve:** when complete, download the output zip (your whole working dir). Extract the returned
   `metrics.json` / logs into `probes/reve_frozen_probe/results/` to turn the TDD test green.

## What to capture the first time (closes must-verify items)
- The exact **tool ID** string (visible on the task page / in the task summary) → put in
  `nsgr/config.example.env` so REST submission works.
- The **GPU count, cores, RAM, max-hours** fields and their defaults → update `FINDINGS.md`.
- Whether there's a field to set a **runfile** or a fixed convention → update the templates' README.
