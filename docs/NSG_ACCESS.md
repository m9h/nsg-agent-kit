# Getting NSG access: account + NSG-R application key

To submit jobs from scripts or an agent (via the REST API, NSG-R), you need two things: an **NSG
account** and an **Application ID** (the REST "app key"). This is a one-time setup, ~5 minutes.
The portal GUI needs only the account; the app key is what unlocks `curl`/agent submission.

---

## Step 1 — Create an NSG account

1. Go to the portal: **<https://nsgprod.sdsc.edu:8443/portal2/>**
2. On the login page, use the **new-user registration form** ("New users who are interested in
   getting an account should fill out the form").
3. Register with an **academic email**. Approval is usually quick.
4. Review and accept the **usage policy**. Note: only **de-identified** human-subject data may be
   run on NSG.

You can now submit jobs through the web portal. For REST/agent access, continue below.

## Step 2 — Create an NSG-R application (get your app key)

Log in to the portal, then:

1. Open the **Developer** menu (top navigation) → **Application Management** → **Create New
   Application**.
2. Fill the form:
   | Field | What to enter |
   |---|---|
   | **Name** | A short name, **letters/numbers/underscores only** — e.g. `nsg_agent_kit` |
   | **Long Name** | A human-readable label — e.g. `NSG Agent Kit` |
   | **Authentication Type** | **DIRECT** ← leave as-is. DIRECT is for scripts/CLI/agents. (UMBRELLA is for web apps with their own user management and triggers a manual review email — you don't want it.) |
   | **Web Site** | Optional (e.g. your repo URL) — fine to leave blank |
   | **Additional Contacts & Information** | Optional — leave blank |
3. Click **Create**. The application page shows **Status: Active** and an **Application ID** that
   looks like:
   ```
   nsg_agent_kit-B142E82FF04F4349A9F59FA384E38ADD
   ```
   That whole string is your **app key** (the `cipres-appkey` header). Copy it.

> You can **Regenerate Application ID** on that page if it's ever leaked, and create **multiple**
> applications — give each agent/pipeline its own so usage is isolated and independently revocable.

## Step 3 — Store your credentials (never commit secrets)

```bash
cp nsgr/config.example.env nsgr/config.env      # gitignored
```
Edit `nsgr/config.env`:
```bash
export NSG_URL="https://nsgr.sdsc.edu:8443/cipresrest/v1"
export NSG_USER="your_nsg_username"
export NSG_APPKEY="your_application_id"          # the string from Step 2
export NSG_TOOL="PYTORCH_PY_EXPANSE"             # or another tool id (see docs/tool-params.md)
```
**Do not put your password in any file in the repo.** Stash it once in a 0600 file:
```bash
umask 077; read -rsp "NSG password: " P; printf 'export NSG_PASSWORD=%q\n' "$P" > ~/.nsg_secret.env; unset P
```

## Step 4 — Test authentication

```bash
source nsgr/config.env; source ~/.nsg_secret.env
curl -sS -u "$NSG_USER:$NSG_PASSWORD" -H "cipres-appkey:$NSG_APPKEY" "$NSG_URL/job/$NSG_USER"
# HTTP 200 + <joblist> (empty if you have no jobs) = you're in.
```
Then submit with `./nsgr/nsgr.sh` — see [`submit-rest.md`](submit-rest.md).

## Security & fair-use notes

- **One app key per agent/user.** Don't share a single key across independent pipelines — separate
  keys are independently revocable, and NSG-R rate limits are per token.
- **Fair-use is per account.** Multiple agents authenticating as the same user share **one**
  allocation and rate-limit budget. Coordinate heavy sweeps.
- **Rotate on exposure.** If a password is ever pasted into a chat/log, change it (portal → My
  Profile → Change Password). App keys can be regenerated on the application page.
- The app key is not a bearer-only secret — it's used **with** your username + password (HTTP Basic
  auth). All three are required to submit.

## No-auth bonus

The **tool catalog** and each tool's **parameter spec** are public — no account needed:
```bash
curl https://nsgr.sdsc.edu:8443/cipresrest/v1/tool                    # all tool ids
curl https://nsgr.sdsc.edu:8443/cipresrest/v1/tool/PYTORCH_PY_EXPANSE  # + /doc/pise for params
```
