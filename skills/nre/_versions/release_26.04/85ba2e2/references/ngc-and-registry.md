# NGC Registry Credentials (Shared)

Read this file when you need to pull `nvcr.io/nvidia/nre/nre-ga:<version>`
(or the legacy `nvcr.io/nvidia/nre/nre:<version>` channel — deprecated but
still valid for cached pins) on the **local Docker backend**. For the OSMO
backend, the canonical reference
is `skills/orion-osmo-cli/SKILL.md` → **Set Up an Image Registry Credential**
— do not re-derive `osmo credential set` flags from memory. The NGC key
resolution (below) is the same on both backends; only where the credential
is installed differs.

---

## Key resolution order

**Do not prompt the user until all automatic sources are exhausted.** Same
rule applies on both backends:

1. **`$NGC_CLI_API_KEY`** — primary source. Many provisioning
   systems (CI runners, cloud images, dev sandboxes) export this in
   `~/.bashrc` / `~/.profile` automatically. Use it if set.
2. **`$NGC_API_KEY`** — secondary fallback.
3. **User prompt** — only if neither env var is present. Point them at
   https://ngc.nvidia.com/setup/api-key.

```bash
NGC_KEY="${NGC_CLI_API_KEY:-${NGC_API_KEY:-}}"
if [ -z "$NGC_KEY" ]; then
  # Ask the user, then set NGC_KEY
  :
fi
```

---

## Local backend — `docker login nvcr.io`

For local `docker run`, the host's Docker config must be authenticated
against `nvcr.io`:

```bash
echo "$NGC_KEY" | docker login nvcr.io -u '$oauthtoken' --password-stdin
```

If login fails or was never performed, `docker pull` / `docker run` will
fail with `unauthorized: authentication required`. Tell the user exactly
that and suggest re-running the login command above.

---

## OSMO backend

See `skills/orion-osmo-cli/SKILL.md` → **Set Up an Image Registry Credential**.
It covers: `osmo credential list` to check for an existing entry, the exact
`osmo credential set --type REGISTRY --payload registry=… username=…
auth=…` command, the `auth=` raw-key (NOT base64) gotcha, and how to
reference the credential from workflow YAML.

The working
[`example-workflows/osmo/render-usdz.yaml`](example-workflows/osmo/render-usdz.yaml)
and [`example-workflows/osmo/pai-nurec.yaml`](example-workflows/osmo/pai-nurec.yaml)
headers show concrete `osmo credential set` invocations matching the
osmo-cli canonical form.

---

## When to ask the user

Only ask for the API key if both env vars are empty. When you do,
mention https://ngc.nvidia.com/setup/api-key and stop work until the
key is provided — there is no useful fallback: the image is private
and cannot be pulled anonymously.
