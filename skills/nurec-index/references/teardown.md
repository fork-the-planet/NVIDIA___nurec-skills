# Cross-skill teardown

A complete NuRec workflow can leave **150 GB+ on disk** between
container images, model weights, code clones, conda envs, and
output directories. Each sibling skill has its own dedicated
"Teardown" section — use them in this order when the workflow is
no longer needed:

| Sibling skill | What to read | Approximate footprint |
|---------------|--------------|------------------------|
| `nre` | `nre/SKILL.md#teardown` + `nre/references/teardown.md` | ~120 GB images + caches + per-run outputs |
| `nurec-fixer` | `nurec-fixer/SKILL.md#teardown` + `nurec-fixer/references/teardown.md` | 100 GB+ possible between Cosmos image/build cache, HF model/dataset, checkout, and outputs |
| `asset-harvester` | `asset-harvester/references/troubleshooting.md#teardown` | ~30 GB conda envs + checkpoints + outputs |
| `ncore` | NCore shards live under `<dataset_dir>/`; delete after NRE training is done | clip-dependent |
| `physical-ai-datasets` | HF caches live under `${HF_HOME:-$HOME/.cache/huggingface}/hub/`; remove per-dataset directories | dataset-dependent |

## Practical tips across every container-based skill

1. **Pin `-u $(id -u):$(id -g)` on every `docker run`** so outputs
   land owned by the invoking user, not by `root`. Every documented
   `docker run` command in `nre` and `nurec-fixer` already does
   this. If outputs end up `root`-owned anyway, recover with
   `sudo chown -R "$(id -u):$(id -g)" <output_dir>` before deleting.
2. **Do not revoke `NGC_API_KEY` / `HF_TOKEN` as part of teardown**
   unless there is reason to believe they were leaked — they are
   per-user and shared across every NVIDIA workflow on the host.

## Secrets handling across every sibling skill

Every sibling skill on this index includes a `Verifying secrets
safely` block in its Prerequisites section. **Always verify
prerequisites by running `scripts/validate_setup.py` (where it
exists) or, for skills without one (`ncore`,
`physical-ai-datasets`, this index), use `hf auth whoami` or a
length-only shell check. Never write ad-hoc bash that interpolates
secret values.**

In particular, do not use this bash anti-pattern:

```bash
# BAD — leaks the secret to the terminal when the variable is set
echo "HF_TOKEN: ${HF_TOKEN:+yes}${HF_TOKEN:-no}"
```

It prints `yes<token-value>` because `${VAR:-no}` only falls back
to "no" when the variable is empty. If a token was echoed, rotate
it (`huggingface.co/settings/tokens`,
`org.ngc.nvidia.com/setup/api-key`) before continuing.
