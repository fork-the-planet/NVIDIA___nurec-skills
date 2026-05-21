# Teardown — nurec-fixer

A complete Fixer V2 workflow leaves roughly **~127 GB** on the host:

| Artifact | Size | Source |
|----------|------|--------|
| `nvcr.io/nvidia/cosmos/cosmos-predict2-container:1.2` base image | ~5 GB | `docker pull` in Container setup |
| `nurec-fixer:v2` layered image | ~120 GB | `docker build -f Dockerfile.cosmos` |
| HuggingFace `nvidia/Fixer` weights (`./models/`) | ~5 GB | `huggingface-cli download` |
| HuggingFace hub cache (`~/.cache/huggingface/hub/...`) | duplicate of `./models` | `huggingface-cli` |
| `nv-tlabs/Fixer` source clone | <1 GB | `git clone` |
| `${HOME}/.cache/nre/difix/` (if `--enable-difix` was used in `nre`) | ~5 GB | NRE on first run |
| Output frames + `metrics.yaml` | sequence-dependent | inference / evaluation runs |

## Reclaim disk

Run these in order. Each block is independent — skip blocks that
don't apply.

### 1. Container images (~125 GB)

```bash
docker image rm nurec-fixer:v2 2>/dev/null || true
docker image rm nvcr.io/nvidia/cosmos/cosmos-predict2-container:1.2 2>/dev/null || true
docker image prune -f
docker builder prune -f       # drops Dockerfile.cosmos build cache
```

### 2. HuggingFace V2 weights (~5 GB)

```bash
rm -rf ./models               # adjust if --local-dir was something else
rm -rf "${HF_HOME:-$HOME/.cache/huggingface}/hub/models--nvidia--Fixer"
```

### 3. V2 source clone (<1 GB)

```bash
rm -rf "${FIXER_REPO_DIR:-$PWD/Fixer}"
```

### 4. Render outputs (sequence-dependent)

These were written with `-u $(id -u):$(id -g)` and can be removed
without `sudo`:

```bash
rm -rf /absolute/path/to/enhanced_frames
rm -rf /absolute/path/to/eval_out
```

### 5. NRE difix cache (~5 GB, only if you used `--enable-difix`)

```bash
rm -rf "${HOME}/.cache/nre/difix"
```

## Outputs already owned by root

If outputs were created **before** the `-u $(id -u):$(id -g)` flag was
added to `docker run`, they are owned by `root:root` and require
`sudo` to delete or modify:

```bash
sudo chown -R "$(id -u):$(id -g)" /absolute/path/to/enhanced_frames
rm -rf /absolute/path/to/enhanced_frames
```

## Verify

```bash
docker images | grep -E 'cosmos-predict2-container|nurec-fixer' || echo "images: clean"
du -sh ./models "${HF_HOME:-$HOME/.cache/huggingface}/hub/models--nvidia--Fixer" 2>/dev/null || echo "weights: clean"
```

## Secrets

Do **not** revoke `HF_TOKEN` as part of teardown unless you suspect it
has been leaked (see `Verifying secrets safely` in `SKILL.md`); the
token is per-user and shared across HuggingFace workflows. If you
*did* echo the token to stdout — for example via the
`${HF_TOKEN:+yes}${HF_TOKEN:-no}` anti-pattern that expands to the
literal value when set — rotate it immediately at
https://huggingface.co/settings/tokens.
