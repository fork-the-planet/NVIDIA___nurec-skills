# Teardown — nurec-fixer

A complete unified Fixer & Harmonizer workflow leaves roughly **~30 GB**
on the host:

| Artifact | Size | Source |
|----------|------|--------|
| `nvcr.io/nvidia/pytorch:24.10-py3` base image | ~25 GB | `docker pull` in Container setup |
| `nurec-fixer_vcosmos_3dgut_fixer_harmonizer/` artifact directory (two `.pt` checkpoints + `inference_jit_harmonizer.py` + `requirements.txt`) | ~5 GB | `ngc registry model download-version` |
| Wrapper image (if you baked one per [`wrapper-image.md`](wrapper-image.md)) | +~2 GB on top of base | `docker build` |
| `${HOME}/.cache/nre/difix/` (if `--enable-difix` was used in `nre`) | ~5 GB | NRE on first run |
| Output frames | sequence-dependent | inference runs |

## Reclaim disk

Run these in order. Each block is independent — skip blocks that
don't apply.

### 1. Container images (~25–27 GB)

```bash
docker image rm nurec-harmonizer:beta 2>/dev/null || true     # only if you baked the wrapper image
docker image rm nvcr.io/nvidia/pytorch:24.10-py3 2>/dev/null || true
docker image prune -f
docker builder prune -f       # drops any wrapper-image build cache
```

### 2. NGC harmonizer artifact (~5 GB)

```bash
rm -rf ./nurec-fixer_vcosmos_3dgut_fixer_harmonizer
```

If you used a custom `--dest` for `ngc registry model download-version`,
substitute that path instead. The `.pt` checkpoints are the bulk of
this directory.

### 3. Render outputs (sequence-dependent)

These were written with `-u $(id -u):$(id -g)` and can be removed
without `sudo`:

```bash
rm -rf /absolute/path/to/enhanced_frames
```

### 4. NRE difix cache (~5 GB, only if you used `--enable-difix`)

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
docker images | grep -E 'pytorch:24\.10-py3|nurec-harmonizer' || echo "images: clean"
du -sh ./nurec-fixer_vcosmos_3dgut_fixer_harmonizer 2>/dev/null || echo "artifact: clean"
```

## Secrets

Do **not** revoke `NGC_API_KEY` as part of teardown unless you suspect
it has been leaked (see `Verifying secrets safely` in `SKILL.md`); the
key is per-user and shared across NGC workflows. If you *did* echo the
key to stdout — for example via the `${NGC_API_KEY:+yes}${NGC_API_KEY:-no}`
anti-pattern that expands to the literal value when set — rotate it
immediately at <https://ngc.nvidia.com/setup/personal-keys>.
