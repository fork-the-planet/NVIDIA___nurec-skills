# Teardown — nre

An end-to-end NRE workflow leaves a substantial footprint on the
host. Reclaim disk in the order below.

| Artifact | Typical size | Source |
|----------|--------------|--------|
| `nvcr.io/nvidia/nre/nre:latest` | ~60 GB | `docker pull nvcr.io/nvidia/nre/nre:latest` |
| `nvcr.io/nvidia/nre/nre-tools:latest` | ~60 GB | `docker pull nvcr.io/nvidia/nre/nre-tools:latest` |
| `${HOME}/.cache/nre/difix/` (Fixer weights) | ~5 GB | First run of `--enable-difix` |
| `<output_dir>/<RUN-ID>/checkpoints/`, `usd-out/`, `val/`, `renders/`, `lidar/` | clip-dependent (often 10–50 GB / clip) | Training, export, and render commands |
| NCore shards + aux data under `<dataset_dir>/` | clip-dependent | `ncore-data-conversion` / `nre-tools` aux generation |

## 1. Stop long-lived processes

```bash
# If you have a serve-grpc instance running, stop it cleanly so
# the gRPC port is freed and any in-flight client requests fail
# loudly instead of timing out.
docker ps --format '{{.ID}} {{.Image}} {{.Command}}' \
  | awk '/serve-grpc/ {print $1}' \
  | xargs -r docker rm -f
```

## 2. Container images (~120 GB total)

```bash
docker image rm nvcr.io/nvidia/nre/nre:latest 2>/dev/null || true
docker image rm nvcr.io/nvidia/nre/nre-tools:latest 2>/dev/null || true
docker image prune -f
docker builder prune -f
```

## 3. NRE-owned caches

```bash
rm -rf "${HOME}/.cache/nre/difix"   # Fixer weights NRE downloaded for --enable-difix
rm -rf "${HOME}/.cache/nre"         # all NRE caches (Hydra resolved configs, etc.)
```

## 4. Outputs and ground-truth exports

Because every documented `docker run` in `SKILL.md` uses
`-u "$(id -u):$(id -g)"`, you do **not** need `sudo` to remove
artifacts under `<output_dir>/<RUN-ID>/`:

```bash
rm -rf /path/to/output/<RUN-ID>
```

If you ever ran a `docker run` **without** the `-u` flag and ended up
with `root`-owned artifacts (`side_by_side.mp4`, harmonized PNGs,
`metrics.yaml`, …), recover ownership first:

```bash
sudo chown -R "$(id -u):$(id -g)" /path/to/output
rm -rf /path/to/output/<RUN-ID>
```

## 5. NCore shards (only if you no longer need the input)

```bash
rm -rf /path/to/dataset/<NAME>.zarr.itar /path/to/dataset/<NAME>.aux.*.zarr
```

## 6. Logout from NGC (optional)

```bash
docker logout nvcr.io
```

Do **not** revoke `NGC_API_KEY` unless you suspect it has been leaked
(see the `Verifying secrets safely` section in `SKILL.md`).

## Verify

```bash
docker images | grep -E 'nre|nre-tools' || echo "images: clean"
du -sh "${HOME}/.cache/nre" 2>/dev/null || echo "caches: clean"
du -sh /path/to/output/<RUN-ID> 2>/dev/null || echo "output: clean"
```
