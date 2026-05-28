# NRE — Teardown / disk cleanup

A full NRE workflow leaves ~120 GB of container images, a Fixer
weights cache, NRE caches under `${HOME}/.cache/nre/`, and a
clip-sized `<output_dir>/<RUN-ID>/` tree. Reclaim disk with:

```bash
# 1. Stop any long-lived serve-grpc container.
docker ps --format '{{.ID}} {{.Image}} {{.Command}}' \
  | awk '/serve-grpc/ {print $1}' | xargs -r docker rm -f

# 2. Remove NRE container images (~120 GB).
docker image rm nvcr.io/nvidia/nre/nre:latest nvcr.io/nvidia/nre/nre-tools:latest 2>/dev/null || true
docker image prune -f && docker builder prune -f

# 3. Remove NRE caches (Fixer weights + Hydra-resolved configs).
rm -rf "${HOME}/.cache/nre"

# 4. Remove outputs (no sudo needed because runs used -u $(id -u):$(id -g)).
rm -rf /path/to/output/<RUN-ID>

# 5. (optional) Remove NCore shards if no longer needed.
rm -rf /path/to/dataset/<NAME>.zarr.itar /path/to/dataset/<NAME>.aux.*.zarr

# 6. (optional) Logout from NGC.
docker logout nvcr.io
```

Recover from outputs created **without** `-u` (root-owned):

```bash
sudo chown -R "$(id -u):$(id -g)" /path/to/output
rm -rf /path/to/output/<RUN-ID>
```

Do **not** revoke `NGC_API_KEY` unless you suspect it has been
leaked (see `install.md` "Verifying secrets safely").

## Post-teardown verification

```bash
docker images | grep -E 'nvcr.io/nvidia/nre/(nre|nre-tools)' || echo "images: clean"
du -sh "${HOME}/.cache/nre" 2>/dev/null || echo "cache: clean"
ls -la /path/to/output/<RUN-ID> 2>/dev/null || echo "output: clean"
```
