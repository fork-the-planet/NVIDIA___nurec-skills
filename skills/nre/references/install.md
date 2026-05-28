# NRE — Install and authenticate

Two public images carry every workflow in `SKILL.md`. Pull both:

```bash
export NGC_API_KEY="<your key from https://org.ngc.nvidia.com/setup/api-key>"
echo "${NGC_API_KEY}" | docker login nvcr.io --username "\$oauthtoken" --password-stdin

docker pull nvcr.io/nvidia/nre/nre:latest          # train / val / export / render / serve-grpc / render-grpc / viewer
docker pull nvcr.io/nvidia/nre/nre-tools:latest    # aux data + asset-harvester subcommand
```

> Pin a specific release tag when re-runnability matters (e.g.
> `:25.11`, `:26.04`). `:latest` floats and may change behaviour
> between checkout sessions.

Some NGC releases publish the same images under the
`nvcr.io/nvidia/nre/nre-ga` and `nvcr.io/nvidia/nre/nre-tools-ga`
names to mark General Availability builds; substitute those names
if the user has been directed to the GA channel.

The container expects two host bind mounts: dataset (read-only is
OK once you've generated aux data) and output (read-write).

## Verifying secrets safely

**Always verify prerequisites by running
[`scripts/validate_setup.py`](../scripts/validate_setup.py); never
by writing ad-hoc bash that interpolates `NGC_API_KEY` / `HF_TOKEN`
values.** The common one-liner

```bash
# BAD — leaks the secret to the terminal when the variable is set
echo "NGC_API_KEY: ${NGC_API_KEY:+yes}${NGC_API_KEY:-no}"
```

prints `yes<token-value>` whenever `NGC_API_KEY` is set, because
`${VAR:-no}` only falls back to "no" when `VAR` is empty — when set
it expands to `$VAR`. Use a length-only check instead, which never
echoes the value:

```bash
# OK — prints "set (N chars)" or "missing", never the value
test -n "$NGC_API_KEY" && echo "NGC_API_KEY: set (${#NGC_API_KEY} chars)" || echo "NGC_API_KEY: missing"
```

`docker login nvcr.io` should always use `--password-stdin` (as
above) so the key never appears in process tables or shell history.
If you suspect a token was echoed, rotate it at
<https://org.ngc.nvidia.com/setup/api-key>.

## Prerequisites (full matrix)

- **OS / arch:** Linux x86_64. Linux aarch64 is **not** supported.
- **GPU / driver:** NVIDIA GPU with CUDA 12.8 capability and >= 24
  GB VRAM (48 GB+ recommended).
  - Ampere (A100/A10/A40/RTX A6000): R550+ required, R570+ recommended.
  - Ada (L20/L40/L40S): same as Ampere.
  - Hopper (H100/H20): same as Ampere.
  - Blackwell (RTX Pro 6000D): R580+ required.
  - Fixer-only host (no NRE training): R535+ acceptable.
- **Container runtime:** Docker >= 23.0.1; NVIDIA Container Toolkit
  >= 1.13.5. Verify with `docker run --rm --runtime=nvidia
  --gpus all ubuntu nvidia-smi`.
- **NGC:** account at <https://catalog.ngc.nvidia.com/> and an API
  key generated at <https://org.ngc.nvidia.com/setup/api-key>.
  Export as `NGC_API_KEY` and `docker login nvcr.io` with
  `Username: $oauthtoken`. The NGC CLI is optional but required for
  downloading the gRPC protobuf bundle (see
  `grpc-api.md`).
- **Disk:** budget tens of GB per clip for NCore shards + aux data
  + checkpoints + exports.
- **Memory / shm:** `--shm-size=64g` for training / export /
  render, `--shm-size=2g` for `nre-tools`.
- **File ownership.** Every documented `docker run` threads
  `-u "$(id -u):$(id -g)"` so artifacts are owned by the host
  user. Without that flag the container runs as `root` and every
  output requires `sudo` to move, rename, or delete. Recover with
  `sudo chown -R "$(id -u):$(id -g)" <output_dir>`.
