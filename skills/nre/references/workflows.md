# NRE — End-to-end workflows (A – I)

Workflow A is the headline path; B – I are variants that compose
the same primitives differently. Always pair these with the CLI
recipes in [`cookbook.md`](cookbook.md).

## Workflow A — NCore clip → reconstruction → novel-view render

1. `python scripts/validate_setup.py` — host prerequisites green.
2. Generate auxiliary data with `nvcr.io/nvidia/nre/nre-tools:latest`
   (see `aux-data.md`). Confirm `<NAME>.aux.sseg.zarr` and
   (optionally) `<NAME>.aux.depth.zarr` sit next to `<NAME>.zarr.itar`.
3. Train + validate with the [cookbook.md train recipe](cookbook.md).
   Wait for `usd-out/last.usdz` to appear and `metrics.yaml`
   `test/psnr` to stabilise.
4. `export-gaussian-plys` (optional) to emit Gaussians for offline
   tooling, or `export-mesh` / `export-ground-mesh` for downstream
   Omniverse use.
5. Render — choose **(a)** local `nre render` for batch novel-view
   jobs without a server, or **(b)** `serve-grpc` + `render-grpc`
   (or your own Python client via `nre.grpc.protos`) when you need
   actor editing, LiDAR rendering, or a long-lived service.

## Workflow B — Render the NVIDIA Physical AI dataset

Skip the training step entirely. Download the pre-reconstructed
USDZs from HuggingFace (gated dataset
`nvidia/PhysicalAI-Autonomous-Vehicles-NuRec`), then jump straight
to `serve-grpc` + a Python client. Full code for coordinate-frame
conversion (NuRec ↔ ECEF ↔ OpenDRIVE ENU) lives in
`physical-ai-render.md`.

## Workflow C — Insert Asset-Harvester actors into a USDZ

1. Run `asset-harvester` on the relevant `track_ids` from the
   source NCore clip to produce a directory of `.ply` +
   `metadata.yaml` per asset.
2. `export-external-assets` to repackage the AH directory into a
   new USDZ and produce a stub `edit-assets.json`.
3. Edit the JSON (`replace`, `remove`, `insert`) — schema lives in
   `asset-editing.md`.
4. `serve-grpc` with `--enable-editing-actors`, then `render-grpc
   --edit-assets <path>` to produce edited frames.

## Workflow D — Multi-GPU training

```bash
# Explicit: 4 GPUs on 1 node.
docker run --shm-size=64g -it --rm --gpus all \
  -u "$(id -u):$(id -g)" \
  -e NGC_API_KEY=${NGC_API_KEY} \
  -e CUDA_VISIBLE_DEVICES=0,1,2,3 \
  --volume /path/to/dataset:/workdir/dataset \
  --volume /path/to/output:/workdir/output \
  nvcr.io/nvidia/nre/nre:latest \
  --config-name=configs/apps/AV/Waymo/3dgut_dynamic_mcmc.yaml \
  mode=trainval \
  dataset.path=/workdir/dataset/<NAME>.json \
  out_dir=/workdir/output \
  logger=tensorboard \
  trainer.world_size=4 trainer.num_nodes=1
```

NRE auto-scales LR, schedule, and dataloader workers (toggle via
`trainer.relative_lr` / `relative_schedule` / `relative_num_workers`).
On SLURM, set `trainer.world_size=0 trainer.num_nodes=0` for
auto-detect. Reconstruction quality plateaus past ~6 GPUs, so for
single-clip jobs prefer 4–6 GPUs over higher counts.

## Workflow E — Resume training

```bash
docker run --shm-size=64g -it --rm --gpus all \
  -u "$(id -u):$(id -g)" \
  -e NGC_API_KEY=${NGC_API_KEY} \
  --volume /path/to/dataset:/workdir/dataset \
  --volume /path/to/output:/workdir/output \
  nvcr.io/nvidia/nre/nre:latest \
  --config-name=/workdir/output/<RUN-ID>/config/parsed.yaml \
  mode=train \
  resume=last
```

`resume=last` resolves to `<RUN-ID>/checkpoints/last.ckpt`; pass an
absolute path or `"epoch=4.ckpt"` to pick a different snapshot. Use
`resume_weights_only=true` for transfer fine-tuning. `out_dir`
cannot change between the original and resumed runs.

## Workflow F — Upgrade an old USDZ once (avoid the in-memory upgrade tax)

```bash
docker run --shm-size=64g -it --rm --gpus all \
  -u "$(id -u):$(id -g)" \
  --volume /path/to/output:/workdir/output \
  nvcr.io/nvidia/nre/nre:latest \
  upgrade-artifact \
  --input  /workdir/output/<OLD-RUN>/usd-out/last.usdz \
  --output /workdir/output/<OLD-RUN>/usd-out/last.upgraded.usdz \
  --target-version 26.4
```

`upgrade-config` is the equivalent for a standalone YAML /
`parsed.yaml` extracted from a USDZ. To inspect the structure of a
checkpoint before/after upgrade use `export-artifact-structure`.

## Workflow G — Headless rendering of LiDAR sweeps

Combine a long-lived `serve-grpc` (e.g. on a render node) with one
or more `render-grpc --lidar` workers:

```bash
# Server (one terminal).
docker run --shm-size=64g -it --rm --gpus all \
  -u "$(id -u):$(id -g)" \
  --net=host --privileged \
  -e NGC_API_KEY=${NGC_API_KEY} \
  --volume /path/to/output:/workdir/output \
  nvcr.io/nvidia/nre/nre:latest \
  serve-grpc \
  --artifact-glob "/workdir/output/<RUN-ID>/usd-out/last.usdz" \
  --renderer nrend \
  --max-workers 4 \
  --cache-size 4

# Client (another terminal / job).
docker run --shm-size=64g -it --rm --gpus all \
  -u "$(id -u):$(id -g)" \
  --network host \
  --volume /path/to/output:/workdir/output \
  nvcr.io/nvidia/nre/nre:latest \
  render-grpc \
  --artifact-path /workdir/output/<RUN-ID>/usd-out/last.usdz \
  --output-dir /workdir/output/<RUN-ID>/lidar \
  --lidar --lidar-id lidar_top \
  --lidar-format ply \
  --lidar-raydrop-threshold 0.5 \
  --lidar-opacity-threshold 0.0
```

For RGB + LiDAR + actor editing in the same loop, write a Python
client against `nre.grpc.protos` — see `grpc-api.md`.

## Workflow H — Browse a USDZ in a browser

```bash
docker run --shm-size=64g -it --rm --gpus all \
  -u "$(id -u):$(id -g)" \
  --net=host \
  -e NGC_API_KEY=${NGC_API_KEY} \
  --volume /path/to/output:/workdir/output \
  nvcr.io/nvidia/nre/nre:latest \
  viewer \
  --artifact-path /workdir/output/<RUN-ID>/usd-out/last.usdz \
  --port 8080
```

Open `http://localhost:8080` (or the host's routable IP). Use
`ply_viewer --ply-path <ply>` to inspect a standalone PLY (e.g.
exported via `export-gaussian-plys`).

## Workflow I — Evaluate rendering quality against ground truth

1. `export-ncore-benchmark-gt` to materialize the GT folder layout
   (`<gt-dir>/camera_images/<seq>/<ts>.<ext>` plus optional
   `<gt-dir>/camera_ego_masks/<seq>.png`).
2. Render the trained USDZ along the same trajectory (`render` or
   `render-grpc`). Use `--frame-naming frame-end-timestamp` so
   frames match the GT naming.
3. Compute metrics:

   ```bash
   docker run --rm --gpus all \
     -u "$(id -u):$(id -g)" \
     --volume /path/to/output:/workdir/output \
     nvcr.io/nvidia/nre/nre:latest \
     eval-rendering-metrics \
     --render-dir /workdir/output/<RUN-ID>/renders \
     --gt-dir     /workdir/output/<RUN-ID>/gt \
     --output-dir /workdir/output/<RUN-ID>/metrics \
     --metrics psnr --metrics ssim --metrics lpips
   ```

   For one-off pairs, `compute-metrics --metric
   psnr|ssim|lpips|fid|...` is a lightweight alternative.

## Warm-server thin-client quick start

Use this path when a session will render the same USDZ repeatedly
or render several cameras per pose. It keeps one `serve-grpc`
container alive and calls it from host-side Python.

```bash
# 1. Authenticate to NGC once per host; see install.md.
NGC_KEY="${NGC_CLI_API_KEY:-${NGC_API_KEY:-}}"
echo "$NGC_KEY" | docker login nvcr.io -u '$oauthtoken' --password-stdin

# 2. Extract protobuf stubs from the cached NRE image.
bash NRE_RenderClient/scripts/setup_protos.sh

# 3. Boot the warm server. NRE_GRPC_USDZ_HOST_DIR is the host
#    root that contains one or more USDZ files.
NRE_GRPC_USDZ_HOST_DIR=/path/to/usdz/root \
  bash scripts/session_warm_server.sh

# 4. Render one or more cameras with the thin client.
python3 NRE_RenderClient/scripts/thin_client.py \
  --cameras camera_front_wide_120fov \
  --rig-file rig-json/augmented_rig.json \
  --output-dir /tmp/nre-out/turn-001

# 5. Encode frames to MP4 if needed, then tear down at session end.
bash scripts/session_teardown.sh
```

Use the local `nre render` CLI instead when the turn needs LiDAR,
actor edits, rolling shutter, arbitrary custom ego trajectories,
or features the thin client deliberately leaves to the NRE
container CLI.
