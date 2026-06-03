# NRE — CLI cookbook

Most-used `docker run` invocations. The single-line summary lives in
`SKILL.md`. See `references/cli-reference.md` for the full command
inventory, the per-sub-command `--help` examples, and the authoritative
flag tables.

## Train + validate (Waymo Open Dataset, dynamic 3DGUT)

> **Recipe scope.** `configs/apps/AV/Waymo/3dgut_dynamic.yaml` is for
> the **Waymo Open Dataset** only — it bakes in the Waymo sensor rig.
> For the NVIDIA Physical AI Autonomous Vehicles (PAI) dataset, use
> the Hyperion-8.1 recipe in the next section instead.

```bash
docker run --shm-size=64g -it --rm --gpus all \
  -u "$(id -u):$(id -g)" \
  -e NGC_API_KEY=${NGC_API_KEY} \
  --volume /path/to/dataset:/workdir/dataset \
  --volume /path/to/output:/workdir/output \
  nvcr.io/nvidia/nre/nre:latest \
  --config-name=configs/apps/AV/Waymo/3dgut_dynamic.yaml \
  mode=trainval \
  dataset.path=/workdir/dataset/<NAME>.json \
  out_dir=/workdir/output \
  logger=tensorboard \
  checkpoint.artifact.enabled=true \
  checkpoint.artifact.rig_trajectories.enabled=true \
  checkpoint.artifact.sequence_tracks.enabled=true
```

## Train + validate (Physical AI Autonomous Vehicles — Hyperion-8.1)

For the gated NVIDIA dataset
[`nvidia/PhysicalAI-Autonomous-Vehicles`](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles),
use `/apps/prod/Hyperion-8.1/car2sim_6cam.yaml` via the small overlay
shipped at [`configs/pai.yaml`](configs/pai.yaml) — **not** the Waymo
recipe above.

The overlay extends `car2sim_6cam.yaml` with the PAI six-camera
validation set, `lidar_top_360fov`, and lidar-`intensity`
supervision. Mount it into the container's config tree as
`external_overrides.yaml`, then drive training with
`--config-name=external_overrides`.

```bash
# Resolve the in-container config dir (e.g. /app/run.runfiles/_main/configs).
NRE_CONFIG_DIR=/app/run.runfiles/_main/configs

docker run --shm-size=64g -it --rm --gpus all \
  -u "$(id -u):$(id -g)" \
  -e NGC_API_KEY=${NGC_API_KEY} \
  --volume /path/to/dataset:/workdir/dataset \
  --volume /path/to/output:/workdir/output \
  --volume $(pwd)/configs/pai.yaml:${NRE_CONFIG_DIR}/external_overrides.yaml:ro \
  nvcr.io/nvidia/nre/nre:latest \
  --config-name=external_overrides \
  mode=trainval \
  dataset.path=/workdir/dataset/<NAME>.json \
  'dataset.camera_ids=[camera_front_wide_120fov,camera_front_tele_30fov,camera_cross_left_120fov,camera_cross_right_120fov,camera_rear_left_70fov,camera_rear_right_70fov]' \
  'dataset.lidar_ids=[lidar_top_360fov]' \
  out_dir=/workdir/output \
  logger=tensorboard \
  checkpoint.artifact.enabled=true \
  checkpoint.artifact.rig_trajectories.enabled=true \
  checkpoint.artifact.sequence_tracks.enabled=true
```

For the OSMO equivalent (which writes the overlay inline as the
`prepare-nre-config` task), see
[`example-workflows/osmo/pai-nurec.yaml`](example-workflows/osmo/pai-nurec.yaml);
for the full host-side PAI walkthrough see
[`example-workflows/bash/nurec_workflow_pai.md`](example-workflows/bash/nurec_workflow_pai.md).

## Re-validate with a 3 m lateral shift

```bash
docker run --shm-size=64g -it --rm --gpus all \
  -u "$(id -u):$(id -g)" \
  -e NGC_API_KEY=${NGC_API_KEY} \
  --volume /path/to/dataset:/workdir/dataset \
  --volume /path/to/output:/workdir/output \
  nvcr.io/nvidia/nre/nre:latest \
  --config-name=/workdir/output/<RUN-ID>/config/parsed.yaml \
  mode=val \
  resume=/workdir/output/<RUN-ID>/checkpoints/last.ckpt \
  out_dir=/workdir/output \
  dataset.val_sensor_transl_delta_m="[0,3,0]"
```

## Render frames locally (no gRPC) at quarter resolution

```bash
docker run --shm-size=64g -it --rm --gpus all \
  -u "$(id -u):$(id -g)" \
  --volume /path/to/output:/workdir/output \
  nvcr.io/nvidia/nre/nre:latest \
  render \
  --artifact-path /workdir/output/<RUN-ID>/usd-out/last.usdz \
  --output-dir   /workdir/output/<RUN-ID>/renders \
  --camera-id camera_front_wide_120fov \
  --image-scale 0.25 \
  --image-format png \
  --frame-naming frame-end-timestamp \
  --replicate-training-views \
  --export-video --video-fps 30 --video-crf 20
```

## Render at native resolution along original sensor poses

`--replicate-training-views` (default ON) locks per-frame poses,
intrinsics, and ISP to the training rig, and overrides
`--calib-source` to `training-sensor-poses-calib` (the optimised
sensor-to-world poses) for the most faithful playback. Repeat
`--camera-id` to render multiple sensors. Budget on a 48 GB
Ampere/Ada GPU: ~270 ms per 1920x1080 frame (~5x slower wall
time vs the 0.25 quarter-res preset above).

```bash
docker run --shm-size=64g -it --rm --gpus all \
  -u "$(id -u):$(id -g)" \
  -e NGC_API_KEY=${NGC_API_KEY} \
  --volume /path/to/scene:/workdir/scene:ro \
  --volume /path/to/output:/workdir/output \
  nvcr.io/nvidia/nre/nre:latest \
  render \
  --artifact-path /workdir/scene/<NAME>.usdz \
  --output-dir   /workdir/output/<RUN-ID>/renders \
  --camera-id camera_front_wide_120fov \
  --image-scale 1.0 \
  --image-format jpg \
  --frame-naming contiguous-output-index \
  --replicate-training-views \
  --renderer nrend

# Stitch into visually-lossless H.264. Use this fallback whenever
# the NRE container predates release 26.04 (where the in-container
# --export-video / --video-fps / --video-crf flags landed); on
# 26.04+ append those flags to the render command above and skip
# ffmpeg entirely.
ffmpeg -framerate 30 \
  -i /path/to/output/<RUN-ID>/renders/camera_front_wide_120fov/%06d.jpg \
  -c:v libx264 -crf 18 -preset slow -pix_fmt yuv420p \
  /path/to/output/<RUN-ID>/renders/camera_front_wide_120fov.mp4
```

> **Quality knobs** for `render`:
> `--image-scale 1.0` (native, never upscale) · `--image-format
> jpg` (default, ~10x lighter than `png`; use `png` only when you
> need bit-exact pixels — GT comparison via
> `eval-rendering-metrics`, perceptual-loss training, or downstream
> perception ingest) · `--replicate-training-views` (keep ON for
> fidelity; flip OFF only when applying rig offsets / actor edits /
> custom rig trajectory) · `--calib-source
> training-sensor-poses-calib` (optimised per-sensor poses;
> auto-selected by `--replicate-training-views`; highest fidelity)
> · `--renderer nrend` (fast C++/CUDA path; same pixels, lower wall
> time) · `--frame-naming contiguous-output-index` (drop-in for
> `ffmpeg %06d`) / `frame-end-timestamp` (deterministic global
> timestamps for benchmarking against GT). Post-encode target
> `crf 18` (visually lossless) or `crf 14` (perceptually lossless).

## Launch sensorsim gRPC server with editing enabled

```bash
docker run --shm-size=64g -it --rm --gpus all \
  -u "$(id -u):$(id -g)" \
  --net=host --privileged \
  -e NGC_API_KEY=${NGC_API_KEY} \
  --volume /path/to/output:/workdir/output \
  nvcr.io/nvidia/nre/nre:latest \
  serve-grpc \
  --artifact-glob /workdir/output/<RUN-ID>/usd-out/last.usdz \
  --renderer nrend \
  --enable-difix \
  --enable-editing-actors \
  --test-scenes-are-valid
```

## Render a LiDAR sweep from a running `serve-grpc`

```bash
docker run --shm-size=64g -it --rm --gpus all \
  -u "$(id -u):$(id -g)" \
  --network host \
  --volume /path/to/output:/workdir/output \
  nvcr.io/nvidia/nre/nre:latest \
  render-grpc \
  --artifact-path /workdir/output/<RUN-ID>/usd-out/last.usdz \
  --output-dir /workdir/output/<RUN-ID>/lidar-renders \
  --lidar --lidar-id lidar_top --lidar-format ply
```
