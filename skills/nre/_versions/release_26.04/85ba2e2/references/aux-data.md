# NuRec Auxiliary Data Tool

Before NRE can train a reconstruction it expects a small set of
per-frame auxiliary signals — semantic segmentation, (optional)
depth, (optional) DINOv2 features, (recommended) LiDAR segmentation
and visibility, ego mask, and metadata. Those signals live next to the
NCore shard as `<NAME>.aux.<kind>.zarr` (or `.zarr.itar`) files and
are produced by the `nvcr.io/nvidia/nre/nre-tools:latest` container.

## Canonical invocation

```bash
docker pull nvcr.io/nvidia/nre/nre-tools:latest

docker run --shm-size=2g -it --rm --gpus all \
  -e NGC_API_KEY=${NGC_API_KEY} \
  --volume /path/to/dataset:/workdir/dataset \
  --volume /path/to/output:/workdir/output \
  nvcr.io/nvidia/nre/nre-tools:latest \
  --dataset-path=/workdir/dataset/<NAME>.json \
  --output-dir=/workdir/output \
  --camera-id=<ID1> --camera-id=<ID2> --camera-id=<ID3> \
  --store-meta \
  --no-seg-logits \
  --lidar-seg-camvis
```

> The legacy `--shard-file-pattern <NAME>.zarr.itar` entry point still
> works and is the only option for V3-only datasets, but emits a
> deprecation warning. Prefer `--dataset-path <NAME>.json` (which
> handles both V3 and V4) for new workflows.

Notes:

- Bind the host directory containing `<NAME>.zarr.itar` and
  `<NAME>.json` to `/workdir/dataset`. For V4 datasets the meta-file
  references one or more `zarr.itar` archives or `zarr` directories
  alongside it.
- Camera IDs come from the NCore JSON. Pass `--camera-id` once per
  camera; omit the flag entirely to process all cameras. Same for
  `--lidar-id`.
- Output filenames get an `.aux.<kind>.zarr` (directory store) or
  `.aux.<kind>.zarr.itar` (indexed-tar store, default) suffix. Move
  them next to the input shards before launching NRE training.
- `--num-threads <N>` controls CPU thread count for parallelizable
  steps. `--num-threads auto` uses all available cores.
- `--store-meta` writes a per-shard `.meta.yaml` file capturing the
  exact CLI arguments and runtime metadata — keep this on for
  reproducibility.

## Sub-commands

The default invocation (no sub-command) processes the full sequence.
Two optional sub-commands restrict the time / frame range:

### `offset` — restrict by seconds

```bash
docker run ... nvcr.io/nvidia/nre/nre-tools:latest \
  --dataset-path=/workdir/dataset/<NAME>.json \
  --output-dir=/workdir/output \
  --camera-id=camera_front_wide_120fov \
  offset \
  --sequence-seek-sec=2.0 \
  --sequence-duration-sec=5.0
```

| Flag | Purpose |
|------|---------|
| `--sequence-seek-sec` | Seconds to skip at the start of each sequence. |
| `--sequence-duration-sec` | Sequence duration to keep, in seconds. |

### `sensor-frames` — restrict by main-sensor frame indices

```bash
docker run ... nvcr.io/nvidia/nre/nre-tools:latest \
  --dataset-path=/workdir/dataset/<NAME>.json \
  --output-dir=/workdir/output \
  --camera-id=camera_front_wide_120fov \
  sensor-frames \
  --main-sensor-id=camera_front_wide_120fov \
  --start-frame=100 --stop-frame=300
```

| Flag | Purpose |
|------|---------|
| `--main-sensor-id` (repeatable, ≥ 1) | One or more reference sensors. The combined min/max timestamp window is used to clip every other sensor. |
| `--start-frame` | First frame index of the main sensor(s) to export. |
| `--stop-frame` | Past-the-end frame index of the main sensor(s) to export. |

## Flag matrix (all sub-commands)

| Option | Required | Default | Purpose |
|--------|----------|---------|---------|
| `--dataset-path` | yes (or `--shard-file-pattern`) | — | NCore V3/V4 sequence meta-file (`.json`). Preferred entry point. |
| `--shard-file-pattern` | yes (deprecated alternative) | — | V3 shard glob (supports range expansion). Each shard is treated as a self-contained sequence. Emits a deprecation warning. |
| `--poses-component-group` | no | `default` | V4 component group for `poses`. |
| `--intrinsics-component-group` | no | `default` | V4 component group for `intrinsics`. |
| `--masks-component-group` | no | `default` | V4 component group for `masks`. |
| `--cuboids-component-group` | no | `default` | V4 component group for `cuboids`. |
| `--output-dir` | yes | — | Output directory inside the container. |
| `--camera-id` | no (repeatable) | all | Cameras to process. Repeat once per ID. |
| `--lidar-id` | no (repeatable) | all | LiDARs to process. Repeat once per ID. |
| `--segmentation-backend` | no | `mask2former` | `none` (skip) or `mask2former`. Mask2Former is required when `--lidar-seg-camvis` is on. |
| `--seg-logits` / `--no-seg-logits` | no | `--no-seg-logits` | Also store Mask2Former logits (for advanced training only; requires `--segmentation-backend=mask2former`). |
| `--enable-trt` / `--disable-trt` | no | `--enable-trt` | Run TRT-optimised models. Disable when running on a GPU TRT cannot target. |
| `--dinov2-backend` | no | `none` | `none`, `nv_dinov2`, `dinov2_vits14`, `dinov2_vitb14`, `dinov2_vitl14`, `dinov2_vitg14`. |
| `--dinov2-pca-dim` | no | `-1` | PCA dim for DINOv2 features; `-1` disables PCA. |
| `--dinov2-width` | no | `256` | DINOv2 feature output width. |
| `--lidar-seg-camvis` / `--no-lidar-seg-camvis` | no | `--lidar-seg-camvis` | Run LiDAR semantic seg + point-in-cameras visibility (recommended). |
| `--lidar-seg-ensemble-cuda` / `--no-lidar-seg-ensemble-cuda` | no | `--lidar-seg-ensemble-cuda` | Use the CUDA ensemble path for LiDAR seg fusion across cameras. |
| `--depth-backend` | no | `none` | `none` or `depthanythingv2`. |
| `--relative-depth` | no | metric | Predict relative depth instead of metric. |
| `--max-depth-m` | no | `80.0` (outdoor) | Maximum metric depth (no effect with `--relative-depth`). |
| `--depth-input-resolution` | no | `1036` | Depth-network input resolution (px). |
| `--store-depth-as-png` | no | float16 | Store quantised PNG instead of raw float16. |
| `--ego-mask` / `--no-ego-mask` | no | `--ego-mask` | Automatic ego-mask estimation (SAM2-based, prompt-free). |
| `--ego-mask-samples-per-second` | no | `0.2` | Sample rate for ego-mask estimation. Range `0.0002 ≤ x ≤ 30.0`. |
| `--ego-mask-aggregation-method` | no | `majority` | Aggregation method across samples (currently only `majority`). |
| `--ego-mask-camera-id` (repeatable) | no | all cameras | Restrict ego-mask estimation to these camera IDs. |
| `--zarr-store-type` | no | `itar` | `itar` (indexed tar archive — default, production) or `directory` (plain Zarr directory — easier introspection). |
| `--open-consolidated` / `--no-open-consolidated` | no | `--open-consolidated` | Open shards with consolidated metadata. Disable to debug missing-metadata corner cases. |
| `--num-threads` | no | `8` (literal `auto` ⇒ all CPUs) | Number of CPU threads for the parallelizable steps. |
| `--parallel-mode` / `--no-parallel-mode` | no | `--no-parallel-mode` | Run multiple LiDAR / camera segmentation tasks concurrently per GPU. |
| `--workers-per-gpu` | no | `6` | Concurrent workers per GPU when `--parallel-mode` is on. |
| `--debug` | no | off | Enable DEBUG-level logging. |
| `--visualize` | no | off | Output visualisations alongside the Zarr stores. |
| `--store-meta` | no | off | Emit a `<NAME>.meta.yaml` file per shard with CLI args + runtime logging. |
| `--help` | no | — | Full flag listing. |

`--num-threads` accepts the literal string `auto` — equivalent to
"use every CPU the container sees".

## Auxiliary-data types

### Semantic segmentation (conditionally required)

- Backend: Mask2Former with a DINOv2 backbone.
- Outputs:
  - Semantic segmentation masks as PNGs (`aux.sseg`).
  - (Optional) logits when `--seg-logits` is set (`aux.sseg-logits`).
- Default: enabled. Disable via `--segmentation-backend=none`.
- Required when LiDAR segmentation is enabled.

### Depth estimation (optional)

- Backend: `depthanythingv2`.
- Modes: relative (`[0, 1]`) or metric (default 80 m max for outdoor).
- Storage: float16 by default, quantised PNG via `--store-depth-as-png`.
- Default: disabled (`--depth-backend=none`).

### DINOv2 feature extraction (optional)

- Backbones: ViT-S/B/L/G (14×14 patches) plus the bundled
  `nv_dinov2`. Selectable via `--dinov2-backend`.
- Optional PCA reduction via `--dinov2-pca-dim` (`-1` disables PCA).
- Output width controlled by `--dinov2-width` (default 256).
- Default: disabled.

### LiDAR segmentation & visibility (optional but recommended)

- Projects camera semantic segmentation onto LiDAR point clouds.
- Outputs per-point labels (`aux.lidar-sseg`) and point-in-camera
  visibility (`aux.lidar-camvis`).
- Default: `--lidar-seg-camvis` (enabled). Requires semantic
  segmentation to be either generated in the same run or
  pre-existing.
- The CUDA ensemble path (`--lidar-seg-ensemble-cuda`, on by default)
  fuses per-camera labels on-GPU; disable for repro on machines
  without a CUDA-capable GPU mid-pipeline.

### Ego mask (optional, on by default)

- Backend: prompt-free SAM2 estimator, with multi-sample aggregation.
- Per-camera RGBA mask (`aux.egomask`). Tune sampling with
  `--ego-mask-samples-per-second`; restrict to a subset of cameras
  with `--ego-mask-camera-id`.

### Metadata (recommended)

- Camera calibration and sensor metadata.
- The CLI arguments used (when `--store-meta` is set).
- Model versions and configuration.
- Runtime / workflow logging.

## Layout after a successful run

The default `--zarr-store-type=itar` writes one indexed-tar archive
per signal kind:

```text
/path/to/output/
├── <NAME>.aux.sseg.zarr.itar              # Mask2Former semantic seg
├── <NAME>.aux.sseg-logits.zarr.itar       # Mask2Former logits (when --seg-logits)
├── <NAME>.aux.lidar-sseg.zarr.itar        # LiDAR semantic seg (when --lidar-seg-camvis)
├── <NAME>.aux.lidar-camvis.zarr.itar      # LiDAR point-in-camera visibility
├── <NAME>.aux.depth.zarr.itar             # Depth (when --depth-backend != none)
├── <NAME>.aux.dinov2.zarr.itar            # DINOv2 features (when --dinov2-backend != none)
├── <NAME>.aux.egomask.zarr.itar           # Ego mask (when --ego-mask)
└── <NAME>.meta.yaml                       # When --store-meta is set
```

> Internally NRE writes per-signal Zarr base groups under stable
> short names — `sseg`, `iseg`, `oflow`, `sflow`, `nrml`, `depth`,
> `lidar-sseg`, `lidar-camvis`, `sseg-logits`, `dinov2`, `egomask`.
> These are the suffixes you'll see on disk; `iseg`, `oflow`,
> `sflow`, `nrml` are reserved for instance / optical-flow / scene-flow
> / surface-normal extensions and are not produced by the default
> pipeline today.

If you set `--zarr-store-type=directory`, each `.zarr.itar` becomes a
plain `.zarr` directory at the same path (useful for introspection /
external tooling).

Move (or symlink) these next to the original `<NAME>.zarr.itar` and
`<NAME>.json` so the NRE training app finds them automatically. NRE
opens them via `dataset.aux_data=true` (the default).

## Recipes

### Minimal aux for a quick training pass (camera only)

```bash
docker run --shm-size=2g -it --rm --gpus all \
  -e NGC_API_KEY=${NGC_API_KEY} \
  --volume /path/to/dataset:/workdir/dataset \
  --volume /path/to/output:/workdir/output \
  nvcr.io/nvidia/nre/nre-tools:latest \
  --dataset-path=/workdir/dataset/<NAME>.json \
  --output-dir=/workdir/output \
  --no-lidar-seg-camvis \
  --no-ego-mask \
  --store-meta
```

### Full multi-modal aux (Mask2Former + LiDAR seg + ego mask + depth)

```bash
docker run --shm-size=2g -it --rm --gpus all \
  -e NGC_API_KEY=${NGC_API_KEY} \
  --volume /path/to/dataset:/workdir/dataset \
  --volume /path/to/output:/workdir/output \
  nvcr.io/nvidia/nre/nre-tools:latest \
  --dataset-path=/workdir/dataset/<NAME>.json \
  --output-dir=/workdir/output \
  --camera-id=camera_front_wide_120fov \
  --camera-id=camera_cross_left_120fov \
  --camera-id=camera_cross_right_120fov \
  --depth-backend=depthanythingv2 \
  --max-depth-m=80 \
  --lidar-seg-camvis \
  --ego-mask \
  --store-meta \
  --num-threads=auto
```

### Faster end-to-end on multi-GPU hosts

```bash
... --parallel-mode --workers-per-gpu=4
```

### Add DINOv2 features for advanced training

```bash
... --dinov2-backend=dinov2_vitl14 --dinov2-pca-dim=64 --dinov2-width=384
```

### Time-restricted aux (skip first 2s, keep next 5s of every clip)

```bash
... offset --sequence-seek-sec=2.0 --sequence-duration-sec=5.0
```

### Frame-restricted aux (export frames 100..300 of the front-wide camera)

```bash
... sensor-frames --main-sensor-id=camera_front_wide_120fov --start-frame=100 --stop-frame=300
```

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `--shard-file-pattern is deprecated` warning | Using V3 entry point. | Switch to `--dataset-path <NAME>.json` (works for both V3 and V4). |
| `segmentation logits requires 'mask2former' segmentation backend` | `--seg-logits` set but `--segmentation-backend=none`. | Either drop `--seg-logits` or re-enable Mask2Former. |
| `--lidar-seg-camvis` fails with missing seg | LiDAR seg projects from camera seg; you turned off Mask2Former without supplying pre-existing seg. | Run with `--segmentation-backend=mask2former` (default), or supply external `aux.sseg.*` shards co-located with the input. |
| TRT-related crash on a less common GPU | TRT-optimized model paths default to ON. | Pass `--disable-trt`. |
| Slow per-shard processing | Default `--num-threads=8`. | Pass `--num-threads=auto`, and consider `--parallel-mode --workers-per-gpu=N` on multi-GPU hosts. |
| Cannot inspect the Zarr archives | Default store is `itar` (single tar with indexed offsets). | Re-run with `--zarr-store-type=directory` for plain `.zarr` directories. |
| Ego mask covers too few frames | Default `--ego-mask-samples-per-second=0.2` (one sample every 5 s). | Increase up to `30.0` for high-frequency clips, or restrict via `--ego-mask-camera-id`. |
| Output files don't appear next to the input shards | Files land in `--output-dir`. | After the run, move/symlink them next to `<NAME>.zarr.itar` so NRE training picks them up via `dataset.aux_data=true`. |

## Asset Harvester sub-command on the same container

The same `nre-tools` image also exposes the
`asset-harvester` pipeline as a
sub-command, which downloads its own model checkpoints from NGC on
first run and caches them under `/cache`:

```bash
docker run -it --rm --gpus=all \
  -e NGC_API_KEY=${NGC_API_KEY} \
  -v /path/to/output:/output \
  -v /path/to/data:/data \
  -v ~/.cache/nre:/cache \
  nvcr.io/nvidia/nre/nre-tools:latest \
  asset-harvester \
  --component-store="/data/component-store.zarr.itar" \
  --output-dir="/output" \
  --track-ids="track_id1,track_id2" \
  --cache-dir="/cache" \
  ncore_parser.camera_ids=["camera_front_wide_120fov"]
```

See `references/asset-editing.md` for how to take the Asset-Harvester
output and merge it into a NuRec USDZ.
