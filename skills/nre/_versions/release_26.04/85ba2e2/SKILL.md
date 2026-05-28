---
name: nre
description: >-
  Use when the user wants to run NVIDIA Omniverse NuRec / Neural
  Reconstruction Engine (NRE) on Linux x86_64 + NVIDIA GPU: train a
  3DGUT Gaussian reconstruction from an NCore camera+LiDAR clip via a
  Hydra recipe like 3dgut_dynamic.yaml, generate aux data via
  nre-tools, render frames locally with `nre render`, run a warm
  `serve-grpc` server plus host-side thin Python client, export PLY /
  depth / mesh / ego mask / tracks, repackage Asset Harvester output
  into a USDZ, or evaluate rendering metrics. Use ONLY through the
  public NGC containers nvcr.io/nvidia/nre/nre and
  nvcr.io/nvidia/nre/nre-tools with an NGC_API_KEY; never clone the
  source. Do NOT use for per-object 3D asset capture (use
  `asset-harvester`) or sensor-data conversion to NCore (use `ncore`).
  Trigger keywords: nre, NRE, neural reconstruction engine, nurec,
  NuRec, Omniverse NuRec, 3DGUT, 3DGRT, 3dgut_dynamic.yaml,
  export-gaussian-plys, export-custom-rig-trajectory, render-grpc,
  serve-grpc, nre thin client, warm nre server, nre-tools, USDZ.
version: "0.2.0"
tools:
  - Shell
  - Read
  - Write
license: CC-BY-4.0 AND Apache-2.0
compatibility: >-
  Linux x86_64. 1+ NVIDIA GPU with CUDA 12.8 (Ampere A100/A10/A40/RTX
  A6000, Ada L20/L40/L40S, Hopper H100/H20, or Blackwell RTX Pro 6000D),
  >= 24 GB VRAM (48+ GB recommended). Driver R570+ recommended (R580+ on
  Blackwell, R535+ minimum for the Fixer model variant). Host needs
  Docker 23.0.1+ and NVIDIA Container Toolkit 1.13.5+, plus an NGC
  account (NGC_API_KEY env var) to pull nvcr.io/nvidia/nre images.
  scripts/validate_setup.py performs no network calls but reads the
  NGC_API_KEY env var.
dependencies:
  - bash
  - docker
  - python3
metadata:
  author: NVIDIA Omniverse
  tags:
    - nurec
    - autonomous-vehicles
    - neural-reconstruction
    - rendering
    - container
  product_page: https://www.nvidia.com/en-us/omniverse/nurec/
  ngc_container: nvcr.io/nvidia/nre/nre:latest
  ngc_tools_container: nvcr.io/nvidia/nre/nre-tools:latest
  ngc_fixer_model: https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nre/models/nurec-fixer
  hf_datasets: https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles-NuRec
  hf_fixer_model: https://huggingface.co/nvidia/Difix3D
  release_date: "2026-04-30"
  time-estimate: "2h"
---

# NRE — NVIDIA Omniverse NuRec (Neural Reconstruction Engine)

## Purpose

Drive the public NVIDIA Omniverse NuRec / Neural Reconstruction
Engine containers (`nvcr.io/nvidia/nre/nre`,
`nvcr.io/nvidia/nre/nre-tools`) to train a 3DGUT/3DGRT Gaussian
reconstruction from an NCore V4 camera+LiDAR clip, render novel
views (locally or via gRPC), generate aux data, export
PLY/depth/mesh/ego-mask/tracks, package Asset Harvester output into a
USDZ, and evaluate rendering metrics.

This skill also carries the host-side toolkit around the NRE CLI:
NGC credential resolution, cached-image/version notes, local render
recipes, MP4 encoding, warm `serve-grpc` boot/teardown scripts, a
thin Python gRPC client for repeated RGB renders, bundled rig JSONs,
pre-baked custom-rig trajectories, and bash / Hydra / OSMO workflow
templates.

**Use this skill when:** the user has an NCore V4 clip (or a USDZ +
NRE artifact pair) on a Linux x86_64 host with an NVIDIA GPU and an
NGC API key, and wants to train, render, or export with NRE.

**Do NOT use this skill when:**

- The user wants per-object 3D asset extraction from sparse views
  (use `asset-harvester`).
- The user still needs to convert raw sensor data into NCore V4
  (use the `ncore` skill first).
- The user only needs to clean up already-rendered frames
  (use `nurec-fixer`).

## Table of Contents

- [When to Use](#when-to-use) · [When NOT to Use](#when-not-to-use) · [Inputs](#inputs) · [Instructions](#instructions) · [Backend Selection](#backend-selection) · [Output Format](#output-format)
- [Scripts](#scripts) · [References](#references) · [Prerequisites](#prerequisites) · [Verifying secrets safely](#verifying-secrets-safely)
- [Limitations](#limitations) · [Troubleshooting](#troubleshooting)
- [Overview](#overview) · [Install — Docker / NGC container](#install--docker--ngc-container) · [Warm-Server Thin Client Quick Start](#warm-server-thin-client-quick-start) · [CLI cookbook](#cli-cookbook)
- [End-to-end workflows](#end-to-end-workflows) — A (NCore → reconstruction → render), B (Physical AI dataset), C (Asset-Harvester actors), D (multi-GPU), E (resume), F (upgrade-artifact), G (LiDAR sweeps), H (viewer), I (eval)
- [Teardown](#teardown)

## When to Use

- User wants to **train a neural reconstruction** of a multi-camera +
  LiDAR AV driving clip in NCore format and emit a renderable `USDZ`
  scene with 3DGUT Gaussians (or 3DGRT for ray-traced Gaussians).
- User wants to **generate NuRec auxiliary data** (semantic
  segmentation, depth, LiDAR-segmentation visibility, ego mask,
  DINOv2 features) for an NCore dataset using the `nre-tools`
  container.
- User wants to **render frames locally** from a trained USDZ along the
  training rig (or a custom rig + offsets) without standing up a gRPC
  server — the in-container `render` sub-command.
- User wants to **render novel views** of an existing NuRec USDZ
  programmatically via the sensorsim gRPC API (e.g. plugging NuRec
  into CARLA, AlpaSim, Isaac Sim, or a custom simulator), with
  optional Difix (Fixer) artifact-removal post-processing.
- User wants to **render LiDAR sweeps** of a USDZ via
  `render-grpc --lidar` (writes raw `.bin` or PLY-with-intensity).
- User wants to **export artifacts** from a trained checkpoint — PLY
  Gaussians (3DGS or 3DGRT format), ego masks, depth, Poisson mesh,
  ground mesh, point clouds, cuboid-track JSON, per-sensor
  `NCore tracks`, custom rig trajectories.
- User wants to **insert / remove / replace 3D actors** in a USDZ by
  combining Asset-Harvester output (`.ply` + `metadata.yaml`) with
  `export-external-assets` + `render-grpc --edit-assets`.
- User wants to **render the NVIDIA Physical AI dataset** (HuggingFace
  `nvidia/PhysicalAI-Autonomous-Vehicles-NuRec`) into custom
  trajectories.
- User wants to **upgrade an old USDZ artifact** so newer NRE releases
  load it without paying the in-memory-upgrade tax every time.
- User wants to **inspect a USDZ / config** (`export-parsed-config`,
  `export-artifact-structure`, `gaussian-statistics`).
- User wants to **evaluate rendering quality** (`eval-rendering-metrics`,
  `compute-metrics`, `eval-ground-mesh`).
- User wants to **browse a USDZ or PLY in the browser** via the
  in-container viewer.

### When NOT to Use

- For ingesting raw sensor data into NCore (camera/LiDAR conversion,
  rig calibration) — use the `ncore` skill first;
  NRE consumes already-NCore-formatted shards.
- For harvesting per-object 3D Gaussian assets from sparse views — use
  `asset-harvester`; NRE only *consumes*
  Asset Harvester outputs via `export-external-assets`.
- For pure post-processing of rendered frames into harmonised /
  temporally-stable images using the standalone Cosmos-based Fixer —
  use `nurec-fixer`. NRE's `--enable-difix`
  flag invokes a built-in Difix variant (Cosmos by default since
  25.09; the Stable-Diffusion variant is still available via
  `difix=sd_difix`).

## Inputs

- **dataset_dir** — host directory holding the NCore shards
  (`<NAME>.zarr.itar`, `<NAME>.json`, and any pre-generated
  `<NAME>.aux.*.zarr` auxiliary shards). (source: user prompt;
  required.)
- **dataset_name** — basename of the NCore dataset (the part before
  `.zarr.itar`). (source: user prompt; required.)
- **output_dir** — host directory NRE will fill with checkpoints,
  parsed config, metrics, depth / opacity / segmentation videos, and
  USDZ artifacts. (source: user prompt; required.)
- **camera_ids / lidar_ids** — sensor IDs from the NCore JSON to
  include. (source: user prompt; optional; default: all sensors per
  recipe.)
- **config_name** — Hydra config path the NRE container resolves
  internally, e.g. `configs/apps/AV/Waymo/3dgut_dynamic.yaml`. (source:
  user prompt; optional; default: `configs/apps/AV/Waymo/3dgut_dynamic.yaml`.)
- **mode** — `train`, `val`, or `trainval`. (source: user prompt;
  optional; default: `trainval`.)
- **NGC_API_KEY** — NGC token used to authenticate `docker pull` and
  in-container artifact downloads. (source: environment variable;
  required. Generate at
  https://org.ngc.nvidia.com/setup/api-keys.)

## Instructions

1. **Validate prerequisites.** Run `scripts/validate_setup.py` to
   confirm Docker, NVIDIA Container Toolkit, GPU/driver, and
   `NGC_API_KEY` are in place. Resolve any FAIL lines before pulling
   the container.
2. **Authenticate Docker to NGC.** `docker login nvcr.io` with
   `Username: $oauthtoken` and password `${NGC_API_KEY}`.
3. **Confirm input layout.** The dataset directory must contain
   `<NAME>.zarr.itar`, `<NAME>.json`, and any `<NAME>.aux.*.zarr`
   shards you produced earlier. If the NCore data is fresh, generate
   auxiliary data first — see `references/aux-data.md`.
4. **Pull the public container(s).**
   `docker pull nvcr.io/nvidia/nre/nre:latest` and (if you'll be
   generating aux data or harvesting assets)
   `docker pull nvcr.io/nvidia/nre/nre-tools:latest`. Some NGC
   releases publish the same images under the
   `nvcr.io/nvidia/nre/nre-ga` and
   `nvcr.io/nvidia/nre/nre-tools-ga` names to mark General
   Availability builds; substitute those names if the user has been
   directed to the GA channel.
5. **Train / validate the reconstruction.** Launch
   `nvcr.io/nvidia/nre/nre:latest` with the chosen Hydra
   `--config-name`, `mode`, `dataset.path`, and `out_dir` (see
   "End-to-end workflow A" below). For multi-GPU systems append
   `trainer.world_size=<N> trainer.num_nodes=<M>`. Set
   `checkpoint.artifact.enabled=true` if you intend to render or
   serve the result.
6. **Export downstream artifacts.** Use the export sub-commands
   (`export-gaussian-plys`, `export-mesh`, `export-ground-mesh`,
   `export-ego-mask`, `export-depth`, `export-sequence-tracks`,
   `export-ncore-tracks`, …) — see `references/cli-reference.md`.
7. **Render novel views.** Choose the backend by latency and feature
   needs:
   - **Local CLI** — `nre render --artifact-path <usdz> --output-dir
     <dir>` writes frames on disk along the training trajectory, with
     optional `--rig-translation-offset` / `--rig-rotation-offset` or a
     `--custom-rig-trajectory`. No gRPC server required; see
     `references/local-render.md`.
   - **Warm RGB service** — boot `serve-grpc` once with
     `scripts/session_warm_server.sh`, extract protobuf stubs, and use
     `references/NRE_RenderClient/scripts/thin_client.py` for repeated
     single-camera or `batch_render_rgb` calls.
   - **Remote CLI / simulator integration** — `serve-grpc` +
     `render-grpc` (or your own client via `nre.grpc.protos`). Use this
     for LiDAR rendering, simulator loops, Difix, or `--edit-assets`.
     See `references/grpc-api.md` and `references/physical-ai-render.md`.
8. **Edit actors (optional).** To swap, remove, or insert assets, run
   `export-external-assets` to repackage Asset-Harvester output into
   a new USDZ, then pass the produced `edit-assets.json` to
   `render-grpc --edit-assets` (with `serve-grpc
   --enable-editing-actors`). See `references/asset-editing.md`.
9. **Validate the result.** Confirm the USDZ in
   `<output_dir>/<RUN-ID>/usd-out/last.usdz` opens, `metrics.yaml`
   reports a reasonable `test/psnr`, and the generated MP4s render.
   For more thorough metrics use `eval-rendering-metrics` against the
   ground-truth folder layout produced by `export-ncore-benchmark-gt`.
   Tear down any gRPC server (`Ctrl-C` or `docker rm -f`).

## Backend Selection

Pick the smallest backend that exposes the requested feature:

- **Local Docker, single command.** Use `nre render`, `render-grpc`, or
  an export sub-command directly. This is simplest for one-off renders,
  LiDAR sweeps, actor edits, rolling shutter, in-container video
  export, or exact `--replicate-training-views` behavior. Read
  `references/local-render.md`, `references/nre-image-notes.md`, and
  `references/mp4-encoding.md` before adapting the command.
- **Local Docker, warm `serve-grpc` + thin host client.** Use this for
  render-heavy RGB sessions where repeated Docker/Python/CUDA
  cold-start dominates latency, or where multiple cameras should be
  rendered through one `batch_render_rgb` RPC. Read
  `references/NRE_RenderClient/SKILL.md` and use
  `scripts/session_warm_server.sh` / `scripts/session_teardown.sh`.
- **OSMO / cluster workflows.** Use the templates under
  `references/example-workflows/osmo/` for multi-clip fan-out,
  isolation from the local machine, or training jobs that should not
  run on the user's workstation. Follow `references/ngc-and-registry.md`
  for registry credentials and `references/long-running-tasks.md` for
  polling discipline.

For any NRE task expected to run 5 minutes or longer, follow
`references/long-running-tasks.md`: delegate the job, run it in the
background, and report compact status at least every 5 minutes.

## Output Format

Structured deliverables placed under `${output_dir}/${RUN_ID}/` by the
NRE container (no JSON state file required from the agent):

- `config/parsed.yaml` — Hydra-resolved training config (re-pass it
  to validation / export sub-commands).
- `checkpoints/last.ckpt` (plus periodic snapshots) — model weights.
- `val/metrics.yaml` — per-frame PSNR / SSIM / LPIPS under `test/*`.
- `val/*.mp4` and `val/<frame>/*.png` — depth, opacity, segmentation,
  RGB visualisations.
- `usd-out/last.usdz` — USDZ package containing the trained
  reconstruction, `data_info.json`, `rig_trajectories.json`,
  `sequence_tracks.json`, `parsed_config.yaml`, `checkpoint.ckpt`,
  optional `mesh.ply`, and `map.xodr`. Render this with
  `nre render`, `serve-grpc` + `render-grpc`, the in-container
  `viewer`, or hand it to a downstream simulator (CARLA, AlpaSim,
  Isaac Sim).
- `*.ply` / `ego_mask/*` / `depth/*` / `sequence_tracks.json` /
  `ncore_tracks.json` / `mesh.ply` / `ground_mesh.ply` — produced by
  the matching export sub-command.

## Scripts

| Script | Purpose | Usage |
|--------|---------|-------|
| `scripts/validate_setup.py` | Verify Docker, NVIDIA Container Toolkit, GPU/driver R570+ (R535+ minimum), NGC login, and `NGC_API_KEY` env var. No network calls. | Execute: `python scripts/validate_setup.py [--strict]` |
| `scripts/session_warm_server.sh` | Idempotently boot a session-scoped `nre serve-grpc` container for the thin Python client. Discovers a cached 26.04+ renderer image, mounts the USDZ root, and waits for readiness. | Execute: `NRE_GRPC_USDZ_HOST_DIR=/path/to/usdz/root bash scripts/session_warm_server.sh` |
| `scripts/session_teardown.sh` | Stop and remove the warm `serve-grpc` container and clear its state file without racing the next boot. | Execute: `bash scripts/session_teardown.sh` |

## References

Read these on demand; keep `SKILL.md` as the routing layer and put
flag-level or workflow-specific detail in the companion files.

- `references/cli-reference.md` — full sub-command surface of the NRE
  container (training, validation, `render`, `serve-grpc`,
  `render-grpc`, `render-novel-trajectory`, every `export-*`,
  `upgrade-config` / `upgrade-artifact`, `gaussian-statistics`,
  `eval-rendering-metrics`, `compute-metrics`, `viewer`, `ply_viewer`,
  `profile-dataloader`, `run-script`, the `nre-tools` aux-data + AH
  entry points).
- `references/configuration.md` — Hydra configuration recipe map for
  Waymo / NV / PandaSet / Tesla / Alpasim, plus the override matrix
  for `dataset`, `trainer`, `model`, `loss`, `difix`, `system`,
  `checkpoint`, `viewer`, `profiling`, `scopedtimer`, the multi-GPU
  / SLURM cookbook, novel-view synthesis, resume, and environment
  variables.
- `references/aux-data.md` — `nre-tools` auxiliary-data CLI options
  (Mask2Former, DepthAnythingV2, DINOv2, LiDAR seg/visibility, ego
  mask).
- `references/local-render.md` — host-side `docker run ... render`
  recipes for rig offsets and `export-custom-rig-trajectory` bakes,
  including measured wall times and pre-baked trajectory-cache rules.
- `references/NRE_RenderClient/SKILL.md` — warm-server thin Python
  gRPC client, `setup_protos.sh`, `thin_client.py`, `batch_render_rgb`,
  runtime scene loading, rig-file overrides, and limitations.
- `references/grpc-api.md` — sensorsim gRPC server flags + Python
  client cookbook (`nre.grpc.protos.sensorsim_pb2_grpc`,
  `RGBRenderRequest`, `LidarRenderRequest`, `--enable-difix` flow).
- `references/nre-image-notes.md` — cached-image discovery, 26.04+
  vs 26.03 vs pre-26.03 flag selection, `--renderer default`, JPEG
  defaults, rig-rotation convention, and common CLI gotchas.
- `references/ngc-and-registry.md` — NGC API key resolution,
  `docker login nvcr.io`, and OSMO registry credential pointers.
- `references/mp4-encoding.md` — host-side ffmpeg recipe, in-container
  `--export-video` caveats, and required `gt_camera_<id>.mp4` /
  `camera_<id>.mp4` filenames for comparison viewers.
- `references/asset-editing.md` — `export-external-assets` +
  `edit-assets.json` schema + `render-grpc --edit-assets` to remove,
  replace, and insert actors.
- `references/physical-ai-render.md` — end-to-end recipe for rendering
  the HuggingFace `nvidia/PhysicalAI-Autonomous-Vehicles-NuRec`
  dataset, including NuRec ↔ ECEF ↔ ENU coordinate transforms.
- `references/example-workflows/` — bash, Hydra, and OSMO templates for
  PAI/NCore/NuRec workflows, USDZ export, `serve-grpc`, and local or
  cluster rendering. Treat them as templates and generate fresh files
  for user-specific paths.
- `references/rig-json/` — bundled `rig.json` and
  `augmented_rig.json` for fallback/demo renders when the caller has
  not supplied a rig directory.
- `references/custom-rig-trajectories/` — pre-baked
  `export-custom-rig-trajectory` outputs keyed by clip UUID prefix for
  the bundled augmented rig and canonical PAI demo clips.
- `references/long-running-tasks.md` — subagent/background-job and
  5-minute status reporting convention for training, OSMO jobs,
  multi-clip renders, large downloads, and other long tasks.
- `references/nurec-skill-catalog.md` — routing table for sibling
  NuRec-stack skills in the canonical `NVIDIA/nurec-skills` repo.
- `references/teardown.md` — full inventory of images, caches, and
  outputs the workflow leaves behind, ownership-recovery commands,
  and post-teardown verification.
- Public product page: https://www.nvidia.com/en-us/omniverse/nurec/
- HuggingFace dataset: https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles-NuRec
- HuggingFace Fixer model: https://huggingface.co/nvidia/Difix3D
- NGC Fixer model card: https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nre/models/nurec-fixer

## Prerequisites

- **OS / arch:** Linux x86_64. Linux aarch64 is **not** supported.
- **GPU / driver:** NVIDIA GPU with CUDA 12.8 capability and >= 24 GB
  VRAM (48 GB+ recommended).
    - Ampere (A100/A10/A40/RTX A6000): driver R550+ required, R570+
      recommended.
    - Ada Lovelace (L20/L40/L40S): same as Ampere.
    - Grace Hopper (H100/H20): same as Ampere.
    - Blackwell (RTX Pro 6000D): driver R580+ required.
    - Fixer-only host (no NRE training): R535+ acceptable.
- **Container runtime:** Docker >= 23.0.1; NVIDIA Container Toolkit
  >= 1.13.5. Verify with `docker run --rm --runtime=nvidia --gpus all
  ubuntu nvidia-smi`.
- **NGC:** account at https://catalog.ngc.nvidia.com/ and an API key
  generated at https://org.ngc.nvidia.com/setup/api-key. Export it as
  `NGC_API_KEY` and `docker login nvcr.io` with
  `Username: $oauthtoken`. The NGC CLI is optional but required for
  downloading the gRPC protobuf bundle (see
  `references/grpc-api.md`).
- **Disk:** budget tens of GB per clip for NCore shards + aux data +
  checkpoints + exports.
- **Memory / shm:** the container expects `--shm-size=64g` for
  training / export / render work and `--shm-size=2g` for `nre-tools`.
- **File ownership.** Every documented `docker run` below threads
  `-u "$(id -u):$(id -g)"` so artifacts (`metrics.yaml`, the USDZ,
  PNG / JPG renders, MP4 videos) are owned by the host user. Without
  this flag the container runs as `root` and every output requires
  `sudo` to move, rename, or delete. If you ever omit it, recover
  with `sudo chown -R "$(id -u):$(id -g)" <output_dir>`.

### Verifying secrets safely

**Always verify prerequisites by running
[`scripts/validate_setup.py`](scripts/validate_setup.py); never by
writing ad-hoc bash that interpolates `NGC_API_KEY` / `HF_TOKEN`
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

`docker login nvcr.io` should always use `--password-stdin` (as in
[Install — Docker / NGC container](#install--docker--ngc-container))
so the key never appears in process tables or shell history. If you
suspect a token was echoed, rotate it at
<https://org.ngc.nvidia.com/setup/api-key>.

## Limitations

- **Linux x86_64 only.** aarch64 (e.g. Jetson) is not supported as a
  host platform.
- **Internal source not redistributable.** The NRE source tree itself
  is not public; treat the NGC containers
  (`nvcr.io/nvidia/nre/nre:latest`,
  `nvcr.io/nvidia/nre/nre-tools:latest`) and the public NuRec docs as
  the only supported surfaces.
- **Multi-GPU defaults are conservative.** Out of the box NRE binds to
  the first visible GPUs only; set `trainer.world_size` /
  `trainer.num_nodes` (or `CUDA_VISIBLE_DEVICES`) explicitly to scale
  out. SLURM is auto-detected when `world_size=0` and `num_nodes=0`.
  Reconstruction quality plateaus past ~6 GPUs; per the release notes,
  multi-GPU + `dataset.aux_data=false` is a known crash combination.
- **`--config-name` paths differ between train and val/export.**
  Training uses recipe paths bundled in the container
  (`configs/apps/AV/Waymo/3dgut_dynamic.yaml`); validation and
  exports re-pass the `parsed.yaml` that training wrote to
  `<output_dir>/<RUN-ID>/config/`.
- **Validation may prompt for `wandb`.** Choose option 3 to skip when
  running non-interactively, or pass `logger=tensorboard` /
  `logger=dummy`.
- **Render gRPC is data-format-pinned.** Older releases warn / reject
  artifacts that pre-date them (e.g. 25.07 broke 25.06 data); see
  release notes when mixing client / server versions.
- **Asset-Harvester input only.** `export-external-assets` requires
  outputs from the `asset-harvester`
  pipeline; raw `.ply` files alone won't carry the per-asset cuboid
  metadata.
- **Difix variants are pluggable.** The container ships both the
  Cosmos Difix variant (default since 25.09 — `difix=cosmos_difix`)
  and the legacy Stable-Diffusion variant
  (`difix=sd_difix`). The newer Cosmos-Predict-based Fixer variants
  with their own inference container live in the
  `nurec-fixer` skill — that pipeline runs
  out-of-band on a directory of frames.
- **`render` ↔ `render-grpc` overlap.** `render` runs in-container
  without a server; `render-grpc` requires an active `serve-grpc`
  process. Choose `render` for batch novel-view jobs and
  `render-grpc` when you need actor editing, LiDAR rendering, or a
  long-lived service.

## Troubleshooting

| Error | Cause | Solution |
|-------|-------|----------|
| `Unable to find image 'nvcr.io/nvidia/nre/nre:latest'` | Docker not authenticated to NGC. | `export NGC_API_KEY=<key>; docker login nvcr.io` with `Username: $oauthtoken`. |
| `RuntimeError: Attempting to deserialize object on a CUDA device but torch.cuda.is_available() is False` | Container started without `--gpus all` / NVIDIA Container Toolkit missing. | Add `--gpus all` (and `--runtime=nvidia` if needed); install NVIDIA Container Toolkit >= 1.13.5 on the host. |
| `OOM Killed` / `CUDA out of memory` during training | Default Hydra recipe targets >= 48 GB VRAM. | Reduce `dataset.camera_ids` to a single ID, lower `trainer.max_epochs`, switch to `trainer.precision=16-mixed`, lower `dataset.n_train_sample_camera_rays`, or run on a 48 GB+ GPU. |
| `wandb` prompt blocks training in non-TTY mode | NRE asks for a wandb mode during validation. | Choose option `3` (skip) interactively, or set `logger=tensorboard` (already the docs default), or set `logger.offline=true`, or `WANDB_MODE=disabled` in the container environment. |
| `export-*` complains the artifact path is wrong | Export commands re-read `<output_dir>/<RUN-ID>/config/parsed.yaml`; the dataset and output volume mounts MUST match those used at training time. | Re-mount with the same host paths as the training command. |
| `serve-grpc` fails to find the USDZ | `--artifact-glob` must end in `.usdz`. | Use e.g. `--artifact-glob /workdir/output/<RUN-ID>/usd-out/last.usdz`. Quote the glob to keep your shell from expanding it. |
| `serve-grpc` starts but USDZs aren't there | Training didn't write a USDZ. | Re-train with `checkpoint.artifact.enabled=true`, or re-export an existing checkpoint with `export-usdz-artifact`. |
| `replacement_id <id> not found in external_assets` from `render-grpc --edit-assets` | The repackage step (`export-external-assets`) was skipped or pointed at a USDZ that did not include that AH output. | Re-run `export-external-assets` with the correct `--external-assets-dir` and use the resulting `target-external-assets.usdz` in `serve-grpc`. |
| Edits silently ignored from `render-grpc --edit-assets` | Server was started without `--enable-editing-actors`. | Restart `serve-grpc` with that flag. |
| `--enable-nrend` / `--use-gsplat` deprecation warning | Old renderer flags. | Replace with `--renderer nrend` or `--renderer gsplat`. |
| `--no-enable-nrend` warning in old asset-editing docs | Same deprecation. | Use `--renderer default` (or `--renderer gsplat`). |
| Driver mismatch on Blackwell (RTX Pro 6000D) | R570 is too old. | Upgrade host driver to R580+. |
| Multi-GPU training crashes with `dataset.aux_data=false` | Known issue carried from 25.06–25.07. | Stay single-GPU when debugging this combination, or generate aux data first (use `nre-tools`). |
| `render-grpc` LiDAR responses too large for gRPC | Default gRPC message limit. | The bundled `render-grpc` client already raises send/receive limits to 50 MB; if writing your own client, set `grpc.max_send_message_length` and `grpc.max_receive_message_length` similarly. |
| Old USDZ loads slowly every run | NRE upgrades artifacts in memory each load. | Run `upgrade-artifact --input old.usdz --output upgraded.usdz` once and use the upgraded file thereafter. |
| `compute-metrics` "CUDA not available" warning | Default `--device=cuda`. | Pass `--device=cpu` or run with `--gpus all`. |

## Overview

NRE (the Neural Reconstruction Engine) is the engine behind **NVIDIA
Omniverse NuRec**. It ingests **NCore-formatted** AV driving logs —
multi-camera, multi-LiDAR clips with rig calibration, ego pose, and
cuboid tracks — and trains a 3D Gaussian reconstruction using the
**3D Gaussian Unscented Transform (3DGUT)** renderer (or the
ray-traced **3DGRT** variant). Outputs are packaged as a single
**USDZ** scene plus a `.ckpt` containing the trained Gaussians; from
there the same container can:

- replay novel trajectories over a **sensorsim gRPC API**
  (`serve-grpc` + `render-grpc`, supports RGB and LiDAR rendering,
  Difix post-processing, and per-frame actor editing),
- render frames locally without a server (`render`),
- export PLY Gaussians, depth, ego masks, Poisson and ground meshes,
  fused / per-frame point clouds, cuboid-track JSON, NCore tracks,
- repackage Asset-Harvester `.ply` actors into the scene,
- evaluate rendering metrics, browse the scene in a viewer, and
- upgrade old artifacts to the current release.

In the broader Physical-AI stack:

```text
                ┌─────────────────────────────────────────────┐
                │           Raw camera + LiDAR clip           │
                └──────────────────────┬──────────────────────┘
                                       │
                            ncore (convert / validate)
                                       │
                              <NAME>.zarr.itar
                                       │
                  nvcr.io/nvidia/nre/nre-tools (aux data)
                                       │
                  nvcr.io/nvidia/nre/nre  (3DGUT train)
                                       │
                            ${RUN}/usd-out/last.usdz
              ┌──────────────┬─────────┴─────────┬──────────────┐
              │              │                   │              │
       serve-grpc        render             export-*       upgrade-artifact
              │
            ┌─┴────────────────────────────────────────┐
   render-grpc / render-novel-trajectory / Python clients
            │
   CARLA / AlpaSim / Isaac Sim / custom client
```

NRE pairs cleanly with the `ncore` skill (data
ingest), the `asset-harvester` skill
(per-object Gaussian assets), and the
`nurec-fixer` skill (artifact harmonisation
of rendered novel views).

## Install — Docker / NGC container

Two public images carry every workflow on this page:

```bash
export NGC_API_KEY="<your key from https://org.ngc.nvidia.com/setup/api-key>"
echo "${NGC_API_KEY}" | docker login nvcr.io --username "\$oauthtoken" --password-stdin

docker pull nvcr.io/nvidia/nre/nre:latest          # train / val / export / render / serve-grpc / render-grpc / viewer
docker pull nvcr.io/nvidia/nre/nre-tools:latest    # aux data + asset-harvester subcommand
```

> Pin a specific release tag when re-runnability matters (e.g.
> `:25.11`, `:26.04`). `:latest` floats and may change behaviour
> between checkout sessions.

The container expects two host bind mounts: dataset (read-only is OK
once you've generated aux data) and output (read-write).

## Warm-Server Thin Client Quick Start

Use this path when a session will render the same USDZ repeatedly or
render several cameras per pose. It keeps one `serve-grpc` container
alive and calls it from host-side Python.

```bash
# 1. Authenticate to NGC once per host; see references/ngc-and-registry.md.
NGC_KEY="${NGC_CLI_API_KEY:-${NGC_API_KEY:-}}"
echo "$NGC_KEY" | docker login nvcr.io -u '$oauthtoken' --password-stdin

# 2. Extract protobuf stubs from the cached NRE image.
bash references/NRE_RenderClient/scripts/setup_protos.sh

# 3. Boot the warm server. NRE_GRPC_USDZ_HOST_DIR is the host root that
#    contains one or more USDZ files.
NRE_GRPC_USDZ_HOST_DIR=/path/to/usdz/root \
  bash scripts/session_warm_server.sh

# 4. Render one or more cameras with the thin client.
python3 references/NRE_RenderClient/scripts/thin_client.py \
  --cameras camera_front_wide_120fov \
  --rig-file references/rig-json/augmented_rig.json \
  --output-dir /tmp/nre-out/turn-001

# 5. Encode frames to MP4 if needed, then tear down at session end.
bash scripts/session_teardown.sh
```

Use the local `nre render` CLI instead when the turn needs LiDAR,
actor edits, rolling shutter, arbitrary custom ego trajectories, or
features the thin client deliberately leaves to the NRE container CLI.

## CLI cookbook

See `references/cli-reference.md` for the full sub-command matrix.
Most-used invocations:

```bash
# Train + validate a Waymo-style dynamic 3DGUT reconstruction.
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

```bash
# Re-validate the trained model with a 3 m lateral shift.
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

```bash
# Render frames locally (no gRPC) along the training trajectory at
# quarter resolution, encoding an MP4 per camera.
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

```bash
# Render at NATIVE RESOLUTION along the original sensor poses, JPEG
# output (smaller files, ~10x lighter than PNG; switch to
# --image-format png if you need bit-exact output for downstream
# eval). --replicate-training-views (default ON) locks per-frame
# poses, intrinsics, and ISP to the training rig, and overrides
# --calib-source to 'training-sensor-poses-calib' (the optimised
# sensor-to-world poses) for the most faithful playback. Repeat
# --camera-id to render multiple sensors in one pass. Budget on a
# 48 GB Ampere/Ada GPU: ~270 ms per 1920x1080 frame (~5x slower wall
# time vs the 0.25 quarter-res preset above).
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
# Then stitch into a visually-lossless H.264 MP4 with ffmpeg. Use this
# fallback whenever your NRE container predates release 26.04 (where
# the in-container --export-video / --video-fps / --video-crf flags
# landed); on 26.04+ append those flags to the render command above
# and skip ffmpeg entirely:
ffmpeg -framerate 30 \
  -i /path/to/output/<RUN-ID>/renders/camera_front_wide_120fov/%06d.jpg \
  -c:v libx264 -crf 18 -preset slow -pix_fmt yuv420p \
  /path/to/output/<RUN-ID>/renders/camera_front_wide_120fov.mp4
```

> **Quality knobs** for `render`:
> `--image-scale 1.0` (native res, never upscale; reconstructions are
> trained at the source resolution) · `--image-format jpg` (default
> for batch/archival; ~10x lighter on disk than `png`. Use
> `--image-format png` whenever you need bit-exact pixels — GT
> comparison via `eval-rendering-metrics`, perceptual-loss training,
> or downstream perception ingest) · `--replicate-training-views`
> (keep ON for fidelity; only flip OFF when you apply rig offsets,
> edit actors, or use a custom rig trajectory) ·
> `--calib-source training-sensor-poses-calib` (the optimised
> per-sensor poses; auto-selected by `--replicate-training-views`
> and the highest-fidelity option of the four) · `--renderer nrend`
> (the fast C++/CUDA path on supported models — same pixels, lower
> wall time) · `--frame-naming contiguous-output-index` (drop-in for
> `ffmpeg %06d`) / `frame-end-timestamp` (deterministic global
> timestamps for benchmarking against GT). Output bitrate
> post-encode: target `crf 18` (visually lossless) or `crf 14`
> (perceptually lossless).

```bash
# Launch the sensorsim gRPC server on a trained USDZ, with editing
# enabled (so render-grpc --edit-assets can apply edits later).
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

```bash
# Render a LiDAR sweep from a running serve-grpc instance.
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

```bash
# In-container CLI help (sub-commands and flags).
docker run --rm --gpus all nvcr.io/nvidia/nre/nre:latest --help
docker run --rm --gpus all nvcr.io/nvidia/nre/nre:latest render --help
docker run --rm --gpus all nvcr.io/nvidia/nre/nre:latest serve-grpc --help
docker run --rm --gpus all nvcr.io/nvidia/nre/nre:latest export-mesh --help
docker run --rm --gpus all nvcr.io/nvidia/nre/nre:latest upgrade-artifact --help
```

## End-to-end workflows

### Workflow A — NCore clip → reconstruction → novel-view render

1. `python scripts/validate_setup.py` — host prerequisites green.
2. Generate auxiliary data with `nvcr.io/nvidia/nre/nre-tools:latest`
   (see `references/aux-data.md`). Confirm `<NAME>.aux.sseg.zarr` and
   (optionally) `<NAME>.aux.depth.zarr` sit next to `<NAME>.zarr.itar`.
3. Train + validate with the CLI cookbook command above. Wait for
   `usd-out/last.usdz` to appear and `metrics.yaml` `test/psnr` to
   stabilise.
4. `export-gaussian-plys` (optional) to emit Gaussians for offline
   tooling, or `export-mesh` / `export-ground-mesh` for downstream
   Omniverse use.
5. Render — choose **(a)** local `nre render` for batch novel-view
   jobs without a server, or **(b)** `serve-grpc` + `render-grpc`
   (or your own Python client via `nre.grpc.protos`) when you need
   actor editing, LiDAR rendering, or a long-lived service.

### Workflow B — Render the NVIDIA Physical AI dataset

Skip the training step entirely. Download the pre-reconstructed USDZs
from HuggingFace (gated dataset
`nvidia/PhysicalAI-Autonomous-Vehicles-NuRec`), then jump straight to
`serve-grpc` + a Python client. Full code for coordinate-frame
conversion (NuRec ↔ ECEF ↔ OpenDRIVE ENU) lives in
`references/physical-ai-render.md`.

### Workflow C — Insert Asset-Harvester actors into a USDZ

1. Run `asset-harvester` on the relevant
   `track_ids` from the source NCore clip to produce a directory of
   `.ply` + `metadata.yaml` per asset.
2. `export-external-assets` to repackage the AH directory into a new
   USDZ and produce a stub `edit-assets.json`.
3. Edit the JSON (`replace`, `remove`, `insert`) — schema lives in
   `references/asset-editing.md`.
4. `serve-grpc` with `--enable-editing-actors`, then `render-grpc
   --edit-assets <path>` to produce edited frames.

### Workflow D — Multi-GPU training

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

### Workflow E — Resume training

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
`resume_weights_only=true` for transfer fine-tuning. `out_dir` cannot
change between the original and resumed runs.

### Workflow F — Upgrade an old USDZ once (avoid in-memory upgrade tax)

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

`upgrade-config` is the equivalent for a standalone YAML / `parsed.yaml`
extracted from a USDZ. To inspect the structure of a checkpoint
before/after upgrade use `export-artifact-structure`.

### Workflow G — Headless rendering of LiDAR sweeps

Combine a long-lived `serve-grpc` (e.g. on a render node) with one or
more `render-grpc --lidar` workers:

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
client against `nre.grpc.protos` — see `references/grpc-api.md`.

### Workflow H — Browse a USDZ in a browser

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

### Workflow I — Evaluate rendering quality against ground truth

1. `export-ncore-benchmark-gt` to materialize the GT folder layout
   (`<gt-dir>/camera_images/<seq>/<ts>.<ext>` plus optional 
   `<gt-dir>/camera_ego_masks/<seq>.png`).
2. Render the trained USDZ along the same trajectory (`render` or
   `render-grpc`). Use `--frame-naming frame-end-timestamp` so frames
   match the GT naming.
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

   For one-off pairs, `compute-metrics --metric psnr|ssim|lpips|fid|...`
   is a lightweight alternative.

## Teardown

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

Do **not** revoke `NGC_API_KEY` unless you suspect it has been leaked
(see [Verifying secrets safely](#verifying-secrets-safely)).

Full inventory, per-artifact sizes, and post-teardown verification
commands live in [`references/teardown.md`](references/teardown.md).
