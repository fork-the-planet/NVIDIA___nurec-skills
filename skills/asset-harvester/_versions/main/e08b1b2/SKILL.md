---
name: asset-harvester
description: >-
  Use when the user wants to install or run NVIDIA Asset Harvester —
  the open-source Apache-2.0 image-to-3D pipeline at
  github.com/NVIDIA/asset-harvester that converts sparse object views
  from autonomous-vehicle NCore V4 driving logs (or single masked
  images) into simulation-ready 3D Gaussian Splat (.ply) assets via
  SparseViewDiT multiview diffusion plus TokenGS Gaussian lifting, and
  optionally emits metadata.yaml for NVIDIA Omniverse NuRec object
  insertion. Do NOT use for full-scene reconstruction (use `nre`) or
  for object inputs without per-object masks. Trigger keywords: asset
  harvester, asset-harvester, nvidia/asset-harvester, SparseViewDiT,
  TokenGS, image-to-3D AV, multiview diffusion, gaussian lifting, 3DGS
  from NCore, gaussian splat from driving log, run_inference.py,
  run.sh, run_ncore_parser.sh, AH_multiview_diffusion,
  AH_tokengs_lifting, external assets metadata, NuRec object
  insertion, NCore V4 parser, harvest 3D assets from AV clips,
  sparse-view to 3D.
version: "0.1.0"
tools:
  - Shell
  - Read
  - Write
license: CC-BY-4.0 AND Apache-2.0
compatibility: >-
  Linux with conda (Miniconda or Miniforge), NVIDIA driver >= 570
  (CUDA 12.8 compatible), GCC 10-13 (tested 12.3), CUDA toolkit 12.8
  (installed by setup.sh), ~16 GB GPU VRAM (use --offload_model_to_cpu
  for lower VRAM). Requires network egress to github.com,
  huggingface.co, and pypi.org. HF_TOKEN env var must be set (or
  `hf auth login` run) to download the gated nvidia/asset-harvester
  checkpoints and the PhysicalAI-Autonomous-Vehicles-NCore dataset.
dependencies:
  - bash
  - conda
  - git
  - python3
metadata:
  author: NVIDIA AV
  tags:
    - asset-harvester
    - autonomous-vehicles
    - 3d-reconstruction
    - gaussian-splatting
    - simulation
  upstream: https://github.com/NVIDIA/asset-harvester
  project_page: https://research.nvidia.com/labs/sil/projects/asset-harvester/
  paper: https://arxiv.org/abs/2604.18468
  hf_model: https://huggingface.co/nvidia/asset-harvester
  hf_demo: https://huggingface.co/spaces/nvidia/asset-harvester
  hf_dataset: https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles-NCore
  hf_benchmark: https://huggingface.co/datasets/nvidia/NuRec-AV-Object-Benchmark
  time-estimate: "45min (20min setup + 15min inference + evaluation)"
---

# Asset Harvester

## Purpose

Install and drive NVIDIA Asset Harvester to extract per-object 3D
Gaussian Splat assets from sparse autonomous-vehicle (AV) object
observations — either a multi-view crop pulled from an NCore V4
driving log or a single masked image. The output is a
simulation-ready `gaussians.ply` plus optional `metadata.yaml` that
NVIDIA Omniverse NuRec can ingest as an external asset.

**Use this skill when:** the user has AV clips or masked single
images and wants per-object 3D assets via the SparseViewDiT + TokenGS
pipeline at <https://github.com/NVIDIA/asset-harvester>.

**Do NOT use this skill when:**

- The user wants a full-scene reconstruction (use the `nre` skill).
- The user has no per-object masks or AV-style object crops (Asset
  Harvester expects object-centric, masked inputs).
- The user wants to re-train SparseViewDiT or TokenGS from scratch —
  this skill is install + inference only.

## Background

Open-source (Apache-2.0) image-to-3D pipeline that turns sparse,
in-the-wild object observations from real autonomous-vehicle driving
logs into complete, simulation-ready 3D Gaussian Splat assets
(`gaussians.ply`). It pairs **SparseViewDiT** (multiview diffusion,
16 consistent views) with **TokenGS** (feed-forward Gaussian lifting),
and plugs into the NVIDIA NCore (ingestion) and NuRec (asset insertion
/ closed-loop simulation) ecosystem.

Pipeline:

```text
NCore V4 clip ──► NCore parsing ──► SparseViewDiT (16-view diffusion)
              ──► TokenGS lifting ──► gaussians.ply
              ──► (optional) metadata.yaml for NuRec object insertion
```

Shipped model checkpoints (single HF repo `nvidia/asset-harvester`):

| Checkpoint | Role |
|------------|------|
| `AH_object_seg_jit.pt` | AV-object Mask2Former (instance segmentation) |
| `AH_multiview_diffusion.safetensors` | SparseViewDiT — sparse views → 16 consistent multi-views |
| `AH_camera_estimator.safetensors` | Camera pose estimator (used when calibration is absent) |
| `AH_tokengs_lifting.safetensors` | TokenGS — multi-view → 3D Gaussians |

## When to Use

- User wants to turn a single AV object image (or a handful) into a
  3D Gaussian Splat asset.
- User has NCore V4 driving-log clips and wants per-track 3D assets
  for simulation.
- User asks about `SparseViewDiT`, `TokenGS`, or wants to reproduce
  the Asset Harvester paper / HF Space demo locally.
- User wants `.ply` Gaussians + `metadata.yaml` suitable for NVIDIA
  Omniverse **NuRec** object insertion.

### When NOT to Use

- General-purpose text-to-3D, indoor scans, or non-AV imagery — the
  model is trained for AV objects only (vehicles, VRUs, road
  objects). Quality is not guaranteed out of domain.
- Closed-loop AV scene reconstruction (not object assets) — use the
  sibling [`nre`](../nre/SKILL.md) skill instead.
- Ingesting raw sensor data into NCore V4 before harvesting — use
  the sibling [`ncore`](../ncore/SKILL.md) skill;
  Asset Harvester consumes NCore V4 stores, it does not produce
  them.
- Running the HF Space demo (no install): point the user at
  https://huggingface.co/spaces/nvidia/asset-harvester.

## Inputs

- **image_root** — directory of per-object folders, each with
  `frame.jpeg` (512×512) and (optional) `mask.png` (source: user
  prompt; required unless `component_store` is given).
- **component_store** — path to NCore V4 clip `.json` manifest,
  comma-separated component-store paths, or `.zarr.itar` glob (source:
  user prompt; required when running the NCore parsing path).
- **output_dir** — where per-sample outputs (`gaussians.ply`,
  `multiview/`, `3d_lifted/`, `*.mp4`) are written (source: user
  prompt; optional; default: `outputs/`).
- **offload_flag** — enable CPU offload (`--offload_model_to_cpu` /
  `--offload`) when VRAM < ~16 GB (source: user prompt; optional;
  default: off).
- **HF_TOKEN** — HuggingFace access token used by `hf download` for
  the gated `nvidia/asset-harvester` repo and (optionally) the
  PhysicalAI NCore dataset (source: agent context / user env;
  required; obtain at https://huggingface.co/settings/tokens).

## Instructions

1. **Validate the host.** Run `scripts/validate_setup.py` to confirm
   conda, NVIDIA driver, GCC, and `HF_TOKEN` are present. Fail fast
   (exit code 1) if a prerequisite is missing.
2. **Clone and install.** Use the one-shot `bash setup.sh` path unless
   the user specifically asks for a manual install. See
   [Installation](#installation). Do **not** skip `gsplat`'s pinned
   commit — order matters.
3. **Download checkpoints.** `hf auth login` first, then
   `hf download nvidia/asset-harvester --local-dir checkpoints`. See
   [Checkpoints](#checkpoints).
4. **Pick the inference path** based on user inputs:
   - Bundled demo → [Quick start](#quick-start-on-bundled-samples).
   - Single user image (+/- mask) → [Single-view workflow](#single-view-workflow).
   - NCore V4 driving log → [End-to-end on NCore V4](#end-to-end-on-ncore-v4-driving-logs).
5. **Execute with appropriate VRAM flag.** If the user has < 16 GB
   VRAM, add `--offload_model_to_cpu` (direct `run_inference.py`) or
   `--offload` (`run.sh` wrapper).
6. **Validate outputs.** Confirm `gaussians.ply` exists under the
   expected `${OUTPUT_DIR}/<sample>/` path, and that `multiview.mp4`
   and `3d_lifted.mp4` render. If the run targets NuRec, also generate
   `metadata.yaml` (see [NuRec handoff](#nurec-handoff-optional)).
7. **(Optional) Benchmark.** Clone the env to `av-object-benchmark`
   and run `benchmark/eval.py` to compute PSNR / LPIPS / SSIM and
   DINOv3 embedding metrics. See
   [references/end-to-end-ncore.md](references/end-to-end-ncore.md)
   for the full flow.
8. **(Optional) Hand off to NuRec.** Rotate Gaussians with
   `orient_gaussians_for_nurec`, emit `metadata.yaml`, and follow the
   [NuRec external-assets docs](https://docs.nvidia.com/nurec/nurec/use-ah-assets.html).

## Output Format

Per input sample (image or NCore track) the pipeline writes:

```text
${OUTPUT_DIR}/<sample_id>/
├── multiview/           # 16 RGB views generated by SparseViewDiT
├── 3d_lifted/           # TokenGS-rendered views of the lifted Gaussians
├── gaussians.ply        # 3D Gaussian Splat asset (ready for Omniverse/NuRec)
├── multiview.mp4        # spin-around of the generated views
└── 3d_lifted.mp4        # spin-around of the rendered Gaussians
```

When step 3 of the NuRec handoff is run, `metadata.yaml` is additionally
written at the root of the oriented output directory and describes each
`gaussians.ply` plus its NuRec insertion parameters.

## Prerequisites

- Linux host (Ubuntu 22.04 tested).
- **conda** (Miniconda or Miniforge).
- NVIDIA driver `>= 570` (CUDA 12.8 compatible).
- GCC 10–13 on PATH (tested 12.3); nvcc needs a compatible host
  compiler.
- ~16 GB GPU VRAM minimum. Add `--offload_model_to_cpu` (or `--offload`
  on `run.sh`) if you have less.
- HuggingFace account + token (`HF_TOKEN`). Accept the
  [nvidia/asset-harvester](https://huggingface.co/nvidia/asset-harvester)
  model card terms so `hf download` succeeds.
- ~30 GB free disk (checkpoints ~5 GB + outputs + conda env).
- Internet egress to `github.com`, `huggingface.co`, `pypi.org`, and
  `download.pytorch.org`.

### Verifying secrets safely

**Always verify prerequisites by running
[`scripts/validate_setup.py`](scripts/validate_setup.py); never by
writing ad-hoc bash that interpolates `HF_TOKEN` values.** The common
one-liner

```bash
# BAD — leaks the secret to the terminal when the variable is set
echo "HF_TOKEN: ${HF_TOKEN:+yes}${HF_TOKEN:-no}"
```

prints `yes<token-value>` whenever `HF_TOKEN` is set, because
`${VAR:-no}` only falls back to "no" when `VAR` is empty — when set
it expands to `$VAR`. Use a length-only check, which never echoes
the value:

```bash
# OK — prints "set (N chars)" or "missing", never the value
test -n "$HF_TOKEN" && echo "HF_TOKEN: set (${#HF_TOKEN} chars)" || echo "HF_TOKEN: missing"
```

Rotate any token you suspect was echoed at
<https://huggingface.co/settings/tokens>.

## Installation

One-shot (recommended, ~20 min):

```bash
git clone https://github.com/NVIDIA/asset-harvester.git
cd asset-harvester
bash setup.sh                 # creates the `asset-harvester` conda env
conda activate asset-harvester
```

Optional flags: `bash setup.sh --env-name asset-harvester --python 3.10`.

`setup.sh` handles: git submodules, conda env creation, `cuda-toolkit=12.8`
install, nvcc host-compiler probing, PyTorch 2.10.0 CUDA wheels, a
**source build of `gsplat` at the pinned commit
`b60e917c95afc449c5be33a634f1f457e116ff5e`**, editable install of
`asset-harvester` with all extras, and `ruff`.

### Manual install (when `setup.sh` is not usable)

Preinstall `gsplat` at the pinned commit **before** the editable
install — otherwise pip will resolve a wheel with the wrong CUDA ABI:

```bash
pip install --extra-index-url https://download.pytorch.org/whl/cu128 \
    torch==2.10.0 torchvision
pip install --no-cache-dir --no-build-isolation \
    "git+https://github.com/nerfstudio-project/gsplat.git@b60e917c95afc449c5be33a634f1f457e116ff5e"
pip install --extra-index-url https://download.pytorch.org/whl/cu128 \
    -e ".[ncore-parser,multiview_diffusion,tokengs,camera-estimator]"
```

Sanity check the CUDA extension:

```bash
python -c "from gsplat.cuda._backend import _C; print('gsplat CUDA ready')"
```

## Checkpoints

```bash
pip install "huggingface_hub[cli]"
hf auth login                 # paste the token from https://huggingface.co/settings/tokens
hf download nvidia/asset-harvester --local-dir checkpoints
```

Result:

```text
checkpoints/
├── AH_multiview_diffusion.safetensors
├── AH_tokengs_lifting.safetensors
├── AH_camera_estimator.safetensors
└── AH_object_seg_jit.pt
```

## Quick start on bundled samples

`data_samples/rectified_AV_objects/` ships with the repo and exercises
the multiview-diffusion + lifting path (no parsing, no camera
estimation — intrinsics are pre-rectified).

```bash
export DATA_ROOT=data_samples/rectified_AV_objects/
export CHECKPOINT_MV=checkpoints/AH_multiview_diffusion.safetensors
export CHECKPOINT_GS=checkpoints/AH_tokengs_lifting.safetensors
export OUTPUT_DIR=outputs/harvesting

python3 run_inference.py \
    --diffusion_checkpoint "${CHECKPOINT_MV}" \
    --data_root "${DATA_ROOT}" \
    --output_dir "${OUTPUT_DIR}" \
    --lifting_checkpoint "${CHECKPOINT_GS}"
```

Add `--offload_model_to_cpu` to `run_inference.py` if you OOM at 16 GB.

## Single-view workflow

Layout each object as one folder with a 512×512 `frame.jpeg` and
`mask.png`:

```text
YOUR_IMAGE_ROOT/
├── object_0/
│   ├── frame.jpeg
│   └── mask.png
└── object_1/
    └── ...
```

If you only have `frame.jpeg`, generate `mask.png` with the bundled
AV-object segmentation model:

```bash
export CHECKPOINT_SEG=checkpoints/AH_object_seg_jit.pt
export IMAGE_ROOT=data_samples/OOD_images

python -m asset_harvester.utils.image_segment \
    --checkpoint "${CHECKPOINT_SEG}" \
    --image_folder "${IMAGE_ROOT}" \
    --frame_name frame.jpeg \
    --mask_name mask.png
```

Then run inference with the built-in camera estimator (no calibration
needed):

```bash
export YOUR_IMAGE_ROOT=data_samples/OOD_images
export CHECKPOINT_MV=checkpoints/AH_multiview_diffusion.safetensors
export CHECKPOINT_GS=checkpoints/AH_tokengs_lifting.safetensors
export CHECKPOINT_CAM=checkpoints/AH_camera_estimator.safetensors
export OUTPUT_DIR=outputs/harvesting_with_camera_estimate

python3 run_inference.py \
    --diffusion_checkpoint "${CHECKPOINT_MV}" \
    --ahc_checkpoint "${CHECKPOINT_CAM}" \
    --image_dir "${YOUR_IMAGE_ROOT}" \
    --output_dir "${OUTPUT_DIR}" \
    --lifting_checkpoint "${CHECKPOINT_GS}"
```

## End-to-end on NCore V4 driving logs

The two-step pipeline is parsing → diffusion+lifting. See
[references/end-to-end-ncore.md](references/end-to-end-ncore.md) for
the full walkthrough including sample-data download, argument
matrices, and the optional benchmark and NuRec handoff.

Condensed form:

```bash
# (optional) pull a sample clip
hf download nvidia/PhysicalAI-Autonomous-Vehicles-NCore \
    --repo-type dataset \
    --local-dir ./ncore-clips \
    --include 'clips/2a6f330-5ab0-4e92-99d4-d19e406952f4/*'

# 1. NCore V4 clip → per-track object crops
bash scripts/run_ncore_parser.sh \
    --component-store "ncore-clips/clips/2a6f330-5ab0-4e92-99d4-d19e406952f4/pai_02a6f330-5ab0-4e92-99d4-d19e406952f4.json"

# 2. Multiview diffusion + Gaussian lifting
bash run.sh \
    --data-root ./outputs/ncore_parser \
    --output-dir ./outputs/ncore_harvest
```

### NuRec handoff (optional)

Rotate Gaussians into the orientation NuRec insertion expects, then
emit `metadata.yaml`:

```bash
python -m asset_harvester.utils.orient_gaussians_for_nurec \
    --input-dir ./outputs/ncore_harvest \
    --output-dir ./outputs/ncore_harvest_nurec

python asset_harvester/utils/generate_external_assets_metadata.py \
    --input-dir ./outputs/ncore_harvest_nurec
```

The resulting `./outputs/ncore_harvest_nurec/` directory is the NuRec
external-assets input — see
[NuRec docs](https://docs.nvidia.com/nurec/nurec/use-ah-assets.html).

> **Note:** Disable PPISP when reconstructing the scene with NuRec;
> PPISP introduces color-space mismatches that over-saturate inserted
> assets. Asset Harvester does **not** predict object scale — NuRec
> uses the scales stored in the clip cuboid dimensions.

## Configuration Matrix (most common flags)

| Entry point | Flag | Default | Purpose |
|-------------|------|---------|---------|
| `run_inference.py` | `--diffusion_checkpoint` | *(required)* | SparseViewDiT `.safetensors` path |
| `run_inference.py` | `--lifting_checkpoint` | *(optional)* | TokenGS `.safetensors`; omit to skip lifting |
| `run_inference.py` | `--ahc_checkpoint` | *(optional)* | Camera estimator; required for single-image mode |
| `run_inference.py` | `--data_root` | — | Directory with `sample_paths.json` (rectified samples) |
| `run_inference.py` | `--image_dir` | — | Directory of `frame.jpeg`/`mask.png` folders (single-view mode) |
| `run_inference.py` | `--output_dir` | `outputs/` | Where to write per-sample outputs |
| `run_inference.py` | `--offload_model_to_cpu` | off | Reduce VRAM (<16 GB) |
| `run.sh` (wraps step 2) | `--num-steps` | 30 | Diffusion inference steps |
| `run.sh` | `--cfg-scale` | 2.0 | Classifier-free guidance |
| `run.sh` | `--skip-lifting` | off | Multiview only (no `.ply`) |
| `run.sh` | `--offload` | off | Offload diffusion to CPU during lifting |
| `run_ncore_parser.sh` | `--component-store` | *(required)* | NCore V4 clip manifest / component stores / zarr glob |
| `run_ncore_parser.sh` | `--camera-ids` | 5 default cameras | Comma-separated sensor IDs |
| `run_ncore_parser.sh` | `--track-ids` | all | Comma-separated track IDs |

Full flag tables (including every option for every wrapper) live in
[references/cli-reference.md](references/cli-reference.md).

## Scripts

| Script | Purpose | Usage |
|--------|---------|-------|
| `scripts/validate_setup.py` | Verify host meets Asset Harvester prerequisites (conda, driver, GCC, `HF_TOKEN`). Does **not** hit the network. | Execute: `python scripts/validate_setup.py` |

## References

- `references/cli-reference.md` — exhaustive flag matrix for
  `run_inference.py`, `run.sh`, `run_ncore_parser.sh`,
  `image_segment`, `orient_gaussians_for_nurec`, and
  `generate_external_assets_metadata.py`.
- `references/end-to-end-ncore.md` — full NCore V4 walkthrough
  including sample-data download, benchmark eval in the cloned
  `av-object-benchmark` env, and the NuRec handoff checklist.
- Sibling skill — NCore V4 ingestion: [`../ncore/SKILL.md`](../ncore/SKILL.md).
- Sibling skill — NRE / NuRec scene reconstruction and
  `export-external-assets` packaging of the produced `.ply` +
  `metadata.yaml`: [`../nre/SKILL.md`](../nre/SKILL.md).
- Sibling skill — NCore V4 sample clips and benchmark dataset
  download recipes: [`../physical-ai-datasets/SKILL.md`](../physical-ai-datasets/SKILL.md).
- Upstream README: https://github.com/NVIDIA/asset-harvester
- Upstream end-to-end doc: https://github.com/NVIDIA/asset-harvester/blob/main/docs/end_to_end_example.md
- Project page: https://research.nvidia.com/labs/sil/projects/asset-harvester/
- Paper: https://arxiv.org/abs/2604.18468
- HF model: https://huggingface.co/nvidia/asset-harvester
- HF dataset (NCore V4 clips): https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles-NCore
- HF benchmark: https://huggingface.co/datasets/nvidia/NuRec-AV-Object-Benchmark
- Live demo (no install): https://huggingface.co/spaces/nvidia/asset-harvester
- NCore V4 docs: https://nvidia.github.io/ncore/index.html
- NuRec external-assets guide: https://docs.nvidia.com/nurec/nurec/use-ah-assets.html

## Limitations

- AV-only domain: trained on autonomous-vehicle imagery. Indoor or
  non-road objects are out of distribution.
- `AH_object_seg_jit.pt` is class-restricted to vehicles, VRUs,
  cyclists, and road objects. Supply your own `mask.png` for arbitrary
  objects.
- Scale is **not predicted**. Downstream NuRec insertion relies on
  scales extracted from the original clip's cuboid tracks.
- 16 GB VRAM is the practical floor; lower-VRAM users must offload to
  CPU (slower).
- The model expects 512×512 square crops; non-square or low-resolution
  inputs degrade quality.
- `benchmark/eval.py` needs a separately cloned conda env
  (`av-object-benchmark`) because `transformers>=4.56.0` conflicts
  with the main env's pinned `transformers==4.48.3`.
- Linux-only install path (tested on Ubuntu 22.04 with CUDA 12.8).
- Optional SAM 3D Body embedding metric requires access to the gated
  `facebook/sam-3d-body-dinov3` repo; eval falls back to PSNR/LPIPS/SSIM
  if unavailable.

## Troubleshooting

| Error | Cause | Solution |
|-------|-------|----------|
| `gsplat` import error or CUDA ABI mismatch | Installed `gsplat` from PyPI wheel instead of the pinned commit | Reinstall from source: `pip install --no-cache-dir --no-build-isolation "git+https://github.com/nerfstudio-project/gsplat.git@b60e917c95afc449c5be33a634f1f457e116ff5e"` then redo the editable install. |
| `nvcc` fails — "unsupported GNU version" | GCC outside the 10–13 window on PATH | Install GCC 12 and export `CC`/`CXX`/`CUDAHOSTCXX` before running `setup.sh`. |
| `CUDA error: out of memory` | GPU VRAM < ~16 GB | Add `--offload_model_to_cpu` to `run_inference.py` or `--offload` to `run.sh`. |
| `401 Unauthorized` on `hf download nvidia/asset-harvester` | Model-card terms not accepted, or `HF_TOKEN` missing | `hf auth login` with a token from https://huggingface.co/settings/tokens after accepting the model card on the HF page. |
| `sample_paths.json` not found (step 2) | Step 1 (NCore parsing) wasn't run or `--data-root` points elsewhere | Point `run.sh --data-root` at the `--output-path` of `run_ncore_parser.sh` (default `outputs/ncore_parser/`). |
| Over-saturated / wrong-color assets in NuRec | PPISP enabled during scene reconstruction | Disable PPISP for the NuRec scene reconstruction used with inserted assets. |
| Inserted asset is wrong size in NuRec | Source clip cuboid dimensions are wrong | Asset Harvester does not predict scale — debug the cuboid track in the source NCore clip. |
| `benchmark/eval.py` crashes on `transformers` import | Wrong env (main env pins 4.48.3; benchmark needs ≥ 4.56.0) | `conda activate av-object-benchmark` (created via `conda create --name av-object-benchmark --clone asset-harvester && bash benchmark/install.sh`). |
| SAM 3D Body checkpoint 403 | Gated repo access not granted | Request access at https://huggingface.co/facebook/sam-3d-body-dinov3; eval continues without embedding metrics. |

## Teardown

A complete Asset Harvester install leaves ~30–40 GB on the host:
checkpoints (~5 GB), the source clone, two conda envs (~10–15 GB
each), per-sample outputs (multiview MP4s + Gaussian PLYs), and any
downloaded NCore sample clips.

```bash
# 1. Conda environments
conda deactivate 2>/dev/null || true
conda env remove --name asset-harvester --yes 2>/dev/null || true
conda env remove --name av-object-benchmark --yes 2>/dev/null || true

# 2. Source clone (and the pinned-commit gsplat build cache it pulled)
rm -rf ./asset-harvester                              # adjust if cloned elsewhere

# 3. HuggingFace checkpoints (~5 GB)
rm -rf ./checkpoints
rm -rf "${HF_HOME:-$HOME/.cache/huggingface}/hub/models--nvidia--asset-harvester"

# 4. NCore sample clips downloaded for the demo (size depends on clips)
rm -rf ./ncore-clips
rm -rf "${HF_HOME:-$HOME/.cache/huggingface}/hub/datasets--nvidia--PhysicalAI-Autonomous-Vehicles-NCore"

# 5. Per-sample outputs (gaussians.ply, multiview.mp4, 3d_lifted.mp4, …)
rm -rf ./outputs

# 6. pip + conda caches (only if disk is tight; safe to keep)
pip cache purge 2>/dev/null || true
conda clean --all --yes 2>/dev/null || true
```

If any of those paths were written by a container or by another user
and ended up `root`-owned, recover ownership first:

```bash
sudo chown -R "$(id -u):$(id -g)" ./outputs ./checkpoints ./ncore-clips
```

Verify:

```bash
conda env list | grep -E 'asset-harvester|av-object-benchmark' || echo "envs: clean"
du -sh ./checkpoints ./outputs ./asset-harvester 2>/dev/null || echo "files: clean"
```

Do **not** revoke `HF_TOKEN` as part of teardown unless you suspect
it has been leaked (see [Verifying secrets safely](#verifying-secrets-safely));
the token is per-user and shared across HuggingFace workflows.
