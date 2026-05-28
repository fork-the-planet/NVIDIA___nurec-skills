---
name: nurec-fixer
description: >-
  Use to run NVIDIA DiffusionHarmonizer (public successor to the
  older Fixer recipes) to enhance, harmonize, evaluate, or
  fine-tune novel-view frames from NRE / NuRec / 3DGS / NeRF
  reconstructions. Do NOT use for training the 3D reconstruction
  itself (use `nre`) or for sensor-to-NCore conversion (use
  `ncore`).
version: "0.4.1"
tools:
  - Shell
  - Read
  - Write
license: CC-BY-4.0 AND Apache-2.0
compatibility: >-
  Linux + NVIDIA GPU Ampere+ (compute capability >= 8.0; A100,
  A10, L40, H100, RTX 30/40/PRO, B200, GB200), Docker + NVIDIA
  Container Toolkit, ~120 GB free disk. HF_TOKEN required for
  gated `nvidia/DiffusionHarmonizer` model weights; NGC_API_KEY
  needed when pulling Cosmos Predict2 from nvcr.io. Code at
  github.com/NVIDIA/harmonizer; runtime uses
  nvcr.io/nvidia/cosmos/cosmos-predict2-container:1.2 or the
  project Dockerfile that layers required packages and patches.
dependencies:
  - bash
  - docker
  - nvidia-container-toolkit
  - python3
  - git
  - huggingface_hub
metadata:
  author: NVIDIA NRS <nurec-skills@nvidia.com>
  tags:
    - nurec
    - fixer
    - harmonizer
    - diffusionharmonizer
    - reconstruction
    - autonomous-vehicles
    - post-processing
    - evaluation
  upstream: https://github.com/NVIDIA/harmonizer
  hf_model: https://huggingface.co/nvidia/DiffusionHarmonizer
  hf_dataset: https://huggingface.co/datasets/nvidia/DiffusionHarmonizer-Dataset
  container: nvcr.io/nvidia/cosmos/cosmos-predict2-container:1.2
  paper: https://arxiv.org/abs/2602.24096
  product_page: https://research.nvidia.com/labs/sil/projects/diffusion-harmonizer/
  model_license: NVIDIA Open Model License Agreement
---

# NVIDIA DiffusionHarmonizer (NuRec post-processing)

## Purpose

Run NVIDIA DiffusionHarmonizer on rendered images from neural
reconstructions. DiffusionHarmonizer is a single-step,
temporally-aware image diffusion enhancer for NeRF / 3DGS /
NuRec-style renderings. It improves realism, reduces
reconstruction artifacts, and harmonizes inserted dynamic
objects with the surrounding scene.

## When to Use / When NOT to Use

**Use this skill when** the user has rendered frames from NRE,
NuRec, 3DGS, NeRF, or a similar reconstruction pipeline and
wants to enhance, harmonize, evaluate, or optionally fine-tune
the DiffusionHarmonizer model.

**Do NOT use this skill when:**

- The user wants to train or render the 3D reconstruction itself
  (use `nre`).
- The user wants to convert raw sensor data to NCore V4 (use
  `ncore`).
- The user wants a generic photo enhancer. DiffusionHarmonizer
  is tuned for neural-reconstruction artifacts and
  object-insertion failures.
- The user only wants NRE inline rendering with
  `--enable-difix`. That remains an NRE runtime feature; use the
  `nre` skill for the complete `serve-grpc` / `render-grpc`
  command shape.

## What changed from the older Fixer skill

This skill now follows the newer DiffusionHarmonizer release
branches, not the older NGC JIT `.pt` artifact recipe. Use these
public release artifacts:

- Code: <https://github.com/NVIDIA/harmonizer>
- Model: `nvidia/DiffusionHarmonizer` on Hugging Face
- Main checkpoint: `models/pretrained/pretrained_harmonizer.pkl`
- Runtime: Cosmos Predict2 environment
  (`nvcr.io/nvidia/cosmos/cosmos-predict2-container:1.2`)
- Inference entry: `src/inference_pretrained_model.py`
- Evaluation entry: `src/evaluate_test_dataset.py`
- Training entry: `src/train_pix2pix_turbo_harmonizer.py`

Do not use the obsolete standalone recipe that downloads
`nvidia/nre/nurec-fixer:cosmos_3dgut_fixer_harmonizer`, mounts
`harmonizer_temporal.pt`, or runs `inference_jit_harmonizer.py`
inside `nvcr.io/nvidia/pytorch:24.10-py3` unless the user
explicitly asks for that older beta artifact.

## Background

DiffusionHarmonizer is described in *DiffusionHarmonizer:
Bridging Neural Reconstruction and Photorealistic Simulation
with Online Diffusion Enhancer*
([arXiv 2602.24096](https://arxiv.org/abs/2602.24096), CVPR 2026).
It distills a pretrained multi-step diffusion model into a
single-step enhancer designed for online simulation and offline
data cleanup.

Two operating modes:

- **Offline:** clean pseudo-training views rendered from a
  reconstruction, then distill the improved views back into the
  3D representation.
- **Online:** enhance frames during simulation/inference by
  harmonizing color and lighting, reconstructing missing or
  inconsistent shadows for inserted actors, and reducing residual
  reconstruction artifacts.

The public model card describes `DiffusionHarmonizer-cosmos-0.6B`,
a Cosmos Predict2 Diffusion Transformer post-trained at
`576x1024` input and output resolution.

## Inputs

- **code_dir** — checkout of
  <https://github.com/NVIDIA/harmonizer>.
- **model_dir** — directory downloaded from Hugging Face. After
  `hf download nvidia/DiffusionHarmonizer --local-dir models`,
  the main checkpoint is
  `models/pretrained/pretrained_harmonizer.pkl`.
- **input_dir** — directory of rendered RGB frames (`.png`,
  `.jpg`, `.jpeg`) to enhance.
- **output_dir** — directory where enhanced frames are written.
- **HF_TOKEN** — Hugging Face token with access to the model
  and, if used, the dataset. Accept the model/dataset license
  terms first.
- **NGC_API_KEY** — often needed to authenticate `docker pull`
  from `nvcr.io`. Use only for container pulls, not model
  download.

## Instructions

1. **Validate the host.** Have the agent execute
   `scripts/validate_setup.py` via its standard script runner —
   e.g. `run_script("scripts/validate_setup.py")` or
   `python scripts/validate_setup.py`. It checks Docker, the
   NVIDIA Container Toolkit, GPU architecture, `git`, the
   Hugging Face CLI, token presence, and free disk space — and
   exits non-zero on any missing prerequisite.
2. **Clone the code and build (or pull) the runtime image.**
   Full commands and the Blackwell patch caveat live in
   [`references/inference.md`](references/inference.md).

   ```bash
   git clone https://github.com/NVIDIA/harmonizer.git
   cd harmonizer
   docker build -t harmonizer-cosmos-env -f Dockerfile.cosmos .
   ```

3. **Download the pretrained model.**

   ```bash
   export HF_TOKEN=<your-hugging-face-token>
   hf auth login --token "$HF_TOKEN"
   hf download nvidia/DiffusionHarmonizer --local-dir models
   ```

   Verify `models/pretrained/pretrained_harmonizer.pkl` exists.
4. **Confirm `input_dir` exists and filenames sort into frame
   order.** The current public inference script uses normal
   string sorting; prefer zero-padded names such as
   `frame_000001.png`.
5. **Run inference inside the container** with input, output,
   and model paths mounted. Pin `-u $(id -u):$(id -g)` so
   outputs are owned by the host user. Full `docker run` recipe
   and flag matrix in
   [`references/inference.md`](references/inference.md).
6. **Validate that the output frame count matches the input
   frame count,** spot-check frames, and (if ground truth is
   available) run paired evaluation with PSNR/LPIPS — see
   [`references/evaluation.md`](references/evaluation.md).
7. **(Optional) Train or fine-tune.** Download the dataset (or
   prepare JSON manifests in the documented format), then run
   `src/train_pix2pix_turbo_harmonizer.py` with the recommended
   hyperparameters. Full recipe + NuRec data-pair recipes in
   [`references/training.md`](references/training.md).
8. **(Optional) Teardown.** Follow
   [`references/teardown.md`](references/teardown.md) to remove
   images, code clones, model weights, datasets, and outputs.

## Examples

### Example 1 — Enhance a folder of rendered frames

```bash
python scripts/validate_setup.py   # then build the image once
docker run --gpus=all --rm --ipc=host \
  -u "$(id -u):$(id -g)" \
  -v $PWD/harmonizer:/work \
  -v /absolute/path/to/rendered_frames:/input:ro \
  -v /absolute/path/to/enhanced_frames:/output \
  -w /work harmonizer-cosmos-env \
  python /work/src/inference_pretrained_model.py \
    --model /work/models/pretrained/pretrained_harmonizer.pkl \
    --input /input --output /output \
    --timestep 250 --resolution 1024
```

Full flag matrix in
[`references/inference.md`](references/inference.md).

### Example 2 — Quantitative PSNR / LPIPS evaluation

Prepare the paired `test_dataset/{scene}/render` +
`test_dataset/{scene}/gt` layout, then run
`src/evaluate_test_dataset.py` inside the container. See
[`references/evaluation.md`](references/evaluation.md) for the
exact directory shape and the `docker run` command.

### Example 3 — Fine-tune from the public checkpoint

Download `nvidia/DiffusionHarmonizer-Dataset`, prepare the
training JSON, then run
`src/train_pix2pix_turbo_harmonizer.py` with the multi-GPU
`accelerate launch` recipe in
[`references/training.md`](references/training.md). For
fine-tuning add `--pretrained_path
/path/to/pretrained_harmonizer.pkl` and use
`--fixing_data_weight 3` on the released dataset.

### Example 4 — Temporal-aware enhancement

Use `src/inference_pix2pix_turbo_harmonizer.py` when the user
explicitly asks for temporal conditioning rather than the
standard pretrained model script. Full command +
`--offset_list` defaults in
[`references/inference.md`](references/inference.md).

## Consuming the inline Fixer via NRE

`nre --enable-difix` is still the right answer when the user
wants NRE to enhance frames as part of rendering without a
separate harmonizer checkout. That path is owned by the sibling
`nre` skill. Do not mix the standalone DiffusionHarmonizer
HF/Cosmos workflow with NRE's internal cache flags unless the
NRE documentation for the user's tag explicitly says they share
weights.

Use this standalone skill when the user wants the public
DiffusionHarmonizer code, model card, training/evaluation
scripts, or post-processing of frames that already exist on
disk.

## Prerequisites

- **OS:** Linux host.
- **GPU / driver:** NVIDIA GPU Ampere or newer (compute
  capability `>= 8.0`; A100, A10, L40, H100, RTX 30/40/PRO,
  B200, GB200).
- **Container runtime:** Docker with the NVIDIA Container
  Toolkit.
- **Tools:** `git`, `python3`, Hugging Face CLI (`hf` or
  `huggingface-cli`).
- **Secrets:**
  - `HF_TOKEN` with the
    [`nvidia/DiffusionHarmonizer`](https://huggingface.co/nvidia/DiffusionHarmonizer)
    license accepted (required to download model weights and
    the optional dataset).
  - `NGC_API_KEY` (often required for `docker login nvcr.io`
    before pulling `nvcr.io/nvidia/cosmos/...`).
- **Disk:** at least ~120 GB free for the Cosmos image, build
  cache, model weights, optional dataset, and outputs combined.
- **Source / model / dataset:**
  - Code: <https://github.com/NVIDIA/harmonizer>.
  - Model: <https://huggingface.co/nvidia/DiffusionHarmonizer>.
  - Optional dataset:
    <https://huggingface.co/datasets/nvidia/DiffusionHarmonizer-Dataset>.

The fail-fast check that enforces all of the above is
`scripts/validate_setup.py`.

## Scripts

| Script | Purpose | Usage |
|--------|---------|-------|
| `scripts/validate_setup.py` | Verify Docker, NVIDIA Container Toolkit, GPU architecture, `git`, Hugging Face CLI, token presence, and disk space. No network calls. | `run_script("scripts/validate_setup.py")` or `python scripts/validate_setup.py` |
| `scripts/.env.example` | Template for `HF_TOKEN` and optional `NGC_API_KEY`. | `cp scripts/.env.example .env && set -a && . ./.env && set +a` |

## References

- [`references/inference.md`](references/inference.md) — container
  build, raw-Cosmos fallback, Blackwell patches, model download,
  `inference_pretrained_model.py` flag matrix, temporal variant.
- [`references/evaluation.md`](references/evaluation.md) — paired
  `test_dataset/` layout and `evaluate_test_dataset.py` command.
- [`references/training.md`](references/training.md) — dataset
  download, training JSON format, multi-GPU `accelerate launch`
  recipe, fine-tuning flags, NuRec data-pair recipes.
- [`references/wrapper-image.md`](references/wrapper-image.md) —
  build and run the project image for repeat inference.
- [`references/troubleshooting.md`](references/troubleshooting.md)
  — extended diagnostic notes.
- [`references/teardown.md`](references/teardown.md) — cleanup
  inventory for images, code, Hugging Face caches, datasets,
  and outputs.
- Public code: <https://github.com/NVIDIA/harmonizer>
- Model card: <https://huggingface.co/nvidia/DiffusionHarmonizer>
- Dataset: <https://huggingface.co/datasets/nvidia/DiffusionHarmonizer-Dataset>
- Paper: <https://arxiv.org/abs/2602.24096>
- Project page: <https://research.nvidia.com/labs/sil/projects/diffusion-harmonizer/>

## Limitations

- **Rendered inputs only.** The model is tuned for
  neural-reconstruction renderings and object-insertion
  artifacts, not arbitrary real photos.
- **Primary model resolution is 576x1024.** The public script
  maps resolution key `1024` to `1024x576` and restores outputs
  to the original image size. Other keys are script-supported
  but may not match the model-card operating point.
- **Filename ordering matters.** The standard inference script
  sorts filenames as strings. Use zero-padded frame numbers.
- **Container builds can be large.** The Cosmos environment,
  build cache, model weights, and optional dataset can exceed
  100 GB.
- **Training is multi-GPU by default.** The README command
  assumes 8 GPUs with bf16 mixed precision.
- **Public code evolves.** If a checkout uses the older
  `inference_pretrained_model_harmonizer.py` name, prefer the
  script present in that checkout; the fixed release branch
  standardises on `inference_pretrained_model.py`.

## Troubleshooting (top 5)

| Error / symptom | Most common cause |
|-----------------|-------------------|
| `docker: could not select device driver ... gpu` | NVIDIA Container Toolkit missing or Docker is not configured for the NVIDIA runtime. |
| `docker pull` 401 / 403 from `nvcr.io` | Docker is not authenticated to NGC, or the API key lacks container access. |
| `hf download ... 401 / 403` | `HF_TOKEN` is missing/expired/lacks read scope, or the model/dataset license has not been accepted. |
| `pretrained_harmonizer.pkl` missing | Model download path is wrong or incomplete. Re-run `hf download nvidia/DiffusionHarmonizer --local-dir models`. |
| Output files owned by `root` | The `docker run` omitted `-u $(id -u):$(id -g)`. |

Full matrix in
[`references/troubleshooting.md`](references/troubleshooting.md).

## Teardown

A full workflow can leave large artifacts on disk: the Cosmos
image, project image, build cache, `harmonizer` code checkout,
Hugging Face model weights, optional dataset, evaluation
outputs, and enhanced frames. Reclaim them with the inventory
in [`references/teardown.md`](references/teardown.md). Do not
revoke `HF_TOKEN` or `NGC_API_KEY` as normal cleanup. Rotate a
token only if you suspect it was leaked.
