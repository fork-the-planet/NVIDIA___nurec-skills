---
name: nurec-fixer
description: >-
  Use when the user wants to enhance, harmonize, evaluate, or fine-tune
  rendered novel-view images from neural reconstructions by running
  NVIDIA DiffusionHarmonizer, the public successor to the older Fixer
  recipes. Covers cloning the public NVIDIA/harmonizer code, building
  the Cosmos Predict2 environment, downloading the Hugging Face model
  `nvidia/DiffusionHarmonizer` to obtain
  `models/pretrained/pretrained_harmonizer.pkl`, running inference with
  `src/inference_pretrained_model.py`, preparing paired train/test data,
  evaluating PSNR/LPIPS with `evaluate_test_dataset.py`, and optional
  fine-tuning. Do NOT use for training the 3D reconstruction itself
  (use `nre`) or for converting sensor data (use `ncore`). Trigger
  keywords: nurec fixer, nurec harmonizer, diffusion harmonizer,
  DiffusionHarmonizer, NVIDIA Harmonizer, nvidia/DiffusionHarmonizer,
  pretrained_harmonizer.pkl, harmonizer-cosmos-env,
  cosmos-predict2-container, inference_pretrained_model.py,
  train_pix2pix_turbo_harmonizer.py, evaluate_test_dataset.py,
  harmonize rendered frames, fix reconstruction artifacts, online
  diffusion enhancer, inserted object harmonization, PSNR, LPIPS.
version: "0.4.0"
tools:
  - Shell
  - Read
  - Write
license: CC-BY-4.0 AND Apache-2.0
compatibility: >-
  Linux host with an NVIDIA GPU of Ampere architecture or newer
  (compute capability >= 8.0; A100, A10, L40, H100, RTX 30/40/PRO,
  B200, GB200). Docker with NVIDIA Container Toolkit. Public code from
  https://github.com/NVIDIA/harmonizer, public model weights from
  https://huggingface.co/nvidia/DiffusionHarmonizer, and optional
  dataset from https://huggingface.co/datasets/nvidia/DiffusionHarmonizer-Dataset.
  The documented runtime uses the Cosmos Predict2 container
  `nvcr.io/nvidia/cosmos/cosmos-predict2-container:1.2` or the
  project Dockerfile that layers the required packages and patches.
dependencies:
  - bash
  - docker
  - nvidia-container-toolkit
  - python3
  - git
  - huggingface_hub
metadata:
  author: NVIDIA NuRec
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
  source_branches_checked:
    - alex/harmonizner@54596de
    - lesliem/harmonizer/update-model-card@2422370
    - lesliem/harmonizer/fix-inconsistencies@3616343
---

# NVIDIA DiffusionHarmonizer (NuRec Post-Processing)

## Purpose

Run NVIDIA DiffusionHarmonizer on rendered images from neural
reconstructions. DiffusionHarmonizer is a single-step, temporally-aware
image diffusion enhancer for NeRF / 3DGS / NuRec-style renderings. It
improves realism, reduces reconstruction artifacts, and harmonizes
inserted dynamic objects with the surrounding scene.

**Use this skill when:** the user has rendered frames from NRE, NuRec,
3D Gaussian Splatting, NeRF, or a similar reconstruction pipeline and
wants to enhance, harmonize, evaluate, or optionally fine-tune the
DiffusionHarmonizer model.

**Do NOT use this skill when:**

- The user wants to train or render the 3D reconstruction itself. Use
  the sibling `nre` skill.
- The user wants to convert raw sensor data to NCore V4. Use `ncore`.
- The user wants a generic photo enhancer. DiffusionHarmonizer is tuned
  for neural-reconstruction artifacts and object-insertion failures.
- The user only wants NRE inline rendering with `--enable-difix`. That
  remains an NRE runtime feature; use the `nre` skill for the complete
  `serve-grpc` / `render-grpc` command shape.

## What Changed From The Older Fixer Skill

This skill now follows the newer DiffusionHarmonizer release branches,
not the older NGC JIT `.pt` artifact recipe. Use these public release
artifacts:

- Code: `https://github.com/NVIDIA/harmonizer`
- Model: `nvidia/DiffusionHarmonizer` on Hugging Face
- Main checkpoint: `models/pretrained/pretrained_harmonizer.pkl`
- Runtime: Cosmos Predict2 environment, usually
  `nvcr.io/nvidia/cosmos/cosmos-predict2-container:1.2`
- Inference entry point: `src/inference_pretrained_model.py`
- Evaluation entry point: `src/evaluate_test_dataset.py`
- Training entry point: `src/train_pix2pix_turbo_harmonizer.py`

Do not use the obsolete standalone recipe that downloads
`nvidia/nre/nurec-fixer:cosmos_3dgut_fixer_harmonizer`, mounts
`harmonizer_temporal.pt`, or runs `inference_jit_harmonizer.py` inside
`nvcr.io/nvidia/pytorch:24.10-py3` unless the user explicitly asks for
that older beta artifact.

## Background

DiffusionHarmonizer is described in
[DiffusionHarmonizer: Bridging Neural Reconstruction and Photorealistic
Simulation with Online Diffusion Enhancer](https://arxiv.org/abs/2602.24096),
CVPR 2026. It distills a pretrained multi-step diffusion model into a
single-step enhancer designed for online simulation and offline data
cleanup.

The release describes two operating modes:

- **Offline mode:** clean pseudo-training views rendered from a
  reconstruction, then distill the improved views back into the 3D
  representation.
- **Online mode:** enhance frames during simulation/inference by
  harmonizing color and lighting, reconstructing missing or inconsistent
  shadows for inserted actors, and reducing residual reconstruction
  artifacts.

The public model card describes `DiffusionHarmonizer-cosmos-0.6B`, a
Cosmos Predict2 Diffusion Transformer post-trained at `576x1024` input
and output resolution. It is intended for Physical AI developers working
on autonomous-vehicle neural-reconstruction simulation.

## Inputs

- **code_dir** — checkout of `https://github.com/NVIDIA/harmonizer`.
- **model_dir** — directory downloaded from Hugging Face. After
  `hf download nvidia/DiffusionHarmonizer --local-dir models`, the main
  checkpoint is `models/pretrained/pretrained_harmonizer.pkl`.
- **input_dir** — directory of rendered RGB frames (`.png`, `.jpg`, or
  `.jpeg`) to enhance.
- **output_dir** — directory where enhanced frames are written.
- **HF_TOKEN** — Hugging Face token with access to the model and, if
  used, the dataset. Accept the model/dataset license terms before
  downloading gated artifacts.
- **NGC_API_KEY** — often needed to authenticate `docker pull` from
  `nvcr.io`, depending on site policy. Use it only for container pulls,
  not for model download.

## Instructions

1. Run `python scripts/validate_setup.py` from this skill to check
   Docker, NVIDIA Container Toolkit, GPU architecture, `git`,
   Hugging Face CLI availability, token presence, and disk space.
2. Clone the public code and build the runtime image:

   ```bash
   git clone https://github.com/NVIDIA/harmonizer.git
   cd harmonizer
   docker build -t harmonizer-cosmos-env -f Dockerfile.cosmos .
   ```

   If using the raw Cosmos Predict2 container instead of the project
   image, pin `nvcr.io/nvidia/cosmos/cosmos-predict2-container:1.2` and
   apply the project patches documented below.
3. Download the pretrained model:

   ```bash
   export HF_TOKEN=<your-hugging-face-token>
   hf auth login --token "$HF_TOKEN"
   hf download nvidia/DiffusionHarmonizer --local-dir models
   ```

   Verify `models/pretrained/pretrained_harmonizer.pkl` exists before
   running inference.
4. Confirm `input_dir` exists and filenames sort into the intended
   frame order. The current public inference script uses normal string
   sorting, so prefer zero-padded frame names such as `frame_000001.png`.
5. Run inference inside the container with input, output, and model
   paths mounted. Pin `-u $(id -u):$(id -g)` so outputs are owned by
   the host user.
6. Validate that the output frame count matches the input frame count,
   spot-check frames visually, and optionally run paired evaluation
   with PSNR/LPIPS if ground truth is available.
7. If the user wants to train or fine-tune the harmonizer, download the
   dataset or prepare JSON manifests in the documented format, then run
   `src/train_pix2pix_turbo_harmonizer.py` with the recommended
   hyperparameters.
8. Follow `references/teardown.md` when the workflow is done to remove
   images, code clones, model weights, datasets, and outputs.

## Container Setup

The release branches document the Cosmos Predict2 environment. The
preferred external path is to clone the public code and build the image
from its `Dockerfile.cosmos`:

```bash
CODE_DIR=$PWD/harmonizer
git clone https://github.com/NVIDIA/harmonizer.git "$CODE_DIR"
cd "$CODE_DIR"

docker build \
  -t harmonizer-cosmos-env \
  -f Dockerfile.cosmos \
  .
```

If the project Dockerfile is unavailable in a particular release
checkout, run from the pinned public Cosmos Predict2 image and install
only the packages listed by that checkout:

```bash
BASE=nvcr.io/nvidia/cosmos/cosmos-predict2-container:1.2

docker run --gpus=all --rm -it --ipc=host \
  -u "$(id -u):$(id -g)" \
  -v "$CODE_DIR":/work \
  -w /work \
  "$BASE" \
  bash
```

For Blackwell GPUs such as B200 or GB200, the README branch calls out a
Text2ImageDIT patch before inference when using the raw container:

```bash
patch /usr/local/lib/python3.12/dist-packages/cosmos_predict2/models/text2image_dit.py text2image_dit.patch
```

For training, apply the tokenizer patch as well if the project image did
not already apply it:

```bash
patch /usr/local/lib/python3.12/dist-packages/cosmos_predict2/tokenizers/tokenizer.py tokenizer.patch
```

The project image should bake these patches in. Only apply them manually
when running directly inside a raw base container.

## Model Download

Install and authenticate the Hugging Face CLI, then download the model:

```bash
python3 -m pip install --user "huggingface_hub[cli]"
export HF_TOKEN=<your-hugging-face-token>
hf auth login --token "$HF_TOKEN"
hf download nvidia/DiffusionHarmonizer --local-dir models
```

Expected checkpoint:

```bash
ls -lh models/pretrained/pretrained_harmonizer.pkl
```

Never commit `models/`, the downloaded checkpoint, or `HF_TOKEN`.

## Run Inference

Use the public inference entry point from the fixed README branch:
`src/inference_pretrained_model.py`.

```bash
IMAGE=harmonizer-cosmos-env
CODE_DIR=/absolute/path/to/harmonizer
MODEL_PATH=/work/models/pretrained/pretrained_harmonizer.pkl
INPUT_DIR=/absolute/path/to/rendered_frames
OUTPUT_DIR=/absolute/path/to/enhanced_frames

mkdir -p "$OUTPUT_DIR"

docker run --gpus=all --rm --ipc=host \
  -u "$(id -u):$(id -g)" \
  -v "$CODE_DIR":/work \
  -v "$INPUT_DIR":/input:ro \
  -v "$OUTPUT_DIR":/output \
  -w /work \
  "$IMAGE" \
  python /work/src/inference_pretrained_model.py \
    --model "$MODEL_PATH" \
    --input /input \
    --output /output \
    --timestep 250 \
    --resolution 1024
```

Useful flags exposed by the public inference script:

| Flag | Default | Purpose |
|------|---------|---------|
| `--model` | required | Path to `pretrained_harmonizer.pkl`. |
| `--input` | required | Directory of input frames. |
| `--output` | `output` | Directory for enhanced frames. |
| `--timestep` | script default may differ; README uses `250` | Diffusion timestep used by the distilled model. |
| `--resolution` | `1024` | Internal size key. `1024` maps to `1024x576`. Other script-supported keys include `960`, `1360`, `704`, `512`, `256`, and `1920`. |
| `--batch_size` | `8` for benchmarking | Batch size for speed testing. The image generation path currently processes frames one at a time. |
| `--max_frames` | very large | Maximum number of frames to process. |
| `--skip_frames` | `1` | Process every Nth frame. |
| `--save_video` | off | Save an MP4 from the output folder. |
| `--test-speed` | off | Run the built-in speed benchmark. |

The branch model card reports H100 testing and an inference time of
about `212 ms` on one H100. The code also has a local speed-test mode;
trust the measurement from your target hardware for capacity planning.

## Temporal Variant

The newer branches also include `src/inference_pix2pix_turbo_harmonizer.py`,
which runs a temporal/autoregressive variant using prior enhanced frames
as references. Use it when the user explicitly asks for temporal
conditioning rather than the standard pretrained model script.

Key differences:

- It accepts `--input_image`, `--model_path`, `--model_identifier`,
  `--offset_list`, and `--nontemporal`.
- It writes to a sibling output folder named
  `<input_dir>_<model_identifier>` instead of taking `--output`.
- The default `--offset_list -1 -2 -3 -4` uses the four previous
  enhanced frames as temporal references once enough history exists.
- Early frames fall back to a non-temporal pass.

Example:

```bash
docker run --gpus=all --rm --ipc=host \
  -u "$(id -u):$(id -g)" \
  -v "$CODE_DIR":/work \
  -v "$INPUT_DIR":/input:ro \
  -w /work \
  harmonizer-cosmos-env \
  python /work/src/inference_pix2pix_turbo_harmonizer.py \
    --model_path /work/models/pretrained/pretrained_harmonizer.pkl \
    --input_image /input \
    --model_identifier harmonized \
    --timestep 250
```

If reproducibility matters, keep each run's output folder separate so
old temporal outputs cannot be reused accidentally.

## Dataset And Training

The release branches describe a training set of approximately 350K
curated synthetic-real image pairs from five complementary curation
pipelines. Download the assembled dataset when the user wants to train,
fine-tune, or reproduce evaluation:

```bash
hf download nvidia/DiffusionHarmonizer-Dataset \
  --repo-type dataset \
  --local-dir data
```

Data sources and targeted failure modes:

| Data source | Failure mode |
|-------------|--------------|
| ISP Modification | ISP-induced color or tone drift between foreground and background. |
| Relighting | Illumination mismatch between inserted objects and scene lighting. |
| Asset Re-insertion | Missing shadows and appearance mismatch when dynamic assets are re-inserted. |
| PBR Shadow Simulation | Missing or unrealistic cast shadows on inserted objects. |
| Artifacts Correction | Novel-view artifacts such as blur, missing regions, ghosting, and spurious geometry. |

Training JSON format:

```json
{
  "train": {
    "{data_id}": {
      "image": "{PATH_TO_IMAGE}",
      "target_image": "{PATH_TO_TARGET_IMAGE}",
      "prompt": "remove degradation"
    }
  },
  "test": {
    "{data_id}": {
      "image": "{PATH_TO_IMAGE}",
      "target_image": "{PATH_TO_TARGET_IMAGE}",
      "prompt": "remove degradation"
    }
  }
}
```

Recommended multi-GPU training shape from the release README:

```bash
export NUM_NODES=1
export NUM_GPUS=8
export OUTPUT_DIR=/path/to/checkpointing_directory
export DATASET_FOLDER=/data/data.json
export WANDB_MODE=offline

accelerate launch \
  --mixed_precision=bf16 \
  --main_process_port 29501 \
  --multi_gpu \
  --num_machines "$NUM_NODES" \
  --num_processes "$NUM_GPUS" \
  src/train_pix2pix_turbo_harmonizer.py \
    --output_dir="${OUTPUT_DIR}" \
    --dataset_folder="${DATASET_FOLDER}" \
    --max_train_steps 10000 \
    --learning_rate 2e-5 \
    --train_batch_size=1 \
    --gradient_accumulation_steps 1 \
    --dataloader_num_workers 8 \
    --checkpointing_steps=2000 \
    --eval_freq 1000 \
    --viz_freq 1000 \
    --train_image_prep resize_576x1024 \
    --test_image_prep resize_576x1024 \
    --lambda_clipsim 0.0 \
    --lambda_lpips 0.3 \
    --lambda_gan 0.0 \
    --lambda_l2 1.0 \
    --lambda_gram 0.0 \
    --use_sched \
    --report_to wandb \
    --tracker_project_name cosmos_harmonizer \
    --tracker_run_name train \
    --train_full_unet \
    --timestep 250 \
    --track_val_fid \
    --num_samples_eval 20 \
    --mixed_precision=bf16
```

For fine-tuning from the released checkpoint, add:

```bash
--pretrained_path /path/to/pretrained_harmonizer.pkl
```

For the released dataset, the README recommends `--fixing_data_weight 3`
to up-weight artifact-correction examples.

## NuRec Data Pair Recipes

The updated tutorials describe four ways to construct image pairs for
training or testing. Three are shown with NuRec command templates:

- **Sparse reconstruction:** train with every Nth frame and pair held-out
  ground-truth images with rendered novel views.
- **Cycle reconstruction:** listed as a supported pair-generation method
  in the tutorial overview.
- **Model underfitting:** train a reconstruction for a reduced schedule
  (roughly 25%-75%) and pair degraded renders with clean targets.
- **Cross reference:** train using one camera and render/evaluate held-out
  cameras.

When adapting the sample NRE commands, keep public container names and
current NRE recipes from the sibling `nre` skill. Do not copy internal
container names or internal dataset paths into user-facing instructions.

## Evaluation

For quantitative PSNR / LPIPS evaluation, prepare paired data with this
layout:

```text
test_dataset/
├── {scene_id_1}/
│   ├── render/
│   │   ├── {camera_id_1}/
│   │   │   ├── {timestamp_1}.png
│   │   │   └── {timestamp_2}.png
│   │   └── {camera_id_2}/
│   └── gt/
│       ├── {camera_id_1}/
│       │   ├── {timestamp_1}.png
│       │   └── {timestamp_2}.png
│       └── {camera_id_2}/
└── {scene_id_2}/
    ├── render/
    └── gt/
```

`render/` and `gt/` must have identical camera subdirectories and
matching filenames. Images may be PNG, JPEG, or JPG.

Run evaluation:

```bash
docker run --gpus=all --rm --ipc=host \
  -u "$(id -u):$(id -g)" \
  -v "$CODE_DIR":/work \
  -v /absolute/path/to/test_dataset:/test_dataset \
  -w /work \
  harmonizer-cosmos-env \
  python /work/src/evaluate_test_dataset.py \
    --model /work/models/pretrained/pretrained_harmonizer.pkl \
    --input /test_dataset
```

Expected outputs:

- Enhanced images under an `evaluation/` directory that mirrors the test
  dataset structure.
- `metrics.yaml` with overall and per-scene PSNR/LPIPS, inference time,
  image counts, and GPU memory statistics.

## Consuming Inline Fixer Via NRE

`nre --enable-difix` is still the right answer when the user wants NRE
to enhance frames as part of rendering without a separate harmonizer
checkout. That path is owned by the sibling `nre` skill. Do not mix the
standalone DiffusionHarmonizer HF/Cosmos workflow with NRE's internal
cache flags unless the NRE documentation for the user's tag explicitly
says they share weights.

Use this standalone skill when the user wants the public
DiffusionHarmonizer code, model card, training/evaluation scripts, or
post-processing of frames that already exist on disk.

## Scripts

| Script | Purpose | Usage |
|--------|---------|-------|
| `scripts/validate_setup.py` | Verify Docker, NVIDIA Container Toolkit, GPU architecture, `git`, Hugging Face CLI, token presence, and disk space. Makes no network calls. | `python scripts/validate_setup.py` |
| `scripts/.env.example` | Template for `HF_TOKEN` and optional `NGC_API_KEY`. | `cp scripts/.env.example .env && set -a && . ./.env && set +a` |

## References

- `references/wrapper-image.md` — build and run the project image for
  repeat inference.
- `references/troubleshooting.md` — common setup, inference, training,
  and evaluation failures.
- `references/teardown.md` — cleanup inventory for images, code,
  Hugging Face caches, datasets, and outputs.
- Public code: <https://github.com/NVIDIA/harmonizer>
- Model card: <https://huggingface.co/nvidia/DiffusionHarmonizer>
- Dataset: <https://huggingface.co/datasets/nvidia/DiffusionHarmonizer-Dataset>
- Paper: <https://arxiv.org/abs/2602.24096>
- Project page: <https://research.nvidia.com/labs/sil/projects/diffusion-harmonizer/>

## Limitations

- **Rendered inputs only.** The model is tuned for neural-reconstruction
  renderings and object insertion artifacts, not arbitrary real photos.
- **Primary model resolution is 576x1024.** The public script maps
  resolution key `1024` to `1024x576` and restores outputs to the
  original image size. Other keys are script-supported but may not match
  the model-card operating point.
- **Filename ordering matters.** The standard inference script sorts
  filenames as strings. Use zero-padded frame numbers.
- **Container builds can be large.** The Cosmos environment, build cache,
  model weights, and optional dataset can exceed 100 GB.
- **Training is multi-GPU by default.** The README command assumes 8 GPUs
  with bf16 mixed precision. Scale carefully and expect to tune batch,
  accumulation, and worker counts.
- **Public code evolves.** If a checkout uses the older
  `inference_pretrained_model_harmonizer.py` name, prefer the script
  present in that checkout, but the fixed release branch standardizes on
  `inference_pretrained_model.py`.

## Troubleshooting

| Error / symptom | Most common cause |
|-----------------|-------------------|
| `docker: could not select device driver ... gpu` | NVIDIA Container Toolkit is missing or Docker is not configured for the NVIDIA runtime. |
| `docker pull` 401 / 403 from `nvcr.io` | Docker is not authenticated to NGC, or the API key lacks container access. |
| `hf download ... 401 / 403` | `HF_TOKEN` is missing, expired, lacks read scope, or the model/dataset license has not been accepted. |
| `pretrained_harmonizer.pkl` missing | Model download path is wrong or incomplete. Re-run `hf download nvidia/DiffusionHarmonizer --local-dir models`. |
| `No such file: inference_pretrained_model_harmonizer.py` | The fixed branch uses `src/inference_pretrained_model.py`. Use the script name in the checkout. |
| Patch fails on Blackwell | The patch was already applied by the Dockerfile, or the installed Cosmos package version differs from the patch target. |
| Output files owned by `root` | The `docker run` omitted `-u $(id -u):$(id -g)`. |
| Evaluation finds no pairs | `render/` and `gt/` structures or filenames do not match exactly. |
| `CUDA out of memory` | Resolution, batch size, TensorRT compilation, or training configuration exceeds GPU memory. Lower resolution/batch or use a larger GPU. |

Full diagnostic notes live in `references/troubleshooting.md`.

## Teardown

A full workflow can leave large artifacts on disk: the Cosmos image,
project image, build cache, `harmonizer` code checkout, Hugging Face
model weights, optional dataset, evaluation outputs, and enhanced
frames. Reclaim them with the inventory in `references/teardown.md`.

Do not revoke `HF_TOKEN` or `NGC_API_KEY` as normal cleanup. Rotate a
token only if you suspect it was leaked.
