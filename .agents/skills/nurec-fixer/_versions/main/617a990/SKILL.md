---
name: nurec-fixer
description: >-
  Use when a user wants to enhance, harmonize, or temporally stabilize
  novel-view renders produced by a 3D reconstruction / neural renderer
  (especially NRE / NuRec outputs) by running NVIDIA Fixer — a
  single-step diffusion model (based on Difix3D+) that removes
  artifacts caused by underconstrained regions of the 3D representation.
  Covers fetching pretrained weights from HuggingFace nvidia/Fixer,
  preparing the cosmos-predict2 container environment, running
  per-sequence inference on a directory of rendered frames, and
  evaluating results with PSNR / LPIPS on paired render↔ground-truth
  test sets. Use only for post-processing rendered images from an
  external 3D reconstruction pipeline — not for generating or training
  the reconstruction itself. Trigger keywords: nurec fixer, nvidia
  fixer, unified fixer, harmonizer, difix, difix3d, difix3d+,
  pretrained_fixer.pkl, nvidia/Fixer, fix rendered frames, harmonize
  rendered frames, remove reconstruction artifacts, temporal
  stabilization, post-process novel views, cosmos fixer,
  cosmos-predict2-container, inference_pretrained_model.py,
  evaluate_test_dataset.py, novel view enhancement, 3D reconstruction
  refinement, NRE difix cache, --enable-difix.
version: "0.1.0"
author: NVIDIA NuRec
tags:
  - nurec
  - fixer
  - reconstruction
  - autonomous-vehicles
  - post-processing
tools:
  - Shell
  - Read
  - Write
license: Apache-2.0 OR CC-BY-4.0
compatibility: >-
  Linux host with an NVIDIA GPU of Ampere architecture or newer
  (compute capability >= 8.0; A100, A10, L40, H100, RTX 30/40/PRO,
  B200, GB200). Requires recent NVIDIA driver compatible with CUDA 12,
  NVIDIA Container Toolkit, Docker (or Podman). ~16 GB VRAM
  recommended for the pretrained Fixer at 1024x576. Internet egress
  required to pull the cosmos-predict2 container from nvcr.io and
  pretrained weights from huggingface.co. Credentials via env vars
  only: HF_TOKEN (HuggingFace, required for weight download),
  NGC_API_KEY (optional, only if your NGC account gates the
  container). Scripts make network calls to huggingface.co and
  nvcr.io.
dependencies:
  - bash
  - docker
  - nvidia-container-toolkit
  - python3
metadata:
  upstream: https://github.com/nv-tlabs/Difix3D
  hf_models: https://huggingface.co/nvidia/Fixer
  ngc_container: nvcr.io/nvidia/cosmos/cosmos-predict2-container:1.2
  paper: https://arxiv.org/abs/2503.01774
  product_page: https://research.nvidia.com/labs/toronto-ai/difix3d/
---

# NVIDIA Fixer (NuRec Post-Processing)

Single-step image diffusion model that removes artifacts from novel-view
renders produced by 3D reconstruction pipelines (Gaussian Splatting,
NeRF, NRE / NuRec, and similar). Based on Difix3D+
(see [paper](https://arxiv.org/abs/2503.01774) and the open-source
[Difix3D+ repository](https://github.com/nv-tlabs/Difix3D)). The
pretrained Fixer checkpoint is published on HuggingFace at
[`nvidia/Fixer`](https://huggingface.co/nvidia/Fixer) and runs inside
the public
[`nvcr.io/nvidia/cosmos/cosmos-predict2-container:1.2`](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/cosmos/containers/cosmos-predict2-container)
container.

Fixer is typically chained **after** a NuRec / NRE reconstruction
pipeline: the reconstruction renders novel views from held-out camera
poses, and Fixer cleans up the rendered frames to improve perceptual
quality, temporal coherence, and downstream metrics (PSNR, LPIPS).

## When to Use

- User has a directory of rendered frames from a 3D reconstruction
  (NRE, NuRec, Gaussian Splatting, NeRF, etc.) and asks to remove
  artifacts, enhance, harmonize, or "fix" them.
- User mentions running the NVIDIA Fixer model, Difix3D+, or
  `pretrained_fixer.pkl` on novel-view renders.
- User has paired render↔ground-truth images and wants to evaluate
  fidelity (PSNR / LPIPS) of their reconstruction after post-processing.
- User mentions `--enable-difix` in an NRE pipeline and asks what
  weights / cache to supply.
- User mentions `nvidia/Fixer` on HuggingFace and wants an end-to-end
  inference recipe.

### When NOT to Use

- The user wants to **train or run the 3D reconstruction itself**
  (e.g. multi-camera NRE reconstruction, Gaussian Splatting training).
  Use the sibling [`nre`](../nre/SKILL.md) skill instead; Fixer only
  post-processes renders.
- The user wants Fixer **inline with NRE rendering** (single pipeline,
  no separate container). Use NRE's built-in
  `--enable-difix` flag with `difix=cosmos_difix` (default since
  25.09) or `difix=sd_difix` for the legacy Stable-Diffusion variant.
  Reach for this standalone skill only when you need to swap model
  variants, post-process frames that were rendered earlier, or run
  PSNR / LPIPS evaluation on a paired test set.
- The user wants to train or fine-tune a new Fixer model. Fine-tuning
  requires the internal training codebase and is not covered by this
  external-facing skill.
- The user wants to apply Fixer to real (non-rendered) camera images —
  the model is trained specifically on reconstruction artifacts.

## Inputs

- **input_dir** — directory of rendered images (`.png`, `.jpg`,
  `.jpeg`) to enhance. Filenames MUST natural-sort into temporal
  order for sequence consistency (source: user prompt; required).
- **output_dir** — directory for enhanced images. MUST be empty or
  non-existent (source: user prompt; required).
- **model_path** — path to the pretrained Fixer checkpoint
  `pretrained_fixer.pkl` (source: user prompt or agent context;
  required; default: `./models/pretrained/pretrained_fixer.pkl` after
  running the download step).
- **timestep** — diffusion timestep for single-step inference (source:
  user prompt; optional; default: `250`).
- **test_dataset_dir** — only when evaluating: directory containing
  `render/` and `gt/` subtrees per scene (source: user prompt;
  required for evaluation; see [test dataset layout](references/test-dataset-layout.md)).
- **HF_TOKEN** — HuggingFace auth token, set as an environment
  variable on the host (source: agent context; required for weight
  download; create at https://huggingface.co/settings/tokens).

## Instructions

1. Run [`scripts/validate_setup.py`](scripts/validate_setup.py) to
   confirm the host has Docker, the NVIDIA Container Toolkit, a GPU
   with compute capability ≥ 8.0, `HF_TOKEN` exported, and enough
   free disk for the container (~40 GB) plus weights (~6 GB). Abort
   on any failure.
2. Pull the container `nvcr.io/nvidia/cosmos/cosmos-predict2-container:1.2`
   if it is not already present (see
   [Container setup](#container-setup)).
3. Download the pretrained Fixer weights from
   [`nvidia/Fixer`](https://huggingface.co/nvidia/Fixer) into a host
   directory (typically `./models/`) using the HuggingFace CLI. See
   [Weight download](#weight-download).
4. Confirm `input_dir` exists, contains supported images, that
   filenames natural-sort into the intended order, and that
   `output_dir` is empty or non-existent. Fail fast otherwise.
5. Run the inference container with `input_dir`, `output_dir`, and
   the weights directory mounted. See
   [Run inference](#run-inference).
6. Validate that the number of output images equals the number of
   input images and spot-check a frame visually.
7. **If evaluating:** prepare a paired test dataset per
   [`references/test-dataset-layout.md`](references/test-dataset-layout.md)
   and run `evaluate_test_dataset.py` inside the container to produce
   `metrics.yaml` with PSNR and LPIPS.
8. When consuming Fixer as part of an NRE pipeline, prefer passing
   `--enable-difix --difix-cache=<path>` to the NRE entrypoint
   instead of running Fixer manually — NRE will download the same
   weights into the cache directory and call the model internally.

## Output Format

Per-frame enhanced images land in `output_dir` with the same basenames
and file extensions as the inputs:

```text
<output_dir>/
├── <frame-0>.png        # or .jpg, matching input
├── <frame-1>.png
└── ...
```

Images are restored to each input's original `(width, height)` after
the model runs at its internal 1024×576 working resolution.

When evaluating, `evaluate_test_dataset.py` additionally writes
`metrics.yaml` at the run root:

```yaml
overall:
  psnr_mean: <float>
  psnr_std: <float>
  lpips_mean: <float>
  lpips_std: <float>
  avg_inference_time: <seconds>
  total_images: <int>
per_scene:
  <scene_id>:
    psnr: <float>
    lpips: <float>
    num_images: <int>
    avg_inference_time: <seconds>
```

## Prerequisites

- Linux host with NVIDIA driver supporting CUDA 12.x.
- Docker (or Podman) + [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
- NVIDIA GPU with compute capability ≥ 8.0 (Ampere or newer).
  ~16 GB VRAM recommended.
- Internet egress to `nvcr.io` and `huggingface.co`.
- HuggingFace account and access token exported as `HF_TOKEN`.
  Create at https://huggingface.co/settings/tokens. No license
  acceptance is currently gated on the `nvidia/Fixer` model page,
  but confirm the card terms before use.
- ~40 GB free disk for the container image and ~6 GB for weights.

## Container setup

Pull the cosmos-predict2 base container (the same image used by the
Fixer release toolchain):

```bash
IMAGE=nvcr.io/nvidia/cosmos/cosmos-predict2-container:1.2
docker image inspect "$IMAGE" >/dev/null 2>&1 || docker pull "$IMAGE"
```

This container ships PyTorch 2.x, `cosmos_predict2`, NVIDIA
Transformer Engine, and the CUDA / cuDNN stack Fixer needs. No
custom image build is required for external inference-only use.

If a site policy requires a mirrored / hardened variant of the
container, consult your NGC account and substitute the image tag.
Do not pin to `:latest` — pin to the dated minor release shown
above for reproducibility.

## Weight download

Export your HuggingFace token, then fetch the pretrained weights into
a host-side `models/` directory (the container will mount this
read-only at inference time):

```bash
export HF_TOKEN=<your-huggingface-token>     # create at https://huggingface.co/settings/tokens
pip install --quiet 'huggingface_hub[cli]'
huggingface-cli login --token "$HF_TOKEN"
huggingface-cli download nvidia/Fixer --local-dir ./models
```

After the download completes, verify the checkpoint exists:

```bash
ls -lh ./models/pretrained/pretrained_fixer.pkl
```

The checkpoint is ~1.5 GB. If the file is much smaller, the download
was truncated — delete and retry. Never commit the checkpoint to
version control and never embed `HF_TOKEN` in a config file.

## Run inference

A single `docker run --rm` invocation handles the end-to-end
inference. Bind-mount the weights (read-only), the input frames
directory (read-only), and the output directory (read-write):

```bash
IMAGE=nvcr.io/nvidia/cosmos/cosmos-predict2-container:1.2
INPUT_DIR=/absolute/path/to/rendered_frames
OUTPUT_DIR=/absolute/path/to/enhanced_frames
MODELS_DIR=$(pwd)/models

mkdir -p "$OUTPUT_DIR"
[ -z "$(ls -A "$OUTPUT_DIR" 2>/dev/null)" ] || {
  echo "ERROR: $OUTPUT_DIR is not empty."
  exit 1
}

docker run --gpus=all --rm --ipc=host \
  -v "$MODELS_DIR":/models:ro \
  -v "$INPUT_DIR":/in:ro \
  -v "$OUTPUT_DIR":/out \
  "$IMAGE" \
  python /opt/fixer/src/inference_pretrained_model.py \
    --model /models/pretrained/pretrained_fixer.pkl \
    --input /in \
    --output /out \
    --timestep 250
```

Notes:

- The script path inside the container (`/opt/fixer/src/...`) is
  illustrative. If your copy of the container ships the Fixer code
  at a different path (`/work/src/`, `/fixer-codebase/src/`, …),
  adjust the path or `docker cp` the corresponding `.py` file from
  the open-source [Difix3D+ repo](https://github.com/nv-tlabs/Difix3D)
  before running. See [Troubleshooting](#troubleshooting).
- `--ipc=host` is standard for NGC containers and harmless.
- Do not pass `torchrun` / `--nproc_per_node` — the inference script
  is single-process, single-GPU.

### Common inference flags

| Flag | Default | Purpose |
|------|---------|---------|
| `--model` | _(required)_ | Path to `pretrained_fixer.pkl` inside the container. |
| `--input` | _(required)_ | Directory of rendered frames. |
| `--output` | _(required)_ | Directory for enhanced frames. |
| `--timestep` | `250` | Diffusion timestep used for single-step inference. The default is the value the release recommends. |
| `--batch-size` | `1` | Inference batch size. Increase if VRAM allows. |
| `--export-path` | _(unset)_ | If set, exports the model to a reloadable directory instead of running per-frame inference. See [`references/export-and-reload.md`](references/export-and-reload.md). |

## Consuming Fixer via NRE (`--enable-difix`)

If you are already running an NRE reconstruction pipeline from
`nvcr.io/nvidia/nre/nre:<tag>` (or the GA-channel
`nvcr.io/nvidia/nre/nre-ga:<tag>`), the cleanest integration is
to let NRE manage Fixer for you. Pass the `--enable-difix` flag with
a cache directory on the host; NRE will download the weights into
the cache on first run (default `${HOME}/.cache/nre/difix/`) and
post-process renders internally. See the sibling
[`nre`](../nre/SKILL.md) skill (`serve-grpc --enable-difix` invocation
in the CLI cookbook) for the full pipeline command shape, and pick
between `difix=cosmos_difix` (default since 25.09) and
`difix=sd_difix` (legacy Stable-Diffusion variant) at the same
time.

When using this path, you do **not** need to run the steps in
[Weight download](#weight-download) or
[Run inference](#run-inference) above — NRE owns the whole cycle.

## Evaluate a paired test dataset

Prepare a dataset in the layout documented in
[`references/test-dataset-layout.md`](references/test-dataset-layout.md),
then evaluate:

```bash
docker run --gpus=all --rm --ipc=host \
  -v "$MODELS_DIR":/models:ro \
  -v /absolute/path/to/test_dataset:/data:ro \
  -v /absolute/path/to/eval_out:/eval \
  "$IMAGE" \
  python /opt/fixer/src/evaluate_test_dataset.py \
    --model /models/pretrained/pretrained_fixer.pkl \
    --input /data \
    --output /eval
```

The script iterates each `<scene_id>/render/<camera_id>/<frame>.png`,
runs Fixer, compares the output to the paired
`<scene_id>/gt/<camera_id>/<frame>.png`, and emits
`eval/metrics.yaml` with PSNR / LPIPS.

## Scripts

| Script | Purpose | Usage |
|--------|---------|-------|
| `scripts/validate_setup.py` | Verify host has Docker, NVIDIA Container Toolkit, a GPU with compute capability ≥ 8.0, `HF_TOKEN` exported, and sufficient free disk. Makes no network calls. | Execute: `python scripts/validate_setup.py` |

## References

- [`references/test-dataset-layout.md`](references/test-dataset-layout.md)
  — directory structure and naming conventions for paired
  render↔gt evaluation datasets.
- [`references/export-and-reload.md`](references/export-and-reload.md)
  — how to export the pretrained model once and reload it for faster
  repeat inference.
- [`references/troubleshooting.md`](references/troubleshooting.md)
  — extended troubleshooting catalogue.
- Sibling skill — NRE reconstruction and rendering pipeline that
  produces the inputs Fixer post-processes:
  [`../nre/SKILL.md`](../nre/SKILL.md).
- Sibling skill — HF dataset catalog including
  `PhysicalAI-Autonomous-Vehicles-NuRec` (USDZs whose renders Fixer
  is most commonly applied to):
  [`../physical-ai-datasets/SKILL.md`](../physical-ai-datasets/SKILL.md).
- Upstream open-source Difix3D+ repo:
  <https://github.com/nv-tlabs/Difix3D>
- Pretrained weights (HuggingFace): <https://huggingface.co/nvidia/Fixer>
- Cosmos-Predict2 container on NGC:
  <https://catalog.ngc.nvidia.com/orgs/nvidia/teams/cosmos/containers/cosmos-predict2-container>
- Difix3D+ paper: <https://arxiv.org/abs/2503.01774>
- NVIDIA Research project page:
  <https://research.nvidia.com/labs/toronto-ai/difix3d/>

## Limitations

- **Rendered inputs only.** Fixer is trained on artifacts specific
  to 3D reconstruction (Gaussian Splatting, NeRF, NRE). It is not a
  general-purpose image enhancer and may distort real photographs.
- **Internal resolution fixed at 1024×576.** Inputs are resized to
  that resolution before the model and resized back to the original
  `(W, H)` after. Non-16:9 content is squashed and unsquashed; content
  is preserved but geometric fidelity may be reduced.
- **Sequence ordering depends on filename natural-sort.** Use zero-
  padded, fixed-width indices (`frame_0001.png`) to avoid misordering.
- **Temporal coherence.** Any temporal stabilisation relies on
  the harmonizer's own prior outputs. Re-running on a non-empty
  output directory mixes stale outputs into the reference window and
  degrades quality — always start from an empty output directory.
- **Beta release.** Flags, default timestep, and checkpoint format
  may change between releases. Pin the container tag and checkpoint
  revision for reproducibility.
- **GPU architecture.** Requires Ampere or newer (compute capability
  ≥ 8.0) because inference uses bfloat16 autocast.

## Troubleshooting

| Error / symptom | Cause | Solution |
|-----------------|-------|----------|
| `ImportError: transformer_engine` | Running inference outside the cosmos-predict2 container. | Use the `docker run` command in [Run inference](#run-inference). Do not install TE manually — it requires a matching CUDA / cuDNN / compiler stack. |
| `CUDA error: out of memory` | GPU has < 12 GB VRAM, or batch size > 1. | Use a GPU with ≥ 16 GB VRAM, reduce `--batch-size` to 1, or close other GPU processes. |
| Output frame count < input frame count | Input directory contains unsupported formats or hidden files. | Ensure all frames end in `.png`, `.jpg`, or `.jpeg`. Remove hidden files. |
| Seam / visual discontinuity at early frames | The first few frames warm up the temporal window and look different from the rest. | Expected. Either discard the first pass of the first few frames, or prepend copies of the first frame and drop them from the output. |
| Model checkpoint not found inside container | Forgot to mount `./models` as `/models`. | Add `-v "$MODELS_DIR":/models:ro` to the `docker run` command and point `--model` at `/models/pretrained/pretrained_fixer.pkl`. |
| `HF_TOKEN` unset on weight download | Credential not exported. | `export HF_TOKEN=<token>` before calling `huggingface-cli`. Never write the token to a file checked into version control. |
| Grid-like artifacts in output | Model regression on certain domains. | Reduce the perceptual loss scale at fine-tune time (`--lambda_lpips 0.03`); for inference-only use, try a different timestep value or revert to the non-Fixer render. |

See [`references/troubleshooting.md`](references/troubleshooting.md)
for an expanded catalogue.
