---
name: nurec-fixer
description: >-
  Use when a user wants to enhance, harmonize, or temporally stabilize
  novel-view renders from a 3D reconstruction (especially NRE / NuRec
  outputs) by running NVIDIA Fixer V2 — a single-step diffusion model
  (Difix3D+ lineage) that removes artifacts in underconstrained regions
  of the 3D representation. Covers fetching the V2 weights from
  HuggingFace nvidia/Fixer, building the inference image via
  Dockerfile.cosmos from nv-tlabs/Fixer on top of cosmos-predict2,
  running per-sequence inference on a folder of rendered frames, and
  evaluating PSNR / LPIPS on paired render↔ground-truth sets.
  Post-processing only — not for training the reconstruction. Trigger
  keywords: nurec fixer, nvidia fixer V2, harmonizer, difix, difix3d+,
  pretrained_fixer.pkl, nvidia/Fixer, nv-tlabs/Fixer, Dockerfile.cosmos,
  fix rendered frames, remove reconstruction artifacts, temporal
  stabilization, cosmos fixer, cosmos-predict2-container,
  inference_pretrained_model.py, evaluate_test_dataset.py,
  --enable-difix.
version: "0.2.0"
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
  B200, GB200). Recent NVIDIA driver supporting CUDA 12, NVIDIA
  Container Toolkit, Docker (or Podman). ~16 GB VRAM at 1024x576.
  Internet egress to nvcr.io, github.com, huggingface.co. Credentials
  via env vars only: HF_TOKEN (required), NGC_API_KEY (optional).
dependencies:
  - bash
  - docker
  - git
  - nvidia-container-toolkit
  - python3
metadata:
  upstream: https://github.com/nv-tlabs/Fixer
  upstream_ancestry: https://github.com/nv-tlabs/Difix3D
  hf_models: https://huggingface.co/nvidia/Fixer
  model_version: "v2"
  ngc_container: nvcr.io/nvidia/cosmos/cosmos-predict2-container:1.2
  paper: https://arxiv.org/abs/2503.01774
  product_page: https://research.nvidia.com/labs/toronto-ai/difix3d/
---

# NVIDIA Fixer V2 (NuRec Post-Processing)

Single-step image diffusion model that removes artifacts from novel-view
renders produced by 3D reconstruction pipelines (Gaussian Splatting,
NeRF, NRE / NuRec, and similar). Fixer V2 is the successor to Difix3D+
(see [paper](https://arxiv.org/abs/2503.01774) for the V1 method); the
V2 inference code lives at
[`nv-tlabs/Fixer`](https://github.com/nv-tlabs/Fixer), and the V2
pretrained checkpoint (`pretrained_fixer.pkl`, the file inspected by
`torch.load(...).keys()` here) is published on HuggingFace at
[`nvidia/Fixer`](https://huggingface.co/nvidia/Fixer). Inference runs
inside a custom image built from `Dockerfile.cosmos` shipped in the V2
repo on top of
[`nvcr.io/nvidia/cosmos/cosmos-predict2-container:1.2`](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/cosmos/containers/cosmos-predict2-container).

> **Version alignment matters.** The legacy `nv-tlabs/Difix3D` repository
> is V1 code and pairs with V1 checkpoints. The `nvidia/Fixer`
> HuggingFace weights are V2 and will hit a `state_dict` mismatch if
> loaded against V1 model definitions. Use the V2 code path on this page
> end-to-end; treat `Difix3D` purely as the academic ancestor.

## Table of Contents

- [When to Use](#when-to-use) · [Inputs](#inputs) · [Instructions](#instructions) · [Output Format](#output-format)
- [Prerequisites](#prerequisites) · [Verifying secrets safely](#verifying-secrets-safely)
- [Container setup](#container-setup) — [Pull base](#pull-the-cosmos-predict2-base-image) · [Build V2 image](#build-the-v2-inference-image) · [File ownership](#file-ownership-must-pin--u-id--u-id--g)
- [Weight download](#weight-download) · [Run inference](#run-inference) · [Evaluate](#evaluate-a-paired-test-dataset)
- [Consuming Fixer via NRE](#consuming-fixer-via-nre---enable-difix) · [Scripts](#scripts) · [References](#references)
- [Limitations](#limitations) · [Troubleshooting](#troubleshooting) · [Teardown](#teardown)

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
   free disk for the container (~120 GB after `Dockerfile.cosmos`
   build) plus weights (~5 GB). Abort on any failure. **Never** verify
   `HF_TOKEN` / `NGC_API_KEY` are set by writing ad-hoc bash that
   substitutes the value (see
   [Verifying secrets safely](#verifying-secrets-safely)).
2. Pull the cosmos-predict2 base image
   `nvcr.io/nvidia/cosmos/cosmos-predict2-container:1.2` if it is not
   already present (see [Container setup](#container-setup)).
3. Clone the V2 Fixer code from
   [`nv-tlabs/Fixer`](https://github.com/nv-tlabs/Fixer) and build the
   inference image with `docker build -f Dockerfile.cosmos .`. See
   [Build the V2 inference image](#build-the-v2-inference-image).
4. Download the V2 pretrained Fixer weights from
   [`nvidia/Fixer`](https://huggingface.co/nvidia/Fixer) into a host
   directory (typically `./models/`) using the HuggingFace CLI. See
   [Weight download](#weight-download).
5. Confirm `input_dir` exists, contains supported images, that
   filenames natural-sort into the intended order, and that
   `output_dir` is empty or non-existent. Fail fast otherwise.
6. Run the inference container with `input_dir`, `output_dir`, and
   the weights directory mounted, and **pin the container UID/GID to
   the host user with `-u $(id -u):$(id -g)`** so outputs are owned by
   you, not by root (see [Run inference](#run-inference) and
   [File ownership](#file-ownership-must-pin--u-id--u-id--g)).
7. Validate that the number of output images equals the number of
   input images, spot-check a frame visually, and confirm
   `ls -l <output_dir>` shows your UID rather than `root`.
8. **If evaluating:** prepare a paired test dataset per
   [`references/test-dataset-layout.md`](references/test-dataset-layout.md)
   and run `evaluate_test_dataset.py` inside the container to produce
   `metrics.yaml` with PSNR and LPIPS.
9. When consuming Fixer as part of an NRE pipeline, prefer passing
   `--enable-difix --difix-cache=<path>` to the NRE entrypoint
   instead of running Fixer manually — NRE will download the same
   weights into the cache directory and call the model internally.
10. When done, follow [Teardown](#teardown) to reclaim ~127 GB of
    images, weights, code clones, and root-owned outputs the workflow
    leaves behind.

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
- `git` on PATH (V2 code lives in a GitHub clone).
- Internet egress to `nvcr.io`, `github.com`, and `huggingface.co`.
- HuggingFace account and access token exported as `HF_TOKEN`.
  Create at https://huggingface.co/settings/tokens. No license
  acceptance is currently gated on the `nvidia/Fixer` model page,
  but confirm the card terms before use.
- ~120 GB free disk: ~5 GB cosmos-predict2 base image,
  ~120 GB after the `Dockerfile.cosmos` build adds the Fixer stack,
  ~5 GB for the V2 HuggingFace weights, plus output room.

### Verifying secrets safely

**Always validate prerequisites by running
[`scripts/validate_setup.py`](scripts/validate_setup.py).** Never write
ad-hoc bash that interpolates `HF_TOKEN` / `NGC_API_KEY` values into
stdout. In particular, the common one-liner

```bash
# BAD — leaks the secret to the terminal when the variable is set
echo "HF_TOKEN: ${HF_TOKEN:+yes}${HF_TOKEN:-no}"
```

prints `yes<token-value>` whenever `HF_TOKEN` is set, because
`${VAR:-no}` falls back to "no" only when `VAR` is empty — when set, it
expands to `$VAR`. The validator uses presence-only checks
(`os.environ.get("HF_TOKEN") is not None`) that never echo the value.
If you must check in a shell, use a length check instead:

```bash
# OK — prints "set (N chars)" or "missing", never the value
test -n "$HF_TOKEN" && echo "HF_TOKEN: set (${#HF_TOKEN} chars)" || echo "HF_TOKEN: missing"
```

Rotate any token you suspect may have been echoed.

## Container setup

V2 inference uses a **two-stage image**: the public cosmos-predict2
base, plus the V2 Fixer code layered on top via
`Dockerfile.cosmos` shipped in the `nv-tlabs/Fixer` repo.

### Pull the cosmos-predict2 base image

```bash
BASE=nvcr.io/nvidia/cosmos/cosmos-predict2-container:1.2
docker image inspect "$BASE" >/dev/null 2>&1 || docker pull "$BASE"
```

This base ships PyTorch 2.x, `cosmos_predict2`, NVIDIA Transformer
Engine, and the CUDA / cuDNN stack Fixer needs.

If a site policy requires a mirrored / hardened variant of the
container, consult your NGC account and substitute the base tag.
Do not pin to `:latest` — pin to the dated minor release shown
above for reproducibility.

### Build the V2 inference image

The V2 codebase (`nv-tlabs/Fixer`) ships `Dockerfile.cosmos`, which
layers the Fixer Python source and its extra dependencies on top of
the cosmos-predict2 base. Build it once per host:

```bash
FIXER_REPO_DIR=${FIXER_REPO_DIR:-$PWD/Fixer}
IMAGE=nurec-fixer:v2

git clone --depth 1 https://github.com/nv-tlabs/Fixer "$FIXER_REPO_DIR"
cd "$FIXER_REPO_DIR"
docker build -f Dockerfile.cosmos -t "$IMAGE" .
```

The build is ~120 GB on disk after the layers settle. The `IMAGE` tag
`nurec-fixer:v2` is referenced by every `docker run` below; substitute
your own if your site uses a registry-prefixed tag.

> If a `docker build` failure cites a `state_dict` mismatch during a
> downstream test step, you almost certainly cloned the legacy
> `nv-tlabs/Difix3D` (V1) instead of `nv-tlabs/Fixer` (V2). Remove the
> wrong clone and re-run the V2 clone above.

### File ownership: MUST pin `-u $(id -u):$(id -g)`

By default, processes inside an `nvcr.io` container run as `root`. Any
file the container writes to a host bind mount lands on the host as
`root:root 644` — which means the host user cannot `mv`, `chmod`,
`cp`, or `rm` the output without `sudo`. Every documented `docker
run` snippet below threads `-u $(id -u):$(id -g)` into the container
so outputs are owned by **your** UID/GID. Do not drop this flag in
copy-pasted variants — verify with `ls -l <output_dir>` after every
run.

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
directory (read-only), and the output directory (read-write), and pin
the container UID/GID to the host user so outputs land owned by you:

```bash
IMAGE=nurec-fixer:v2                          # built from Dockerfile.cosmos above
INPUT_DIR=/absolute/path/to/rendered_frames
OUTPUT_DIR=/absolute/path/to/enhanced_frames
MODELS_DIR=$(pwd)/models

mkdir -p "$OUTPUT_DIR"
[ -z "$(ls -A "$OUTPUT_DIR" 2>/dev/null)" ] || {
  echo "ERROR: $OUTPUT_DIR is not empty."
  exit 1
}

docker run --gpus=all --rm --ipc=host \
  -u "$(id -u):$(id -g)" \
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

Verify ownership afterwards:

```bash
ls -l "$OUTPUT_DIR" | head            # first column should be your user, not 'root'
stat -c '%U:%G' "$OUTPUT_DIR"/*.png | sort -u
```

Notes:

- The in-container script path `/opt/fixer/src/inference_pretrained_model.py`
  is where `Dockerfile.cosmos` copies the V2 source. If your build
  ships it at a different path, run `docker run --rm "$IMAGE" find / -name
  inference_pretrained_model.py 2>/dev/null` once to locate it. Do
  **not** fall back to the V1 `nv-tlabs/Difix3D` script — it expects a
  V1 checkpoint and will crash with a state-dict mismatch against the
  `nvidia/Fixer` weights.
- `--ipc=host` is standard for NGC containers and harmless.
- Do not pass `torchrun` / `--nproc_per_node` — the inference script
  is single-process, single-GPU.
- The first run with `-u $(id -u):$(id -g)` may emit harmless
  warnings about an unknown UID inside the container (no entry in
  `/etc/passwd`); ignore them.

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
IMAGE=nurec-fixer:v2

docker run --gpus=all --rm --ipc=host \
  -u "$(id -u):$(id -g)" \
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
- [`references/teardown.md`](references/teardown.md) — full disk
  reclaim inventory, ownership-recovery commands, and post-teardown
  verification.
- [`references/troubleshooting.md`](references/troubleshooting.md)
  — extended troubleshooting catalogue.
- Sibling skill — NRE reconstruction and rendering pipeline that
  produces the inputs Fixer post-processes:
  [`../nre/SKILL.md`](../nre/SKILL.md).
- Sibling skill — HF dataset catalog including
  `PhysicalAI-Autonomous-Vehicles-NuRec` (USDZs whose renders Fixer
  is most commonly applied to):
  [`../physical-ai-datasets/SKILL.md`](../physical-ai-datasets/SKILL.md).
- V2 inference code (canonical):
  <https://github.com/nv-tlabs/Fixer>
- V2 pretrained weights (HuggingFace):
  <https://huggingface.co/nvidia/Fixer>
- V1 / academic ancestor — Difix3D+:
  <https://github.com/nv-tlabs/Difix3D> (incompatible state-dict; do
  **not** load `nvidia/Fixer` weights against this code)
- Cosmos-Predict2 container on NGC:
  <https://catalog.ngc.nvidia.com/orgs/nvidia/teams/cosmos/containers/cosmos-predict2-container>
- Difix3D+ paper: <https://arxiv.org/abs/2503.01774>
- NVIDIA Research project page:
  <https://research.nvidia.com/labs/toronto-ai/difix3d/>

## Limitations

- **Rendered inputs only.** Fixer is trained on 3D-reconstruction
  artifacts (3DGS, NeRF, NRE); it is not a general image enhancer and
  may distort real photographs.
- **Internal resolution fixed at 1024×576.** Inputs are resized to
  that resolution before the model and back to `(W, H)` after.
  Non-16:9 content is preserved but geometric fidelity may suffer.
- **Sequence ordering depends on filename natural-sort.** Use
  zero-padded indices (`frame_0001.png`) to avoid misordering.
- **Temporal coherence relies on prior outputs.** Re-running on a
  non-empty `output_dir` mixes stale frames into the reference window
  and degrades quality — always start from an empty directory.
- **Beta release.** Flags, timestep, and checkpoint format may change
  between releases. Pin container tag + checkpoint for reproducibility.
- **GPU architecture.** Requires Ampere or newer (compute capability
  ≥ 8.0; inference uses bfloat16 autocast).
- **V2 weights are not compatible with V1 code** (`nv-tlabs/Difix3D`)
  — see the [Run inference](#run-inference) note.

## Troubleshooting

| Error / symptom | Most common cause |
|-----------------|-------------------|
| `state_dict` mismatch at model load | Cloned `nv-tlabs/Difix3D` (V1) instead of `nv-tlabs/Fixer` (V2). |
| Output files owned by `root:root` | Forgot `-u $(id -u):$(id -g)` on `docker run`. |
| `ImportError: transformer_engine` | Running inference outside the cosmos-predict2-based image. |
| `CUDA error: out of memory` | GPU < 16 GB VRAM or batch size > 1. |
| Model checkpoint not found in container | Forgot the `-v "$MODELS_DIR":/models:ro` mount. |
| Output frame count < input frame count | Hidden files / unsupported extensions in input. |

Full diagnostic table — root cause and fix for every row above — lives
in [`references/troubleshooting.md`](references/troubleshooting.md).

## Teardown

A complete Fixer V2 workflow leaves roughly **~127 GB** on the host:
~125 GB of container images (cosmos-predict2 base + the
`Dockerfile.cosmos` build), ~5 GB of HuggingFace V2 weights, the V2
source clone, and a sequence-dependent output directory. Reclaim disk
with:

```bash
docker image rm nurec-fixer:v2 nvcr.io/nvidia/cosmos/cosmos-predict2-container:1.2 2>/dev/null || true
docker image prune -f && docker builder prune -f
rm -rf ./models "${HF_HOME:-$HOME/.cache/huggingface}/hub/models--nvidia--Fixer"
rm -rf "${FIXER_REPO_DIR:-$PWD/Fixer}"
rm -rf /absolute/path/to/enhanced_frames /absolute/path/to/eval_out
rm -rf "${HOME}/.cache/nre/difix"     # only if you used --enable-difix
```

If outputs were created **before** `-u $(id -u):$(id -g)` was added to
the `docker run` command, they are owned by `root` and need
`sudo chown -R "$(id -u):$(id -g)" <dir>` first. See
[`references/teardown.md`](references/teardown.md) for the full
inventory, ownership-recovery commands, and post-teardown
verification.

Do **not** revoke `HF_TOKEN` as part of teardown unless you suspect it
has been leaked (see [Verifying secrets safely](#verifying-secrets-safely)).
