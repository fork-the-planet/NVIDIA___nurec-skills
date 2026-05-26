---
name: nurec-fixer
description: >-
  Use when the user wants to enhance, harmonize, or temporally
  stabilize novel-view renders from a 3D reconstruction (especially
  NRE / NuRec outputs) by running NVIDIA's unified Fixer & Harmonizer
  — a JIT TorchScript model (Difix3D+ lineage) shipped on NGC at
  `nvidia/nre/nurec-fixer:cosmos_3dgut_fixer_harmonizer` that removes
  artifacts and harmonizes appearance in underconstrained regions.
  Covers fetching the artifact (inference script +
  `harmonizer_temporal.pt` + `harmonizer_nontemporal.pt`) from NGC,
  running per-sequence inference inside
  `nvcr.io/nvidia/pytorch:24.10-py3`, and consuming the same weights
  via `nre --enable-difix`. Post-processing only; do NOT use to
  retrain a reconstruction or as a general image super-resolution
  model. Trigger keywords: nurec fixer, nurec harmonizer, nvidia
  fixer, unified fixer harmonizer, cosmos_3dgut_fixer_harmonizer,
  harmonizer_temporal.pt, harmonizer_nontemporal.pt,
  inference_jit_harmonizer.py, difix, difix3d+, fix rendered frames,
  temporal stabilization, --enable-difix.
version: "0.3.0"
tools:
  - Shell
  - Read
  - Write
license: CC-BY-4.0 AND Apache-2.0
compatibility: >-
  Linux host with an NVIDIA GPU of Ampere architecture or newer
  (compute capability >= 8.0; A100, A10, L40, H100, RTX 30/40/PRO,
  B200, GB200). Recent NVIDIA driver supporting CUDA 12, NVIDIA
  Container Toolkit, Docker (or Podman). ~16 GB VRAM at 1024x576.
  Internet egress to nvcr.io and ngc.nvidia.com. Credentials via env
  vars only: NGC_API_KEY (required for both the model artifact and the
  pytorch base image pull).
dependencies:
  - bash
  - docker
  - nvidia-container-toolkit
  - python3
  - ngc-cli
metadata:
  author: NVIDIA NuRec
  tags:
    - nurec
    - fixer
    - harmonizer
    - reconstruction
    - autonomous-vehicles
    - post-processing
  ngc_model: nvcr.io/nvidia/nre/nurec-fixer
  ngc_model_version: cosmos_3dgut_fixer_harmonizer
  ngc_model_page: https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nre/models/nurec-fixer?version=cosmos_3dgut_fixer_harmonizer
  ngc_container: nvcr.io/nvidia/pytorch:24.10-py3
  release_status: beta
  paper: https://arxiv.org/abs/2503.01774
  product_page: https://research.nvidia.com/labs/toronto-ai/difix3d/
---

# NVIDIA Unified Fixer & Harmonizer (NuRec Post-Processing)

## Purpose

Run NVIDIA's unified Fixer & Harmonizer JIT TorchScript model
(Difix3D+ lineage) on novel-view renders from a 3D reconstruction to
remove artifacts (ghosting, floaters, splotches) and harmonize
appearance in underconstrained regions. Either standalone via
`inference_jit_harmonizer.py` inside `nvcr.io/nvidia/pytorch:24.10-py3`
or through `nre --enable-difix` during NRE rendering.

**Use this skill when:** the user has rendered frames from NRE or
another 3D reconstruction pipeline and wants to clean them up before
downstream consumption (training data, qualitative review, sensor
sim, etc.).

**Do NOT use this skill when:**

- The user wants to retrain or fine-tune the reconstruction itself
  (the Fixer is inference-only post-processing).
- The user wants a general image super-resolution / denoiser — Fixer
  is tuned for 3D-reconstruction artifacts, not arbitrary photos.
- No NGC API key is available — both the model and the base image
  live behind NGC auth.

## Background

A single TorchScript JIT model that removes artifacts and harmonizes
appearance in novel-view renders produced by 3D reconstruction
pipelines (Gaussian Splatting, NeRF, NRE / NuRec, and similar). The
unified release replaces the earlier HuggingFace `nvidia/Fixer` V2
artifact and is published on NGC at
[`nvidia/nre/nurec-fixer`, version
`cosmos_3dgut_fixer_harmonizer`](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nre/models/nurec-fixer?version=cosmos_3dgut_fixer_harmonizer).
Inference runs inside the public PyTorch base image
[`nvcr.io/nvidia/pytorch:24.10-py3`](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch);
no custom image build is required.

> **Beta release.** Flag names, model-file layout, and even the
> required PyTorch base may shift between minor versions. Pin both the
> NGC artifact version (`cosmos_3dgut_fixer_harmonizer`) and the base
> image tag (`24.10-py3`) for reproducibility.

The artifact ships three files plus a `requirements.txt`:

| File | Purpose |
|------|---------|
| `inference_jit_harmonizer.py` | Entry point. Iterates an input directory and writes harmonized frames. |
| `harmonizer_temporal.pt`      | Temporal TorchScript JIT model. Used after the warm-up window (consumes the previous 4 outputs as references). |
| `harmonizer_nontemporal.pt`   | Nontemporal TorchScript JIT model. Used for the first frames before a temporal window exists. |
| `requirements.txt`            | Extra Python dependencies installed on top of `pytorch:24.10-py3`. |

## Table of Contents

- [When to Use](#when-to-use) · [Inputs](#inputs) · [Instructions](#instructions) · [Output Format](#output-format)
- [Prerequisites](#prerequisites) · [Verifying secrets safely](#verifying-secrets-safely)
- [Container setup](#container-setup) · [File ownership](#file-ownership-must-pin--u-id--u-id--g)
- [Model download](#model-download) · [Run inference](#run-inference)
- [Consuming Fixer via NRE](#consuming-fixer-via-nre---enable-difix) · [Scripts](#scripts) · [References](#references)
- [Limitations](#limitations) · [Troubleshooting](#troubleshooting) · [Teardown](#teardown)

The harmonizer is typically chained **after** a NuRec / NRE
reconstruction: the reconstruction renders novel views from held-out
camera poses, and the harmonizer cleans up the rendered frames and
stabilizes them temporally to improve perceptual quality and downstream
metrics (LPIPS, PSNR).

## When to Use

- User has a directory of rendered frames from a 3D reconstruction
  (NRE, NuRec, Gaussian Splatting, NeRF, etc.) and asks to remove
  artifacts, enhance, harmonize, or "fix" them.
- User mentions running the NVIDIA Fixer / Harmonizer model, Difix3D+,
  the NGC model `nvidia/nre/nurec-fixer`, or the
  `cosmos_3dgut_fixer_harmonizer` version on novel-view renders.
- User mentions `--enable-difix` in an NRE pipeline and asks what
  weights / cache to supply.
- User mentions `harmonizer_temporal.pt` / `harmonizer_nontemporal.pt`
  / `inference_jit_harmonizer.py` and wants an end-to-end recipe.

### When NOT to Use

- The user wants to **train or run the 3D reconstruction itself**
  (e.g. multi-camera NRE reconstruction, Gaussian Splatting training).
  Use the sibling [`nre`](../nre/SKILL.md) skill instead; the
  harmonizer only post-processes renders.
- The user wants the harmonizer **inline with NRE rendering** (single
  pipeline, no separate container). Use NRE's built-in
  `--enable-difix` flag with `difix=cosmos_difix` (default since
  25.09) or `difix=sd_difix` for the legacy Stable-Diffusion variant.
  Reach for this standalone skill only when you need to swap variants,
  post-process frames that were rendered earlier, or apply the
  harmonizer to non-NRE renders.
- The user wants to train or fine-tune a new harmonizer / fixer model.
  Fine-tuning requires the internal training codebase and is not
  covered by this external-facing skill.
- The user wants to apply the harmonizer to real (non-rendered) camera
  images — the model is trained specifically on reconstruction
  artifacts and will distort real photographs.

## Inputs

- **input_dir** — directory of rendered images (`.png`, `.jpg`) to
  enhance. Filenames MUST natural-sort into temporal order; the
  harmonizer reads the **previous 4 outputs** as references after
  warm-up, so misordered filenames silently degrade temporal coherence.
  (source: user prompt; required.)
- **output_dir** — directory for harmonized images. MUST be empty or
  non-existent — the temporal pass would otherwise reuse stale frames
  as references. (source: user prompt; required.)
- **artifact_dir** — host directory holding the unpacked NGC artifact
  (`inference_jit_harmonizer.py`, `harmonizer_temporal.pt`,
  `harmonizer_nontemporal.pt`, `requirements.txt`). Defaults to
  `./nurec-fixer_vcosmos_3dgut_fixer_harmonizer/` after running the
  download step. (source: user prompt or agent context; required.)
- **NGC_API_KEY** — NGC personal API key, set as an environment
  variable on the host (source: agent context; required for both the
  model download and the base-image pull; create at
  <https://ngc.nvidia.com/setup/personal-keys>).

## Instructions

1. Run [`scripts/validate_setup.py`](scripts/validate_setup.py) to
   confirm the host has Docker, the NVIDIA Container Toolkit, a GPU
   with compute capability ≥ 8.0, `NGC_API_KEY` exported, and enough
   free disk for the base image (~25 GB) plus the harmonizer artifact
   (~5 GB). Abort on any failure. **Never** verify `NGC_API_KEY` is
   set by writing ad-hoc bash that substitutes the value (see
   [Verifying secrets safely](#verifying-secrets-safely)).
2. Authenticate to NGC and pull the PyTorch base image
   `nvcr.io/nvidia/pytorch:24.10-py3` if it is not already present
   (see [Container setup](#container-setup)).
3. Download the harmonizer artifact from
   [`nvidia/nre/nurec-fixer:cosmos_3dgut_fixer_harmonizer`](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nre/models/nurec-fixer?version=cosmos_3dgut_fixer_harmonizer)
   into a host directory using `ngc registry model download-version`.
   Verify the four expected files (`inference_jit_harmonizer.py`,
   `harmonizer_temporal.pt`, `harmonizer_nontemporal.pt`,
   `requirements.txt`) are present. See [Model download](#model-download).
4. Confirm `input_dir` exists, contains supported images, that
   filenames natural-sort into the intended order, and that
   `output_dir` is empty or non-existent. Fail fast otherwise.
5. Run the inference container with `input_dir`, `output_dir`, and
   `artifact_dir` mounted. Install the artifact's `requirements.txt`
   on top of the base image inside the container, then invoke
   `inference_jit_harmonizer.py`. **Pin the container UID/GID to the
   host user with `-u $(id -u):$(id -g)`** so outputs are owned by
   you, not by root (see [Run inference](#run-inference) and
   [File ownership](#file-ownership-must-pin--u-id--u-id--g)).
6. Validate that the number of output images equals the number of
   input images, spot-check a frame visually, and confirm
   `ls -l <output_dir>` shows your UID rather than `root`.
7. When consuming the harmonizer as part of an NRE pipeline, prefer
   passing `--enable-difix --difix-cache=<path>` to the NRE entrypoint
   instead of running the harmonizer manually — NRE will pull the same
   NGC artifact into the cache directory and call the model
   internally.
8. When done, follow [Teardown](#teardown) to reclaim the ~30 GB of
   images, weights, and outputs the workflow leaves behind.

## Output Format

Per-frame harmonized images land in `output_dir` with the same
basenames and file extensions as the inputs:

```text
<output_dir>/
├── <frame-0>.png        # or .jpg, matching input
├── <frame-1>.png
└── ...
```

Images are restored to each input's original `(width, height)` after
the model runs at its internal 1024×576 working resolution.

## Prerequisites

- Linux host with NVIDIA driver supporting CUDA 12.x.
- Docker (or Podman) + [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
- NVIDIA GPU with compute capability ≥ 8.0 (Ampere or newer).
  ~16 GB VRAM recommended.
- Internet egress to `nvcr.io` and `ngc.nvidia.com`.
- NGC personal API key exported as `NGC_API_KEY`. Create at
  <https://ngc.nvidia.com/setup/personal-keys>. The same key is used
  for `docker login nvcr.io` (username `$oauthtoken`, password
  `$NGC_API_KEY`) and for `ngc registry model download-version`.
- [`ngc` CLI](https://docs.ngc.nvidia.com/cli/) on PATH for downloading
  the harmonizer artifact. Tested with NGC CLI 3.x.
- ~30 GB free disk: ~25 GB for `pytorch:24.10-py3`, ~5 GB for the
  harmonizer artifact and per-run pip cache, plus output room.

### Verifying secrets safely

**Always validate prerequisites by running
[`scripts/validate_setup.py`](scripts/validate_setup.py).** Never write
ad-hoc bash that interpolates `NGC_API_KEY` values into stdout. In
particular, the common one-liner

```bash
# BAD — leaks the secret to the terminal when the variable is set
echo "NGC_API_KEY: ${NGC_API_KEY:+yes}${NGC_API_KEY:-no}"
```

prints `yes<key-value>` whenever `NGC_API_KEY` is set, because
`${VAR:-no}` falls back to "no" only when `VAR` is empty — when set, it
expands to `$VAR`. The validator uses presence-only checks
(`os.environ.get("NGC_API_KEY") is not None`) that never echo the
value. If you must check in a shell, use a length check instead:

```bash
# OK — prints "set (N chars)" or "missing", never the value
test -n "$NGC_API_KEY" && echo "NGC_API_KEY: set (${#NGC_API_KEY} chars)" || echo "NGC_API_KEY: missing"
```

Rotate any key you suspect may have been echoed.

## Container setup

The unified harmonizer runs inside the unmodified
`nvcr.io/nvidia/pytorch:24.10-py3` base; there is no custom image
build (this is a deliberate simplification compared to the older
`Dockerfile.cosmos` recipe used for the V2 HF artifact).

### Pull the PyTorch base image

```bash
BASE=nvcr.io/nvidia/pytorch:24.10-py3

# `docker login` once per host. Username MUST be the literal string
# `$oauthtoken`; the password is your NGC personal API key.
echo "$NGC_API_KEY" | docker login nvcr.io --username '$oauthtoken' --password-stdin

docker image inspect "$BASE" >/dev/null 2>&1 || docker pull "$BASE"
```

This base ships PyTorch 2.x and the CUDA / cuDNN stack the harmonizer
needs to load `harmonizer_*.pt` TorchScript JIT modules.

If a site policy requires a mirrored / hardened variant of the
container, consult your NGC account and substitute the base tag — but
**do not** pin to `:latest`; pin to the dated tag shown above for
reproducibility.

### File ownership: MUST pin `-u $(id -u):$(id -g)`

By default, processes inside an `nvcr.io` container run as `root`.
Any file the container writes to a host bind mount lands on the host
as `root:root 644` — which means the host user cannot `mv`, `chmod`,
`cp`, or `rm` the output without `sudo`. Every documented `docker
run` snippet below threads `-u $(id -u):$(id -g)` into the container
so outputs are owned by **your** UID/GID. Do not drop this flag in
copy-pasted variants — verify with `ls -l <output_dir>` after every
run.

## Model download

Export your NGC API key, configure the NGC CLI, then download the
harmonizer artifact into a host-side directory. The container will
mount this read-only at inference time:

```bash
export NGC_API_KEY=<your-ngc-personal-key>   # create at https://ngc.nvidia.com/setup/personal-keys

# Configure the NGC CLI once per host (org=nvidia, team=nre).
# The CLI reads $NGC_API_KEY from the environment when --api-key is
# omitted; it never prints the value back.
ngc config set --format_type ascii

ngc registry model download-version \
  "nvidia/nre/nurec-fixer:cosmos_3dgut_fixer_harmonizer" \
  --dest .
```

After the download completes, the artifact lives in
`./nurec-fixer_vcosmos_3dgut_fixer_harmonizer/`. Verify the four
expected files exist:

```bash
ARTIFACT_DIR=$(pwd)/nurec-fixer_vcosmos_3dgut_fixer_harmonizer
ls -lh "$ARTIFACT_DIR"/{inference_jit_harmonizer.py,harmonizer_temporal.pt,harmonizer_nontemporal.pt,requirements.txt}
```

The two `.pt` checkpoints together are roughly 3 GB. If either file is
much smaller, the download was truncated — `rm -rf "$ARTIFACT_DIR"` and
retry. Never commit the checkpoints to version control and never embed
`NGC_API_KEY` in a config file.

## Run inference

A single `docker run --rm` invocation handles end-to-end inference.
Bind-mount the artifact directory (read-only), the input frames
(read-only), and the output directory (read-write); install the
artifact's `requirements.txt` on top of the base image inside the
container; then call `inference_jit_harmonizer.py`. Pin the container
UID/GID to the host user so outputs land owned by you:

```bash
BASE=nvcr.io/nvidia/pytorch:24.10-py3
ARTIFACT_DIR=$(pwd)/nurec-fixer_vcosmos_3dgut_fixer_harmonizer
INPUT_DIR=/absolute/path/to/rendered_frames
OUTPUT_DIR=/absolute/path/to/enhanced_frames

mkdir -p "$OUTPUT_DIR"
[ -z "$(ls -A "$OUTPUT_DIR" 2>/dev/null)" ] || {
  echo "ERROR: $OUTPUT_DIR is not empty (temporal pass would reuse stale frames)."
  exit 1
}

docker run --gpus=all --rm --ipc=host \
  -u "$(id -u):$(id -g)" \
  -e PIP_DISABLE_PIP_VERSION_CHECK=1 \
  -e PYTHONUSERBASE=/tmp/pyuser \
  -v "$ARTIFACT_DIR":/work:ro \
  -v "$INPUT_DIR":/in:ro \
  -v "$OUTPUT_DIR":/out \
  -w /work \
  "$BASE" \
  bash -c '
    set -euo pipefail
    pip install --user --no-warn-script-location -r /work/requirements.txt
    python /work/inference_jit_harmonizer.py \
      --input_image /in \
      --output_dir /out \
      --temporal_model_path /work/harmonizer_temporal.pt \
      --nontemporal_model_path /work/harmonizer_nontemporal.pt
  '
```

Verify ownership afterwards:

```bash
ls -l "$OUTPUT_DIR" | head            # first column should be your user, not 'root'
stat -c '%U:%G' "$OUTPUT_DIR"/*.png | sort -u
```

Notes:

- The artifact is mounted read-only on `/work`; the script does not
  write into `$ARTIFACT_DIR` and the only writable path inside the
  container is `/out` plus the user pip prefix at `/tmp/pyuser`.
- `--ipc=host` is standard for NGC containers and harmless.
- The inference script is **single-process, single-GPU**. Do not pass
  `torchrun` / `--nproc_per_node`.
- The first run with `-u $(id -u):$(id -g)` may emit harmless
  warnings about an unknown UID inside the container (no entry in
  `/etc/passwd`); ignore them.
- For repeat runs on the same host, the simplest amortization is to
  bake a small wrapper image that installs `requirements.txt` once.
  See [`references/wrapper-image.md`](references/wrapper-image.md).

### Inference flags

| Flag | Default | Purpose |
|------|---------|---------|
| `--input_image` | _(required)_ | Directory of rendered frames inside the container. |
| `--output_dir` | `output` | Directory for harmonized frames. |
| `--temporal_model_path` | _(required)_ | Path to `harmonizer_temporal.pt` inside the container. |
| `--nontemporal_model_path` | _(required)_ | Path to `harmonizer_nontemporal.pt` inside the container. The artifact docs occasionally spell this `--non_temporal_model_path`; both forms appear in the beta tarball — check `python inference_jit_harmonizer.py --help` if either name errors. |

## Consuming Fixer via NRE (`--enable-difix`)

If you are already running an NRE reconstruction pipeline from
`nvcr.io/nvidia/nre/nre:<tag>` (or the GA-channel
`nvcr.io/nvidia/nre/nre-ga:<tag>`), the cleanest integration is to
let NRE manage the harmonizer for you. Pass the `--enable-difix`
flag with a cache directory on the host; NRE will download the same
NGC artifact into the cache on first run (default
`${HOME}/.cache/nre/difix/`) and post-process renders internally.
See the sibling [`nre`](../nre/SKILL.md) skill (`serve-grpc
--enable-difix` invocation in the CLI cookbook) for the full
pipeline command shape, and pick between `difix=cosmos_difix`
(default since 25.09) and `difix=sd_difix` (legacy
Stable-Diffusion variant) at the same time.

When using this path, you do **not** need to run the steps in
[Model download](#model-download) or [Run inference](#run-inference)
above — NRE owns the whole cycle.

## Scripts

| Script | Purpose | Usage |
|--------|---------|-------|
| `scripts/validate_setup.py` | Verify host has Docker, NVIDIA Container Toolkit, a GPU with compute capability ≥ 8.0, `NGC_API_KEY` exported, and sufficient free disk. Makes no network calls. | Execute: `python scripts/validate_setup.py` |
| `scripts/.env.example`      | Template for `NGC_API_KEY`. Copy to `.env`, fill in, and source before running the workflow.                                   | `cp scripts/.env.example .env && set -a && . ./.env && set +a` |

## References

- [`references/wrapper-image.md`](references/wrapper-image.md) — how
  to bake a small `Dockerfile` on top of `pytorch:24.10-py3` that
  pre-installs `requirements.txt` for faster repeat inference.
- [`references/teardown.md`](references/teardown.md) — full disk
  reclaim inventory, ownership-recovery commands, and post-teardown
  verification.
- [`references/troubleshooting.md`](references/troubleshooting.md)
  — extended troubleshooting catalogue.
- Sibling skill — NRE reconstruction and rendering pipeline that
  produces the inputs the harmonizer post-processes:
  [`../nre/SKILL.md`](../nre/SKILL.md).
- Sibling skill — HF dataset catalog including
  `PhysicalAI-Autonomous-Vehicles-NuRec` (USDZs whose renders the
  harmonizer is most commonly applied to):
  [`../physical-ai-datasets/SKILL.md`](../physical-ai-datasets/SKILL.md).
- NGC harmonizer model card:
  <https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nre/models/nurec-fixer?version=cosmos_3dgut_fixer_harmonizer>
- NGC PyTorch base container:
  <https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch>
- Difix3D+ paper (academic ancestor):
  <https://arxiv.org/abs/2503.01774>
- NVIDIA Research project page:
  <https://research.nvidia.com/labs/toronto-ai/difix3d/>

## Limitations

- **Rendered inputs only.** The harmonizer is trained on
  3D-reconstruction artifacts (3DGS, NeRF, NRE); it is not a general
  image enhancer and may distort real photographs.
- **Internal resolution fixed at 1024×576.** Inputs are resized to
  that resolution before the model and back to `(W, H)` after.
  Non-16:9 content is preserved but geometric fidelity may suffer.
- **Sequence ordering depends on filename natural-sort.** Use
  zero-padded indices (`frame_0001.png`) to avoid misordering. The
  temporal model consumes the previous 4 outputs as references; if
  filenames misorder, references are wrong and quality degrades
  silently.
- **Temporal coherence relies on prior outputs.** Re-running on a
  non-empty `output_dir` mixes stale frames into the reference window
  and degrades quality — always start from an empty directory.
- **First 4 frames are nontemporal.** The opening frames go through
  `harmonizer_nontemporal.pt` and may show a visible seam where the
  temporal model takes over; either discard the warm-up frames or
  prepend copies of the first frame and drop those copies on output.
- **Beta release.** Flag names, the model file layout, and the
  required base image may change between minor versions. Pin the NGC
  artifact version (`cosmos_3dgut_fixer_harmonizer`) and the base
  image tag (`24.10-py3`) for reproducibility.
- **GPU architecture.** Requires Ampere or newer (compute capability
  ≥ 8.0; inference uses bfloat16 autocast).
- **No standalone evaluation script.** Unlike the previous V2 HF
  release, the unified beta artifact does **not** ship
  `evaluate_test_dataset.py`. PSNR / LPIPS evaluation must be wired
  up by the caller (e.g. via `lpips` and `torchmetrics` against
  paired ground-truth frames).

## Troubleshooting

| Error / symptom | Most common cause |
|-----------------|-------------------|
| `docker pull` 401 / 403 on `pytorch:24.10-py3` | `docker login nvcr.io` not run, or `NGC_API_KEY` expired. |
| `ngc registry model download-version ... 403` | NGC CLI not configured, or the API key lacks the `nvidia/nre` scope. |
| `RuntimeError` loading `harmonizer_*.pt` | Mismatched torch / CUDA version. Always run inside `pytorch:24.10-py3`, not on the host. |
| Output files owned by `root:root` | Forgot `-u $(id -u):$(id -g)` on `docker run`. |
| `CUDA error: out of memory` | GPU < 16 GB VRAM. Run on Ampere+ with at least 16 GB. |
| Script can't find `harmonizer_temporal.pt` | Forgot the `-v "$ARTIFACT_DIR":/work:ro` mount, or `--temporal_model_path` points at a host path instead of `/work/...`. |
| `unrecognized arguments: --nontemporal_model_path` | Beta artifact spells the flag `--non_temporal_model_path`. Run `python /work/inference_jit_harmonizer.py --help` and use whichever the help message lists. |
| Output frame count < input frame count | Hidden files / unsupported extensions in input. |
| Visible seam at frame 5 of a sequence | Expected — temporal model takes over from the nontemporal warm-up. Discard warm-up frames or prepend copies of frame 0. |

Full diagnostic table — root cause and fix for every row above — lives
in [`references/troubleshooting.md`](references/troubleshooting.md).

## Teardown

A complete harmonizer workflow leaves roughly **~30 GB** on the host:
~25 GB for `nvcr.io/nvidia/pytorch:24.10-py3`, ~5 GB of NGC harmonizer
artifact (mostly the two `.pt` checkpoints), plus a sequence-dependent
output directory. Reclaim disk with:

```bash
docker image rm nvcr.io/nvidia/pytorch:24.10-py3 2>/dev/null || true
docker image prune -f
rm -rf ./nurec-fixer_vcosmos_3dgut_fixer_harmonizer
rm -rf /absolute/path/to/enhanced_frames
rm -rf "${HOME}/.cache/nre/difix"     # only if you used --enable-difix
```

If outputs were created **before** `-u $(id -u):$(id -g)` was added to
the `docker run` command, they are owned by `root` and need
`sudo chown -R "$(id -u):$(id -g)" <dir>` first. See
[`references/teardown.md`](references/teardown.md) for the full
inventory, ownership-recovery commands, and post-teardown
verification.

Do **not** revoke `NGC_API_KEY` as part of teardown unless you suspect
it has been leaked (see [Verifying secrets safely](#verifying-secrets-safely)).
