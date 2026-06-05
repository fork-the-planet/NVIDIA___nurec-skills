# DiffusionHarmonizer — Extended Troubleshooting

## Installation / Environment

**`docker: Error response from daemon: could not select device driver "" with capabilities: [[gpu]]`**

The NVIDIA Container Toolkit is not installed or Docker is not
configured for the NVIDIA runtime. Install from
<https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html>
and restart Docker.

**`docker pull` from `nvcr.io` returns 401 / 403**

Authenticate Docker to NGC if your environment requires it:

```bash
echo "$NGC_API_KEY" | docker login nvcr.io --username '$oauthtoken' --password-stdin
```

The username must be the literal string `$oauthtoken`. Do not paste the
key into shell history.

**`./download_checkpoints.sh` returns 401 / 403**

`HF_TOKEN` is missing, expired, lacks read scope, or the
`nvidia/Harmonizer` or `nvidia/Cosmos-Predict2-0.6B-Text2Image` license
has not been accepted. Visit each model page in a browser, accept the
license if gated, then run `hf auth login --token "$HF_TOKEN"`.

**`hf: command not found`**

Install the Hugging Face CLI:

```bash
python3 -m pip install --user "huggingface_hub[cli]"
```

Depending on package version, the executable may be `hf` or
`huggingface-cli`.

**Build fails because a patch does not apply**

The patch may already be applied, or the installed Cosmos Predict2
package version differs from the patch target. Prefer the public project
Dockerfile for the checkout you are using. Apply `text2image_dit.patch`
and `tokenizer.patch` manually only when running the raw base container.

## Runtime

**`diffusion_harmonizer.pkl` cannot be found**

The expected checkpoint path after download is:

```text
models/diffusion_harmonizer.pkl
```

Run:

```bash
./download_checkpoints.sh
```

from the code checkout, or adjust `--model_path` to the actual
in-container path.

**`ModuleNotFoundError` / cannot import sibling modules, or base Cosmos
model not found**

`inference_pix2pix_turbo_harmonizer.py` imports sibling modules from
`src/` and loads the base model from `src/checkpoints/`. Launch it from
`/work/src` (`-w /work/src`) and mount the whole checkout at `/work` so
both `models/` and `src/checkpoints/` are visible — do not mount only
the frames directory.

**Output frames are not where I expected**

`inference_pix2pix_turbo_harmonizer.py` has no `--output` flag. It
writes to a sibling folder named `<input_dir>_<model_identifier>` next
to the input directory. Make sure that parent directory is writable and
mounted on the host.

**Output files are owned by root**

The container was run without `-u $(id -u):$(id -g)`. Recover once with:

```bash
sudo chown -R "$(id -u):$(id -g)" /absolute/path/to/output
```

Then rerun with the `-u` flag.

**Frames appear out of temporal order / temporal artifacts**

`inference_pix2pix_turbo_harmonizer.py` enhances frames in natural-sorted
order and feeds previous outputs back as temporal references. Use
fixed-width zero-padded names such as `frame_000010.png` so ordering is
unambiguous, or pass `--nontemporal` to disable temporal conditioning for
unordered images.

**CUDA out of memory**

Lower `--resolution`, disable speed/export modes, avoid TensorRT compile
on small GPUs, or move to a larger Ampere/Hopper/Blackwell GPU. Training
is much heavier than inference and the README command assumes 8 GPUs.

## Evaluation

**Evaluation reports missing pairs or zero images**

The test dataset must mirror `render/` and `gt/` exactly:

```text
{scene}/render/{camera}/{timestamp}.png
{scene}/gt/{camera}/{timestamp}.png
```

Every render file needs a matching ground-truth file with the same
camera subdirectory and filename.

**`metrics.yaml` is missing**

Check that `evaluate_test_dataset.py` completed without exceptions and
that the output path is writable by the container user. The documented
evaluation command mounts the test dataset read-write because the
released script writes `evaluation/` and `metrics.yaml` next to the
input structure. Keep `-u $(id -u):$(id -g)` so those outputs are owned
by the host user.

## Training

**Training exits before loading data**

Confirm the JSON manifest contains `train` and `test` keys and each item
has `image`, `target_image`, and `prompt`. Paths must be visible inside
the container, not just on the host.

**Training is unstable or too slow**

Start from the release README defaults: learning rate `2e-5`, timestep
`250`, `resize_576x1024`, `lambda_lpips 0.3`, bf16 mixed precision, and
`--pretrained_path` when fine-tuning. For the released dataset, add
`--fixing_data_weight 3` to up-weight artifact-correction pairs.

## Secrets

Never check token presence with a command that can print the token value.
Use the validator or a length-only check:

```bash
test -n "$HF_TOKEN" && echo "HF_TOKEN: set (${#HF_TOKEN} chars)" || echo "HF_TOKEN: missing"
```

Rotate any token that was echoed into logs or committed to disk.
