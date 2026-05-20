# Fixer — Extended Troubleshooting

## Installation / environment

**`docker: Error response from daemon: could not select device driver "" with capabilities: [[gpu]]`**

The NVIDIA Container Toolkit is not installed or not configured for
the Docker runtime. Install from
<https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html>
and restart the Docker daemon.

**`Unable to find image 'nvcr.io/nvidia/cosmos/cosmos-predict2-container:1.2' locally`** followed by a 401 / 403

NGC requires authentication for the container. Run
`docker login nvcr.io` with your NGC API key (the password field is
the key; username is `$oauthtoken`). Generate a key at
<https://ngc.nvidia.com/setup/api-keys>.

**HuggingFace download hangs or errors with 401**

`HF_TOKEN` is missing, expired, or the model license was not
accepted. Re-issue a token at
<https://huggingface.co/settings/tokens>, visit
<https://huggingface.co/nvidia/Fixer>, accept any gating terms, and
retry `huggingface-cli login --token "$HF_TOKEN"`.

## Runtime

**Script can't find `pretrained_fixer.pkl`**

Inside the container, the host `./models` directory must be mounted
at `/models` (read-only is fine). Double-check the `-v` flag in the
`docker run` command and that `--model` points at the path
**inside** the container (`/models/pretrained/pretrained_fixer.pkl`),
not the host path.

**Inference script path differs from the README**

The release container may ship the inference scripts at
`/work/src/`, `/opt/fixer/src/`, or `/fixer-codebase/src/` depending
on the release. `docker run --rm -it <image> find / -name inference_pretrained_model.py 2>/dev/null`
reveals the actual path. Alternatively, `docker cp` the script from
the open-source Difix3D+ repo
(<https://github.com/nv-tlabs/Difix3D>) into a host directory and
bind-mount it at `/opt/fixer/src`.

**Aspect-ratio distortion in output**

Inputs are force-resized to 1024×576 and restored to the original
resolution after the model runs. For non-16:9 content (4:3, 1:1,
portrait), the model still improves perceptual quality and temporal
coherence, but pixel-exact fidelity can suffer. There is no user-
facing flag to change the internal resolution — the checkpoint was
trained at 1024×576.

**Output count does not match input count**

Confirm all inputs end in `.png`, `.jpg`, or `.jpeg`. Remove any
hidden `.DS_Store`, `Thumbs.db`, or editor backup files. The script
skips unsupported formats silently.

**Fresh instance missing `${HOME}/.cache/nre/difix/`**

Only relevant when consuming Fixer via NRE. On a fresh Brev / cloud
instance, the cache is not pre-populated. Run any NRE pipeline with
`--enable-difix --difix-cache=${HOME}/.cache/nre/difix`; NRE
downloads the weights on first use. Alternatively, fetch the
weights directly with `huggingface-cli download nvidia/Fixer` and
point `--difix-cache` at the directory you downloaded into.

## Quality

**Visible seam at frame 5 of a sequence**

Expected — the first few frames warm up the temporal reference
window. Either discard them, or prepend copies of the first frame
and drop those copies from the output.

**PSNR regresses after applying Fixer**

Fixer optimises perceptual quality (LPIPS) over pixel-exact fidelity
(PSNR). A small PSNR drop paired with a larger LPIPS improvement is
expected. If PSNR drops dramatically (> 2 dB) on real reconstruction
outputs, the render domain likely differs from the pretraining
distribution — consider fine-tuning Fixer on in-domain data (not
covered by this external skill).

**Grid-like artifacts in output**

Known failure mode on some render domains. At inference time, try a
different `--timestep` value (e.g. 100, 500) to see if the artifact
diminishes. For fine-tuning users, reducing the perceptual loss scale
(`--lambda_lpips 0.03`) is the documented mitigation.
