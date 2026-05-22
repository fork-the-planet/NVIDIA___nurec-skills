# Unified Fixer & Harmonizer — Extended Troubleshooting

## Installation / environment

**`docker: Error response from daemon: could not select device driver "" with capabilities: [[gpu]]`**

The NVIDIA Container Toolkit is not installed or not configured for
the Docker runtime. Install from
<https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html>
and restart the Docker daemon.

**`Unable to find image 'nvcr.io/nvidia/pytorch:24.10-py3' locally`** followed by a 401 / 403

NGC requires authentication to pull NVIDIA-hosted containers. Run
`docker login nvcr.io` with the literal username `$oauthtoken` and
your `NGC_API_KEY` as the password (use `--password-stdin` to avoid
echoing the key into shell history). Generate a key at
<https://ngc.nvidia.com/setup/personal-keys>.

**`ngc registry model download-version` returns 401 / 403**

The NGC CLI is not authenticated with a key that has access to the
`nvidia/nre` model namespace. Confirm `NGC_API_KEY` is exported in
the same shell, then run `ngc config set` to wire the CLI to the
`nvidia` org and `nre` team. Re-issuing a personal key from
<https://ngc.nvidia.com/setup/personal-keys> usually clears expired-key
errors. If the model card itself appears empty in a browser, your
account may not yet have NRE entitlement — file a support ticket
rather than retrying the CLI.

**`ngc: command not found`**

The NGC CLI is not on PATH. Install per
<https://docs.ngc.nvidia.com/cli/>. As an alternative for one-off
downloads, the model card on the NGC site has a "Download" button
that returns a tarball; unpack it next to where the CLI would have
written `nurec-fixer_vcosmos_3dgut_fixer_harmonizer/`.

## Runtime

**Script can't find `harmonizer_temporal.pt` / `harmonizer_nontemporal.pt`**

Inside the container, the host artifact directory must be mounted
at `/work` (read-only is fine). Double-check the `-v` flag in the
`docker run` command and that `--temporal_model_path` /
`--nontemporal_model_path` point at the path **inside** the
container (`/work/harmonizer_temporal.pt`,
`/work/harmonizer_nontemporal.pt`), not the host path.

**`unrecognized arguments: --nontemporal_model_path`**

The beta artifact has been observed to spell this argument both as
`--nontemporal_model_path` (in the inference example) and
`--non_temporal_model_path` (in the argument table). Run
`python /work/inference_jit_harmonizer.py --help` inside the
container and use whichever name the live `--help` output advertises;
the underlying parameter is the same. Do not patch the script — the
.pt files are wired to the script's exact `argparse` definition.

**`pip install -r requirements.txt` fails with permission denied**

You probably dropped `-e PYTHONUSERBASE=/tmp/pyuser` from the
`docker run` command. The container's site-packages directory is
owned by root and cannot be written to under `-u $(id -u):$(id -g)`.
Either keep `--user` + a writable `PYTHONUSERBASE`, or bake a wrapper
image per [`wrapper-image.md`](wrapper-image.md) so the install
happens at build time as root.

**`RuntimeError: PytorchStreamReader failed reading zip archive`** when loading a `.pt` file

The .pt download was truncated. Run
`du -h nurec-fixer_vcosmos_3dgut_fixer_harmonizer/*.pt` — both files
should be on the order of 1–2 GB each. If either is much smaller,
delete the artifact directory and re-run
`ngc registry model download-version`.

**`RuntimeError: Could not run 'aten::...' with arguments from the 'CUDA' backend`** loading a `.pt` file

You're running outside `nvcr.io/nvidia/pytorch:24.10-py3`. The JIT
modules embed CUDA-specific kernels and only load reliably under the
PyTorch / CUDA / cuDNN stack the artifact was traced against. Always
run inference inside the documented base image; do not try to load
the .pt files in a host Python environment.

**Output files owned by root after a `docker run`**

The container ran without `-u $(id -u):$(id -g)`. Run
`sudo chown -R "$(id -u):$(id -g)" <output_dir>` once, then re-issue
the original command **with** the `-u` flag so subsequent runs land
correctly. Every `docker run` snippet in `SKILL.md` includes this
flag; verify your shell history did not drop it.

**Aspect-ratio distortion in output**

Inputs are force-resized to 1024×576 and restored to the original
resolution after the model runs. For non-16:9 content (4:3, 1:1,
portrait), the model still improves perceptual quality and temporal
coherence, but pixel-exact fidelity can suffer. There is no
user-facing flag to change the internal resolution — the JIT models
were traced at 1024×576.

**Output count does not match input count**

Confirm all inputs end in `.png` or `.jpg`. Remove any hidden
`.DS_Store`, `Thumbs.db`, or editor backup files. The script skips
unsupported formats silently.

**Fresh instance missing `${HOME}/.cache/nre/difix/`**

Only relevant when consuming the harmonizer via NRE. On a fresh Brev
/ cloud instance, the cache is not pre-populated. Run any NRE
pipeline with `--enable-difix --difix-cache=${HOME}/.cache/nre/difix`;
NRE will pull the same NGC harmonizer artifact on first use.
Alternatively, fetch the artifact directly with
`ngc registry model download-version "nvidia/nre/nurec-fixer:cosmos_3dgut_fixer_harmonizer"`
and point `--difix-cache` at the directory you downloaded into.

## Quality

**Visible seam at frame 5 of a sequence**

Expected — the first 4 frames go through `harmonizer_nontemporal.pt`
to warm up the temporal reference window; frame 5 onward switches to
`harmonizer_temporal.pt`. Either discard the warm-up frames, or
prepend copies of the first frame and drop those copies on output.

**Quality regresses on a re-run**

You re-ran into a non-empty `output_dir`. The temporal pass uses the
previous 4 *output* frames as references, so stale outputs poison the
window. Always start from an empty output directory.

**PSNR regresses after harmonization**

The harmonizer optimises perceptual quality (LPIPS) over pixel-exact
fidelity (PSNR), inheriting the Difix3D+ training objective. A small
PSNR drop paired with a larger LPIPS improvement is expected. If PSNR
drops dramatically (> 2 dB) on real reconstruction outputs, the render
domain likely differs from the pretraining distribution — fine-tuning
the harmonizer on in-domain data is not covered by this external
skill.

**Grid-like artifacts in output**

Known beta failure mode on some render domains. The unified beta
release does not expose a tunable diffusion timestep flag (the JIT
models bake the schedule in), so the documented mitigations are:
verify input filenames truly natural-sort, verify
`harmonizer_temporal.pt` is being used after the warm-up window
(`grep -i temporal` the script's stderr), and report the failure on
the NGC model card so the next release can address it.

## Filenames

**`frame_1.png ... frame_10.png` ordering**

Lexicographic (default) sort places `frame_10.png` **before**
`frame_2.png`, which silently misorders the temporal references. Use
zero-padded fixed-width indices (`frame_0001.png ... frame_0010.png`)
or pre-sort with the natural-sort key documented in the upstream
inference script. The harmonizer reads the *previous 4 outputs* as
references, so misordering compounds quality loss across the entire
sequence.
