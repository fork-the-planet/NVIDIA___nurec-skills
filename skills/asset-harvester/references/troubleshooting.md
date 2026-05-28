# Asset Harvester — Troubleshooting and teardown

Detail moved out of `SKILL.md`.

## Troubleshooting matrix

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
it has been leaked (see `installation.md` "Verifying secrets
safely"); the token is per-user and shared across HuggingFace
workflows.
