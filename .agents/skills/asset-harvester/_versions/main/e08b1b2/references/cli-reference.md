# Asset Harvester — CLI reference

Exhaustive flag matrix for every wrapper and entry point. See the top
of `SKILL.md` for the common-case cookbook; this file is the complete
surface.

## `run_inference.py` (direct image-to-3D entry)

| Flag | Default | Purpose |
|------|---------|---------|
| `--diffusion_checkpoint` | *(required)* | Path to `AH_multiview_diffusion.safetensors`. |
| `--lifting_checkpoint` | *(optional)* | Path to `AH_tokengs_lifting.safetensors`. Omit to skip TokenGS lifting (multiview-only output). |
| `--ahc_checkpoint` | *(optional)* | Path to `AH_camera_estimator.safetensors`. **Required** when using `--image_dir` (single-view mode) — an AV camera pose is estimated from the single input. |
| `--data_root` | — | Root of a rectified-samples directory containing `sample_paths.json` (e.g. `data_samples/rectified_AV_objects/` or `outputs/ncore_parser/`). Mutually exclusive with `--image_dir`. |
| `--image_dir` | — | Root of a single-view directory with `<object_id>/{frame.jpeg,mask.png}` sub-folders. Mutually exclusive with `--data_root`. |
| `--output_dir` | `outputs/` | Per-sample outputs (`multiview/`, `3d_lifted/`, `gaussians.ply`, `*.mp4`). |
| `--offload_model_to_cpu` | off | Offload unused model components to CPU to fit < 16 GB VRAM. |

## `run.sh` — step-2 wrapper (diffusion + lifting)

| Flag | Default | Purpose |
|------|---------|---------|
| `--data-root` | `outputs/ncore_parser/` | Input directory containing `sample_paths.json`. |
| `--diffusion-ckpt` | `checkpoints/AH_multiview_diffusion.safetensors` | SparseViewDiT checkpoint. |
| `--lifting-ckpt` | `checkpoints/AH_tokengs_lifting.safetensors` | TokenGS checkpoint. |
| `--output-dir` | `outputs/` | Where per-sample outputs are written. |
| `--num-steps` | 30 | Diffusion inference steps. |
| `--cfg-scale` | 2.0 | Classifier-free guidance scale. |
| `--max-samples` | 0 | Maximum samples to process; `0` = all. |
| `--skip-lifting` | off | Disable TokenGS lifting — emit multiview outputs only. |
| `--offload` | off | Offload diffusion models to CPU while lifting runs. |

Per-sample outputs:

```text
${OUTPUT_DIR}/<sample_id>/
├── multiview/
├── 3d_lifted/
├── gaussians.ply
├── multiview.mp4
└── 3d_lifted.mp4
```

## `scripts/run_ncore_parser.sh` — step-1 wrapper

| Flag | Default | Purpose |
|------|---------|---------|
| `--component-store` | *(required)* | Clip `.json` manifest, comma-separated NCore V4 component-store paths, or `.zarr.itar` glob(s). |
| `--output-path` | `outputs/ncore_parser/` | Output directory for per-track object crops and `sample_paths.json`. |
| `--segmentation-ckpt` | `checkpoints/AH_object_seg_jit.pt` | Mask2Former JIT checkpoint for foreground masking. |
| `--camera-ids` | all 5 default cameras | Comma-separated camera sensor IDs (e.g. `front_wide,front_tele`). |
| `--track-ids` | all tracks | Comma-separated track IDs to process (filter to specific objects). |

The module can also be invoked directly via the installed console
script (`ncore-parser`) or `python -m asset_harvester.ncore_parser`.

## `asset_harvester.utils.image_segment` — standalone segmentation

Generate `mask.png` from `frame.jpeg` for each sub-folder using the
AV-object Mask2Former.

| Flag | Default | Purpose |
|------|---------|---------|
| `--checkpoint` | *(required)* | Path to `AH_object_seg_jit.pt`. |
| `--image_folder` | *(required)* | Directory of `<object_id>/` sub-folders. |
| `--frame_name` | `frame.jpeg` | Input filename to segment. |
| `--mask_name` | `mask.png` | Output mask filename. |

Run via:

```bash
python -m asset_harvester.utils.image_segment --help
```

## `asset_harvester.utils.orient_gaussians_for_nurec`

Rotate Gaussian PLY files into the orientation NuRec external-assets
insertion expects.

| Flag | Default | Purpose |
|------|---------|---------|
| `--input-dir` | *(required)* | Step-2 output directory (contains per-sample `gaussians.ply`). |
| `--output-dir` | — | If given, writes a transformed copy; otherwise requires `--in-place`. |
| `--in-place` | off | Overwrite `gaussians.ply` under `--input-dir`. |
| `--degrees` | 90 | Y-axis rotation applied (degrees). |

## `asset_harvester.utils.generate_external_assets_metadata`

Emit `metadata.yaml` describing each `gaussians.ply` for NuRec
insertion.

| Flag | Default | Purpose |
|------|---------|---------|
| `--input-dir` | *(required)* | Root of the (oriented) lifting output. |

## Installed console scripts (from `pyproject.toml`)

| Script | Target |
|--------|--------|
| `ncore-parser` | `asset_harvester.ncore_parser.__main__:main` |
| `tokengs-train` | `asset_harvester.tokengs.main:main` (see `docs/tokengs.md` in the upstream repo) |

## `benchmark/eval.py` — reconstruction metrics

| Flag | Default | Purpose |
|------|---------|---------|
| `--output_dir` | *(required)* | Root output directory from `run_inference.py` / `run.sh`. |
| `--eval_output_dir` | `<output_dir>/eval` | Where eval results are written. |
| `--output_size` | 512 | Render resolution for comparisons. |
| `--no_comparisons` | off | Skip saving side-by-side comparison images. |

Summary lands in `rendering_metrics_summary.txt`. Must be run inside
the `av-object-benchmark` conda env (clone of `asset-harvester` +
`bash benchmark/install.sh`).
