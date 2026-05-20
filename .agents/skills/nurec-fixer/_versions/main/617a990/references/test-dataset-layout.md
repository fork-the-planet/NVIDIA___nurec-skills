# Fixer Evaluation — Test Dataset Layout

The evaluation script expects a directory tree with paired
`render/` (model input) and `gt/` (ground-truth) subtrees per scene.
The two subtrees MUST mirror each other exactly: every file
`render/<camera_id>/<timestamp>.png` MUST have a matching
`gt/<camera_id>/<timestamp>.png`.

## Directory structure

```text
test_dataset/
├── <scene_id_1>/
│   ├── render/
│   │   ├── <camera_id_1>/
│   │   │   ├── <timestamp_1>.png
│   │   │   ├── <timestamp_2>.png
│   │   │   └── ...
│   │   ├── <camera_id_2>/
│   │   │   └── ...
│   │   └── ...
│   └── gt/
│       ├── <camera_id_1>/
│       │   ├── <timestamp_1>.png
│       │   ├── <timestamp_2>.png
│       │   └── ...
│       └── ...
├── <scene_id_2>/
│   ├── render/
│   └── gt/
└── ...
```

## Naming conventions

- **`<scene_id>`** — unique per scene. A UUID such as
  `05443ac1-8125-4ace-9daa-6ecdc0df43ff` works well. Avoid spaces.
- **`render/`** — degraded / rendered images fed to the model as input.
- **`gt/`** — ground-truth camera images used for metric comparison.
- **`<camera_id>`** — stable camera identifier, e.g.
  `camera_front_wide_120fov`, `camera_front_tele_30fov`. Same string in
  both `render/` and `gt/`.
- **`<timestamp>.png`** — timestamp-based filename. Zero-pad to a
  fixed width to avoid natural-sort surprises.

## Supported formats

- `.png`, `.jpg`, `.jpeg`.
- Any resolution; images are resized to 1024×576 internally and
  restored to the original `(W, H)` on output.
- For metric-critical work, prefer `.png` (lossless).

## Generating the dataset

External customers typically generate test datasets from an NRE /
NuRec reconstruction run with held-out camera frames (sparse
reconstruction, model-underfitting, or cross-reference protocols).
See the NRE skill for command shapes and the Difix3D+ paper
(<https://arxiv.org/abs/2503.01774>) for protocol details.

## Output of evaluation

Running the evaluation script emits `metrics.yaml` in the output
directory:

```yaml
overall:
  psnr_mean: 29.335
  psnr_std: 2.028
  lpips_mean: 0.109
  lpips_std: 0.043
  avg_inference_time: 0.025
  total_images: 117
per_scene:
  05443ac1-8125-4ace-9daa-6ecdc0df43ff:
    psnr: 28.716
    lpips: 0.187
    num_images: 12
    avg_inference_time: 0.025
```

The same directory also holds the enhanced images under an
`evaluation/` sub-tree mirroring the test dataset layout.
