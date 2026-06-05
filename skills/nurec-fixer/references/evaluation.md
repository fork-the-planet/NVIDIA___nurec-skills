# DiffusionHarmonizer — Quantitative evaluation

For PSNR / LPIPS evaluation, prepare paired data with this
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

`render/` and `gt/` must have identical camera subdirectories
and matching filenames. Images may be PNG, JPEG, or JPG.

## Run evaluation

`evaluate_test_dataset.py` loads the base Cosmos model from
`src/checkpoints/`, so mount the whole checkout at `/work` and run from
`/work/src`. Place the test dataset under `/work` as well (or mount it
read-write so the script can write `evaluation/` and `metrics.yaml`):

```bash
docker run --gpus=all --rm --ipc=host \
  -u "$(id -u):$(id -g)" \
  -v "$CODE_DIR":/work \
  -v /absolute/path/to/test_dataset:/work/test_dataset \
  -w /work/src \
  harmonizer-cosmos-env \
  python evaluate_test_dataset.py \
    --model /work/models/diffusion_harmonizer.pkl \
    --input /work/test_dataset \
    --output /work/test_dataset/evaluation
```

Add `--calculate-for-input` to also report metrics between the raw
input renders and ground truth (in addition to enhanced output vs. GT).

## Expected outputs

- Enhanced images under the `--output` directory (default
  `evaluation/`) that mirrors the test dataset structure.
- `metrics.yaml` in the output directory with overall and per-scene
  PSNR/LPIPS, inference time, image counts, and GPU memory statistics.
