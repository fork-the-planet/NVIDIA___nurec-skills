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

```bash
docker run --gpus=all --rm --ipc=host \
  -u "$(id -u):$(id -g)" \
  -v "$CODE_DIR":/work \
  -v /absolute/path/to/test_dataset:/test_dataset \
  -w /work \
  harmonizer-cosmos-env \
  python /work/src/evaluate_test_dataset.py \
    --model /work/models/pretrained/pretrained_harmonizer.pkl \
    --input /test_dataset
```

## Expected outputs

- Enhanced images under an `evaluation/` directory that mirrors
  the test dataset structure.
- `metrics.yaml` with overall and per-scene PSNR/LPIPS, inference
  time, image counts, and GPU memory statistics.
