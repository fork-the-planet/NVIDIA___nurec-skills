# DiffusionHarmonizer — Dataset, training, and NuRec pair recipes

Detail moved out of `SKILL.md`. Run only when the user explicitly
asks for training, fine-tuning, or evaluation-pair construction.

## Dataset

The release branches describe a training set of approximately
350K curated synthetic-real image pairs from five complementary
curation pipelines. Download the assembled dataset when the user
wants to train, fine-tune, or reproduce evaluation:

```bash
hf download nvidia/Harmonizer-Dataset \
  --repo-type dataset \
  --local-dir data
```

The `./download_checkpoints.sh --with-dataset` helper performs the same
download as part of fetching the inference checkpoints.

Data sources and targeted failure modes:

| Data source | Failure mode |
|-------------|--------------|
| ISP Modification | ISP-induced color or tone drift between foreground and background. |
| Relighting | Illumination mismatch between inserted objects and scene lighting. |
| Asset Re-insertion | Missing shadows and appearance mismatch when dynamic assets are re-inserted. |
| PBR Shadow Simulation | Missing or unrealistic cast shadows on inserted objects. |
| Artifacts Correction | Novel-view artifacts such as blur, missing regions, ghosting, and spurious geometry. |

Training JSON format:

```json
{
  "train": {
    "{data_id}": {
      "image": "{PATH_TO_IMAGE}",
      "target_image": "{PATH_TO_TARGET_IMAGE}",
      "prompt": "remove degradation"
    }
  },
  "test": {
    "{data_id}": {
      "image": "{PATH_TO_IMAGE}",
      "target_image": "{PATH_TO_TARGET_IMAGE}",
      "prompt": "remove degradation"
    }
  }
}
```

## Training

Recommended multi-GPU training shape from the release README:

```bash
export NUM_NODES=1
export NUM_GPUS=8
export OUTPUT_DIR=/path/to/checkpointing_directory
export DATASET_FOLDER=/data/data.json
export WANDB_MODE=offline

accelerate launch \
  --mixed_precision=bf16 \
  --main_process_port 29501 \
  --multi_gpu \
  --num_machines "$NUM_NODES" \
  --num_processes "$NUM_GPUS" \
  src/train_pix2pix_turbo_harmonizer.py \
    --output_dir="${OUTPUT_DIR}" \
    --dataset_folder="${DATASET_FOLDER}" \
    --max_train_steps 10000 \
    --learning_rate 2e-5 \
    --train_batch_size=1 \
    --gradient_accumulation_steps 1 \
    --dataloader_num_workers 8 \
    --checkpointing_steps=2000 \
    --eval_freq 1000 \
    --viz_freq 1000 \
    --train_image_prep resize_576x1024 \
    --test_image_prep resize_576x1024 \
    --lambda_clipsim 0.0 \
    --lambda_lpips 0.3 \
    --lambda_gan 0.0 \
    --lambda_l2 1.0 \
    --lambda_gram 0.0 \
    --use_sched \
    --report_to wandb \
    --tracker_project_name cosmos_harmonizer \
    --tracker_run_name train \
    --train_full_unet \
    --timestep 250 \
    --track_val_fid \
    --num_samples_eval 20 \
    --mixed_precision=bf16
```

For fine-tuning from the released checkpoint, add:

```bash
--pretrained_path /path/to/diffusion_harmonizer.pkl
```

When omitted, the model is fine-tuned directly from the raw Cosmos 0.6B
image model.

For the released dataset, the README recommends
`--fixing_data_weight 3` to up-weight artifact-correction
examples.

## NuRec data-pair recipes

The updated tutorials describe four ways to construct image pairs
for training or testing. Three are shown with NuRec command
templates:

- **Sparse reconstruction:** train with every Nth frame and pair
  held-out ground-truth images with rendered novel views.
- **Cycle reconstruction:** listed as a supported pair-generation
  method in the tutorial overview.
- **Model underfitting:** train a reconstruction for a reduced
  schedule (roughly 25%-75%) and pair degraded renders with clean
  targets.
- **Cross reference:** train using one camera and render/evaluate
  held-out cameras.

When adapting the sample NRE commands, keep public container
names and current NRE recipes from the sibling `nre` skill. Do
not copy internal container names or internal dataset paths into
user-facing instructions.
