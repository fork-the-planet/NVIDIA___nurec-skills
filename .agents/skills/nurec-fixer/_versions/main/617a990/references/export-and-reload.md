# Exporting and Reloading Fixer for Faster Repeat Inference

For workflows that call Fixer many times (batch jobs, CI, deployment
pipelines), exporting the pretrained model once and reloading the
exported artifact amortises model-loading cost.

## Export the model

Pass `--no-generate-images --export-path <dir>` to the inference
script inside the container. The command processes zero frames and
writes an exportable artifact to `<dir>`:

```bash
docker run --gpus=all --rm --ipc=host \
  -v "$MODELS_DIR":/models:ro \
  -v "$(pwd)/exported_model":/exported \
  "$IMAGE" \
  python /opt/fixer/src/inference_pretrained_model.py \
    --model /models/pretrained/pretrained_fixer.pkl \
    --timestep 250 \
    --no-generate-images \
    --batch-size 1 \
    --export-path /exported
```

Pin `--timestep` and `--batch-size` at export time to the values you
intend to reuse — changing them at reload time requires re-exporting.

## Reload the exported model

Inside the container, use the helper loader:

```python
from inference_exported_model import load_exported_model

model = load_exported_model(
    "exported_model",
    timestep=250,
    vae_skip_connection=False,
    batch_size=1,
    image_size=(576, 1024),
    device="cuda",
    dtype="bfloat16",
)
# model(...) now runs per-frame inference without re-loading the checkpoint.
```

Treat `exported_model/` as an opaque artifact — the layout is
implementation-specific and may change across Fixer releases. Do not
commit it to version control; re-export whenever you upgrade the
container or the pretrained checkpoint.
