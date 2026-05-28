# Runtime Image — DiffusionHarmonizer

The current DiffusionHarmonizer release is intended to run in a Cosmos
Predict2 environment. For repeat inference, build the project image once
from the public `NVIDIA/harmonizer` checkout instead of installing
packages interactively on every run.

## Build The Image

```bash
CODE_DIR=$PWD/harmonizer
git clone https://github.com/NVIDIA/harmonizer.git "$CODE_DIR"
cd "$CODE_DIR"

docker build \
  -t harmonizer-cosmos-env \
  -f Dockerfile.cosmos \
  .
```

If your site requires authenticated NGC pulls, login first:

```bash
echo "$NGC_API_KEY" | docker login nvcr.io --username '$oauthtoken' --password-stdin
```

The release README names `nvcr.io/nvidia/cosmos/cosmos-predict2-container:1.2`
as the base environment. Do not use `:latest` in reproducible workflows.

## Run Inference

```bash
IMAGE=harmonizer-cosmos-env
CODE_DIR=/absolute/path/to/harmonizer
INPUT_DIR=/absolute/path/to/rendered_frames
OUTPUT_DIR=/absolute/path/to/enhanced_frames

mkdir -p "$OUTPUT_DIR"

docker run --gpus=all --rm --ipc=host \
  -u "$(id -u):$(id -g)" \
  -v "$CODE_DIR":/work \
  -v "$INPUT_DIR":/input:ro \
  -v "$OUTPUT_DIR":/output \
  -w /work \
  "$IMAGE" \
  python /work/src/inference_pretrained_model.py \
    --model /work/models/pretrained/pretrained_harmonizer.pkl \
    --input /input \
    --output /output \
    --timestep 250 \
    --resolution 1024
```

Keep the model weights under `$CODE_DIR/models` by running:

```bash
hf download nvidia/DiffusionHarmonizer --local-dir "$CODE_DIR/models"
```

## Raw Container Mode

If you run directly from `nvcr.io/nvidia/cosmos/cosmos-predict2-container:1.2`,
install the checkout's requirements and apply the patches documented in
`SKILL.md`. For Blackwell inference, the README calls out
`text2image_dit.patch`; for training, it also calls out `tokenizer.patch`.
The project image should apply these patches during build.

## Rebuild Policy

Rebuild the image whenever:

- The `NVIDIA/harmonizer` checkout changes.
- The base Cosmos Predict2 container tag changes.
- `requirements.txt`, `tokenizer.patch`, or `text2image_dit.patch` changes.

Treat the image as disposable. It should not bake in `HF_TOKEN`,
`NGC_API_KEY`, datasets, or downloaded model checkpoints.
