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

Once the image is built, the canonical `docker run` invocation, the
full flag matrix for `inference_pretrained_model.py`, and the
temporal-variant entry point all live in
[`inference.md`](inference.md). Keep the downloaded weights under
`$CODE_DIR/models` (see the `hf download` step in `inference.md §
Model download`) so the `-v "$CODE_DIR":/work` mount picks them up at
`/work/models/pretrained/pretrained_harmonizer.pkl`.

## Raw Container Mode

If you run directly from
`nvcr.io/nvidia/cosmos/cosmos-predict2-container:1.2` instead of the
project image, install the checkout's requirements and apply the
Blackwell / training patches called out in
[`inference.md § Container setup`](inference.md#container-setup). The
project Dockerfile applies these patches during build, so this step
is only needed in raw-container mode.

## Rebuild Policy

Rebuild the image whenever:

- The `NVIDIA/harmonizer` checkout changes.
- The base Cosmos Predict2 container tag changes.
- `requirements.txt`, `tokenizer.patch`, or `text2image_dit.patch` changes.

Treat the image as disposable. It should not bake in `HF_TOKEN`,
`NGC_API_KEY`, datasets, or downloaded model checkpoints.
