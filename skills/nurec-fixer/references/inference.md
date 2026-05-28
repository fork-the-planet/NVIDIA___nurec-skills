# DiffusionHarmonizer — Container setup, model download, inference

Detail moved out of `SKILL.md`.

## Container setup

The release branches target the Cosmos Predict2 environment. The
preferred external path is to clone the public code and build the
image from its `Dockerfile.cosmos`:

```bash
CODE_DIR=$PWD/harmonizer
git clone https://github.com/NVIDIA/harmonizer.git "$CODE_DIR"
cd "$CODE_DIR"

docker build \
  -t harmonizer-cosmos-env \
  -f Dockerfile.cosmos \
  .
```

If the project Dockerfile is unavailable in a particular release
checkout, run from the pinned public Cosmos Predict2 image and
install only the packages listed by that checkout:

```bash
BASE=nvcr.io/nvidia/cosmos/cosmos-predict2-container:1.2

docker run --gpus=all --rm -it --ipc=host \
  -u "$(id -u):$(id -g)" \
  -v "$CODE_DIR":/work \
  -w /work \
  "$BASE" \
  bash
```

For Blackwell GPUs such as B200 or GB200, the README branch calls
out a Text2ImageDIT patch before inference when using the raw
container:

```bash
patch /usr/local/lib/python3.12/dist-packages/cosmos_predict2/models/text2image_dit.py text2image_dit.patch
```

For training, apply the tokenizer patch as well if the project
image did not already apply it:

```bash
patch /usr/local/lib/python3.12/dist-packages/cosmos_predict2/tokenizers/tokenizer.py tokenizer.patch
```

The project image should bake these patches in. Only apply them
manually when running directly inside a raw base container.

## Model download

Install and authenticate the Hugging Face CLI, then download the
model:

```bash
python3 -m pip install --user "huggingface_hub[cli]"
export HF_TOKEN=<your-hugging-face-token>
hf auth login --token "$HF_TOKEN"
hf download nvidia/DiffusionHarmonizer --local-dir models
```

Expected checkpoint:

```bash
ls -lh models/pretrained/pretrained_harmonizer.pkl
```

Never commit `models/`, the downloaded checkpoint, or `HF_TOKEN`.

## Run inference (`src/inference_pretrained_model.py`)

```bash
IMAGE=harmonizer-cosmos-env
CODE_DIR=/absolute/path/to/harmonizer
MODEL_PATH=/work/models/pretrained/pretrained_harmonizer.pkl
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
    --model "$MODEL_PATH" \
    --input /input \
    --output /output \
    --timestep 250 \
    --resolution 1024
```

Useful flags exposed by the public inference script:

| Flag | Default | Purpose |
|------|---------|---------|
| `--model` | required | Path to `pretrained_harmonizer.pkl`. |
| `--input` | required | Directory of input frames. |
| `--output` | `output` | Directory for enhanced frames. |
| `--timestep` | script default (README uses `250`) | Diffusion timestep used by the distilled model. |
| `--resolution` | `1024` | Internal size key. `1024` maps to `1024x576`. Other supported keys: `960`, `1360`, `704`, `512`, `256`, `1920`. |
| `--batch_size` | `8` for benchmarking | Batch size for speed testing. Image generation path currently processes frames one at a time. |
| `--max_frames` | very large | Maximum number of frames to process. |
| `--skip_frames` | `1` | Process every Nth frame. |
| `--save_video` | off | Save an MP4 from the output folder. |
| `--test-speed` | off | Run the built-in speed benchmark. |

The branch model card reports H100 testing and an inference time
of about `212 ms` on one H100. The code also has a local
speed-test mode; trust the measurement from your target hardware
for capacity planning.

## Temporal variant

The newer branches also include
`src/inference_pix2pix_turbo_harmonizer.py`, which runs a
temporal/autoregressive variant using prior enhanced frames as
references. Use it when the user explicitly asks for temporal
conditioning rather than the standard pretrained model script.

Key differences:

- Accepts `--input_image`, `--model_path`, `--model_identifier`,
  `--offset_list`, and `--nontemporal`.
- Writes to a sibling output folder named
  `<input_dir>_<model_identifier>` instead of taking `--output`.
- Default `--offset_list -1 -2 -3 -4` uses the four previous
  enhanced frames as temporal references once enough history
  exists.
- Early frames fall back to a non-temporal pass.

Example:

```bash
docker run --gpus=all --rm --ipc=host \
  -u "$(id -u):$(id -g)" \
  -v "$CODE_DIR":/work \
  -v "$INPUT_DIR":/input:ro \
  -w /work \
  harmonizer-cosmos-env \
  python /work/src/inference_pix2pix_turbo_harmonizer.py \
    --model_path /work/models/pretrained/pretrained_harmonizer.pkl \
    --input_image /input \
    --model_identifier harmonized \
    --timestep 250
```

If reproducibility matters, keep each run's output folder
separate so old temporal outputs cannot be reused accidentally.
