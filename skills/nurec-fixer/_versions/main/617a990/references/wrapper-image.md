# Wrapper Image — Pre-installed `requirements.txt` for Faster Repeat Inference

The default [Run inference](../SKILL.md#run-inference) recipe installs
the harmonizer artifact's `requirements.txt` on every `docker run` (to
the per-run `PYTHONUSERBASE=/tmp/pyuser`). That is fine for one-off
runs but wastes ~30–60 seconds per invocation on a CI server or batch
host. For repeat-use workflows, bake a small wrapper image once that
pre-installs the deps on top of `pytorch:24.10-py3`.

## Build the wrapper image

From the parent directory of `nurec-fixer_vcosmos_3dgut_fixer_harmonizer/`:

```bash
ARTIFACT_DIR=$(pwd)/nurec-fixer_vcosmos_3dgut_fixer_harmonizer

cat > Dockerfile.harmonizer <<'EOF'
# Layer harmonizer requirements on top of the public PyTorch base.
FROM nvcr.io/nvidia/pytorch:24.10-py3

# Pin the artifact version this image was built for so a future
# version bump is forced through a fresh `docker build`.
LABEL nvidia.harmonizer.version="cosmos_3dgut_fixer_harmonizer"

WORKDIR /opt/harmonizer
COPY artifact/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
EOF

# Stage just the requirements.txt so the build context stays small.
mkdir -p ./build/artifact
cp "$ARTIFACT_DIR/requirements.txt" ./build/artifact/

docker build \
  -f Dockerfile.harmonizer \
  -t nurec-harmonizer:beta \
  ./build
```

Substitute the image tag if your site uses a registry-prefixed name.

## Run inference against the wrapper image

```bash
IMAGE=nurec-harmonizer:beta
ARTIFACT_DIR=$(pwd)/nurec-fixer_vcosmos_3dgut_fixer_harmonizer
INPUT_DIR=/absolute/path/to/rendered_frames
OUTPUT_DIR=/absolute/path/to/enhanced_frames

mkdir -p "$OUTPUT_DIR"
[ -z "$(ls -A "$OUTPUT_DIR" 2>/dev/null)" ] || {
  echo "ERROR: $OUTPUT_DIR is not empty."
  exit 1
}

docker run --gpus=all --rm --ipc=host \
  -u "$(id -u):$(id -g)" \
  -v "$ARTIFACT_DIR":/work:ro \
  -v "$INPUT_DIR":/in:ro \
  -v "$OUTPUT_DIR":/out \
  -w /work \
  "$IMAGE" \
  python /work/inference_jit_harmonizer.py \
    --input_image /in \
    --output_dir /out \
    --temporal_model_path /work/harmonizer_temporal.pt \
    --nontemporal_model_path /work/harmonizer_nontemporal.pt
```

No `pip install` step is needed at runtime, so the image can run
without a writable `PYTHONUSERBASE`. The artifact is still mounted
read-only at `/work` so a future version bump does not require a
rebuild — only a new `ngc registry model download-version` and a
swapped mount.

## Rebuild policy

Re-run `docker build` whenever:

- You bump the harmonizer NGC version (e.g. a future
  `cosmos_3dgut_fixer_harmonizer_v2`) and its `requirements.txt`
  changes.
- NGC publishes a new dated tag of `pytorch:24.10-py3` and you want
  the security updates.

Treat the wrapper image as disposable; it does not contain the model
weights and is cheap to recreate.
