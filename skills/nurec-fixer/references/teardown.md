# Teardown — nurec-fixer / DiffusionHarmonizer

A complete DiffusionHarmonizer workflow can leave **100 GB+** on disk,
especially if you build the runtime image and download the optional
training dataset.

| Artifact | Approximate size | Source |
|----------|------------------|--------|
| `harmonizer-cosmos-env` image (base `nvcr.io/nvidia/pytorch:25.10-py3`) | tens of GB | `docker build` / `docker pull` |
| Docker build cache | tens of GB | `docker build -f Dockerfile.cosmos` |
| `harmonizer/` code checkout | repo-dependent | `git clone https://github.com/NVIDIA/harmonizer.git` |
| `models/` with `diffusion_harmonizer.pkl` | model-dependent | `./download_checkpoints.sh` (`nvidia/Harmonizer`) |
| `src/checkpoints/nvidia/Cosmos-Predict2-0.6B-Text2Image/` | model-dependent | `./download_checkpoints.sh` (base Cosmos model) |
| Hugging Face hub cache | model/dataset-dependent | `hf download` |
| `data/` training dataset | large | `./download_checkpoints.sh --with-dataset` (`nvidia/Harmonizer-Dataset`) |
| Enhanced/evaluation outputs | sequence-dependent | inference/evaluation runs |

## Reclaim Disk

Run only the blocks that apply to your host.

### 1. Container Images And Build Cache

```bash
docker image rm harmonizer-cosmos-env 2>/dev/null || true
docker image rm nvcr.io/nvidia/pytorch:25.10-py3 2>/dev/null || true
docker image prune -f
docker builder prune -f
```

### 2. Code Checkout, Model, And Dataset Copies

```bash
rm -rf /absolute/path/to/harmonizer/models
rm -rf /absolute/path/to/harmonizer/src/checkpoints
rm -rf /absolute/path/to/harmonizer/data
rm -rf /absolute/path/to/harmonizer
```

If you downloaded model or dataset artifacts elsewhere, remove those
paths instead.

### 3. Hugging Face Cache

For targeted cleanup, inspect the cache first:

```bash
hf cache ls | grep -E 'Harmonizer|Cosmos-Predict2' || true
```

Then delete the specific cached repos with `hf cache delete`, or remove
only the known repo cache directories under
`${HF_HOME:-$HOME/.cache/huggingface}/hub/` if you are sure they are not
shared by another workflow.

### 4. Outputs

```bash
rm -rf /absolute/path/to/enhanced_frames
rm -rf /absolute/path/to/test_dataset/evaluation
rm -f /absolute/path/to/test_dataset/metrics.yaml
```

## Outputs Already Owned By Root

If a previous `docker run` omitted `-u $(id -u):$(id -g)`, fix ownership
before deleting or editing outputs:

```bash
sudo chown -R "$(id -u):$(id -g)" /absolute/path/to/enhanced_frames
rm -rf /absolute/path/to/enhanced_frames
```

## Verify

```bash
docker images | grep -E 'harmonizer-cosmos-env|nvcr.io/nvidia/pytorch' || echo "images: clean"
du -sh /absolute/path/to/harmonizer 2>/dev/null || echo "checkout: clean"
```

## Secrets

Do **not** revoke `HF_TOKEN` or `NGC_API_KEY` as part of routine
teardown. Rotate a token only if you suspect it was printed, committed,
or otherwise leaked. Use length-only checks such as `${#HF_TOKEN}`;
never echo token values.
