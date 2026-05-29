#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
SAMPLE_DATA_DIR="${SAMPLE_DATA_DIR:-${SCRIPT_DIR}/sample_data}"
RUN_ID="${RUN_ID:-latest}"

docker run --shm-size=64g -it --rm --gpus all \
    -u "$(id -u):$(id -g)" \
    --net=host \
    --privileged \
    --ulimit nofile=65536:65536 \
    --volume "${SAMPLE_DATA_DIR}":/workdir/output \
    --volume "${SAMPLE_DATA_DIR}/${RUN_ID}/usd-out":/workdir/data \
    nvcr.io/nvidia/nre/nre-ga:latest \
    render-grpc \
    --artifact-path /workdir/data/last.usdz \
    --output-dir /workdir/output/reconstructed_scenes
