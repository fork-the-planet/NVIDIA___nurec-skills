#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
SAMPLE_DATA_DIR="${SAMPLE_DATA_DIR:-${SCRIPT_DIR}/sample_data}"
RUN_ID="${RUN_ID:-latest}"

docker run --shm-size=64g -it --rm --gpus all \
    -u "$(id -u):$(id -g)" \
    --net=host \
    --privileged \
    -e NGC_API_KEY="${NGC_API_KEY}" \
    --volume "${SAMPLE_DATA_DIR}":/workdir/data \
    nvcr.io/nvidia/nre/nre-ga:latest \
    serve-grpc \
    --artifact-glob "/workdir/data/${RUN_ID}/usd-out/last.usdz" \
    --renderer default \
    --enable-editing-actors \
    --test-scenes-are-valid
