#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
SAMPLE_DATA_DIR="${SAMPLE_DATA_DIR:-${SCRIPT_DIR}/sample_data}"
RUN_ID="${RUN_ID:-latest}"

docker run --shm-size=64g -it --rm --ulimit nofile=2048:2048 --gpus all \
    -u "$(id -u):$(id -g)" \
    -e NGC_API_KEY="${NGC_API_KEY}" \
    --volume "${SAMPLE_DATA_DIR}":/workdir/dataset \
    --volume "${SAMPLE_DATA_DIR}":/workdir/output \
    nvcr.io/nvidia/nre/nre-ga:latest \
    export-usdz-artifact \
    --config-name=/workdir/output/${RUN_ID}/config/parsed.yaml \
    --checkpoint-name=last.ckpt \
    --output-dir=/workdir/output/${RUN_ID}/usd-out
