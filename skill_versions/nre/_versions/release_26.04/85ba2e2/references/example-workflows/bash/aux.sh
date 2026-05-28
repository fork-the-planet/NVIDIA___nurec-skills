#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
SAMPLE_DATA_DIR="${SAMPLE_DATA_DIR:-${SCRIPT_DIR}/sample_data}"
CLIP_ID="${CLIP_ID:-5479852f-647e-41e6-9a25-2d75bc8f6e46}"

docker run --shm-size=2g -it --rm --gpus all \
    -u "$(id -u):$(id -g)" \
    -e NGC_API_KEY="${NGC_API_KEY}" \
    --volume "${SAMPLE_DATA_DIR}":/workdir/dataset \
    --volume "${SAMPLE_DATA_DIR}":/workdir/output \
    nvcr.io/nvidia/nre/nre-tools-ga:latest \
    --dataset-path=/workdir/dataset/pai_${CLIP_ID}/pai_${CLIP_ID}.json \
    --output-dir=/workdir/output/pai_${CLIP_ID} \
    --lidar-seg-camvis \
    --no-seg-logits \
    --store-meta \
    --camera-id camera_front_wide_120fov \
    --camera-id camera_front_tele_30fov \
    --camera-id camera_cross_right_120fov \
    --camera-id camera_cross_left_120fov \
    --camera-id camera_rear_left_70fov \
    --camera-id camera_rear_right_70fov
