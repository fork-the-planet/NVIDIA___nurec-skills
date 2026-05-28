#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
SAMPLE_DATA_DIR="${SAMPLE_DATA_DIR:-${SCRIPT_DIR}/sample_data}"
CLIP_ID="${CLIP_ID:-5479852f-647e-41e6-9a25-2d75bc8f6e46}"

docker run --shm-size=64g -it --rm --ulimit nofile=2048:2048 --gpus all \
    -u "$(id -u):$(id -g)" \
    -e NGC_API_KEY="${NGC_API_KEY}" \
    --volume "${SAMPLE_DATA_DIR}":/workdir/dataset \
    --volume "${SAMPLE_DATA_DIR}":/workdir/output \
    nvcr.io/nvidia/nre/nre-ga:latest \
    --config-name=configs/apps/AV/Waymo/3dgut_dynamic.yaml \
    mode=train \
    out_dir=/workdir/output \
    dataset.path=/workdir/dataset/pai_${CLIP_ID}/pai_${CLIP_ID}.json \
    dataset.lidar_ids="[lidar_top_360fov]" \
    dataset.camera_ids="[camera_front_wide_120fov,camera_cross_left_120fov,camera_cross_right_120fov,camera_rear_left_70fov,camera_rear_right_70fov]" \
    logger=tensorboard \
    logger.run_id="latest" \
    checkpoint.artifact.enabled=true \
    checkpoint.artifact.rig_trajectories.enabled=true \
    checkpoint.artifact.sequence_tracks.enabled=true \
    dataset.n_train_sample_camera_rays=3072
