#!/usr/bin/env bash
# Train a NuRec reconstruction on a single Physical AI Autonomous
# Vehicles (PAI) clip already converted to NCore V4.
#
# PAI clips MUST use the Hyperion-8.1 `car2sim_6cam` recipe, NOT the
# Waymo `configs/apps/AV/Waymo/3dgut_dynamic.yaml` recipe. The latter
# bakes in the Waymo Open Dataset sensor rig and is not valid for PAI.
#
# We drive training off the small overlay shipped at
# `references/configs/pai.yaml`, which uses
# `/apps/prod/Hyperion-8.1/car2sim_6cam.yaml` as its `defaults` base
# and adds the PAI six-camera validation set, the `lidar_top_360fov`
# lidar id, and lidar `intensity` supervision. The overlay is mounted
# into the in-container Hydra config tree as
# `${NRE_CONFIG_DIR}/external_overrides.yaml`, then selected with
# `--config-name=external_overrides`.
set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
SAMPLE_DATA_DIR="${SAMPLE_DATA_DIR:-${SCRIPT_DIR}/sample_data}"
CLIP_ID="${CLIP_ID:-5479852f-647e-41e6-9a25-2d75bc8f6e46}"

# Path to the PAI overlay shipped with this skill.
PAI_OVERLAY="${PAI_OVERLAY:-${SCRIPT_DIR}/../../configs/pai.yaml}"

# In-container Hydra config dir for the GA image. Override if a future
# image moves the runfiles tree (`docker run --rm --entrypoint sh
# nvcr.io/nvidia/nre/nre-ga:latest -c 'ls /app/run.runfiles/_main/configs'`).
NRE_CONFIG_DIR="${NRE_CONFIG_DIR:-/app/run.runfiles/_main/configs}"

docker run --shm-size=64g -it --rm --ulimit nofile=2048:2048 --gpus all \
    -u "$(id -u):$(id -g)" \
    -e NGC_API_KEY="${NGC_API_KEY}" \
    --volume "${SAMPLE_DATA_DIR}":/workdir/dataset \
    --volume "${SAMPLE_DATA_DIR}":/workdir/output \
    --volume "${PAI_OVERLAY}":"${NRE_CONFIG_DIR}/external_overrides.yaml":ro \
    nvcr.io/nvidia/nre/nre-ga:latest \
    --config-name=external_overrides \
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
