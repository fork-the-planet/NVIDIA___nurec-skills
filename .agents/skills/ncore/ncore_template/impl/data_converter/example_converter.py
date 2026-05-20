# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

"""
Example (skeleton) NCore V4 converter.

Start from this file when adding your own format. It demonstrates the converter
contract: get_sequence_ids, from_config, convert_sequence, using
SequenceComponentGroupsWriter and V4 component writers. Input is minimal (empty
root_dir or a single placeholder sequence); output is valid V4 with all major
component types: Poses, Intrinsics, Masks, Camera, Lidar, Radar, and Cuboids.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Literal

import numpy as np


from ncore.data import (
    BBox3,
    CuboidTrackObservation,
    FThetaCameraModelParameters,
    LabelSource,
    OpenCVFisheyeCameraModelParameters,
    OpenCVPinholeCameraModelParameters,
    RowOffsetStructuredSpinningLidarModelParameters,
    ShutterType,
)
from ncore.sensors import RowOffsetStructuredSpinningLidarModel
from ncore.data.v4 import (
    CameraSensorComponent,
    CuboidsComponent,
    IntrinsicsComponent,
    LidarSensorComponent,
    MasksComponent,
    PosesComponent,
    RadarSensorComponent,
    SequenceComponentGroupsReader,
    SequenceComponentGroupsWriter,
)
from ncore.data_converter import FileBasedDataConverter, FileBasedDataConverterConfig
from ncore.impl.common.transformations import HalfClosedInterval  # internal API; no public equivalent yet
from ncore.impl.data.v4.types import ComponentGroupAssignments  # internal API; no public equivalent yet
from upath import UPath


_PLACEHOLDER_JPEG = b"\xff\xd8\xff\xd9"
"""Minimal JPEG (SOI + EOI) used by the skeleton converter.

In a real converter, read actual image files::

    with open(image_path, "rb") as f:
        jpeg_bytes = f.read()
"""

# --- Spinning LiDAR model parameters (shared by intrinsics + frame writers) ---
#
# FILL IN: Replace ALL values below with real values from your sensor datasheet
# or dataset calibration.  Four parameters are silent correctness traps:
#
#   spinning_direction:
#     Determines which way column index increases around the spin axis.
#     "cw"  = clockwise when viewed from above (rig +Z looking down).
#     "ccw" = counter-clockwise when viewed from above.
#     Using the wrong value mirrors the Y-axis of every direction vector and
#     produces a Z-flip in NuRec where surfaces appear reflected through the
#     ground plane.  Verify from the sensor datasheet.
#
#   row_elevations_rad:
#     Must be the actual per-beam inclination angles from the sensor calibration.
#     A synthetic linspace is wrong for every common spinning LiDAR -- all use
#     non-uniform beam spacing.  NCore requires elevations in descending order
#     (highest beam first); reverse the array if your calibration source provides
#     ascending order.  See NCore sensor model docs:
#     https://nvidia.github.io/ncore/data/sensor_models.html#lidar-models
#
#   column_azimuths_rad:
#     Must reflect the actual per-column azimuth of the sensor at frame start.
#     NCore validates ordering via relative_angle(azim[0], azim, spinning_direction)
#     and asserts the result is strictly increasing.  This means each successive
#     column must be further along in the spin direction from column 0.  Raw values
#     can use any range (e.g. (-pi, pi], [0, 2*pi), negative) -- NCore projects to
#     [0, 2*pi) internally before comparing.  For example, linspace(0, 2*pi, N,
#     endpoint=False) works for "ccw" and linspace(0, -2*pi, N, endpoint=False)
#     works for "cw", but these are convenient special cases, not the only valid
#     inputs.  If the dataset embeds per-column pose or heading data, derive
#     azimuths from that rather than a synthetic sweep; a wrong starting azimuth
#     rotates the entire point cloud horizontally.
#
#   row_azimuth_offsets_rad:
#     Per-beam azimuth offset added on top of the column azimuth.  Zero is
#     correct for most sensors.  Some sensors (notably Ouster) provide per-beam
#     azimuth offsets that must be included for accurate LiDAR-to-camera alignment.
#     Check your sensor's calibration data or datasheet.
#
# All common automotive spinning LiDARs (Velodyne, Ouster, Hesai, Robosense)
# spin clockwise ("cw").  The placeholder below defaults to "ccw" -- you almost
# certainly need to change it.  Beam elevations are non-uniform for all families;
# always use real per-beam angles from sensor calibration, not linspace.

_SPINNING_LIDAR_PARAMS = RowOffsetStructuredSpinningLidarModelParameters(
    spinning_frequency_hz=10.0,  # FILL IN: sensor spin rate (Hz)
    spinning_direction="ccw",  # FILL IN: verify from sensor datasheet; most automotive LiDARs are "cw"
    n_rows=4,  # FILL IN: number of laser beams (channels)
    n_columns=360,  # FILL IN: columns per revolution
    row_elevations_rad=np.linspace(
        np.deg2rad(15), np.deg2rad(-15), 4, dtype=np.float32
    ),  # FILL IN: use real per-beam angles (non-uniform); descending order
    column_azimuths_rad=np.linspace(
        0, 2 * np.pi, 360, endpoint=False, dtype=np.float32
    ),  # FILL IN: derive from sensor data if per-column pose is available
    row_azimuth_offsets_rad=np.zeros(
        4, dtype=np.float32
    ),  # FILL IN: per-row azimuth offsets from sensor datasheet (often called beam_azimuth_angles)
)
"""Placeholder spinning LiDAR intrinsics (``lidar_top``)."""


def _evaluate_model_directions(
    params: RowOffsetStructuredSpinningLidarModelParameters,
    model_element: np.ndarray,
) -> np.ndarray:
    """Compute unit-direction vectors from a spinning model for given ``(row, col)`` indices.

    **This helper is for placeholder/smoke-test data only.** Real converters
    should compute directions via direct trigonometry from
    ``column_azimuths_rad`` and ``row_elevations_rad`` (i.e.
    ``[cos_e*cos_az, cos_e*sin_az, sin_e]``), not via this function. Using
    ``elements_to_sensor_rays`` for direction derivation creates a coupling
    between ``column_azimuths_rad`` and ``T_lidar_rig`` that is easy to
    break.

    Delegates to
    ``RowOffsetStructuredSpinningLidarModel.elements_to_sensor_rays``.

    Args:
        params: Spinning LiDAR model parameters.
        model_element: ``[N, 2]`` uint16 array of ``(row, col)`` indices.

    Returns:
        ``[N, 3]`` float32 unit-direction vectors in sensor frame.
    """
    model = RowOffsetStructuredSpinningLidarModel(params, device="cpu")
    # Cast to int64: elements_to_sensor_rays internally converts to
    # torch.long but its uint16→int16 guard rejects values > 32767.
    return model.elements_to_sensor_rays(model_element.astype(np.int64)).cpu().numpy().astype(np.float32)


@dataclass(kw_only=True, slots=True)
class ExampleConverterConfig(FileBasedDataConverterConfig):
    """Configuration for the example (skeleton) converter.

    Extends FileBasedDataConverterConfig with store type, component group
    profile, and whether to write sequence meta JSON after conversion.

    Attributes:
        store_type: Output store type ("itar" or "directory").
        component_group_profile: Component group layout ("default",
            "separate-sensors", or "separate-all").
        store_sequence_meta: Whether to write {sequence_id}.json after conversion.
    """

    store_type: Literal["itar", "directory"] = "itar"
    component_group_profile: Literal["default", "separate-sensors", "separate-all"] = "separate-sensors"
    store_sequence_meta: bool = True


class ExampleConverter(FileBasedDataConverter):
    """
    Skeleton converter that writes minimal valid NCore V4 output.

    Use this as a reference when implementing your own converter.
    """

    def __init__(self, config: ExampleConverterConfig) -> None:
        """Initialize the converter from an example config.

        Args:
            config: ExampleConverterConfig with root_dir, output_dir, store_type,
                component_group_profile, store_sequence_meta, and base sensor options.
        """
        super().__init__(config)
        self.store_type: Literal["itar", "directory"] = config.store_type
        self.component_group_profile: Literal["default", "separate-sensors", "separate-all"] = (
            config.component_group_profile
        )
        self.store_sequence_meta: bool = config.store_sequence_meta
        self.logger = logging.getLogger(__name__)

    @staticmethod
    def get_sequence_ids(config: ExampleConverterConfig) -> list[str]:
        """Return sequence IDs to convert (example discovery convention).

        If root_dir does not exist or has no subdirs, returns a single
        placeholder ID. Otherwise returns the first subdirectory name.

        Args:
            config: Converter config with root_dir to scan.

        Returns:
            List of sequence identifier strings (e.g. ["example_sequence_001"]).
        """
        # --- FILL IN: Return the list of sequence IDs your dataset exposes. ---
        # Scan config.root_dir (e.g. list subdirectories, read a manifest CSV/JSON,
        # or parse your dataset index). Each ID will be passed to convert_sequence().
        root = UPath(config.root_dir)
        if not root.exists():
            return ["example_sequence_001"]
        subdirs = [p for p in root.iterdir() if p.is_dir()]
        if not subdirs:
            return ["example_sequence_001"]
        return [str(p.name) for p in sorted(subdirs)[:1]]

    @staticmethod
    def from_config(config: ExampleConverterConfig) -> ExampleConverter:
        """Build an ExampleConverter instance from the given config.

        Args:
            config: ExampleConverterConfig instance.

        Returns:
            ExampleConverter ready to run convert_sequence.
        """
        # --- FILL IN: Return an instance of your converter. ---
        # Use this to pass through config and optionally do one-time setup
        # (e.g. load calibration, open a dataset index) that all sequences will use.
        #
        # TIP: NCore provides shared utilities for real converters in
        # ncore.impl.common.transformations: se3_inverse, transform_point_cloud,
        # transform_bbox, PoseInterpolator, MotionCompensator, time_bounds.
        # See the Waymo and PAI converters for usage examples.
        return ExampleConverter(config)

    def convert_sequence(self, sequence_id: str) -> None:
        """Write valid NCore V4 output for one sequence using placeholder data.

        Demonstrates all V4 component types: Poses (dynamic rig->world,
        static camera->rig, optional static world->world_global), Intrinsics, Masks,
        Camera frames, LiDAR frames, Radar frames, and Cuboids. Uses synthetic
        data so the example runs without any real sensor input.

        Args:
            sequence_id: Identifier for this sequence (used for output path and meta).
        """
        self.logger.info("Converting sequence %s", sequence_id)

        # --- Sequence time range (required by NCore / NuRec). ---
        # FILL IN: Read the start and end timestamps for this sequence from your
        # source data. NCore uses microseconds everywhere.
        #
        # from_start_end(start, end) treats `end` as **inclusive** and internally
        # stores stop = end + 1, producing a half-closed [start, stop) interval.
        # Do NOT pass end + 1 yourself -- that double-increments and causes:
        #   "Dynamic poses must cover the full sequence time range"
        #
        # IMPORTANT: Derive the interval from the sensor that provides the rig
        # trajectory (typically your ego-pose / LiDAR source). NCore requires
        # dynamic poses to *exactly* span the interval (first timestamp ==
        # interval.start, last timestamp == interval.stop - 1). If other sensors
        # (e.g. cameras) have timestamps slightly outside this range, clamp them
        # when writing frames.
        t_start_us = 0  # FILL IN: sequence start timestamp (microseconds)
        t_end_us = int(1e6)  # FILL IN: sequence end timestamp (microseconds, inclusive)
        sequence_timestamp_interval_us = HalfClosedInterval.from_start_end(t_start_us, t_end_us)

        # --- Which sensors to export. ---
        # get_active_* respects --no-cameras, --camera-id, etc. Pass the full
        # list of sensor IDs your format has; the method returns the filtered list.
        # FILL IN: Replace these with your dataset's actual sensor IDs.
        camera_ids = self.get_active_camera_ids(
            ["pinhole_camera", "fisheye_camera", "ftheta_camera"]
        )  # FILL IN: real camera IDs
        lidar_ids = self.get_active_lidar_ids(["lidar_top"])  # FILL IN: real LiDAR IDs
        radar_ids = self.get_active_radar_ids(["radar_front"])  # FILL IN: real radar IDs (or remove if no radar)

        # --- Component group layout (required so NuRec finds components). ---
        # This defines how Poses, Cameras, Lidars, etc. are grouped in the store.
        # Use the same profile (e.g. separate-sensors) as in this example unless
        # you have a reason to change it.
        component_groups = ComponentGroupAssignments.create(
            camera_ids=camera_ids,
            lidar_ids=lidar_ids,
            radar_ids=list(radar_ids),
            profile=self.component_group_profile,
        )

        # --- Create the NCore V4 writer. ---
        # One writer per sequence. output_dir/sequence_id will hold the store
        # (itar file or directory). sequence_id and sequence_timestamp_interval_us
        # are written into the store and must match what NuRec expects.
        output_path = self.output_dir / sequence_id
        store_writer = SequenceComponentGroupsWriter(
            output_dir_path=output_path,
            store_base_name=sequence_id,
            sequence_id=sequence_id,
            sequence_timestamp_interval_us=sequence_timestamp_interval_us,
            store_type=self.store_type,
            generic_meta_data={"source": "ncore_template_example"},  # FILL IN: update source tag
        )

        # --- Register component writers and write data. ---
        # Order: Poses -> Intrinsics -> Masks -> Camera / Lidar / Radar -> Cuboids.
        self._write_poses(store_writer, component_groups, camera_ids, lidar_ids, radar_ids, t_start_us, t_end_us)
        self._write_intrinsics(store_writer, component_groups, camera_ids, lidar_ids)
        self._write_masks(store_writer, component_groups, camera_ids)
        self._write_cameras(store_writer, component_groups, camera_ids, t_start_us)
        self._write_lidars(store_writer, component_groups, lidar_ids, t_start_us)
        self._write_radars(store_writer, component_groups, radar_ids, t_start_us)
        self._write_cuboids(store_writer, component_groups, t_start_us)

        # --- Finalize the store. ---
        # Call finalize() once all components are written. No further writes
        # to this sequence are allowed. You get back the paths to the store.
        ncore_4_paths = store_writer.finalize()

        # --- Write sequence meta JSON (required for NuRec and ncore_vis). ---
        # NuRec and the NCore viewer need a {sequence_id}.json file that
        # describes the sequence and where to find the component groups. Write
        # it from the reader's get_sequence_meta() after finalize().
        if self.store_sequence_meta:
            reader = SequenceComponentGroupsReader(ncore_4_paths)
            sequence_meta_path = output_path / f"{reader.sequence_id}.json"
            with sequence_meta_path.open("w") as f:
                json.dump(reader.get_sequence_meta().to_dict(), f, indent=2)
            self.logger.info("Wrote sequence meta %s", sequence_meta_path)

    # ------------------------------------------------------------------
    # Per-component write methods
    # ------------------------------------------------------------------

    def _write_poses(
        self,
        store_writer: SequenceComponentGroupsWriter,
        component_groups: ComponentGroupAssignments,
        camera_ids: list[str],
        lidar_ids: list[str],
        radar_ids: list[str],
        t_start_us: int,
        t_end_us: int,
    ) -> None:
        """Write Poses component: rig->world dynamic + static transforms.

        FILL IN: Decode poses from your format and write them.

        NCore expects rig->world as a *dynamic* pose (multiple 4x4 matrices
        with timestamps_us). Store world->world_global as a *static* pose
        (identity if no global CRS; see inline comment below for details).

        Pose trajectory density:
            NuRec interpolates rig->world at every per-ray LiDAR timestamp during
            motion compensation.  A sparse trajectory (e.g. one pose per LiDAR frame)
            causes rowing artifacts -- horizontal stripes where adjacent sweep columns
            snap to a staircase-interpolated pose.

            Combine **every** available pose source into a single trajectory:
              1. Per-camera-image poses (primary) -- iterate ALL cameras, not just
                 one.  N cameras x M frames yields N x M waypoints.
              2. Frame-level / LiDAR-level poses -- if the dataset provides a separate
                 pose per frame at a distinct timestamp, include it.
              3. IMU / GPS / odometry -- if available at higher frequency, include too.
            Concatenate all sources, deduplicate by timestamp (np.unique), and sort.
            Waypoint spacing should be smaller than the LiDAR sweep duration
            (e.g. <50 ms for a 10 Hz LiDAR).

        Rig frame: +X forward, +Y left, +Z up.  Timestamps in microseconds.

        Constraint: timestamps_us[0] must equal the sequence interval start
        and timestamps_us[-1] must equal the inclusive end (interval.stop - 1).
        If these don't match, the writer raises an AssertionError.

        Args:
            store_writer: V4 sequence writer to register the PosesComponent on.
            component_groups: Component group assignments for the store layout.
            camera_ids: Camera sensor IDs needing camera->rig static poses.
            lidar_ids: LiDAR sensor IDs needing lidar->rig static poses.
            radar_ids: Radar sensor IDs needing radar->rig static poses.
            t_start_us: Sequence start timestamp in microseconds.
            t_end_us: Sequence end timestamp in microseconds (inclusive).
        """
        poses_writer = store_writer.register_component_writer(
            PosesComponent.Writer,
            component_instance_name="default",
            group_name=component_groups.poses_component_group,
            generic_meta_data={  # FILL IN: update calibration and egomotion source tags
                "calibration_type": "example:calib",
                "egomotion_type": "example:egomotion",
            },
        )

        # FILL IN: Collect rig->world poses from EVERY available source:
        #   1. Per-camera-image poses (primary -- iterate ALL cameras, not just one)
        #   2. Frame-level / LiDAR-level poses (if at distinct timestamps)
        #   3. IMU / GPS / odometry poses (if available at higher frequency)
        # Concatenate all poses + timestamps, deduplicate by timestamp (np.unique),
        # sort, and store.  See NCore reference Waymo converter for the canonical
        # pattern (tools/data_converter/waymo/converter.py :: decode_poses).
        #
        # IMPORTANT: All pose computation (collection, concatenation,
        # deduplication, sorting, re-referencing, extrapolation) must stay in
        # float64. Cast to float32 only as the very last step before storing.
        # Re-reference to the first ego pose so the world origin is near the
        # vehicle start:
        #   T_ref_inv = np.linalg.inv(poses_f64[0])
        #   poses_f64 = T_ref_inv @ poses_f64
        #   poses_f32 = poses_f64.astype(np.float32)
        # Without re-referencing, raw global coordinates (GPS/UTM/ENU at 10+ km)
        # lose precision in float32, causing motion-compensation artifacts.
        T_rig_world_start = np.eye(4, dtype=np.float32)
        T_rig_world_end = np.eye(4, dtype=np.float32)
        T_rig_world_end[0, 3] = 10.0  # 10 m forward along +x
        T_rig_worlds = np.stack([T_rig_world_start, T_rig_world_end])
        timestamps_us = np.array([t_start_us, t_end_us], dtype=np.uint64)
        poses_writer.store_dynamic_pose(
            source_frame_id="rig",
            target_frame_id="world",
            poses=T_rig_worlds,
            timestamps_us=timestamps_us,
        )
        # Store world->world_global static pose.  NuRec currently requires this
        # edge in the pose graph even for datasets with no global CRS.  Use
        # identity if you don't have a global coordinate system; replace with
        # the actual world->world_global transform (4x4 SE3) if your dataset
        # provides one (e.g. ECEF).
        # dtype MUST be float64 (NuRec's RigTrajectories.T_world_base is float64).
        # TODO: This pose will become optional in a future NuRec release.
        poses_writer.store_static_pose(
            source_frame_id="world",
            target_frame_id="world_global",
            pose=np.eye(4, dtype=np.float64),  # FILL IN: replace with actual transform if global CRS exists
        )

        # FILL IN: Store a static camera->rig pose for each camera.
        # NuRec needs camera->rig to project points into images. Use your
        # dataset's extrinsic calibration (4x4 SE3: camera frame -> rig frame).
        #
        # NOTE: If your dataset uses a different camera frame convention than
        # NCore (camera +Z optical, +X right, +Y down), apply a rotation to
        # the extrinsic here. E.g. Waymo cameras use +X as the principal axis
        # and require a 3x3 rotation; see the Waymo converter for details.
        #
        # dtype MUST be float32. NuRec's FrameConversion.transform_poses leaks
        # the input dtype through an internal matmul, so a float64 sensor->rig
        # pose collides with the float32 rig trajectory at "Get Lidar Point
        # Clouds" with `RuntimeError: double != float`. The only pose that
        # stays float64 is world->world_global (stored above).
        # TODO: revert to float64 once NuRec's transform_poses promotes inputs
        # consistently (future NuRec release).
        for camera_id in camera_ids:
            T_camera_rig = np.eye(4, dtype=np.float32)
            poses_writer.store_static_pose(
                source_frame_id=camera_id,
                target_frame_id="rig",
                pose=T_camera_rig,
            )

        # FILL IN: Store a static T_lidar_rig for each LiDAR.
        # If the LiDAR is the rig reference sensor (its poses define the rig
        # trajectory), this is identity. Otherwise use the LiDAR-to-rig
        # extrinsic calibration. The pose graph needs this edge for motion
        # compensation and LiDAR-to-camera projection. T_camera_rig and
        # T_lidar_rig must encode the correct physical sensor-to-sensor
        # relationship. Verify alignment with ncore_vis /
        # ncore_project_pc_to_img.
        #
        # dtype MUST be float32 (same reason as T_camera_rig above).
        # TODO: revert to float64 in a future NuRec release.
        for lidar_id in lidar_ids:
            T_lidar_rig = np.eye(4, dtype=np.float32)
            poses_writer.store_static_pose(
                source_frame_id=lidar_id,
                target_frame_id="rig",
                pose=T_lidar_rig,
            )

        # FILL IN: Store a static T_radar_rig for each radar.
        # Same principle as camera and lidar: use the radar-to-rig extrinsic
        # calibration from your dataset.
        #
        # dtype MUST be float32 (same reason as T_camera_rig above).
        # TODO: revert to float64 in a future NuRec release.
        for radar_id in radar_ids:
            T_radar_rig = np.eye(4, dtype=np.float32)
            poses_writer.store_static_pose(
                source_frame_id=radar_id,
                target_frame_id="rig",
                pose=T_radar_rig,
            )

    def _write_intrinsics(
        self,
        store_writer: SequenceComponentGroupsWriter,
        component_groups: ComponentGroupAssignments,
        camera_ids: list[str],
        lidar_ids: list[str],
    ) -> None:
        """Write Intrinsics component: camera + LiDAR model parameters.

        Registers one IntrinsicsComponent.Writer and delegates to
        _write_camera_intrinsics() and _write_lidar_intrinsics().

        For vehicles with windshield refraction, NCore provides
        BivariateWindshieldModelParameters as an external distortion model.
        This is specialized and omitted here.

        Args:
            store_writer: V4 sequence writer to register the IntrinsicsComponent on.
            component_groups: Component group assignments for the store layout.
            camera_ids: Camera sensor IDs to store camera intrinsics for.
            lidar_ids: LiDAR sensor IDs to store LiDAR intrinsics for.
        """
        intrinsics_writer = store_writer.register_component_writer(
            IntrinsicsComponent.Writer,
            component_instance_name="default",
            group_name=component_groups.intrinsics_component_group,
        )
        self._write_camera_intrinsics(intrinsics_writer, camera_ids)
        self._write_lidar_intrinsics(intrinsics_writer, lidar_ids)

    def _write_camera_intrinsics(
        self,
        intrinsics_writer: IntrinsicsComponent.Writer,
        camera_ids: list[str],
    ) -> None:
        """Store per-camera intrinsics (model parameters).

        FILL IN: Use the camera model that matches your dataset's cameras.

        NCore supports three camera models (all from ncore.data):
        OpenCVPinholeCameraModelParameters, OpenCVFisheyeCameraModelParameters,
        and FThetaCameraModelParameters. One example of each is shown below.

        Each camera model includes a ShutterType. Most datasets use GLOBAL
        (instantaneous capture) or ROLLING_TOP_TO_BOTTOM. All variants:
        GLOBAL, ROLLING_TOP_TO_BOTTOM, ROLLING_LEFT_TO_RIGHT,
        ROLLING_BOTTOM_TO_TOP, ROLLING_RIGHT_TO_LEFT.

        Args:
            intrinsics_writer: Already-registered IntrinsicsComponent.Writer.
            camera_ids: Camera sensor IDs to store intrinsics for.
        """
        camera_intrinsics = {  # FILL IN: real camera models and parameters per sensor
            "pinhole_camera": OpenCVPinholeCameraModelParameters(
                resolution=np.array([1920, 1080], dtype=np.uint64),
                shutter_type=ShutterType.GLOBAL,
                principal_point=np.array([960.0, 540.0], dtype=np.float32),
                focal_length=np.array([1000.0, 1000.0], dtype=np.float32),
                radial_coeffs=np.zeros(6, dtype=np.float32),
                tangential_coeffs=np.zeros(2, dtype=np.float32),
                thin_prism_coeffs=np.zeros(4, dtype=np.float32),
            ),
            "fisheye_camera": OpenCVFisheyeCameraModelParameters(
                resolution=np.array([1920, 1080], dtype=np.uint64),
                shutter_type=ShutterType.GLOBAL,
                principal_point=np.array([960.0, 540.0], dtype=np.float32),
                focal_length=np.array([500.0, 500.0], dtype=np.float32),
                radial_coeffs=np.zeros(4, dtype=np.float32),
                max_angle=np.pi / 2.0,
            ),
            "ftheta_camera": FThetaCameraModelParameters(
                resolution=np.array([1920, 1080], dtype=np.uint64),
                shutter_type=ShutterType.GLOBAL,
                principal_point=np.array([960.0, 540.0], dtype=np.float32),
                reference_poly=FThetaCameraModelParameters.PolynomialType.ANGLE_TO_PIXELDIST,
                pixeldist_to_angle_poly=np.array([0.0, 1.0 / 500.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
                angle_to_pixeldist_poly=np.array([0.0, 500.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
                max_angle=np.pi / 2.0,
            ),
        }
        for camera_id in camera_ids:
            intrinsics_writer.store_camera_intrinsics(
                camera_id=camera_id,
                camera_model_parameters=camera_intrinsics[camera_id],
            )

    def _write_lidar_intrinsics(
        self,
        intrinsics_writer: IntrinsicsComponent.Writer,
        lidar_ids: list[str],
    ) -> None:
        """Store per-LiDAR intrinsics for structured spinning sensors.

        NuRec and current NCore tooling expect a spinning LiDAR model; use
        store_lidar_intrinsics() with parameters from your sensor datasheet or
        calibration (see NCore data format docs). FILL IN: replace placeholder
        dimensions and angles with real values.

        Args:
            intrinsics_writer: Already-registered IntrinsicsComponent.Writer.
            lidar_ids: LiDAR sensor IDs to store intrinsics for.
        """
        for lidar_id in lidar_ids:
            intrinsics_writer.store_lidar_intrinsics(
                lidar_id=lidar_id,
                lidar_model_parameters=_SPINNING_LIDAR_PARAMS,
            )

    def _write_masks(
        self,
        store_writer: SequenceComponentGroupsWriter,
        component_groups: ComponentGroupAssignments,
        camera_ids: list[str],
    ) -> None:
        """Write Masks component (per-camera static masks).

        NuRec expects this component to exist. The NuRec data quality guide
        **requires** ego-vehicle masks per camera for best reconstruction
        quality -- without them, ego-vehicle pixels (hood, bumper, roof rack)
        leak into the reconstruction. Pass an empty dict only as a fallback
        when the dataset provides no masks and they cannot be generated.

        Real masks are PIL images keyed by name, e.g.::

            from PIL import Image
            sky_mask = Image.open("sky_mask.png")
            masks_writer.store_camera_masks(camera_id, {"sky": sky_mask, "ego": ego_mask})

        Args:
            store_writer: V4 sequence writer to register the MasksComponent on.
            component_groups: Component group assignments for the store layout.
            camera_ids: Camera sensor IDs to store masks for.
        """
        masks_writer = store_writer.register_component_writer(
            MasksComponent.Writer,
            component_instance_name="default",
            group_name=component_groups.masks_component_group,
        )
        for camera_id in camera_ids:
            # FILL IN: Replace {} with {"mask_name": pil_image, ...} if your
            # dataset provides static camera masks (e.g. sky, ego-vehicle).
            masks_writer.store_camera_masks(camera_id=camera_id, mask_images={})

    def _write_cameras(
        self,
        store_writer: SequenceComponentGroupsWriter,
        component_groups: ComponentGroupAssignments,
        camera_ids: list[str],
        t_start_us: int,
    ) -> None:
        """Write Camera frames (images).

        Register one CameraSensorComponent.Writer per camera, then call
        store_frame() for every frame. image_binary_data is the raw encoded
        bytes (JPEG, PNG, etc.); NCore stores them as-is without re-encoding.

        Frame timestamps:
            frame_timestamps_us is [exposure_start, exposure_end] in microseconds
            **per camera per frame** -- not a single global frame timestamp shared
            across cameras. Each camera has its own trigger and readout timing:

            - Rolling-shutter: start = trigger + half shutter, end = readout done -
              half shutter (exposure-centered bounds).
            - Global-shutter: start == end is valid (instantaneous capture).

            Using a global frame timestamp (e.g. ``frame.timestamp_micros``) for all
            cameras is wrong -- NuRec handles rolling-shutter compensation at incorrect
            times, causing motion blur or ghosting.

        Constraints: (1) each frame's end timestamp (frame_timestamps_us[1])
        must be unique within the component writer -- no two frames can share
        the same end timestamp. (2) Both timestamps must fall within the
        sequence interval [start, stop).

        Args:
            store_writer: V4 sequence writer to register CameraSensorComponents on.
            component_groups: Component group assignments for the store layout.
            camera_ids: Camera sensor IDs to write frames for.
            t_start_us: Timestamp (microseconds) used for placeholder frame data.
        """
        t_second_us = t_start_us + 500_000  # 0.5 s into the sequence
        for camera_id in camera_ids:
            camera_writer = store_writer.register_component_writer(
                CameraSensorComponent.Writer,
                component_instance_name=camera_id,
                group_name=component_groups.camera_component_groups[camera_id],
            )
            # FILL IN: Loop over your actual frames and read real image bytes.
            camera_writer.store_frame(
                image_binary_data=_PLACEHOLDER_JPEG,
                image_format="jpeg",
                frame_timestamps_us=np.array([t_start_us, t_start_us], dtype=np.uint64),
                generic_data={},
                generic_meta_data={},
            )
            camera_writer.store_frame(
                image_binary_data=_PLACEHOLDER_JPEG,
                image_format="jpeg",
                frame_timestamps_us=np.array([t_second_us, t_second_us], dtype=np.uint64),
                generic_data={},
                generic_meta_data={},
            )

    def _write_lidars(
        self,
        store_writer: SequenceComponentGroupsWriter,
        component_groups: ComponentGroupAssignments,
        lidar_ids: list[str],
        t_start_us: int,
    ) -> None:
        """Write LiDAR frames (point clouds).

        Register one LidarSensorComponent.Writer per LiDAR sensor, then call
        store_frame() for every sweep. Key arrays::

            direction    -- [N, 3] float32 unit-norm rays in sensor frame
            timestamp_us -- [N] uint64 per-ray timestamps (see below)
            distance_m   -- [R, N] float32 metric distances (R = number of returns)
            intensity    -- [R, N] float32 normalized to [0, 1]
            model_element -- [N, 2] uint16 (row, col) or None

        Same timestamp constraints as camera frames: end timestamp must be
        unique per frame, within the sequence interval, and per-point
        timestamps must be within [frame_start, frame_end].

        **Per-ray timestamps**: NuRec uses per-ray ``timestamp_us`` for motion
        compensation — always supply real per-ray capture times (e.g.
        column-based for spinning LiDARs).

        **Sweep start convention**: ``frame_timestamps_us[0]`` must be the
        sweep **start** (the physical time the first ray fires), NOT the sweep
        center.  Per-ray timestamps span ``[frame_ts, frame_ts + sweep_duration]``.
        Treating the source frame timestamp as the sweep center and subtracting
        half the sweep duration shifts every per-ray timestamp by ~50 ms
        (for a 10 Hz LiDAR), causing severe motion-compensation blur.  The
        ``check_lidar_camera_sweep_alignment`` validation check detects this
        by comparing LiDAR sweep midpoints against camera exposure midpoints.

        Args:
            store_writer: V4 sequence writer to register LidarSensorComponents on.
            component_groups: Component group assignments for the store layout.
            lidar_ids: LiDAR sensor IDs to write frames for.
            t_start_us: Timestamp (microseconds) used for placeholder frame data.
        """
        for lidar_id in lidar_ids:
            lidar_writer = store_writer.register_component_writer(
                LidarSensorComponent.Writer,
                component_instance_name=lidar_id,
                group_name=component_groups.lidar_component_groups[lidar_id],
            )
            self._write_spinning_lidar_frame(lidar_writer, t_start_us)

    def _write_spinning_lidar_frame(
        self,
        lidar_writer: LidarSensorComponent.Writer,
        t_start_us: int,
    ) -> None:
        """Write a placeholder frame for a spinning LiDAR.

        Picks model elements from opposite sides of the grid so the rig-frame
        AABB spans the origin (cameras use identity sensor->rig).

        Per-ray timestamps are distributed across the sweep based on column
        index to demonstrate realistic timing for motion compensation.

        LiDAR point frame requirement:
            NCore V4 expects direction vectors in **sensor coordinates** (per-ray
            normalized ray directions at each point's measurement time).  Raw LiDAR
            data falls into three categories:

            1. **Polar range image** (row/column grid with beam geometry metadata).
               Derive directions from the intrinsic beam model::

                   az = column_azimuths_rad[col_idx]
                   elev = row_elevations_rad[row_idx]
                   cos_e = np.cos(elev)
                   direction = np.stack([cos_e*np.cos(az), cos_e*np.sin(az), np.sin(elev)], -1)
                   distance = range_image[row_idx, col_idx]

               Prefer this direct approach over converting range images to
               world-frame cartesian (via per-pixel poses or the dataset's SDK)
               and decompensating back, as it avoids an unnecessary round-trip.

            2. **Sensor-frame XYZ** (cartesian points already in the sensor frame
               at each ray's measurement time).  Normalize directly:
               ``direction = xyz / np.linalg.norm(xyz, axis=1, keepdims=True)``.

            3. **World-frame or ego-compensated XYZ** (cartesian points in a global
               or vehicle frame, no range image available).  **Decompensate** to
               per-ray sensor-frame using
               ``MotionCompensator.motion_decompensate_points``, then normalize.

        Args:
            lidar_writer: Already-registered LidarSensorComponent.Writer.
            t_start_us: Timestamp (microseconds) for the placeholder frame.
        """
        # FILL IN: Loop over your actual sweeps and read real point clouds.
        # Direction vectors MUST be unit-norm (L2 norm ~= 1.0) in sensor coordinates.
        # If your data provides Cartesian XYZ in sensor frame, normalize:
        #   direction = xyz / norm(xyz).
        # If your data provides world-frame or ego-compensated XYZ, you MUST
        # decompensate first -- see the docstring above.
        # For spinning LiDARs, supply per-ray model_element (row, col) indices
        # consistent with _write_lidar_intrinsics; see NCore V4 LiDAR format.
        params = _SPINNING_LIDAR_PARAMS
        n_half = 5
        fwd_cols = np.arange(n_half, dtype=np.uint16)
        bwd_cols = np.array([params.n_columns // 2 + i for i in range(n_half)], dtype=np.uint16)
        # Use top row (positive elevation) for forward rays and bottom row
        # (negative elevation) for backward rays so the rig-frame AABB spans
        # the origin in all three axes.
        rows = np.concatenate([np.zeros(n_half, dtype=np.uint16), np.full(n_half, params.n_rows - 1, dtype=np.uint16)])
        cols = np.concatenate([fwd_cols, bwd_cols])
        model_element = np.column_stack([rows, cols])
        direction = _evaluate_model_directions(params, model_element)

        n_points = direction.shape[0]
        distances = np.ones(n_points, dtype=np.float32) * 10.0
        distances[:n_half] = 5.0  # forward rays at 5 m land inside the example cuboid

        # Per-ray timestamps: column index maps linearly to time within the sweep.
        sweep_duration_us = 100_000  # 100 ms typical for a 10 Hz spinning LiDAR
        frame_end_us = t_start_us + sweep_duration_us
        col_frac = cols.astype(np.float64) / max(params.n_columns - 1, 1)
        timestamp_us = (t_start_us + col_frac * sweep_duration_us).astype(np.uint64)

        lidar_writer.store_frame(
            direction=direction,
            timestamp_us=timestamp_us,
            model_element=model_element,
            distance_m=distances.reshape(1, -1),
            intensity=np.ones((1, n_points), dtype=np.float32) * 0.5,
            frame_timestamps_us=np.array([t_start_us, frame_end_us], dtype=np.uint64),
            generic_data={},
            generic_meta_data={},
        )

    def _write_radars(
        self,
        store_writer: SequenceComponentGroupsWriter,
        component_groups: ComponentGroupAssignments,
        radar_ids: list[str],
        t_start_us: int,
    ) -> None:
        """Write Radar frames (not required for NuRec).

        RadarSensorComponent is structurally similar to LiDAR but without
        intensity or model_element. Key arrays::

            direction    -- [N, 3] float32 unit-norm rays in sensor frame
            timestamp_us -- [N] uint64 per-ray timestamps
            distance_m   -- [R, N] float32 metric distances (R = number of returns)

        Same timestamp constraints as LiDAR / camera frames apply.
        Radar data is optional and not used by NuRec; include it only if your
        dataset provides radar measurements and you want them in the shard.

        Args:
            store_writer: V4 sequence writer to register RadarSensorComponents on.
            component_groups: Component group assignments for the store layout.
            radar_ids: Radar sensor IDs to write frames for.
            t_start_us: Timestamp (microseconds) used for placeholder frame data.
        """
        for radar_id in radar_ids:
            radar_writer = store_writer.register_component_writer(
                RadarSensorComponent.Writer,
                component_instance_name=radar_id,
                group_name=component_groups.radar_component_groups[radar_id],
            )
            # FILL IN: Loop over your actual radar sweeps.
            n_points = 5
            direction = np.zeros((n_points, 3), dtype=np.float32)
            direction[:, 0] = 1.0  # all rays point forward (+X)
            radar_writer.store_frame(
                direction=direction,
                timestamp_us=np.full(n_points, t_start_us, dtype=np.uint64),
                distance_m=np.ones((1, n_points), dtype=np.float32) * 20.0,
                frame_timestamps_us=np.array([t_start_us, t_start_us], dtype=np.uint64),
                generic_data={},
                generic_meta_data={},
            )

    def _write_cuboids(
        self,
        store_writer: SequenceComponentGroupsWriter,
        component_groups: ComponentGroupAssignments,
        t_start_us: int,
    ) -> None:
        """Write Cuboids component (3-D bounding boxes).

        Register one CuboidsComponent.Writer, collect all CuboidTrackObservation
        objects across frames, then call store_observations() once. Each
        observation has: track_id, class_id, timestamp, a reference frame (e.g.
        a LiDAR sensor) with its timestamp, and a BBox3 (centroid + dimensions +
        XYZ Euler rotation in radians). reference_frame_id must match a known
        frame name (e.g. a LiDAR instance name registered in Poses).

        Timestamp alignment:
            Both ``timestamp_us`` and ``reference_frame_timestamp_us`` must
            fall within the sequence time interval (``store_observations``
            asserts this). NCore's ``PoseGraphInterpolator`` interpolates
            dynamic poses to arbitrary timestamps via SLERP/linear, so
            exact matches to stored pose waypoints are not required.
            However, timestamps far outside the convex hull of the pose
            trajectory will be extrapolated, which can produce wrong
            transforms. Best practice: use the annotation-native timestamp
            directly -- it will work as long as it lies within the sequence
            interval.

            **Caveat for rig/world-frame cuboids**: when cuboid labels are
            defined relative to a specific ego pose (e.g. a frame-level
            vehicle pose at a LiDAR frame timestamp), the pose trajectory
            should contain a waypoint at that exact timestamp.  An
            interpolated pose at a nearby time is a *different* transform,
            causing a spatial offset proportional to vehicle speed x time
            gap.  This makes dynamic objects blurry in NuRec renders.
            Either include the label-time pose in the trajectory, or
            transform cuboids into the sensor frame (where only a static
            pose is needed).

        Centroid convention:
            BBox3.centroid must be the **geometric center** of the box -- the
            midpoint of all three axes. Some datasets use bottom-center origin;
            for these, add half the box height to the Z component:
            ``centroid_z += dim_z / 2``. NuRec requires "object poses at the
            center of the bounding box aligned with the principal directions,
            without any offsets."

        Args:
            store_writer: V4 sequence writer to register the CuboidsComponent on.
            component_groups: Component group assignments for the store layout.
            t_start_us: Timestamp (microseconds) used for placeholder cuboid data.
        """
        cuboids_writer = store_writer.register_component_writer(
            CuboidsComponent.Writer,
            component_instance_name="default",
            group_name=component_groups.cuboid_track_observations_component_group,
        )
        # FILL IN: Build observations from your annotation data.
        # The three observations below demonstrate all reference_frame_id
        # variants: rig frame, sensor frame, and world frame.  Because the
        # example uses identity for all sensor→rig poses and rig→world at
        # t=0 is also identity, all three resolve to the same rig-frame
        # position.
        cuboids_writer.store_observations(
            [
                CuboidTrackObservation(
                    track_id="example-track-001",
                    class_id="car",
                    timestamp_us=t_start_us,
                    reference_frame_id="rig",
                    reference_frame_timestamp_us=t_start_us,
                    bbox3=BBox3(
                        centroid=(5.0, 0.0, 0.0),
                        dim=(4.5, 2.0, 3.0),
                        rot=(0.0, 0.0, 0.0),
                    ),
                    source=LabelSource.EXTERNAL,
                ),
                CuboidTrackObservation(
                    track_id="example-track-002",
                    class_id="car",
                    timestamp_us=t_start_us,
                    reference_frame_id="lidar_top",
                    reference_frame_timestamp_us=t_start_us,
                    bbox3=BBox3(
                        centroid=(5.0, 0.0, 0.0),
                        dim=(4.5, 2.0, 3.0),
                        rot=(0.0, 0.0, 0.0),
                    ),
                    source=LabelSource.EXTERNAL,
                ),
                CuboidTrackObservation(
                    track_id="example-track-003",
                    class_id="car",
                    timestamp_us=t_start_us,
                    reference_frame_id="world",
                    reference_frame_timestamp_us=t_start_us,
                    bbox3=BBox3(
                        centroid=(5.0, 0.0, 0.0),
                        dim=(4.5, 2.0, 3.0),
                        rot=(0.0, 0.0, 0.0),
                    ),
                    source=LabelSource.EXTERNAL,
                ),
            ]
        )
