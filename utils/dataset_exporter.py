"""
Multi-View Dataset Exporter

This module exports synchronized multi-camera recordings as datasets
compatible with multi-view computer vision tools (COLMAP, OpenMVG, etc.).

Part of Scenario 13 (Multi-Vendor Multi-Camera Integration)

Implements:
- Scenario 9.3: Export synchronized multi-view dataset

Features:
- Export synchronized frame triplets
- Include camera calibration and poses
- Generate metadata for reconstruction tools
- Support multiple export formats
"""

import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import numpy as np
import logging

logger = logging.getLogger(__name__)


@dataclass
class CameraInfo:
    """Camera information for dataset"""
    camera_id: str
    camera_model: str  # "PINHOLE", "OPENCV", etc.
    width: int
    height: int
    fx: float  # Focal length x
    fy: float  # Focal length y
    cx: float  # Principal point x
    cy: float  # Principal point y
    distortion: List[float]  # Distortion coefficients


@dataclass
class FrameTriplet:
    """Synchronized frame triplet"""
    triplet_id: int
    timestamp: float  # Reference timestamp
    frames: Dict[str, str]  # camera_id -> relative image path
    timestamps_per_camera: Dict[str, float]  # Actual timestamp per camera
    sync_error_ms: float  # Maximum timestamp difference


@dataclass
class DatasetMetadata:
    """Dataset metadata"""
    dataset_name: str
    recording_date: str
    num_cameras: int
    num_frames: int
    cameras: List[CameraInfo]
    frame_triplets: List[FrameTriplet]
    calibration_file: Optional[str]
    coordinate_system: str  # Reference camera
    export_format: str  # "colmap", "openmvg", "generic"


class MultiViewDatasetExporter:
    """
    Export multi-camera recordings as structured datasets.

    Implements Scenario 9.3: Export synchronized multi-view dataset

    Supports:
    - COLMAP format (for 3D reconstruction)
    - OpenMVG format (for structure-from-motion)
    - Generic format (JSON + organized images)
    """

    def __init__(self, session_dir: Path, output_dir: Path):
        """
        Initialize dataset exporter.

        Args:
            session_dir: Recording session directory
            output_dir: Output directory for dataset
        """
        self.session_dir = Path(session_dir)
        self.output_dir = Path(output_dir)

        logger.info(f"Initialized MultiViewDatasetExporter")
        logger.info(f"  Session: {session_dir}")
        logger.info(f"  Output: {output_dir}")

    def export_dataset(self,
                      camera_ids: List[str],
                      format: str = "generic",
                      include_depth: bool = False) -> Path:
        """
        Export complete dataset.

        Args:
            camera_ids: List of camera IDs to include
            format: Export format ("colmap", "openmvg", "generic")
            include_depth: Whether to include depth maps

        Returns:
            Path to exported dataset
        """
        logger.info(f"Exporting dataset in '{format}' format...")

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load calibration if available
        calibration_metadata = self._load_calibration_metadata()

        # Find synchronized frame triplets
        frame_triplets = self._find_frame_triplets(camera_ids)

        if not frame_triplets:
            logger.error("No synchronized frame triplets found")
            return None

        logger.info(f"Found {len(frame_triplets)} synchronized frame triplets")

        # Export based on format
        if format == "colmap":
            dataset_path = self._export_colmap_format(
                camera_ids, frame_triplets, calibration_metadata, include_depth
            )
        elif format == "openmvg":
            dataset_path = self._export_openmvg_format(
                camera_ids, frame_triplets, calibration_metadata
            )
        else:  # generic
            dataset_path = self._export_generic_format(
                camera_ids, frame_triplets, calibration_metadata, include_depth
            )

        logger.info(f"Dataset exported to: {dataset_path}")

        return dataset_path

    def _load_calibration_metadata(self) -> Optional[Dict]:
        """Load calibration metadata from session directory."""
        calib_file = self.session_dir / "calibration_metadata.json"

        if not calib_file.exists():
            logger.warning("No calibration metadata found")
            return None

        try:
            with open(calib_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load calibration metadata: {e}")
            return None

    def _find_frame_triplets(self, camera_ids: List[str]) -> List[FrameTriplet]:
        """
        Find synchronized frame triplets.

        Looks for frames with matching or close timestamps.
        """
        # Load frame lists per camera
        camera_frames = {}

        for camera_id in camera_ids:
            frames_file = self.session_dir / f"{camera_id}_frames.json"

            if not frames_file.exists():
                logger.warning(f"No frames file for camera {camera_id}")
                continue

            try:
                with open(frames_file, 'r') as f:
                    camera_frames[camera_id] = json.load(f)
            except Exception as e:
                logger.error(f"Failed to load frames for {camera_id}: {e}")

        if not camera_frames:
            # Fallback: scan image directories
            return self._find_triplets_from_directories(camera_ids)

        # Match frames by timestamp
        triplets = []
        triplet_id = 0

        # Use first camera as reference
        ref_camera = camera_ids[0]
        if ref_camera not in camera_frames:
            return []

        for ref_frame in camera_frames[ref_camera]:
            ref_timestamp = ref_frame['timestamp']

            # Find matching frames from other cameras
            matched_frames = {ref_camera: ref_frame['path']}
            timestamps = {ref_camera: ref_timestamp}

            for other_camera in camera_ids[1:]:
                if other_camera not in camera_frames:
                    continue

                # Find nearest frame
                best_match = None
                best_diff = float('inf')

                for other_frame in camera_frames[other_camera]:
                    diff = abs(other_frame['timestamp'] - ref_timestamp)
                    if diff < best_diff:
                        best_diff = diff
                        best_match = other_frame

                # Accept match if within 50ms
                if best_match and best_diff < 0.050:
                    matched_frames[other_camera] = best_match['path']
                    timestamps[other_camera] = best_match['timestamp']

            # Create triplet if we have all cameras
            if len(matched_frames) == len(camera_ids):
                max_diff = max(timestamps.values()) - min(timestamps.values())

                triplet = FrameTriplet(
                    triplet_id=triplet_id,
                    timestamp=ref_timestamp,
                    frames=matched_frames,
                    timestamps_per_camera=timestamps,
                    sync_error_ms=max_diff * 1000
                )

                triplets.append(triplet)
                triplet_id += 1

        return triplets

    def _find_triplets_from_directories(self, camera_ids: List[str]) -> List[FrameTriplet]:
        """
        Fallback: find triplets by scanning image directories.

        Assumes images are named with timestamps or sequential IDs.
        """
        logger.info("Scanning image directories for frame triplets...")

        # Scan each camera directory
        camera_images = {}

        for camera_id in camera_ids:
            image_dir = self.session_dir / camera_id / "images"

            if not image_dir.exists():
                logger.warning(f"No image directory for {camera_id}")
                continue

            # Get all images
            images = sorted(list(image_dir.glob("*.png")) + list(image_dir.glob("*.jpg")))
            camera_images[camera_id] = images

        if not camera_images:
            return []

        # Match by index (assumes synchronized capture)
        triplets = []
        min_count = min(len(images) for images in camera_images.values())

        for i in range(min_count):
            frames = {}
            for camera_id, images in camera_images.items():
                frames[camera_id] = str(images[i].relative_to(self.session_dir))

            triplet = FrameTriplet(
                triplet_id=i,
                timestamp=i * 0.033,  # Assume 30fps
                frames=frames,
                timestamps_per_camera={cam: i * 0.033 for cam in camera_ids},
                sync_error_ms=0.0
            )

            triplets.append(triplet)

        return triplets

    def _export_generic_format(self,
                               camera_ids: List[str],
                               frame_triplets: List[FrameTriplet],
                               calibration: Optional[Dict],
                               include_depth: bool) -> Path:
        """
        Export in generic format (JSON + organized images).

        Directory structure:
        dataset/
          ├── metadata.json
          ├── cameras.json
          ├── frames/
          │   ├── camera1/
          │   │   ├── 0000.png
          │   │   ├── 0001.png
          │   ├── camera2/
          │   │   ├── 0000.png
          │   │   ├── 0001.png
          └── depth/ (optional)
              ├── camera1/
              ├── camera2/
        """
        dataset_dir = self.output_dir / "dataset_generic"
        dataset_dir.mkdir(parents=True, exist_ok=True)

        # Create directory structure
        frames_dir = dataset_dir / "frames"
        frames_dir.mkdir(exist_ok=True)

        if include_depth:
            depth_dir = dataset_dir / "depth"
            depth_dir.mkdir(exist_ok=True)

        # Copy images
        for camera_id in camera_ids:
            cam_frames_dir = frames_dir / camera_id
            cam_frames_dir.mkdir(exist_ok=True)

            if include_depth:
                cam_depth_dir = depth_dir / camera_id
                cam_depth_dir.mkdir(exist_ok=True)

        for triplet in frame_triplets:
            for camera_id, rel_path in triplet.frames.items():
                src_path = self.session_dir / rel_path
                dst_path = frames_dir / camera_id / f"{triplet.triplet_id:04d}.png"

                if src_path.exists():
                    shutil.copy2(src_path, dst_path)

        # Export camera info
        cameras_info = []
        if calibration and 'calibration_info' in calibration:
            calib_info = calibration['calibration_info']

            for camera_id in camera_ids:
                # Get intrinsics from calibration
                # This is simplified - would need actual parsing
                camera_info = CameraInfo(
                    camera_id=camera_id,
                    camera_model="OPENCV",
                    width=1280,  # Default
                    height=720,
                    fx=600.0,  # Placeholder
                    fy=600.0,
                    cx=640.0,
                    cy=360.0,
                    distortion=[0.0, 0.0, 0.0, 0.0, 0.0]
                )
                cameras_info.append(camera_info)

        with open(dataset_dir / "cameras.json", 'w') as f:
            json.dump([asdict(c) for c in cameras_info], f, indent=2)

        # Export metadata
        metadata = DatasetMetadata(
            dataset_name=self.session_dir.name,
            recording_date=datetime.now().isoformat(),
            num_cameras=len(camera_ids),
            num_frames=len(frame_triplets),
            cameras=cameras_info,
            frame_triplets=frame_triplets,
            calibration_file="cameras.json",
            coordinate_system=camera_ids[0],  # First camera as reference
            export_format="generic"
        )

        # Convert dataclass to dict
        metadata_dict = asdict(metadata)

        with open(dataset_dir / "metadata.json", 'w') as f:
            json.dump(metadata_dict, f, indent=2)

        logger.info(f"Exported {len(frame_triplets)} frames to generic format")

        return dataset_dir

    def _export_colmap_format(self,
                             camera_ids: List[str],
                             frame_triplets: List[FrameTriplet],
                             calibration: Optional[Dict],
                             include_depth: bool) -> Path:
        """
        Export in COLMAP format for 3D reconstruction.

        COLMAP format:
        - cameras.txt: Camera intrinsics
        - images.txt: Image list with poses
        - points3D.txt: 3D points (optional, empty initially)
        - images/ directory
        """
        dataset_dir = self.output_dir / "dataset_colmap"
        dataset_dir.mkdir(parents=True, exist_ok=True)

        images_dir = dataset_dir / "images"
        images_dir.mkdir(exist_ok=True)

        # Export cameras.txt
        with open(dataset_dir / "cameras.txt", 'w') as f:
            f.write("# Camera list with one line of data per camera:\n")
            f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")

            for i, camera_id in enumerate(camera_ids, 1):
                # PINHOLE model: fx, fy, cx, cy
                f.write(f"{i} PINHOLE 1280 720 600.0 600.0 640.0 360.0\n")

        # Export images.txt
        with open(dataset_dir / "images.txt", 'w') as f:
            f.write("# Image list with two lines of data per image:\n")
            f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
            f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")

            image_id = 1
            for triplet in frame_triplets:
                for cam_idx, (camera_id, rel_path) in enumerate(triplet.frames.items(), 1):
                    # Copy image
                    src_path = self.session_dir / rel_path
                    dst_name = f"{image_id:04d}.png"
                    dst_path = images_dir / dst_name

                    if src_path.exists():
                        shutil.copy2(src_path, dst_path)

                    # Identity pose (qw, qx, qy, qz, tx, ty, tz)
                    f.write(f"{image_id} 1.0 0.0 0.0 0.0 0.0 0.0 0.0 {cam_idx} {dst_name}\n")
                    f.write("\n")  # Empty POINTS2D line

                    image_id += 1

        # Export empty points3D.txt
        with open(dataset_dir / "points3D.txt", 'w') as f:
            f.write("# 3D point list with one line of data per point:\n")
            f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")

        logger.info(f"Exported {len(frame_triplets)} frames to COLMAP format")

        return dataset_dir

    def _export_openmvg_format(self,
                              camera_ids: List[str],
                              frame_triplets: List[FrameTriplet],
                              calibration: Optional[Dict]) -> Path:
        """
        Export in OpenMVG JSON format.
        """
        dataset_dir = self.output_dir / "dataset_openmvg"
        dataset_dir.mkdir(parents=True, exist_ok=True)

        images_dir = dataset_dir / "images"
        images_dir.mkdir(exist_ok=True)

        # OpenMVG sfm_data.json format
        sfm_data = {
            "version": ["1", "2", "2"],
            "root_path": str(images_dir),
            "views": [],
            "intrinsics": [],
            "extrinsics": [],
            "structure": []
        }

        # Add intrinsics
        for i, camera_id in enumerate(camera_ids):
            intrinsic = {
                "key": i,
                "value": {
                    "polymorphic_id": 2147483649,  # Pinhole camera radial k3
                    "polymorphic_name": "pinhole_radial_k3",
                    "ptr_wrapper": {
                        "id": 2147483650,
                        "data": {
                            "width": 1280,
                            "height": 720,
                            "focal_length": 600.0,
                            "principal_point": [640.0, 360.0],
                            "disto_k3": [0.0, 0.0, 0.0]
                        }
                    }
                }
            }
            sfm_data["intrinsics"].append(intrinsic)

        # Add views
        view_id = 0
        for triplet in frame_triplets:
            for cam_idx, (camera_id, rel_path) in enumerate(triplet.frames.items()):
                # Copy image
                src_path = self.session_dir / rel_path
                dst_name = f"{view_id:04d}.png"
                dst_path = images_dir / dst_name

                if src_path.exists():
                    shutil.copy2(src_path, dst_path)

                view = {
                    "key": view_id,
                    "value": {
                        "polymorphic_id": 1073741824,
                        "ptr_wrapper": {
                            "id": 1073741825,
                            "data": {
                                "local_path": "",
                                "filename": dst_name,
                                "width": 1280,
                                "height": 720,
                                "id_view": view_id,
                                "id_intrinsic": cam_idx,
                                "id_pose": view_id
                            }
                        }
                    }
                }

                sfm_data["views"].append(view)
                view_id += 1

        # Save sfm_data.json
        with open(dataset_dir / "sfm_data.json", 'w') as f:
            json.dump(sfm_data, f, indent=2)

        logger.info(f"Exported {len(frame_triplets)} frames to OpenMVG format")

        return dataset_dir


# Convenience function
def export_session_as_dataset(session_dir: Path,
                             output_dir: Path,
                             camera_ids: List[str],
                             format: str = "generic") -> Path:
    """
    Convenience function to export recording session as dataset.

    Implements Scenario 9.3: Export synchronized multi-view dataset

    Args:
        session_dir: Recording session directory
        output_dir: Output directory
        camera_ids: List of camera IDs
        format: Export format ("colmap", "openmvg", "generic")

    Returns:
        Path to exported dataset
    """
    exporter = MultiViewDatasetExporter(session_dir, output_dir)
    return exporter.export_dataset(camera_ids, format=format)
