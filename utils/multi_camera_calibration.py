"""
Multi-Vendor Multi-Camera Calibration System

This module provides comprehensive calibration for multi-vendor camera systems:
- 2x ZED 2i cameras (Stereolabs)
- 1x RealSense D455i camera (Intel)

Capabilities:
- Intrinsic calibration per camera
- Extrinsic calibration (relative poses)
- Global coordinate system establishment
- Transformation matrix computation
- Calibration quality metrics

Part of Scenario 13 (Multi-Vendor Multi-Camera Integration)

Implements:
- Scenario 1.1: Calibrate multi-vendor camera network
- Scenario 1.2: Validate calibration pattern visibility
- Scenario 1.3: Compute intrinsic parameters
- Scenario 3.1: Define RealSense as global reference
- Scenario 3.2: Compute ZED poses relative to RealSense
- Scenario 4.1-4.4: Transformation matrices
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime
import json
import logging
from pathlib import Path

from .calibration_pattern_detector import CalibrationPatternDetector, PatternDetectionResult
from .transformation_matrix import TransformationMatrix, CameraPose

logger = logging.getLogger(__name__)


@dataclass
class IntrinsicCalibration:
    """Intrinsic calibration parameters for a single camera"""
    camera_id: str
    camera_matrix: np.ndarray  # 3x3 matrix (fx, fy, cx, cy)
    distortion_coeffs: np.ndarray  # Distortion coefficients (k1, k2, p1, p2, k3)
    image_size: Tuple[int, int]  # (width, height)
    reprojection_error: float  # RMS reprojection error in pixels
    num_calibration_images: int
    calibration_date: str

    def to_dict(self) -> Dict:
        """Convert to JSON-serializable dictionary"""
        return {
            "camera_id": self.camera_id,
            "camera_matrix": self.camera_matrix.tolist(),
            "distortion_coeffs": self.distortion_coeffs.flatten().tolist(),
            "image_size": {"width": self.image_size[0], "height": self.image_size[1]},
            "reprojection_error": float(self.reprojection_error),
            "num_calibration_images": self.num_calibration_images,
            "calibration_date": self.calibration_date,
            "focal_length": {
                "fx": float(self.camera_matrix[0, 0]),
                "fy": float(self.camera_matrix[1, 1])
            },
            "principal_point": {
                "cx": float(self.camera_matrix[0, 2]),
                "cy": float(self.camera_matrix[1, 2])
            }
        }


@dataclass
class ExtrinsicCalibration:
    """Extrinsic calibration (camera poses in global system)"""
    reference_camera: str  # e.g., "realsense_d455i_0"
    camera_poses: Dict[str, CameraPose] = field(default_factory=dict)
    transformation_matrices: Dict[str, Dict[str, np.ndarray]] = field(default_factory=dict)
    calibration_date: str = ""

    def add_camera_pose(self, pose: CameraPose):
        """Add camera pose to calibration"""
        self.camera_poses[pose.camera_id] = pose

    def get_transformation(self, from_camera: str, to_camera: str) -> Optional[np.ndarray]:
        """
        Get transformation matrix from one camera to another.

        Args:
            from_camera: Source camera ID
            to_camera: Target camera ID

        Returns:
            4x4 transformation matrix or None if not available
        """
        if from_camera == to_camera:
            return TransformationMatrix.create_identity()

        # Check if direct transformation exists
        if from_camera in self.transformation_matrices:
            if to_camera in self.transformation_matrices[from_camera]:
                return self.transformation_matrices[from_camera][to_camera]

        # Compute via reference frame
        if from_camera in self.camera_poses and to_camera in self.camera_poses:
            T_from_to_ref = self.camera_poses[from_camera].transformation_matrix
            T_to_to_ref = self.camera_poses[to_camera].transformation_matrix

            T_from_to_to = TransformationMatrix.transform_from_to(
                T_from_to_ref, T_to_to_ref
            )
            return T_from_to_to

        return None

    def to_dict(self) -> Dict:
        """Convert to JSON-serializable dictionary"""
        return {
            "reference_camera": self.reference_camera,
            "calibration_date": self.calibration_date,
            "coordinate_system": {
                "convention": "right_handed",
                "axes": {"x": "right", "y": "down", "z": "forward"},
                "units": "meters"
            },
            "camera_poses": {
                camera_id: pose.to_dict()
                for camera_id, pose in self.camera_poses.items()
            },
            "transformations": {
                from_cam: {
                    to_cam: matrix.tolist()
                    for to_cam, matrix in transforms.items()
                }
                for from_cam, transforms in self.transformation_matrices.items()
            }
        }


@dataclass
class CalibrationResult:
    """Complete calibration result for multi-camera system"""
    intrinsic_calibrations: Dict[str, IntrinsicCalibration]
    extrinsic_calibration: ExtrinsicCalibration
    pattern_config: Dict
    quality_metrics: Dict[str, float]
    calibration_date: str

    def to_dict(self) -> Dict:
        """Convert to JSON-serializable dictionary"""
        return {
            "calibration_date": self.calibration_date,
            "pattern_config": self.pattern_config,
            "intrinsic_calibrations": {
                camera_id: calib.to_dict()
                for camera_id, calib in self.intrinsic_calibrations.items()
            },
            "extrinsic_calibration": self.extrinsic_calibration.to_dict(),
            "quality_metrics": self.quality_metrics
        }

    def save(self, filepath: Path):
        """Save calibration to JSON file"""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

        logger.info(f"Calibration saved to: {filepath}")

    @classmethod
    def load(cls, filepath: Path) -> 'CalibrationResult':
        """Load calibration from JSON file"""
        with open(filepath, 'r') as f:
            data = json.load(f)

        # Reconstruct intrinsic calibrations
        intrinsic_calibrations = {}
        for camera_id, calib_data in data["intrinsic_calibrations"].items():
            intrinsic_calibrations[camera_id] = IntrinsicCalibration(
                camera_id=calib_data["camera_id"],
                camera_matrix=np.array(calib_data["camera_matrix"]),
                distortion_coeffs=np.array(calib_data["distortion_coeffs"]),
                image_size=(calib_data["image_size"]["width"],
                           calib_data["image_size"]["height"]),
                reprojection_error=calib_data["reprojection_error"],
                num_calibration_images=calib_data["num_calibration_images"],
                calibration_date=calib_data["calibration_date"]
            )

        # Reconstruct extrinsic calibration
        ext_data = data["extrinsic_calibration"]
        extrinsic_calibration = ExtrinsicCalibration(
            reference_camera=ext_data["reference_camera"],
            calibration_date=ext_data["calibration_date"]
        )

        # Reconstruct camera poses
        for camera_id, pose_data in ext_data["camera_poses"].items():
            T = np.array(pose_data["transformation_matrix"])
            pose = TransformationMatrix.create_camera_pose(camera_id, T)
            extrinsic_calibration.add_camera_pose(pose)

        # Reconstruct transformations
        for from_cam, transforms in ext_data["transformations"].items():
            extrinsic_calibration.transformation_matrices[from_cam] = {}
            for to_cam, matrix_list in transforms.items():
                extrinsic_calibration.transformation_matrices[from_cam][to_cam] = \
                    np.array(matrix_list)

        return cls(
            intrinsic_calibrations=intrinsic_calibrations,
            extrinsic_calibration=extrinsic_calibration,
            pattern_config=data["pattern_config"],
            quality_metrics=data["quality_metrics"],
            calibration_date=data["calibration_date"]
        )


class MultiCameraCalibrator:
    """
    Multi-vendor multi-camera calibration system.

    Supports:
    - Multiple camera vendors (ZED 2i, RealSense D455i)
    - Intrinsic and extrinsic calibration
    - Global coordinate system establishment
    - Transformation matrix computation
    - Quality validation

    Usage:
        calibrator = MultiCameraCalibrator(
            camera_ids=["realsense_d455i_0", "zed_2i_0", "zed_2i_1"],
            reference_camera="realsense_d455i_0"
        )

        # Capture calibration images
        for i in range(30):
            images = capture_from_all_cameras()
            calibrator.add_calibration_images(images)

        # Perform calibration
        result = calibrator.calibrate()
        result.save("calibration.json")
    """

    def __init__(self,
                 camera_ids: List[str],
                 reference_camera: str,
                 pattern_size: Tuple[int, int] = (9, 6),
                 square_size_mm: float = 30.0):
        """
        Initialize multi-camera calibrator.

        Args:
            camera_ids: List of camera IDs (e.g., ["realsense_d455i_0", "zed_2i_0", "zed_2i_1"])
            reference_camera: ID of reference camera (origin of global coordinates)
            pattern_size: Chessboard pattern size (internal corners)
            square_size_mm: Physical size of squares in mm
        """
        self.camera_ids = camera_ids
        self.reference_camera = reference_camera

        if reference_camera not in camera_ids:
            raise ValueError(f"Reference camera '{reference_camera}' not in camera_ids")

        # Pattern detector
        self.pattern_detector = CalibrationPatternDetector(
            pattern_size=pattern_size,
            square_size_mm=square_size_mm
        )

        # Storage for calibration images
        self.calibration_images: Dict[str, List[np.ndarray]] = {
            camera_id: [] for camera_id in camera_ids
        }

        # Storage for detected points
        self.image_points: Dict[str, List[np.ndarray]] = {
            camera_id: [] for camera_id in camera_ids
        }
        self.object_points: List[np.ndarray] = []

        logger.info(f"Initialized MultiCameraCalibrator with {len(camera_ids)} cameras")
        logger.info(f"Reference camera: {reference_camera}")

    def add_calibration_images(self, images: Dict[str, np.ndarray]) -> bool:
        """
        Add calibration images from all cameras.

        Images should be captured simultaneously showing the same calibration pattern.

        Args:
            images: Dict mapping camera_id -> image

        Returns:
            True if pattern detected in all cameras, False otherwise
        """
        # Detect pattern in all images
        results = self.pattern_detector.detect_multi_camera(images, require_all=True)

        if results is None:
            logger.warning("Pattern not detected in all cameras, images not added")
            return False

        # Validate all detections
        all_valid = True
        for camera_id, result in results.items():
            valid, msg = self.pattern_detector.validate_detection(result, min_corners=20)
            if not valid:
                logger.warning(f"Camera {camera_id}: {msg}")
                all_valid = False

        if not all_valid:
            logger.warning("Some detections invalid, images not added")
            return False

        # Add images and points
        for camera_id, image in images.items():
            self.calibration_images[camera_id].append(image)
            self.image_points[camera_id].append(results[camera_id].corners)

        # Object points are same for all cameras (only add once)
        if len(self.object_points) < len(self.calibration_images[self.camera_ids[0]]):
            self.object_points.append(results[self.camera_ids[0]].object_points)

        num_images = len(self.calibration_images[self.camera_ids[0]])
        logger.info(f"Added calibration image set #{num_images}")

        return True

    def calibrate_intrinsics(self) -> Dict[str, IntrinsicCalibration]:
        """
        Calibrate intrinsic parameters for each camera.

        Implements Scenario 1.3: Compute intrinsic parameters

        Returns:
            Dict mapping camera_id -> IntrinsicCalibration
        """
        intrinsic_calibrations = {}

        for camera_id in self.camera_ids:
            logger.info(f"Calibrating intrinsics for camera: {camera_id}")

            images = self.calibration_images[camera_id]
            if len(images) == 0:
                raise ValueError(f"No calibration images for camera {camera_id}")

            image_size = (images[0].shape[1], images[0].shape[0])

            # Perform calibration
            ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
                self.object_points,
                self.image_points[camera_id],
                image_size,
                None,
                None
            )

            intrinsic_calibrations[camera_id] = IntrinsicCalibration(
                camera_id=camera_id,
                camera_matrix=camera_matrix,
                distortion_coeffs=dist_coeffs,
                image_size=image_size,
                reprojection_error=ret,
                num_calibration_images=len(images),
                calibration_date=datetime.now().isoformat()
            )

            logger.info(f"Camera {camera_id}: Reprojection error = {ret:.4f} pixels")

        return intrinsic_calibrations

    def calibrate_extrinsics(self,
                            intrinsic_calibrations: Dict[str, IntrinsicCalibration]
                            ) -> ExtrinsicCalibration:
        """
        Calibrate extrinsic parameters (camera poses in global system).

        Implements:
        - Scenario 3.1: Define RealSense as global reference
        - Scenario 3.2: Compute ZED poses relative to RealSense
        - Scenario 4.1-4.4: Transformation matrices

        Args:
            intrinsic_calibrations: Intrinsic calibrations for all cameras

        Returns:
            ExtrinsicCalibration with all camera poses
        """
        logger.info("Calibrating extrinsic parameters (camera poses)")

        extrinsic = ExtrinsicCalibration(
            reference_camera=self.reference_camera,
            calibration_date=datetime.now().isoformat()
        )

        # Reference camera has identity pose (Scenario 3.1)
        ref_pose = TransformationMatrix.create_camera_pose(
            self.reference_camera,
            TransformationMatrix.create_identity()
        )
        extrinsic.add_camera_pose(ref_pose)
        logger.info(f"Reference camera {self.reference_camera} set to origin (identity pose)")

        # Compute poses for other cameras relative to reference (Scenario 3.2)
        for camera_id in self.camera_ids:
            if camera_id == self.reference_camera:
                continue

            logger.info(f"Computing pose for camera: {camera_id}")

            # Compute transformation from this camera to reference camera
            # using all calibration images
            transformations = []

            for i in range(len(self.object_points)):
                # Get pose of pattern relative to reference camera
                ret_ref, rvec_ref, tvec_ref = cv2.solvePnP(
                    self.object_points[i],
                    self.image_points[self.reference_camera][i],
                    intrinsic_calibrations[self.reference_camera].camera_matrix,
                    intrinsic_calibrations[self.reference_camera].distortion_coeffs
                )

                # Get pose of pattern relative to this camera
                ret_cam, rvec_cam, tvec_cam = cv2.solvePnP(
                    self.object_points[i],
                    self.image_points[camera_id][i],
                    intrinsic_calibrations[camera_id].camera_matrix,
                    intrinsic_calibrations[camera_id].distortion_coeffs
                )

                if ret_ref and ret_cam:
                    # Convert to transformation matrices
                    R_ref, _ = cv2.Rodrigues(rvec_ref)
                    R_cam, _ = cv2.Rodrigues(rvec_cam)

                    T_ref_to_pattern = TransformationMatrix.create_from_rotation_translation(
                        R_ref, tvec_ref
                    )
                    T_cam_to_pattern = TransformationMatrix.create_from_rotation_translation(
                        R_cam, tvec_cam
                    )

                    # Compute transformation: camera -> reference
                    # T_cam_to_ref = inv(T_ref_to_pattern) @ T_cam_to_pattern
                    T_pattern_to_ref = TransformationMatrix.invert(T_ref_to_pattern)
                    T_cam_to_ref = T_pattern_to_ref @ T_cam_to_pattern

                    transformations.append(T_cam_to_ref)

            # Average transformations from all images
            T_avg = TransformationMatrix.average_transformations(transformations)

            # Create camera pose
            pose = TransformationMatrix.create_camera_pose(camera_id, T_avg)
            extrinsic.add_camera_pose(pose)

            logger.info(f"Camera {camera_id} pose computed:")
            logger.info(f"  Position: ({pose.position[0]:.3f}, {pose.position[1]:.3f}, {pose.position[2]:.3f}) m")
            logger.info(f"  Euler angles: ({np.degrees(pose.euler_angles[0]):.1f}, "
                       f"{np.degrees(pose.euler_angles[1]):.1f}, "
                       f"{np.degrees(pose.euler_angles[2]):.1f}) deg")

        # Compute all pairwise transformations (Scenario 4.1-4.4)
        self._compute_pairwise_transformations(extrinsic)

        return extrinsic

    def _compute_pairwise_transformations(self, extrinsic: ExtrinsicCalibration):
        """Compute transformation matrices between all camera pairs"""
        logger.info("Computing pairwise transformation matrices")

        for from_camera in self.camera_ids:
            extrinsic.transformation_matrices[from_camera] = {}

            for to_camera in self.camera_ids:
                if from_camera == to_camera:
                    T = TransformationMatrix.create_identity()
                else:
                    T = extrinsic.get_transformation(from_camera, to_camera)

                extrinsic.transformation_matrices[from_camera][to_camera] = T

                # Log baseline distance for stereo pairs
                if from_camera != to_camera:
                    baseline = TransformationMatrix.compute_baseline_distance(T)
                    logger.info(f"  {from_camera} → {to_camera}: baseline = {baseline:.3f} m")

    def compute_quality_metrics(self,
                                intrinsic_calibrations: Dict[str, IntrinsicCalibration],
                                extrinsic_calibration: ExtrinsicCalibration) -> Dict[str, float]:
        """
        Compute calibration quality metrics.

        Implements Scenario 8.1: Compute reprojection error

        Returns:
            Dict of quality metrics
        """
        metrics = {}

        # Reprojection errors per camera
        for camera_id, calib in intrinsic_calibrations.items():
            metrics[f"reprojection_error_{camera_id}"] = calib.reprojection_error

        # Mean reprojection error
        mean_reproj_error = np.mean([
            calib.reprojection_error for calib in intrinsic_calibrations.values()
        ])
        metrics["mean_reprojection_error"] = mean_reproj_error

        # Max reprojection error
        max_reproj_error = np.max([
            calib.reprojection_error for calib in intrinsic_calibrations.values()
        ])
        metrics["max_reprojection_error"] = max_reproj_error

        # Number of calibration images
        metrics["num_calibration_images"] = len(self.object_points)

        # Quality assessment
        if mean_reproj_error < 0.5:
            quality = "excellent"
        elif mean_reproj_error < 1.0:
            quality = "good"
        elif mean_reproj_error < 2.0:
            quality = "acceptable"
        else:
            quality = "poor"

        metrics["overall_quality"] = quality

        logger.info(f"Calibration quality: {quality} (mean reprojection error: {mean_reproj_error:.4f} pixels)")

        return metrics

    def calibrate(self) -> CalibrationResult:
        """
        Perform complete multi-camera calibration.

        Implements:
        - Scenario 1.1: Calibrate multi-vendor camera network
        - Scenario 1.3: Compute intrinsic parameters
        - Scenario 3.1-3.2: Global coordinate system
        - Scenario 4.1-4.4: Transformation matrices

        Returns:
            CalibrationResult with complete calibration data
        """
        logger.info("Starting multi-camera calibration")

        if len(self.object_points) < 10:
            raise ValueError(f"Need at least 10 calibration images, got {len(self.object_points)}")

        # Step 1: Intrinsic calibration
        logger.info("Step 1/3: Intrinsic calibration")
        intrinsic_calibrations = self.calibrate_intrinsics()

        # Step 2: Extrinsic calibration
        logger.info("Step 2/3: Extrinsic calibration")
        extrinsic_calibration = self.calibrate_extrinsics(intrinsic_calibrations)

        # Step 3: Quality metrics
        logger.info("Step 3/3: Computing quality metrics")
        quality_metrics = self.compute_quality_metrics(
            intrinsic_calibrations,
            extrinsic_calibration
        )

        result = CalibrationResult(
            intrinsic_calibrations=intrinsic_calibrations,
            extrinsic_calibration=extrinsic_calibration,
            pattern_config=self.pattern_detector.get_pattern_info(),
            quality_metrics=quality_metrics,
            calibration_date=datetime.now().isoformat()
        )

        logger.info("Multi-camera calibration complete!")
        return result

    def get_calibration_progress(self) -> Dict:
        """Get current calibration progress"""
        return {
            "num_cameras": len(self.camera_ids),
            "num_images_captured": len(self.object_points),
            "images_per_camera": {
                camera_id: len(images)
                for camera_id, images in self.calibration_images.items()
            },
            "recommended_minimum": 20,
            "ready_to_calibrate": len(self.object_points) >= 10
        }
