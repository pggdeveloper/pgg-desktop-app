"""
Point Cloud Transformation for Multi-Camera Systems

This module provides transformation of point clouds from camera-local
coordinates to global coordinates using calibrated transformation matrices.

Features:
- Transform point clouds using 4x4 matrices
- Preserve colors and intensities
- Batch transformation for efficiency
- Validation with known landmarks

Part of Scenario 13 (Multi-Vendor Multi-Camera Integration)

Implements:
- Scenario 6.1: Transform ZED 2i #0 point cloud
- Scenario 6.2: Transform ZED 2i #1 point cloud
- Scenario 6.3: Transform RealSense point cloud (identity)
"""

import numpy as np
from typing import Optional, Tuple, Dict
import logging

try:
    import open3d as o3d
    _HAS_OPEN3D = True
except ImportError:
    _HAS_OPEN3D = False
    logging.warning("Open3D not available, some functionality will be limited")

from .transformation_matrix import TransformationMatrix

logger = logging.getLogger(__name__)


class PointCloudTransformer:
    """
    Transform point clouds between coordinate systems.

    Supports:
    - NumPy arrays (Nx3 or Nx6 with colors)
    - Open3D PointCloud objects
    - Batch transformation
    - Color preservation

    Usage:
        transformer = PointCloudTransformer()

        # Load transformation from calibration
        T_zed0_to_global = load_transformation("zed_2i_0", "realsense_d455i_0")

        # Transform point cloud
        points_global = transformer.transform_points(points_local, T_zed0_to_global)
    """

    def __init__(self):
        """Initialize point cloud transformer"""
        logger.info("Initialized PointCloudTransformer")

    def transform_points(self,
                        points: np.ndarray,
                        transformation: np.ndarray,
                        preserve_colors: bool = True) -> np.ndarray:
        """
        Transform 3D points using 4x4 transformation matrix.

        Args:
            points: Nx3 array (xyz) or Nx6 array (xyzrgb)
            transformation: 4x4 homogeneous transformation matrix
            preserve_colors: If True, preserve RGB colors (if present)

        Returns:
            Transformed points (same shape as input)
        """
        if points.shape[1] not in [3, 6]:
            raise ValueError(f"Points must be Nx3 or Nx6, got shape {points.shape}")

        # Extract xyz
        xyz = points[:, :3]

        # Transform xyz using matrix
        xyz_transformed = TransformationMatrix.transform_points(xyz, transformation)

        # Preserve colors if present
        if preserve_colors and points.shape[1] == 6:
            colors = points[:, 3:6]
            points_transformed = np.hstack([xyz_transformed, colors])
        else:
            points_transformed = xyz_transformed

        return points_transformed

    def transform_point_cloud_o3d(self,
                                  point_cloud: 'o3d.geometry.PointCloud',
                                  transformation: np.ndarray) -> 'o3d.geometry.PointCloud':
        """
        Transform Open3D point cloud.

        Args:
            point_cloud: Open3D PointCloud object
            transformation: 4x4 transformation matrix

        Returns:
            Transformed Open3D PointCloud
        """
        if not _HAS_OPEN3D:
            raise ImportError("Open3D not available")

        # Create copy to avoid modifying original
        pc_transformed = o3d.geometry.PointCloud(point_cloud)

        # Transform (Open3D has built-in method)
        pc_transformed.transform(transformation)

        return pc_transformed

    def numpy_to_o3d(self,
                    points: np.ndarray,
                    has_colors: bool = False) -> 'o3d.geometry.PointCloud':
        """
        Convert NumPy array to Open3D PointCloud.

        Args:
            points: Nx3 (xyz) or Nx6 (xyzrgb) array
            has_colors: True if points contain RGB (columns 3-5)

        Returns:
            Open3D PointCloud object
        """
        if not _HAS_OPEN3D:
            raise ImportError("Open3D not available")

        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(points[:, :3])

        if has_colors and points.shape[1] >= 6:
            colors = points[:, 3:6]
            # Normalize colors to [0, 1] if they're in [0, 255]
            if colors.max() > 1.0:
                colors = colors / 255.0
            pc.colors = o3d.utility.Vector3dVector(colors)

        return pc

    def o3d_to_numpy(self,
                    point_cloud: 'o3d.geometry.PointCloud',
                    include_colors: bool = True) -> np.ndarray:
        """
        Convert Open3D PointCloud to NumPy array.

        Args:
            point_cloud: Open3D PointCloud object
            include_colors: Include colors in output

        Returns:
            Nx3 or Nx6 NumPy array
        """
        if not _HAS_OPEN3D:
            raise ImportError("Open3D not available")

        points = np.asarray(point_cloud.points)

        if include_colors and point_cloud.has_colors():
            colors = np.asarray(point_cloud.colors)
            # Scale colors to [0, 255]
            if colors.max() <= 1.0:
                colors = colors * 255.0
            points = np.hstack([points, colors])

        return points

    def validate_transformation(self,
                               points_before: np.ndarray,
                               points_after: np.ndarray,
                               transformation: np.ndarray,
                               tolerance: float = 1e-5) -> Tuple[bool, float]:
        """
        Validate that transformation was applied correctly.

        Args:
            points_before: Original points
            points_after: Transformed points
            transformation: Transformation matrix used
            tolerance: Numerical tolerance

        Returns:
            (is_valid, max_error)
        """
        # Manually transform and compare
        expected = self.transform_points(points_before, transformation, preserve_colors=False)

        # Compare xyz only
        actual_xyz = points_after[:, :3] if points_after.shape[1] > 3 else points_after
        expected_xyz = expected[:, :3] if expected.shape[1] > 3 else expected

        errors = np.linalg.norm(actual_xyz - expected_xyz, axis=1)
        max_error = np.max(errors)

        is_valid = max_error < tolerance

        if not is_valid:
            logger.warning(f"Transformation validation failed: max error = {max_error:.6f}")

        return is_valid, float(max_error)

    def transform_with_validation(self,
                                 points: np.ndarray,
                                 transformation: np.ndarray,
                                 landmark_indices: Optional[np.ndarray] = None,
                                 expected_landmark_positions: Optional[np.ndarray] = None,
                                 tolerance: float = 0.02) -> Tuple[np.ndarray, bool, float]:
        """
        Transform point cloud with validation using known landmarks.

        Args:
            points: Points to transform
            transformation: Transformation matrix
            landmark_indices: Indices of landmark points
            expected_landmark_positions: Expected positions of landmarks (in global coords)
            tolerance: Tolerance for landmark validation (meters)

        Returns:
            (transformed_points, is_valid, max_landmark_error)
        """
        # Transform points
        points_transformed = self.transform_points(points, transformation)

        # Validate landmarks if provided
        if landmark_indices is not None and expected_landmark_positions is not None:
            landmark_positions = points_transformed[landmark_indices, :3]
            errors = np.linalg.norm(
                landmark_positions - expected_landmark_positions,
                axis=1
            )
            max_error = np.max(errors)
            is_valid = max_error < tolerance

            if not is_valid:
                logger.warning(
                    f"Landmark validation failed: max error = {max_error*1000:.1f}mm "
                    f"(threshold = {tolerance*1000:.1f}mm)"
                )
        else:
            is_valid = True
            max_error = 0.0

        return points_transformed, is_valid, float(max_error)

    def compute_transformation_error(self,
                                    source_points: np.ndarray,
                                    target_points: np.ndarray,
                                    transformation: np.ndarray) -> Dict[str, float]:
        """
        Compute error metrics for transformation.

        Args:
            source_points: Source points (before transformation)
            target_points: Expected target points
            transformation: Transformation matrix

        Returns:
            Dict with error metrics
        """
        # Transform source
        transformed = self.transform_points(source_points, transformation, preserve_colors=False)
        transformed_xyz = transformed[:, :3] if transformed.shape[1] > 3 else transformed
        target_xyz = target_points[:, :3] if target_points.shape[1] > 3 else target_points

        # Compute errors
        errors = np.linalg.norm(transformed_xyz - target_xyz, axis=1)

        return {
            "mean_error_m": float(np.mean(errors)),
            "median_error_m": float(np.median(errors)),
            "max_error_m": float(np.max(errors)),
            "std_error_m": float(np.std(errors)),
            "mean_error_cm": float(np.mean(errors) * 100),
            "max_error_cm": float(np.max(errors) * 100)
        }


class MultiCameraPointCloudTransformer:
    """
    Manage point cloud transformations for multiple cameras.

    Loads transformation matrices from calibration and provides
    convenient interface for transforming point clouds from each camera
    to global coordinates.

    Usage:
        transformer = MultiCameraPointCloudTransformer(calibration_result)

        # Transform point clouds from all cameras
        pc_rs_global = transformer.transform_to_global(pc_rs, "realsense_d455i_0")
        pc_zed0_global = transformer.transform_to_global(pc_zed0, "zed_2i_0")
        pc_zed1_global = transformer.transform_to_global(pc_zed1, "zed_2i_1")
    """

    def __init__(self, calibration_result):
        """
        Initialize with calibration result.

        Args:
            calibration_result: CalibrationResult object with transformations
        """
        self.calibration = calibration_result
        self.reference_camera = calibration_result.extrinsic_calibration.reference_camera

        self.transformer = PointCloudTransformer()

        # Load transformation matrices
        self.transformations = {}
        for camera_id in calibration_result.intrinsic_calibrations.keys():
            if camera_id == self.reference_camera:
                # Reference camera has identity transformation
                self.transformations[camera_id] = TransformationMatrix.create_identity()
            else:
                # Get transformation to reference
                T = calibration_result.extrinsic_calibration.get_transformation(
                    camera_id,
                    self.reference_camera
                )
                self.transformations[camera_id] = T

        logger.info(f"Initialized MultiCameraPointCloudTransformer with {len(self.transformations)} cameras")
        logger.info(f"Reference camera: {self.reference_camera}")

    def transform_to_global(self,
                           points: np.ndarray,
                           camera_id: str) -> np.ndarray:
        """
        Transform point cloud from camera coordinates to global coordinates.

        Args:
            points: Point cloud in camera coordinates
            camera_id: Source camera ID

        Returns:
            Point cloud in global coordinates
        """
        if camera_id not in self.transformations:
            raise ValueError(f"Unknown camera_id: {camera_id}")

        T = self.transformations[camera_id]

        # Reference camera needs no transformation
        if camera_id == self.reference_camera:
            logger.debug(f"Camera {camera_id} is reference, no transformation applied")
            return points.copy()

        # Transform to global
        points_global = self.transformer.transform_points(points, T)

        logger.debug(f"Transformed {len(points)} points from {camera_id} to global coordinates")

        return points_global

    def get_transformation(self, camera_id: str) -> np.ndarray:
        """
        Get transformation matrix for camera.

        Args:
            camera_id: Camera ID

        Returns:
            4x4 transformation matrix to global coordinates
        """
        return self.transformations.get(camera_id)

    def get_baseline_distance(self, camera_id_1: str, camera_id_2: str) -> float:
        """
        Get baseline distance between two cameras.

        Args:
            camera_id_1: First camera ID
            camera_id_2: Second camera ID

        Returns:
            Baseline distance in meters
        """
        T = self.calibration.extrinsic_calibration.get_transformation(
            camera_id_1, camera_id_2
        )
        return TransformationMatrix.compute_baseline_distance(T)


# Convenience functions
def transform_point_cloud(points: np.ndarray,
                         transformation: np.ndarray) -> np.ndarray:
    """
    Convenience function to transform point cloud.

    Args:
        points: Nx3 or Nx6 array
        transformation: 4x4 transformation matrix

    Returns:
        Transformed points
    """
    transformer = PointCloudTransformer()
    return transformer.transform_points(points, transformation)


def load_point_cloud_from_file(filepath: str) -> np.ndarray:
    """
    Load point cloud from file (PLY, PCD, etc.).

    Args:
        filepath: Path to point cloud file

    Returns:
        Nx3 or Nx6 NumPy array
    """
    if not _HAS_OPEN3D:
        raise ImportError("Open3D required for loading point cloud files")

    pc = o3d.io.read_point_cloud(filepath)
    transformer = PointCloudTransformer()
    return transformer.o3d_to_numpy(pc)


def save_point_cloud_to_file(points: np.ndarray,
                             filepath: str,
                             has_colors: bool = None):
    """
    Save point cloud to file.

    Args:
        points: Nx3 or Nx6 NumPy array
        filepath: Output file path
        has_colors: True if points contain RGB (auto-detect if None)
    """
    if not _HAS_OPEN3D:
        raise ImportError("Open3D required for saving point cloud files")

    if has_colors is None:
        has_colors = points.shape[1] == 6

    transformer = PointCloudTransformer()
    pc = transformer.numpy_to_o3d(points, has_colors)
    o3d.io.write_point_cloud(filepath, pc)

    logger.info(f"Saved point cloud with {len(points)} points to {filepath}")
