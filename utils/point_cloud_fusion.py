"""
Point Cloud Fusion for Multi-Camera Systems

This module provides fusion of point clouds from multiple cameras into
a single unified 360° representation in global coordinates.

Features:
- Merge point clouds from multiple cameras
- Outlier removal (statistical, radius-based)
- Voxel grid downsampling
- Coverage analysis
- Quality metrics

Part of Scenario 13 (Multi-Vendor Multi-Camera Integration)

Implements:
- Scenario 6.4: Merge point clouds from all cameras
- Scenario 10.1: 360° body volume estimation
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import logging

try:
    import open3d as o3d
    _HAS_OPEN3D = True
except ImportError:
    _HAS_OPEN3D = False
    logging.warning("Open3D not available, some functionality will be limited")

logger = logging.getLogger(__name__)


@dataclass
class FusionResult:
    """Result of point cloud fusion"""
    fused_points: np.ndarray  # Nx3 or Nx6 (with colors)
    num_input_points: int
    num_output_points: int
    num_outliers_removed: int
    num_downsampled: int
    reduction_ratio: float
    bounding_box_min: np.ndarray
    bounding_box_max: np.ndarray
    bounding_box_size: np.ndarray
    coverage_quality: str  # "excellent", "good", "acceptable", "poor"


class PointCloudFusion:
    """
    Fuse point clouds from multiple cameras into unified 360° representation.

    Features:
    - Concatenation of multiple point clouds
    - Outlier removal for cleaning
    - Voxel grid downsampling for efficiency
    - Color preservation
    - Quality metrics

    Usage:
        fusion = PointCloudFusion(
            voxel_size=0.01,  # 1cm voxels
            remove_outliers=True
        )

        # Fuse point clouds from 3 cameras (already in global coordinates)
        result = fusion.fuse([pc_rs_global, pc_zed0_global, pc_zed1_global])

        # Save fused point cloud
        fusion.save_fused_cloud(result.fused_points, "fused_360.ply")
    """

    def __init__(self,
                 voxel_size: float = 0.01,
                 remove_outliers: bool = True,
                 outlier_removal_method: str = "statistical",
                 outlier_nb_neighbors: int = 20,
                 outlier_std_ratio: float = 2.0):
        """
        Initialize point cloud fusion.

        Args:
            voxel_size: Voxel size for downsampling (meters), e.g., 0.01 = 1cm
            remove_outliers: Whether to remove outliers
            outlier_removal_method: "statistical" or "radius"
            outlier_nb_neighbors: Number of neighbors for statistical outlier removal
            outlier_std_ratio: Standard deviation ratio for statistical outlier removal
        """
        self.voxel_size = voxel_size
        self.remove_outliers = remove_outliers
        self.outlier_removal_method = outlier_removal_method
        self.outlier_nb_neighbors = outlier_nb_neighbors
        self.outlier_std_ratio = outlier_std_ratio

        logger.info(f"Initialized PointCloudFusion (voxel_size={voxel_size}m)")

    def fuse(self,
            point_clouds: List[np.ndarray],
            camera_ids: Optional[List[str]] = None) -> FusionResult:
        """
        Fuse multiple point clouds into single unified cloud.

        Point clouds should already be in global coordinates.

        Args:
            point_clouds: List of point clouds (Nx3 or Nx6 arrays)
            camera_ids: Optional list of camera IDs for logging

        Returns:
            FusionResult with fused point cloud and metrics
        """
        if len(point_clouds) == 0:
            raise ValueError("No point clouds provided for fusion")

        logger.info(f"Fusing {len(point_clouds)} point clouds...")

        # Count input points
        num_input_points = sum(len(pc) for pc in point_clouds)
        logger.info(f"Total input points: {num_input_points}")

        # Step 1: Concatenate all point clouds
        merged = np.vstack(point_clouds)
        logger.info(f"After concatenation: {len(merged)} points")

        # Step 2: Remove outliers
        num_outliers_removed = 0
        if self.remove_outliers:
            merged, num_outliers_removed = self._remove_outliers(merged)
            logger.info(f"After outlier removal: {len(merged)} points ({num_outliers_removed} removed)")

        # Step 3: Voxel grid downsampling
        num_before_downsample = len(merged)
        if self.voxel_size > 0:
            merged = self._voxel_downsample(merged)
            num_downsampled = num_before_downsample - len(merged)
            logger.info(f"After downsampling: {len(merged)} points ({num_downsampled} removed)")
        else:
            num_downsampled = 0

        # Compute metrics
        num_output_points = len(merged)
        reduction_ratio = 1.0 - (num_output_points / num_input_points)

        # Compute bounding box
        bbox_min = np.min(merged[:, :3], axis=0)
        bbox_max = np.max(merged[:, :3], axis=0)
        bbox_size = bbox_max - bbox_min

        # Assess coverage quality
        coverage_quality = self._assess_coverage_quality(merged, bbox_size)

        result = FusionResult(
            fused_points=merged,
            num_input_points=num_input_points,
            num_output_points=num_output_points,
            num_outliers_removed=num_outliers_removed,
            num_downsampled=num_downsampled,
            reduction_ratio=reduction_ratio,
            bounding_box_min=bbox_min,
            bounding_box_max=bbox_max,
            bounding_box_size=bbox_size,
            coverage_quality=coverage_quality
        )

        logger.info(f"Fusion complete: {num_output_points} points "
                   f"({reduction_ratio*100:.1f}% reduction)")
        logger.info(f"Bounding box size: "
                   f"X={bbox_size[0]:.2f}m, Y={bbox_size[1]:.2f}m, Z={bbox_size[2]:.2f}m")
        logger.info(f"Coverage quality: {coverage_quality}")

        return result

    def _remove_outliers(self, points: np.ndarray) -> Tuple[np.ndarray, int]:
        """
        Remove outliers from point cloud.

        Args:
            points: Input points

        Returns:
            (cleaned_points, num_outliers_removed)
        """
        if not _HAS_OPEN3D:
            logger.warning("Open3D not available, skipping outlier removal")
            return points, 0

        # Convert to Open3D
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(points[:, :3])

        has_colors = points.shape[1] >= 6
        if has_colors:
            colors = points[:, 3:6]
            if colors.max() > 1.0:
                colors = colors / 255.0
            pc.colors = o3d.utility.Vector3dVector(colors)

        # Remove outliers
        if self.outlier_removal_method == "statistical":
            pc_clean, inlier_indices = pc.remove_statistical_outlier(
                nb_neighbors=self.outlier_nb_neighbors,
                std_ratio=self.outlier_std_ratio
            )
        elif self.outlier_removal_method == "radius":
            pc_clean, inlier_indices = pc.remove_radius_outlier(
                nb_points=self.outlier_nb_neighbors,
                radius=self.voxel_size * 5
            )
        else:
            logger.warning(f"Unknown outlier removal method: {self.outlier_removal_method}")
            return points, 0

        # Convert back to NumPy
        points_clean = np.asarray(pc_clean.points)

        if has_colors and pc_clean.has_colors():
            colors_clean = np.asarray(pc_clean.colors)
            if colors_clean.max() <= 1.0:
                colors_clean = colors_clean * 255.0
            points_clean = np.hstack([points_clean, colors_clean])

        num_removed = len(points) - len(points_clean)

        return points_clean, num_removed

    def _voxel_downsample(self, points: np.ndarray) -> np.ndarray:
        """
        Downsample point cloud using voxel grid.

        Args:
            points: Input points

        Returns:
            Downsampled points
        """
        if not _HAS_OPEN3D:
            logger.warning("Open3D not available, skipping downsampling")
            return points

        # Convert to Open3D
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(points[:, :3])

        has_colors = points.shape[1] >= 6
        if has_colors:
            colors = points[:, 3:6]
            if colors.max() > 1.0:
                colors = colors / 255.0
            pc.colors = o3d.utility.Vector3dVector(colors)

        # Voxel downsampling
        pc_down = pc.voxel_down_sample(voxel_size=self.voxel_size)

        # Convert back to NumPy
        points_down = np.asarray(pc_down.points)

        if has_colors and pc_down.has_colors():
            colors_down = np.asarray(pc_down.colors)
            if colors_down.max() <= 1.0:
                colors_down = colors_down * 255.0
            points_down = np.hstack([points_down, colors_down])

        return points_down

    def _assess_coverage_quality(self,
                                points: np.ndarray,
                                bbox_size: np.ndarray) -> str:
        """
        Assess 360° coverage quality.

        Args:
            points: Fused point cloud
            bbox_size: Bounding box size

        Returns:
            Quality string: "excellent", "good", "acceptable", "poor"
        """
        # Simple heuristic based on point density and bounding box
        volume = np.prod(bbox_size)

        if volume > 0:
            density = len(points) / volume  # points per cubic meter
        else:
            density = 0

        # Assess based on density (heuristic for cattle: ~100k-500k points/m³)
        if density > 300000:
            return "excellent"
        elif density > 150000:
            return "good"
        elif density > 50000:
            return "acceptable"
        else:
            return "poor"

    def save_fused_cloud(self,
                        points: np.ndarray,
                        filepath: str,
                        metadata: Optional[Dict] = None):
        """
        Save fused point cloud to file.

        Args:
            points: Fused point cloud
            filepath: Output file path (PLY format)
            metadata: Optional metadata to include
        """
        if not _HAS_OPEN3D:
            raise ImportError("Open3D required for saving point clouds")

        # Convert to Open3D
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(points[:, :3])

        if points.shape[1] >= 6:
            colors = points[:, 3:6]
            if colors.max() > 1.0:
                colors = colors / 255.0
            pc.colors = o3d.utility.Vector3dVector(colors)

        # Save
        o3d.io.write_point_cloud(filepath, pc)

        logger.info(f"Saved fused point cloud to {filepath} ({len(points)} points)")


class MultiCamera360Reconstruction:
    """
    Complete 360° reconstruction from multiple cameras.

    Combines calibration, transformation, and fusion for complete pipeline.

    Usage:
        reconstruction = MultiCamera360Reconstruction(calibration_result)

        # Add point clouds from all cameras (in local coordinates)
        reconstruction.add_point_cloud("realsense_d455i_0", pc_rs)
        reconstruction.add_point_cloud("zed_2i_0", pc_zed0)
        reconstruction.add_point_cloud("zed_2i_1", pc_zed1)

        # Fuse into 360° representation
        result = reconstruction.fuse_360()
    """

    def __init__(self,
                 calibration_result,
                 voxel_size: float = 0.01):
        """
        Initialize 360° reconstruction.

        Args:
            calibration_result: CalibrationResult object
            voxel_size: Voxel size for downsampling (meters)
        """
        from .point_cloud_transformation import MultiCameraPointCloudTransformer

        self.calibration = calibration_result
        self.transformer = MultiCameraPointCloudTransformer(calibration_result)
        self.fusion = PointCloudFusion(voxel_size=voxel_size)

        self.point_clouds_local: Dict[str, np.ndarray] = {}
        self.point_clouds_global: Dict[str, np.ndarray] = {}

        logger.info("Initialized MultiCamera360Reconstruction")

    def add_point_cloud(self, camera_id: str, points: np.ndarray):
        """
        Add point cloud from camera (in camera-local coordinates).

        Args:
            camera_id: Camera identifier
            points: Point cloud in camera-local coordinates
        """
        self.point_clouds_local[camera_id] = points

        # Transform to global
        points_global = self.transformer.transform_to_global(points, camera_id)
        self.point_clouds_global[camera_id] = points_global

        logger.info(f"Added point cloud from {camera_id}: {len(points)} points")

    def fuse_360(self) -> FusionResult:
        """
        Fuse all point clouds into 360° representation.

        Returns:
            FusionResult with fused point cloud
        """
        if len(self.point_clouds_global) == 0:
            raise ValueError("No point clouds added")

        # Fuse all global point clouds
        point_clouds = list(self.point_clouds_global.values())
        camera_ids = list(self.point_clouds_global.keys())

        result = self.fusion.fuse(point_clouds, camera_ids)

        return result

    def get_coverage_map(self) -> Dict[str, Dict]:
        """
        Get coverage information per camera.

        Returns:
            Dict mapping camera_id -> coverage_info
        """
        coverage = {}

        for camera_id, points in self.point_clouds_global.items():
            bbox_min = np.min(points[:, :3], axis=0)
            bbox_max = np.max(points[:, :3], axis=0)
            bbox_size = bbox_max - bbox_min

            coverage[camera_id] = {
                "num_points": len(points),
                "bounding_box_min": bbox_min.tolist(),
                "bounding_box_max": bbox_max.tolist(),
                "bounding_box_size": bbox_size.tolist(),
                "volume_m3": float(np.prod(bbox_size))
            }

        return coverage


# Convenience functions
def fuse_point_clouds(point_clouds: List[np.ndarray],
                     voxel_size: float = 0.01,
                     remove_outliers: bool = True) -> FusionResult:
    """
    Convenience function to fuse point clouds.

    Args:
        point_clouds: List of point clouds (in global coordinates)
        voxel_size: Voxel size for downsampling (meters)
        remove_outliers: Whether to remove outliers

    Returns:
        FusionResult
    """
    fusion = PointCloudFusion(voxel_size=voxel_size, remove_outliers=remove_outliers)
    return fusion.fuse(point_clouds)


def compute_fusion_quality_metrics(result: FusionResult) -> Dict:
    """
    Compute quality metrics for fusion result.

    Args:
        result: FusionResult object

    Returns:
        Dict with quality metrics
    """
    return {
        "num_input_points": result.num_input_points,
        "num_output_points": result.num_output_points,
        "reduction_ratio": result.reduction_ratio,
        "num_outliers_removed": result.num_outliers_removed,
        "num_downsampled": result.num_downsampled,
        "bounding_box_size_m": {
            "x": float(result.bounding_box_size[0]),
            "y": float(result.bounding_box_size[1]),
            "z": float(result.bounding_box_size[2])
        },
        "volume_m3": float(np.prod(result.bounding_box_size)),
        "point_density_per_m3": float(result.num_output_points / np.prod(result.bounding_box_size)),
        "coverage_quality": result.coverage_quality
    }


def estimate_occlusion_reduction(num_cameras: int,
                                 single_camera_coverage: float = 0.6) -> float:
    """
    Estimate occlusion reduction from multi-camera fusion.

    Args:
        num_cameras: Number of cameras
        single_camera_coverage: Coverage from single camera (0-1)

    Returns:
        Estimated total coverage (0-1)
    """
    # Simple model: coverage increases with overlapping views
    # Assumes independent random occlusions
    occlusion_rate = 1.0 - single_camera_coverage
    combined_occlusion_rate = occlusion_rate ** num_cameras
    combined_coverage = 1.0 - combined_occlusion_rate

    return combined_coverage
