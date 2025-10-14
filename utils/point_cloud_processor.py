"""
Point Cloud Processing for RealSense D455i (CPU-Only)

This module provides comprehensive point cloud generation, filtering,
processing, and export capabilities using pyrealsense2 and Open3D.

Author: PGG Desktop App Team
Date: 2025-10-10
"""

import numpy as np
import pyrealsense2 as rs
from pathlib import Path
from typing import Optional, Tuple, Dict, List, Literal
from dataclasses import dataclass
import json

try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False
    print("Open3D not available. Advanced point cloud features disabled.")

try:
    import laspy
    LASPY_AVAILABLE = True
except ImportError:
    LASPY_AVAILABLE = False

try:
    import pye57
    PYE57_AVAILABLE = True
except ImportError:
    PYE57_AVAILABLE = False


@dataclass
class PointCloudStats:
    """Statistics for point cloud analysis."""
    total_points: int
    valid_points: int
    invalid_points: int
    min_bounds: Tuple[float, float, float]
    max_bounds: Tuple[float, float, float]
    dimensions: Tuple[float, float, float]
    density: float
    completeness: float


class PointCloudProcessor:
    """
    Advanced point cloud processing for RealSense cameras (CPU-only).

    Features:
    - Point cloud generation (dense, colored)
    - Filtering (outliers, depth range, spatial bounds)
    - Downsampling (voxel, uniform, random, adaptive)
    - Surface analysis (normals, curvature, density)
    - Registration and transformation
    - Multi-format export (PLY, PCD, XYZ, NPY, LAS, E57)
    - Quality metrics and statistics
    """

    def __init__(self):
        """Initialize point cloud processor."""
        self.rs_pointcloud = rs.pointcloud()
        self.current_points = None
        self.current_colors = None
        self.intrinsics = None

    # ========================================================================
    # POINT CLOUD GENERATION
    # ========================================================================

    def generate_dense_point_cloud(
        self,
        depth_frame: rs.depth_frame
    ) -> Tuple[np.ndarray, int]:
        """
        Generate dense point cloud from depth frame.

        Args:
            depth_frame: RealSense depth frame

        Returns:
            Tuple of (points array [N, 3], point count)
        """
        # Calculate point cloud using RealSense
        points_rs = self.rs_pointcloud.calculate(depth_frame)

        # Get vertices as NumPy array
        vertices = np.asanyarray(points_rs.get_vertices())

        # Convert structured array to regular array [N, 3]
        points = np.zeros((len(vertices), 3), dtype=np.float32)
        points[:, 0] = vertices['f0']  # x
        points[:, 1] = vertices['f1']  # y
        points[:, 2] = vertices['f2']  # z

        self.current_points = points
        valid_count = np.sum(~np.isnan(points[:, 0]))

        return points, valid_count

    def generate_colored_point_cloud(
        self,
        depth_frame: rs.depth_frame,
        color_frame: rs.video_frame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate colored point cloud (XYZRGB) from aligned depth and RGB.

        Args:
            depth_frame: RealSense depth frame
            color_frame: RealSense color frame (aligned to depth)

        Returns:
            Tuple of (points [N, 3], colors [N, 3])
        """
        # Map color to point cloud
        self.rs_pointcloud.map_to(color_frame)

        # Calculate point cloud
        points_rs = self.rs_pointcloud.calculate(depth_frame)

        # Get vertices and texture coordinates
        vertices = np.asanyarray(points_rs.get_vertices())
        texcoords = np.asanyarray(points_rs.get_texture_coordinates())

        # Convert vertices to [N, 3] array
        points = np.zeros((len(vertices), 3), dtype=np.float32)
        points[:, 0] = vertices['f0']
        points[:, 1] = vertices['f1']
        points[:, 2] = vertices['f2']

        # Get color data
        color_data = np.asanyarray(color_frame.get_data())
        height, width = color_data.shape[:2]

        # Map texture coordinates to RGB colors
        colors = np.zeros((len(texcoords), 3), dtype=np.uint8)
        for i, tc in enumerate(texcoords):
            u = int(tc['f0'] * width)
            v = int(tc['f1'] * height)

            # Clamp to image bounds
            u = max(0, min(u, width - 1))
            v = max(0, min(v, height - 1))

            colors[i] = color_data[v, u][:3]  # RGB

        self.current_points = points
        self.current_colors = colors

        return points, colors

    def calculate_point_cloud_coordinates(
        self,
        depth_frame: rs.depth_frame
    ) -> np.ndarray:
        """
        Calculate point cloud coordinates using camera intrinsics.

        Each (u, v, depth) pixel maps to (x, y, z) in camera coordinates.

        Args:
            depth_frame: RealSense depth frame

        Returns:
            Points array [N, 3] in camera-centered coordinate system
        """
        # Get intrinsics
        if self.intrinsics is None:
            self.intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics

        width = self.intrinsics.width
        height = self.intrinsics.height

        # Get depth data
        depth_data = np.asanyarray(depth_frame.get_data())

        # Create pixel coordinates
        u_coords, v_coords = np.meshgrid(np.arange(width), np.arange(height))
        u_coords = u_coords.flatten()
        v_coords = v_coords.flatten()
        depth_values = depth_data.flatten()

        # Deproject to 3D
        points = []
        for u, v, d in zip(u_coords, v_coords, depth_values):
            if d > 0:  # Valid depth
                point = rs.rs2_deproject_pixel_to_point(
                    self.intrinsics, [u, v], d * depth_frame.get_units()
                )
                points.append(point)

        points = np.array(points, dtype=np.float32)
        self.current_points = points

        return points

    # ========================================================================
    # FILTERING
    # ========================================================================

    def filter_invalid_points(
        self,
        points: np.ndarray,
        colors: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Filter invalid points (NaN, infinite, zero-depth).

        Args:
            points: Point cloud [N, 3]
            colors: Optional colors [N, 3]

        Returns:
            Tuple of (filtered_points, filtered_colors)
        """
        # Find valid points
        valid_mask = (
            ~np.isnan(points).any(axis=1) &
            ~np.isinf(points).any(axis=1) &
            (points[:, 2] != 0)  # Non-zero depth
        )

        filtered_points = points[valid_mask]
        filtered_colors = colors[valid_mask] if colors is not None else None

        return filtered_points, filtered_colors

    def apply_statistical_outlier_removal(
        self,
        points: np.ndarray,
        colors: Optional[np.ndarray] = None,
        nb_neighbors: int = 20,
        std_ratio: float = 2.0
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Apply statistical outlier removal using Open3D.

        Removes points that are farther from their neighbors than average.

        Args:
            points: Point cloud [N, 3]
            colors: Optional colors [N, 3]
            nb_neighbors: Number of neighbors to analyze
            std_ratio: Standard deviation ratio threshold

        Returns:
            Tuple of (filtered_points, filtered_colors)
        """
        if not OPEN3D_AVAILABLE:
            print("Open3D not available. Skipping statistical outlier removal.")
            return points, colors

        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        if colors is not None:
            pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)

        # Apply statistical outlier removal
        cl, ind = pcd.remove_statistical_outlier(
            nb_neighbors=nb_neighbors,
            std_ratio=std_ratio
        )

        # Extract filtered points
        filtered_points = np.asarray(cl.points)
        filtered_colors = None
        if colors is not None:
            filtered_colors = (np.asarray(cl.colors) * 255).astype(np.uint8)

        return filtered_points, filtered_colors

    def apply_radius_outlier_removal(
        self,
        points: np.ndarray,
        colors: Optional[np.ndarray] = None,
        radius: float = 0.05,
        min_neighbors: int = 10
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Apply radius outlier removal using Open3D.

        Removes isolated points with fewer than min_neighbors within radius.

        Args:
            points: Point cloud [N, 3]
            colors: Optional colors [N, 3]
            radius: Search radius in meters
            min_neighbors: Minimum number of neighbors required

        Returns:
            Tuple of (filtered_points, filtered_colors)
        """
        if not OPEN3D_AVAILABLE:
            print("Open3D not available. Skipping radius outlier removal.")
            return points, colors

        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        if colors is not None:
            pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)

        # Apply radius outlier removal
        cl, ind = pcd.remove_radius_outlier(
            nb_points=min_neighbors,
            radius=radius
        )

        # Extract filtered points
        filtered_points = np.asarray(cl.points)
        filtered_colors = None
        if colors is not None:
            filtered_colors = (np.asarray(cl.colors) * 255).astype(np.uint8)

        return filtered_points, filtered_colors

    def apply_depth_range_filter(
        self,
        points: np.ndarray,
        colors: Optional[np.ndarray] = None,
        min_depth: float = 0.5,
        max_depth: float = 3.0
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Apply conditional filtering by depth range.

        Args:
            points: Point cloud [N, 3]
            colors: Optional colors [N, 3]
            min_depth: Minimum depth (z) in meters
            max_depth: Maximum depth (z) in meters

        Returns:
            Tuple of (filtered_points, filtered_colors)
        """
        # Filter by z coordinate (depth)
        valid_mask = (points[:, 2] >= min_depth) & (points[:, 2] <= max_depth)

        filtered_points = points[valid_mask]
        filtered_colors = colors[valid_mask] if colors is not None else None

        return filtered_points, filtered_colors

    def apply_spatial_bounds_filter(
        self,
        points: np.ndarray,
        colors: Optional[np.ndarray] = None,
        x_range: Tuple[float, float] = (-1.0, 1.0),
        y_range: Tuple[float, float] = (-1.0, 1.0),
        z_range: Tuple[float, float] = (0.0, 3.0)
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Apply pass-through filtering by spatial bounds (ROI extraction).

        Args:
            points: Point cloud [N, 3]
            colors: Optional colors [N, 3]
            x_range: (min_x, max_x) bounds
            y_range: (min_y, max_y) bounds
            z_range: (min_z, max_z) bounds

        Returns:
            Tuple of (filtered_points, filtered_colors)
        """
        valid_mask = (
            (points[:, 0] >= x_range[0]) & (points[:, 0] <= x_range[1]) &
            (points[:, 1] >= y_range[0]) & (points[:, 1] <= y_range[1]) &
            (points[:, 2] >= z_range[0]) & (points[:, 2] <= z_range[1])
        )

        filtered_points = points[valid_mask]
        filtered_colors = colors[valid_mask] if colors is not None else None

        return filtered_points, filtered_colors

    # ========================================================================
    # DOWNSAMPLING
    # ========================================================================

    def apply_voxel_grid_downsampling(
        self,
        points: np.ndarray,
        colors: Optional[np.ndarray] = None,
        voxel_size: float = 0.01
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Apply voxel grid downsampling.

        One representative point per voxel, preserving cloud shape.

        Args:
            points: Point cloud [N, 3]
            colors: Optional colors [N, 3]
            voxel_size: Voxel size in meters (e.g., 0.01 = 1cm)

        Returns:
            Tuple of (downsampled_points, downsampled_colors)
        """
        if not OPEN3D_AVAILABLE:
            print("Open3D not available. Skipping voxel downsampling.")
            return points, colors

        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        if colors is not None:
            pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)

        # Apply voxel grid downsampling
        downsampled_pcd = pcd.voxel_down_sample(voxel_size=voxel_size)

        # Extract downsampled points
        downsampled_points = np.asarray(downsampled_pcd.points)
        downsampled_colors = None
        if colors is not None:
            downsampled_colors = (np.asarray(downsampled_pcd.colors) * 255).astype(np.uint8)

        return downsampled_points, downsampled_colors

    def apply_uniform_downsampling(
        self,
        points: np.ndarray,
        colors: Optional[np.ndarray] = None,
        every_n: int = 10
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Apply uniform downsampling (select every Nth point).

        Args:
            points: Point cloud [N, 3]
            colors: Optional colors [N, 3]
            every_n: Select every Nth point

        Returns:
            Tuple of (downsampled_points, downsampled_colors)
        """
        downsampled_points = points[::every_n]
        downsampled_colors = colors[::every_n] if colors is not None else None

        return downsampled_points, downsampled_colors

    def apply_random_downsampling(
        self,
        points: np.ndarray,
        colors: Optional[np.ndarray] = None,
        ratio: float = 0.1
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Apply random downsampling.

        Args:
            points: Point cloud [N, 3]
            colors: Optional colors [N, 3]
            ratio: Sampling ratio (e.g., 0.1 = 10% of points)

        Returns:
            Tuple of (downsampled_points, downsampled_colors)
        """
        n_points = len(points)
        n_sample = int(n_points * ratio)

        indices = np.random.choice(n_points, n_sample, replace=False)

        downsampled_points = points[indices]
        downsampled_colors = colors[indices] if colors is not None else None

        return downsampled_points, downsampled_colors

    def apply_adaptive_downsampling(
        self,
        points: np.ndarray,
        colors: Optional[np.ndarray] = None,
        target_density: float = 1000.0
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Apply adaptive downsampling (downsample dense regions more).

        Args:
            points: Point cloud [N, 3]
            colors: Optional colors [N, 3]
            target_density: Target points per cubic meter

        Returns:
            Tuple of (downsampled_points, downsampled_colors)
        """
        # Calculate voxel size based on target density
        # density = points per m^3, so voxel_size = (1 / density)^(1/3)
        voxel_size = (1.0 / target_density) ** (1.0 / 3.0)

        # Use voxel downsampling to achieve uniform density
        return self.apply_voxel_grid_downsampling(points, colors, voxel_size)

    # ========================================================================
    # SURFACE ANALYSIS
    # ========================================================================

    def estimate_surface_normals(
        self,
        points: np.ndarray,
        search_radius: float = 0.05,
        max_nn: int = 30
    ) -> np.ndarray:
        """
        Estimate surface normals using Open3D.

        Args:
            points: Point cloud [N, 3]
            search_radius: Search radius for normal estimation
            max_nn: Maximum nearest neighbors

        Returns:
            Normals array [N, 3] (unit vectors)
        """
        if not OPEN3D_AVAILABLE:
            print("Open3D not available. Cannot estimate normals.")
            return np.zeros_like(points)

        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        # Estimate normals
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=search_radius,
                max_nn=max_nn
            )
        )

        # Extract normals
        normals = np.asarray(pcd.normals)

        return normals

    def calculate_surface_curvature(
        self,
        points: np.ndarray,
        normals: np.ndarray,
        search_radius: float = 0.05
    ) -> np.ndarray:
        """
        Calculate surface curvature.

        Curvature indicates surface flatness (high = edges/corners).

        Args:
            points: Point cloud [N, 3]
            normals: Normals [N, 3]
            search_radius: Neighborhood radius

        Returns:
            Curvature values [N] (0 = flat, higher = curved)
        """
        if not OPEN3D_AVAILABLE:
            print("Open3D not available. Cannot compute curvature.")
            return np.zeros(len(points))

        # Create Open3D point cloud with normals
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.normals = o3d.utility.Vector3dVector(normals)

        # Build KD tree
        pcd_tree = o3d.geometry.KDTreeFlann(pcd)

        curvatures = np.zeros(len(points))

        for i in range(len(points)):
            # Find neighbors
            [k, idx, _] = pcd_tree.search_radius_vector_3d(pcd.points[i], search_radius)

            if k > 1:
                # Calculate normal variation
                neighbor_normals = normals[idx]
                normal_diffs = np.linalg.norm(neighbor_normals - normals[i], axis=1)
                curvatures[i] = np.mean(normal_diffs)

        return curvatures

    def calculate_point_cloud_density(
        self,
        points: np.ndarray,
        voxel_size: float = 0.1
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Calculate point cloud density (points per unit volume).

        Args:
            points: Point cloud [N, 3]
            voxel_size: Voxel size for density calculation

        Returns:
            Tuple of (density_map [grid], density_stats dict)
        """
        # Calculate bounding box
        min_bounds = points.min(axis=0)
        max_bounds = points.max(axis=0)

        # Create voxel grid
        grid_size = ((max_bounds - min_bounds) / voxel_size).astype(int) + 1
        density_grid = np.zeros(grid_size, dtype=np.int32)

        # Count points per voxel
        for point in points:
            idx = ((point - min_bounds) / voxel_size).astype(int)
            idx = np.clip(idx, 0, grid_size - 1)
            density_grid[tuple(idx)] += 1

        # Calculate statistics
        voxel_volume = voxel_size ** 3
        densities = density_grid.flatten() / voxel_volume

        stats = {
            'mean_density': float(np.mean(densities[densities > 0])),
            'max_density': float(np.max(densities)),
            'min_density': float(np.min(densities[densities > 0])) if np.any(densities > 0) else 0,
            'total_occupied_voxels': int(np.sum(density_grid > 0)),
        }

        return density_grid, stats

    def calculate_coverage_area(
        self,
        points: np.ndarray,
        grid_resolution: float = 0.01
    ) -> Dict[str, float]:
        """
        Calculate coverage area (projected 2D area).

        Args:
            points: Point cloud [N, 3]
            grid_resolution: Grid resolution for coverage calculation

        Returns:
            Coverage statistics dict
        """
        # Project to XY plane
        xy_points = points[:, :2]

        # Calculate bounding box
        min_xy = xy_points.min(axis=0)
        max_xy = xy_points.max(axis=0)

        # Create 2D grid
        grid_size = ((max_xy - min_xy) / grid_resolution).astype(int) + 1
        coverage_grid = np.zeros(grid_size, dtype=bool)

        # Mark occupied cells
        for point in xy_points:
            idx = ((point - min_xy) / grid_resolution).astype(int)
            idx = np.clip(idx, 0, grid_size - 1)
            coverage_grid[tuple(idx)] = True

        # Calculate coverage
        cell_area = grid_resolution ** 2
        covered_area = np.sum(coverage_grid) * cell_area
        total_area = grid_size[0] * grid_size[1] * cell_area
        coverage_percentage = (covered_area / total_area) * 100

        return {
            'covered_area_m2': float(covered_area),
            'total_area_m2': float(total_area),
            'coverage_percentage': float(coverage_percentage),
            'gap_count': int(np.sum(~coverage_grid)),
        }

    # ========================================================================
    # CLOUD OPERATIONS
    # ========================================================================

    def merge_point_clouds(
        self,
        clouds: List[np.ndarray],
        colors_list: Optional[List[np.ndarray]] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Merge multiple point clouds into single cloud.

        Args:
            clouds: List of point clouds [N_i, 3]
            colors_list: Optional list of color arrays [N_i, 3]

        Returns:
            Tuple of (merged_points, merged_colors)
        """
        merged_points = np.vstack(clouds)

        merged_colors = None
        if colors_list is not None:
            merged_colors = np.vstack(colors_list)

        return merged_points, merged_colors

    def register_point_clouds_icp(
        self,
        source_points: np.ndarray,
        target_points: np.ndarray,
        max_correspondence_distance: float = 0.05,
        max_iterations: int = 50
    ) -> Tuple[np.ndarray, float]:
        """
        Register point clouds using ICP (Iterative Closest Point).

        Args:
            source_points: Source point cloud [N, 3]
            target_points: Target point cloud [M, 3]
            max_correspondence_distance: Max distance for correspondence
            max_iterations: Maximum ICP iterations

        Returns:
            Tuple of (transformation_matrix [4, 4], registration_error)
        """
        if not OPEN3D_AVAILABLE:
            print("Open3D not available. Cannot perform ICP.")
            return np.eye(4), float('inf')

        # Create Open3D point clouds
        source_pcd = o3d.geometry.PointCloud()
        source_pcd.points = o3d.utility.Vector3dVector(source_points)

        target_pcd = o3d.geometry.PointCloud()
        target_pcd.points = o3d.utility.Vector3dVector(target_points)

        # Perform ICP registration
        reg_result = o3d.pipelines.registration.registration_icp(
            source_pcd,
            target_pcd,
            max_correspondence_distance,
            np.eye(4),
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(
                max_iteration=max_iterations
            )
        )

        return np.asarray(reg_result.transformation), reg_result.inlier_rmse

    def transform_point_cloud(
        self,
        points: np.ndarray,
        transformation_matrix: np.ndarray
    ) -> np.ndarray:
        """
        Transform point cloud coordinates.

        Args:
            points: Point cloud [N, 3]
            transformation_matrix: 4x4 transformation matrix

        Returns:
            Transformed points [N, 3]
        """
        # Convert to homogeneous coordinates
        points_homogeneous = np.hstack([points, np.ones((len(points), 1))])

        # Apply transformation
        transformed_homogeneous = (transformation_matrix @ points_homogeneous.T).T

        # Convert back to 3D
        transformed_points = transformed_homogeneous[:, :3]

        return transformed_points

    def crop_point_cloud_to_roi(
        self,
        points: np.ndarray,
        colors: Optional[np.ndarray] = None,
        min_bound: Tuple[float, float, float] = (-1, -1, 0),
        max_bound: Tuple[float, float, float] = (1, 1, 3)
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Crop point cloud to ROI (bounding box).

        Args:
            points: Point cloud [N, 3]
            colors: Optional colors [N, 3]
            min_bound: Minimum (x, y, z) bound
            max_bound: Maximum (x, y, z) bound

        Returns:
            Tuple of (cropped_points, cropped_colors)
        """
        if not OPEN3D_AVAILABLE:
            # Use NumPy fallback
            valid_mask = (
                (points[:, 0] >= min_bound[0]) & (points[:, 0] <= max_bound[0]) &
                (points[:, 1] >= min_bound[1]) & (points[:, 1] <= max_bound[1]) &
                (points[:, 2] >= min_bound[2]) & (points[:, 2] <= max_bound[2])
            )
            return points[valid_mask], colors[valid_mask] if colors is not None else None

        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        if colors is not None:
            pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)

        # Create bounding box
        bbox = o3d.geometry.AxisAlignedBoundingBox(
            min_bound=min_bound,
            max_bound=max_bound
        )

        # Crop
        cropped_pcd = pcd.crop(bbox)

        cropped_points = np.asarray(cropped_pcd.points)
        cropped_colors = None
        if colors is not None:
            cropped_colors = (np.asarray(cropped_pcd.colors) * 255).astype(np.uint8)

        return cropped_points, cropped_colors

    # ========================================================================
    # EXPORT FORMATS
    # ========================================================================

    def export_to_ply_ascii(
        self,
        points: np.ndarray,
        colors: Optional[np.ndarray] = None,
        output_path: Path = Path("pointcloud.ply")
    ):
        """
        Export point cloud to PLY ASCII format.

        Args:
            points: Point cloud [N, 3]
            colors: Optional colors [N, 3]
            output_path: Output file path
        """
        if OPEN3D_AVAILABLE:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            if colors is not None:
                pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)
            o3d.io.write_point_cloud(str(output_path), pcd, write_ascii=True)
        else:
            # Manual PLY writing
            n_points = len(points)
            has_color = colors is not None

            with open(output_path, 'w') as f:
                # Header
                f.write("ply\n")
                f.write("format ascii 1.0\n")
                f.write(f"element vertex {n_points}\n")
                f.write("property float x\n")
                f.write("property float y\n")
                f.write("property float z\n")
                if has_color:
                    f.write("property uchar red\n")
                    f.write("property uchar green\n")
                    f.write("property uchar blue\n")
                f.write("end_header\n")

                # Vertices
                for i in range(n_points):
                    if has_color:
                        f.write(f"{points[i, 0]} {points[i, 1]} {points[i, 2]} "
                               f"{colors[i, 0]} {colors[i, 1]} {colors[i, 2]}\n")
                    else:
                        f.write(f"{points[i, 0]} {points[i, 1]} {points[i, 2]}\n")

    def export_to_ply_binary(
        self,
        points: np.ndarray,
        colors: Optional[np.ndarray] = None,
        output_path: Path = Path("pointcloud.ply")
    ):
        """
        Export point cloud to PLY binary format.

        Args:
            points: Point cloud [N, 3]
            colors: Optional colors [N, 3]
            output_path: Output file path
        """
        if OPEN3D_AVAILABLE:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            if colors is not None:
                pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)
            o3d.io.write_point_cloud(str(output_path), pcd, write_ascii=False)
        else:
            print("Open3D required for binary PLY export.")

    def export_to_pcd(
        self,
        points: np.ndarray,
        colors: Optional[np.ndarray] = None,
        output_path: Path = Path("pointcloud.pcd")
    ):
        """
        Export point cloud to PCD (Point Cloud Data) format.

        Args:
            points: Point cloud [N, 3]
            colors: Optional colors [N, 3]
            output_path: Output file path
        """
        if not OPEN3D_AVAILABLE:
            print("Open3D required for PCD export.")
            return

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        if colors is not None:
            pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)
        o3d.io.write_point_cloud(str(output_path), pcd)

    def export_to_xyz(
        self,
        points: np.ndarray,
        colors: Optional[np.ndarray] = None,
        output_path: Path = Path("pointcloud.xyz")
    ):
        """
        Export point cloud to XYZ text format.

        Simple ASCII format: x y z [r g b]

        Args:
            points: Point cloud [N, 3]
            colors: Optional colors [N, 3]
            output_path: Output file path
        """
        with open(output_path, 'w') as f:
            for i in range(len(points)):
                if colors is not None:
                    f.write(f"{points[i, 0]} {points[i, 1]} {points[i, 2]} "
                           f"{colors[i, 0]} {colors[i, 1]} {colors[i, 2]}\n")
                else:
                    f.write(f"{points[i, 0]} {points[i, 1]} {points[i, 2]}\n")

    def export_to_npy(
        self,
        points: np.ndarray,
        colors: Optional[np.ndarray] = None,
        output_path: Path = Path("pointcloud.npy")
    ):
        """
        Export point cloud to NPY format.

        Args:
            points: Point cloud [N, 3]
            colors: Optional colors [N, 3]
            output_path: Output file path
        """
        np.save(output_path, points)

        if colors is not None:
            colors_path = output_path.parent / f"{output_path.stem}_colors.npy"
            np.save(colors_path, colors)

    def export_to_las(
        self,
        points: np.ndarray,
        colors: Optional[np.ndarray] = None,
        output_path: Path = Path("pointcloud.las")
    ):
        """
        Export point cloud to LAS (LiDAR) format.

        Args:
            points: Point cloud [N, 3]
            colors: Optional colors [N, 3]
            output_path: Output file path
        """
        if not LASPY_AVAILABLE:
            print("laspy library required for LAS export. Install: pip install laspy")
            return

        # Create LAS file
        header = laspy.LasHeader(version="1.4", point_format=3 if colors is not None else 1)
        las = laspy.LasData(header)

        # Set coordinates
        las.x = points[:, 0]
        las.y = points[:, 1]
        las.z = points[:, 2]

        # Set colors if available
        if colors is not None:
            las.red = (colors[:, 0] * 256).astype(np.uint16)
            las.green = (colors[:, 1] * 256).astype(np.uint16)
            las.blue = (colors[:, 2] * 256).astype(np.uint16)

        # Write file
        las.write(str(output_path))

    def export_to_e57(
        self,
        points: np.ndarray,
        colors: Optional[np.ndarray] = None,
        output_path: Path = Path("pointcloud.e57")
    ):
        """
        Export point cloud to E57 (ASTM standard) format.

        Args:
            points: Point cloud [N, 3]
            colors: Optional colors [N, 3]
            output_path: Output file path
        """
        if not PYE57_AVAILABLE:
            print("pye57 library required for E57 export. Install: pip install pye57")
            return

        # Create E57 file
        e57 = pye57.E57(str(output_path), mode='w')

        # Prepare data
        data = {
            "cartesianX": points[:, 0].astype(np.float64),
            "cartesianY": points[:, 1].astype(np.float64),
            "cartesianZ": points[:, 2].astype(np.float64),
        }

        if colors is not None:
            data["colorRed"] = colors[:, 0].astype(np.uint8)
            data["colorGreen"] = colors[:, 1].astype(np.uint8)
            data["colorBlue"] = colors[:, 2].astype(np.uint8)

        # Write scan
        e57.write_scan_raw(data)
        e57.close()

    # ========================================================================
    # STATISTICS & METRICS
    # ========================================================================

    def calculate_point_count_statistics(
        self,
        points: np.ndarray
    ) -> Dict[str, int]:
        """
        Calculate point count statistics.

        Args:
            points: Point cloud [N, 3]

        Returns:
            Statistics dict
        """
        total_count = len(points)

        # Count valid points (non-NaN, non-inf, non-zero-depth)
        valid_mask = (
            ~np.isnan(points).any(axis=1) &
            ~np.isinf(points).any(axis=1) &
            (points[:, 2] != 0)
        )
        valid_count = np.sum(valid_mask)
        invalid_count = total_count - valid_count

        return {
            'total_points': total_count,
            'valid_points': int(valid_count),
            'invalid_points': int(invalid_count),
        }

    def calculate_bounding_box(
        self,
        points: np.ndarray
    ) -> Dict[str, Tuple[float, float, float]]:
        """
        Calculate point cloud bounding box.

        Args:
            points: Point cloud [N, 3]

        Returns:
            Bounding box dict with min, max, dimensions
        """
        min_bounds = tuple(points.min(axis=0))
        max_bounds = tuple(points.max(axis=0))
        dimensions = tuple(max_bounds[i] - min_bounds[i] for i in range(3))

        return {
            'min_bounds': min_bounds,
            'max_bounds': max_bounds,
            'dimensions': dimensions,
        }

    def calculate_quality_metrics(
        self,
        points: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate point cloud quality metrics.

        Args:
            points: Point cloud [N, 3]

        Returns:
            Quality metrics dict
        """
        # Completeness
        valid_mask = (
            ~np.isnan(points).any(axis=1) &
            ~np.isinf(points).any(axis=1) &
            (points[:, 2] != 0)
        )
        completeness = np.sum(valid_mask) / len(points) * 100

        # Noise level (std of z-coordinates)
        valid_points = points[valid_mask]
        noise_level = float(np.std(valid_points[:, 2])) if len(valid_points) > 0 else 0

        # Uniformity (coefficient of variation of point distances)
        if len(valid_points) > 1 and OPEN3D_AVAILABLE:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(valid_points)
            distances = pcd.compute_nearest_neighbor_distance()
            uniformity = float(np.std(distances) / np.mean(distances)) if np.mean(distances) > 0 else 0
        else:
            uniformity = 0.0

        return {
            'completeness_percentage': float(completeness),
            'noise_level': noise_level,
            'uniformity_score': uniformity,
        }

    def generate_density_map(
        self,
        points: np.ndarray,
        voxel_size: float = 0.1
    ) -> np.ndarray:
        """
        Generate density heatmap.

        Args:
            points: Point cloud [N, 3]
            voxel_size: Voxel size for density calculation

        Returns:
            Density grid [Nx, Ny, Nz]
        """
        density_grid, _ = self.calculate_point_cloud_density(points, voxel_size)
        return density_grid

    def check_point_cloud_completeness(
        self,
        points: np.ndarray
    ) -> Dict[str, float]:
        """
        Check point cloud completeness.

        Args:
            points: Point cloud [N, 3]

        Returns:
            Completeness metrics dict
        """
        valid_mask = (
            ~np.isnan(points).any(axis=1) &
            ~np.isinf(points).any(axis=1) &
            (points[:, 2] != 0)
        )

        valid_percentage = np.sum(valid_mask) / len(points) * 100
        missing_percentage = 100 - valid_percentage

        return {
            'valid_percentage': float(valid_percentage),
            'missing_percentage': float(missing_percentage),
            'completeness_score': float(valid_percentage / 100),
        }

    # ========================================================================
    # ADDITIONAL OPERATIONS (Scenario 5 Feature 1)
    # ========================================================================

    def orient_normals_consistently(
        self,
        points: np.ndarray,
        normals: np.ndarray,
        k: int = 10
    ) -> np.ndarray:
        """
        Orient normals consistently using tangent plane method.

        Args:
            points: Point cloud [N, 3]
            normals: Normals [N, 3]
            k: Number of neighbors for consistency check

        Returns:
            Consistently oriented normals [N, 3]
        """
        if not OPEN3D_AVAILABLE:
            print("Open3D not available. Cannot orient normals.")
            return normals

        # Create Open3D point cloud with normals
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.normals = o3d.utility.Vector3dVector(normals)

        # Orient normals consistently
        pcd.orient_normals_consistent_tangent_plane(k)

        # Extract oriented normals
        oriented_normals = np.asarray(pcd.normals)

        return oriented_normals

    def calculate_centroid(
        self,
        points: np.ndarray
    ) -> Tuple[float, float, float]:
        """
        Calculate point cloud centroid (center point).

        Args:
            points: Point cloud [N, 3]

        Returns:
            Centroid (x, y, z)
        """
        if OPEN3D_AVAILABLE:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            centroid = pcd.get_center()
            return tuple(centroid)
        else:
            # NumPy fallback
            centroid = np.mean(points, axis=0)
            return tuple(centroid)

    def visualize_point_cloud(
        self,
        points: np.ndarray,
        colors: Optional[np.ndarray] = None,
        normals: Optional[np.ndarray] = None,
        window_name: str = "Point Cloud Viewer"
    ):
        """
        Visualize point cloud with Open3D viewer.

        Args:
            points: Point cloud [N, 3]
            colors: Optional colors [N, 3]
            normals: Optional normals [N, 3]
            window_name: Viewer window title
        """
        if not OPEN3D_AVAILABLE:
            print("Open3D not available. Cannot visualize point cloud.")
            return

        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        if colors is not None:
            pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)

        if normals is not None:
            pcd.normals = o3d.utility.Vector3dVector(normals)

        # Visualize
        o3d.visualization.draw_geometries(
            [pcd],
            window_name=window_name,
            width=1920,
            height=1080,
            left=50,
            top=50,
            point_show_normal=normals is not None
        )

    def save_point_cloud_metadata(
        self,
        points: np.ndarray,
        colors: Optional[np.ndarray] = None,
        normals: Optional[np.ndarray] = None,
        camera_intrinsics: Optional[Dict] = None,
        output_path: Path = Path("pointcloud_metadata.json")
    ):
        """
        Save point cloud metadata to JSON file.

        Args:
            points: Point cloud [N, 3]
            colors: Optional colors [N, 3]
            normals: Optional normals [N, 3]
            camera_intrinsics: Optional camera intrinsics dict
            output_path: Output JSON file path
        """
        from datetime import datetime

        # Calculate bounding box
        bbox = self.calculate_bounding_box(points)

        # Calculate statistics
        stats = self.calculate_point_count_statistics(points)

        # Build metadata
        metadata = {
            'timestamp': datetime.now().isoformat(),
            'point_count': int(len(points)),
            'valid_points': stats['valid_points'],
            'invalid_points': stats['invalid_points'],
            'bounding_box': {
                'min': list(bbox['min_bounds']),
                'max': list(bbox['max_bounds']),
                'dimensions': list(bbox['dimensions']),
            },
            'has_colors': colors is not None,
            'has_normals': normals is not None,
        }

        if camera_intrinsics is not None:
            metadata['camera_intrinsics'] = camera_intrinsics

        # Calculate additional metrics
        if len(points) > 0:
            centroid = self.calculate_centroid(points)
            metadata['centroid'] = list(centroid)

            density_stats = self.calculate_point_cloud_density(points)
            metadata['density_statistics'] = density_stats[1]

        # Write to JSON
        with open(output_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"Metadata saved to {output_path}")

    def process_point_cloud_pipeline(
        self,
        depth_frame: rs.depth_frame,
        color_frame: rs.video_frame,
        output_dir: Path = Path("output"),
        voxel_size: float = 0.01,
        remove_outliers: bool = True,
        estimate_normals: bool = True,
        export_format: Literal['ply', 'pcd', 'xyz', 'npy'] = 'ply'
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        Execute complete point cloud processing pipeline end-to-end.

        Pipeline steps:
        1. Calculate point cloud from depth
        2. Map RGB colors to points
        3. Convert to NumPy arrays
        4. Filter invalid points
        5. Create Open3D point cloud
        6. Downsample with voxel grid
        7. Remove outliers (optional)
        8. Estimate normals (optional)
        9. Export to specified format
        10. Save metadata

        Args:
            depth_frame: RealSense depth frame
            color_frame: RealSense color frame (aligned)
            output_dir: Output directory for exports
            voxel_size: Voxel size for downsampling (meters)
            remove_outliers: Whether to apply outlier removal
            estimate_normals: Whether to estimate surface normals
            export_format: Export format ('ply', 'pcd', 'xyz', 'npy')

        Returns:
            Tuple of (final_points, final_colors, final_normals)
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        print("Starting point cloud processing pipeline...")

        # Step 1-2: Calculate colored point cloud
        print("  1. Calculating point cloud from depth...")
        print("  2. Mapping RGB colors to points...")
        points, colors = self.generate_colored_point_cloud(depth_frame, color_frame)
        print(f"     Generated {len(points)} points")

        # Step 3-4: Filter invalid points
        print("  3. Converting to NumPy arrays...")
        print("  4. Filtering invalid points...")
        points, colors = self.filter_invalid_points(points, colors)
        print(f"     {len(points)} valid points after filtering")

        if len(points) == 0:
            print("No valid points after filtering. Aborting pipeline.")
            return np.array([]), np.array([]), None

        # Step 5: Create Open3D point cloud (implicit in subsequent operations)
        print("  5. Creating Open3D point cloud...")

        # Step 6: Downsample with voxel grid
        print(f"  6. Downsampling with voxel grid (voxel_size={voxel_size}m)...")
        points, colors = self.apply_voxel_grid_downsampling(points, colors, voxel_size)
        print(f"     {len(points)} points after downsampling")

        # Step 7: Remove outliers
        normals = None
        if remove_outliers:
            print("  7. Removing statistical outliers...")
            points, colors = self.apply_statistical_outlier_removal(
                points, colors, nb_neighbors=20, std_ratio=2.0
            )
            print(f"     {len(points)} points after outlier removal")
        else:
            print("  7. Skipping outlier removal")

        # Step 8: Estimate normals
        if estimate_normals:
            print("  8. Estimating surface normals...")
            normals = self.estimate_surface_normals(
                points, search_radius=voxel_size * 5, max_nn=30
            )
            print(f"     Estimated normals for {len(normals)} points")
        else:
            print("  8. Skipping normal estimation")

        # Step 9: Export to specified format
        print(f"  9. Exporting to {export_format.upper()} format...")
        if export_format == 'ply':
            output_path = output_dir / "pointcloud.ply"
            self.export_to_ply_binary(points, colors, output_path)
        elif export_format == 'pcd':
            output_path = output_dir / "pointcloud.pcd"
            self.export_to_pcd(points, colors, output_path)
        elif export_format == 'xyz':
            output_path = output_dir / "pointcloud.xyz"
            self.export_to_xyz(points, colors, output_path)
        elif export_format == 'npy':
            output_path = output_dir / "pointcloud.npy"
            self.export_to_npy(points, colors, output_path)
        print(f"     Exported to {output_path}")

        # Step 10: Save metadata
        print(" 10. Saving metadata...")
        metadata_path = output_dir / "pointcloud_metadata.json"

        # Extract camera intrinsics if available
        camera_intrinsics = None
        if hasattr(depth_frame, 'profile'):
            profile = depth_frame.profile.as_video_stream_profile()
            intrinsics = profile.intrinsics
            camera_intrinsics = {
                'width': intrinsics.width,
                'height': intrinsics.height,
                'fx': intrinsics.fx,
                'fy': intrinsics.fy,
                'ppx': intrinsics.ppx,
                'ppy': intrinsics.ppy,
                'model': str(intrinsics.model),
                'coeffs': list(intrinsics.coeffs),
            }

        self.save_point_cloud_metadata(
            points, colors, normals, camera_intrinsics, metadata_path
        )

        print("Point cloud processing pipeline completed successfully!")
        print(f"   Final point count: {len(points)}")
        print(f"   Output directory: {output_dir}")

        return points, colors, normals
