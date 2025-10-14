"""
Cow Volume and Dimensional Measurements (CPU-Only)

This module provides comprehensive volume and weight estimation specifically for
cattle (cow, bull, calf, heifer, steer) using RealSense depth data.

Author: PGG Desktop App Team
Date: 2025-10-10
"""

import numpy as np
import pyrealsense2 as rs
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Literal
from datetime import datetime
import json
import csv

try:
    from scipy.spatial import ConvexHull
    from scipy.optimize import curve_fit
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("scipy not available. Some volume calculations disabled.")

try:
    import trimesh
    TRIMESH_AVAILABLE = True
except ImportError:
    TRIMESH_AVAILABLE = False
    print("trimesh not available. Mesh volume calculations disabled.")

try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False
    print("Open3D not available. Advanced features disabled.")

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    print("pandas not available. CSV operations limited.")

try:
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("scikit-learn not available. Weight regression disabled.")

from .animal_volume_measurements import (
    AnimalVolumeEstimator,
    VolumeMetrics,
    BodyDimensions,
    WeightEstimate,
    MeasurementReport
)


class CowVolumeEstimator(AnimalVolumeEstimator):
    """
    Cow-specific volume and weight estimation.

    Supports: cow, bull, calf, heifer, steer

    Features:
    - Multiple volume calculation methods
    - Body dimension measurements
    - Weight estimation with calibration
    - Growth tracking over time
    - Herd statistics and analysis
    """

    # Cattle-specific constants
    COW_DENSITY_KG_M3 = 1000.0  # Average cattle density
    COW_VOXEL_SIZE = 0.01  # 1cm voxels for large animals

    # Weight estimation models (can be calibrated)
    _weight_model_linear = None  # sklearn LinearRegression
    _weight_model_poly = None  # sklearn PolynomialFeatures + LinearRegression
    _calibration_r2 = 0.0

    def __init__(self, animal_subtype: str = 'cow'):
        """
        Initialize cow volume estimator.

        Args:
            animal_subtype: 'cow', 'bull', 'calf', 'heifer', or 'steer'
        """
        super().__init__(animal_type='cattle')
        self.animal_subtype = animal_subtype
        self.volume_history: Dict[str, List[Tuple[datetime, float]]] = {}

    # ========================================================================
    # POINT CLOUD EXTRACTION
    # ========================================================================

    def extract_animal_point_cloud(
        self,
        depth_frame: np.ndarray,
        bounding_box: Tuple[int, int, int, int]
    ) -> np.ndarray:
        """
        Extract cow point cloud from bounding box.

        Args:
            depth_frame: Depth frame data [H, W]
            bounding_box: (x_min, x_max, y_min, y_max)

        Returns:
            Point cloud [N, 3] in meters
        """
        x_min, x_max, y_min, y_max = bounding_box

        # Crop depth ROI
        depth_roi = depth_frame[y_min:y_max, x_min:x_max]

        # Convert to point cloud (simplified - assumes known intrinsics)
        # In real implementation, use camera intrinsics from RealSense
        points = []
        for v in range(depth_roi.shape[0]):
            for u in range(depth_roi.shape[1]):
                depth = depth_roi[v, u]
                if depth > 0:  # Valid depth
                    # Simplified projection (replace with proper intrinsics)
                    x = (u - depth_roi.shape[1] / 2) * depth * 0.001
                    y = (v - depth_roi.shape[0] / 2) * depth * 0.001
                    z = depth * 0.001  # Convert mm to meters
                    points.append([x, y, z])

        return np.array(points, dtype=np.float32)

    def remove_ground_plane(
        self,
        points: np.ndarray,
        distance_threshold: float = 0.02
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Remove ground plane from point cloud using RANSAC.

        Args:
            points: Point cloud [N, 3]
            distance_threshold: RANSAC distance threshold in meters

        Returns:
            Tuple of (animal_points [M, 3], plane_coefficients [4])
        """
        if not OPEN3D_AVAILABLE:
            # Simple fallback: remove bottom 10% of points
            z_threshold = np.percentile(points[:, 2], 10)
            animal_points = points[points[:, 2] > z_threshold]
            return animal_points, np.array([0, 0, 1, -z_threshold])

        # Use Open3D RANSAC for robust plane fitting
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        plane_model, inliers = pcd.segment_plane(
            distance_threshold=distance_threshold,
            ransac_n=3,
            num_iterations=1000
        )

        # Extract plane coefficients [a, b, c, d]
        a, b, c, d = plane_model

        # Remove ground points (inliers)
        animal_points = points[np.setdiff1d(np.arange(len(points)), inliers)]

        print(f" Ground plane removed: {len(inliers)} ground points, "
              f"{len(animal_points)} cow points remaining")

        return animal_points, np.array([a, b, c, d])

    # ========================================================================
    # VOLUME CALCULATION METHODS
    # ========================================================================

    def calculate_convex_hull_volume(self, points: np.ndarray) -> float:
        """
        Calculate convex hull volume.

        Args:
            points: Point cloud [N, 3]

        Returns:
            Volume in m�
        """
        if not SCIPY_AVAILABLE or len(points) < 4:
            return 0.0

        try:
            hull = ConvexHull(points)
            volume = hull.volume
            print(f"  Convex hull volume: {volume:.6f} m�")
            return float(volume)
        except Exception as e:
            print(f"Convex hull failed: {e}")
            return 0.0

    def calculate_voxel_volume(
        self,
        points: np.ndarray,
        voxel_size: Optional[float] = None
    ) -> float:
        """
        Calculate voxel-based volume.

        Args:
            points: Point cloud [N, 3]
            voxel_size: Voxel size in meters (default: COW_VOXEL_SIZE)

        Returns:
            Volume in m�
        """
        if voxel_size is None:
            voxel_size = self.COW_VOXEL_SIZE

        # Voxelize point cloud
        voxel_grid = set()
        for point in points:
            voxel_key = (
                int(point[0] / voxel_size),
                int(point[1] / voxel_size),
                int(point[2] / voxel_size)
            )
            voxel_grid.add(voxel_key)

        occupied_voxels = len(voxel_grid)
        voxel_volume_m3 = voxel_size ** 3
        total_volume = occupied_voxels * voxel_volume_m3

        print(f"  Voxel volume: {total_volume:.6f} m� ({occupied_voxels} voxels @ {voxel_size}m)")

        return float(total_volume)

    def calculate_mesh_volume(
        self,
        points: np.ndarray,
        normals: Optional[np.ndarray] = None
    ) -> Optional[float]:
        """
        Calculate mesh volume using Poisson reconstruction.

        Args:
            points: Point cloud [N, 3]
            normals: Normals [N, 3] (optional, will estimate if None)

        Returns:
            Volume in m� or None if reconstruction fails
        """
        if not OPEN3D_AVAILABLE or not TRIMESH_AVAILABLE:
            return None

        try:
            # Create Open3D point cloud
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)

            # Estimate normals if not provided
            if normals is None:
                pcd.estimate_normals(
                    search_param=o3d.geometry.KDTreeSearchParamHybrid(
                        radius=0.1, max_nn=30
                    )
                )
            else:
                pcd.normals = o3d.utility.Vector3dVector(normals)

            # Poisson surface reconstruction
            mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                pcd, depth=9
            )

            # Convert to trimesh for volume calculation
            vertices = np.asarray(mesh.vertices)
            triangles = np.asarray(mesh.triangles)

            mesh_obj = trimesh.Trimesh(vertices=vertices, faces=triangles)

            # Check if watertight
            if not mesh_obj.is_watertight:
                print("Mesh is not watertight, volume may be unreliable")

            volume = abs(mesh_obj.volume)
            print(f"  Mesh volume (Poisson): {volume:.6f} m� (watertight={mesh_obj.is_watertight})")

            return float(volume)

        except Exception as e:
            print(f"Mesh volume calculation failed: {e}")
            return None

    def calculate_alpha_shape_volume(
        self,
        points: np.ndarray,
        alpha: float = 0.03
    ) -> Optional[float]:
        """
        Calculate alpha shape volume.

        Args:
            points: Point cloud [N, 3]
            alpha: Alpha parameter (smaller = tighter fit)

        Returns:
            Volume in m� or None if fails
        """
        if not OPEN3D_AVAILABLE or not TRIMESH_AVAILABLE:
            return None

        try:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)

            # Create alpha shape
            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
                pcd, alpha
            )

            vertices = np.asarray(mesh.vertices)
            triangles = np.asarray(mesh.triangles)

            mesh_obj = trimesh.Trimesh(vertices=vertices, faces=triangles)
            volume = abs(mesh_obj.volume) if mesh_obj.is_watertight else None

            if volume:
                print(f"  Alpha shape volume (�={alpha}): {volume:.6f} m�")
            else:
                print(f"Alpha shape not watertight")

            return float(volume) if volume else None

        except Exception as e:
            print(f"Alpha shape calculation failed: {e}")
            return None

    def calculate_statistical_volume(
        self,
        points: np.ndarray,
        occupancy_ratio: float = 0.5
    ) -> float:
        """
        Calculate statistical volume from bounding box and occupancy ratio.

        Args:
            points: Point cloud [N, 3]
            occupancy_ratio: Estimated occupancy (0.3-0.7 typical)

        Returns:
            Volume in m�
        """
        # Calculate bounding box
        min_bound = points.min(axis=0)
        max_bound = points.max(axis=0)
        dimensions = max_bound - min_bound

        bbox_volume = dimensions[0] * dimensions[1] * dimensions[2]
        statistical_volume = bbox_volume * occupancy_ratio

        print(f"  Statistical volume: {statistical_volume:.6f} m� "
              f"(bbox={bbox_volume:.6f} m� � {occupancy_ratio})")

        return float(statistical_volume)

    def calculate_all_volumes(
        self,
        points: np.ndarray,
        normals: Optional[np.ndarray] = None
    ) -> VolumeMetrics:
        """
        Calculate volume using all methods and select best estimate.

        Args:
            points: Point cloud [N, 3]
            normals: Optional normals [N, 3]

        Returns:
            VolumeMetrics with all methods and best estimate
        """
        print("= Calculating volumes using all methods...")

        # Calculate using all methods
        convex_hull_vol = self.calculate_convex_hull_volume(points)
        voxel_vol = self.calculate_voxel_volume(points)
        mesh_vol = self.calculate_mesh_volume(points, normals)
        alpha_vol = self.calculate_alpha_shape_volume(points)
        statistical_vol = self.calculate_statistical_volume(points)

        # Select best estimate (priority: mesh > voxel > alpha > convex > statistical)
        if mesh_vol is not None and mesh_vol > 0:
            best_volume = mesh_vol
            method = 'mesh_poisson'
            confidence = 0.95
        elif voxel_vol > 0:
            best_volume = voxel_vol
            method = 'voxel_grid'
            confidence = 0.85
        elif alpha_vol is not None and alpha_vol > 0:
            best_volume = alpha_vol
            method = 'alpha_shape'
            confidence = 0.80
        elif convex_hull_vol > 0:
            best_volume = convex_hull_vol
            method = 'convex_hull'
            confidence = 0.70
        else:
            best_volume = statistical_vol
            method = 'statistical'
            confidence = 0.60

        metrics = VolumeMetrics(
            convex_hull_volume=convex_hull_vol,
            voxel_volume=voxel_vol,
            mesh_volume=mesh_vol,
            alpha_shape_volume=alpha_vol,
            statistical_volume=statistical_vol,
            best_estimate=best_volume,
            method_used=method,
            confidence=confidence
        )

        print(f" Best volume estimate: {best_volume:.6f} m� (method={method}, confidence={confidence})")

        return metrics

    # ========================================================================
    # DIMENSIONAL MEASUREMENTS
    # ========================================================================

    def measure_body_length(self, points: np.ndarray) -> float:
        """
        Measure cow body length (nose-to-tail).

        Args:
            points: Point cloud [N, 3]

        Returns:
            Length in meters
        """
        # Find extreme points along principal axis
        # Simplified: use min/max along x-axis
        length = float(points[:, 0].max() - points[:, 0].min())
        print(f"  Body length: {length:.3f} m")
        return length

    def measure_body_height(
        self,
        points: np.ndarray,
        ground_plane: Optional[np.ndarray] = None
    ) -> float:
        """
        Measure cow height from ground (withers/shoulder).

        Args:
            points: Point cloud [N, 3]
            ground_plane: Plane coefficients [a, b, c, d] or None

        Returns:
            Height in meters
        """
        if ground_plane is not None:
            # Calculate perpendicular distance from highest point to plane
            a, b, c, d = ground_plane
            highest_point = points[np.argmax(points[:, 2])]

            # Distance from point to plane: |ax + by + cz + d| / sqrt(a� + b� + c�)
            distance = abs(a * highest_point[0] + b * highest_point[1] +
                          c * highest_point[2] + d) / np.sqrt(a**2 + b**2 + c**2)
            height = float(distance)
        else:
            # Simple: max - min z-coordinate
            height = float(points[:, 2].max() - points[:, 2].min())

        print(f"  Body height: {height:.3f} m")
        return height

    def measure_body_width(self, points: np.ndarray) -> float:
        """
        Measure cow body width (chest/abdomen).

        Args:
            points: Point cloud [N, 3]

        Returns:
            Width in meters
        """
        # Simplified: max width along y-axis
        width = float(points[:, 1].max() - points[:, 1].min())
        print(f"  Body width: {width:.3f} m")
        return width

    def calculate_bounding_box_dimensions(
        self,
        points: np.ndarray
    ) -> BodyDimensions:
        """
        Calculate comprehensive body dimensions.

        Args:
            points: Point cloud [N, 3]

        Returns:
            BodyDimensions object
        """
        # Calculate bounding box
        min_bound = points.min(axis=0)
        max_bound = points.max(axis=0)
        dimensions = max_bound - min_bound

        bbox_length = float(dimensions[0])
        bbox_width = float(dimensions[1])
        bbox_height = float(dimensions[2])
        bbox_volume = bbox_length * bbox_width * bbox_height

        # Calculate body measurements
        length = self.measure_body_length(points)
        height = self.measure_body_height(points)
        width = self.measure_body_width(points)

        body_dims = BodyDimensions(
            length=length,
            height=height,
            width=width,
            bbox_length=bbox_length,
            bbox_width=bbox_width,
            bbox_height=bbox_height,
            bbox_volume=bbox_volume
        )

        print(f" Body dimensions: L={length:.2f}m W={width:.2f}m H={height:.2f}m")

        return body_dims

    # ========================================================================
    # WEIGHT ESTIMATION
    # ========================================================================

    def get_species_density(self) -> float:
        """
        Get cattle-specific density.

        Returns:
            Density in kg/m�
        """
        # Adjust density based on subtype
        if self.animal_subtype == 'calf':
            return 950.0  # Calves slightly less dense
        else:
            return self.COW_DENSITY_KG_M3

    def estimate_weight_from_volume(
        self,
        volume: float,
        method: Literal['density', 'linear_regression', 'polynomial_regression'] = 'density'
    ) -> WeightEstimate:
        """
        Estimate cow weight from volume.

        Args:
            volume: Volume in m�
            method: Estimation method

        Returns:
            WeightEstimate object
        """
        if method == 'density':
            # Simple density-based estimation
            density = self.get_species_density()
            weight_kg = volume * density
            uncertainty_kg = weight_kg * 0.10  # �10% typical uncertainty

            return WeightEstimate(
                estimated_weight_kg=weight_kg,
                volume_m3=volume,
                density_used=density,
                method='density',
                confidence='medium',
                r_squared=None,
                uncertainty_kg=uncertainty_kg
            )

        elif method == 'linear_regression' and self._weight_model_linear is not None:
            # Use calibrated linear model
            weight_kg = float(self._weight_model_linear.predict([[volume]])[0])
            uncertainty_kg = weight_kg * 0.05  # �5% with calibration

            # Determine confidence based on R�
            if self._calibration_r2 > 0.95:
                confidence = 'high'
            elif self._calibration_r2 > 0.85:
                confidence = 'medium'
            else:
                confidence = 'low'

            return WeightEstimate(
                estimated_weight_kg=weight_kg,
                volume_m3=volume,
                density_used=None,
                method='linear_regression',
                confidence=confidence,
                r_squared=self._calibration_r2,
                uncertainty_kg=uncertainty_kg
            )

        elif method == 'polynomial_regression' and self._weight_model_poly is not None:
            # Use calibrated polynomial model
            poly_features = PolynomialFeatures(degree=2)
            volume_poly = poly_features.fit_transform([[volume]])
            weight_kg = float(self._weight_model_poly.predict(volume_poly)[0])
            uncertainty_kg = weight_kg * 0.05

            if self._calibration_r2 > 0.95:
                confidence = 'high'
            elif self._calibration_r2 > 0.85:
                confidence = 'medium'
            else:
                confidence = 'low'

            return WeightEstimate(
                estimated_weight_kg=weight_kg,
                volume_m3=volume,
                density_used=None,
                method='polynomial_regression',
                confidence=confidence,
                r_squared=self._calibration_r2,
                uncertainty_kg=uncertainty_kg
            )

        else:
            # Fallback to density method
            return self.estimate_weight_from_volume(volume, 'density')

    def calibrate_weight_model(
        self,
        volumes: np.ndarray,
        actual_weights: np.ndarray
    ) -> Dict[str, float]:
        """
        Calibrate weight estimation model with ground truth.

        Args:
            volumes: Array of volumes [N] in m�
            actual_weights: Array of actual weights [N] in kg

        Returns:
            Dict with calibration metrics (R�, RMSE, MAE)
        """
        if not SKLEARN_AVAILABLE:
            print("scikit-learn not available for calibration")
            return {}

        # Reshape for sklearn
        X = volumes.reshape(-1, 1)
        y = actual_weights

        # Train linear model
        self._weight_model_linear = LinearRegression()
        self._weight_model_linear.fit(X, y)

        # Train polynomial model (degree 2)
        poly = PolynomialFeatures(degree=2)
        X_poly = poly.fit_transform(X)
        self._weight_model_poly = LinearRegression()
        self._weight_model_poly.fit(X_poly, y)

        # Calculate metrics for linear model
        y_pred_linear = self._weight_model_linear.predict(X)
        r2_linear = r2_score(y, y_pred_linear)
        rmse_linear = np.sqrt(mean_squared_error(y, y_pred_linear))
        mae_linear = mean_absolute_error(y, y_pred_linear)

        # Calculate metrics for polynomial model
        y_pred_poly = self._weight_model_poly.predict(X_poly)
        r2_poly = r2_score(y, y_pred_poly)
        rmse_poly = np.sqrt(mean_squared_error(y, y_pred_poly))
        mae_poly = mean_absolute_error(y, y_pred_poly)

        # Use best model
        if r2_poly > r2_linear:
            self._calibration_r2 = r2_poly
            best_model = 'polynomial'
        else:
            self._calibration_r2 = r2_linear
            best_model = 'linear'

        metrics = {
            'r2_linear': r2_linear,
            'rmse_linear': rmse_linear,
            'mae_linear': mae_linear,
            'r2_polynomial': r2_poly,
            'rmse_polynomial': rmse_poly,
            'mae_polynomial': mae_poly,
            'best_model': best_model
        }

        print(f" Weight model calibrated:")
        print(f"   Linear: R�={r2_linear:.3f}, RMSE={rmse_linear:.2f} kg, MAE={mae_linear:.2f} kg")
        print(f"   Polynomial: R�={r2_poly:.3f}, RMSE={rmse_poly:.2f} kg, MAE={mae_poly:.2f} kg")
        print(f"   Best model: {best_model}")

        return metrics

    # ========================================================================
    # TRACKING & ANALYSIS
    # ========================================================================

    def track_volume_over_time(
        self,
        animal_id: str,
        volume: float,
        timestamp: datetime
    ):
        """
        Track volume measurements over time.

        Args:
            animal_id: Animal identifier
            volume: Volume in m�
            timestamp: Measurement timestamp
        """
        if animal_id not in self.volume_history:
            self.volume_history[animal_id] = []

        self.volume_history[animal_id].append((timestamp, volume))
        print(f" Volume tracked for {animal_id}: {volume:.6f} m� @ {timestamp}")

    def calculate_growth_rate(
        self,
        animal_id: str
    ) -> Optional[float]:
        """
        Calculate growth rate in m�/day.

        Args:
            animal_id: Animal identifier

        Returns:
            Growth rate in m�/day or None if insufficient data
        """
        if animal_id not in self.volume_history or len(self.volume_history[animal_id]) < 2:
            return None

        # Get first and last measurements
        measurements = sorted(self.volume_history[animal_id], key=lambda x: x[0])
        first_time, first_volume = measurements[0]
        last_time, last_volume = measurements[-1]

        # Calculate growth rate
        time_delta_days = (last_time - first_time).total_seconds() / 86400
        volume_change = last_volume - first_volume
        growth_rate = volume_change / time_delta_days if time_delta_days > 0 else 0

        print(f"  Growth rate for {animal_id}: {growth_rate:.8f} m�/day")

        return float(growth_rate)

    def save_measurements_to_csv(
        self,
        report: MeasurementReport,
        csv_path: Path
    ):
        """
        Save measurements to CSV file.

        Args:
            report: Measurement report
            csv_path: Output CSV file path
        """
        csv_path = Path(csv_path)
        file_exists = csv_path.exists()

        with open(csv_path, 'a', newline='') as f:
            fieldnames = [
                'timestamp', 'animal_id', 'animal_type',
                'volume_m3', 'method', 'confidence',
                'weight_kg', 'length_m', 'height_m', 'width_m',
                'growth_rate_m3_per_day'
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)

            if not file_exists:
                writer.writeheader()

            writer.writerow({
                'timestamp': report.timestamp.isoformat(),
                'animal_id': report.animal_id,
                'animal_type': report.animal_type,
                'volume_m3': report.volume_metrics.best_estimate,
                'method': report.volume_metrics.method_used,
                'confidence': report.volume_metrics.confidence,
                'weight_kg': report.weight_estimate.estimated_weight_kg,
                'length_m': report.body_dimensions.length,
                'height_m': report.body_dimensions.height,
                'width_m': report.body_dimensions.width,
                'growth_rate_m3_per_day': report.growth_rate_m3_per_day or 0.0
            })

        print(f" Measurement saved to {csv_path}")

    def generate_measurement_report(
        self,
        animal_id: str,
        points: np.ndarray,
        normals: Optional[np.ndarray] = None,
        timestamp: Optional[datetime] = None
    ) -> MeasurementReport:
        """
        Generate comprehensive measurement report for cow.

        Args:
            animal_id: Animal identifier
            points: Point cloud [N, 3]
            normals: Optional normals [N, 3]
            timestamp: Measurement timestamp (default: now)

        Returns:
            MeasurementReport object
        """
        if timestamp is None:
            timestamp = datetime.now()

        print(f"\n=. Generating measurement report for {self.animal_subtype} {animal_id}...")

        # Calculate volumes
        volume_metrics = self.calculate_all_volumes(points, normals)

        # Calculate dimensions
        body_dimensions = self.calculate_bounding_box_dimensions(points)

        # Estimate weight
        weight_estimate = self.estimate_weight_from_volume(
            volume_metrics.best_estimate,
            method='linear_regression' if self._weight_model_linear else 'density'
        )

        # Track volume over time
        self.track_volume_over_time(animal_id, volume_metrics.best_estimate, timestamp)

        # Calculate growth rate
        growth_rate = self.calculate_growth_rate(animal_id)

        # Get previous measurement
        history = self.volume_history.get(animal_id, [])
        previous_volume = history[-2][1] if len(history) >= 2 else None
        volume_change = (volume_metrics.best_estimate - previous_volume) if previous_volume else None

        # Create report
        report = MeasurementReport(
            animal_id=animal_id,
            animal_type=self.animal_subtype,
            timestamp=timestamp,
            volume_metrics=volume_metrics,
            body_dimensions=body_dimensions,
            weight_estimate=weight_estimate,
            growth_rate_m3_per_day=growth_rate,
            previous_volume=previous_volume,
            volume_change=volume_change,
            herd_average_comparison=None  # Set externally if needed
        )

        # Add to measurements history
        self.measurements_history.append(report)

        print(f" Measurement report generated for {animal_id}")
        print(f"   Volume: {volume_metrics.best_estimate:.6f} m�")
        print(f"   Weight: {weight_estimate.estimated_weight_kg:.2f} kg")
        print(f"   Dimensions: {body_dimensions.length:.2f}m � {body_dimensions.width:.2f}m � {body_dimensions.height:.2f}m")

        return report

    # ========================================================================
    # MULTI-ANIMAL ANALYSIS
    # ========================================================================

    def compute_herd_statistics(
        self,
        volumes: List[float]
    ) -> Dict[str, float]:
        """
        Compute summary statistics for herd.

        Args:
            volumes: List of volumes in m�

        Returns:
            Dict with mean, median, std, min, max
        """
        volumes_array = np.array(volumes)

        stats = {
            'mean_volume': float(np.mean(volumes_array)),
            'median_volume': float(np.median(volumes_array)),
            'std_volume': float(np.std(volumes_array)),
            'min_volume': float(np.min(volumes_array)),
            'max_volume': float(np.max(volumes_array)),
            'count': len(volumes)
        }

        print(f"=� Herd statistics (n={stats['count']}):")
        print(f"   Mean: {stats['mean_volume']:.6f} m�")
        print(f"   Median: {stats['median_volume']:.6f} m�")
        print(f"   Std: {stats['std_volume']:.6f} m�")
        print(f"   Range: [{stats['min_volume']:.6f}, {stats['max_volume']:.6f}] m�")

        return stats

    def identify_growth_outliers(
        self,
        growth_rates: List[float],
        threshold: float = 2.5
    ) -> List[int]:
        """
        Identify animals with abnormal growth rates (outliers).

        Args:
            growth_rates: List of growth rates in m�/day
            threshold: Z-score threshold (default: 2.5 standard deviations)

        Returns:
            List of indices of outlier animals
        """
        rates_array = np.array(growth_rates)
        mean_rate = np.mean(rates_array)
        std_rate = np.std(rates_array)

        if std_rate == 0:
            return []

        z_scores = (rates_array - mean_rate) / std_rate
        outliers = np.where(np.abs(z_scores) > threshold)[0].tolist()

        print(f"Growth outliers detected: {len(outliers)}/{len(growth_rates)} animals")
        for idx in outliers:
            print(f"   Animal {idx}: rate={growth_rates[idx]:.8f} m�/day (z={z_scores[idx]:.2f})")

        return outliers
