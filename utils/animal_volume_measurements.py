"""
Abstract Base Class for Animal Volume and Dimensional Measurements

This module provides an abstract interface for animal-specific volume and weight
estimation implementations. Enables future extensibility for different animal types
(cows, pigs, sheep, etc.) while maintaining consistent API.

Author: PGG Desktop App Team
Date: 2025-10-10
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, List, Tuple, Optional, Literal
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime


@dataclass
class VolumeMetrics:
    """Volume measurement metrics."""
    convex_hull_volume: float
    voxel_volume: float
    mesh_volume: Optional[float]
    alpha_shape_volume: Optional[float]
    statistical_volume: float
    best_estimate: float
    method_used: str
    confidence: float


@dataclass
class BodyDimensions:
    """Animal body dimension measurements."""
    length: float  # nose-to-tail in meters
    height: float  # withers/shoulder height in meters
    width: float  # chest/abdomen width in meters
    bbox_length: float
    bbox_width: float
    bbox_height: float
    bbox_volume: float


@dataclass
class WeightEstimate:
    """Weight estimation results."""
    estimated_weight_kg: float
    volume_m3: float
    density_used: float
    method: Literal['density', 'linear_regression', 'polynomial_regression']
    confidence: str  # 'high', 'medium', 'low'
    r_squared: Optional[float]
    uncertainty_kg: float


@dataclass
class MeasurementReport:
    """Comprehensive measurement report."""
    animal_id: str
    animal_type: str  # 'cow', 'bull', 'calf', 'heifer', 'steer', etc.
    timestamp: datetime
    volume_metrics: VolumeMetrics
    body_dimensions: BodyDimensions
    weight_estimate: WeightEstimate
    growth_rate_m3_per_day: Optional[float]
    previous_volume: Optional[float]
    volume_change: Optional[float]
    herd_average_comparison: Optional[float]


class AnimalVolumeEstimator(ABC):
    """
    Abstract base class for animal volume and weight estimation.

    Provides consistent interface for different animal types while allowing
    species-specific implementations.
    """

    def __init__(self, animal_type: str):
        """
        Initialize animal volume estimator.

        Args:
            animal_type: Type of animal ('cow', 'pig', 'sheep', etc.)
        """
        self.animal_type = animal_type
        self.measurements_history: List[MeasurementReport] = []

    # ========================================================================
    # POINT CLOUD EXTRACTION (Abstract Methods)
    # ========================================================================

    @abstractmethod
    def extract_animal_point_cloud(
        self,
        depth_frame: np.ndarray,
        bounding_box: Tuple[int, int, int, int]
    ) -> np.ndarray:
        """
        Extract animal point cloud from bounding box.

        Args:
            depth_frame: Depth frame data
            bounding_box: (x_min, x_max, y_min, y_max)

        Returns:
            Point cloud [N, 3]
        """
        pass

    @abstractmethod
    def remove_ground_plane(
        self,
        points: np.ndarray,
        distance_threshold: float = 0.02
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Remove ground plane from point cloud using RANSAC.

        Args:
            points: Point cloud [N, 3]
            distance_threshold: RANSAC distance threshold

        Returns:
            Tuple of (animal_points, plane_coefficients [a, b, c, d])
        """
        pass

    # ========================================================================
    # VOLUME CALCULATION (Abstract Methods)
    # ========================================================================

    @abstractmethod
    def calculate_convex_hull_volume(self, points: np.ndarray) -> float:
        """Calculate volume using convex hull."""
        pass

    @abstractmethod
    def calculate_voxel_volume(
        self,
        points: np.ndarray,
        voxel_size: float
    ) -> float:
        """Calculate volume using voxel grid."""
        pass

    @abstractmethod
    def calculate_mesh_volume(
        self,
        points: np.ndarray,
        normals: Optional[np.ndarray] = None
    ) -> Optional[float]:
        """Calculate volume using watertight mesh."""
        pass

    @abstractmethod
    def calculate_alpha_shape_volume(
        self,
        points: np.ndarray,
        alpha: float
    ) -> Optional[float]:
        """Calculate volume using alpha shape."""
        pass

    @abstractmethod
    def calculate_statistical_volume(
        self,
        points: np.ndarray,
        occupancy_ratio: float = 0.5
    ) -> float:
        """Calculate statistical volume from bounding box."""
        pass

    @abstractmethod
    def calculate_all_volumes(
        self,
        points: np.ndarray,
        normals: Optional[np.ndarray] = None
    ) -> VolumeMetrics:
        """Calculate volume using all methods and select best estimate."""
        pass

    # ========================================================================
    # DIMENSIONAL MEASUREMENTS (Abstract Methods)
    # ========================================================================

    @abstractmethod
    def measure_body_length(self, points: np.ndarray) -> float:
        """Measure body length (nose-to-tail)."""
        pass

    @abstractmethod
    def measure_body_height(
        self,
        points: np.ndarray,
        ground_plane: Optional[np.ndarray] = None
    ) -> float:
        """Measure body height from ground."""
        pass

    @abstractmethod
    def measure_body_width(self, points: np.ndarray) -> float:
        """Measure body width (chest/abdomen)."""
        pass

    @abstractmethod
    def calculate_bounding_box_dimensions(
        self,
        points: np.ndarray
    ) -> BodyDimensions:
        """Calculate comprehensive body dimensions."""
        pass

    # ========================================================================
    # WEIGHT ESTIMATION (Abstract Methods)
    # ========================================================================

    @abstractmethod
    def get_species_density(self) -> float:
        """Get species-specific density in kg/m³."""
        pass

    @abstractmethod
    def estimate_weight_from_volume(
        self,
        volume: float,
        method: Literal['density', 'linear_regression', 'polynomial_regression'] = 'density'
    ) -> WeightEstimate:
        """Estimate weight from volume."""
        pass

    @abstractmethod
    def calibrate_weight_model(
        self,
        volumes: np.ndarray,
        actual_weights: np.ndarray
    ) -> Dict[str, float]:
        """Calibrate weight estimation model with ground truth."""
        pass

    # ========================================================================
    # TRACKING & ANALYSIS (Abstract Methods)
    # ========================================================================

    @abstractmethod
    def track_volume_over_time(
        self,
        animal_id: str,
        volume: float,
        timestamp: datetime
    ):
        """Track volume measurements over time."""
        pass

    @abstractmethod
    def calculate_growth_rate(
        self,
        animal_id: str
    ) -> Optional[float]:
        """Calculate growth rate in m³/day."""
        pass

    @abstractmethod
    def save_measurements_to_csv(
        self,
        report: MeasurementReport,
        csv_path: Path
    ):
        """Save measurements to CSV file."""
        pass

    @abstractmethod
    def generate_measurement_report(
        self,
        animal_id: str,
        points: np.ndarray,
        normals: Optional[np.ndarray] = None,
        timestamp: Optional[datetime] = None
    ) -> MeasurementReport:
        """Generate comprehensive measurement report."""
        pass

    # ========================================================================
    # MULTI-ANIMAL ANALYSIS (Abstract Methods)
    # ========================================================================

    @abstractmethod
    def compute_herd_statistics(
        self,
        volumes: List[float]
    ) -> Dict[str, float]:
        """Compute summary statistics for herd."""
        pass

    @abstractmethod
    def identify_growth_outliers(
        self,
        growth_rates: List[float],
        threshold: float = 2.5
    ) -> List[int]:
        """Identify animals with abnormal growth rates."""
        pass
