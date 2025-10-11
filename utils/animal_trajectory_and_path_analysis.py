"""
Abstract Base Class for Animal Trajectory and Path Analysis

This module provides an abstract interface for animal-specific trajectory tracking
and path analysis implementations. Enables future extensibility for different animal
types (cows, pigs, sheep, etc.) while maintaining consistent API.

Author: PGG Desktop App Team
Date: 2025-10-10
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime


@dataclass
class TrajectoryPoint:
    """Single point in a trajectory."""
    position: np.ndarray  # 3D position (x, y, z) in meters
    timestamp: datetime
    frame_number: int
    velocity: Optional[np.ndarray] = None  # Optional velocity vector


@dataclass
class Trajectory:
    """Complete trajectory for an animal."""
    animal_id: str
    points: List[TrajectoryPoint]
    total_length: float  # Total path length in meters
    smoothed: bool = False  # Whether trajectory has been smoothed


@dataclass
class PathMetrics:
    """Path analysis metrics."""
    total_length_m: float  # Total path length in meters
    average_speed_m_s: float  # Average speed
    max_speed_m_s: float  # Maximum speed
    turning_angles: List[float]  # Turning angles in radians
    sharp_turns_count: int  # Number of sharp turns (> threshold)
    total_duration_s: float  # Total duration in seconds
    timestamp: datetime


@dataclass
class TrajectoryCluster:
    """Trajectory clustering result."""
    cluster_id: int
    trajectories: List[Trajectory]
    centroid: Optional[np.ndarray]  # Cluster centroid (representative path)
    size: int  # Number of trajectories in cluster


@dataclass
class CameraPose:
    """6DOF camera pose."""
    position: np.ndarray  # 3D position (x, y, z)
    rotation_matrix: np.ndarray  # 3x3 rotation matrix
    translation_vector: np.ndarray  # 3D translation vector
    euler_angles: np.ndarray  # (roll, pitch, yaw) in radians
    timestamp: datetime
    confidence: float  # Confidence score (0-1)


@dataclass
class TrajectoryReport:
    """Comprehensive trajectory analysis report."""
    animal_id: str
    animal_type: str
    trajectory: Trajectory
    path_metrics: PathMetrics
    predicted_positions: Optional[List[np.ndarray]]  # Future position predictions
    cluster_assignment: Optional[int]  # Cluster ID if clustering applied
    timestamp: datetime


class AnimalTrajectoryAnalyzer(ABC):
    """
    Abstract base class for animal trajectory and path analysis.

    Provides consistent interface for different animal types while allowing
    species-specific implementations for trajectory tracking, path smoothing,
    clustering, and prediction.
    """

    def __init__(self, animal_type: str, fps: int = 30):
        """
        Initialize animal trajectory analyzer.

        Args:
            animal_type: Type of animal ('cow', 'pig', 'sheep', etc.)
            fps: Frame rate for temporal calculations
        """
        self.animal_type = animal_type
        self.fps = fps
        self.frame_time = 1.0 / fps

        # Trajectory storage
        self.trajectories: Dict[str, Trajectory] = {}
        self.active_trajectories: Dict[str, List[TrajectoryPoint]] = {}

        # Clustering results
        self.trajectory_clusters: List[TrajectoryCluster] = []

        # Camera pose tracking
        self.camera_poses: List[CameraPose] = []

    # ========================================================================
    # TRAJECTORY RECORDING (Abstract Methods)
    # ========================================================================

    @abstractmethod
    def record_trajectory_point(
        self,
        animal_id: str,
        position: np.ndarray,
        timestamp: datetime,
        frame_number: int
    ):
        """
        Record a 3D trajectory point for an animal.

        Args:
            animal_id: Unique animal identifier
            position: 3D position (x, y, z) in meters
            timestamp: Measurement timestamp
            frame_number: Frame number
        """
        pass

    @abstractmethod
    def get_trajectory(self, animal_id: str) -> Optional[Trajectory]:
        """Get complete trajectory for an animal."""
        pass

    # ========================================================================
    # TRAJECTORY SMOOTHING (Abstract Methods)
    # ========================================================================

    @abstractmethod
    def smooth_trajectory_kalman(
        self,
        trajectory: Trajectory,
        process_noise: float = 0.01,
        measurement_noise: float = 0.1
    ) -> Trajectory:
        """
        Smooth trajectory using Kalman filter.

        Args:
            trajectory: Noisy trajectory
            process_noise: Process noise covariance
            measurement_noise: Measurement noise covariance

        Returns:
            Smoothed trajectory
        """
        pass

    @abstractmethod
    def smooth_trajectory_moving_average(
        self,
        trajectory: Trajectory,
        window_size: int = 5
    ) -> Trajectory:
        """
        Smooth trajectory using moving average filter.

        Args:
            trajectory: Noisy trajectory
            window_size: Number of consecutive points to average

        Returns:
            Smoothed trajectory
        """
        pass

    # ========================================================================
    # PATH ANALYSIS (Abstract Methods)
    # ========================================================================

    @abstractmethod
    def calculate_path_length(self, trajectory: Trajectory) -> float:
        """
        Calculate total path length.

        Args:
            trajectory: Trajectory to analyze

        Returns:
            Total path length in meters
        """
        pass

    @abstractmethod
    def analyze_turning_angles(
        self,
        trajectory: Trajectory,
        sharp_turn_threshold: float = 45.0
    ) -> Tuple[List[float], int]:
        """
        Analyze turning angles in trajectory.

        Args:
            trajectory: Trajectory to analyze
            sharp_turn_threshold: Threshold for sharp turns in degrees

        Returns:
            Tuple of (turning_angles, sharp_turns_count)
        """
        pass

    @abstractmethod
    def calculate_path_metrics(self, trajectory: Trajectory) -> PathMetrics:
        """
        Calculate comprehensive path metrics.

        Args:
            trajectory: Trajectory to analyze

        Returns:
            PathMetrics with all path statistics
        """
        pass

    # ========================================================================
    # TRAJECTORY CLUSTERING (Abstract Methods)
    # ========================================================================

    @abstractmethod
    def cluster_trajectories_dbscan(
        self,
        trajectories: List[Trajectory],
        eps: float = 0.5,
        min_samples: int = 2
    ) -> List[TrajectoryCluster]:
        """
        Cluster trajectories using DBSCAN.

        Args:
            trajectories: List of trajectories to cluster
            eps: Maximum distance between samples
            min_samples: Minimum samples in a neighborhood

        Returns:
            List of trajectory clusters
        """
        pass

    @abstractmethod
    def cluster_trajectories_kmeans(
        self,
        trajectories: List[Trajectory],
        n_clusters: int = 3
    ) -> List[TrajectoryCluster]:
        """
        Cluster trajectories using k-means.

        Args:
            trajectories: List of trajectories to cluster
            n_clusters: Number of clusters

        Returns:
            List of trajectory clusters
        """
        pass

    @abstractmethod
    def identify_common_patterns(
        self,
        clusters: List[TrajectoryCluster]
    ) -> Dict[int, Dict[str, Any]]:
        """
        Identify common movement patterns from clusters.

        Args:
            clusters: List of trajectory clusters

        Returns:
            Dictionary mapping cluster_id to pattern statistics
        """
        pass

    # ========================================================================
    # POSITION PREDICTION (Abstract Methods)
    # ========================================================================

    @abstractmethod
    def predict_position_linear(
        self,
        trajectory: Trajectory,
        prediction_time: float
    ) -> np.ndarray:
        """
        Predict future position using linear extrapolation.

        Args:
            trajectory: Historical trajectory
            prediction_time: Time into future in seconds

        Returns:
            Predicted 3D position
        """
        pass

    @abstractmethod
    def predict_position_lstm(
        self,
        trajectory: Trajectory,
        prediction_steps: int = 10
    ) -> List[np.ndarray]:
        """
        Predict future positions using LSTM/GRU.

        Args:
            trajectory: Historical trajectory
            prediction_steps: Number of future steps to predict

        Returns:
            List of predicted 3D positions
        """
        pass

    # ========================================================================
    # CAMERA POSE & ODOMETRY (Abstract Methods)
    # ========================================================================

    @abstractmethod
    def estimate_camera_pose(
        self,
        object_points: np.ndarray,
        image_points: np.ndarray,
        camera_matrix: np.ndarray,
        dist_coeffs: np.ndarray
    ) -> CameraPose:
        """
        Estimate 6DOF camera pose using PnP.

        Args:
            object_points: 3D points in world coordinates (N, 3)
            image_points: 2D points in image coordinates (N, 2)
            camera_matrix: Camera intrinsic matrix (3, 3)
            dist_coeffs: Distortion coefficients

        Returns:
            CameraPose with 6DOF pose
        """
        pass

    @abstractmethod
    def calculate_visual_odometry(
        self,
        prev_frame: np.ndarray,
        curr_frame: np.ndarray,
        camera_matrix: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate visual odometry between frames.

        Args:
            prev_frame: Previous frame
            curr_frame: Current frame
            camera_matrix: Camera intrinsic matrix

        Returns:
            Tuple of (rotation_matrix, translation_vector)
        """
        pass

    @abstractmethod
    def calculate_euler_angles(
        self,
        rotation_matrix: np.ndarray
    ) -> np.ndarray:
        """
        Calculate Euler angles from rotation matrix.

        Args:
            rotation_matrix: 3x3 rotation matrix

        Returns:
            Euler angles (roll, pitch, yaw) in radians
        """
        pass

    # ========================================================================
    # REPORT GENERATION (Abstract Methods)
    # ========================================================================

    @abstractmethod
    def generate_trajectory_report(
        self,
        animal_id: str,
        timestamp: Optional[datetime] = None
    ) -> Optional[TrajectoryReport]:
        """
        Generate comprehensive trajectory analysis report.

        Args:
            animal_id: Unique animal identifier
            timestamp: Optional timestamp

        Returns:
            TrajectoryReport or None if no trajectory exists
        """
        pass

    @abstractmethod
    def save_trajectory_to_csv(
        self,
        trajectory: Trajectory,
        csv_path: str
    ):
        """
        Save trajectory to CSV file.

        Args:
            trajectory: Trajectory to save
            csv_path: Path to CSV file
        """
        pass

    @abstractmethod
    def save_trajectory_report_to_csv(
        self,
        report: TrajectoryReport,
        csv_path: str
    ):
        """
        Save trajectory report to CSV file.

        Args:
            report: Trajectory report to save
            csv_path: Path to CSV file
        """
        pass
