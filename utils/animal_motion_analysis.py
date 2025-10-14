"""
Abstract Base Class for Animal Motion Analysis

This module provides an abstract interface for animal-specific motion analysis
implementations. Enables future extensibility for different animal types
(cows, pigs, sheep, etc.) while maintaining consistent API.

Author: PGG Desktop App Team
Date: 2025-10-10
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, List, Tuple, Optional, Literal
from dataclasses import dataclass
from datetime import datetime


@dataclass
class OpticalFlowResult:
    """Optical flow computation result."""
    flow: np.ndarray  # Flow field (H, W, 2) with (flow_x, flow_y)
    magnitude: np.ndarray  # Flow magnitude (H, W)
    direction: np.ndarray  # Flow direction in radians (H, W)
    method: Literal['farneback', 'lucas_kanade']
    timestamp: datetime


@dataclass
class SpeedMetrics:
    """Real-world speed and motion metrics."""
    speed_m_s: float  # Speed in meters per second
    acceleration_m_s2: float  # Acceleration in m/s²
    distance_traveled_m: float  # Total distance traveled in meters
    average_speed_m_s: float  # Average speed over period
    max_speed_m_s: float  # Maximum speed recorded
    timestamp: datetime


@dataclass
class ActivityMetrics:
    """Activity level and motion detection metrics."""
    activity_level: Literal['active', 'resting', 'moderate']
    total_motion: float  # Sum of motion magnitude
    motion_percentage: float  # Percentage of frame with motion
    foreground_area: int  # Pixels classified as foreground
    background_area: int  # Pixels classified as background
    timestamp: datetime


@dataclass
class GaitMetrics:
    """Gait and periodic motion analysis metrics."""
    step_frequency_hz: float  # Steps per second
    stride_length_m: float  # Stride length in meters
    gait_cycle_duration_s: float  # Gait cycle duration in seconds
    dominant_frequency_hz: float  # Dominant frequency from FFT
    periodicity_score: float  # Autocorrelation peak (0-1)
    symmetry_score: float  # Gait symmetry (0-1, 1=perfect)
    stance_phase_ratio: float  # Ratio of stance to total cycle
    timestamp: datetime


@dataclass
class MotionReport:
    """Comprehensive motion analysis report."""
    animal_id: str
    animal_type: str
    timestamp: datetime
    optical_flow: Optional[OpticalFlowResult]
    speed_metrics: Optional[SpeedMetrics]
    activity_metrics: Optional[ActivityMetrics]
    gait_metrics: Optional[GaitMetrics]
    frame_number: int


class AnimalMotionAnalyzer(ABC):
    """
    Abstract base class for animal motion analysis.

    Provides consistent interface for different animal types while allowing
    species-specific implementations for motion detection, activity tracking,
    and gait analysis.
    """

    def __init__(self, animal_type: str, fps: int = 30):
        """
        Initialize animal motion analyzer.

        Args:
            animal_type: Type of animal ('cow', 'pig', 'sheep', etc.)
            fps: Frame rate for temporal calculations
        """
        self.animal_type = animal_type
        self.fps = fps
        self.frame_time = 1.0 / fps  # Time between frames in seconds

        # Motion history
        self.motion_history: List[MotionReport] = []
        self.previous_frame: Optional[np.ndarray] = None
        self.previous_flow: Optional[np.ndarray] = None
        self.previous_positions: List[np.ndarray] = []  # For trajectory tracking

        # Background subtractors (initialized by subclass)
        self.bg_subtractor_mog2 = None
        self.bg_subtractor_knn = None

        # Motion history image
        self.motion_history_image: Optional[np.ndarray] = None
        self.motion_history_duration = 5.0  # seconds

    # ========================================================================
    # OPTICAL FLOW (Abstract Methods)
    # ========================================================================

    @abstractmethod
    def calculate_dense_optical_flow(
        self,
        frame1: np.ndarray,
        frame2: np.ndarray
    ) -> OpticalFlowResult:
        """
        Calculate dense optical flow using Farneback algorithm.

        Args:
            frame1: Previous frame (grayscale)
            frame2: Current frame (grayscale)

        Returns:
            OpticalFlowResult with flow field, magnitude, and direction
        """
        pass

    @abstractmethod
    def calculate_sparse_optical_flow(
        self,
        frame1: np.ndarray,
        frame2: np.ndarray,
        feature_points: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate sparse optical flow using Lucas-Kanade.

        Args:
            frame1: Previous frame (grayscale)
            frame2: Current frame (grayscale)
            feature_points: Feature points to track (N, 2)

        Returns:
            Tuple of (new_points, status, error)
        """
        pass

    @abstractmethod
    def generate_motion_magnitude_map(self, flow: np.ndarray) -> np.ndarray:
        """Generate motion magnitude map from optical flow."""
        pass

    @abstractmethod
    def generate_motion_direction_map(self, flow: np.ndarray) -> np.ndarray:
        """Generate motion direction map from optical flow."""
        pass

    @abstractmethod
    def visualize_flow_vectors(
        self,
        frame: np.ndarray,
        flow: np.ndarray,
        step: int = 16
    ) -> np.ndarray:
        """
        Visualize optical flow as arrows on frame.

        Args:
            frame: Base frame for visualization
            flow: Optical flow field
            step: Sampling step for arrows

        Returns:
            Frame with flow arrows drawn
        """
        pass

    # ========================================================================
    # BACKGROUND SUBTRACTION (Abstract Methods)
    # ========================================================================

    @abstractmethod
    def apply_background_subtraction_mog2(
        self,
        frame: np.ndarray
    ) -> np.ndarray:
        """Apply MOG2 background subtraction."""
        pass

    @abstractmethod
    def apply_background_subtraction_knn(
        self,
        frame: np.ndarray
    ) -> np.ndarray:
        """Apply KNN background subtraction."""
        pass

    @abstractmethod
    def generate_motion_history_image(
        self,
        foreground_mask: np.ndarray,
        timestamp: float
    ) -> np.ndarray:
        """
        Generate motion history image.

        Args:
            foreground_mask: Binary foreground mask
            timestamp: Current timestamp

        Returns:
            Motion history image (recent motion brighter)
        """
        pass

    @abstractmethod
    def apply_frame_differencing(
        self,
        frame1: np.ndarray,
        frame2: np.ndarray,
        threshold: int = 25
    ) -> np.ndarray:
        """
        Apply frame differencing to detect motion.

        Args:
            frame1: Previous frame
            frame2: Current frame
            threshold: Difference threshold

        Returns:
            Binary motion mask
        """
        pass

    # ========================================================================
    # ACTIVITY DETECTION (Abstract Methods)
    # ========================================================================

    @abstractmethod
    def detect_activity_level(
        self,
        motion_magnitude: np.ndarray,
        threshold_active: float = 2.0,
        threshold_resting: float = 0.5
    ) -> ActivityMetrics:
        """
        Detect activity level from motion magnitude.

        Args:
            motion_magnitude: Motion magnitude map
            threshold_active: Threshold for active state
            threshold_resting: Threshold for resting state

        Returns:
            ActivityMetrics with activity classification
        """
        pass

    # ========================================================================
    # SPEED & ACCELERATION (Abstract Methods)
    # ========================================================================

    @abstractmethod
    def calculate_real_world_speed(
        self,
        flow: np.ndarray,
        depth_frame: np.ndarray,
        camera_intrinsics: Dict[str, float]
    ) -> float:
        """
        Calculate real-world speed using depth and camera intrinsics.

        Args:
            flow: Optical flow field
            depth_frame: Depth frame in meters
            camera_intrinsics: Camera parameters (fx, fy, cx, cy)

        Returns:
            Speed in meters per second
        """
        pass

    @abstractmethod
    def detect_acceleration(
        self,
        current_speed: float,
        previous_speed: float
    ) -> float:
        """
        Detect acceleration from speed changes.

        Args:
            current_speed: Current speed in m/s
            previous_speed: Previous speed in m/s

        Returns:
            Acceleration in m/s²
        """
        pass

    @abstractmethod
    def track_movement_speed_over_time(
        self,
        animal_id: str,
        speed: float,
        timestamp: datetime
    ):
        """Track movement speed over time for an animal."""
        pass

    @abstractmethod
    def calculate_distance_traveled(
        self,
        trajectory: List[np.ndarray]
    ) -> float:
        """
        Calculate total distance traveled from trajectory.

        Args:
            trajectory: List of 3D positions (x, y, z) in meters

        Returns:
            Total distance in meters
        """
        pass

    # ========================================================================
    # PERIODIC MOTION ANALYSIS (Abstract Methods)
    # ========================================================================

    @abstractmethod
    def detect_periodic_motion_fft(
        self,
        motion_signal: np.ndarray,
        sampling_rate: float
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Detect periodic motion using FFT.

        Args:
            motion_signal: 1D motion signal over time
            sampling_rate: Sampling rate in Hz

        Returns:
            Tuple of (dominant_frequency, frequencies, power_spectrum)
        """
        pass

    @abstractmethod
    def detect_periodic_motion_autocorrelation(
        self,
        motion_signal: np.ndarray
    ) -> Tuple[float, float]:
        """
        Detect periodic motion using autocorrelation.

        Args:
            motion_signal: 1D motion signal over time

        Returns:
            Tuple of (period_duration, periodicity_score)
        """
        pass

    @abstractmethod
    def analyze_step_frequency(
        self,
        dominant_frequency: float
    ) -> float:
        """
        Analyze step frequency from dominant frequency.

        Args:
            dominant_frequency: Dominant frequency in Hz

        Returns:
            Steps per second
        """
        pass

    @abstractmethod
    def measure_stride_length(
        self,
        speed: float,
        step_frequency: float
    ) -> float:
        """
        Measure stride length from speed and step frequency.

        Args:
            speed: Speed in m/s
            step_frequency: Steps per second

        Returns:
            Stride length in meters
        """
        pass

    @abstractmethod
    def analyze_gait_cycle(
        self,
        motion_signal: np.ndarray,
        sampling_rate: float
    ) -> GaitMetrics:
        """
        Analyze gait cycle for periodic motion patterns.

        Args:
            motion_signal: 1D motion signal over time
            sampling_rate: Sampling rate in Hz

        Returns:
            GaitMetrics with gait analysis results
        """
        pass

    # ========================================================================
    # REPORT GENERATION (Abstract Methods)
    # ========================================================================

    @abstractmethod
    def generate_motion_report(
        self,
        animal_id: str,
        frame: np.ndarray,
        depth_frame: Optional[np.ndarray],
        camera_intrinsics: Optional[Dict[str, float]],
        frame_number: int,
        timestamp: Optional[datetime] = None
    ) -> MotionReport:
        """
        Generate comprehensive motion analysis report.

        Args:
            animal_id: Unique animal identifier
            frame: Current RGB frame
            depth_frame: Optional depth frame for real-world measurements
            camera_intrinsics: Optional camera parameters
            frame_number: Current frame number
            timestamp: Optional timestamp

        Returns:
            MotionReport with all motion metrics
        """
        pass

    @abstractmethod
    def save_motion_data_to_csv(
        self,
        report: MotionReport,
        csv_path: str
    ):
        """Save motion report to CSV file."""
        pass
