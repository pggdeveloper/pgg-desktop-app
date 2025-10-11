"""
Abstract Base Class for Animal IMU Integration

This module provides an abstract interface for animal-specific IMU data processing
and sensor fusion. Enables future extensibility for different animal types and
agricultural environments (cows, pigs, sheep, etc.) while maintaining consistent API.

Author: PGG Desktop App Team
Date: 2025-10-10
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Literal
from dataclasses import dataclass
from datetime import datetime


@dataclass
class AccelerometerData:
    """Accelerometer measurement with 3-axis acceleration."""
    acceleration: np.ndarray  # (x, y, z) in m/s²
    timestamp: datetime
    frame_number: Optional[int] = None


@dataclass
class GyroscopeData:
    """Gyroscope measurement with 3-axis angular velocity."""
    angular_velocity: np.ndarray  # (x, y, z) in rad/s (roll, pitch, yaw rates)
    timestamp: datetime
    frame_number: Optional[int] = None


@dataclass
class SynchronizedIMUData:
    """Synchronized IMU data matched to a video frame."""
    frame_number: int
    frame_timestamp: datetime
    accelerometer: AccelerometerData
    gyroscope: GyroscopeData
    time_offset_ms: float  # Time difference between frame and IMU sample


@dataclass
class OrientationEstimate:
    """Device orientation estimate from IMU fusion."""
    roll: float  # Roll angle in radians
    pitch: float  # Pitch angle in radians
    yaw: float  # Yaw angle in radians
    rotation_matrix: np.ndarray  # 3x3 rotation matrix
    confidence: float  # 0-1
    timestamp: datetime


@dataclass
class GravityVector:
    """Estimated gravity vector from accelerometer."""
    direction: np.ndarray  # Unit vector (x, y, z)
    magnitude: float  # Should be ~9.81 m/s²
    down_vector: np.ndarray  # Unit vector pointing down
    timestamp: datetime


@dataclass
class VibrationAnalysis:
    """Vibration analysis in frequency domain."""
    frequencies: np.ndarray  # Frequency bins (Hz)
    power_spectrum: np.ndarray  # Power spectrum magnitude
    dominant_frequency: float  # Dominant vibration frequency (Hz)
    vibration_magnitude: float  # Overall vibration magnitude
    is_shaking: bool  # True if significant vibration detected
    timestamp: datetime


@dataclass
class MotionBlurEstimate:
    """Motion blur estimate correlated with IMU."""
    angular_velocity_magnitude: float  # rad/s
    estimated_blur_pixels: float  # Estimated blur in pixels
    blur_level: Literal['none', 'low', 'medium', 'high']
    should_skip_frame: bool  # True if blur is too high
    timestamp: datetime


@dataclass
class CameraPoseIMU:
    """Camera pose estimated from IMU."""
    rotation_matrix: np.ndarray  # 3x3 rotation from IMU
    pointing_direction: np.ndarray  # Unit vector (x, y, z) camera is pointing
    up_vector: np.ndarray  # Unit vector (x, y, z) camera's up direction
    orientation: OrientationEstimate  # Roll, pitch, yaw
    timestamp: datetime


@dataclass
class DepthCorrectionResult:
    """Motion-corrected depth map."""
    corrected_depth: np.ndarray  # Motion-corrected depth map
    original_depth: np.ndarray  # Original depth map
    motion_vector: np.ndarray  # Estimated camera motion (x, y, z)
    rotation_correction: np.ndarray  # 3x3 rotation correction matrix
    correction_magnitude: float  # Magnitude of correction applied
    timestamp: datetime


@dataclass
class IMUReport:
    """Comprehensive IMU integration report."""
    frame_number: int
    timestamp: datetime
    synchronized_data: SynchronizedIMUData
    orientation: OrientationEstimate
    gravity: GravityVector
    vibration: Optional[VibrationAnalysis]
    motion_blur: Optional[MotionBlurEstimate]
    camera_pose: CameraPoseIMU
    depth_correction: Optional[DepthCorrectionResult]


class AnimalIMUIntegration(ABC):
    """
    Abstract base class for animal IMU integration.

    Provides consistent interface for different animal types and agricultural
    environments while allowing species-specific implementations for sensor fusion,
    orientation estimation, and motion correction.
    """

    def __init__(
        self,
        animal_type: str,
        environment_type: Literal['feedlot_closed', 'feedlot_outdoor', 'pasture_natural', 'pasture_fodder'] = 'feedlot_outdoor',
        sampling_rate: float = 200.0  # IMU sampling rate in Hz
    ):
        """
        Initialize animal IMU integration.

        Args:
            animal_type: Type of animal ('cow', 'pig', 'sheep', etc.)
            environment_type: Type of agricultural environment
            sampling_rate: IMU sampling rate in Hz (typically 200-400 Hz)
        """
        self.animal_type = animal_type
        self.environment_type = environment_type
        self.sampling_rate = sampling_rate

        # IMU data buffers
        self.accelerometer_buffer: List[AccelerometerData] = []
        self.gyroscope_buffer: List[GyroscopeData] = []
        self.synchronized_buffer: List[SynchronizedIMUData] = []
        self.imu_reports: List[IMUReport] = []

        # Sensor fusion state
        self.current_orientation: Optional[OrientationEstimate] = None
        self.gravity_estimate: Optional[GravityVector] = None

    # ========================================================================
    # DATA CAPTURE (Abstract Methods)
    # ========================================================================

    @abstractmethod
    def capture_accelerometer_data(
        self,
        accel_x: float,
        accel_y: float,
        accel_z: float,
        timestamp: datetime,
        frame_number: Optional[int] = None
    ) -> AccelerometerData:
        """
        Capture accelerometer data.

        Args:
            accel_x: Acceleration in x-axis (m/s²)
            accel_y: Acceleration in y-axis (m/s²)
            accel_z: Acceleration in z-axis (m/s²)
            timestamp: Timestamp of measurement
            frame_number: Optional associated frame number

        Returns:
            AccelerometerData with acceleration vector
        """
        pass

    @abstractmethod
    def capture_gyroscope_data(
        self,
        gyro_x: float,
        gyro_y: float,
        gyro_z: float,
        timestamp: datetime,
        frame_number: Optional[int] = None
    ) -> GyroscopeData:
        """
        Capture gyroscope data.

        Args:
            gyro_x: Angular velocity around x-axis (rad/s) - roll rate
            gyro_y: Angular velocity around y-axis (rad/s) - pitch rate
            gyro_z: Angular velocity around z-axis (rad/s) - yaw rate
            timestamp: Timestamp of measurement
            frame_number: Optional associated frame number

        Returns:
            GyroscopeData with angular velocity vector
        """
        pass

    # ========================================================================
    # TIMESTAMP SYNCHRONIZATION (Abstract Methods)
    # ========================================================================

    @abstractmethod
    def synchronize_imu_with_frame(
        self,
        frame_number: int,
        frame_timestamp: datetime,
        max_time_offset_ms: float = 10.0
    ) -> Optional[SynchronizedIMUData]:
        """
        Synchronize IMU samples with video frame.

        Args:
            frame_number: Video frame number
            frame_timestamp: Timestamp of video frame
            max_time_offset_ms: Maximum allowed time offset in milliseconds

        Returns:
            SynchronizedIMUData with matched IMU samples or None if no match
        """
        pass

    @abstractmethod
    def interpolate_imu_data(
        self,
        target_timestamp: datetime,
        before_sample: Optional[Any],
        after_sample: Optional[Any]
    ) -> Optional[Any]:
        """
        Interpolate IMU data between two samples.

        Args:
            target_timestamp: Target timestamp for interpolation
            before_sample: IMU sample before target time
            after_sample: IMU sample after target time

        Returns:
            Interpolated IMU sample or None
        """
        pass

    # ========================================================================
    # SENSOR FUSION (Abstract Methods)
    # ========================================================================

    @abstractmethod
    def apply_complementary_filter(
        self,
        accel_data: AccelerometerData,
        gyro_data: GyroscopeData,
        alpha: float = 0.98,
        dt: Optional[float] = None
    ) -> OrientationEstimate:
        """
        Apply complementary filter for sensor fusion.

        Args:
            accel_data: Accelerometer data (provides gravity direction)
            gyro_data: Gyroscope data (provides rotation rate)
            alpha: Filter coefficient (0-1, typically 0.98)
            dt: Time step in seconds (auto-calculated if None)

        Returns:
            OrientationEstimate with fused orientation
        """
        pass

    @abstractmethod
    def estimate_gravity_vector(
        self,
        accel_data: AccelerometerData,
        window_size: int = 10
    ) -> GravityVector:
        """
        Estimate gravity vector from accelerometer.

        Args:
            accel_data: Accelerometer data
            window_size: Number of samples to average

        Returns:
            GravityVector with direction and magnitude
        """
        pass

    @abstractmethod
    def calculate_orientation_angles(
        self,
        rotation_matrix: np.ndarray
    ) -> Tuple[float, float, float]:
        """
        Calculate roll, pitch, yaw from rotation matrix.

        Args:
            rotation_matrix: 3x3 rotation matrix

        Returns:
            Tuple of (roll, pitch, yaw) in radians
        """
        pass

    # ========================================================================
    # VIBRATION ANALYSIS (Abstract Methods)
    # ========================================================================

    @abstractmethod
    def analyze_vibration_fft(
        self,
        accel_signal: np.ndarray,
        sampling_rate: Optional[float] = None
    ) -> VibrationAnalysis:
        """
        Analyze vibration in frequency domain using FFT.

        Args:
            accel_signal: Accelerometer time-series (1D array)
            sampling_rate: Sampling rate in Hz (uses self.sampling_rate if None)

        Returns:
            VibrationAnalysis with frequency spectrum
        """
        pass

    @abstractmethod
    def detect_camera_shake(
        self,
        vibration_analysis: VibrationAnalysis,
        shake_threshold: float = 2.0
    ) -> bool:
        """
        Detect camera shake from vibration analysis.

        Args:
            vibration_analysis: VibrationAnalysis result
            shake_threshold: Threshold for shake detection

        Returns:
            True if camera shake detected
        """
        pass

    # ========================================================================
    # MOTION BLUR CORRELATION (Abstract Methods)
    # ========================================================================

    @abstractmethod
    def estimate_motion_blur(
        self,
        gyro_data: GyroscopeData,
        exposure_time_ms: float = 33.0,
        focal_length_pixels: float = 500.0
    ) -> MotionBlurEstimate:
        """
        Estimate motion blur from IMU angular velocity.

        Args:
            gyro_data: Gyroscope data
            exposure_time_ms: Camera exposure time in milliseconds
            focal_length_pixels: Camera focal length in pixels

        Returns:
            MotionBlurEstimate with blur level
        """
        pass

    @abstractmethod
    def assess_blur_level(
        self,
        blur_pixels: float,
        low_threshold: float = 0.5,
        medium_threshold: float = 2.0,
        high_threshold: float = 5.0
    ) -> Literal['none', 'low', 'medium', 'high']:
        """
        Assess blur level from estimated blur pixels.

        Args:
            blur_pixels: Estimated blur in pixels
            low_threshold: Threshold for low blur
            medium_threshold: Threshold for medium blur
            high_threshold: Threshold for high blur

        Returns:
            Blur level classification
        """
        pass

    # ========================================================================
    # CAMERA POSE ESTIMATION (Abstract Methods)
    # ========================================================================

    @abstractmethod
    def estimate_camera_pose_from_imu(
        self,
        orientation: OrientationEstimate,
        gravity: GravityVector
    ) -> CameraPoseIMU:
        """
        Estimate camera pose from IMU orientation.

        Args:
            orientation: OrientationEstimate from sensor fusion
            gravity: GravityVector for reference frame

        Returns:
            CameraPoseIMU with rotation and pointing direction
        """
        pass

    @abstractmethod
    def calculate_pointing_direction(
        self,
        rotation_matrix: np.ndarray
    ) -> np.ndarray:
        """
        Calculate camera pointing direction from rotation matrix.

        Args:
            rotation_matrix: 3x3 rotation matrix

        Returns:
            Unit vector (x, y, z) of pointing direction
        """
        pass

    # ========================================================================
    # GRAVITY COMPENSATION (Abstract Methods)
    # ========================================================================

    @abstractmethod
    def apply_gravity_compensation(
        self,
        measurements: np.ndarray,
        gravity: GravityVector
    ) -> np.ndarray:
        """
        Apply gravity compensation to vertical measurements.

        Args:
            measurements: Array of measurements to correct
            gravity: GravityVector for compensation

        Returns:
            Gravity-compensated measurements
        """
        pass

    @abstractmethod
    def calculate_true_vertical(
        self,
        gravity: GravityVector
    ) -> np.ndarray:
        """
        Calculate true vertical direction from gravity.

        Args:
            gravity: GravityVector

        Returns:
            Unit vector (x, y, z) of true vertical
        """
        pass

    # ========================================================================
    # DEPTH CORRECTION (Abstract Methods)
    # ========================================================================

    @abstractmethod
    def generate_motion_corrected_depth(
        self,
        depth_map: np.ndarray,
        orientation_before: OrientationEstimate,
        orientation_after: OrientationEstimate,
        dt: float
    ) -> DepthCorrectionResult:
        """
        Generate motion-corrected depth map.

        Args:
            depth_map: Original depth map
            orientation_before: Orientation at previous frame
            orientation_after: Orientation at current frame
            dt: Time step in seconds

        Returns:
            DepthCorrectionResult with corrected depth
        """
        pass

    @abstractmethod
    def calculate_rotation_correction(
        self,
        orientation_before: OrientationEstimate,
        orientation_after: OrientationEstimate
    ) -> np.ndarray:
        """
        Calculate rotation correction matrix between orientations.

        Args:
            orientation_before: Previous orientation
            orientation_after: Current orientation

        Returns:
            3x3 rotation correction matrix
        """
        pass

    # ========================================================================
    # REPORT GENERATION (Abstract Methods)
    # ========================================================================

    @abstractmethod
    def generate_imu_report(
        self,
        frame_number: int,
        timestamp: datetime,
        synchronized_data: SynchronizedIMUData,
        depth_map: Optional[np.ndarray] = None,
        previous_orientation: Optional[OrientationEstimate] = None
    ) -> IMUReport:
        """
        Generate comprehensive IMU integration report.

        Args:
            frame_number: Current frame number
            timestamp: Current timestamp
            synchronized_data: Synchronized IMU data
            depth_map: Optional depth map for correction
            previous_orientation: Previous orientation for depth correction

        Returns:
            IMUReport with all IMU analysis results
        """
        pass

    @abstractmethod
    def save_imu_report_to_csv(
        self,
        report: IMUReport,
        csv_path: str
    ):
        """
        Save IMU report to CSV file.

        Args:
            report: IMUReport to save
            csv_path: Path to CSV file
        """
        pass
