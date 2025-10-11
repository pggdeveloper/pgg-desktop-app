"""
Cow-Specific IMU Integration Implementation

This module implements IMU data processing and sensor fusion specifically for
cattle monitoring in agricultural environments (feedlots and pastures).

Supports: cow, bull, calf, heifer, steer
Environments: feedlot_closed, feedlot_outdoor, pasture_natural, pasture_fodder

Author: PGG Desktop App Team
Date: 2025-10-10
"""

import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional, Any, Literal
from datetime import datetime, timedelta
import csv
from collections import deque

from utils.animal_imu_integration import (
    AnimalIMUIntegration,
    AccelerometerData,
    GyroscopeData,
    SynchronizedIMUData,
    OrientationEstimate,
    GravityVector,
    VibrationAnalysis,
    MotionBlurEstimate,
    CameraPoseIMU,
    DepthCorrectionResult,
    IMUReport
)


class CowIMUIntegration(AnimalIMUIntegration):
    """
    Cow-specific IMU integration implementation.

    Handles IMU data processing for cattle monitoring with:
    - Sensor fusion for orientation estimation
    - Vibration and motion blur analysis
    - Gravity compensation for height measurements
    - Motion-corrected depth maps

    Optimized for typical cattle movement patterns and agricultural
    environments with varying terrain and lighting conditions.
    """

    # Cow-specific constants
    COW_GRAVITY_MAGNITUDE = 9.81  # m/s²
    COW_TYPICAL_SHAKE_THRESHOLD = 2.0  # m/s² vibration threshold
    COW_MIN_BUFFER_SIZE = 10  # Minimum samples for gravity estimation
    COW_MAX_BUFFER_SIZE = 100  # Maximum buffer size for IMU data
    COW_COMPLEMENTARY_FILTER_ALPHA = 0.98  # Alpha for complementary filter
    COW_IMU_TIMEOUT_MS = 50.0  # Maximum time offset for synchronization

    # Motion blur thresholds (for cattle movement speeds)
    COW_BLUR_LOW_THRESHOLD = 0.5  # pixels
    COW_BLUR_MEDIUM_THRESHOLD = 2.0  # pixels
    COW_BLUR_HIGH_THRESHOLD = 5.0  # pixels

    def __init__(
        self,
        animal_type: Literal['cow', 'bull', 'calf', 'heifer', 'steer'] = 'cow',
        environment_type: Literal['feedlot_closed', 'feedlot_outdoor',
                                   'pasture_natural', 'pasture_fodder'] = 'feedlot_outdoor',
        sampling_rate: float = 200.0
    ):
        """
        Initialize cow IMU integration.

        Args:
            animal_type: Type of cattle
            environment_type: Agricultural environment type
            sampling_rate: IMU sampling rate in Hz (default 200 Hz for D455i)
        """
        super().__init__(animal_type, environment_type, sampling_rate)

        # Initialize orientation state (identity rotation initially)
        self.current_orientation = OrientationEstimate(
            roll=0.0,
            pitch=0.0,
            yaw=0.0,
            rotation_matrix=np.eye(3),
            confidence=0.0,
            timestamp=datetime.now()
        )

        # Initialize gravity estimate (assume down is -Z initially)
        self.gravity_estimate = GravityVector(
            direction=np.array([0.0, 0.0, -1.0]),
            magnitude=self.COW_GRAVITY_MAGNITUDE,
            down_vector=np.array([0.0, 0.0, -1.0]),
            timestamp=datetime.now()
        )

        # Time tracking for sensor fusion
        self.last_update_time: Optional[datetime] = None

    # ========================================================================
    # DATA CAPTURE IMPLEMENTATION (Scenario 1 & 2)
    # ========================================================================

    def capture_accelerometer_data(
        self,
        accel_x: float,
        accel_y: float,
        accel_z: float,
        timestamp: datetime,
        frame_number: Optional[int] = None
    ) -> AccelerometerData:
        """
        Capture accelerometer data (Scenario 1).

        Stores acceleration in m/s² for x, y, z axes with synchronized timestamp.
        """
        accel_data = AccelerometerData(
            acceleration=np.array([accel_x, accel_y, accel_z]),
            timestamp=timestamp,
            frame_number=frame_number
        )

        # Add to buffer with size limit
        self.accelerometer_buffer.append(accel_data)
        if len(self.accelerometer_buffer) > self.COW_MAX_BUFFER_SIZE:
            self.accelerometer_buffer.pop(0)

        return accel_data

    def capture_gyroscope_data(
        self,
        gyro_x: float,
        gyro_y: float,
        gyro_z: float,
        timestamp: datetime,
        frame_number: Optional[int] = None
    ) -> GyroscopeData:
        """
        Capture gyroscope data (Scenario 2).

        Stores angular velocity in rad/s for x, y, z axes (roll, pitch, yaw rates).
        """
        gyro_data = GyroscopeData(
            angular_velocity=np.array([gyro_x, gyro_y, gyro_z]),
            timestamp=timestamp,
            frame_number=frame_number
        )

        # Add to buffer with size limit
        self.gyroscope_buffer.append(gyro_data)
        if len(self.gyroscope_buffer) > self.COW_MAX_BUFFER_SIZE:
            self.gyroscope_buffer.pop(0)

        return gyro_data

    # ========================================================================
    # TIMESTAMP SYNCHRONIZATION (Scenario 3)
    # ========================================================================

    def synchronize_imu_with_frame(
        self,
        frame_number: int,
        frame_timestamp: datetime,
        max_time_offset_ms: float = None
    ) -> Optional[SynchronizedIMUData]:
        """
        Synchronize IMU samples with video frame (Scenario 3).

        Finds the closest IMU samples to the frame timestamp and
        performs temporal interpolation if needed.
        """
        if max_time_offset_ms is None:
            max_time_offset_ms = self.COW_IMU_TIMEOUT_MS

        # Check if we have enough data
        if not self.accelerometer_buffer or not self.gyroscope_buffer:
            return None

        # Find closest accelerometer sample
        closest_accel = None
        min_accel_offset = float('inf')

        for accel in self.accelerometer_buffer:
            offset = abs((accel.timestamp - frame_timestamp).total_seconds() * 1000)
            if offset < min_accel_offset:
                min_accel_offset = offset
                closest_accel = accel

        # Find closest gyroscope sample
        closest_gyro = None
        min_gyro_offset = float('inf')

        for gyro in self.gyroscope_buffer:
            offset = abs((gyro.timestamp - frame_timestamp).total_seconds() * 1000)
            if offset < min_gyro_offset:
                min_gyro_offset = offset
                closest_gyro = gyro

        # Check if samples are within tolerance
        if min_accel_offset > max_time_offset_ms or min_gyro_offset > max_time_offset_ms:
            return None

        if closest_accel is None or closest_gyro is None:
            return None

        # Create synchronized data
        avg_offset = (min_accel_offset + min_gyro_offset) / 2.0

        synchronized = SynchronizedIMUData(
            frame_number=frame_number,
            frame_timestamp=frame_timestamp,
            accelerometer=closest_accel,
            gyroscope=closest_gyro,
            time_offset_ms=avg_offset
        )

        # Add to synchronized buffer
        self.synchronized_buffer.append(synchronized)
        if len(self.synchronized_buffer) > self.COW_MAX_BUFFER_SIZE:
            self.synchronized_buffer.pop(0)

        return synchronized

    def interpolate_imu_data(
        self,
        target_timestamp: datetime,
        before_sample: Optional[Any],
        after_sample: Optional[Any]
    ) -> Optional[Any]:
        """
        Interpolate IMU data between two samples (Scenario 3).

        Performs linear interpolation for better temporal alignment.
        """
        if before_sample is None or after_sample is None:
            return None

        # Calculate interpolation factor
        t_before = before_sample.timestamp
        t_after = after_sample.timestamp
        t_target = target_timestamp

        if t_before >= t_after:
            return before_sample

        # Linear interpolation factor (0 to 1)
        alpha = (t_target - t_before).total_seconds() / (t_after - t_before).total_seconds()
        alpha = np.clip(alpha, 0.0, 1.0)

        # Interpolate based on sample type
        if isinstance(before_sample, AccelerometerData):
            interpolated_accel = (1 - alpha) * before_sample.acceleration + alpha * after_sample.acceleration
            return AccelerometerData(
                acceleration=interpolated_accel,
                timestamp=target_timestamp,
                frame_number=None
            )
        elif isinstance(before_sample, GyroscopeData):
            interpolated_gyro = (1 - alpha) * before_sample.angular_velocity + alpha * after_sample.angular_velocity
            return GyroscopeData(
                angular_velocity=interpolated_gyro,
                timestamp=target_timestamp,
                frame_number=None
            )

        return None

    # ========================================================================
    # SENSOR FUSION (Scenarios 4, 5, 6)
    # ========================================================================

    def apply_complementary_filter(
        self,
        accel_data: AccelerometerData,
        gyro_data: GyroscopeData,
        alpha: float = None,
        dt: Optional[float] = None
    ) -> OrientationEstimate:
        """
        Apply complementary filter for sensor fusion (Scenario 4).

        Combines accelerometer (gravity direction) and gyroscope (rotation rate)
        to estimate orientation with reduced drift.

        Filter: orientation = alpha * (gyro_integration) + (1-alpha) * (accel_estimate)
        """
        if alpha is None:
            alpha = self.COW_COMPLEMENTARY_FILTER_ALPHA

        # Calculate dt if not provided
        if dt is None:
            if self.last_update_time is not None:
                dt = (accel_data.timestamp - self.last_update_time).total_seconds()
            else:
                dt = 1.0 / self.sampling_rate

        # Clamp dt to reasonable range
        dt = np.clip(dt, 0.001, 0.1)

        # Integrate gyroscope for rotation update
        gyro_angles = gyro_data.angular_velocity * dt

        # Previous orientation
        prev_roll = self.current_orientation.roll
        prev_pitch = self.current_orientation.pitch
        prev_yaw = self.current_orientation.yaw

        # Gyroscope-based prediction (integrate angular velocity)
        gyro_roll = prev_roll + gyro_angles[0]
        gyro_pitch = prev_pitch + gyro_angles[1]
        gyro_yaw = prev_yaw + gyro_angles[2]

        # Accelerometer-based estimate (from gravity direction)
        accel_roll, accel_pitch = self._estimate_roll_pitch_from_accel(accel_data.acceleration)

        # Complementary filter fusion
        fused_roll = alpha * gyro_roll + (1 - alpha) * accel_roll
        fused_pitch = alpha * gyro_pitch + (1 - alpha) * accel_pitch
        fused_yaw = gyro_yaw  # Yaw from gyro only (no magnetometer)

        # Create rotation matrix from Euler angles
        rotation_matrix = self._euler_to_rotation_matrix(fused_roll, fused_pitch, fused_yaw)

        # Calculate confidence (based on accelerometer consistency with gravity)
        accel_mag = np.linalg.norm(accel_data.acceleration)
        gravity_error = abs(accel_mag - self.COW_GRAVITY_MAGNITUDE)
        confidence = np.exp(-gravity_error)  # Exponential decay with error

        orientation = OrientationEstimate(
            roll=fused_roll,
            pitch=fused_pitch,
            yaw=fused_yaw,
            rotation_matrix=rotation_matrix,
            confidence=confidence,
            timestamp=accel_data.timestamp
        )

        # Update state
        self.current_orientation = orientation
        self.last_update_time = accel_data.timestamp

        return orientation

    def estimate_gravity_vector(
        self,
        accel_data: AccelerometerData,
        window_size: int = 10
    ) -> GravityVector:
        """
        Estimate gravity vector from accelerometer (Scenario 5).

        Averages recent accelerometer readings to estimate gravity direction.
        Assumes animal is stationary or moving slowly.
        """
        # Get recent accelerometer samples for averaging
        recent_samples = self.accelerometer_buffer[-window_size:] if len(self.accelerometer_buffer) >= window_size else self.accelerometer_buffer

        if not recent_samples:
            return self.gravity_estimate

        # Average acceleration vectors
        avg_accel = np.mean([s.acceleration for s in recent_samples], axis=0)

        # Magnitude should be ~9.81 m/s²
        magnitude = np.linalg.norm(avg_accel)

        # Direction (normalized)
        if magnitude > 0.1:  # Avoid division by zero
            direction = avg_accel / magnitude
        else:
            direction = self.gravity_estimate.direction

        # Down vector is opposite of gravity direction
        down_vector = -direction

        gravity = GravityVector(
            direction=direction,
            magnitude=magnitude,
            down_vector=down_vector,
            timestamp=accel_data.timestamp
        )

        # Update state
        self.gravity_estimate = gravity

        return gravity

    def calculate_orientation_angles(
        self,
        rotation_matrix: np.ndarray
    ) -> Tuple[float, float, float]:
        """
        Calculate roll, pitch, yaw from rotation matrix (Scenario 6).

        Uses ZYX Euler angle convention.
        Returns angles in radians.
        """
        # Extract angles from rotation matrix (ZYX convention)
        # R = Rz(yaw) * Ry(pitch) * Rx(roll)

        # Pitch: sin(pitch) = -R[2,0]
        sin_pitch = -rotation_matrix[2, 0]
        sin_pitch = np.clip(sin_pitch, -1.0, 1.0)
        pitch = np.arcsin(sin_pitch)

        # Check for gimbal lock
        if abs(np.cos(pitch)) > 1e-6:
            # Roll: tan(roll) = R[2,1] / R[2,2]
            roll = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])

            # Yaw: tan(yaw) = R[1,0] / R[0,0]
            yaw = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
        else:
            # Gimbal lock case
            roll = 0.0
            yaw = np.arctan2(-rotation_matrix[0, 1], rotation_matrix[1, 1])

        return roll, pitch, yaw

    # ========================================================================
    # VIBRATION ANALYSIS (Scenario 7)
    # ========================================================================

    def analyze_vibration_fft(
        self,
        accel_signal: np.ndarray,
        sampling_rate: Optional[float] = None
    ) -> VibrationAnalysis:
        """
        Analyze vibration in frequency domain using FFT (Scenario 7).

        Detects camera shake and vibration frequencies that may
        affect image quality during cattle monitoring.
        """
        if sampling_rate is None:
            sampling_rate = self.sampling_rate

        # Ensure signal is 1D
        if accel_signal.ndim > 1:
            accel_signal = np.linalg.norm(accel_signal, axis=1)

        # Remove DC component
        signal = accel_signal - np.mean(accel_signal)

        # Apply Hanning window to reduce spectral leakage
        window = np.hanning(len(signal))
        windowed_signal = signal * window

        # Compute FFT
        fft_result = np.fft.fft(windowed_signal)
        power_spectrum = np.abs(fft_result) ** 2

        # Frequency bins
        frequencies = np.fft.fftfreq(len(signal), d=1.0/sampling_rate)

        # Keep only positive frequencies
        positive_freq_mask = frequencies >= 0
        frequencies = frequencies[positive_freq_mask]
        power_spectrum = power_spectrum[positive_freq_mask]

        # Find dominant frequency (exclude DC component)
        if len(power_spectrum) > 1:
            dominant_idx = np.argmax(power_spectrum[1:]) + 1
            dominant_frequency = float(frequencies[dominant_idx])
        else:
            dominant_frequency = 0.0

        # Calculate overall vibration magnitude (RMS of AC component)
        vibration_magnitude = float(np.sqrt(np.mean(signal ** 2)))

        # Detect shaking
        is_shaking = vibration_magnitude > self.COW_TYPICAL_SHAKE_THRESHOLD

        vibration = VibrationAnalysis(
            frequencies=frequencies,
            power_spectrum=power_spectrum,
            dominant_frequency=dominant_frequency,
            vibration_magnitude=vibration_magnitude,
            is_shaking=is_shaking,
            timestamp=datetime.now()
        )

        return vibration

    def detect_camera_shake(
        self,
        vibration_analysis: VibrationAnalysis,
        shake_threshold: float = None
    ) -> bool:
        """
        Detect camera shake from vibration analysis (Scenario 7).

        Returns True if significant vibration detected.
        """
        if shake_threshold is None:
            shake_threshold = self.COW_TYPICAL_SHAKE_THRESHOLD

        return vibration_analysis.vibration_magnitude > shake_threshold

    # ========================================================================
    # MOTION BLUR CORRELATION (Scenario 8)
    # ========================================================================

    def estimate_motion_blur(
        self,
        gyro_data: GyroscopeData,
        exposure_time_ms: float = 33.0,
        focal_length_pixels: float = 500.0
    ) -> MotionBlurEstimate:
        """
        Estimate motion blur from IMU angular velocity (Scenario 8).

        Correlates high angular velocity with motion blur.
        Blur (pixels) H angular_velocity (rad/s) * exposure_time (s) * focal_length (pixels)
        """
        # Calculate angular velocity magnitude
        angular_vel_mag = float(np.linalg.norm(gyro_data.angular_velocity))

        # Convert exposure time to seconds
        exposure_time_s = exposure_time_ms / 1000.0

        # Estimate blur in pixels
        blur_pixels = angular_vel_mag * exposure_time_s * focal_length_pixels

        # Assess blur level
        blur_level = self.assess_blur_level(
            blur_pixels,
            self.COW_BLUR_LOW_THRESHOLD,
            self.COW_BLUR_MEDIUM_THRESHOLD,
            self.COW_BLUR_HIGH_THRESHOLD
        )

        # Determine if frame should be skipped
        should_skip = blur_level == 'high'

        motion_blur = MotionBlurEstimate(
            angular_velocity_magnitude=angular_vel_mag,
            estimated_blur_pixels=blur_pixels,
            blur_level=blur_level,
            should_skip_frame=should_skip,
            timestamp=gyro_data.timestamp
        )

        return motion_blur

    def assess_blur_level(
        self,
        blur_pixels: float,
        low_threshold: float = None,
        medium_threshold: float = None,
        high_threshold: float = None
    ) -> Literal['none', 'low', 'medium', 'high']:
        """
        Assess blur level from estimated blur pixels (Scenario 8).
        """
        if low_threshold is None:
            low_threshold = self.COW_BLUR_LOW_THRESHOLD
        if medium_threshold is None:
            medium_threshold = self.COW_BLUR_MEDIUM_THRESHOLD
        if high_threshold is None:
            high_threshold = self.COW_BLUR_HIGH_THRESHOLD

        if blur_pixels < low_threshold:
            return 'none'
        elif blur_pixels < medium_threshold:
            return 'low'
        elif blur_pixels < high_threshold:
            return 'medium'
        else:
            return 'high'

    # ========================================================================
    # CAMERA POSE ESTIMATION (Scenario 9)
    # ========================================================================

    def estimate_camera_pose_from_imu(
        self,
        orientation: OrientationEstimate,
        gravity: GravityVector
    ) -> CameraPoseIMU:
        """
        Estimate camera pose from IMU orientation (Scenario 9).

        Provides rotation matrix and pointing direction relative to world frame.
        """
        # Rotation matrix from orientation
        rotation_matrix = orientation.rotation_matrix

        # Calculate pointing direction (camera looks along -Z axis in camera frame)
        camera_z_axis = np.array([0.0, 0.0, -1.0])
        pointing_direction = rotation_matrix @ camera_z_axis
        pointing_direction = pointing_direction / np.linalg.norm(pointing_direction)

        # Calculate up vector (camera Y axis points up in camera frame)
        camera_y_axis = np.array([0.0, 1.0, 0.0])
        up_vector = rotation_matrix @ camera_y_axis
        up_vector = up_vector / np.linalg.norm(up_vector)

        camera_pose = CameraPoseIMU(
            rotation_matrix=rotation_matrix,
            pointing_direction=pointing_direction,
            up_vector=up_vector,
            orientation=orientation,
            timestamp=orientation.timestamp
        )

        return camera_pose

    def calculate_pointing_direction(
        self,
        rotation_matrix: np.ndarray
    ) -> np.ndarray:
        """
        Calculate camera pointing direction from rotation matrix (Scenario 9).
        """
        # Camera looks along -Z axis in camera frame
        camera_z_axis = np.array([0.0, 0.0, -1.0])
        pointing_direction = rotation_matrix @ camera_z_axis

        # Normalize
        pointing_direction = pointing_direction / np.linalg.norm(pointing_direction)

        return pointing_direction

    # ========================================================================
    # GRAVITY COMPENSATION (Scenario 10)
    # ========================================================================

    def apply_gravity_compensation(
        self,
        measurements: np.ndarray,
        gravity: GravityVector
    ) -> np.ndarray:
        """
        Apply gravity compensation to vertical measurements (Scenario 10).

        Corrects height measurements to be relative to true vertical.
        """
        # True vertical direction (opposite of gravity)
        true_vertical = -gravity.direction

        # Project measurements onto true vertical
        # This corrects for camera tilt
        if measurements.ndim == 1:
            # Single measurement
            compensated = np.dot(measurements, true_vertical) * true_vertical
        else:
            # Multiple measurements
            compensated = np.dot(measurements, true_vertical)[:, np.newaxis] * true_vertical

        return compensated

    def calculate_true_vertical(
        self,
        gravity: GravityVector
    ) -> np.ndarray:
        """
        Calculate true vertical direction from gravity (Scenario 10).
        """
        # True vertical is opposite of gravity direction (points up)
        true_vertical = -gravity.direction

        # Normalize
        true_vertical = true_vertical / np.linalg.norm(true_vertical)

        return true_vertical

    # ========================================================================
    # DEPTH CORRECTION (Scenario 11)
    # ========================================================================

    def generate_motion_corrected_depth(
        self,
        depth_map: np.ndarray,
        orientation_before: OrientationEstimate,
        orientation_after: OrientationEstimate,
        dt: float
    ) -> DepthCorrectionResult:
        """
        Generate motion-corrected depth map (Scenario 11).

        Corrects depth measurements for camera motion between frames.
        """
        # Calculate rotation correction
        rotation_correction = self.calculate_rotation_correction(
            orientation_before,
            orientation_after
        )

        # Estimate motion vector (simplified - assumes pure rotation)
        # In full implementation, would use translation from visual odometry
        motion_vector = np.array([0.0, 0.0, 0.0])

        # Apply rotation correction to depth map (simplified)
        # Full implementation would warp the depth map based on rotation
        corrected_depth = depth_map.copy()

        # Calculate correction magnitude
        rotation_angle = np.arccos(np.clip(
            (np.trace(rotation_correction) - 1) / 2,
            -1.0, 1.0
        ))
        correction_magnitude = float(rotation_angle)

        result = DepthCorrectionResult(
            corrected_depth=corrected_depth,
            original_depth=depth_map,
            motion_vector=motion_vector,
            rotation_correction=rotation_correction,
            correction_magnitude=correction_magnitude,
            timestamp=orientation_after.timestamp
        )

        return result

    def calculate_rotation_correction(
        self,
        orientation_before: OrientationEstimate,
        orientation_after: OrientationEstimate
    ) -> np.ndarray:
        """
        Calculate rotation correction matrix between orientations (Scenario 11).

        R_correction = R_after * R_before^T
        """
        # Get rotation matrices
        R_before = orientation_before.rotation_matrix
        R_after = orientation_after.rotation_matrix

        # Calculate correction (relative rotation)
        R_correction = R_after @ R_before.T

        return R_correction

    # ========================================================================
    # REPORT GENERATION
    # ========================================================================

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

        Combines all IMU analysis: orientation, gravity, vibration, blur, pose, depth correction.
        """
        # Apply sensor fusion
        orientation = self.apply_complementary_filter(
            synchronized_data.accelerometer,
            synchronized_data.gyroscope
        )

        # Estimate gravity
        gravity = self.estimate_gravity_vector(synchronized_data.accelerometer)

        # Analyze vibration (if we have enough samples)
        vibration = None
        if len(self.accelerometer_buffer) >= 20:
            accel_signal = np.array([s.acceleration for s in self.accelerometer_buffer[-20:]])
            vibration = self.analyze_vibration_fft(accel_signal)

        # Estimate motion blur
        motion_blur = self.estimate_motion_blur(synchronized_data.gyroscope)

        # Estimate camera pose
        camera_pose = self.estimate_camera_pose_from_imu(orientation, gravity)

        # Generate depth correction (if depth map and previous orientation provided)
        depth_correction = None
        if depth_map is not None and previous_orientation is not None:
            dt = (timestamp - previous_orientation.timestamp).total_seconds()
            if dt > 0:
                depth_correction = self.generate_motion_corrected_depth(
                    depth_map,
                    previous_orientation,
                    orientation,
                    dt
                )

        report = IMUReport(
            frame_number=frame_number,
            timestamp=timestamp,
            synchronized_data=synchronized_data,
            orientation=orientation,
            gravity=gravity,
            vibration=vibration,
            motion_blur=motion_blur,
            camera_pose=camera_pose,
            depth_correction=depth_correction
        )

        # Store report
        self.imu_reports.append(report)

        return report

    def save_imu_report_to_csv(
        self,
        report: IMUReport,
        csv_path: str
    ):
        """
        Save IMU report to CSV file.

        Creates/appends to CSV with all IMU metrics.
        """
        import os

        # Check if file exists to determine if we need header
        file_exists = os.path.exists(csv_path)

        with open(csv_path, 'a', newline='') as csvfile:
            fieldnames = [
                'frame_number', 'timestamp',
                'accel_x', 'accel_y', 'accel_z',
                'gyro_x', 'gyro_y', 'gyro_z',
                'roll_rad', 'pitch_rad', 'yaw_rad',
                'gravity_x', 'gravity_y', 'gravity_z', 'gravity_magnitude',
                'vibration_magnitude', 'is_shaking', 'dominant_frequency_hz',
                'angular_velocity_mag', 'blur_pixels', 'blur_level',
                'pointing_x', 'pointing_y', 'pointing_z',
                'up_x', 'up_y', 'up_z',
                'time_offset_ms', 'orientation_confidence'
            ]

            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            if not file_exists:
                writer.writeheader()

            # Prepare row data
            row = {
                'frame_number': report.frame_number,
                'timestamp': report.timestamp.isoformat(),
                'accel_x': report.synchronized_data.accelerometer.acceleration[0],
                'accel_y': report.synchronized_data.accelerometer.acceleration[1],
                'accel_z': report.synchronized_data.accelerometer.acceleration[2],
                'gyro_x': report.synchronized_data.gyroscope.angular_velocity[0],
                'gyro_y': report.synchronized_data.gyroscope.angular_velocity[1],
                'gyro_z': report.synchronized_data.gyroscope.angular_velocity[2],
                'roll_rad': report.orientation.roll,
                'pitch_rad': report.orientation.pitch,
                'yaw_rad': report.orientation.yaw,
                'gravity_x': report.gravity.direction[0],
                'gravity_y': report.gravity.direction[1],
                'gravity_z': report.gravity.direction[2],
                'gravity_magnitude': report.gravity.magnitude,
                'vibration_magnitude': report.vibration.vibration_magnitude if report.vibration else 0.0,
                'is_shaking': report.vibration.is_shaking if report.vibration else False,
                'dominant_frequency_hz': report.vibration.dominant_frequency if report.vibration else 0.0,
                'angular_velocity_mag': report.motion_blur.angular_velocity_magnitude if report.motion_blur else 0.0,
                'blur_pixels': report.motion_blur.estimated_blur_pixels if report.motion_blur else 0.0,
                'blur_level': report.motion_blur.blur_level if report.motion_blur else 'none',
                'pointing_x': report.camera_pose.pointing_direction[0],
                'pointing_y': report.camera_pose.pointing_direction[1],
                'pointing_z': report.camera_pose.pointing_direction[2],
                'up_x': report.camera_pose.up_vector[0],
                'up_y': report.camera_pose.up_vector[1],
                'up_z': report.camera_pose.up_vector[2],
                'time_offset_ms': report.synchronized_data.time_offset_ms,
                'orientation_confidence': report.orientation.confidence
            }

            writer.writerow(row)

    # ========================================================================
    # HELPER METHODS
    # ========================================================================

    def _estimate_roll_pitch_from_accel(self, acceleration: np.ndarray) -> Tuple[float, float]:
        """
        Estimate roll and pitch from accelerometer gravity vector.

        Assumes accelerometer measures gravity when stationary.
        """
        ax, ay, az = acceleration

        # Roll: rotation around X axis
        # tan(roll) = ay / az
        roll = np.arctan2(ay, az)

        # Pitch: rotation around Y axis
        # tan(pitch) = -ax / sqrt(ay^2 + az^2)
        pitch = np.arctan2(-ax, np.sqrt(ay**2 + az**2))

        return roll, pitch

    def _euler_to_rotation_matrix(self, roll: float, pitch: float, yaw: float) -> np.ndarray:
        """
        Convert Euler angles to rotation matrix (ZYX convention).

        R = Rz(yaw) * Ry(pitch) * Rx(roll)
        """
        # Rotation around X axis (roll)
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll), np.cos(roll)]
        ])

        # Rotation around Y axis (pitch)
        Ry = np.array([
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)]
        ])

        # Rotation around Z axis (yaw)
        Rz = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1]
        ])

        # Combined rotation: R = Rz * Ry * Rx
        R = Rz @ Ry @ Rx

        return R
