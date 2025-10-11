"""
Cow-Specific Motion Analysis Implementation

Implements motion analysis for cattle (cow, bull, calf, heifer, steer) using
classical computer vision algorithms on CPU.

Covers all 19 scenarios from scenario-8.feature:
- Optical flow (dense Farneback, sparse Lucas-Kanade)
- Motion visualization (magnitude, direction, vectors)
- Background subtraction (MOG2, KNN)
- Motion detection (history, frame differencing, activity level)
- Speed/acceleration analysis
- Periodic motion analysis (FFT, autocorrelation, gait)

Author: PGG Desktop App Team
Date: 2025-10-10
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from pathlib import Path
import csv

from utils.animal_motion_analysis import (
    AnimalMotionAnalyzer,
    OpticalFlowResult,
    SpeedMetrics,
    ActivityMetrics,
    GaitMetrics,
    MotionReport
)


class CowMotionAnalyzer(AnimalMotionAnalyzer):
    """
    Cow-specific motion analysis implementation.

    Supports: cow, bull, calf, heifer, steer
    """

    # Cow-specific constants
    COW_ACTIVE_THRESHOLD = 2.0  # Motion magnitude threshold for active state
    COW_RESTING_THRESHOLD = 0.5  # Motion magnitude threshold for resting state
    COW_TYPICAL_SPEED = 1.0  # Typical walking speed in m/s
    COW_MAX_SPEED = 4.0  # Maximum running speed in m/s

    def __init__(self, animal_type: str = 'cow', fps: int = 30):
        """
        Initialize cow motion analyzer.

        Args:
            animal_type: Type of cattle ('cow', 'bull', 'calf', 'heifer', 'steer')
            fps: Frame rate for temporal calculations
        """
        super().__init__(animal_type, fps)

        # Speed tracking
        self.speed_history: Dict[str, List[Tuple[datetime, float]]] = {}
        self.previous_speed: Dict[str, float] = {}

        # Motion signal tracking for periodic analysis
        self.motion_signals: Dict[str, List[float]] = {}

        # Initialize background subtractors
        self._initialize_background_subtractors()

    def _initialize_background_subtractors(self):
        """Initialize MOG2 and KNN background subtractors."""
        # MOG2: Gaussian Mixture-based Background/Foreground Segmentation
        self.bg_subtractor_mog2 = cv2.createBackgroundSubtractorMOG2(
            history=500,
            varThreshold=16,
            detectShadows=True
        )

        # KNN: K-Nearest Neighbors Background Subtractor
        self.bg_subtractor_knn = cv2.createBackgroundSubtractorKNN(
            history=500,
            dist2Threshold=400.0,
            detectShadows=True
        )

    # ========================================================================
    # OPTICAL FLOW IMPLEMENTATION
    # ========================================================================

    def calculate_dense_optical_flow(
        self,
        frame1: np.ndarray,
        frame2: np.ndarray
    ) -> OpticalFlowResult:
        """
        Calculate dense optical flow using Farneback algorithm.

        Scenario: Calculate dense optical flow with Farneback algorithm

        Args:
            frame1: Previous frame (grayscale)
            frame2: Current frame (grayscale)

        Returns:
            OpticalFlowResult with flow field, magnitude, and direction
        """
        # Convert to grayscale if needed
        if len(frame1.shape) == 3:
            frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        if len(frame2.shape) == 3:
            frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        # Calculate dense optical flow using Farneback
        flow = cv2.calcOpticalFlowFarneback(
            frame1, frame2,
            None,  # No pre-computed flow
            pyr_scale=0.5,  # Pyramid scale
            levels=3,  # Number of pyramid levels
            winsize=15,  # Window size
            iterations=3,  # Iterations at each pyramid level
            poly_n=5,  # Polynomial expansion neighborhood
            poly_sigma=1.2,  # Polynomial expansion sigma
            flags=0
        )

        # Calculate magnitude and direction
        magnitude = self.generate_motion_magnitude_map(flow)
        direction = self.generate_motion_direction_map(flow)

        return OpticalFlowResult(
            flow=flow,
            magnitude=magnitude,
            direction=direction,
            method='farneback',
            timestamp=datetime.now()
        )

    def calculate_sparse_optical_flow(
        self,
        frame1: np.ndarray,
        frame2: np.ndarray,
        feature_points: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate sparse optical flow using Lucas-Kanade.

        Scenario: Calculate sparse optical flow with Lucas-Kanade

        Args:
            frame1: Previous frame (grayscale)
            frame2: Current frame (grayscale)
            feature_points: Feature points to track (N, 2)

        Returns:
            Tuple of (new_points, status, error)
        """
        # Convert to grayscale if needed
        if len(frame1.shape) == 3:
            frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        if len(frame2.shape) == 3:
            frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        # Lucas-Kanade parameters
        lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        )

        # Calculate sparse optical flow
        new_points, status, error = cv2.calcOpticalFlowPyrLK(
            frame1, frame2,
            feature_points,
            None,
            **lk_params
        )

        return new_points, status, error

    def generate_motion_magnitude_map(self, flow: np.ndarray) -> np.ndarray:
        """
        Generate motion magnitude map from optical flow.

        Scenario: Generate motion magnitude map

        Args:
            flow: Optical flow field (H, W, 2)

        Returns:
            Motion magnitude map (H, W)
        """
        # magnitude = sqrt(flow_x² + flow_y²)
        magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
        return magnitude

    def generate_motion_direction_map(self, flow: np.ndarray) -> np.ndarray:
        """
        Generate motion direction map from optical flow.

        Scenario: Generate motion direction map

        Args:
            flow: Optical flow field (H, W, 2)

        Returns:
            Motion direction map in radians (H, W)
        """
        # direction = atan2(flow_y, flow_x)
        direction = np.arctan2(flow[..., 1], flow[..., 0])
        return direction

    def visualize_flow_vectors(
        self,
        frame: np.ndarray,
        flow: np.ndarray,
        step: int = 16
    ) -> np.ndarray:
        """
        Visualize optical flow as arrows on frame.

        Scenario: Visualize flow vectors

        Args:
            frame: Base frame for visualization
            flow: Optical flow field
            step: Sampling step for arrows

        Returns:
            Frame with flow arrows drawn
        """
        h, w = frame.shape[:2]
        vis = frame.copy()

        # Create grid of points
        y, x = np.mgrid[step//2:h:step, step//2:w:step].reshape(2, -1).astype(int)

        # Get flow at each point
        fx, fy = flow[y, x].T

        # Calculate magnitude for color
        mag = np.sqrt(fx**2 + fy**2)

        # Normalize magnitude for visualization
        mag_normalized = np.clip(mag * 5, 0, 255).astype(np.uint8)

        # Draw arrows
        for i in range(len(x)):
            # Skip if motion is negligible
            if mag[i] < 0.5:
                continue

            # Calculate arrow endpoints
            x1, y1 = x[i], y[i]
            x2, y2 = int(x1 + fx[i] * 3), int(y1 + fy[i] * 3)

            # Color based on magnitude (red for high, blue for low)
            color = (0, 255 - mag_normalized[i], mag_normalized[i])

            # Draw arrow
            cv2.arrowedLine(vis, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA, tipLength=0.3)

        return vis

    # ========================================================================
    # BACKGROUND SUBTRACTION IMPLEMENTATION
    # ========================================================================

    def apply_background_subtraction_mog2(
        self,
        frame: np.ndarray
    ) -> np.ndarray:
        """
        Apply MOG2 background subtraction.

        Scenario: Apply background subtraction with MOG2

        Args:
            frame: Current frame (RGB or grayscale)

        Returns:
            Foreground mask (binary)
        """
        # Apply MOG2
        fg_mask = self.bg_subtractor_mog2.apply(frame)

        # Post-process: remove shadows (value 127 in mask)
        fg_mask[fg_mask == 127] = 0

        # Apply morphological operations to reduce noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)

        return fg_mask

    def apply_background_subtraction_knn(
        self,
        frame: np.ndarray
    ) -> np.ndarray:
        """
        Apply KNN background subtraction.

        Scenario: Apply background subtraction with KNN

        Args:
            frame: Current frame (RGB or grayscale)

        Returns:
            Foreground mask (binary)
        """
        # Apply KNN
        fg_mask = self.bg_subtractor_knn.apply(frame)

        # Post-process: remove shadows
        fg_mask[fg_mask == 127] = 0

        # Apply morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)

        return fg_mask

    def generate_motion_history_image(
        self,
        foreground_mask: np.ndarray,
        timestamp: float
    ) -> np.ndarray:
        """
        Generate motion history image.

        Scenario: Generate motion history image

        Args:
            foreground_mask: Binary foreground mask
            timestamp: Current timestamp

        Returns:
            Motion history image (recent motion brighter)
        """
        # Initialize motion history image if needed
        if self.motion_history_image is None:
            self.motion_history_image = np.zeros(foreground_mask.shape, dtype=np.float32)

        # Update motion history: set current motion to timestamp, decay old motion
        # Recent motion = high values, old motion = low values
        self.motion_history_image = np.where(
            foreground_mask > 0,
            timestamp,  # Set to current timestamp where motion detected
            self.motion_history_image  # Keep existing value
        )

        # Create visualization (0-255)
        # Recent motion (within duration) will be bright, old motion fades
        current_time = timestamp
        age = current_time - self.motion_history_image

        # Normalize to 0-255 (recent = bright, old = dark)
        vis = np.where(
            age < self.motion_history_duration,
            255 * (1.0 - age / self.motion_history_duration),
            0
        ).astype(np.uint8)

        return vis

    def apply_frame_differencing(
        self,
        frame1: np.ndarray,
        frame2: np.ndarray,
        threshold: int = 25
    ) -> np.ndarray:
        """
        Apply frame differencing to detect motion.

        Scenario: Apply frame differencing

        Args:
            frame1: Previous frame
            frame2: Current frame
            threshold: Difference threshold

        Returns:
            Binary motion mask
        """
        # Convert to grayscale if needed
        if len(frame1.shape) == 3:
            frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        if len(frame2.shape) == 3:
            frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        # Calculate absolute difference
        diff = cv2.absdiff(frame1, frame2)

        # Apply threshold
        _, motion_mask = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)

        # Clean up noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, kernel)

        return motion_mask

    # ========================================================================
    # ACTIVITY DETECTION IMPLEMENTATION
    # ========================================================================

    def detect_activity_level(
        self,
        motion_magnitude: np.ndarray,
        threshold_active: float = None,
        threshold_resting: float = None
    ) -> ActivityMetrics:
        """
        Detect activity level from motion magnitude.

        Scenario: Detect activity level (active vs. resting)

        Args:
            motion_magnitude: Motion magnitude map
            threshold_active: Threshold for active state (uses cow default if None)
            threshold_resting: Threshold for resting state (uses cow default if None)

        Returns:
            ActivityMetrics with activity classification
        """
        # Use cow-specific thresholds if not provided
        if threshold_active is None:
            threshold_active = self.COW_ACTIVE_THRESHOLD
        if threshold_resting is None:
            threshold_resting = self.COW_RESTING_THRESHOLD

        # Calculate total motion
        total_motion = np.sum(motion_magnitude)

        # Calculate motion statistics
        mean_motion = np.mean(motion_magnitude)
        motion_pixels = np.sum(motion_magnitude > 0.1)
        total_pixels = motion_magnitude.size
        motion_percentage = (motion_pixels / total_pixels) * 100

        # Classify activity level
        if mean_motion > threshold_active:
            activity_level = 'active'
        elif mean_motion < threshold_resting:
            activity_level = 'resting'
        else:
            activity_level = 'moderate'

        return ActivityMetrics(
            activity_level=activity_level,
            total_motion=float(total_motion),
            motion_percentage=float(motion_percentage),
            foreground_area=int(motion_pixels),
            background_area=int(total_pixels - motion_pixels),
            timestamp=datetime.now()
        )

    # ========================================================================
    # SPEED & ACCELERATION IMPLEMENTATION
    # ========================================================================

    def calculate_real_world_speed(
        self,
        flow: np.ndarray,
        depth_frame: np.ndarray,
        camera_intrinsics: Dict[str, float]
    ) -> float:
        """
        Calculate real-world speed using depth and camera intrinsics.

        Scenario: Calculate real-world speed using depth

        Args:
            flow: Optical flow field
            depth_frame: Depth frame in meters
            camera_intrinsics: Camera parameters (fx, fy, cx, cy)

        Returns:
            Speed in meters per second
        """
        # Get camera parameters
        fx = camera_intrinsics.get('fx', 1.0)
        fy = camera_intrinsics.get('fy', 1.0)

        # Get median depth (ignore invalid depths)
        valid_depth = depth_frame[depth_frame > 0]
        if len(valid_depth) == 0:
            return 0.0

        median_depth = np.median(valid_depth)

        # Calculate motion in pixels
        flow_magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
        median_flow = np.median(flow_magnitude[flow_magnitude > 0.5]) if np.any(flow_magnitude > 0.5) else 0.0

        # Convert pixel motion to real-world motion using depth
        # real_motion = (pixel_motion / focal_length) * depth
        real_motion_x = (median_flow / fx) * median_depth

        # Speed = distance / time
        # time = frame_time (1 / fps)
        speed = real_motion_x / self.frame_time

        return float(speed)

    def detect_acceleration(
        self,
        current_speed: float,
        previous_speed: float
    ) -> float:
        """
        Detect acceleration from speed changes.

        Scenario: Detect acceleration

        Args:
            current_speed: Current speed in m/s
            previous_speed: Previous speed in m/s

        Returns:
            Acceleration in m/s²
        """
        # acceleration = delta_speed / delta_time
        delta_speed = current_speed - previous_speed
        acceleration = delta_speed / self.frame_time

        return float(acceleration)

    def track_movement_speed_over_time(
        self,
        animal_id: str,
        speed: float,
        timestamp: datetime
    ):
        """
        Track movement speed over time for an animal.

        Scenario: Track movement speed over time

        Args:
            animal_id: Unique animal identifier
            speed: Speed in m/s
            timestamp: Measurement timestamp
        """
        if animal_id not in self.speed_history:
            self.speed_history[animal_id] = []

        self.speed_history[animal_id].append((timestamp, speed))

        # Update previous speed for acceleration calculation
        self.previous_speed[animal_id] = speed

    def calculate_distance_traveled(
        self,
        trajectory: List[np.ndarray]
    ) -> float:
        """
        Calculate total distance traveled from trajectory.

        Scenario: Calculate distance traveled

        Args:
            trajectory: List of 3D positions (x, y, z) in meters

        Returns:
            Total distance in meters
        """
        if len(trajectory) < 2:
            return 0.0

        total_distance = 0.0

        # Sum segment lengths
        for i in range(1, len(trajectory)):
            segment = trajectory[i] - trajectory[i-1]
            segment_length = np.linalg.norm(segment)
            total_distance += segment_length

        return float(total_distance)

    # ========================================================================
    # PERIODIC MOTION ANALYSIS IMPLEMENTATION
    # ========================================================================

    def detect_periodic_motion_fft(
        self,
        motion_signal: np.ndarray,
        sampling_rate: float
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Detect periodic motion using FFT.

        Scenario: Detect periodic motion with FFT

        Args:
            motion_signal: 1D motion signal over time
            sampling_rate: Sampling rate in Hz

        Returns:
            Tuple of (dominant_frequency, frequencies, power_spectrum)
        """
        # Remove DC component (mean)
        signal = motion_signal - np.mean(motion_signal)

        # Apply FFT
        fft_result = np.fft.fft(signal)
        power_spectrum = np.abs(fft_result) ** 2

        # Get frequencies
        n = len(signal)
        frequencies = np.fft.fftfreq(n, d=1.0/sampling_rate)

        # Only use positive frequencies
        positive_freqs = frequencies[:n//2]
        positive_power = power_spectrum[:n//2]

        # Find dominant frequency (ignore DC component at index 0)
        if len(positive_power) > 1:
            dominant_idx = np.argmax(positive_power[1:]) + 1
            dominant_frequency = positive_freqs[dominant_idx]
        else:
            dominant_frequency = 0.0

        return float(dominant_frequency), positive_freqs, positive_power

    def detect_periodic_motion_autocorrelation(
        self,
        motion_signal: np.ndarray
    ) -> Tuple[float, float]:
        """
        Detect periodic motion using autocorrelation.

        Scenario: Detect periodic motion with autocorrelation

        Args:
            motion_signal: 1D motion signal over time

        Returns:
            Tuple of (period_duration, periodicity_score)
        """
        # Normalize signal
        signal = motion_signal - np.mean(motion_signal)

        # Compute autocorrelation using numpy correlate
        autocorr = np.correlate(signal, signal, mode='full')
        autocorr = autocorr[len(autocorr)//2:]  # Keep only positive lags

        # Normalize by zero-lag value
        if autocorr[0] > 0:
            autocorr = autocorr / autocorr[0]

        # Find first peak after lag 1 (ignore lag 0 which is always 1.0)
        # Look for local maxima
        peaks = []
        for i in range(2, len(autocorr) - 1):
            if autocorr[i] > autocorr[i-1] and autocorr[i] > autocorr[i+1]:
                peaks.append((i, autocorr[i]))

        if peaks:
            # Get highest peak
            period_lag, periodicity_score = max(peaks, key=lambda x: x[1])
            period_duration = period_lag / self.fps  # Convert lag to seconds
        else:
            period_duration = 0.0
            periodicity_score = 0.0

        return float(period_duration), float(periodicity_score)

    def analyze_step_frequency(
        self,
        dominant_frequency: float
    ) -> float:
        """
        Analyze step frequency from dominant frequency.

        Scenario: Analyze step frequency

        Args:
            dominant_frequency: Dominant frequency in Hz

        Returns:
            Steps per second
        """
        # For walking/gait, dominant frequency often corresponds to step frequency
        # For cows, typical walking frequency is 0.5-2 Hz (0.5-2 steps per second)
        steps_per_second = dominant_frequency

        return float(steps_per_second)

    def measure_stride_length(
        self,
        speed: float,
        step_frequency: float
    ) -> float:
        """
        Measure stride length from speed and step frequency.

        Scenario: Measure stride length

        Args:
            speed: Speed in m/s
            step_frequency: Steps per second

        Returns:
            Stride length in meters
        """
        if step_frequency == 0:
            return 0.0

        # stride_length = speed / step_frequency
        stride_length = speed / step_frequency

        return float(stride_length)

    def analyze_gait_cycle(
        self,
        motion_signal: np.ndarray,
        sampling_rate: float
    ) -> GaitMetrics:
        """
        Analyze gait cycle for periodic motion patterns.

        Scenario: Analyze gait cycle

        Args:
            motion_signal: 1D motion signal over time
            sampling_rate: Sampling rate in Hz

        Returns:
            GaitMetrics with gait analysis results
        """
        # Detect periodicity using both FFT and autocorrelation
        dominant_freq, freqs, power = self.detect_periodic_motion_fft(
            motion_signal, sampling_rate
        )
        period_duration, periodicity_score = self.detect_periodic_motion_autocorrelation(
            motion_signal
        )

        # Calculate step frequency
        step_frequency = self.analyze_step_frequency(dominant_freq)

        # Estimate stride length (assuming typical cow walking speed)
        stride_length = self.measure_stride_length(self.COW_TYPICAL_SPEED, step_frequency)

        # Gait cycle duration
        if step_frequency > 0:
            gait_cycle_duration = 1.0 / step_frequency
        else:
            gait_cycle_duration = 0.0

        # Analyze symmetry (simplified: check if signal is symmetric around peaks)
        # High symmetry score means regular, repeating pattern
        symmetry_score = periodicity_score  # Use autocorrelation peak as proxy

        # Estimate stance phase ratio (simplified)
        # For cows, stance phase is typically 60-70% of gait cycle
        # We'll estimate from signal: ratio of time above mean
        stance_phase_ratio = np.sum(motion_signal > np.mean(motion_signal)) / len(motion_signal)

        return GaitMetrics(
            step_frequency_hz=step_frequency,
            stride_length_m=stride_length,
            gait_cycle_duration_s=gait_cycle_duration,
            dominant_frequency_hz=dominant_freq,
            periodicity_score=periodicity_score,
            symmetry_score=symmetry_score,
            stance_phase_ratio=stance_phase_ratio,
            timestamp=datetime.now()
        )

    # ========================================================================
    # REPORT GENERATION IMPLEMENTATION
    # ========================================================================

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
        if timestamp is None:
            timestamp = datetime.now()

        # Initialize report components
        optical_flow_result = None
        speed_metrics = None
        activity_metrics = None
        gait_metrics = None

        # Calculate optical flow if we have previous frame
        if self.previous_frame is not None:
            optical_flow_result = self.calculate_dense_optical_flow(
                self.previous_frame,
                frame
            )

            # Detect activity level
            activity_metrics = self.detect_activity_level(
                optical_flow_result.magnitude
            )

            # Calculate speed if depth frame available
            if depth_frame is not None and camera_intrinsics is not None:
                current_speed = self.calculate_real_world_speed(
                    optical_flow_result.flow,
                    depth_frame,
                    camera_intrinsics
                )

                # Track speed over time
                self.track_movement_speed_over_time(animal_id, current_speed, timestamp)

                # Calculate acceleration if we have previous speed
                prev_speed = self.previous_speed.get(animal_id, 0.0)
                acceleration = self.detect_acceleration(current_speed, prev_speed)

                # Get speed statistics
                if animal_id in self.speed_history:
                    speeds = [s for _, s in self.speed_history[animal_id]]
                    avg_speed = np.mean(speeds) if speeds else 0.0
                    max_speed = np.max(speeds) if speeds else 0.0
                else:
                    avg_speed = current_speed
                    max_speed = current_speed

                # Calculate distance traveled (simplified: speed * time)
                distance_traveled = current_speed * self.frame_time

                speed_metrics = SpeedMetrics(
                    speed_m_s=current_speed,
                    acceleration_m_s2=acceleration,
                    distance_traveled_m=distance_traveled,
                    average_speed_m_s=avg_speed,
                    max_speed_m_s=max_speed,
                    timestamp=timestamp
                )

                # Track motion signal for gait analysis
                if animal_id not in self.motion_signals:
                    self.motion_signals[animal_id] = []
                self.motion_signals[animal_id].append(np.mean(optical_flow_result.magnitude))

                # Perform gait analysis if we have enough data (at least 2 seconds)
                if len(self.motion_signals[animal_id]) >= int(2 * self.fps):
                    motion_signal = np.array(self.motion_signals[animal_id][-int(5 * self.fps):])
                    gait_metrics = self.analyze_gait_cycle(motion_signal, self.fps)

        # Update previous frame
        self.previous_frame = frame.copy()

        # Create and store report
        report = MotionReport(
            animal_id=animal_id,
            animal_type=self.animal_type,
            timestamp=timestamp,
            optical_flow=optical_flow_result,
            speed_metrics=speed_metrics,
            activity_metrics=activity_metrics,
            gait_metrics=gait_metrics,
            frame_number=frame_number
        )

        self.motion_history.append(report)

        return report

    def save_motion_data_to_csv(
        self,
        report: MotionReport,
        csv_path: str
    ):
        """
        Save motion report to CSV file.

        Args:
            report: MotionReport to save
            csv_path: Path to CSV file
        """
        # Check if file exists to determine if we need headers
        file_exists = Path(csv_path).exists()

        # Open in append mode
        with open(csv_path, 'a', newline='') as csvfile:
            fieldnames = [
                'timestamp', 'animal_id', 'animal_type', 'frame_number',
                'activity_level', 'total_motion', 'motion_percentage',
                'speed_m_s', 'acceleration_m_s2', 'distance_traveled_m',
                'avg_speed_m_s', 'max_speed_m_s',
                'step_frequency_hz', 'stride_length_m', 'gait_cycle_duration_s',
                'dominant_frequency_hz', 'periodicity_score', 'symmetry_score'
            ]

            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            # Write header if new file
            if not file_exists:
                writer.writeheader()

            # Prepare row data
            row = {
                'timestamp': report.timestamp.isoformat(),
                'animal_id': report.animal_id,
                'animal_type': report.animal_type,
                'frame_number': report.frame_number,
                'activity_level': report.activity_metrics.activity_level if report.activity_metrics else '',
                'total_motion': report.activity_metrics.total_motion if report.activity_metrics else 0.0,
                'motion_percentage': report.activity_metrics.motion_percentage if report.activity_metrics else 0.0,
                'speed_m_s': report.speed_metrics.speed_m_s if report.speed_metrics else 0.0,
                'acceleration_m_s2': report.speed_metrics.acceleration_m_s2 if report.speed_metrics else 0.0,
                'distance_traveled_m': report.speed_metrics.distance_traveled_m if report.speed_metrics else 0.0,
                'avg_speed_m_s': report.speed_metrics.average_speed_m_s if report.speed_metrics else 0.0,
                'max_speed_m_s': report.speed_metrics.max_speed_m_s if report.speed_metrics else 0.0,
                'step_frequency_hz': report.gait_metrics.step_frequency_hz if report.gait_metrics else 0.0,
                'stride_length_m': report.gait_metrics.stride_length_m if report.gait_metrics else 0.0,
                'gait_cycle_duration_s': report.gait_metrics.gait_cycle_duration_s if report.gait_metrics else 0.0,
                'dominant_frequency_hz': report.gait_metrics.dominant_frequency_hz if report.gait_metrics else 0.0,
                'periodicity_score': report.gait_metrics.periodicity_score if report.gait_metrics else 0.0,
                'symmetry_score': report.gait_metrics.symmetry_score if report.gait_metrics else 0.0,
            }

            writer.writerow(row)
