"""
Cow-Specific Trajectory and Path Analysis Implementation

Implements trajectory tracking and path analysis for cattle (cow, bull, calf,
heifer, steer) using classical algorithms on CPU.

Covers all 12 scenarios from scenario-9.feature:
- Trajectory recording (3D positions over time)
- Trajectory smoothing (Kalman filter, moving average)
- Path analysis (length, turning angles)
- Clustering (DBSCAN, k-means, pattern identification)
- Position prediction (linear extrapolation, LSTM/GRU)
- Camera pose estimation (6DOF, visual odometry, Euler angles)

Author: PGG Desktop App Team
Date: 2025-10-10
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
from pathlib import Path
import csv

from utils.animal_trajectory_and_path_analysis import (
    AnimalTrajectoryAnalyzer,
    TrajectoryPoint,
    Trajectory,
    PathMetrics,
    TrajectoryCluster,
    CameraPose,
    TrajectoryReport
)


class CowTrajectoryAnalyzer(AnimalTrajectoryAnalyzer):
    """
    Cow-specific trajectory and path analysis implementation.

    Supports: cow, bull, calf, heifer, steer
    """

    # Cow-specific constants
    COW_SHARP_TURN_THRESHOLD = 45.0  # Degrees
    COW_TYPICAL_SPEED = 1.0  # m/s
    COW_MIN_TRAJECTORY_LENGTH = 10  # Minimum points for valid trajectory

    def __init__(self, animal_type: str = 'cow', fps: int = 30):
        """
        Initialize cow trajectory analyzer.

        Args:
            animal_type: Type of cattle ('cow', 'bull', 'calf', 'heifer', 'steer')
            fps: Frame rate for temporal calculations
        """
        super().__init__(animal_type, fps)

        # Kalman filter for trajectory smoothing (initialized per trajectory)
        self.kalman_filters: Dict[str, Any] = {}

    # ========================================================================
    # TRAJECTORY RECORDING IMPLEMENTATION
    # ========================================================================

    def record_trajectory_point(
        self,
        animal_id: str,
        position: np.ndarray,
        timestamp: datetime,
        frame_number: int
    ):
        """
        Record a 3D trajectory point for an animal.

        Scenario: Record 3D trajectory of moving object

        Args:
            animal_id: Unique animal identifier
            position: 3D position (x, y, z) in meters
            timestamp: Measurement timestamp
            frame_number: Frame number
        """
        # Calculate velocity if we have previous point
        velocity = None
        if animal_id in self.active_trajectories and len(self.active_trajectories[animal_id]) > 0:
            prev_point = self.active_trajectories[animal_id][-1]
            time_delta = (timestamp - prev_point.timestamp).total_seconds()
            if time_delta > 0:
                displacement = position - prev_point.position
                velocity = displacement / time_delta

        # Create trajectory point
        point = TrajectoryPoint(
            position=position,
            timestamp=timestamp,
            frame_number=frame_number,
            velocity=velocity
        )

        # Add to active trajectory
        if animal_id not in self.active_trajectories:
            self.active_trajectories[animal_id] = []

        self.active_trajectories[animal_id].append(point)

    def get_trajectory(self, animal_id: str) -> Optional[Trajectory]:
        """
        Get complete trajectory for an animal.

        Args:
            animal_id: Unique animal identifier

        Returns:
            Trajectory or None if not found
        """
        # Check completed trajectories first
        if animal_id in self.trajectories:
            return self.trajectories[animal_id]

        # Check active trajectories
        if animal_id in self.active_trajectories:
            points = self.active_trajectories[animal_id]
            if len(points) >= self.COW_MIN_TRAJECTORY_LENGTH:
                # Calculate total length
                total_length = self._calculate_trajectory_length(points)

                return Trajectory(
                    animal_id=animal_id,
                    points=points,
                    total_length=total_length,
                    smoothed=False
                )

        return None

    def _calculate_trajectory_length(self, points: List[TrajectoryPoint]) -> float:
        """Calculate total trajectory length."""
        if len(points) < 2:
            return 0.0

        total_length = 0.0
        for i in range(1, len(points)):
            segment = points[i].position - points[i-1].position
            total_length += np.linalg.norm(segment)

        return float(total_length)

    # ========================================================================
    # TRAJECTORY SMOOTHING IMPLEMENTATION
    # ========================================================================

    def smooth_trajectory_kalman(
        self,
        trajectory: Trajectory,
        process_noise: float = 0.01,
        measurement_noise: float = 0.1
    ) -> Trajectory:
        """
        Smooth trajectory using Kalman filter.

        Scenario: Smooth trajectory with Kalman filter

        Args:
            trajectory: Noisy trajectory
            process_noise: Process noise covariance
            measurement_noise: Measurement noise covariance

        Returns:
            Smoothed trajectory
        """
        if len(trajectory.points) < 2:
            return trajectory

        # Initialize Kalman filter for 3D position + 3D velocity (6 state variables)
        # State: [x, y, z, vx, vy, vz]
        kalman = cv2.KalmanFilter(6, 3)  # 6 state variables, 3 measurements

        # State transition matrix (constant velocity model)
        dt = self.frame_time
        kalman.transitionMatrix = np.array([
            [1, 0, 0, dt, 0, 0],
            [0, 1, 0, 0, dt, 0],
            [0, 0, 1, 0, 0, dt],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ], dtype=np.float32)

        # Measurement matrix (we only measure position)
        kalman.measurementMatrix = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0]
        ], dtype=np.float32)

        # Process noise covariance
        kalman.processNoiseCov = np.eye(6, dtype=np.float32) * process_noise

        # Measurement noise covariance
        kalman.measurementNoiseCov = np.eye(3, dtype=np.float32) * measurement_noise

        # Initialize with first point
        first_pos = trajectory.points[0].position
        kalman.statePre = np.array([
            [first_pos[0]], [first_pos[1]], [first_pos[2]],
            [0.0], [0.0], [0.0]
        ], dtype=np.float32)

        # Smooth all points
        smoothed_points = []
        for point in trajectory.points:
            # Predict
            prediction = kalman.predict()

            # Update with measurement
            measurement = np.array([[point.position[0]],
                                   [point.position[1]],
                                   [point.position[2]]], dtype=np.float32)
            kalman.correct(measurement)

            # Get smoothed position
            smoothed_pos = kalman.statePost[:3].flatten()

            # Create smoothed point
            smoothed_point = TrajectoryPoint(
                position=smoothed_pos,
                timestamp=point.timestamp,
                frame_number=point.frame_number,
                velocity=kalman.statePost[3:6].flatten() if len(smoothed_points) > 0 else None
            )
            smoothed_points.append(smoothed_point)

        # Calculate new total length
        total_length = self._calculate_trajectory_length(smoothed_points)

        return Trajectory(
            animal_id=trajectory.animal_id,
            points=smoothed_points,
            total_length=total_length,
            smoothed=True
        )

    def smooth_trajectory_moving_average(
        self,
        trajectory: Trajectory,
        window_size: int = 5
    ) -> Trajectory:
        """
        Smooth trajectory using moving average filter.

        Scenario: Smooth trajectory with moving average

        Args:
            trajectory: Noisy trajectory
            window_size: Number of consecutive points to average

        Returns:
            Smoothed trajectory
        """
        if len(trajectory.points) < window_size:
            return trajectory

        smoothed_points = []
        positions = np.array([p.position for p in trajectory.points])

        # Apply moving average
        for i in range(len(positions)):
            # Calculate window bounds
            start = max(0, i - window_size // 2)
            end = min(len(positions), i + window_size // 2 + 1)

            # Average positions in window
            smoothed_pos = np.mean(positions[start:end], axis=0)

            # Calculate velocity if we have previous smoothed point
            velocity = None
            if len(smoothed_points) > 0:
                prev_point = smoothed_points[-1]
                time_delta = (trajectory.points[i].timestamp - prev_point.timestamp).total_seconds()
                if time_delta > 0:
                    displacement = smoothed_pos - prev_point.position
                    velocity = displacement / time_delta

            # Create smoothed point
            smoothed_point = TrajectoryPoint(
                position=smoothed_pos,
                timestamp=trajectory.points[i].timestamp,
                frame_number=trajectory.points[i].frame_number,
                velocity=velocity
            )
            smoothed_points.append(smoothed_point)

        # Calculate new total length
        total_length = self._calculate_trajectory_length(smoothed_points)

        return Trajectory(
            animal_id=trajectory.animal_id,
            points=smoothed_points,
            total_length=total_length,
            smoothed=True
        )

    # ========================================================================
    # PATH ANALYSIS IMPLEMENTATION
    # ========================================================================

    def calculate_path_length(self, trajectory: Trajectory) -> float:
        """
        Calculate total path length.

        Scenario: Calculate path length

        Args:
            trajectory: Trajectory to analyze

        Returns:
            Total path length in meters
        """
        return self._calculate_trajectory_length(trajectory.points)

    def analyze_turning_angles(
        self,
        trajectory: Trajectory,
        sharp_turn_threshold: float = None
    ) -> Tuple[List[float], int]:
        """
        Analyze turning angles in trajectory.

        Scenario: Analyze turning angles

        Args:
            trajectory: Trajectory to analyze
            sharp_turn_threshold: Threshold for sharp turns in degrees

        Returns:
            Tuple of (turning_angles, sharp_turns_count)
        """
        if sharp_turn_threshold is None:
            sharp_turn_threshold = self.COW_SHARP_TURN_THRESHOLD

        if len(trajectory.points) < 3:
            return [], 0

        turning_angles = []
        sharp_turns_count = 0

        # Calculate angle between consecutive segments
        for i in range(1, len(trajectory.points) - 1):
            # Get three consecutive points
            p1 = trajectory.points[i - 1].position
            p2 = trajectory.points[i].position
            p3 = trajectory.points[i + 1].position

            # Calculate vectors
            v1 = p2 - p1
            v2 = p3 - p2

            # Calculate norms
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)

            # Avoid division by zero
            if norm1 < 1e-6 or norm2 < 1e-6:
                continue

            # Calculate angle using dot product
            cos_angle = np.dot(v1, v2) / (norm1 * norm2)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)  # Ensure valid range
            angle_rad = np.arccos(cos_angle)
            angle_deg = np.degrees(angle_rad)

            turning_angles.append(angle_deg)

            # Count sharp turns
            if angle_deg > sharp_turn_threshold:
                sharp_turns_count += 1

        return turning_angles, sharp_turns_count

    def calculate_path_metrics(self, trajectory: Trajectory) -> PathMetrics:
        """
        Calculate comprehensive path metrics.

        Args:
            trajectory: Trajectory to analyze

        Returns:
            PathMetrics with all path statistics
        """
        if len(trajectory.points) < 2:
            return PathMetrics(
                total_length_m=0.0,
                average_speed_m_s=0.0,
                max_speed_m_s=0.0,
                turning_angles=[],
                sharp_turns_count=0,
                total_duration_s=0.0,
                timestamp=datetime.now()
            )

        # Calculate path length
        total_length = self.calculate_path_length(trajectory)

        # Calculate duration
        start_time = trajectory.points[0].timestamp
        end_time = trajectory.points[-1].timestamp
        total_duration = (end_time - start_time).total_seconds()

        # Calculate speeds
        speeds = []
        for point in trajectory.points:
            if point.velocity is not None:
                speed = np.linalg.norm(point.velocity)
                speeds.append(speed)

        average_speed = np.mean(speeds) if speeds else 0.0
        max_speed = np.max(speeds) if speeds else 0.0

        # Analyze turning angles
        turning_angles, sharp_turns_count = self.analyze_turning_angles(trajectory)

        return PathMetrics(
            total_length_m=total_length,
            average_speed_m_s=float(average_speed),
            max_speed_m_s=float(max_speed),
            turning_angles=turning_angles,
            sharp_turns_count=sharp_turns_count,
            total_duration_s=total_duration,
            timestamp=datetime.now()
        )

    # ========================================================================
    # TRAJECTORY CLUSTERING IMPLEMENTATION
    # ========================================================================

    def cluster_trajectories_dbscan(
        self,
        trajectories: List[Trajectory],
        eps: float = 0.5,
        min_samples: int = 2
    ) -> List[TrajectoryCluster]:
        """
        Cluster trajectories using DBSCAN.

        Scenario: Cluster trajectories with DBSCAN

        Args:
            trajectories: List of trajectories to cluster
            eps: Maximum distance between samples
            min_samples: Minimum samples in a neighborhood

        Returns:
            List of trajectory clusters
        """
        try:
            from sklearn.cluster import DBSCAN
        except ImportError:
            print("Warning: scikit-learn not available. Returning single cluster.")
            return [TrajectoryCluster(
                cluster_id=0,
                trajectories=trajectories,
                centroid=None,
                size=len(trajectories)
            )]

        if len(trajectories) < min_samples:
            return []

        # Extract features from trajectories (start point, end point, length, duration)
        features = []
        for traj in trajectories:
            if len(traj.points) < 2:
                continue

            start_pos = traj.points[0].position
            end_pos = traj.points[-1].position
            length = traj.total_length
            duration = (traj.points[-1].timestamp - traj.points[0].timestamp).total_seconds()

            # Feature vector: [start_x, start_y, start_z, end_x, end_y, end_z, length, duration]
            feature = np.concatenate([start_pos, end_pos, [length, duration]])
            features.append(feature)

        features = np.array(features)

        # Apply DBSCAN
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(features)
        labels = clustering.labels_

        # Group trajectories by cluster
        clusters_dict = {}
        for idx, label in enumerate(labels):
            if label not in clusters_dict:
                clusters_dict[label] = []
            clusters_dict[label].append(trajectories[idx])

        # Create TrajectoryCluster objects
        clusters = []
        for cluster_id, cluster_trajs in clusters_dict.items():
            # Calculate centroid (average of all trajectory positions)
            all_positions = []
            for traj in cluster_trajs:
                for point in traj.points:
                    all_positions.append(point.position)

            centroid = np.mean(all_positions, axis=0) if all_positions else None

            clusters.append(TrajectoryCluster(
                cluster_id=int(cluster_id),
                trajectories=cluster_trajs,
                centroid=centroid,
                size=len(cluster_trajs)
            ))

        self.trajectory_clusters = clusters
        return clusters

    def cluster_trajectories_kmeans(
        self,
        trajectories: List[Trajectory],
        n_clusters: int = 3
    ) -> List[TrajectoryCluster]:
        """
        Cluster trajectories using k-means.

        Scenario: Cluster trajectories with k-means

        Args:
            trajectories: List of trajectories to cluster
            n_clusters: Number of clusters

        Returns:
            List of trajectory clusters
        """
        try:
            from sklearn.cluster import KMeans
        except ImportError:
            print("Warning: scikit-learn not available. Returning single cluster.")
            return [TrajectoryCluster(
                cluster_id=0,
                trajectories=trajectories,
                centroid=None,
                size=len(trajectories)
            )]

        if len(trajectories) < n_clusters:
            n_clusters = len(trajectories)

        # Extract features
        features = []
        for traj in trajectories:
            if len(traj.points) < 2:
                continue

            start_pos = traj.points[0].position
            end_pos = traj.points[-1].position
            length = traj.total_length
            duration = (traj.points[-1].timestamp - traj.points[0].timestamp).total_seconds()

            feature = np.concatenate([start_pos, end_pos, [length, duration]])
            features.append(feature)

        features = np.array(features)

        # Apply k-means
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit(features)
        labels = kmeans.labels_

        # Group trajectories by cluster
        clusters_dict = {}
        for idx, label in enumerate(labels):
            if label not in clusters_dict:
                clusters_dict[label] = []
            clusters_dict[label].append(trajectories[idx])

        # Create TrajectoryCluster objects
        clusters = []
        for cluster_id, cluster_trajs in clusters_dict.items():
            # Calculate centroid
            all_positions = []
            for traj in cluster_trajs:
                for point in traj.points:
                    all_positions.append(point.position)

            centroid = np.mean(all_positions, axis=0) if all_positions else None

            clusters.append(TrajectoryCluster(
                cluster_id=int(cluster_id),
                trajectories=cluster_trajs,
                centroid=centroid,
                size=len(cluster_trajs)
            ))

        self.trajectory_clusters = clusters
        return clusters

    def identify_common_patterns(
        self,
        clusters: List[TrajectoryCluster]
    ) -> Dict[int, Dict[str, Any]]:
        """
        Identify common movement patterns from clusters.

        Scenario: Identify common movement patterns

        Args:
            clusters: List of trajectory clusters

        Returns:
            Dictionary mapping cluster_id to pattern statistics
        """
        patterns = {}

        for cluster in clusters:
            if cluster.size == 0:
                continue

            # Calculate pattern statistics
            lengths = [traj.total_length for traj in cluster.trajectories]
            durations = [
                (traj.points[-1].timestamp - traj.points[0].timestamp).total_seconds()
                for traj in cluster.trajectories if len(traj.points) >= 2
            ]

            # Calculate average speed for each trajectory
            speeds = []
            for traj in cluster.trajectories:
                if len(traj.points) >= 2:
                    duration = (traj.points[-1].timestamp - traj.points[0].timestamp).total_seconds()
                    if duration > 0:
                        speeds.append(traj.total_length / duration)

            patterns[cluster.cluster_id] = {
                'size': cluster.size,
                'avg_length': float(np.mean(lengths)) if lengths else 0.0,
                'std_length': float(np.std(lengths)) if lengths else 0.0,
                'avg_duration': float(np.mean(durations)) if durations else 0.0,
                'avg_speed': float(np.mean(speeds)) if speeds else 0.0,
                'centroid': cluster.centroid.tolist() if cluster.centroid is not None else None,
                'frequency': cluster.size / sum(c.size for c in clusters) if clusters else 0.0
            }

        return patterns

    # ========================================================================
    # POSITION PREDICTION IMPLEMENTATION
    # ========================================================================

    def predict_position_linear(
        self,
        trajectory: Trajectory,
        prediction_time: float
    ) -> np.ndarray:
        """
        Predict future position using linear extrapolation.

        Scenario: Predict future position with linear extrapolation

        Args:
            trajectory: Historical trajectory
            prediction_time: Time into future in seconds

        Returns:
            Predicted 3D position
        """
        if len(trajectory.points) < 2:
            return trajectory.points[-1].position if trajectory.points else np.zeros(3)

        # Use last point's velocity if available
        last_point = trajectory.points[-1]

        if last_point.velocity is not None:
            velocity = last_point.velocity
        else:
            # Calculate velocity from last two points
            prev_point = trajectory.points[-2]
            time_delta = (last_point.timestamp - prev_point.timestamp).total_seconds()
            if time_delta > 0:
                displacement = last_point.position - prev_point.position
                velocity = displacement / time_delta
            else:
                velocity = np.zeros(3)

        # Linear extrapolation: future_pos = current_pos + velocity × delta_t
        future_position = last_point.position + velocity * prediction_time

        return future_position

    def predict_position_lstm(
        self,
        trajectory: Trajectory,
        prediction_steps: int = 10
    ) -> List[np.ndarray]:
        """
        Predict future positions using LSTM/GRU.

        Scenario: Predict future position with LSTM/GRU

        Note: This is a placeholder implementation. For production, you would
        need to train an LSTM/GRU model with historical trajectory data.

        Args:
            trajectory: Historical trajectory
            prediction_steps: Number of future steps to predict

        Returns:
            List of predicted 3D positions
        """
        # For now, use linear extrapolation as fallback
        # In production, this would use a trained LSTM/GRU model

        predictions = []
        for step in range(1, prediction_steps + 1):
            prediction_time = step * self.frame_time
            predicted_pos = self.predict_position_linear(trajectory, prediction_time)
            predictions.append(predicted_pos)

        return predictions

    # ========================================================================
    # CAMERA POSE & ODOMETRY IMPLEMENTATION
    # ========================================================================

    def estimate_camera_pose(
        self,
        object_points: np.ndarray,
        image_points: np.ndarray,
        camera_matrix: np.ndarray,
        dist_coeffs: np.ndarray
    ) -> CameraPose:
        """
        Estimate 6DOF camera pose using PnP.

        Scenario: Estimate camera pose (6DOF)

        Args:
            object_points: 3D points in world coordinates (N, 3)
            image_points: 2D points in image coordinates (N, 2)
            camera_matrix: Camera intrinsic matrix (3, 3)
            dist_coeffs: Distortion coefficients

        Returns:
            CameraPose with 6DOF pose
        """
        # Solve PnP with RANSAC
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            object_points.astype(np.float32),
            image_points.astype(np.float32),
            camera_matrix,
            dist_coeffs
        )

        if not success:
            # Return identity pose if estimation failed
            return CameraPose(
                position=np.zeros(3),
                rotation_matrix=np.eye(3),
                translation_vector=np.zeros(3),
                euler_angles=np.zeros(3),
                timestamp=datetime.now(),
                confidence=0.0
            )

        # Convert rotation vector to rotation matrix
        rotation_matrix, _ = cv2.Rodrigues(rvec)

        # Calculate Euler angles
        euler_angles = self.calculate_euler_angles(rotation_matrix)

        # Calculate position (camera center in world coordinates)
        position = -rotation_matrix.T @ tvec.flatten()

        # Calculate confidence (ratio of inliers)
        confidence = len(inliers) / len(object_points) if inliers is not None else 0.0

        pose = CameraPose(
            position=position,
            rotation_matrix=rotation_matrix,
            translation_vector=tvec.flatten(),
            euler_angles=euler_angles,
            timestamp=datetime.now(),
            confidence=float(confidence)
        )

        self.camera_poses.append(pose)
        return pose

    def calculate_visual_odometry(
        self,
        prev_frame: np.ndarray,
        curr_frame: np.ndarray,
        camera_matrix: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate visual odometry between frames.

        Scenario: Calculate visual odometry

        Args:
            prev_frame: Previous frame
            curr_frame: Current frame
            camera_matrix: Camera intrinsic matrix

        Returns:
            Tuple of (rotation_matrix, translation_vector)
        """
        # Convert to grayscale if needed
        if len(prev_frame.shape) == 3:
            prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        else:
            prev_gray = prev_frame

        if len(curr_frame.shape) == 3:
            curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        else:
            curr_gray = curr_frame

        # Detect features in previous frame
        orb = cv2.ORB_create(nfeatures=1000)
        kp1, des1 = orb.detectAndCompute(prev_gray, None)

        # Detect features in current frame
        kp2, des2 = orb.detectAndCompute(curr_gray, None)

        # Match features
        if des1 is None or des2 is None or len(kp1) < 5 or len(kp2) < 5:
            # Not enough features, return identity
            return np.eye(3), np.zeros(3)

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)

        # Keep best matches
        good_matches = matches[:min(50, len(matches))]

        if len(good_matches) < 5:
            return np.eye(3), np.zeros(3)

        # Extract matched points
        pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])

        # Calculate essential matrix
        E, mask = cv2.findEssentialMat(
            pts1, pts2, camera_matrix, method=cv2.RANSAC, prob=0.999, threshold=1.0
        )

        if E is None:
            return np.eye(3), np.zeros(3)

        # Recover pose
        _, R, t, mask = cv2.recoverPose(E, pts1, pts2, camera_matrix)

        return R, t.flatten()

    def calculate_euler_angles(
        self,
        rotation_matrix: np.ndarray
    ) -> np.ndarray:
        """
        Calculate Euler angles from rotation matrix.

        Scenario: Calculate Euler angles from rotation matrix

        Args:
            rotation_matrix: 3x3 rotation matrix

        Returns:
            Euler angles (roll, pitch, yaw) in radians
        """
        # Extract Euler angles using ZYX convention (yaw-pitch-roll)
        sy = np.sqrt(rotation_matrix[0, 0]**2 + rotation_matrix[1, 0]**2)

        singular = sy < 1e-6

        if not singular:
            roll = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
            pitch = np.arctan2(-rotation_matrix[2, 0], sy)
            yaw = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
        else:
            roll = np.arctan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
            pitch = np.arctan2(-rotation_matrix[2, 0], sy)
            yaw = 0

        return np.array([roll, pitch, yaw])

    # ========================================================================
    # REPORT GENERATION IMPLEMENTATION
    # ========================================================================

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
        if timestamp is None:
            timestamp = datetime.now()

        # Get trajectory
        trajectory = self.get_trajectory(animal_id)
        if trajectory is None:
            return None

        # Calculate path metrics
        path_metrics = self.calculate_path_metrics(trajectory)

        # Predict future positions (next 5 seconds)
        predicted_positions = []
        for t in range(1, 6):
            pred_pos = self.predict_position_linear(trajectory, t * 1.0)
            predicted_positions.append(pred_pos)

        # Get cluster assignment if clustering has been performed
        cluster_assignment = None
        for cluster in self.trajectory_clusters:
            for traj in cluster.trajectories:
                if traj.animal_id == animal_id:
                    cluster_assignment = cluster.cluster_id
                    break

        return TrajectoryReport(
            animal_id=animal_id,
            animal_type=self.animal_type,
            trajectory=trajectory,
            path_metrics=path_metrics,
            predicted_positions=predicted_positions,
            cluster_assignment=cluster_assignment,
            timestamp=timestamp
        )

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
        file_exists = Path(csv_path).exists()

        with open(csv_path, 'a', newline='') as csvfile:
            fieldnames = [
                'animal_id', 'frame_number', 'timestamp',
                'x', 'y', 'z',
                'vx', 'vy', 'vz',
                'speed'
            ]

            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            if not file_exists:
                writer.writeheader()

            for point in trajectory.points:
                speed = np.linalg.norm(point.velocity) if point.velocity is not None else 0.0

                row = {
                    'animal_id': trajectory.animal_id,
                    'frame_number': point.frame_number,
                    'timestamp': point.timestamp.isoformat(),
                    'x': point.position[0],
                    'y': point.position[1],
                    'z': point.position[2],
                    'vx': point.velocity[0] if point.velocity is not None else 0.0,
                    'vy': point.velocity[1] if point.velocity is not None else 0.0,
                    'vz': point.velocity[2] if point.velocity is not None else 0.0,
                    'speed': speed
                }

                writer.writerow(row)

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
        file_exists = Path(csv_path).exists()

        with open(csv_path, 'a', newline='') as csvfile:
            fieldnames = [
                'timestamp', 'animal_id', 'animal_type',
                'total_length_m', 'average_speed_m_s', 'max_speed_m_s',
                'total_duration_s', 'sharp_turns_count',
                'cluster_id', 'num_points', 'smoothed'
            ]

            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            if not file_exists:
                writer.writeheader()

            row = {
                'timestamp': report.timestamp.isoformat(),
                'animal_id': report.animal_id,
                'animal_type': report.animal_type,
                'total_length_m': report.path_metrics.total_length_m,
                'average_speed_m_s': report.path_metrics.average_speed_m_s,
                'max_speed_m_s': report.path_metrics.max_speed_m_s,
                'total_duration_s': report.path_metrics.total_duration_s,
                'sharp_turns_count': report.path_metrics.sharp_turns_count,
                'cluster_id': report.cluster_assignment if report.cluster_assignment is not None else -1,
                'num_points': len(report.trajectory.points),
                'smoothed': report.trajectory.smoothed
            }

            writer.writerow(row)
