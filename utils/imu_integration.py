"""
IMU Integration for RealSense D455i

This module integrates IMU data (accelerometer and gyroscope) from the
RealSense D455i with visual data for improved multi-camera calibration
and pose estimation.

Part of Scenario 13 (Multi-Vendor Multi-Camera Integration)

Implements:
- Scenario 12.3: Integrate IMU data from RealSense D455i

IMU capabilities:
- 6-DoF IMU (3-axis accelerometer + 3-axis gyroscope)
- Gravity vector estimation
- Camera orientation tracking
- Motion compensation
"""

import numpy as np
from typing import Tuple, Optional, List
from dataclasses import dataclass
from scipy.spatial.transform import Rotation
import logging

logger = logging.getLogger(__name__)


@dataclass
class IMUSample:
    """Single IMU measurement"""
    timestamp: float  # Epoch timestamp
    acceleration: np.ndarray  # 3D acceleration (m/s²) in camera frame
    angular_velocity: np.ndarray  # 3D angular velocity (rad/s) in camera frame


@dataclass
class IMUPose:
    """Camera pose estimated from IMU"""
    timestamp: float
    orientation: np.ndarray  # 3x3 rotation matrix
    orientation_quaternion: np.ndarray  # Quaternion (w, x, y, z)
    euler_angles: np.ndarray  # Roll, pitch, yaw (radians)
    gravity_vector: np.ndarray  # Estimated gravity direction
    confidence: float  # 0-1, confidence in estimate


@dataclass
class IMUCalibration:
    """IMU calibration parameters"""
    accel_bias: np.ndarray  # Accelerometer bias (3,)
    gyro_bias: np.ndarray  # Gyroscope bias (3,)
    accel_scale: np.ndarray  # Accelerometer scale factors (3,)
    gyro_scale: np.ndarray  # Gyroscope scale factors (3,)
    R_imu_to_camera: np.ndarray  # 3x3 rotation from IMU to camera frame


class IMUIntegrator:
    """
    Integrate IMU data for camera orientation tracking.

    Implements Scenario 12.3: Integrate IMU data from RealSense D455i

    Features:
    - Gravity vector estimation
    - Orientation tracking
    - Bias estimation
    - Complementary filter for sensor fusion
    """

    # Earth gravity magnitude (m/s²)
    GRAVITY = 9.81

    def __init__(self, imu_calibration: Optional[IMUCalibration] = None):
        """
        Initialize IMU integrator.

        Args:
            imu_calibration: Optional IMU calibration (uses defaults if None)
        """
        if imu_calibration is None:
            # Default calibration (identity)
            imu_calibration = IMUCalibration(
                accel_bias=np.zeros(3),
                gyro_bias=np.zeros(3),
                accel_scale=np.ones(3),
                gyro_scale=np.ones(3),
                R_imu_to_camera=np.eye(3)
            )

        self.calibration = imu_calibration

        # State
        self.current_orientation = Rotation.identity()
        self.gravity_vector = np.array([0, 0, -1])  # Initial guess (Z-down)

        # Complementary filter parameters
        self.alpha = 0.98  # Gyro trust (0.98 = trust gyro 98%)

        logger.info("Initialized IMUIntegrator")

    def calibrate_static(self, imu_samples: List[IMUSample], duration_s: float = 5.0):
        """
        Calibrate IMU biases from static measurements.

        Args:
            imu_samples: List of IMU samples while camera is static
            duration_s: Expected duration of static period
        """
        if not imu_samples:
            logger.warning("No IMU samples for calibration")
            return

        # Compute mean acceleration and angular velocity
        accels = np.array([s.acceleration for s in imu_samples])
        gyros = np.array([s.angular_velocity for s in imu_samples])

        # Gyro bias = mean angular velocity (should be zero when static)
        self.calibration.gyro_bias = np.mean(gyros, axis=0)

        # Accel bias = mean acceleration - gravity
        mean_accel = np.mean(accels, axis=0)
        gravity_magnitude = np.linalg.norm(mean_accel)

        # Gravity vector in sensor frame
        self.gravity_vector = mean_accel / gravity_magnitude

        # Accel bias (assuming gravity is well-measured)
        expected_accel = self.gravity_vector * self.GRAVITY
        self.calibration.accel_bias = mean_accel - expected_accel

        logger.info(
            f"IMU calibration complete: "
            f"gyro_bias={self.calibration.gyro_bias}, "
            f"gravity_mag={gravity_magnitude:.2f} m/s²"
        )

    def process_imu_sample(self,
                          imu_sample: IMUSample,
                          dt: float) -> IMUPose:
        """
        Process single IMU sample and update orientation estimate.

        Uses complementary filter combining gyroscope (high-frequency)
        and accelerometer (low-frequency) measurements.

        Args:
            imu_sample: IMU measurement
            dt: Time step (seconds)

        Returns:
            IMUPose with updated orientation
        """
        # Apply calibration
        accel_corrected = (imu_sample.acceleration - self.calibration.accel_bias) * \
                         self.calibration.accel_scale
        gyro_corrected = (imu_sample.angular_velocity - self.calibration.gyro_bias) * \
                        self.calibration.gyro_scale

        # === Gyroscope integration ===
        # Predict orientation from gyroscope
        angle_increment = gyro_corrected * dt
        rotation_increment = Rotation.from_rotvec(angle_increment)
        orientation_pred = self.current_orientation * rotation_increment

        # === Accelerometer correction ===
        # Use accelerometer to correct orientation (assume measuring gravity)
        accel_magnitude = np.linalg.norm(accel_corrected)

        # Check if accelerometer is measuring primarily gravity
        # (magnitude close to g, not much linear acceleration)
        is_static = np.abs(accel_magnitude - self.GRAVITY) < 0.5

        if is_static:
            # Estimate gravity direction from accelerometer
            measured_gravity = -accel_corrected / accel_magnitude

            # Expected gravity in world frame
            expected_gravity = np.array([0, 0, -1])

            # Compute correction rotation
            # This aligns measured gravity with expected gravity
            correction_axis = np.cross(
                orientation_pred.apply(measured_gravity),
                expected_gravity
            )

            correction_angle = np.arcsin(
                np.clip(np.linalg.norm(correction_axis), -1.0, 1.0)
            )

            if correction_angle > 1e-6:
                correction_axis /= np.linalg.norm(correction_axis)
                correction_rotation = Rotation.from_rotvec(
                    correction_axis * correction_angle * (1 - self.alpha)
                )

                # Apply correction
                orientation_corrected = correction_rotation * orientation_pred
            else:
                orientation_corrected = orientation_pred

            # Update gravity vector estimate
            self.gravity_vector = measured_gravity

            confidence = 0.9
        else:
            # Moving - trust gyroscope only
            orientation_corrected = orientation_pred
            confidence = 0.5

        # Update state
        self.current_orientation = orientation_corrected

        # Convert to various representations
        rotation_matrix = orientation_corrected.as_matrix()
        quaternion = orientation_corrected.as_quat()  # (x, y, z, w) in scipy
        # Convert to (w, x, y, z)
        quaternion = np.array([quaternion[3], quaternion[0], quaternion[1], quaternion[2]])

        euler_angles = orientation_corrected.as_euler('xyz', degrees=False)

        return IMUPose(
            timestamp=imu_sample.timestamp,
            orientation=rotation_matrix,
            orientation_quaternion=quaternion,
            euler_angles=euler_angles,
            gravity_vector=self.gravity_vector,
            confidence=confidence
        )

    def process_imu_sequence(self, imu_samples: List[IMUSample]) -> List[IMUPose]:
        """
        Process sequence of IMU samples.

        Args:
            imu_samples: List of IMU samples (must be time-ordered)

        Returns:
            List of IMUPose estimates
        """
        if not imu_samples:
            return []

        poses = []

        for i, sample in enumerate(imu_samples):
            # Compute time step
            if i == 0:
                dt = 0.01  # Assume 100Hz
            else:
                dt = sample.timestamp - imu_samples[i-1].timestamp

            # Process
            pose = self.process_imu_sample(sample, dt)
            poses.append(pose)

        logger.info(f"Processed {len(poses)} IMU samples")

        return poses

    def get_camera_to_world_transform(self) -> np.ndarray:
        """
        Get current camera-to-world transformation matrix.

        Returns:
            4x4 transformation matrix
        """
        # Camera orientation in world frame
        R_world_to_camera = self.current_orientation.as_matrix()

        # For now, assume no translation (orientation only)
        T = np.eye(4)
        T[:3, :3] = R_world_to_camera

        return T

    def align_coordinate_frame_with_gravity(self) -> np.ndarray:
        """
        Compute transformation to align coordinate frame with gravity.

        Useful for ensuring Z-axis points up (against gravity).

        Returns:
            4x4 transformation matrix
        """
        # Current gravity in camera frame
        gravity_camera = self.gravity_vector

        # Desired gravity direction (Z-up convention: gravity = [0, 0, -1])
        gravity_world = np.array([0, 0, -1])

        # Compute rotation to align
        rotation_axis = np.cross(gravity_camera, gravity_world)
        rotation_angle = np.arccos(
            np.clip(np.dot(gravity_camera, gravity_world), -1.0, 1.0)
        )

        if np.linalg.norm(rotation_axis) > 1e-6:
            rotation_axis /= np.linalg.norm(rotation_axis)
            R = Rotation.from_rotvec(rotation_axis * rotation_angle)
        else:
            R = Rotation.identity()

        T = np.eye(4)
        T[:3, :3] = R.as_matrix()

        return T


class IMUVisualFusion:
    """
    Fuse IMU and visual data for improved pose estimation.

    Provides complementary information:
    - IMU: High-frequency orientation, no drift in gravity direction
    - Visual: Absolute position, long-term stability
    """

    def __init__(self, imu_integrator: IMUIntegrator):
        """
        Initialize IMU-visual fusion.

        Args:
            imu_integrator: IMUIntegrator instance
        """
        self.imu_integrator = imu_integrator
        logger.info("Initialized IMUVisualFusion")

    def compensate_camera_motion(self,
                                point_cloud: np.ndarray,
                                imu_pose_start: IMUPose,
                                imu_pose_end: IMUPose) -> np.ndarray:
        """
        Compensate point cloud for camera motion during capture.

        Args:
            point_cloud: Point cloud (N, 3)
            imu_pose_start: Camera pose at capture start
            imu_pose_end: Camera pose at capture end

        Returns:
            Motion-compensated point cloud
        """
        # Compute relative rotation
        R_start = imu_pose_start.orientation
        R_end = imu_pose_end.orientation

        # Relative rotation from start to end
        R_relative = R_end @ R_start.T

        # Compensate: rotate points by inverse of motion
        R_compensation = R_relative.T

        # Apply compensation
        compensated_points = (R_compensation @ point_cloud.T).T

        logger.debug(
            f"Applied motion compensation: "
            f"rotation = {np.linalg.norm(Rotation.from_matrix(R_relative).as_rotvec()):.3f} rad"
        )

        return compensated_points

    def refine_calibration_with_imu(self,
                                    visual_transforms: List[np.ndarray],
                                    imu_poses: List[IMUPose]) -> List[np.ndarray]:
        """
        Refine multi-camera calibration using IMU data.

        IMU provides absolute orientation reference (gravity).

        Args:
            visual_transforms: Visual-based transformation estimates
            imu_poses: Corresponding IMU pose estimates

        Returns:
            Refined transformations
        """
        if len(visual_transforms) != len(imu_poses):
            logger.warning("Mismatch in number of visual transforms and IMU poses")
            return visual_transforms

        refined_transforms = []

        for T_visual, imu_pose in zip(visual_transforms, imu_poses):
            # Extract visual rotation
            R_visual = T_visual[:3, :3]
            t_visual = T_visual[:3, 3]

            # Get IMU rotation
            R_imu = imu_pose.orientation

            # Fuse rotations (weighted average using quaternions)
            q_visual = Rotation.from_matrix(R_visual).as_quat()
            q_imu = Rotation.from_matrix(R_imu).as_quat()

            # Weighted average (70% visual, 30% IMU)
            # For quaternions, use SLERP
            alpha = 0.7
            q_fused = self._slerp_quaternions(q_visual, q_imu, alpha)

            R_fused = Rotation.from_quat(q_fused).as_matrix()

            # Create refined transform
            T_refined = np.eye(4)
            T_refined[:3, :3] = R_fused
            T_refined[:3, 3] = t_visual  # Keep visual translation

            refined_transforms.append(T_refined)

        logger.info(f"Refined {len(refined_transforms)} transformations with IMU")

        return refined_transforms

    def _slerp_quaternions(self,
                          q1: np.ndarray,
                          q2: np.ndarray,
                          t: float) -> np.ndarray:
        """
        Spherical linear interpolation between quaternions.

        Args:
            q1: First quaternion (x, y, z, w)
            q2: Second quaternion
            t: Interpolation parameter (0 = q1, 1 = q2)

        Returns:
            Interpolated quaternion
        """
        # Ensure quaternions are unit
        q1 = q1 / np.linalg.norm(q1)
        q2 = q2 / np.linalg.norm(q2)

        # Compute dot product
        dot = np.dot(q1, q2)

        # If dot < 0, negate q2 to take shorter path
        if dot < 0:
            q2 = -q2
            dot = -dot

        # Clamp to avoid numerical issues
        dot = np.clip(dot, -1.0, 1.0)

        # Compute angle
        theta = np.arccos(dot)

        # SLERP
        if theta < 1e-6:
            # Quaternions very close - linear interpolation
            return (1 - t) * q1 + t * q2
        else:
            sin_theta = np.sin(theta)
            w1 = np.sin((1 - t) * theta) / sin_theta
            w2 = np.sin(t * theta) / sin_theta
            return w1 * q1 + w2 * q2


# Convenience functions
def create_imu_sample(timestamp: float,
                     accel_x: float, accel_y: float, accel_z: float,
                     gyro_x: float, gyro_y: float, gyro_z: float) -> IMUSample:
    """
    Create IMU sample from raw measurements.

    Args:
        timestamp: Epoch timestamp
        accel_x, accel_y, accel_z: Acceleration (m/s²)
        gyro_x, gyro_y, gyro_z: Angular velocity (rad/s)

    Returns:
        IMUSample
    """
    return IMUSample(
        timestamp=timestamp,
        acceleration=np.array([accel_x, accel_y, accel_z]),
        angular_velocity=np.array([gyro_x, gyro_y, gyro_z])
    )
