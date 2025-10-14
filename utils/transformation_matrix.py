"""
Transformation Matrix Operations for Multi-Camera Calibration

This module provides utilities for computing and applying 4x4 homogeneous
transformation matrices between camera coordinate systems.

Supports:
- Rotation matrices (3x3) and translation vectors (3x1)
- 4x4 homogeneous transformations
- Point and point cloud transformations
- Matrix averaging and validation
- Pose composition and inversion

Part of Scenario 13 (Multi-Vendor Multi-Camera Integration)
"""

import numpy as np
from typing import Tuple, List, Optional, Dict
from dataclasses import dataclass
import logging
from scipy.spatial.transform import Rotation

logger = logging.getLogger(__name__)


@dataclass
class CameraPose:
    """Camera pose in global coordinate system"""
    camera_id: str
    transformation_matrix: np.ndarray  # 4x4 homogeneous transformation
    rotation_matrix: np.ndarray  # 3x3 rotation
    translation_vector: np.ndarray  # 3x1 translation
    euler_angles: Tuple[float, float, float]  # (roll, pitch, yaw) in radians
    position: Tuple[float, float, float]  # (x, y, z) in meters

    def to_dict(self) -> Dict:
        """Convert to JSON-serializable dictionary"""
        return {
            "camera_id": self.camera_id,
            "transformation_matrix": self.transformation_matrix.tolist(),
            "rotation_matrix": self.rotation_matrix.tolist(),
            "translation_vector": self.translation_vector.flatten().tolist(),
            "euler_angles_rad": {
                "roll": float(self.euler_angles[0]),
                "pitch": float(self.euler_angles[1]),
                "yaw": float(self.euler_angles[2])
            },
            "euler_angles_deg": {
                "roll": float(np.degrees(self.euler_angles[0])),
                "pitch": float(np.degrees(self.euler_angles[1])),
                "yaw": float(np.degrees(self.euler_angles[2]))
            },
            "position": {
                "x": float(self.position[0]),
                "y": float(self.position[1]),
                "z": float(self.position[2])
            }
        }


class TransformationMatrix:
    """
    Utilities for 4x4 homogeneous transformation matrices.

    Coordinate System Convention:
    - Right-handed coordinate system
    - X: right
    - Y: down
    - Z: forward (camera viewing direction)

    Transformation Matrix Format (4x4):
        [[R11, R12, R13, tx],
         [R21, R22, R23, ty],
         [R31, R32, R33, tz],
         [0,   0,   0,   1]]

    Where:
    - R: 3x3 rotation matrix
    - t: 3x1 translation vector
    """

    @staticmethod
    def create_identity() -> np.ndarray:
        """
        Create 4x4 identity transformation matrix.

        Returns:
            4x4 identity matrix
        """
        return np.eye(4, dtype=np.float64)

    @staticmethod
    def create_from_rotation_translation(rotation: np.ndarray,
                                        translation: np.ndarray) -> np.ndarray:
        """
        Create 4x4 transformation matrix from rotation and translation.

        Args:
            rotation: 3x3 rotation matrix or 3x1 rotation vector (Rodrigues)
            translation: 3x1 translation vector

        Returns:
            4x4 homogeneous transformation matrix
        """
        T = np.eye(4, dtype=np.float64)

        # Handle rotation vector (Rodrigues format)
        if rotation.shape == (3, 1) or (rotation.shape == (3,) and len(rotation.shape) == 1):
            rotation, _ = cv2.Rodrigues(rotation)

        T[:3, :3] = rotation
        T[:3, 3] = translation.flatten()
        return T

    @staticmethod
    def decompose(transformation: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Decompose 4x4 transformation into rotation and translation.

        Args:
            transformation: 4x4 transformation matrix

        Returns:
            (rotation_matrix, translation_vector)
            rotation_matrix: 3x3
            translation_vector: 3x1
        """
        rotation = transformation[:3, :3].copy()
        translation = transformation[:3, 3].copy().reshape(3, 1)
        return rotation, translation

    @staticmethod
    def invert(transformation: np.ndarray) -> np.ndarray:
        """
        Compute inverse of 4x4 transformation matrix.

        For rigid body transformation:
        T^-1 = [[R^T, -R^T * t],
                [0,    1      ]]

        Args:
            transformation: 4x4 transformation matrix

        Returns:
            4x4 inverse transformation matrix
        """
        R, t = TransformationMatrix.decompose(transformation)

        T_inv = np.eye(4, dtype=np.float64)
        T_inv[:3, :3] = R.T
        T_inv[:3, 3] = (-R.T @ t).flatten()

        return T_inv

    @staticmethod
    def compose(T1: np.ndarray, T2: np.ndarray) -> np.ndarray:
        """
        Compose two transformations: T_result = T1 @ T2

        Applies T2 first, then T1.

        Args:
            T1: First transformation (4x4)
            T2: Second transformation (4x4)

        Returns:
            Composed transformation (4x4)
        """
        return T1 @ T2

    @staticmethod
    def transform_from_to(T_A_to_ref: np.ndarray,
                         T_B_to_ref: np.ndarray) -> np.ndarray:
        """
        Compute transformation from frame A to frame B.

        Given:
        - T_A_to_ref: Transformation from A to reference frame
        - T_B_to_ref: Transformation from B to reference frame

        Returns:
        - T_A_to_B: Transformation from A to B

        Formula: T_A_to_B = inv(T_B_to_ref) @ T_A_to_ref

        Args:
            T_A_to_ref: 4x4 transformation from A to reference
            T_B_to_ref: 4x4 transformation from B to reference

        Returns:
            4x4 transformation from A to B
        """
        T_ref_to_B = TransformationMatrix.invert(T_B_to_ref)
        T_A_to_B = T_ref_to_B @ T_A_to_ref
        return T_A_to_B

    @staticmethod
    def validate(transformation: np.ndarray,
                tolerance: float = 1e-6) -> Tuple[bool, str]:
        """
        Validate 4x4 transformation matrix.

        Checks:
        1. Shape is 4x4
        2. Bottom row is [0, 0, 0, 1]
        3. Rotation part is orthogonal (R^T @ R = I)
        4. Determinant of rotation is ~1 (proper rotation)

        Args:
            transformation: 4x4 matrix to validate
            tolerance: Numerical tolerance

        Returns:
            (is_valid, message)
        """
        # Check shape
        if transformation.shape != (4, 4):
            return False, f"Shape is {transformation.shape}, expected (4, 4)"

        # Check bottom row
        bottom_row = transformation[3, :]
        expected_bottom = np.array([0, 0, 0, 1])
        if not np.allclose(bottom_row, expected_bottom, atol=tolerance):
            return False, f"Bottom row is {bottom_row}, expected [0, 0, 0, 1]"

        # Extract rotation
        R = transformation[:3, :3]

        # Check orthogonality (R^T @ R = I)
        RTR = R.T @ R
        I = np.eye(3)
        if not np.allclose(RTR, I, atol=tolerance):
            return False, "Rotation matrix is not orthogonal"

        # Check determinant (should be +1 for proper rotation)
        det = np.linalg.det(R)
        if not np.isclose(det, 1.0, atol=tolerance):
            return False, f"Rotation determinant is {det:.6f}, expected 1.0"

        return True, "Valid transformation matrix"

    @staticmethod
    def rotation_matrix_to_euler(R: np.ndarray,
                                 convention: str = 'xyz') -> Tuple[float, float, float]:
        """
        Convert rotation matrix to Euler angles.

        Args:
            R: 3x3 rotation matrix
            convention: Euler angle convention (default: 'xyz')

        Returns:
            (roll, pitch, yaw) in radians
        """
        rot = Rotation.from_matrix(R)
        euler = rot.as_euler(convention, degrees=False)
        return tuple(euler)

    @staticmethod
    def euler_to_rotation_matrix(roll: float, pitch: float, yaw: float,
                                 convention: str = 'xyz') -> np.ndarray:
        """
        Convert Euler angles to rotation matrix.

        Args:
            roll: Roll angle in radians
            pitch: Pitch angle in radians
            yaw: Yaw angle in radians
            convention: Euler angle convention (default: 'xyz')

        Returns:
            3x3 rotation matrix
        """
        rot = Rotation.from_euler(convention, [roll, pitch, yaw], degrees=False)
        return rot.as_matrix()

    @staticmethod
    def average_transformations(transformations: List[np.ndarray]) -> np.ndarray:
        """
        Average multiple transformation matrices.

        Uses:
        - Translation: arithmetic mean
        - Rotation: quaternion averaging

        Args:
            transformations: List of 4x4 transformation matrices

        Returns:
            Averaged 4x4 transformation matrix
        """
        if len(transformations) == 0:
            raise ValueError("Cannot average empty list of transformations")

        if len(transformations) == 1:
            return transformations[0].copy()

        # Extract rotations and translations
        rotations = []
        translations = []

        for T in transformations:
            R, t = TransformationMatrix.decompose(T)
            rotations.append(R)
            translations.append(t.flatten())

        # Average translations (simple arithmetic mean)
        avg_translation = np.mean(translations, axis=0).reshape(3, 1)

        # Average rotations using quaternions
        quaternions = [Rotation.from_matrix(R).as_quat() for R in rotations]
        quaternions = np.array(quaternions)

        # Ensure all quaternions are in same hemisphere (avoid sign ambiguity)
        for i in range(1, len(quaternions)):
            if np.dot(quaternions[0], quaternions[i]) < 0:
                quaternions[i] = -quaternions[i]

        # Average quaternions
        avg_quat = np.mean(quaternions, axis=0)
        avg_quat /= np.linalg.norm(avg_quat)  # Normalize

        # Convert back to rotation matrix
        avg_rotation = Rotation.from_quat(avg_quat).as_matrix()

        # Create averaged transformation
        T_avg = TransformationMatrix.create_from_rotation_translation(
            avg_rotation, avg_translation
        )

        return T_avg

    @staticmethod
    def create_camera_pose(camera_id: str,
                          transformation: np.ndarray) -> CameraPose:
        """
        Create CameraPose object from transformation matrix.

        Args:
            camera_id: Camera identifier
            transformation: 4x4 transformation matrix

        Returns:
            CameraPose object
        """
        R, t = TransformationMatrix.decompose(transformation)
        euler = TransformationMatrix.rotation_matrix_to_euler(R)
        position = (t[0, 0], t[1, 0], t[2, 0])

        return CameraPose(
            camera_id=camera_id,
            transformation_matrix=transformation.copy(),
            rotation_matrix=R.copy(),
            translation_vector=t.copy(),
            euler_angles=euler,
            position=position
        )

    @staticmethod
    def transform_point(point: np.ndarray,
                       transformation: np.ndarray) -> np.ndarray:
        """
        Transform a single 3D point using 4x4 transformation matrix.

        Args:
            point: 3D point as (3,) or (3, 1) array
            transformation: 4x4 transformation matrix

        Returns:
            Transformed 3D point as (3,) array
        """
        # Convert to homogeneous coordinates
        point_flat = point.flatten()
        point_homogeneous = np.append(point_flat, 1.0)

        # Apply transformation
        transformed = transformation @ point_homogeneous

        # Convert back to 3D
        return transformed[:3]

    @staticmethod
    def transform_points(points: np.ndarray,
                        transformation: np.ndarray) -> np.ndarray:
        """
        Transform multiple 3D points using 4x4 transformation matrix.

        Args:
            points: Nx3 array of points
            transformation: 4x4 transformation matrix

        Returns:
            Nx3 array of transformed points
        """
        N = points.shape[0]

        # Convert to homogeneous coordinates (Nx4)
        points_homogeneous = np.hstack([points, np.ones((N, 1))])

        # Apply transformation (4x4) @ (4xN) = (4xN)
        transformed = (transformation @ points_homogeneous.T).T

        # Convert back to 3D (Nx3)
        return transformed[:, :3]

    @staticmethod
    def compute_baseline_distance(T_A_to_B: np.ndarray) -> float:
        """
        Compute baseline distance between two cameras.

        Args:
            T_A_to_B: Transformation from camera A to camera B

        Returns:
            Baseline distance in meters
        """
        _, t = TransformationMatrix.decompose(T_A_to_B)
        return np.linalg.norm(t)

    @staticmethod
    def get_baseline_vector(T_A_to_B: np.ndarray) -> np.ndarray:
        """
        Get baseline vector from camera A to camera B.

        Args:
            T_A_to_B: Transformation from camera A to camera B

        Returns:
            3D baseline vector
        """
        _, t = TransformationMatrix.decompose(T_A_to_B)
        return t.flatten()


# Import cv2 for Rodrigues conversion (optional, only if needed)
try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False
    logger.warning("OpenCV not available, Rodrigues conversion disabled")


# Convenience functions
def create_identity_pose(camera_id: str = "reference") -> CameraPose:
    """Create identity pose (origin of coordinate system)"""
    T = TransformationMatrix.create_identity()
    return TransformationMatrix.create_camera_pose(camera_id, T)


def validate_transformation_pair(T_forward: np.ndarray,
                                 T_inverse: np.ndarray,
                                 tolerance: float = 1e-5) -> bool:
    """
    Validate that T_forward and T_inverse are true inverses.

    Args:
        T_forward: Forward transformation
        T_inverse: Inverse transformation
        tolerance: Numerical tolerance

    Returns:
        True if T_forward @ T_inverse ≈ I
    """
    product = T_forward @ T_inverse
    identity = np.eye(4)
    return np.allclose(product, identity, atol=tolerance)
