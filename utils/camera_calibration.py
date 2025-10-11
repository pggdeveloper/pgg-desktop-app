"""RealSense camera calibration and intrinsics utilities (CPU-only)."""
from typing import Optional, Tuple, Dict, List
from pathlib import Path
import numpy as np
import cv2
import json
from datetime import datetime


class CameraCalibration:
    """
    Camera calibration and intrinsics management for RealSense cameras.

    Provides CPU-only calibration utilities including:
    - Camera intrinsics export (focal length, principal point, distortion)
    - Stereo calibration parameters (baseline, rotation, translation)
    - Extrinsics transformations (depth-to-RGB, IR-to-RGB, multi-camera)
    - Field of view calculations
    - Calibration verification (checkerboard detection, reprojection error)
    - Calibration drift detection
    """

    def __init__(self):
        """Initialize camera calibration manager."""
        try:
            import pyrealsense2 as rs
            self.rs = rs
        except ImportError:
            raise ImportError("pyrealsense2 is required for CameraCalibration")

        # Store calibration data
        self.intrinsics_cache: Dict[str, 'rs.intrinsics'] = {}
        self.extrinsics_cache: Dict[str, 'rs.extrinsics'] = {}
        self.stored_calibration: Optional[Dict] = None

    # ============================================================================
    # RGB CAMERA INTRINSICS
    # ============================================================================

    def get_rgb_intrinsics(self, profile) -> 'rs.intrinsics':
        """
        Get RGB camera intrinsics from stream profile.

        Args:
            profile: RealSense stream profile

        Returns:
            RGB camera intrinsics object
        """
        color_stream = profile.get_stream(self.rs.stream.color)
        intrinsics = color_stream.as_video_stream_profile().get_intrinsics()
        self.intrinsics_cache['rgb'] = intrinsics
        return intrinsics

    def export_rgb_focal_length(self, profile) -> Dict[str, float]:
        """
        Export RGB camera focal length.

        Args:
            profile: RealSense stream profile

        Returns:
            Dictionary with fx and fy in pixels
            {
                'fx': float,  # Horizontal focal length (pixels)
                'fy': float,  # Vertical focal length (pixels)
            }
        """
        intrinsics = self.get_rgb_intrinsics(profile)

        return {
            'fx': intrinsics.fx,
            'fy': intrinsics.fy,
        }

    def export_rgb_principal_point(self, profile) -> Dict[str, float]:
        """
        Export RGB camera principal point.

        The principal point is where the optical axis intersects the image plane.
        Typically near the image center.

        Args:
            profile: RealSense stream profile

        Returns:
            Dictionary with cx and cy in pixels
            {
                'cx': float,  # Horizontal principal point (pixels)
                'cy': float,  # Vertical principal point (pixels)
            }
        """
        intrinsics = self.get_rgb_intrinsics(profile)

        return {
            'cx': intrinsics.ppx,
            'cy': intrinsics.ppy,
        }

    def export_rgb_distortion_coefficients(self, profile) -> Dict[str, any]:
        """
        Export RGB camera distortion coefficients.

        Follows Brown-Conrady distortion model:
        - Radial distortion: k1, k2, k3, k4, k5, k6
        - Tangential distortion: p1, p2

        Args:
            profile: RealSense stream profile

        Returns:
            Dictionary with distortion coefficients
            {
                'model': str,  # Distortion model name
                'k1': float, 'k2': float, 'k3': float,
                'k4': float, 'k5': float, 'k6': float,
                'p1': float, 'p2': float,
                'coeffs': list  # All coefficients as array
            }
        """
        intrinsics = self.get_rgb_intrinsics(profile)

        # RealSense stores 5 distortion coefficients
        # Format depends on model, but typically: [k1, k2, p1, p2, k3]
        coeffs = intrinsics.coeffs

        return {
            'model': str(intrinsics.model),
            'k1': coeffs[0] if len(coeffs) > 0 else 0.0,
            'k2': coeffs[1] if len(coeffs) > 1 else 0.0,
            'p1': coeffs[2] if len(coeffs) > 2 else 0.0,
            'p2': coeffs[3] if len(coeffs) > 3 else 0.0,
            'k3': coeffs[4] if len(coeffs) > 4 else 0.0,
            'k4': 0.0,  # Extended Brown-Conrady (not in RealSense)
            'k5': 0.0,
            'k6': 0.0,
            'coeffs': list(coeffs),
        }

    def export_camera_lens_model(self, profile) -> Dict[str, any]:
        """
        Export camera lens distortion model.

        RealSense supports multiple distortion models:
        - Brown-Conrady (most common)
        - Inverse Brown-Conrady
        - Ftheta
        - Kannala-Brandt4

        Args:
            profile: RealSense stream profile

        Returns:
            Dictionary with lens model information
            {
                'model': str,  # Model name
                'model_id': int,  # Model enum value
                'parameters': list  # Model-specific parameters
            }
        """
        intrinsics = self.get_rgb_intrinsics(profile)

        model_names = {
            self.rs.distortion.none: 'None',
            self.rs.distortion.modified_brown_conrady: 'Modified Brown-Conrady',
            self.rs.distortion.inverse_brown_conrady: 'Inverse Brown-Conrady',
            self.rs.distortion.ftheta: 'F-Theta',
            self.rs.distortion.brown_conrady: 'Brown-Conrady',
            self.rs.distortion.kannala_brandt4: 'Kannala-Brandt4',
        }

        return {
            'model': model_names.get(intrinsics.model, 'Unknown'),
            'model_id': int(intrinsics.model),
            'parameters': list(intrinsics.coeffs),
        }

    def calculate_rgb_field_of_view(self, profile) -> Dict[str, float]:
        """
        Calculate RGB camera field of view.

        FOV is calculated from focal length and sensor dimensions:
        FOV_horizontal = 2 * atan(width / (2 * fx))
        FOV_vertical = 2 * atan(height / (2 * fy))

        Args:
            profile: RealSense stream profile

        Returns:
            Dictionary with FOV in degrees
            {
                'fov_horizontal': float,  # Horizontal FOV (degrees)
                'fov_vertical': float,    # Vertical FOV (degrees)
                'fov_diagonal': float,    # Diagonal FOV (degrees)
            }
        """
        intrinsics = self.get_rgb_intrinsics(profile)

        # Calculate horizontal FOV
        fov_h = 2 * np.arctan(intrinsics.width / (2 * intrinsics.fx))

        # Calculate vertical FOV
        fov_v = 2 * np.arctan(intrinsics.height / (2 * intrinsics.fy))

        # Calculate diagonal FOV
        diagonal = np.sqrt(intrinsics.width ** 2 + intrinsics.height ** 2)
        focal_diagonal = np.sqrt(intrinsics.fx ** 2 + intrinsics.fy ** 2)
        fov_d = 2 * np.arctan(diagonal / (2 * focal_diagonal))

        return {
            'fov_horizontal': np.degrees(fov_h),
            'fov_vertical': np.degrees(fov_v),
            'fov_diagonal': np.degrees(fov_d),
        }

    # ============================================================================
    # STEREO CALIBRATION
    # ============================================================================

    def export_depth_camera_baseline(self, profile) -> float:
        """
        Export stereo depth camera baseline.

        The baseline is the distance between the left and right infrared cameras.
        Typical values: 0.05m to 0.10m (50mm to 100mm)

        Args:
            profile: RealSense stream profile

        Returns:
            Baseline distance in meters
        """
        # Get extrinsics from left IR to right IR
        ir_left_stream = profile.get_stream(self.rs.stream.infrared, 1)
        ir_right_stream = profile.get_stream(self.rs.stream.infrared, 2)

        extrinsics = ir_left_stream.as_video_stream_profile().get_extrinsics_to(
            ir_right_stream.as_video_stream_profile()
        )

        # Baseline is the magnitude of the translation vector (primarily X-axis)
        translation = np.array(extrinsics.translation)
        baseline = np.linalg.norm(translation)

        return baseline

    def export_stereo_rotation_matrix(self, profile) -> np.ndarray:
        """
        Export stereo rotation matrix (left IR to right IR).

        The rotation matrix is a 3x3 orthonormal matrix representing
        the rotation from left IR camera to right IR camera coordinate system.

        Args:
            profile: RealSense stream profile

        Returns:
            3x3 rotation matrix (numpy array)
        """
        ir_left_stream = profile.get_stream(self.rs.stream.infrared, 1)
        ir_right_stream = profile.get_stream(self.rs.stream.infrared, 2)

        extrinsics = ir_left_stream.as_video_stream_profile().get_extrinsics_to(
            ir_right_stream.as_video_stream_profile()
        )

        # Rotation matrix is stored as flat array (row-major)
        rotation_matrix = np.array(extrinsics.rotation).reshape(3, 3)

        return rotation_matrix

    def export_stereo_translation_vector(self, profile) -> np.ndarray:
        """
        Export stereo translation vector (left IR to right IR).

        The translation vector represents the baseline offset in 3D space.
        Typically, translation is primarily along the X-axis.

        Args:
            profile: RealSense stream profile

        Returns:
            3D translation vector in meters [tx, ty, tz]
        """
        ir_left_stream = profile.get_stream(self.rs.stream.infrared, 1)
        ir_right_stream = profile.get_stream(self.rs.stream.infrared, 2)

        extrinsics = ir_left_stream.as_video_stream_profile().get_extrinsics_to(
            ir_right_stream.as_video_stream_profile()
        )

        translation_vector = np.array(extrinsics.translation)

        return translation_vector

    def export_depth_scale_factor(self, device) -> float:
        """
        Export depth scale factor.

        The depth scale converts raw depth units to meters:
        depth_meters = depth_units * depth_scale

        Typical value: 0.001 (1mm per unit)

        Args:
            device: RealSense device object

        Returns:
            Depth scale factor (meters per unit)
        """
        depth_sensor = device.first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()

        return depth_scale

    def export_stereo_rectification_parameters(self, profile) -> Dict[str, np.ndarray]:
        """
        Export stereo rectification parameters.

        Rectification transforms stereo images so that epipolar lines are horizontal,
        simplifying stereo matching.

        Args:
            profile: RealSense stream profile

        Returns:
            Dictionary with rectification parameters
            {
                'rotation_left': np.ndarray,   # 3x3 left rectification matrix
                'rotation_right': np.ndarray,  # 3x3 right rectification matrix
                'baseline': float,             # Baseline in meters
            }
        """
        # Get intrinsics
        ir_left_stream = profile.get_stream(self.rs.stream.infrared, 1)
        ir_right_stream = profile.get_stream(self.rs.stream.infrared, 2)

        intrinsics_left = ir_left_stream.as_video_stream_profile().get_intrinsics()
        intrinsics_right = ir_right_stream.as_video_stream_profile().get_intrinsics()

        # Get extrinsics (rotation and translation)
        extrinsics = ir_left_stream.as_video_stream_profile().get_extrinsics_to(
            ir_right_stream.as_video_stream_profile()
        )

        rotation_matrix = np.array(extrinsics.rotation).reshape(3, 3)
        translation_vector = np.array(extrinsics.translation)

        # For stereo rectification, we typically use cv2.stereoRectify
        # Here we return the basic parameters needed
        return {
            'rotation_left': np.eye(3),  # Left is typically identity
            'rotation_right': rotation_matrix,
            'baseline': np.linalg.norm(translation_vector),
            'translation': translation_vector,
        }

    # ============================================================================
    # EXTRINSICS TRANSFORMATIONS
    # ============================================================================

    def export_depth_to_rgb_transformation(self, profile) -> Dict[str, np.ndarray]:
        """
        Export depth-to-RGB transformation matrix.

        This transformation maps depth camera coordinates to RGB camera coordinates.

        Args:
            profile: RealSense stream profile

        Returns:
            Dictionary with transformation parameters
            {
                'rotation': np.ndarray,     # 3x3 rotation matrix
                'translation': np.ndarray,  # 3D translation vector (meters)
                'transform_4x4': np.ndarray # 4x4 homogeneous transformation matrix
            }
        """
        depth_stream = profile.get_stream(self.rs.stream.depth)
        color_stream = profile.get_stream(self.rs.stream.color)

        extrinsics = depth_stream.as_video_stream_profile().get_extrinsics_to(
            color_stream.as_video_stream_profile()
        )

        rotation = np.array(extrinsics.rotation).reshape(3, 3)
        translation = np.array(extrinsics.translation)

        # Build 4x4 homogeneous transformation matrix
        transform_4x4 = np.eye(4)
        transform_4x4[:3, :3] = rotation
        transform_4x4[:3, 3] = translation

        return {
            'rotation': rotation,
            'translation': translation,
            'transform_4x4': transform_4x4,
        }

    def export_ir_to_rgb_transformation(self, profile, ir_index: int = 1) -> Dict[str, np.ndarray]:
        """
        Export IR-to-RGB transformation matrix.

        Args:
            profile: RealSense stream profile
            ir_index: IR camera index (1=left, 2=right)

        Returns:
            Dictionary with transformation parameters
            {
                'rotation': np.ndarray,
                'translation': np.ndarray,
                'transform_4x4': np.ndarray
            }
        """
        ir_stream = profile.get_stream(self.rs.stream.infrared, ir_index)
        color_stream = profile.get_stream(self.rs.stream.color)

        extrinsics = ir_stream.as_video_stream_profile().get_extrinsics_to(
            color_stream.as_video_stream_profile()
        )

        rotation = np.array(extrinsics.rotation).reshape(3, 3)
        translation = np.array(extrinsics.translation)

        # Build 4x4 homogeneous transformation matrix
        transform_4x4 = np.eye(4)
        transform_4x4[:3, :3] = rotation
        transform_4x4[:3, 3] = translation

        return {
            'rotation': rotation,
            'translation': translation,
            'transform_4x4': transform_4x4,
        }

    def export_per_frame_camera_pose(
        self,
        profile,
        imu_data: Optional[Dict] = None
    ) -> Dict[str, np.ndarray]:
        """
        Export per-frame camera pose (position and orientation).

        If IMU data is available, the pose can be computed from accelerometer
        and gyroscope readings. Otherwise, returns identity transformation.

        Args:
            profile: RealSense stream profile
            imu_data: Optional IMU data (accelerometer, gyroscope)

        Returns:
            Dictionary with pose information
            {
                'position': np.ndarray,     # 3D position [x, y, z]
                'orientation': np.ndarray,  # 3x3 rotation or quaternion
                'transform_4x4': np.ndarray # 4x4 transformation matrix
            }
        """
        # For static camera, return identity transform
        # In practice, this would integrate IMU data for moving cameras
        transform_4x4 = np.eye(4)

        if imu_data is not None:
            # TODO: Integrate IMU data to compute pose
            # This requires sensor fusion (e.g., Madgwick, Mahony filters)
            pass

        return {
            'position': transform_4x4[:3, 3],
            'orientation': transform_4x4[:3, :3],
            'transform_4x4': transform_4x4,
        }

    # ============================================================================
    # CALIBRATION VERIFICATION
    # ============================================================================

    def detect_checkerboard_pattern(
        self,
        image: np.ndarray,
        pattern_size: Tuple[int, int] = (9, 6),
        subpix_criteria: Optional[Tuple] = None
    ) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Detect checkerboard pattern for calibration verification.

        Args:
            image: Grayscale or BGR image
            pattern_size: Checkerboard size (columns, rows) of inner corners
            subpix_criteria: Subpixel refinement criteria (optional)

        Returns:
            Tuple of (found, corners)
            - found: Boolean indicating if pattern was detected
            - corners: Array of corner coordinates (N, 1, 2) or None
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Find checkerboard corners
        found, corners = cv2.findChessboardCorners(
            gray,
            pattern_size,
            flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        )

        if found and corners is not None:
            # Refine corner positions to subpixel accuracy
            if subpix_criteria is None:
                subpix_criteria = (
                    cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                    30,
                    0.001
                )

            corners = cv2.cornerSubPix(
                gray,
                corners,
                (11, 11),
                (-1, -1),
                subpix_criteria
            )

        return found, corners

    def calculate_reprojection_error(
        self,
        object_points: np.ndarray,
        image_points: np.ndarray,
        camera_matrix: np.ndarray,
        dist_coeffs: np.ndarray,
        rvec: np.ndarray,
        tvec: np.ndarray
    ) -> Tuple[float, np.ndarray]:
        """
        Calculate reprojection error for calibration verification.

        Reprojection error is the distance between detected points and
        projected 3D points using the calibration.

        Args:
            object_points: 3D points in world coordinates (N, 3)
            image_points: 2D detected points in image (N, 2)
            camera_matrix: 3x3 camera intrinsic matrix
            dist_coeffs: Distortion coefficients
            rvec: Rotation vector
            tvec: Translation vector

        Returns:
            Tuple of (mean_error, errors)
            - mean_error: Mean reprojection error in pixels
            - errors: Per-point errors
        """
        # Project 3D points to 2D
        projected_points, _ = cv2.projectPoints(
            object_points,
            rvec,
            tvec,
            camera_matrix,
            dist_coeffs
        )

        # Calculate error
        projected_points = projected_points.reshape(-1, 2)
        errors = np.linalg.norm(image_points - projected_points, axis=1)
        mean_error = np.mean(errors)

        return mean_error, errors

    def calculate_calibration_accuracy_metrics(
        self,
        object_points_list: List[np.ndarray],
        image_points_list: List[np.ndarray],
        camera_matrix: np.ndarray,
        dist_coeffs: np.ndarray,
        rvecs: List[np.ndarray],
        tvecs: List[np.ndarray]
    ) -> Dict[str, any]:
        """
        Calculate comprehensive calibration accuracy metrics.

        Args:
            object_points_list: List of 3D point arrays (one per image)
            image_points_list: List of 2D point arrays (one per image)
            camera_matrix: Camera intrinsic matrix
            dist_coeffs: Distortion coefficients
            rvecs: List of rotation vectors
            tvecs: List of translation vectors

        Returns:
            Dictionary with accuracy metrics
            {
                'rms_error': float,
                'mean_error': float,
                'per_image_errors': list,
                'outlier_images': list,  # Indices of images with high error
                'num_images': int,
                'num_outliers': int,
            }
        """
        per_image_errors = []

        for i, (obj_pts, img_pts, rvec, tvec) in enumerate(
            zip(object_points_list, image_points_list, rvecs, tvecs)
        ):
            mean_err, _ = self.calculate_reprojection_error(
                obj_pts, img_pts, camera_matrix, dist_coeffs, rvec, tvec
            )
            per_image_errors.append(mean_err)

        per_image_errors = np.array(per_image_errors)

        # Calculate RMS error
        rms_error = np.sqrt(np.mean(per_image_errors ** 2))

        # Identify outliers (error > mean + 2*std)
        mean_error = np.mean(per_image_errors)
        std_error = np.std(per_image_errors)
        outlier_threshold = mean_error + 2 * std_error
        outlier_images = np.where(per_image_errors > outlier_threshold)[0].tolist()

        return {
            'rms_error': float(rms_error),
            'mean_error': float(mean_error),
            'std_error': float(std_error),
            'per_image_errors': per_image_errors.tolist(),
            'outlier_images': outlier_images,
            'num_images': len(per_image_errors),
            'num_outliers': len(outlier_images),
        }

    def detect_calibration_drift(
        self,
        current_intrinsics: Dict,
        stored_intrinsics: Dict,
        threshold: float = 0.01
    ) -> Dict[str, any]:
        """
        Detect calibration drift by comparing current and stored intrinsics.

        Args:
            current_intrinsics: Current camera intrinsics
            stored_intrinsics: Previously stored intrinsics
            threshold: Drift threshold (fractional change, e.g., 0.01 = 1%)

        Returns:
            Dictionary with drift analysis
            {
                'has_drift': bool,
                'fx_drift': float,  # Fractional change
                'fy_drift': float,
                'cx_drift': float,
                'cy_drift': float,
                'max_drift': float,
                'drift_magnitude': float,
            }
        """
        # Calculate fractional changes
        fx_drift = abs(current_intrinsics['fx'] - stored_intrinsics['fx']) / stored_intrinsics['fx']
        fy_drift = abs(current_intrinsics['fy'] - stored_intrinsics['fy']) / stored_intrinsics['fy']
        cx_drift = abs(current_intrinsics['cx'] - stored_intrinsics['cx']) / stored_intrinsics['cx']
        cy_drift = abs(current_intrinsics['cy'] - stored_intrinsics['cy']) / stored_intrinsics['cy']

        max_drift = max(fx_drift, fy_drift, cx_drift, cy_drift)
        drift_magnitude = np.sqrt(fx_drift**2 + fy_drift**2 + cx_drift**2 + cy_drift**2)

        has_drift = max_drift > threshold

        return {
            'has_drift': has_drift,
            'fx_drift': float(fx_drift),
            'fy_drift': float(fy_drift),
            'cx_drift': float(cx_drift),
            'cy_drift': float(cy_drift),
            'max_drift': float(max_drift),
            'drift_magnitude': float(drift_magnitude),
        }

    # ============================================================================
    # CALIBRATION EXPORT / IMPORT
    # ============================================================================

    def export_full_calibration(
        self,
        profile,
        device,
        output_path: Path
    ):
        """
        Export complete calibration data to JSON file.

        Args:
            profile: RealSense stream profile
            device: RealSense device object
            output_path: Output JSON file path
        """
        calibration_data = {
            'timestamp': datetime.now().isoformat(),
            'device_serial': device.get_info(self.rs.camera_info.serial_number),
            'device_name': device.get_info(self.rs.camera_info.name),
            'firmware_version': device.get_info(self.rs.camera_info.firmware_version),

            # RGB intrinsics
            'rgb_intrinsics': {
                'focal_length': self.export_rgb_focal_length(profile),
                'principal_point': self.export_rgb_principal_point(profile),
                'distortion': self.export_rgb_distortion_coefficients(profile),
                'lens_model': self.export_camera_lens_model(profile),
                'field_of_view': self.calculate_rgb_field_of_view(profile),
            },

            # Stereo calibration
            'stereo_calibration': {
                'baseline': float(self.export_depth_camera_baseline(profile)),
                'rotation_matrix': self.export_stereo_rotation_matrix(profile).tolist(),
                'translation_vector': self.export_stereo_translation_vector(profile).tolist(),
                'depth_scale': float(self.export_depth_scale_factor(device)),
            },

            # Extrinsics
            'extrinsics': {
                'depth_to_rgb': {
                    k: v.tolist() if isinstance(v, np.ndarray) else v
                    for k, v in self.export_depth_to_rgb_transformation(profile).items()
                },
                'ir_left_to_rgb': {
                    k: v.tolist() if isinstance(v, np.ndarray) else v
                    for k, v in self.export_ir_to_rgb_transformation(profile, 1).items()
                },
                'ir_right_to_rgb': {
                    k: v.tolist() if isinstance(v, np.ndarray) else v
                    for k, v in self.export_ir_to_rgb_transformation(profile, 2).items()
                },
            },
        }

        # Save to JSON
        with open(output_path, 'w') as f:
            json.dump(calibration_data, f, indent=2)

        # Store for drift detection
        self.stored_calibration = calibration_data

    def load_stored_calibration(self, input_path: Path) -> Dict:
        """
        Load stored calibration data from JSON file.

        Args:
            input_path: Input JSON file path

        Returns:
            Calibration data dictionary
        """
        with open(input_path, 'r') as f:
            calibration_data = json.load(f)

        self.stored_calibration = calibration_data
        return calibration_data

    def get_opencv_camera_matrix(self, profile) -> np.ndarray:
        """
        Get OpenCV-compatible camera matrix from RealSense intrinsics.

        Camera matrix format:
        [[fx,  0, cx],
         [ 0, fy, cy],
         [ 0,  0,  1]]

        Args:
            profile: RealSense stream profile

        Returns:
            3x3 camera matrix
        """
        intrinsics = self.get_rgb_intrinsics(profile)

        camera_matrix = np.array([
            [intrinsics.fx, 0, intrinsics.ppx],
            [0, intrinsics.fy, intrinsics.ppy],
            [0, 0, 1]
        ], dtype=np.float64)

        return camera_matrix

    def get_opencv_distortion_coeffs(self, profile) -> np.ndarray:
        """
        Get OpenCV-compatible distortion coefficients.

        Args:
            profile: RealSense stream profile

        Returns:
            Distortion coefficients array
        """
        intrinsics = self.get_rgb_intrinsics(profile)
        return np.array(intrinsics.coeffs, dtype=np.float64)
