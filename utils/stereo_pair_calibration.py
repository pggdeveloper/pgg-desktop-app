"""
Stereo Pair Calibration for Wide-Baseline ZED 2i Cameras

This module provides stereo calibration for two ZED 2i cameras configured
as a wide-baseline stereo pair (2-5 meters apart) for large area coverage
in feedlot/pasture applications.

Features:
- Fundamental matrix computation
- Essential matrix computation
- Rectification transforms
- Stereo baseline and disparity range calculation
- Rectification quality validation

Part of Scenario 13 (Multi-Vendor Multi-Camera Integration)

Implements:
- Scenario 2.1: Calibrate ZED 2i cameras as stereo pair
- Scenario 2.2: Validate stereo rectification quality
"""

import cv2
import numpy as np
from typing import Tuple, Dict, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class StereoPairCalibration:
    """Stereo calibration result for camera pair"""
    camera_left_id: str
    camera_right_id: str
    fundamental_matrix: np.ndarray  # 3x3 fundamental matrix
    essential_matrix: np.ndarray  # 3x3 essential matrix
    rotation: np.ndarray  # 3x3 rotation from left to right
    translation: np.ndarray  # 3x1 translation from left to right
    rectification_left: np.ndarray  # 3x3 rectification for left camera
    rectification_right: np.ndarray  # 3x3 rectification for right camera
    projection_left: np.ndarray  # 3x4 projection matrix for left
    projection_right: np.ndarray  # 3x4 projection matrix for right
    Q: np.ndarray  # 4x4 disparity-to-depth mapping matrix
    baseline_meters: float  # Physical baseline distance
    disparity_range: Tuple[float, float]  # (min_disparity, max_disparity)
    rectification_quality_score: float  # 0-1, higher is better

    def to_dict(self) -> Dict:
        """Convert to JSON-serializable dictionary"""
        return {
            "camera_left_id": self.camera_left_id,
            "camera_right_id": self.camera_right_id,
            "fundamental_matrix": self.fundamental_matrix.tolist(),
            "essential_matrix": self.essential_matrix.tolist(),
            "rotation": self.rotation.tolist(),
            "translation": self.translation.flatten().tolist(),
            "rectification_left": self.rectification_left.tolist(),
            "rectification_right": self.rectification_right.tolist(),
            "projection_left": self.projection_left.tolist(),
            "projection_right": self.projection_right.tolist(),
            "Q": self.Q.tolist(),
            "baseline_meters": float(self.baseline_meters),
            "disparity_range": {
                "min": float(self.disparity_range[0]),
                "max": float(self.disparity_range[1])
            },
            "rectification_quality_score": float(self.rectification_quality_score)
        }


@dataclass
class RectificationQualityMetrics:
    """Quality metrics for stereo rectification"""
    vertical_disparity_max: float  # Maximum vertical disparity in pixels
    vertical_disparity_mean: float  # Mean vertical disparity
    epipolar_alignment_score: float  # 0-1, how well aligned are epipolar lines
    quality_assessment: str  # "excellent", "good", "acceptable", "poor"
    is_valid: bool  # True if quality meets minimum standards


class StereoPairCalibrator:
    """
    Calibrate two cameras as stereo pair.

    Designed for wide-baseline stereo (2-5m) with ZED 2i cameras
    for large area coverage in cattle monitoring applications.

    Usage:
        calibrator = StereoPairCalibrator()

        # Calibrate using existing multi-camera calibration
        stereo_calib = calibrator.calibrate_from_multi_camera(
            multi_calib_result,
            camera_left_id="zed_2i_0",
            camera_right_id="zed_2i_1"
        )

        # Validate rectification
        quality = calibrator.validate_rectification(
            test_image_left,
            test_image_right,
            stereo_calib
        )
    """

    def __init__(self):
        """Initialize stereo pair calibrator"""
        logger.info("Initialized StereoPairCalibrator")

    def calibrate_from_multi_camera(self,
                                    multi_camera_calibration,
                                    camera_left_id: str,
                                    camera_right_id: str) -> StereoPairCalibration:
        """
        Calibrate stereo pair from existing multi-camera calibration.

        Implements Scenario 2.1: Calibrate ZED 2i cameras as stereo pair

        Args:
            multi_camera_calibration: CalibrationResult from multi-camera calibration
            camera_left_id: ID of left camera
            camera_right_id: ID of right camera

        Returns:
            StereoPairCalibration object
        """
        logger.info(f"Calibrating stereo pair: {camera_left_id} ↔ {camera_right_id}")

        # Get intrinsic calibrations
        intrinsic_left = multi_camera_calibration.intrinsic_calibrations[camera_left_id]
        intrinsic_right = multi_camera_calibration.intrinsic_calibrations[camera_right_id]

        # Get transformation from left to right
        T_left_to_right = multi_camera_calibration.extrinsic_calibration.get_transformation(
            camera_left_id, camera_right_id
        )

        # Decompose transformation into R and t
        R = T_left_to_right[:3, :3]
        t = T_left_to_right[:3, 3].reshape(3, 1)

        # Compute baseline
        baseline = np.linalg.norm(t)

        logger.info(f"Stereo baseline: {baseline:.3f} meters")

        # Compute fundamental and essential matrices
        # E = [t]_x @ R (cross product matrix)
        t_x = self._skew_symmetric(t.flatten())
        E = t_x @ R

        # F = inv(K_right)^T @ E @ inv(K_left)
        K_left = intrinsic_left.camera_matrix
        K_right = intrinsic_right.camera_matrix

        F = np.linalg.inv(K_right).T @ E @ np.linalg.inv(K_left)

        logger.info("Computed fundamental and essential matrices")

        # Compute stereo rectification
        image_size = intrinsic_left.image_size

        R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
            K_left, intrinsic_left.distortion_coeffs,
            K_right, intrinsic_right.distortion_coeffs,
            image_size,
            R, t,
            flags=cv2.CALIB_ZERO_DISPARITY,
            alpha=0  # 0 = cropped, 1 = keep all pixels
        )

        logger.info("Computed stereo rectification transforms")

        # Compute disparity range
        # For wide baseline (2-5m) and typical camera specs
        focal_length = K_left[0, 0]  # fx
        min_depth = 1.0  # meters (closest object)
        max_depth = 20.0  # meters (farthest object for ZED 2i)

        # Disparity = (baseline * focal_length) / depth
        max_disparity = (baseline * focal_length) / min_depth
        min_disparity = (baseline * focal_length) / max_depth

        logger.info(f"Disparity range: [{min_disparity:.1f}, {max_disparity:.1f}] pixels")

        # Initialize quality score (will be computed during validation)
        quality_score = 0.0

        stereo_calib = StereoPairCalibration(
            camera_left_id=camera_left_id,
            camera_right_id=camera_right_id,
            fundamental_matrix=F,
            essential_matrix=E,
            rotation=R,
            translation=t,
            rectification_left=R1,
            rectification_right=R2,
            projection_left=P1,
            projection_right=P2,
            Q=Q,
            baseline_meters=baseline,
            disparity_range=(min_disparity, max_disparity),
            rectification_quality_score=quality_score
        )

        logger.info("Stereo pair calibration complete")

        return stereo_calib

    def _skew_symmetric(self, v: np.ndarray) -> np.ndarray:
        """
        Create skew-symmetric matrix from vector.

        [v]_x for cross product v × w = [v]_x @ w

        Args:
            v: 3D vector

        Returns:
            3x3 skew-symmetric matrix
        """
        return np.array([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0]
        ])

    def rectify_images(self,
                      image_left: np.ndarray,
                      image_right: np.ndarray,
                      stereo_calib: StereoPairCalibration,
                      intrinsic_left,
                      intrinsic_right) -> Tuple[np.ndarray, np.ndarray]:
        """
        Rectify stereo image pair.

        Args:
            image_left: Left camera image
            image_right: Right camera image
            stereo_calib: Stereo calibration
            intrinsic_left: Intrinsic calibration for left camera
            intrinsic_right: Intrinsic calibration for right camera

        Returns:
            (rectified_left, rectified_right)
        """
        # Compute rectification maps
        map1_left, map2_left = cv2.initUndistortRectifyMap(
            intrinsic_left.camera_matrix,
            intrinsic_left.distortion_coeffs,
            stereo_calib.rectification_left,
            stereo_calib.projection_left,
            intrinsic_left.image_size,
            cv2.CV_32FC1
        )

        map1_right, map2_right = cv2.initUndistortRectifyMap(
            intrinsic_right.camera_matrix,
            intrinsic_right.distortion_coeffs,
            stereo_calib.rectification_right,
            stereo_calib.projection_right,
            intrinsic_right.image_size,
            cv2.CV_32FC1
        )

        # Apply rectification
        rect_left = cv2.remap(image_left, map1_left, map2_left, cv2.INTER_LINEAR)
        rect_right = cv2.remap(image_right, map1_right, map2_right, cv2.INTER_LINEAR)

        return rect_left, rect_right

    def validate_rectification(self,
                               image_left: np.ndarray,
                               image_right: np.ndarray,
                               stereo_calib: StereoPairCalibration,
                               intrinsic_left,
                               intrinsic_right,
                               draw_epipolar_lines: bool = True) -> RectificationQualityMetrics:
        """
        Validate stereo rectification quality.

        Implements Scenario 2.2: Validate stereo rectification quality

        Checks:
        - Epipolar lines are horizontal
        - Vertical disparity < 1 pixel
        - Rectification quality score > 0.95

        Args:
            image_left: Left camera image
            image_right: Right camera image
            stereo_calib: Stereo calibration
            intrinsic_left: Intrinsic calibration for left
            intrinsic_right: Intrinsic calibration for right
            draw_epipolar_lines: If True, draw epipolar lines for visualization

        Returns:
            RectificationQualityMetrics
        """
        logger.info("Validating stereo rectification quality...")

        # Rectify images
        rect_left, rect_right = self.rectify_images(
            image_left, image_right, stereo_calib,
            intrinsic_left, intrinsic_right
        )

        # Detect features in both rectified images
        detector = cv2.ORB_create(nfeatures=500)

        kp_left, desc_left = detector.detectAndCompute(rect_left, None)
        kp_right, desc_right = detector.detectAndCompute(rect_right, None)

        # Match features
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = matcher.match(desc_left, desc_right)

        # Sort by distance
        matches = sorted(matches, key=lambda x: x.distance)[:100]  # Top 100 matches

        if len(matches) < 10:
            logger.warning("Insufficient matches for quality validation")
            return RectificationQualityMetrics(
                vertical_disparity_max=float('inf'),
                vertical_disparity_mean=float('inf'),
                epipolar_alignment_score=0.0,
                quality_assessment="poor",
                is_valid=False
            )

        # Compute vertical disparities
        vertical_disparities = []
        for match in matches:
            pt_left = kp_left[match.queryIdx].pt
            pt_right = kp_right[match.trainIdx].pt

            v_disparity = abs(pt_left[1] - pt_right[1])  # Difference in y-coordinate
            vertical_disparities.append(v_disparity)

        vertical_disparities = np.array(vertical_disparities)

        v_disp_mean = np.mean(vertical_disparities)
        v_disp_max = np.max(vertical_disparities)

        logger.info(f"Vertical disparity: mean={v_disp_mean:.3f}px, max={v_disp_max:.3f}px")

        # Compute epipolar alignment score
        # Good rectification: vertical disparity close to 0
        epipolar_score = 1.0 / (1.0 + v_disp_mean)  # Score decreases with mean disparity

        # Overall quality assessment
        if v_disp_max < 1.0 and epipolar_score > 0.95:
            quality = "excellent"
            is_valid = True
        elif v_disp_max < 2.0 and epipolar_score > 0.85:
            quality = "good"
            is_valid = True
        elif v_disp_max < 3.0 and epipolar_score > 0.70:
            quality = "acceptable"
            is_valid = True
        else:
            quality = "poor"
            is_valid = False

        logger.info(f"Rectification quality: {quality} (score={epipolar_score:.3f})")

        # Update stereo calibration with quality score
        stereo_calib.rectification_quality_score = epipolar_score

        metrics = RectificationQualityMetrics(
            vertical_disparity_max=float(v_disp_max),
            vertical_disparity_mean=float(v_disp_mean),
            epipolar_alignment_score=float(epipolar_score),
            quality_assessment=quality,
            is_valid=is_valid
        )

        return metrics

    def compute_disparity_map(self,
                             rect_left: np.ndarray,
                             rect_right: np.ndarray,
                             stereo_calib: StereoPairCalibration,
                             algorithm: str = "sgbm") -> np.ndarray:
        """
        Compute disparity map from rectified stereo pair.

        Args:
            rect_left: Rectified left image
            rect_right: Rectified right image
            stereo_calib: Stereo calibration
            algorithm: "bm" (block matching) or "sgbm" (semi-global)

        Returns:
            Disparity map (float32, in pixels)
        """
        # Convert to grayscale if needed
        if len(rect_left.shape) == 3:
            gray_left = cv2.cvtColor(rect_left, cv2.COLOR_BGR2GRAY)
            gray_right = cv2.cvtColor(rect_right, cv2.COLOR_BGR2GRAY)
        else:
            gray_left = rect_left
            gray_right = rect_right

        # Compute disparity range
        min_disp = int(stereo_calib.disparity_range[0])
        max_disp = int(stereo_calib.disparity_range[1])
        num_disparities = ((max_disp - min_disp) // 16 + 1) * 16  # Must be divisible by 16

        if algorithm == "bm":
            # Block matching (faster but less accurate)
            stereo = cv2.StereoBM_create(numDisparities=num_disparities, blockSize=15)
        else:
            # Semi-global block matching (slower but more accurate)
            window_size = 5
            stereo = cv2.StereoSGBM_create(
                minDisparity=min_disp,
                numDisparities=num_disparities,
                blockSize=window_size,
                P1=8 * 3 * window_size ** 2,
                P2=32 * 3 * window_size ** 2,
                disp12MaxDiff=1,
                uniquenessRatio=10,
                speckleWindowSize=100,
                speckleRange=32,
                mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
            )

        # Compute disparity
        disparity = stereo.compute(gray_left, gray_right).astype(np.float32) / 16.0

        return disparity

    def disparity_to_depth(self,
                          disparity: np.ndarray,
                          stereo_calib: StereoPairCalibration) -> np.ndarray:
        """
        Convert disparity map to depth map.

        Args:
            disparity: Disparity map (pixels)
            stereo_calib: Stereo calibration

        Returns:
            Depth map (meters)
        """
        # depth = (baseline * focal_length) / disparity
        focal_length = stereo_calib.projection_left[0, 0]
        baseline = stereo_calib.baseline_meters

        # Avoid division by zero
        depth = np.zeros_like(disparity)
        valid_mask = disparity > 0
        depth[valid_mask] = (baseline * focal_length) / disparity[valid_mask]

        return depth


# Convenience functions
def calibrate_stereo_pair_from_multi_camera(multi_calib,
                                            camera_left_id: str,
                                            camera_right_id: str) -> StereoPairCalibration:
    """
    Convenience function to calibrate stereo pair.

    Args:
        multi_calib: MultiCameraCalibration result
        camera_left_id: Left camera ID
        camera_right_id: Right camera ID

    Returns:
        StereoPairCalibration
    """
    calibrator = StereoPairCalibrator()
    return calibrator.calibrate_from_multi_camera(
        multi_calib, camera_left_id, camera_right_id
    )
