"""
Dense Stereo Reconstruction for Wide-Baseline ZED Pair

This module provides dense stereo reconstruction capabilities for the
wide-baseline ZED 2i camera pair (2-5m baseline).

Part of Scenario 13 (Multi-Vendor Multi-Camera Integration)

Implements:
- Scenario 12.1: Dense stereo reconstruction from ZED 2i pair

Uses OpenCV for CPU-based stereo matching (no GPU required).
"""

import numpy as np
import cv2
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class StereoCalibration:
    """Stereo calibration parameters"""
    camera_matrix_left: np.ndarray  # 3x3 intrinsic matrix
    camera_matrix_right: np.ndarray  # 3x3 intrinsic matrix
    dist_coeffs_left: np.ndarray  # Distortion coefficients
    dist_coeffs_right: np.ndarray  # Distortion coefficients
    R: np.ndarray  # 3x3 rotation matrix (left to right)
    T: np.ndarray  # 3x1 translation vector
    baseline_m: float  # Baseline in meters


@dataclass
class StereoRectification:
    """Stereo rectification transforms"""
    R1: np.ndarray  # 3x3 rectification rotation (left)
    R2: np.ndarray  # 3x3 rectification rotation (right)
    P1: np.ndarray  # 3x4 projection matrix (left)
    P2: np.ndarray  # 3x4 projection matrix (right)
    Q: np.ndarray  # 4x4 disparity-to-depth mapping matrix
    roi_left: Tuple[int, int, int, int]  # Valid ROI (x, y, w, h)
    roi_right: Tuple[int, int, int, int]  # Valid ROI
    map1_left: np.ndarray  # Rectification map x (left)
    map2_left: np.ndarray  # Rectification map y (left)
    map1_right: np.ndarray  # Rectification map x (right)
    map2_right: np.ndarray  # Rectification map y (right)


@dataclass
class DisparityResult:
    """Result of stereo disparity computation"""
    disparity_map: np.ndarray  # Disparity in pixels
    depth_map: np.ndarray  # Depth in meters
    point_cloud: np.ndarray  # 3D points (N, 3)
    colors: Optional[np.ndarray]  # RGB colors (N, 3)
    confidence_map: Optional[np.ndarray]  # Confidence per pixel
    valid_mask: np.ndarray  # Boolean mask of valid depths


class DenseStereoReconstructor:
    """
    Dense stereo reconstruction for wide-baseline ZED pair.

    Implements Scenario 12.1: Dense stereo reconstruction from ZED 2i pair

    Uses OpenCV StereoSGBM (Semi-Global Block Matching) for CPU-based
    stereo matching. This is suitable for the wide baseline (2-5m) between
    the two ZED 2i cameras.
    """

    def __init__(self, stereo_calibration: StereoCalibration):
        """
        Initialize stereo reconstructor.

        Args:
            stereo_calibration: Stereo calibration parameters
        """
        self.calibration = stereo_calibration
        self.rectification: Optional[StereoRectification] = None

        # Default SGBM parameters (can be tuned)
        self.min_disparity = 0
        self.num_disparities = 128  # Must be divisible by 16
        self.block_size = 11  # Odd number, typically 3-11

        # SGBM parameters for wide baseline
        self.p1 = 8 * 3 * self.block_size ** 2
        self.p2 = 32 * 3 * self.block_size ** 2
        self.disp12_max_diff = 1
        self.uniqueness_ratio = 10
        self.speckle_window_size = 100
        self.speckle_range = 32
        self.prefilter_cap = 63
        self.mode = cv2.STEREO_SGBM_MODE_SGBM_3WAY

        logger.info("Initialized DenseStereoReconstructor")

    def compute_rectification(self, image_size: Tuple[int, int]):
        """
        Compute stereo rectification transforms.

        Args:
            image_size: Image size (width, height)
        """
        # Stereo rectification
        R1, R2, P1, P2, Q, roi_left, roi_right = cv2.stereoRectify(
            cameraMatrix1=self.calibration.camera_matrix_left,
            distCoeffs1=self.calibration.dist_coeffs_left,
            cameraMatrix2=self.calibration.camera_matrix_right,
            distCoeffs2=self.calibration.dist_coeffs_right,
            imageSize=image_size,
            R=self.calibration.R,
            T=self.calibration.T,
            flags=cv2.CALIB_ZERO_DISPARITY,
            alpha=0.0  # 0 = crop to valid pixels only
        )

        # Compute rectification maps
        map1_left, map2_left = cv2.initUndistortRectifyMap(
            self.calibration.camera_matrix_left,
            self.calibration.dist_coeffs_left,
            R1, P1, image_size,
            cv2.CV_32FC1
        )

        map1_right, map2_right = cv2.initUndistortRectifyMap(
            self.calibration.camera_matrix_right,
            self.calibration.dist_coeffs_right,
            R2, P2, image_size,
            cv2.CV_32FC1
        )

        self.rectification = StereoRectification(
            R1=R1, R2=R2, P1=P1, P2=P2, Q=Q,
            roi_left=roi_left, roi_right=roi_right,
            map1_left=map1_left, map2_left=map2_left,
            map1_right=map1_right, map2_right=map2_right
        )

        logger.info(
            f"Computed stereo rectification for {image_size[0]}×{image_size[1]} images"
        )

    def rectify_images(self,
                      image_left: np.ndarray,
                      image_right: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Rectify stereo image pair.

        Args:
            image_left: Left image
            image_right: Right image

        Returns:
            (rectified_left, rectified_right)
        """
        if self.rectification is None:
            # Compute rectification on first use
            image_size = (image_left.shape[1], image_left.shape[0])
            self.compute_rectification(image_size)

        # Apply rectification
        rectified_left = cv2.remap(
            image_left,
            self.rectification.map1_left,
            self.rectification.map2_left,
            cv2.INTER_LINEAR
        )

        rectified_right = cv2.remap(
            image_right,
            self.rectification.map1_right,
            self.rectification.map2_right,
            cv2.INTER_LINEAR
        )

        return rectified_left, rectified_right

    def compute_disparity(self,
                         image_left: np.ndarray,
                         image_right: np.ndarray,
                         rectify: bool = True) -> np.ndarray:
        """
        Compute disparity map using SGBM.

        Args:
            image_left: Left image (grayscale or BGR)
            image_right: Right image (grayscale or BGR)
            rectify: Whether to rectify images first

        Returns:
            Disparity map (float32, in pixels)
        """
        # Convert to grayscale if needed
        if len(image_left.shape) == 3:
            gray_left = cv2.cvtColor(image_left, cv2.COLOR_BGR2GRAY)
            gray_right = cv2.cvtColor(image_right, cv2.COLOR_BGR2GRAY)
        else:
            gray_left = image_left
            gray_right = image_right

        # Rectify if requested
        if rectify:
            gray_left, gray_right = self.rectify_images(gray_left, gray_right)

        # Create SGBM matcher
        stereo = cv2.StereoSGBM_create(
            minDisparity=self.min_disparity,
            numDisparities=self.num_disparities,
            blockSize=self.block_size,
            P1=self.p1,
            P2=self.p2,
            disp12MaxDiff=self.disp12_max_diff,
            uniquenessRatio=self.uniqueness_ratio,
            speckleWindowSize=self.speckle_window_size,
            speckleRange=self.speckle_range,
            preFilterCap=self.prefilter_cap,
            mode=self.mode
        )

        # Compute disparity
        disparity = stereo.compute(gray_left, gray_right)

        # Convert to float and scale
        disparity = disparity.astype(np.float32) / 16.0

        logger.info(
            f"Computed disparity map: "
            f"min={np.min(disparity):.1f}, "
            f"max={np.max(disparity):.1f} pixels"
        )

        return disparity

    def disparity_to_depth(self, disparity: np.ndarray) -> np.ndarray:
        """
        Convert disparity to depth using calibration.

        Args:
            disparity: Disparity map (pixels)

        Returns:
            Depth map (meters)
        """
        if self.rectification is None:
            raise ValueError("Rectification not computed")

        # Avoid division by zero
        valid_mask = disparity > 0

        # depth = (baseline * focal_length) / disparity
        # From projection matrix P2: focal_length = P2[0,0]
        # baseline = -P2[0,3] / P2[0,0]
        focal_length = self.rectification.P2[0, 0]
        baseline = -self.rectification.P2[0, 3] / focal_length

        depth = np.zeros_like(disparity)
        depth[valid_mask] = (baseline * focal_length) / disparity[valid_mask]

        return depth

    def disparity_to_point_cloud(self,
                                 disparity: np.ndarray,
                                 image_left: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert disparity to 3D point cloud.

        Args:
            disparity: Disparity map
            image_left: Left image for colors

        Returns:
            (points, colors) - both as (N, 3) arrays
        """
        if self.rectification is None:
            raise ValueError("Rectification not computed")

        # Reproject to 3D using Q matrix
        points_3d = cv2.reprojectImageTo3D(disparity, self.rectification.Q)

        # Filter invalid points
        valid_mask = (disparity > 0) & (points_3d[:, :, 2] > 0)

        # Extract valid points
        points = points_3d[valid_mask]

        # Extract colors
        if len(image_left.shape) == 3:
            colors = image_left[valid_mask]
        else:
            # Grayscale - replicate to RGB
            gray = image_left[valid_mask]
            colors = np.stack([gray, gray, gray], axis=1)

        logger.info(f"Generated point cloud with {points.shape[0]} points")

        return points, colors

    def reconstruct_dense(self,
                         image_left: np.ndarray,
                         image_right: np.ndarray,
                         return_point_cloud: bool = True) -> DisparityResult:
        """
        Perform complete dense stereo reconstruction.

        Implements Scenario 12.1: Dense stereo reconstruction from ZED 2i pair

        Args:
            image_left: Left camera image
            image_right: Right camera image
            return_point_cloud: Whether to generate point cloud

        Returns:
            DisparityResult with disparity, depth, and optionally point cloud
        """
        logger.info("Starting dense stereo reconstruction...")

        # Compute disparity
        disparity = self.compute_disparity(image_left, image_right, rectify=True)

        # Convert to depth
        depth = self.disparity_to_depth(disparity)

        # Valid mask
        valid_mask = (disparity > 0) & (depth > 0) & (depth < 50.0)  # Max 50m

        # Point cloud
        points = None
        colors = None
        if return_point_cloud:
            points, colors = self.disparity_to_point_cloud(disparity, image_left)

        # Confidence (based on disparity consistency)
        # Higher disparity = closer = more confident
        confidence = None
        if valid_mask.any():
            confidence = np.zeros_like(disparity)
            confidence[valid_mask] = np.clip(disparity[valid_mask] / self.num_disparities, 0, 1)

        num_valid = np.count_nonzero(valid_mask)
        total_pixels = valid_mask.size
        valid_ratio = num_valid / total_pixels * 100

        logger.info(
            f"Dense reconstruction complete: "
            f"{valid_ratio:.1f}% valid pixels, "
            f"depth range: {np.min(depth[valid_mask]):.2f}-{np.max(depth[valid_mask]):.2f}m"
        )

        return DisparityResult(
            disparity_map=disparity,
            depth_map=depth,
            point_cloud=points,
            colors=colors,
            confidence_map=confidence,
            valid_mask=valid_mask
        )


# Convenience functions
def create_stereo_reconstructor_from_multi_camera_calibration(
    multi_camera_calibration,
    camera_left_id: str,
    camera_right_id: str
) -> DenseStereoReconstructor:
    """
    Create stereo reconstructor from multi-camera calibration.

    Args:
        multi_camera_calibration: CalibrationResult from multi-camera calibration
        camera_left_id: Left camera ID
        camera_right_id: Right camera ID

    Returns:
        DenseStereoReconstructor instance
    """
    # Get intrinsics
    intrinsic_left = multi_camera_calibration.intrinsic_calibrations[camera_left_id]
    intrinsic_right = multi_camera_calibration.intrinsic_calibrations[camera_right_id]

    # Get transformation from left to right
    T_left_to_right = multi_camera_calibration.extrinsic_calibration.get_transformation(
        camera_left_id, camera_right_id
    )

    R = T_left_to_right[:3, :3]
    T = T_left_to_right[:3, 3].reshape(3, 1)
    baseline = np.linalg.norm(T)

    # Create stereo calibration
    stereo_calib = StereoCalibration(
        camera_matrix_left=intrinsic_left.camera_matrix,
        camera_matrix_right=intrinsic_right.camera_matrix,
        dist_coeffs_left=intrinsic_left.dist_coeffs,
        dist_coeffs_right=intrinsic_right.dist_coeffs,
        R=R,
        T=T,
        baseline_m=baseline
    )

    return DenseStereoReconstructor(stereo_calib)
