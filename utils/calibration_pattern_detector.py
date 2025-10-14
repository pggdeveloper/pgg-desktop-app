"""
Calibration Pattern Detector for Multi-Vendor Camera Calibration

This module provides pattern detection functionality for camera calibration,
supporting chessboard patterns with OpenCV (vendor-agnostic).

Supports:
- ZED 2i cameras (left image)
- RealSense D455i cameras (RGB image)
- Standard chessboard patterns (OpenCV compatible)

Part of Scenario 13 (Multi-Vendor Multi-Camera Integration)
"""

import cv2
import numpy as np
from typing import Tuple, Optional, Dict, List
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class PatternDetectionResult:
    """Result of pattern detection operation"""
    success: bool
    corners: Optional[np.ndarray]  # 2D image coordinates of corners (N, 1, 2)
    object_points: Optional[np.ndarray]  # 3D world coordinates (N, 3)
    num_corners: int
    reprojection_error: Optional[float]
    image_with_overlay: Optional[np.ndarray]


@dataclass
class PatternConfig:
    """Configuration for calibration pattern"""
    pattern_type: str = "chessboard"  # Only chessboard supported for now
    pattern_size: Tuple[int, int] = (9, 6)  # Internal corners (width, height)
    square_size_mm: float = 30.0  # Size of squares in millimeters

    @property
    def square_size_m(self) -> float:
        """Square size in meters"""
        return self.square_size_mm / 1000.0

    @property
    def num_corners(self) -> int:
        """Total number of internal corners"""
        return self.pattern_size[0] * self.pattern_size[1]


class CalibrationPatternDetector:
    """
    Detects calibration patterns in images for multi-vendor camera calibration.

    Features:
    - Chessboard pattern detection (OpenCV compatible)
    - Sub-pixel corner refinement
    - Quality validation (minimum corners, reprojection error)
    - Visual feedback with corner overlays

    Usage:
        detector = CalibrationPatternDetector(pattern_size=(9, 6), square_size_mm=30.0)
        result = detector.detect(image)
        if result.success:
            print(f"Detected {result.num_corners} corners")
    """

    def __init__(self,
                 pattern_size: Tuple[int, int] = (9, 6),
                 square_size_mm: float = 30.0):
        """
        Initialize pattern detector.

        Args:
            pattern_size: (width, height) of internal corners
            square_size_mm: Physical size of squares in millimeters
        """
        self.config = PatternConfig(
            pattern_size=pattern_size,
            square_size_mm=square_size_mm
        )

        # Sub-pixel refinement criteria
        self.subpix_criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            30,  # Max iterations
            0.001  # Epsilon
        )

        # Prepare 3D object points (chessboard in world coordinates)
        self.object_points_template = self._prepare_object_points()

        logger.info(f"Initialized CalibrationPatternDetector with {self.config.pattern_size} "
                   f"pattern, {self.config.square_size_mm}mm squares")

    def _prepare_object_points(self) -> np.ndarray:
        """
        Prepare 3D coordinates of chessboard corners in world space.

        Returns:
            Array of shape (N, 3) with coordinates in meters
            Z=0 (pattern lies on plane), X-Y grid of corners
        """
        objp = np.zeros((self.config.num_corners, 3), np.float32)
        objp[:, :2] = np.mgrid[
            0:self.config.pattern_size[0],
            0:self.config.pattern_size[1]
        ].T.reshape(-1, 2)
        objp *= self.config.square_size_m
        return objp

    def detect(self,
               image: np.ndarray,
               refine_corners: bool = True,
               draw_overlay: bool = True) -> PatternDetectionResult:
        """
        Detect calibration pattern in image.

        Args:
            image: Input image (BGR or grayscale)
            refine_corners: Apply sub-pixel refinement
            draw_overlay: Draw detected corners on image

        Returns:
            PatternDetectionResult with detection status and data
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Detect chessboard corners
        ret, corners = cv2.findChessboardCorners(
            gray,
            self.config.pattern_size,
            None
        )

        if not ret or corners is None:
            logger.warning("Chessboard pattern not detected in image")
            return PatternDetectionResult(
                success=False,
                corners=None,
                object_points=None,
                num_corners=0,
                reprojection_error=None,
                image_with_overlay=None
            )

        # Refine corners with sub-pixel accuracy
        if refine_corners:
            corners = cv2.cornerSubPix(
                gray,
                corners,
                (11, 11),  # Window size
                (-1, -1),  # Zero zone (not used)
                self.subpix_criteria
            )

        # Create overlay visualization
        image_with_overlay = None
        if draw_overlay:
            if len(image.shape) == 3:
                image_with_overlay = image.copy()
            else:
                image_with_overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

            cv2.drawChessboardCorners(
                image_with_overlay,
                self.config.pattern_size,
                corners,
                ret
            )

        logger.info(f"Successfully detected {len(corners)} corners")

        return PatternDetectionResult(
            success=True,
            corners=corners,
            object_points=self.object_points_template.copy(),
            num_corners=len(corners),
            reprojection_error=None,  # Calculated during calibration
            image_with_overlay=image_with_overlay
        )

    def validate_detection(self,
                          result: PatternDetectionResult,
                          min_corners: int = 20) -> Tuple[bool, str]:
        """
        Validate quality of pattern detection.

        Args:
            result: Detection result to validate
            min_corners: Minimum required corners

        Returns:
            (is_valid, message)
        """
        if not result.success:
            return False, "Pattern not detected"

        if result.num_corners < min_corners:
            return False, f"Only {result.num_corners} corners detected (minimum {min_corners})"

        if result.num_corners != self.config.num_corners:
            return False, (f"Expected {self.config.num_corners} corners, "
                          f"got {result.num_corners}")

        return True, "Detection valid"

    def detect_multi_camera(self,
                           images: Dict[str, np.ndarray],
                           require_all: bool = True) -> Dict[str, PatternDetectionResult]:
        """
        Detect pattern in images from multiple cameras.

        Args:
            images: Dict mapping camera_id -> image
            require_all: If True, return None if any detection fails

        Returns:
            Dict mapping camera_id -> PatternDetectionResult
            Returns None if require_all=True and any detection fails
        """
        results = {}

        for camera_id, image in images.items():
            logger.info(f"Detecting pattern in camera: {camera_id}")
            result = self.detect(image)
            results[camera_id] = result

            if require_all and not result.success:
                logger.error(f"Pattern not detected in camera {camera_id}, aborting")
                return None

        logger.info(f"Pattern detected in {len(results)}/{len(images)} cameras")
        return results

    def batch_detect(self,
                    images: List[np.ndarray],
                    min_valid_ratio: float = 0.7) -> Tuple[List[np.ndarray],
                                                           List[np.ndarray]]:
        """
        Detect pattern in batch of images (e.g., calibration sequence).

        Args:
            images: List of images
            min_valid_ratio: Minimum ratio of valid detections required

        Returns:
            (image_points_list, object_points_list)
            Lists of valid 2D and 3D points for calibration
        """
        image_points = []
        object_points = []

        for i, image in enumerate(images):
            result = self.detect(image, draw_overlay=False)

            if result.success:
                valid, msg = self.validate_detection(result)
                if valid:
                    image_points.append(result.corners)
                    object_points.append(result.object_points)
                    logger.debug(f"Image {i+1}/{len(images)}: Pattern detected ✓")
                else:
                    logger.warning(f"Image {i+1}/{len(images)}: {msg}")
            else:
                logger.warning(f"Image {i+1}/{len(images)}: Pattern not detected ✗")

        valid_ratio = len(image_points) / len(images)

        if valid_ratio < min_valid_ratio:
            logger.warning(
                f"Only {valid_ratio*100:.1f}% valid detections "
                f"(minimum {min_valid_ratio*100:.1f}% required)"
            )
        else:
            logger.info(
                f"Batch detection complete: {len(image_points)}/{len(images)} valid "
                f"({valid_ratio*100:.1f}%)"
            )

        return image_points, object_points

    def get_pattern_info(self) -> Dict:
        """Get information about configured pattern"""
        return {
            "pattern_type": self.config.pattern_type,
            "pattern_size": self.config.pattern_size,
            "square_size_mm": self.config.square_size_mm,
            "square_size_m": self.config.square_size_m,
            "num_corners": self.config.num_corners,
            "total_points": self.config.num_corners
        }


# Convenience functions for direct usage
def detect_chessboard(image: np.ndarray,
                      pattern_size: Tuple[int, int] = (9, 6),
                      square_size_mm: float = 30.0) -> PatternDetectionResult:
    """
    Convenience function to detect chessboard pattern.

    Args:
        image: Input image
        pattern_size: (width, height) of internal corners
        square_size_mm: Physical size of squares in mm

    Returns:
        PatternDetectionResult
    """
    detector = CalibrationPatternDetector(pattern_size, square_size_mm)
    return detector.detect(image)


def validate_pattern_visibility(images: Dict[str, np.ndarray],
                                pattern_size: Tuple[int, int] = (9, 6),
                                min_corners: int = 20) -> Dict[str, bool]:
    """
    Validate that pattern is visible from all cameras.

    Args:
        images: Dict mapping camera_id -> image
        pattern_size: Pattern size
        min_corners: Minimum corners required

    Returns:
        Dict mapping camera_id -> is_visible
    """
    detector = CalibrationPatternDetector(pattern_size)
    results = detector.detect_multi_camera(images, require_all=False)

    visibility = {}
    for camera_id, result in results.items():
        is_valid, _ = detector.validate_detection(result, min_corners)
        visibility[camera_id] = is_valid

    return visibility
