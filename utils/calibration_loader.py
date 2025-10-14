"""
Calibration Loader for Camera Orchestrator

This module provides utilities to load and validate multi-camera calibration
data for use during recording sessions.

Features:
- Load calibration from JSON
- Validate calibration (age, camera IDs, quality)
- Provide transformation matrices
- Check if recalibration needed

Part of Scenario 13 (Multi-Vendor Multi-Camera Integration)

Implements:
- Scenario 9.1: Load camera calibration at orchestrator initialization
- Scenario 11.1: Monitor calibration drift over time
- Scenario 11.3: Validate multi-camera system before recording
"""

from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, Tuple
import logging
import numpy as np

from .multi_camera_calibration import CalibrationResult
from .transformation_matrix import TransformationMatrix

logger = logging.getLogger(__name__)


class CalibrationLoader:
    """
    Load and manage multi-camera calibration for runtime use.

    Features:
    - Load from JSON file
    - Validation (expiry, camera IDs, quality)
    - Warning system for old calibrations
    - Easy access to transformation matrices

    Usage:
        loader = CalibrationLoader("calibration.json")

        if loader.is_valid():
            # Use calibration
            T = loader.get_transformation("zed_2i_0", "realsense_d455i_0")
        else:
            print(f"Calibration issues: {loader.get_validation_messages()}")
    """

    def __init__(self,
                 calibration_path: Path,
                 max_age_days: int = 30,
                 max_reprojection_error: float = 2.0):
        """
        Initialize calibration loader.

        Args:
            calibration_path: Path to calibration JSON file
            max_age_days: Maximum calibration age before warning (default: 30 days)
            max_reprojection_error: Maximum acceptable reprojection error (default: 2.0 pixels)
        """
        self.calibration_path = Path(calibration_path)
        self.max_age_days = max_age_days
        self.max_reprojection_error = max_reprojection_error

        self.calibration: Optional[CalibrationResult] = None
        self.validation_messages = []
        self.warnings = []

        # Load calibration
        self._load()

    def _load(self):
        """Load calibration from file"""
        if not self.calibration_path.exists():
            self.validation_messages.append(f"Calibration file not found: {self.calibration_path}")
            logger.error(f"Calibration file not found: {self.calibration_path}")
            return

        try:
            self.calibration = CalibrationResult.load(self.calibration_path)
            logger.info(f"Loaded calibration from: {self.calibration_path}")

            # Validate
            self._validate()

        except Exception as e:
            self.validation_messages.append(f"Failed to load calibration: {e}")
            logger.exception("Failed to load calibration")

    def _validate(self):
        """Validate loaded calibration"""
        if self.calibration is None:
            return

        # Check calibration age
        calib_date = datetime.fromisoformat(self.calibration.calibration_date)
        age = datetime.now() - calib_date
        age_days = age.days

        if age_days > self.max_age_days:
            self.warnings.append(
                f"Calibration is {age_days} days old (recommended: <{self.max_age_days} days). "
                "Consider recalibration."
            )
            logger.warning(f"Calibration is {age_days} days old")

        # Check quality metrics
        mean_error = self.calibration.quality_metrics.get('mean_reprojection_error', 0.0)

        if mean_error > self.max_reprojection_error:
            self.validation_messages.append(
                f"Poor calibration quality: reprojection error {mean_error:.2f} pixels "
                f"(threshold: {self.max_reprojection_error} pixels)"
            )
            logger.error(f"Poor calibration quality: {mean_error:.2f} pixels")

        overall_quality = self.calibration.quality_metrics.get('overall_quality', 'unknown')
        if overall_quality == 'poor':
            self.validation_messages.append("Calibration quality is marked as 'poor'. Recalibration recommended.")

        # Log if good
        if not self.validation_messages:
            logger.info(f"Calibration validation passed (age: {age_days} days, quality: {overall_quality})")

    def is_valid(self) -> bool:
        """
        Check if calibration is valid for use.

        Returns:
            True if valid, False if critical issues found
        """
        return self.calibration is not None and len(self.validation_messages) == 0

    def has_warnings(self) -> bool:
        """Check if there are warnings (non-critical issues)"""
        return len(self.warnings) > 0

    def get_validation_messages(self) -> list:
        """Get list of validation error messages"""
        return self.validation_messages.copy()

    def get_warnings(self) -> list:
        """Get list of warning messages"""
        return self.warnings.copy()

    def get_transformation(self,
                          from_camera: str,
                          to_camera: str) -> Optional[np.ndarray]:
        """
        Get transformation matrix from one camera to another.

        Args:
            from_camera: Source camera ID
            to_camera: Target camera ID

        Returns:
            4x4 transformation matrix or None if not available
        """
        if not self.is_valid():
            return None

        return self.calibration.extrinsic_calibration.get_transformation(
            from_camera, to_camera
        )

    def get_camera_pose(self, camera_id: str):
        """Get camera pose in global coordinates"""
        if not self.is_valid():
            return None

        return self.calibration.extrinsic_calibration.camera_poses.get(camera_id)

    def get_intrinsics(self, camera_id: str):
        """Get intrinsic calibration for camera"""
        if not self.is_valid():
            return None

        return self.calibration.intrinsic_calibrations.get(camera_id)

    def get_reference_camera(self) -> Optional[str]:
        """Get reference camera ID (origin of global coordinates)"""
        if not self.is_valid():
            return None

        return self.calibration.extrinsic_calibration.reference_camera

    def get_camera_ids(self) -> list:
        """Get list of calibrated camera IDs"""
        if not self.is_valid():
            return []

        return list(self.calibration.intrinsic_calibrations.keys())

    def validate_camera_ids(self, expected_camera_ids: list) -> Tuple[bool, str]:
        """
        Validate that calibration has expected cameras.

        Args:
            expected_camera_ids: List of expected camera IDs

        Returns:
            (is_valid, message)
        """
        if not self.is_valid():
            return False, "Calibration not loaded"

        calibrated_ids = set(self.get_camera_ids())
        expected_ids = set(expected_camera_ids)

        missing = expected_ids - calibrated_ids
        extra = calibrated_ids - expected_ids

        if missing or extra:
            msg = "Camera ID mismatch: "
            if missing:
                msg += f"missing {list(missing)} "
            if extra:
                msg += f"unexpected {list(extra)}"
            return False, msg

        return True, "Camera IDs match"

    def get_calibration_info(self) -> Dict:
        """Get calibration information summary"""
        if not self.is_valid():
            return {
                "valid": False,
                "errors": self.validation_messages,
                "warnings": self.warnings
            }

        calib_date = datetime.fromisoformat(self.calibration.calibration_date)
        age_days = (datetime.now() - calib_date).days

        return {
            "valid": True,
            "calibration_date": self.calibration.calibration_date,
            "age_days": age_days,
            "reference_camera": self.get_reference_camera(),
            "camera_ids": self.get_camera_ids(),
            "quality": self.calibration.quality_metrics.get('overall_quality'),
            "mean_reprojection_error": self.calibration.quality_metrics.get('mean_reprojection_error'),
            "warnings": self.warnings,
            "pattern_config": self.calibration.pattern_config
        }

    def needs_recalibration(self) -> Tuple[bool, str]:
        """
        Check if recalibration is recommended.

        Returns:
            (needs_recalibration, reason)
        """
        if not self.is_valid():
            return True, "Calibration invalid or not loaded"

        # Check age
        calib_date = datetime.fromisoformat(self.calibration.calibration_date)
        age_days = (datetime.now() - calib_date).days

        if age_days > self.max_age_days * 2:  # 2x threshold
            return True, f"Calibration too old ({age_days} days)"

        # Check quality
        quality = self.calibration.quality_metrics.get('overall_quality')
        if quality == 'poor':
            return True, "Calibration quality is poor"

        mean_error = self.calibration.quality_metrics.get('mean_reprojection_error', 0.0)
        if mean_error > self.max_reprojection_error:
            return True, f"Reprojection error too high ({mean_error:.2f} pixels)"

        # All good
        return False, ""


class CalibrationValidator:
    """
    Validate multi-camera system before recording.

    Implements Scenario 11.3: Validate multi-camera system before recording session
    """

    @staticmethod
    def validate_pre_recording(calibration_loader: CalibrationLoader,
                               expected_cameras: list,
                               min_disk_space_gb: float = 10.0) -> Tuple[bool, list]:
        """
        Validate system before starting recording.

        Checks:
        1. Calibration loaded and valid
        2. Camera IDs match
        3. Calibration quality acceptable
        4. Sufficient disk space

        Args:
            calibration_loader: CalibrationLoader instance
            expected_cameras: List of expected camera IDs
            min_disk_space_gb: Minimum disk space required (GB)

        Returns:
            (is_valid, issues) - issues is list of problem descriptions
        """
        issues = []

        # Check 1: Calibration valid
        if not calibration_loader.is_valid():
            issues.extend(calibration_loader.get_validation_messages())

        # Check 2: Camera IDs match
        valid_ids, msg = calibration_loader.validate_camera_ids(expected_cameras)
        if not valid_ids:
            issues.append(msg)

        # Check 3: Calibration quality
        if calibration_loader.has_warnings():
            # Warnings are non-critical, just log them
            for warning in calibration_loader.get_warnings():
                logger.warning(f"Pre-recording warning: {warning}")

        # Check 4: Disk space
        try:
            import shutil
            stats = shutil.disk_usage('/')
            free_gb = stats.free / (1024 ** 3)

            if free_gb < min_disk_space_gb:
                issues.append(
                    f"Insufficient disk space: {free_gb:.1f}GB free "
                    f"(minimum: {min_disk_space_gb}GB)"
                )
        except Exception as e:
            logger.warning(f"Could not check disk space: {e}")

        # Determine overall validity
        is_valid = len(issues) == 0

        if is_valid:
            logger.info("Pre-recording validation passed")
        else:
            logger.error(f"Pre-recording validation failed: {issues}")

        return is_valid, issues


# Convenience functions
def load_calibration(filepath: str,
                    max_age_days: int = 30) -> Optional[CalibrationLoader]:
    """
    Convenience function to load calibration.

    Args:
        filepath: Path to calibration JSON
        max_age_days: Maximum calibration age

    Returns:
        CalibrationLoader or None if failed
    """
    loader = CalibrationLoader(filepath, max_age_days=max_age_days)

    if not loader.is_valid():
        logger.error(f"Failed to load valid calibration from {filepath}")
        return None

    return loader


def check_calibration_health(filepath: str) -> Dict:
    """
    Check calibration health status.

    Args:
        filepath: Path to calibration JSON

    Returns:
        Dict with health status
    """
    loader = CalibrationLoader(filepath)

    if not loader.is_valid():
        return {
            "healthy": False,
            "status": "invalid",
            "errors": loader.get_validation_messages()
        }

    needs_recalib, reason = loader.needs_recalibration()

    if needs_recalib:
        return {
            "healthy": False,
            "status": "recalibration_recommended",
            "reason": reason,
            "warnings": loader.get_warnings()
        }

    if loader.has_warnings():
        return {
            "healthy": True,
            "status": "healthy_with_warnings",
            "warnings": loader.get_warnings(),
            "info": loader.get_calibration_info()
        }

    return {
        "healthy": True,
        "status": "excellent",
        "info": loader.get_calibration_info()
    }
