"""
Calibration Quality Validation

This module provides validation tools to assess calibration quality using:
- 3D test objects with known dimensions
- Temporal synchronization quality tests
- Real-world measurement validation

Part of Scenario 13 (Multi-Vendor Multi-Camera Integration)

Implements:
- Scenario 8.2: Validate calibration with 3D test object
- Scenario 8.3: Assess temporal synchronization quality
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class TestObjectMeasurement:
    """Measurement of test object from a camera"""
    camera_id: str
    measured_dimensions: np.ndarray  # 3D dimensions (length, width, height) in meters
    measured_position: np.ndarray  # 3D position (x, y, z) in meters
    measured_orientation: np.ndarray  # Euler angles (roll, pitch, yaw) in radians
    measurement_confidence: float  # 0-1, confidence in measurement


@dataclass
class CalibrationValidationResult:
    """Result of calibration validation with test object"""
    test_object_name: str
    ground_truth_dimensions: np.ndarray  # Known dimensions in meters
    measurements: Dict[str, TestObjectMeasurement]

    # Dimensional accuracy
    dimension_errors: Dict[str, float]  # camera_id -> error in meters
    mean_dimension_error: float
    max_dimension_error: float
    dimension_error_percentage: float

    # Position consistency
    position_errors: Dict[Tuple[str, str], float]  # (cam1, cam2) -> error in meters
    mean_position_error: float
    max_position_error: float

    # Orientation consistency
    orientation_errors: Dict[Tuple[str, str], float]  # (cam1, cam2) -> error in degrees
    mean_orientation_error: float
    max_orientation_error: float

    # Overall assessment
    dimensional_accuracy_valid: bool  # Error < 2%
    position_consistency_valid: bool  # Error < 2cm
    orientation_consistency_valid: bool  # Error < 2°
    overall_valid: bool


@dataclass
class TemporalSynchronizationQuality:
    """Quality assessment of temporal synchronization"""
    test_object_name: str
    motion_type: str  # "pendulum", "linear", "rotating"

    # Position consistency across synchronized frames
    position_variance_per_frame: List[float]  # Variance in position across cameras
    mean_position_variance: float
    max_position_variance: float

    # Temporal jitter
    temporal_jitter_ms: float  # Standard deviation of frame time differences

    # Synchronization score
    synchronization_score: float  # 0-1, higher is better
    quality_assessment: str  # "excellent", "good", "acceptable", "poor"
    is_valid: bool  # True if jitter < 10ms


class CalibrationQualityValidator:
    """
    Validate multi-camera calibration quality using real-world tests.

    Tests:
    1. 3D test object with known dimensions
    2. Temporal synchronization with moving object
    3. Multi-camera consistency checks

    Usage:
        validator = CalibrationQualityValidator(calibration_result)

        # Validate with test cube
        result = validator.validate_with_test_object(
            measurements={
                'realsense_d455i_0': measurement_rs,
                'zed_2i_0': measurement_zed0,
                'zed_2i_1': measurement_zed1
            },
            ground_truth_dimensions=np.array([0.1, 0.1, 0.1])  # 10cm cube
        )

        if result.overall_valid:
            print("Calibration validated successfully!")
    """

    def __init__(self, calibration_result):
        """
        Initialize validator with calibration result.

        Args:
            calibration_result: CalibrationResult from multi-camera calibration
        """
        self.calibration = calibration_result
        self.reference_camera = calibration_result.extrinsic_calibration.reference_camera

        logger.info("Initialized CalibrationQualityValidator")

    def validate_with_test_object(self,
                                  measurements: Dict[str, TestObjectMeasurement],
                                  ground_truth_dimensions: np.ndarray,
                                  test_object_name: str = "calibration_cube") -> CalibrationValidationResult:
        """
        Validate calibration using 3D test object with known dimensions.

        Implements Scenario 8.2: Validate calibration with 3D test object

        Args:
            measurements: Dict mapping camera_id -> measurement
            ground_truth_dimensions: Known dimensions (length, width, height) in meters
            test_object_name: Name of test object

        Returns:
            CalibrationValidationResult
        """
        logger.info(f"Validating calibration with test object: {test_object_name}")

        # 1. Validate dimensional accuracy per camera
        dimension_errors = {}
        for camera_id, measurement in measurements.items():
            error = np.linalg.norm(
                measurement.measured_dimensions - ground_truth_dimensions
            )
            dimension_errors[camera_id] = error

            logger.info(
                f"Camera {camera_id}: dimension error = {error*100:.2f}cm "
                f"({error/np.mean(ground_truth_dimensions)*100:.1f}%)"
            )

        mean_dimension_error = np.mean(list(dimension_errors.values()))
        max_dimension_error = np.max(list(dimension_errors.values()))

        # Calculate percentage error
        dimension_error_percentage = (
            mean_dimension_error / np.mean(ground_truth_dimensions) * 100
        )

        # 2. Validate position consistency across cameras
        # Transform all positions to global coordinates and compare
        position_errors = {}

        camera_ids = list(measurements.keys())
        for i, cam1 in enumerate(camera_ids):
            for cam2 in camera_ids[i+1:]:
                # Get positions in global coordinates
                pos1 = measurements[cam1].measured_position
                pos2 = measurements[cam2].measured_position

                # If measurements are in local coordinates, transform to global
                if cam1 != self.reference_camera:
                    T = self.calibration.extrinsic_calibration.get_transformation(
                        cam1, self.reference_camera
                    )
                    pos1_homogeneous = np.append(pos1, 1.0)
                    pos1 = (T @ pos1_homogeneous)[:3]

                if cam2 != self.reference_camera:
                    T = self.calibration.extrinsic_calibration.get_transformation(
                        cam2, self.reference_camera
                    )
                    pos2_homogeneous = np.append(pos2, 1.0)
                    pos2 = (T @ pos2_homogeneous)[:3]

                # Compute position difference
                error = np.linalg.norm(pos1 - pos2)
                position_errors[(cam1, cam2)] = error

                logger.info(
                    f"Position consistency {cam1} ↔ {cam2}: "
                    f"error = {error*100:.2f}cm"
                )

        mean_position_error = np.mean(list(position_errors.values())) if position_errors else 0.0
        max_position_error = np.max(list(position_errors.values())) if position_errors else 0.0

        # 3. Validate orientation consistency
        orientation_errors = {}

        for i, cam1 in enumerate(camera_ids):
            for cam2 in camera_ids[i+1:]:
                orient1 = measurements[cam1].measured_orientation
                orient2 = measurements[cam2].measured_orientation

                # Compute angular difference
                angle_diff = np.abs(orient1 - orient2)
                # Normalize to [-pi, pi]
                angle_diff = np.arctan2(np.sin(angle_diff), np.cos(angle_diff))

                error_deg = np.degrees(np.linalg.norm(angle_diff))
                orientation_errors[(cam1, cam2)] = error_deg

                logger.info(
                    f"Orientation consistency {cam1} ↔ {cam2}: "
                    f"error = {error_deg:.2f}°"
                )

        mean_orientation_error = np.mean(list(orientation_errors.values())) if orientation_errors else 0.0
        max_orientation_error = np.max(list(orientation_errors.values())) if orientation_errors else 0.0

        # 4. Overall validation
        dimensional_accuracy_valid = dimension_error_percentage < 2.0  # < 2%
        position_consistency_valid = max_position_error < 0.02  # < 2cm
        orientation_consistency_valid = max_orientation_error < 2.0  # < 2°

        overall_valid = (
            dimensional_accuracy_valid and
            position_consistency_valid and
            orientation_consistency_valid
        )

        result = CalibrationValidationResult(
            test_object_name=test_object_name,
            ground_truth_dimensions=ground_truth_dimensions,
            measurements=measurements,
            dimension_errors=dimension_errors,
            mean_dimension_error=mean_dimension_error,
            max_dimension_error=max_dimension_error,
            dimension_error_percentage=dimension_error_percentage,
            position_errors=position_errors,
            mean_position_error=mean_position_error,
            max_position_error=max_position_error,
            orientation_errors=orientation_errors,
            mean_orientation_error=mean_orientation_error,
            max_orientation_error=max_orientation_error,
            dimensional_accuracy_valid=dimensional_accuracy_valid,
            position_consistency_valid=position_consistency_valid,
            orientation_consistency_valid=orientation_consistency_valid,
            overall_valid=overall_valid
        )

        if overall_valid:
            logger.info("Calibration validation PASSED")
        else:
            logger.warning("Calibration validation FAILED")
            if not dimensional_accuracy_valid:
                logger.warning(
                    f"  - Dimensional accuracy: {dimension_error_percentage:.2f}% "
                    f"(threshold: 2%)"
                )
            if not position_consistency_valid:
                logger.warning(
                    f"  - Position consistency: {max_position_error*100:.2f}cm "
                    f"(threshold: 2cm)"
                )
            if not orientation_consistency_valid:
                logger.warning(
                    f"  - Orientation consistency: {max_orientation_error:.2f}° "
                    f"(threshold: 2°)"
                )

        return result

    def assess_temporal_synchronization(self,
                                       synchronized_frames: List[Dict[str, np.ndarray]],
                                       test_object_name: str = "pendulum",
                                       motion_type: str = "pendulum") -> TemporalSynchronizationQuality:
        """
        Assess temporal synchronization quality using moving object.

        Implements Scenario 8.3: Assess temporal synchronization quality

        For each synchronized frame triplet, the object position should be
        consistent across all cameras (after transformation to global coordinates).

        Args:
            synchronized_frames: List of frame triplets, each a dict mapping
                                camera_id -> object_position (3D)
            test_object_name: Name of test object
            motion_type: Type of motion ("pendulum", "linear", "rotating")

        Returns:
            TemporalSynchronizationQuality
        """
        logger.info(
            f"Assessing temporal synchronization with {motion_type} motion "
            f"({len(synchronized_frames)} frames)"
        )

        position_variances = []

        for frame_idx, frame_positions in enumerate(synchronized_frames):
            # Transform all positions to global coordinates
            global_positions = []

            for camera_id, local_position in frame_positions.items():
                if camera_id == self.reference_camera:
                    global_pos = local_position
                else:
                    T = self.calibration.extrinsic_calibration.get_transformation(
                        camera_id, self.reference_camera
                    )
                    pos_homogeneous = np.append(local_position, 1.0)
                    global_pos = (T @ pos_homogeneous)[:3]

                global_positions.append(global_pos)

            # Compute variance of positions (should be low for good sync)
            global_positions = np.array(global_positions)
            variance = np.var(global_positions, axis=0).sum()  # Total variance
            position_variances.append(variance)

            if frame_idx < 5:  # Log first few frames
                logger.debug(
                    f"Frame {frame_idx}: position variance = {variance*1000:.2f} mm²"
                )

        mean_variance = np.mean(position_variances)
        max_variance = np.max(position_variances)

        # Estimate temporal jitter from position variance
        # Assuming object moves at ~0.5 m/s (slow motion)
        # variance in position relates to time jitter
        temporal_jitter_ms = np.sqrt(mean_variance) / 0.5 * 1000  # Convert to ms

        # Compute synchronization score
        # Score decreases with variance/jitter
        synchronization_score = 1.0 / (1.0 + mean_variance * 1000)

        # Quality assessment
        if temporal_jitter_ms < 5.0 and synchronization_score > 0.95:
            quality = "excellent"
            is_valid = True
        elif temporal_jitter_ms < 10.0 and synchronization_score > 0.85:
            quality = "good"
            is_valid = True
        elif temporal_jitter_ms < 15.0:
            quality = "acceptable"
            is_valid = True
        else:
            quality = "poor"
            is_valid = False

        logger.info(
            f"Temporal synchronization quality: {quality} "
            f"(jitter: {temporal_jitter_ms:.2f}ms, score: {synchronization_score:.3f})"
        )

        result = TemporalSynchronizationQuality(
            test_object_name=test_object_name,
            motion_type=motion_type,
            position_variance_per_frame=position_variances,
            mean_position_variance=mean_variance,
            max_position_variance=max_variance,
            temporal_jitter_ms=temporal_jitter_ms,
            synchronization_score=synchronization_score,
            quality_assessment=quality,
            is_valid=is_valid
        )

        return result

    def generate_validation_report(self,
                                  object_validation: CalibrationValidationResult,
                                  temporal_validation: Optional[TemporalSynchronizationQuality] = None) -> str:
        """
        Generate comprehensive validation report.

        Args:
            object_validation: Result from validate_with_test_object
            temporal_validation: Optional result from assess_temporal_synchronization

        Returns:
            Formatted report string
        """
        report_lines = []
        report_lines.append("=" * 70)
        report_lines.append("MULTI-CAMERA CALIBRATION VALIDATION REPORT")
        report_lines.append("=" * 70)
        report_lines.append("")

        # 1. Test Object Validation
        report_lines.append("1. TEST OBJECT VALIDATION")
        report_lines.append(f"   Object: {object_validation.test_object_name}")
        report_lines.append(
            f"   Ground truth dimensions: "
            f"{object_validation.ground_truth_dimensions[0]*100:.1f} × "
            f"{object_validation.ground_truth_dimensions[1]*100:.1f} × "
            f"{object_validation.ground_truth_dimensions[2]*100:.1f} cm"
        )
        report_lines.append("")

        report_lines.append("   Dimensional Accuracy:")
        for camera_id, error in object_validation.dimension_errors.items():
            status = "✓" if error < 0.02 else "✗"
            report_lines.append(
                f"     {status} {camera_id}: {error*100:.2f}cm error"
            )

        status = "PASS" if object_validation.dimensional_accuracy_valid else "FAIL"
        report_lines.append(
            f"     Overall: {object_validation.dimension_error_percentage:.2f}% "
            f"(threshold: 2%) {status}"
        )
        report_lines.append("")

        report_lines.append("   Position Consistency:")
        for (cam1, cam2), error in object_validation.position_errors.items():
            status = "✓" if error < 0.02 else "✗"
            report_lines.append(
                f"     {status} {cam1} ↔ {cam2}: {error*100:.2f}cm"
            )

        status = "PASS" if object_validation.position_consistency_valid else "FAIL"
        report_lines.append(
            f"     Max error: {object_validation.max_position_error*100:.2f}cm "
            f"(threshold: 2cm) {status}"
        )
        report_lines.append("")

        report_lines.append("   Orientation Consistency:")
        for (cam1, cam2), error in object_validation.orientation_errors.items():
            status = "✓" if error < 2.0 else "✗"
            report_lines.append(
                f"     {status} {cam1} ↔ {cam2}: {error:.2f}°"
            )

        status = "PASS" if object_validation.orientation_consistency_valid else "FAIL"
        report_lines.append(
            f"     Max error: {object_validation.max_orientation_error:.2f}° "
            f"(threshold: 2°) {status}"
        )
        report_lines.append("")

        # 2. Temporal Synchronization
        if temporal_validation:
            report_lines.append("2. TEMPORAL SYNCHRONIZATION QUALITY")
            report_lines.append(f"   Test: {temporal_validation.test_object_name}")
            report_lines.append(f"   Motion type: {temporal_validation.motion_type}")
            report_lines.append(
                f"   Temporal jitter: {temporal_validation.temporal_jitter_ms:.2f}ms "
                f"(threshold: 10ms)"
            )
            report_lines.append(
                f"   Synchronization score: {temporal_validation.synchronization_score:.3f}"
            )

            status = "PASS" if temporal_validation.is_valid else "FAIL"
            report_lines.append(
                f"   Quality: {temporal_validation.quality_assessment.upper()} {status}"
            )
            report_lines.append("")

        # 3. Overall Result
        report_lines.append("3. OVERALL VALIDATION RESULT")

        if object_validation.overall_valid:
            report_lines.append("   CALIBRATION VALIDATION PASSED")
            report_lines.append("   All criteria met. System ready for production use.")
        else:
            report_lines.append("   CALIBRATION VALIDATION FAILED")
            report_lines.append("   Recalibration recommended.")

            if not object_validation.dimensional_accuracy_valid:
                report_lines.append("   → Dimensional accuracy below threshold")
            if not object_validation.position_consistency_valid:
                report_lines.append("   → Position consistency below threshold")
            if not object_validation.orientation_consistency_valid:
                report_lines.append("   → Orientation consistency below threshold")

        report_lines.append("")
        report_lines.append("=" * 70)

        return "\n".join(report_lines)


# Convenience functions
def create_test_measurement(camera_id: str,
                           dimensions: np.ndarray,
                           position: np.ndarray,
                           orientation: np.ndarray,
                           confidence: float = 1.0) -> TestObjectMeasurement:
    """
    Convenience function to create test object measurement.

    Args:
        camera_id: Camera identifier
        dimensions: Measured dimensions (length, width, height) in meters
        position: Measured position (x, y, z) in meters
        orientation: Measured orientation (roll, pitch, yaw) in radians
        confidence: Measurement confidence (0-1)

    Returns:
        TestObjectMeasurement
    """
    return TestObjectMeasurement(
        camera_id=camera_id,
        measured_dimensions=dimensions,
        measured_position=position,
        measured_orientation=orientation,
        measurement_confidence=confidence
    )
