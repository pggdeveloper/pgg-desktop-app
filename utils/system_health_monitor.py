"""
System Health and Monitoring for Multi-Camera System

This module provides system health monitoring, calibration drift detection,
and pre-recording validation for long-term system reliability.

Part of Scenario 13 (Multi-Vendor Multi-Camera Integration)

Implements:
- Scenario 11.1: Monitor calibration drift over time
- Scenario 11.2: Detect camera misalignment/physical movement
- Scenario 11.3: Validate multi-camera system before recording
"""

import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class CalibrationDriftMeasurement:
    """Measurement of calibration drift"""
    camera_pair: Tuple[str, str]  # (camera_from, camera_to)
    baseline_date: str  # ISO format
    current_date: str  # ISO format
    translation_drift_m: float  # Translation change in meters
    rotation_drift_deg: float  # Rotation change in degrees
    exceeds_threshold: bool  # True if drift > threshold


@dataclass
class CalibrationDriftReport:
    """Report on calibration drift over time"""
    baseline_calibration_date: str
    current_calibration_date: str
    days_since_baseline: int
    drift_measurements: List[CalibrationDriftMeasurement]
    max_translation_drift_m: float
    max_rotation_drift_deg: float
    recalibration_needed: bool  # True if any drift exceeds threshold
    affected_cameras: List[str]  # Cameras with excessive drift


@dataclass
class LandmarkMeasurement:
    """Measurement of static landmark position"""
    landmark_id: str
    camera_id: str
    position: np.ndarray  # 3D position in camera frame
    timestamp: str  # ISO format
    confidence: float  # 0-1


@dataclass
class CameraMisalignmentReport:
    """Report on camera misalignment detection"""
    camera_id: str
    baseline_date: str
    current_date: str
    landmark_position_changes: Dict[str, float]  # landmark_id -> change (m)
    max_position_change_m: float
    misalignment_detected: bool  # True if change > threshold
    recommended_action: str  # "recalibrate", "inspect", "none"


@dataclass
class SystemHealthStatus:
    """Overall system health status"""
    timestamp: str
    cameras_healthy: Dict[str, bool]  # camera_id -> healthy
    calibration_valid: bool
    calibration_age_days: int
    drift_detected: bool
    misalignment_detected: bool
    disk_space_sufficient: bool
    overall_health: str  # "healthy", "warning", "critical"
    issues: List[str]  # List of detected issues
    recommendations: List[str]  # Recommended actions


class CalibrationDriftMonitor:
    """
    Monitor calibration drift over time.

    Implements Scenario 11.1: Monitor calibration drift over time
    """

    # Thresholds for recalibration
    TRANSLATION_THRESHOLD_M = 0.05  # 5cm
    ROTATION_THRESHOLD_DEG = 2.0  # 2 degrees

    def __init__(self, history_file: Optional[Path] = None):
        """
        Initialize drift monitor.

        Args:
            history_file: Path to calibration history JSON file
        """
        self.history_file = history_file
        self.calibration_history: List[Dict] = []

        if history_file and history_file.exists():
            self._load_history()

        logger.info("Initialized CalibrationDriftMonitor")

    def _load_history(self):
        """Load calibration history from file"""
        try:
            with open(self.history_file, 'r') as f:
                self.calibration_history = json.load(f)
            logger.info(f"Loaded {len(self.calibration_history)} calibration records")
        except Exception as e:
            logger.error(f"Failed to load calibration history: {e}")
            self.calibration_history = []

    def add_calibration_record(self, calibration_data: Dict):
        """
        Add calibration record to history.

        Args:
            calibration_data: Calibration data with transformations and metadata
        """
        record = {
            "date": datetime.now().isoformat(),
            "calibration_data": calibration_data
        }

        self.calibration_history.append(record)

        # Save to file
        if self.history_file:
            self._save_history()

        logger.info(f"Added calibration record: {record['date']}")

    def _save_history(self):
        """Save calibration history to file"""
        try:
            self.history_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.history_file, 'w') as f:
                json.dump(self.calibration_history, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save calibration history: {e}")

    def check_drift(self,
                   baseline_calibration: Dict,
                   current_calibration: Dict) -> CalibrationDriftReport:
        """
        Check calibration drift between baseline and current calibration.

        Args:
            baseline_calibration: Baseline calibration (from history)
            current_calibration: Current calibration

        Returns:
            CalibrationDriftReport
        """
        baseline_date = baseline_calibration.get('calibration_date', 'unknown')
        current_date = current_calibration.get('calibration_date', 'unknown')

        # Calculate days between calibrations
        try:
            baseline_dt = datetime.fromisoformat(baseline_date)
            current_dt = datetime.fromisoformat(current_date)
            days_diff = (current_dt - baseline_dt).days
        except:
            days_diff = 0

        # Get transformations from both calibrations
        baseline_transforms = baseline_calibration.get('extrinsic_calibration', {}).get('transformations', {})
        current_transforms = current_calibration.get('extrinsic_calibration', {}).get('transformations', {})

        # Compare each camera pair
        drift_measurements = []
        max_translation_drift = 0.0
        max_rotation_drift = 0.0
        affected_cameras = set()

        for key in baseline_transforms.keys():
            if key not in current_transforms:
                continue

            # Parse camera pair from key (e.g., "camera1_to_camera2")
            parts = key.split('_to_')
            if len(parts) != 2:
                continue

            camera_from, camera_to = parts

            # Get transformation matrices
            baseline_T = np.array(baseline_transforms[key])
            current_T = np.array(current_transforms[key])

            # Compute drift
            trans_drift, rot_drift = self._compute_transformation_drift(baseline_T, current_T)

            exceeds = (trans_drift > self.TRANSLATION_THRESHOLD_M or
                      rot_drift > self.ROTATION_THRESHOLD_DEG)

            measurement = CalibrationDriftMeasurement(
                camera_pair=(camera_from, camera_to),
                baseline_date=baseline_date,
                current_date=current_date,
                translation_drift_m=trans_drift,
                rotation_drift_deg=rot_drift,
                exceeds_threshold=exceeds
            )

            drift_measurements.append(measurement)

            # Track maximums
            max_translation_drift = max(max_translation_drift, trans_drift)
            max_rotation_drift = max(max_rotation_drift, rot_drift)

            if exceeds:
                affected_cameras.add(camera_from)
                affected_cameras.add(camera_to)

        # Determine if recalibration needed
        recalibration_needed = len(affected_cameras) > 0

        report = CalibrationDriftReport(
            baseline_calibration_date=baseline_date,
            current_calibration_date=current_date,
            days_since_baseline=days_diff,
            drift_measurements=drift_measurements,
            max_translation_drift_m=max_translation_drift,
            max_rotation_drift_deg=max_rotation_drift,
            recalibration_needed=recalibration_needed,
            affected_cameras=list(affected_cameras)
        )

        if recalibration_needed:
            logger.warning(
                f"Calibration drift detected: "
                f"translation={max_translation_drift*100:.2f}cm, "
                f"rotation={max_rotation_drift:.2f}°"
            )
        else:
            logger.info(
                f"Calibration drift within acceptable limits: "
                f"translation={max_translation_drift*100:.2f}cm, "
                f"rotation={max_rotation_drift:.2f}°"
            )

        return report

    def _compute_transformation_drift(self,
                                     baseline_T: np.ndarray,
                                     current_T: np.ndarray) -> Tuple[float, float]:
        """
        Compute drift between two transformation matrices.

        Returns:
            (translation_drift_m, rotation_drift_deg)
        """
        # Translation drift
        baseline_trans = baseline_T[:3, 3]
        current_trans = current_T[:3, 3]
        translation_drift = np.linalg.norm(current_trans - baseline_trans)

        # Rotation drift
        baseline_R = baseline_T[:3, :3]
        current_R = current_T[:3, :3]

        # Compute relative rotation
        R_diff = current_R @ baseline_R.T

        # Convert to angle-axis representation
        trace = np.trace(R_diff)
        angle_rad = np.arccos((trace - 1.0) / 2.0)
        rotation_drift_deg = np.degrees(angle_rad)

        return float(translation_drift), float(rotation_drift_deg)


class LandmarkBasedAlignmentMonitor:
    """
    Monitor camera alignment using static landmarks.

    Implements Scenario 11.2: Detect camera misalignment or physical movement
    """

    # Threshold for misalignment detection
    POSITION_CHANGE_THRESHOLD_M = 0.02  # 2cm

    def __init__(self, landmark_history_file: Optional[Path] = None):
        """
        Initialize alignment monitor.

        Args:
            landmark_history_file: Path to landmark measurement history
        """
        self.landmark_history_file = landmark_history_file
        self.landmark_measurements: Dict[str, List[LandmarkMeasurement]] = {}

        if landmark_history_file and landmark_history_file.exists():
            self._load_landmark_history()

        logger.info("Initialized LandmarkBasedAlignmentMonitor")

    def _load_landmark_history(self):
        """Load landmark measurement history from file"""
        try:
            with open(self.landmark_history_file, 'r') as f:
                data = json.load(f)

            # Reconstruct measurements
            for landmark_id, measurements_list in data.items():
                self.landmark_measurements[landmark_id] = [
                    LandmarkMeasurement(
                        landmark_id=m['landmark_id'],
                        camera_id=m['camera_id'],
                        position=np.array(m['position']),
                        timestamp=m['timestamp'],
                        confidence=m['confidence']
                    )
                    for m in measurements_list
                ]

            logger.info(f"Loaded landmark history for {len(self.landmark_measurements)} landmarks")
        except Exception as e:
            logger.error(f"Failed to load landmark history: {e}")

    def add_landmark_measurement(self, measurement: LandmarkMeasurement):
        """
        Add landmark measurement to history.

        Args:
            measurement: LandmarkMeasurement
        """
        key = f"{measurement.camera_id}_{measurement.landmark_id}"

        if key not in self.landmark_measurements:
            self.landmark_measurements[key] = []

        self.landmark_measurements[key].append(measurement)

        # Save to file
        if self.landmark_history_file:
            self._save_landmark_history()

        logger.debug(f"Added landmark measurement: {key} at {measurement.timestamp}")

    def _save_landmark_history(self):
        """Save landmark history to file"""
        try:
            # Convert to JSON-serializable format
            data = {}
            for key, measurements in self.landmark_measurements.items():
                data[key] = [
                    {
                        'landmark_id': m.landmark_id,
                        'camera_id': m.camera_id,
                        'position': m.position.tolist(),
                        'timestamp': m.timestamp,
                        'confidence': m.confidence
                    }
                    for m in measurements
                ]

            self.landmark_history_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.landmark_history_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save landmark history: {e}")

    def check_camera_alignment(self,
                              camera_id: str,
                              baseline_date: Optional[datetime] = None) -> CameraMisalignmentReport:
        """
        Check if camera has moved since baseline.

        Args:
            camera_id: Camera to check
            baseline_date: Baseline date (uses earliest if None)

        Returns:
            CameraMisalignmentReport
        """
        # Get all measurements for this camera
        camera_measurements = {
            key: meas for key, meas in self.landmark_measurements.items()
            if key.startswith(f"{camera_id}_")
        }

        if not camera_measurements:
            logger.warning(f"No landmark measurements found for camera {camera_id}")
            return CameraMisalignmentReport(
                camera_id=camera_id,
                baseline_date="unknown",
                current_date="unknown",
                landmark_position_changes={},
                max_position_change_m=0.0,
                misalignment_detected=False,
                recommended_action="none"
            )

        # Group by landmark and compute position changes
        position_changes = {}
        max_change = 0.0

        for key, measurements in camera_measurements.items():
            if len(measurements) < 2:
                continue

            # Sort by timestamp
            sorted_meas = sorted(measurements, key=lambda m: m.timestamp)

            # Get baseline (first) and current (last) measurements
            baseline_meas = sorted_meas[0]
            current_meas = sorted_meas[-1]

            # Compute position change
            position_change = np.linalg.norm(
                current_meas.position - baseline_meas.position
            )

            landmark_id = baseline_meas.landmark_id
            position_changes[landmark_id] = position_change
            max_change = max(max_change, position_change)

        # Determine misalignment
        misalignment_detected = max_change > self.POSITION_CHANGE_THRESHOLD_M

        # Recommend action
        if max_change > 0.10:  # > 10cm
            recommended_action = "recalibrate"
        elif misalignment_detected:
            recommended_action = "inspect"
        else:
            recommended_action = "none"

        # Get dates
        all_measurements = [m for meas_list in camera_measurements.values() for m in meas_list]
        if all_measurements:
            sorted_all = sorted(all_measurements, key=lambda m: m.timestamp)
            baseline_date_str = sorted_all[0].timestamp
            current_date_str = sorted_all[-1].timestamp
        else:
            baseline_date_str = "unknown"
            current_date_str = "unknown"

        report = CameraMisalignmentReport(
            camera_id=camera_id,
            baseline_date=baseline_date_str,
            current_date=current_date_str,
            landmark_position_changes=position_changes,
            max_position_change_m=max_change,
            misalignment_detected=misalignment_detected,
            recommended_action=recommended_action
        )

        if misalignment_detected:
            logger.warning(
                f"Camera {camera_id} misalignment detected: "
                f"max_change={max_change*100:.2f}cm"
            )
        else:
            logger.info(
                f"Camera {camera_id} alignment stable: "
                f"max_change={max_change*100:.2f}cm"
            )

        return report


class SystemHealthChecker:
    """
    Comprehensive system health checking.

    Implements Scenario 11.3: Validate multi-camera system before recording
    """

    def __init__(self):
        """Initialize system health checker"""
        logger.info("Initialized SystemHealthChecker")

    def check_system_health(self,
                           orchestrator,
                           min_disk_space_gb: float = 10.0) -> SystemHealthStatus:
        """
        Perform comprehensive system health check.

        Args:
            orchestrator: CameraRecordingOrchestrator instance
            min_disk_space_gb: Minimum required disk space

        Returns:
            SystemHealthStatus
        """
        timestamp = datetime.now().isoformat()
        issues = []
        recommendations = []

        # Check 1: Cameras healthy
        cameras_healthy = {}
        for camera in orchestrator.cameras:
            # TODO: Implement actual camera health check
            # For now, assume healthy if in camera list
            cameras_healthy[str(camera.camera_type)] = True

        all_cameras_healthy = all(cameras_healthy.values())
        if not all_cameras_healthy:
            issues.append("One or more cameras not responding")
            recommendations.append("Check camera connections and restart cameras")

        # Check 2: Calibration valid
        calibration_status = orchestrator.get_calibration_status()
        calibration_valid = calibration_status.get('valid', False)
        calibration_age_days = calibration_status.get('info', {}).get('age_days', 999)

        if not calibration_valid:
            issues.append("Calibration not loaded or invalid")
            recommendations.append("Load valid calibration file")

        # Check 3: Calibration age
        if calibration_age_days > 60:
            issues.append(f"Calibration is {calibration_age_days} days old")
            recommendations.append("Consider recalibration (recommended every 30 days)")

        # Check 4: Disk space
        disk_space_sufficient = self._check_disk_space(min_disk_space_gb)
        if not disk_space_sufficient:
            issues.append(f"Insufficient disk space (need {min_disk_space_gb}GB)")
            recommendations.append("Free up disk space before recording")

        # Check 5: Drift/misalignment (if available)
        # TODO: Integrate with drift monitor
        drift_detected = False
        misalignment_detected = False

        # Determine overall health
        if not calibration_valid or not disk_space_sufficient or not all_cameras_healthy:
            overall_health = "critical"
        elif len(issues) > 0:
            overall_health = "warning"
        else:
            overall_health = "healthy"

        status = SystemHealthStatus(
            timestamp=timestamp,
            cameras_healthy=cameras_healthy,
            calibration_valid=calibration_valid,
            calibration_age_days=calibration_age_days,
            drift_detected=drift_detected,
            misalignment_detected=misalignment_detected,
            disk_space_sufficient=disk_space_sufficient,
            overall_health=overall_health,
            issues=issues,
            recommendations=recommendations
        )

        if overall_health == "healthy":
            logger.info("System health check: HEALTHY")
        elif overall_health == "warning":
            logger.warning(f"System health check: WARNING - {len(issues)} issues")
        else:
            logger.error(f"System health check: CRITICAL - {len(issues)} issues")

        return status

    def _check_disk_space(self, min_required_gb: float) -> bool:
        """Check if sufficient disk space available"""
        try:
            import shutil
            stats = shutil.disk_usage('/')
            free_gb = stats.free / (1024 ** 3)
            return free_gb >= min_required_gb
        except Exception as e:
            logger.error(f"Failed to check disk space: {e}")
            return False


# Convenience functions
def generate_health_report(status: SystemHealthStatus) -> str:
    """
    Generate human-readable health report.

    Args:
        status: SystemHealthStatus

    Returns:
        Formatted health report string
    """
    lines = []
    lines.append("=" * 70)
    lines.append("MULTI-CAMERA SYSTEM HEALTH REPORT")
    lines.append("=" * 70)
    lines.append(f"Timestamp: {status.timestamp}")
    lines.append(f"Overall Health: {status.overall_health.upper()}")
    lines.append("")

    # Cameras
    lines.append("Camera Status:")
    for camera_id, healthy in status.cameras_healthy.items():
        status_str = "Healthy" if healthy else "Unhealthy"
        lines.append(f"  {camera_id}: {status_str}")
    lines.append("")

    # Calibration
    lines.append("Calibration:")
    calib_str = "Valid" if status.calibration_valid else "Invalid"
    lines.append(f"  Status: {calib_str}")
    lines.append(f"  Age: {status.calibration_age_days} days")
    lines.append("")

    # Drift/Misalignment
    lines.append("Alignment:")
    drift_str = "Detected" if status.drift_detected else "Stable"
    misalign_str = "Detected" if status.misalignment_detected else "Aligned"
    lines.append(f"  Calibration drift: {drift_str}")
    lines.append(f"  Camera misalignment: {misalign_str}")
    lines.append("")

    # Disk space
    lines.append("Storage:")
    disk_str = "Sufficient" if status.disk_space_sufficient else "Insufficient"
    lines.append(f"  Disk space: {disk_str}")
    lines.append("")

    # Issues
    if status.issues:
        lines.append("Issues:")
        for issue in status.issues:
            lines.append(f"  {issue}")
        lines.append("")

    # Recommendations
    if status.recommendations:
        lines.append("Recommendations:")
        for rec in status.recommendations:
            lines.append(f"  → {rec}")
        lines.append("")

    lines.append("=" * 70)

    return "\n".join(lines)
