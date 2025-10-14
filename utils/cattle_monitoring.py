"""
Cattle Monitoring Applications with Multi-Camera Fusion

This module provides cattle-specific applications leveraging multi-camera
360° reconstruction and global coordinate tracking.

Part of Scenario 13 (Multi-Vendor Multi-Camera Integration)

Implements:
- Scenario 10.1: 360° body volume estimation
- Scenario 10.2: Multi-angle BCS detection
- Scenario 10.3: Global coordinate tracking
- Scenario 10.4: Multi-view gait analysis for lameness detection
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class BCSRegion(Enum):
    """Body regions for BCS assessment"""
    SPINE = "spine"  # Dorsal view
    RIBS = "ribs"  # Side view
    HIPS = "hips"  # Rear view


class LamenessLevel(Enum):
    """Lameness severity levels"""
    NONE = 0  # Sound
    MILD = 1  # Slight lameness
    MODERATE = 2  # Obvious lameness
    SEVERE = 3  # Cannot bear weight


@dataclass
class BodyVolumeEstimate:
    """Result of 360° body volume estimation"""
    volume_m3: float  # Body volume in cubic meters
    estimated_weight_kg: float  # Weight estimate from volume
    point_count: int  # Number of points in merged cloud
    coverage_quality: str  # "excellent", "good", "acceptable", "poor"
    occlusion_percentage: float  # % of expected surface occluded
    confidence: float  # 0-1, confidence in estimate


@dataclass
class BCSMeasurement:
    """BCS measurement from single camera"""
    camera_id: str
    view_angle: str  # "dorsal", "side", "rear"
    spine_prominence: Optional[float]  # 0-1, higher = more prominent
    rib_visibility: Optional[float]  # 0-1, higher = more visible
    hip_prominence: Optional[float]  # 0-1, higher = more prominent
    confidence: float  # 0-1


@dataclass
class BCSResult:
    """Multi-view BCS assessment result"""
    bcs_score: float  # 1-9 scale (industry standard)
    measurements: Dict[str, BCSMeasurement]  # camera_id -> measurement
    confidence: float  # 0-1, overall confidence
    quality: str  # "excellent", "good", "acceptable", "poor"
    best_camera_per_region: Dict[BCSRegion, str]  # region -> camera_id


@dataclass
class TrackingPoint:
    """Cattle position at specific timestamp"""
    timestamp: float  # Epoch timestamp
    position: np.ndarray  # Global coordinates (x, y, z)
    camera_id: str  # Camera that detected position
    confidence: float  # 0-1


@dataclass
class TrackingResult:
    """Cattle tracking result in global coordinates"""
    trajectory: List[TrackingPoint]  # Temporal sequence of positions
    total_distance_m: float  # Total distance traveled
    average_speed_mps: float  # Average speed (m/s)
    max_speed_mps: float  # Maximum speed
    duration_seconds: float  # Tracking duration
    camera_handoffs: int  # Number of handoffs between cameras


@dataclass
class GaitFeatures:
    """Gait analysis features from single camera"""
    camera_id: str
    stride_length_m: float  # Stride length
    stride_frequency_hz: float  # Steps per second
    stride_symmetry: float  # 0-1, 1 = perfect symmetry
    head_bobbing_amplitude_m: float  # Vertical head movement
    weight_distribution_lr: float  # -1 to 1, left/right balance


@dataclass
class LamenessResult:
    """Multi-view lameness assessment"""
    lameness_score: float  # 0-3 scale
    lameness_level: LamenessLevel
    affected_limb: Optional[str]  # "front_left", "front_right", "rear_left", "rear_right"
    gait_features: Dict[str, GaitFeatures]  # camera_id -> features
    confidence: float  # 0-1
    quality: str  # "excellent", "good", "acceptable", "poor"


class CattleVolumeEstimator:
    """
    Estimate cattle body volume from 360° point cloud fusion.

    Implements Scenario 10.1: Perform 360° body volume estimation
    """

    # Empirical density for cattle (kg/m³)
    # Average cattle body density ~1050 kg/m³ (slightly denser than water)
    CATTLE_DENSITY_KG_M3 = 1050.0

    def __init__(self):
        """Initialize volume estimator"""
        logger.info("Initialized CattleVolumeEstimator")

    def estimate_volume(self,
                       merged_point_cloud: np.ndarray,
                       voxel_size_m: float = 0.01) -> BodyVolumeEstimate:
        """
        Estimate body volume from merged 360° point cloud.

        Method:
        1. Voxelize point cloud
        2. Count occupied voxels
        3. Volume = voxel_count * voxel_volume
        4. Weight = volume * density

        Args:
            merged_point_cloud: Merged point cloud (N, 3) in global coordinates
            voxel_size_m: Voxel size for volume calculation (default: 1cm)

        Returns:
            BodyVolumeEstimate with volume and weight
        """
        if merged_point_cloud.shape[0] == 0:
            logger.error("Empty point cloud provided")
            return BodyVolumeEstimate(
                volume_m3=0.0,
                estimated_weight_kg=0.0,
                point_count=0,
                coverage_quality="poor",
                occlusion_percentage=100.0,
                confidence=0.0
            )

        point_count = merged_point_cloud.shape[0]

        # Voxelize point cloud
        voxel_coords = np.floor(merged_point_cloud / voxel_size_m).astype(np.int32)

        # Count unique voxels
        unique_voxels = np.unique(voxel_coords, axis=0)
        voxel_count = unique_voxels.shape[0]

        # Calculate volume
        voxel_volume = voxel_size_m ** 3
        volume_m3 = voxel_count * voxel_volume

        # Estimate weight using empirical density
        estimated_weight_kg = volume_m3 * self.CATTLE_DENSITY_KG_M3

        # Assess coverage quality
        # Typical cattle: ~600-1000kg → ~0.57-0.95 m³
        # Point density: expect >10,000 points for good coverage
        coverage_quality = self._assess_coverage_quality(point_count, volume_m3)

        # Estimate occlusion percentage
        # Expected point density: ~15,000 points/m² surface area
        # Cattle surface area ~4-6 m² → expect 60,000-90,000 points
        expected_points = 75000  # Midpoint
        occlusion_percentage = max(0.0, (1.0 - point_count / expected_points) * 100)

        # Confidence based on coverage and point density
        confidence = self._calculate_confidence(point_count, volume_m3, occlusion_percentage)

        logger.info(
            f"Volume estimate: {volume_m3:.3f} m³, "
            f"weight: {estimated_weight_kg:.1f} kg, "
            f"points: {point_count}, "
            f"quality: {coverage_quality}"
        )

        return BodyVolumeEstimate(
            volume_m3=volume_m3,
            estimated_weight_kg=estimated_weight_kg,
            point_count=point_count,
            coverage_quality=coverage_quality,
            occlusion_percentage=occlusion_percentage,
            confidence=confidence
        )

    def _assess_coverage_quality(self, point_count: int, volume_m3: float) -> str:
        """Assess point cloud coverage quality"""
        # Point density (points per cubic meter)
        if volume_m3 > 0:
            density = point_count / volume_m3
        else:
            return "poor"

        # Quality thresholds
        if density > 50000:  # Very dense
            return "excellent"
        elif density > 30000:
            return "good"
        elif density > 15000:
            return "acceptable"
        else:
            return "poor"

    def _calculate_confidence(self, point_count: int, volume_m3: float,
                             occlusion_percentage: float) -> float:
        """Calculate confidence in volume estimate"""
        # Factors: point count, volume reasonableness, occlusion

        # Point count factor (0-1)
        point_factor = min(1.0, point_count / 50000)

        # Volume reasonableness (typical cattle: 0.5-1.0 m³)
        if 0.5 <= volume_m3 <= 1.0:
            volume_factor = 1.0
        elif 0.3 <= volume_m3 <= 1.5:
            volume_factor = 0.8
        else:
            volume_factor = 0.5

        # Occlusion factor
        occlusion_factor = max(0.0, 1.0 - occlusion_percentage / 100)

        # Combined confidence
        confidence = (point_factor * 0.4 + volume_factor * 0.3 + occlusion_factor * 0.3)

        return confidence


class MultiAngleBCSDetector:
    """
    Detect Body Condition Score from multiple camera angles.

    Implements Scenario 10.2: Detect BCS from multiple angles
    """

    def __init__(self):
        """Initialize BCS detector"""
        logger.info("Initialized MultiAngleBCSDetector")

    def assess_bcs(self,
                   measurements: Dict[str, BCSMeasurement]) -> BCSResult:
        """
        Assess BCS from multiple camera measurements.

        BCS Scale (1-9):
        1-3: Thin (spine/ribs very prominent)
        4-6: Ideal (moderate coverage)
        7-9: Fat (minimal bone visibility)

        Args:
            measurements: Dict mapping camera_id -> BCSMeasurement

        Returns:
            BCSResult with fused BCS score
        """
        if not measurements:
            logger.error("No measurements provided")
            return BCSResult(
                bcs_score=5.0,
                measurements={},
                confidence=0.0,
                quality="poor",
                best_camera_per_region={}
            )

        # Select best camera for each region
        best_cameras = self._select_best_cameras(measurements)

        # Extract features from best views
        spine_prominence = self._get_best_feature(
            measurements, best_cameras.get(BCSRegion.SPINE),
            lambda m: m.spine_prominence
        )

        rib_visibility = self._get_best_feature(
            measurements, best_cameras.get(BCSRegion.RIBS),
            lambda m: m.rib_visibility
        )

        hip_prominence = self._get_best_feature(
            measurements, best_cameras.get(BCSRegion.HIPS),
            lambda m: m.hip_prominence
        )

        # Compute BCS score from features
        bcs_score = self._compute_bcs_score(
            spine_prominence, rib_visibility, hip_prominence
        )

        # Assess quality and confidence
        confidence = self._calculate_bcs_confidence(measurements, best_cameras)
        quality = self._assess_bcs_quality(confidence)

        logger.info(
            f"BCS assessment: score={bcs_score:.1f}, "
            f"confidence={confidence:.2f}, quality={quality}"
        )

        return BCSResult(
            bcs_score=bcs_score,
            measurements=measurements,
            confidence=confidence,
            quality=quality,
            best_camera_per_region=best_cameras
        )

    def _select_best_cameras(self,
                            measurements: Dict[str, BCSMeasurement]) -> Dict[BCSRegion, str]:
        """Select best camera for each body region"""
        best = {}

        # Group by view angle
        dorsal = {k: v for k, v in measurements.items() if v.view_angle == "dorsal"}
        side = {k: v for k, v in measurements.items() if v.view_angle == "side"}
        rear = {k: v for k, v in measurements.items() if v.view_angle == "rear"}

        # Select highest confidence for each region
        if dorsal:
            best[BCSRegion.SPINE] = max(dorsal.keys(), key=lambda k: dorsal[k].confidence)
        if side:
            best[BCSRegion.RIBS] = max(side.keys(), key=lambda k: side[k].confidence)
        if rear:
            best[BCSRegion.HIPS] = max(rear.keys(), key=lambda k: rear[k].confidence)

        return best

    def _get_best_feature(self, measurements: Dict[str, BCSMeasurement],
                         camera_id: Optional[str],
                         feature_fn) -> Optional[float]:
        """Extract feature from best camera"""
        if camera_id and camera_id in measurements:
            return feature_fn(measurements[camera_id])
        return None

    def _compute_bcs_score(self,
                          spine_prominence: Optional[float],
                          rib_visibility: Optional[float],
                          hip_prominence: Optional[float]) -> float:
        """
        Compute BCS score from anatomical features.

        Mapping:
        - High prominence/visibility → Low BCS (thin)
        - Low prominence/visibility → High BCS (fat)
        """
        features = []
        weights = []

        if spine_prominence is not None:
            # Invert: high prominence = low BCS
            features.append(1.0 - spine_prominence)
            weights.append(0.4)

        if rib_visibility is not None:
            features.append(1.0 - rib_visibility)
            weights.append(0.3)

        if hip_prominence is not None:
            features.append(1.0 - hip_prominence)
            weights.append(0.3)

        if not features:
            return 5.0  # Default to mid-range

        # Weighted average
        feature_score = np.average(features, weights=weights)

        # Map [0, 1] to BCS [1, 9]
        bcs_score = 1.0 + feature_score * 8.0

        return float(bcs_score)

    def _calculate_bcs_confidence(self,
                                 measurements: Dict[str, BCSMeasurement],
                                 best_cameras: Dict[BCSRegion, str]) -> float:
        """Calculate confidence in BCS assessment"""
        # Average confidence from best cameras
        confidences = []

        for region, camera_id in best_cameras.items():
            if camera_id in measurements:
                confidences.append(measurements[camera_id].confidence)

        if confidences:
            return float(np.mean(confidences))
        return 0.0

    def _assess_bcs_quality(self, confidence: float) -> str:
        """Assess BCS quality from confidence"""
        if confidence > 0.8:
            return "excellent"
        elif confidence > 0.6:
            return "good"
        elif confidence > 0.4:
            return "acceptable"
        else:
            return "poor"


class GlobalCattleTracker:
    """
    Track cattle movement in global coordinates across multiple cameras.

    Implements Scenario 10.3: Track cattle movement in global coordinates
    """

    def __init__(self, position_smoothing_window: int = 5):
        """
        Initialize tracker.

        Args:
            position_smoothing_window: Window size for position smoothing
        """
        self.smoothing_window = position_smoothing_window
        logger.info("Initialized GlobalCattleTracker")

    def track_trajectory(self,
                        tracking_points: List[TrackingPoint]) -> TrackingResult:
        """
        Analyze cattle trajectory in global coordinates.

        Args:
            tracking_points: List of tracking points (already in global coordinates)

        Returns:
            TrackingResult with trajectory analysis
        """
        if len(tracking_points) < 2:
            logger.error("Insufficient tracking points for trajectory analysis")
            return TrackingResult(
                trajectory=[],
                total_distance_m=0.0,
                average_speed_mps=0.0,
                max_speed_mps=0.0,
                duration_seconds=0.0,
                camera_handoffs=0
            )

        # Sort by timestamp
        sorted_points = sorted(tracking_points, key=lambda p: p.timestamp)

        # Smooth positions
        smoothed_trajectory = self._smooth_trajectory(sorted_points)

        # Calculate distances
        total_distance = 0.0
        max_speed = 0.0
        speeds = []

        for i in range(1, len(smoothed_trajectory)):
            p1 = smoothed_trajectory[i-1]
            p2 = smoothed_trajectory[i]

            # Distance
            distance = np.linalg.norm(p2.position - p1.position)
            total_distance += distance

            # Speed
            time_delta = p2.timestamp - p1.timestamp
            if time_delta > 0:
                speed = distance / time_delta
                speeds.append(speed)
                max_speed = max(max_speed, speed)

        # Duration
        duration = sorted_points[-1].timestamp - sorted_points[0].timestamp

        # Average speed
        avg_speed = np.mean(speeds) if speeds else 0.0

        # Count camera handoffs
        handoffs = sum(1 for i in range(1, len(sorted_points))
                      if sorted_points[i].camera_id != sorted_points[i-1].camera_id)

        logger.info(
            f"Trajectory: {len(smoothed_trajectory)} points, "
            f"distance={total_distance:.2f}m, "
            f"avg_speed={avg_speed:.2f}m/s, "
            f"handoffs={handoffs}"
        )

        return TrackingResult(
            trajectory=smoothed_trajectory,
            total_distance_m=total_distance,
            average_speed_mps=avg_speed,
            max_speed_mps=max_speed,
            duration_seconds=duration,
            camera_handoffs=handoffs
        )

    def _smooth_trajectory(self, points: List[TrackingPoint]) -> List[TrackingPoint]:
        """Apply moving average smoothing to trajectory"""
        if len(points) <= self.smoothing_window:
            return points

        smoothed = []

        for i, point in enumerate(points):
            # Get window
            start = max(0, i - self.smoothing_window // 2)
            end = min(len(points), i + self.smoothing_window // 2 + 1)

            window_points = points[start:end]

            # Weighted average (higher weight for closer points)
            weights = np.exp(-np.abs(np.arange(len(window_points)) - (i - start)))
            weights /= weights.sum()

            positions = np.array([p.position for p in window_points])
            smoothed_position = np.average(positions, axis=0, weights=weights)

            smoothed_point = TrackingPoint(
                timestamp=point.timestamp,
                position=smoothed_position,
                camera_id=point.camera_id,
                confidence=point.confidence
            )
            smoothed.append(smoothed_point)

        return smoothed


class MultiViewGaitAnalyzer:
    """
    Analyze cattle gait from multiple views for lameness detection.

    Implements Scenario 10.4: Multi-view gait analysis for lameness
    """

    def __init__(self):
        """Initialize gait analyzer"""
        logger.info("Initialized MultiViewGaitAnalyzer")

    def analyze_lameness(self,
                        gait_features: Dict[str, GaitFeatures]) -> LamenessResult:
        """
        Analyze lameness from multi-view gait features.

        Lameness indicators:
        - Asymmetric stride (overhead view)
        - Head bobbing (side view)
        - Uneven weight distribution

        Args:
            gait_features: Dict mapping camera_id -> GaitFeatures

        Returns:
            LamenessResult with lameness assessment
        """
        if not gait_features:
            logger.error("No gait features provided")
            return LamenessResult(
                lameness_score=0.0,
                lameness_level=LamenessLevel.NONE,
                affected_limb=None,
                gait_features={},
                confidence=0.0,
                quality="poor"
            )

        # Extract key indicators
        stride_asymmetry = self._compute_stride_asymmetry(gait_features)
        head_bobbing = self._compute_head_bobbing_severity(gait_features)
        weight_imbalance = self._compute_weight_imbalance(gait_features)

        # Compute lameness score (0-3)
        lameness_score = self._compute_lameness_score(
            stride_asymmetry, head_bobbing, weight_imbalance
        )

        # Classify lameness level
        lameness_level = self._classify_lameness_level(lameness_score)

        # Identify affected limb
        affected_limb = self._identify_affected_limb(gait_features, weight_imbalance)

        # Confidence
        confidence = self._calculate_lameness_confidence(gait_features)
        quality = self._assess_lameness_quality(confidence)

        logger.info(
            f"Lameness analysis: score={lameness_score:.2f}, "
            f"level={lameness_level.name}, "
            f"affected={affected_limb}, "
            f"confidence={confidence:.2f}"
        )

        return LamenessResult(
            lameness_score=lameness_score,
            lameness_level=lameness_level,
            affected_limb=affected_limb,
            gait_features=gait_features,
            confidence=confidence,
            quality=quality
        )

    def _compute_stride_asymmetry(self, features: Dict[str, GaitFeatures]) -> float:
        """Compute stride asymmetry (0 = symmetric, 1 = very asymmetric)"""
        # Get stride symmetry from overhead views
        symmetries = [f.stride_symmetry for f in features.values()
                     if f.stride_symmetry is not None]

        if symmetries:
            # Average symmetry, then invert to asymmetry
            avg_symmetry = np.mean(symmetries)
            return 1.0 - avg_symmetry
        return 0.0

    def _compute_head_bobbing_severity(self, features: Dict[str, GaitFeatures]) -> float:
        """Compute head bobbing severity (0 = none, 1 = severe)"""
        # Get head bobbing from side views
        bobbings = [f.head_bobbing_amplitude_m for f in features.values()
                   if f.head_bobbing_amplitude_m is not None]

        if bobbings:
            # Normalize to [0, 1] (>10cm bobbing = severe)
            avg_bobbing = np.mean(bobbings)
            return min(1.0, avg_bobbing / 0.10)
        return 0.0

    def _compute_weight_imbalance(self, features: Dict[str, GaitFeatures]) -> float:
        """Compute weight distribution imbalance"""
        # Get weight distribution (left/right)
        distributions = [f.weight_distribution_lr for f in features.values()
                        if f.weight_distribution_lr is not None]

        if distributions:
            # Absolute imbalance
            avg_distribution = np.mean(distributions)
            return abs(avg_distribution)
        return 0.0

    def _compute_lameness_score(self,
                               stride_asymmetry: float,
                               head_bobbing: float,
                               weight_imbalance: float) -> float:
        """
        Compute overall lameness score (0-3 scale).

        Weighted combination of indicators.
        """
        # Weights
        w_stride = 0.4
        w_head = 0.3
        w_weight = 0.3

        # Combine
        score = (stride_asymmetry * w_stride +
                head_bobbing * w_head +
                weight_imbalance * w_weight) * 3.0

        return float(score)

    def _classify_lameness_level(self, score: float) -> LamenessLevel:
        """Classify lameness level from score"""
        if score < 0.5:
            return LamenessLevel.NONE
        elif score < 1.5:
            return LamenessLevel.MILD
        elif score < 2.5:
            return LamenessLevel.MODERATE
        else:
            return LamenessLevel.SEVERE

    def _identify_affected_limb(self,
                               features: Dict[str, GaitFeatures],
                               weight_imbalance: float) -> Optional[str]:
        """Identify which limb is affected"""
        # Use weight distribution to identify side
        distributions = [f.weight_distribution_lr for f in features.values()
                        if f.weight_distribution_lr is not None]

        if not distributions or weight_imbalance < 0.2:
            return None

        avg_dist = np.mean(distributions)

        # Negative = left, positive = right
        side = "left" if avg_dist < 0 else "right"

        # TODO: Distinguish front/rear (requires additional analysis)
        # For now, return side only
        return f"side_{side}"

    def _calculate_lameness_confidence(self,
                                      features: Dict[str, GaitFeatures]) -> float:
        """Calculate confidence in lameness assessment"""
        # More cameras = higher confidence
        camera_count_factor = min(1.0, len(features) / 3.0)

        # Feature completeness
        complete_features = sum(
            1 for f in features.values()
            if f.stride_symmetry is not None
            and f.head_bobbing_amplitude_m is not None
            and f.weight_distribution_lr is not None
        )
        completeness_factor = complete_features / len(features) if features else 0.0

        confidence = (camera_count_factor * 0.4 + completeness_factor * 0.6)

        return confidence

    def _assess_lameness_quality(self, confidence: float) -> str:
        """Assess lameness detection quality"""
        if confidence > 0.8:
            return "excellent"
        elif confidence > 0.6:
            return "good"
        elif confidence > 0.4:
            return "acceptable"
        else:
            return "poor"
