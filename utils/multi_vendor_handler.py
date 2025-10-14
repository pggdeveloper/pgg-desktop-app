"""
Multi-Vendor Camera Handler

Handles specific challenges when integrating cameras from different vendors:
- ZED 2i (Stereolabs) - Passive stereo
- RealSense D455i (Intel) - Active IR structured light

Challenges addressed:
- Different depth ranges
- Different depth sensing technologies
- Coordinate frame convention differences
- Depth map normalization and fusion

Part of Scenario 13 (Multi-Vendor Multi-Camera Integration)

Implements:
- Scenario 7.1: Handle different depth ranges
- Scenario 7.2: Compensate for coordinate frame conventions
- Scenario 7.3: Synchronize different depth technologies
"""

import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class DepthTechnology(Enum):
    """Depth sensing technology type"""
    PASSIVE_STEREO = "passive_stereo"  # ZED 2i
    ACTIVE_STRUCTURED_LIGHT = "active_structured_light"  # RealSense D455i
    TIME_OF_FLIGHT = "time_of_flight"  # Alternative tech


class Environment(Enum):
    """Environment type for depth sensing"""
    INDOOR = "indoor"
    OUTDOOR = "outdoor"
    MIXED = "mixed"


@dataclass
class CameraDepthConfig:
    """Depth configuration for a camera"""
    camera_id: str
    technology: DepthTechnology
    depth_range_min_m: float  # Minimum depth in meters
    depth_range_max_m: float  # Maximum depth in meters
    optimal_range_min_m: float  # Optimal performance range min
    optimal_range_max_m: float  # Optimal performance range max
    confidence_threshold: float  # Minimum confidence for valid depth
    works_outdoor: bool  # Whether it works in outdoor conditions


# Default configurations for supported cameras
DEFAULT_CONFIGS = {
    "zed_2i": CameraDepthConfig(
        camera_id="zed_2i",
        technology=DepthTechnology.PASSIVE_STEREO,
        depth_range_min_m=0.3,
        depth_range_max_m=20.0,
        optimal_range_min_m=0.5,
        optimal_range_max_m=15.0,
        confidence_threshold=50.0,  # ZED confidence score
        works_outdoor=True
    ),
    "realsense_d455i": CameraDepthConfig(
        camera_id="realsense_d455i",
        technology=DepthTechnology.ACTIVE_STRUCTURED_LIGHT,
        depth_range_min_m=0.4,
        depth_range_max_m=6.0,
        optimal_range_min_m=0.6,
        optimal_range_max_m=4.0,
        confidence_threshold=2.0,  # RealSense confidence (lower is better)
        works_outdoor=False  # IR struggles in sunlight
    )
}


class MultiVendorDepthHandler:
    """
    Handle depth data from multiple camera vendors.

    Features:
    - Depth range normalization
    - Confidence threshold application
    - Invalid depth masking
    - Adaptive fusion weights by environment
    - Technology-specific processing

    Usage:
        handler = MultiVendorDepthHandler()

        # Configure cameras
        handler.add_camera("zed_2i_0", DEFAULT_CONFIGS["zed_2i"])
        handler.add_camera("realsense_d455i_0", DEFAULT_CONFIGS["realsense_d455i"])

        # Normalize depth maps
        depth_normalized = handler.normalize_depth(depth_raw, "zed_2i_0")

        # Get fusion weights
        weights = handler.get_fusion_weights(Environment.INDOOR)
    """

    def __init__(self):
        """Initialize multi-vendor depth handler"""
        self.camera_configs: Dict[str, CameraDepthConfig] = {}
        logger.info("Initialized MultiVendorDepthHandler")

    def add_camera(self, camera_id: str, config: CameraDepthConfig):
        """
        Add camera configuration.

        Args:
            camera_id: Camera identifier
            config: Depth configuration
        """
        self.camera_configs[camera_id] = config
        logger.info(f"Added camera {camera_id} ({config.technology.value})")

    def add_camera_by_type(self, camera_id: str, camera_type: str):
        """
        Add camera using default configuration for type.

        Args:
            camera_id: Camera identifier
            camera_type: "zed_2i" or "realsense_d455i"
        """
        if camera_type not in DEFAULT_CONFIGS:
            raise ValueError(f"Unknown camera type: {camera_type}")

        config = DEFAULT_CONFIGS[camera_type]
        # Create copy with specific ID
        config_copy = CameraDepthConfig(
            camera_id=camera_id,
            technology=config.technology,
            depth_range_min_m=config.depth_range_min_m,
            depth_range_max_m=config.depth_range_max_m,
            optimal_range_min_m=config.optimal_range_min_m,
            optimal_range_max_m=config.optimal_range_max_m,
            confidence_threshold=config.confidence_threshold,
            works_outdoor=config.works_outdoor
        )

        self.add_camera(camera_id, config_copy)

    def normalize_depth(self,
                       depth_map: np.ndarray,
                       camera_id: str,
                       confidence_map: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Normalize depth map to meters with quality filtering.

        Implements Scenario 7.1: Handle different depth ranges

        Args:
            depth_map: Raw depth map (units depend on camera)
            camera_id: Camera identifier
            confidence_map: Optional confidence/quality map

        Returns:
            Normalized depth map in meters (0 = invalid)
        """
        if camera_id not in self.camera_configs:
            raise ValueError(f"Unknown camera: {camera_id}")

        config = self.camera_configs[camera_id]

        # Copy depth map
        depth_normalized = depth_map.copy().astype(np.float32)

        # Apply confidence threshold if available
        if confidence_map is not None:
            if config.technology == DepthTechnology.ACTIVE_STRUCTURED_LIGHT:
                # RealSense: lower confidence is better (std dev)
                invalid_mask = confidence_map > config.confidence_threshold
            else:
                # ZED: higher confidence is better (score)
                invalid_mask = confidence_map < config.confidence_threshold

            depth_normalized[invalid_mask] = 0.0

        # Mask depth values outside valid range
        depth_normalized[depth_normalized < config.depth_range_min_m] = 0.0
        depth_normalized[depth_normalized > config.depth_range_max_m] = 0.0

        # Mask zero/NaN values
        depth_normalized[np.isnan(depth_normalized)] = 0.0
        depth_normalized[depth_normalized <= 0] = 0.0

        num_valid = np.count_nonzero(depth_normalized)
        total_pixels = depth_normalized.size
        valid_ratio = num_valid / total_pixels

        logger.debug(
            f"Normalized depth for {camera_id}: "
            f"{valid_ratio*100:.1f}% valid pixels"
        )

        return depth_normalized

    def get_fusion_weights(self,
                          environment: Environment,
                          distance_map: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Get fusion weights for depth maps based on environment and distance.

        Implements Scenario 7.3: Synchronize different depth technologies

        Strategy:
        - Indoor: Prefer RealSense (more accurate close range)
        - Outdoor: Use only ZED (RealSense IR fails in sunlight)
        - Close range (<3m): Prefer RealSense
        - Far range (>3m): Prefer ZED

        Args:
            environment: Environment type
            distance_map: Optional distance map for adaptive weighting

        Returns:
            Dict mapping camera_id -> weight (0-1)
        """
        weights = {}

        # Get cameras by technology
        realsense_cameras = [
            cam_id for cam_id, cfg in self.camera_configs.items()
            if cfg.technology == DepthTechnology.ACTIVE_STRUCTURED_LIGHT
        ]

        zed_cameras = [
            cam_id for cam_id, cfg in self.camera_configs.items()
            if cfg.technology == DepthTechnology.PASSIVE_STEREO
        ]

        # Environment-based weights
        if environment == Environment.OUTDOOR:
            # Outdoor: ZED only (RealSense fails)
            for cam in realsense_cameras:
                weights[cam] = 0.0
            for cam in zed_cameras:
                weights[cam] = 1.0

        elif environment == Environment.INDOOR:
            # Indoor: prefer RealSense close, ZED far
            for cam in realsense_cameras:
                weights[cam] = 0.7  # Higher weight close range
            for cam in zed_cameras:
                weights[cam] = 0.3  # Lower weight, but still contributes

        else:  # MIXED
            # Balanced weights
            for cam in realsense_cameras:
                weights[cam] = 0.5
            for cam in zed_cameras:
                weights[cam] = 0.5

        logger.debug(f"Fusion weights for {environment.value}: {weights}")

        return weights

    def get_adaptive_fusion_weights(self,
                                   depth_map: np.ndarray,
                                   camera_id: str) -> np.ndarray:
        """
        Get per-pixel adaptive fusion weights based on depth.

        For each pixel, weight depends on:
        - Distance (RealSense better <3m, ZED better >3m)
        - Optimal range of camera

        Args:
            depth_map: Depth map in meters
            camera_id: Camera identifier

        Returns:
            Weight map (0-1) same size as depth_map
        """
        if camera_id not in self.camera_configs:
            return np.ones_like(depth_map)

        config = self.camera_configs[camera_id]

        # Initialize weights
        weights = np.zeros_like(depth_map)

        # Compute weights based on optimal range
        # Weight = 1.0 in optimal range, falls off outside

        optimal_min = config.optimal_range_min_m
        optimal_max = config.optimal_range_max_m

        # In optimal range: weight = 1.0
        in_optimal = (depth_map >= optimal_min) & (depth_map <= optimal_max)
        weights[in_optimal] = 1.0

        # Below optimal: linear falloff
        below_optimal = depth_map < optimal_min
        if np.any(below_optimal):
            # Weight decreases linearly from 1.0 at optimal_min to 0.0 at depth_range_min
            depth_below = depth_map[below_optimal]
            weights[below_optimal] = (depth_below - config.depth_range_min_m) / \
                                    (optimal_min - config.depth_range_min_m)

        # Above optimal: linear falloff
        above_optimal = depth_map > optimal_max
        if np.any(above_optimal):
            # Weight decreases linearly from 1.0 at optimal_max to 0.0 at depth_range_max
            depth_above = depth_map[above_optimal]
            weights[above_optimal] = (config.depth_range_max_m - depth_above) / \
                                    (config.depth_range_max_m - optimal_max)

        # Clip to [0, 1]
        weights = np.clip(weights, 0.0, 1.0)

        return weights

    def fuse_depth_maps(self,
                       depth_maps: Dict[str, np.ndarray],
                       environment: Environment,
                       use_adaptive_weights: bool = True) -> np.ndarray:
        """
        Fuse depth maps from multiple cameras.

        Args:
            depth_maps: Dict mapping camera_id -> depth_map (in meters)
            environment: Environment type
            use_adaptive_weights: Use per-pixel adaptive weights

        Returns:
            Fused depth map
        """
        if not depth_maps:
            raise ValueError("No depth maps provided")

        # Get first depth map for size
        first_depth = next(iter(depth_maps.values()))
        fused = np.zeros_like(first_depth)
        total_weights = np.zeros_like(first_depth)

        # Get base weights by environment
        base_weights = self.get_fusion_weights(environment)

        for camera_id, depth_map in depth_maps.items():
            if camera_id not in base_weights:
                logger.warning(f"No fusion weight for camera {camera_id}, skipping")
                continue

            base_weight = base_weights[camera_id]

            if base_weight == 0.0:
                continue

            # Get adaptive weights if enabled
            if use_adaptive_weights:
                adaptive_weights = self.get_adaptive_fusion_weights(depth_map, camera_id)
                pixel_weights = base_weight * adaptive_weights
            else:
                pixel_weights = np.full_like(depth_map, base_weight)

            # Only fuse valid depth pixels
            valid_mask = depth_map > 0

            fused[valid_mask] += depth_map[valid_mask] * pixel_weights[valid_mask]
            total_weights[valid_mask] += pixel_weights[valid_mask]

        # Normalize by total weights
        valid_fusion = total_weights > 0
        fused[valid_fusion] /= total_weights[valid_fusion]

        num_fused = np.count_nonzero(fused)
        logger.info(f"Fused depth maps: {num_fused} valid pixels")

        return fused


class CoordinateFrameConverter:
    """
    Handle coordinate frame convention differences between vendors.

    Implements Scenario 7.2: Compensate for coordinate frame conventions

    Both ZED 2i and RealSense D455i use right-handed, Z-forward frames,
    but there may be subtle differences that need verification.
    """

    @staticmethod
    def verify_frame_consistency(transformation_matrix: np.ndarray) -> Tuple[bool, str]:
        """
        Verify coordinate frame consistency in transformation matrix.

        Checks:
        - Right-handed coordinate system
        - Proper rotation matrix (orthogonal, det=1)

        Args:
            transformation_matrix: 4x4 transformation matrix

        Returns:
            (is_consistent, message)
        """
        R = transformation_matrix[:3, :3]

        # Check orthogonality
        RTR = R.T @ R
        I = np.eye(3)
        if not np.allclose(RTR, I, atol=1e-5):
            return False, "Rotation matrix not orthogonal"

        # Check determinant (should be +1 for right-handed, -1 for left-handed)
        det = np.linalg.det(R)
        if not np.isclose(det, 1.0, atol=1e-5):
            if np.isclose(det, -1.0, atol=1e-5):
                return False, "Left-handed coordinate system detected (determinant = -1)"
            else:
                return False, f"Invalid rotation matrix (determinant = {det})"

        # Check right-handed: X × Y = Z
        X = R[:, 0]
        Y = R[:, 1]
        Z = R[:, 2]
        cross_XY = np.cross(X, Y)

        if not np.allclose(cross_XY, Z, atol=1e-5):
            return False, "Not right-handed: X × Y ≠ Z"

        return True, "Coordinate frame consistent (right-handed)"


# Convenience functions
def create_default_handler_for_setup() -> MultiVendorDepthHandler:
    """
    Create handler with default configuration for standard setup:
    - 2x ZED 2i
    - 1x RealSense D455i
    """
    handler = MultiVendorDepthHandler()

    handler.add_camera_by_type("zed_2i_0", "zed_2i")
    handler.add_camera_by_type("zed_2i_1", "zed_2i")
    handler.add_camera_by_type("realsense_d455i_0", "realsense_d455i")

    return handler
