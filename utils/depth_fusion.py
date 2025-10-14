"""
Depth Map Fusion for Multi-Vendor Cameras

This module provides advanced depth fusion capabilities for combining
depth maps from RealSense D455i (active IR) and ZED 2i (passive stereo).

Part of Scenario 13 (Multi-Vendor Multi-Camera Integration)

Implements:
- Scenario 12.2: Fuse depth from RealSense and ZED cameras

Leverages complementary characteristics of each sensor:
- RealSense: Accurate close range (<3m), struggles outdoors
- ZED: Good range (0.3-20m), works indoors/outdoors
"""

import numpy as np
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class FusionStrategy(Enum):
    """Depth fusion strategy"""
    WEIGHTED_AVERAGE = "weighted_average"  # Confidence-weighted average
    SELECTIVE = "selective"  # Choose best sensor per region
    BAYESIAN = "bayesian"  # Bayesian fusion with uncertainty
    MEDIAN = "median"  # Median of valid depths


@dataclass
class DepthSource:
    """Depth data from single camera"""
    camera_id: str
    depth_map: np.ndarray  # Depth in meters (H, W)
    confidence_map: Optional[np.ndarray]  # Confidence 0-1 (H, W)
    technology: str  # "active_ir" or "passive_stereo"
    timestamp: float  # Epoch timestamp


@dataclass
class FusedDepthResult:
    """Result of depth fusion"""
    fused_depth: np.ndarray  # Fused depth map (H, W)
    confidence: np.ndarray  # Fusion confidence (H, W)
    source_weights: Dict[str, np.ndarray]  # Per-pixel weights by camera
    valid_mask: np.ndarray  # Valid pixels mask
    fusion_strategy: str  # Strategy used
    num_sources: int  # Number of depth sources
    coverage_improvement: float  # Improvement vs. single sensor


class DepthFusionEngine:
    """
    Advanced depth fusion for multi-vendor cameras.

    Implements Scenario 12.2: Fuse depth from RealSense and ZED cameras

    Features:
    - Multiple fusion strategies
    - Confidence-weighted fusion
    - Range-based sensor selection
    - Outlier rejection
    - Coverage analysis
    """

    # Distance thresholds for sensor preference (meters)
    CLOSE_RANGE_THRESHOLD = 3.0  # RealSense better below this
    FAR_RANGE_THRESHOLD = 3.0  # ZED better above this

    def __init__(self, fusion_strategy: FusionStrategy = FusionStrategy.WEIGHTED_AVERAGE):
        """
        Initialize depth fusion engine.

        Args:
            fusion_strategy: Fusion strategy to use
        """
        self.fusion_strategy = fusion_strategy
        logger.info(f"Initialized DepthFusionEngine with strategy: {fusion_strategy.value}")

    def fuse_depth_maps(self,
                       depth_sources: List[DepthSource],
                       transformation_matrices: Optional[Dict[str, np.ndarray]] = None) -> FusedDepthResult:
        """
        Fuse multiple depth maps into single refined depth map.

        Args:
            depth_sources: List of depth sources
            transformation_matrices: Optional dict of camera_id -> 4x4 transformation
                                    to common reference frame

        Returns:
            FusedDepthResult
        """
        if not depth_sources:
            raise ValueError("No depth sources provided")

        logger.info(f"Fusing {len(depth_sources)} depth sources")

        # Get reference size from first source
        ref_shape = depth_sources[0].depth_map.shape

        # Transform all depth maps to common frame if needed
        if transformation_matrices:
            depth_sources = self._transform_depth_sources(depth_sources, transformation_matrices)

        # Ensure all depth maps have same size
        for source in depth_sources:
            if source.depth_map.shape != ref_shape:
                logger.warning(
                    f"Depth map size mismatch: {source.camera_id} "
                    f"{source.depth_map.shape} vs reference {ref_shape}"
                )
                # Resize to match
                source.depth_map = self._resize_depth_map(source.depth_map, ref_shape)
                if source.confidence_map is not None:
                    source.confidence_map = self._resize_depth_map(source.confidence_map, ref_shape)

        # Apply fusion strategy
        if self.fusion_strategy == FusionStrategy.WEIGHTED_AVERAGE:
            result = self._fuse_weighted_average(depth_sources)
        elif self.fusion_strategy == FusionStrategy.SELECTIVE:
            result = self._fuse_selective(depth_sources)
        elif self.fusion_strategy == FusionStrategy.BAYESIAN:
            result = self._fuse_bayesian(depth_sources)
        elif self.fusion_strategy == FusionStrategy.MEDIAN:
            result = self._fuse_median(depth_sources)
        else:
            raise ValueError(f"Unknown fusion strategy: {self.fusion_strategy}")

        # Compute coverage improvement
        result.coverage_improvement = self._compute_coverage_improvement(depth_sources, result)

        logger.info(
            f"Depth fusion complete: {np.count_nonzero(result.valid_mask)} valid pixels, "
            f"coverage improvement: {result.coverage_improvement:.1f}%"
        )

        return result

    def _fuse_weighted_average(self, sources: List[DepthSource]) -> FusedDepthResult:
        """
        Fuse using confidence-weighted average.

        Implements the primary fusion strategy for Scenario 12.2.
        """
        ref_shape = sources[0].depth_map.shape
        fused_depth = np.zeros(ref_shape, dtype=np.float32)
        fused_confidence = np.zeros(ref_shape, dtype=np.float32)
        total_weights = np.zeros(ref_shape, dtype=np.float32)

        source_weights = {}

        for source in sources:
            # Get base confidence
            if source.confidence_map is not None:
                confidence = source.confidence_map.copy()
            else:
                # No confidence provided - use depth-based heuristic
                confidence = np.ones_like(source.depth_map)

            # Compute range-based weights
            range_weights = self._compute_range_weights(source.depth_map, source.technology)

            # Combined weights
            weights = confidence * range_weights

            # Only use valid depths
            valid = (source.depth_map > 0) & np.isfinite(source.depth_map)
            weights[~valid] = 0

            # Accumulate
            fused_depth += source.depth_map * weights
            fused_confidence += confidence * weights
            total_weights += weights

            source_weights[source.camera_id] = weights

        # Normalize
        valid_mask = total_weights > 0
        fused_depth[valid_mask] /= total_weights[valid_mask]
        fused_confidence[valid_mask] /= total_weights[valid_mask]

        return FusedDepthResult(
            fused_depth=fused_depth,
            confidence=fused_confidence,
            source_weights=source_weights,
            valid_mask=valid_mask,
            fusion_strategy="weighted_average",
            num_sources=len(sources),
            coverage_improvement=0.0  # Computed later
        )

    def _fuse_selective(self, sources: List[DepthSource]) -> FusedDepthResult:
        """
        Fuse using selective strategy - choose best sensor per pixel.
        """
        ref_shape = sources[0].depth_map.shape
        fused_depth = np.zeros(ref_shape, dtype=np.float32)
        fused_confidence = np.zeros(ref_shape, dtype=np.float32)
        source_weights = {src.camera_id: np.zeros(ref_shape) for src in sources}

        # For each pixel, select source with highest confidence
        for i in range(ref_shape[0]):
            for j in range(ref_shape[1]):
                best_confidence = 0.0
                best_depth = 0.0
                best_source = None

                for source in sources:
                    if source.depth_map[i, j] > 0:
                        # Get confidence
                        if source.confidence_map is not None:
                            conf = source.confidence_map[i, j]
                        else:
                            conf = 1.0

                        # Apply range weighting
                        range_weight = self._compute_range_weight_scalar(
                            source.depth_map[i, j], source.technology
                        )
                        conf *= range_weight

                        if conf > best_confidence:
                            best_confidence = conf
                            best_depth = source.depth_map[i, j]
                            best_source = source.camera_id

                if best_source is not None:
                    fused_depth[i, j] = best_depth
                    fused_confidence[i, j] = best_confidence
                    source_weights[best_source][i, j] = 1.0

        valid_mask = fused_depth > 0

        return FusedDepthResult(
            fused_depth=fused_depth,
            confidence=fused_confidence,
            source_weights=source_weights,
            valid_mask=valid_mask,
            fusion_strategy="selective",
            num_sources=len(sources),
            coverage_improvement=0.0
        )

    def _fuse_bayesian(self, sources: List[DepthSource]) -> FusedDepthResult:
        """
        Fuse using Bayesian approach with uncertainty estimates.

        Assumes Gaussian uncertainty for each measurement.
        """
        ref_shape = sources[0].depth_map.shape

        # Initialize with first source
        fused_depth = sources[0].depth_map.copy()
        # Uncertainty (variance) - inverse of confidence
        fused_variance = np.ones_like(fused_depth)
        if sources[0].confidence_map is not None:
            fused_variance = 1.0 / (sources[0].confidence_map + 1e-6)

        source_weights = {}

        # Sequential Bayesian update
        for i, source in enumerate(sources):
            if i == 0:
                source_weights[source.camera_id] = np.ones_like(fused_depth)
                continue

            # Measurement variance
            if source.confidence_map is not None:
                meas_variance = 1.0 / (source.confidence_map + 1e-6)
            else:
                meas_variance = np.ones_like(source.depth_map)

            # Valid measurements
            valid = (source.depth_map > 0) & (meas_variance > 0)

            # Kalman gain
            kalman_gain = np.zeros_like(fused_depth)
            kalman_gain[valid] = fused_variance[valid] / (
                fused_variance[valid] + meas_variance[valid]
            )

            # Update estimate
            fused_depth[valid] = fused_depth[valid] + kalman_gain[valid] * (
                source.depth_map[valid] - fused_depth[valid]
            )

            # Update variance
            fused_variance[valid] = (1 - kalman_gain[valid]) * fused_variance[valid]

            source_weights[source.camera_id] = kalman_gain

        # Confidence from variance
        fused_confidence = 1.0 / (fused_variance + 1e-6)
        fused_confidence = np.clip(fused_confidence, 0, 1)

        valid_mask = fused_depth > 0

        return FusedDepthResult(
            fused_depth=fused_depth,
            confidence=fused_confidence,
            source_weights=source_weights,
            valid_mask=valid_mask,
            fusion_strategy="bayesian",
            num_sources=len(sources),
            coverage_improvement=0.0
        )

    def _fuse_median(self, sources: List[DepthSource]) -> FusedDepthResult:
        """
        Fuse using median (robust to outliers).
        """
        ref_shape = sources[0].depth_map.shape

        # Stack all depth maps
        depth_stack = np.stack([src.depth_map for src in sources], axis=0)

        # Valid masks
        valid_stack = (depth_stack > 0) & np.isfinite(depth_stack)

        # Compute median (ignoring zeros)
        fused_depth = np.zeros(ref_shape, dtype=np.float32)
        fused_confidence = np.zeros(ref_shape, dtype=np.float32)

        for i in range(ref_shape[0]):
            for j in range(ref_shape[1]):
                valid_depths = depth_stack[valid_stack[:, i, j], i, j]
                if len(valid_depths) > 0:
                    fused_depth[i, j] = np.median(valid_depths)
                    # Confidence = number of agreeing sensors / total
                    fused_confidence[i, j] = len(valid_depths) / len(sources)

        valid_mask = fused_depth > 0

        source_weights = {
            src.camera_id: (src.depth_map > 0).astype(float) / len(sources)
            for src in sources
        }

        return FusedDepthResult(
            fused_depth=fused_depth,
            confidence=fused_confidence,
            source_weights=source_weights,
            valid_mask=valid_mask,
            fusion_strategy="median",
            num_sources=len(sources),
            coverage_improvement=0.0
        )

    def _compute_range_weights(self, depth_map: np.ndarray, technology: str) -> np.ndarray:
        """
        Compute per-pixel range-based weights.

        RealSense preferred < 3m (close range)
        ZED preferred > 3m (far range)
        """
        weights = np.ones_like(depth_map)

        if technology == "active_ir":
            # RealSense: high weight close, low weight far
            weights = np.where(
                depth_map < self.CLOSE_RANGE_THRESHOLD,
                1.0,  # Full weight close
                np.maximum(0.3, 1.0 - (depth_map - self.CLOSE_RANGE_THRESHOLD) / 3.0)  # Falloff
            )
        elif technology == "passive_stereo":
            # ZED: lower weight close, high weight far
            weights = np.where(
                depth_map > self.FAR_RANGE_THRESHOLD,
                1.0,  # Full weight far
                np.maximum(0.5, depth_map / self.FAR_RANGE_THRESHOLD)  # Linear ramp-up
            )

        return weights

    def _compute_range_weight_scalar(self, depth: float, technology: str) -> float:
        """Compute range weight for single depth value."""
        if technology == "active_ir":
            if depth < self.CLOSE_RANGE_THRESHOLD:
                return 1.0
            else:
                return max(0.3, 1.0 - (depth - self.CLOSE_RANGE_THRESHOLD) / 3.0)
        elif technology == "passive_stereo":
            if depth > self.FAR_RANGE_THRESHOLD:
                return 1.0
            else:
                return max(0.5, depth / self.FAR_RANGE_THRESHOLD)
        return 1.0

    def _transform_depth_sources(self,
                                sources: List[DepthSource],
                                transformations: Dict[str, np.ndarray]) -> List[DepthSource]:
        """Transform depth sources to common reference frame."""
        # TODO: Implement depth map transformation with warping
        # For now, assume aligned
        logger.warning("Depth map transformation not yet implemented, assuming aligned")
        return sources

    def _resize_depth_map(self, depth_map: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
        """Resize depth map to target shape."""
        import cv2
        return cv2.resize(depth_map, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_NEAREST)

    def _compute_coverage_improvement(self,
                                     sources: List[DepthSource],
                                     result: FusedDepthResult) -> float:
        """
        Compute coverage improvement vs. best single sensor.

        Returns improvement percentage.
        """
        # Count valid pixels in fused result
        fused_coverage = np.count_nonzero(result.valid_mask)

        # Find best single sensor coverage
        best_single_coverage = 0
        for source in sources:
            source_coverage = np.count_nonzero(source.depth_map > 0)
            best_single_coverage = max(best_single_coverage, source_coverage)

        if best_single_coverage == 0:
            return 0.0

        # Improvement percentage
        improvement = ((fused_coverage - best_single_coverage) / best_single_coverage) * 100

        return improvement


# Convenience function
def fuse_realsense_and_zed_depth(
    realsense_depth: np.ndarray,
    zed_depth: np.ndarray,
    realsense_confidence: Optional[np.ndarray] = None,
    zed_confidence: Optional[np.ndarray] = None,
    strategy: FusionStrategy = FusionStrategy.WEIGHTED_AVERAGE
) -> FusedDepthResult:
    """
    Convenience function to fuse RealSense and ZED depth maps.

    Implements Scenario 12.2: Fuse depth from RealSense and ZED cameras

    Args:
        realsense_depth: RealSense depth map (meters)
        zed_depth: ZED depth map (meters)
        realsense_confidence: Optional RealSense confidence map
        zed_confidence: Optional ZED confidence map
        strategy: Fusion strategy

    Returns:
        FusedDepthResult
    """
    import time

    sources = [
        DepthSource(
            camera_id="realsense_d455i",
            depth_map=realsense_depth,
            confidence_map=realsense_confidence,
            technology="active_ir",
            timestamp=time.time()
        ),
        DepthSource(
            camera_id="zed_2i",
            depth_map=zed_depth,
            confidence_map=zed_confidence,
            technology="passive_stereo",
            timestamp=time.time()
        )
    ]

    engine = DepthFusionEngine(fusion_strategy=strategy)
    return engine.fuse_depth_maps(sources)
