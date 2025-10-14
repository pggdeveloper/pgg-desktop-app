"""
Frame Synchronization for Multi-Camera Systems

This module provides frame matching and synchronization for multi-vendor cameras
based on timestamps. Matches frames from different cameras that were captured
at approximately the same time.

Features:
- Temporal frame matching (nearest neighbor)
- Frame triplet creation (3 cameras)
- Unmatched frame detection
- Synchronization quality metrics

Part of Scenario 13 (Multi-Vendor Multi-Camera Integration)

Implements:
- Scenario 5.6: Match frames by nearest timestamp
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class FrameInfo:
    """Information about a single frame"""
    camera_id: str
    frame_number: int
    timestamp: float  # Synchronized Unix epoch timestamp
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FrameTriplet:
    """Synchronized frame triplet from 3 cameras"""
    timestamp: float  # Common reference timestamp
    frames: Dict[str, FrameInfo]  # camera_id -> FrameInfo
    temporal_errors: Dict[str, float]  # camera_id -> error in milliseconds
    max_temporal_error: float  # Maximum error across all cameras
    is_valid: bool  # True if all temporal errors below threshold


@dataclass
class MatchingStatistics:
    """Statistics for frame matching process"""
    total_frames_per_camera: Dict[str, int]
    matched_triplets: int
    unmatched_frames_per_camera: Dict[str, int]
    matching_rate: float  # Percentage of frames successfully matched
    mean_temporal_error_ms: float
    max_temporal_error_ms: float
    quality: str  # "excellent", "good", "acceptable", "poor"


class FrameSynchronizer:
    """
    Synchronize frames from multiple cameras using timestamps.

    Matches frames from different cameras based on nearest timestamp,
    creating synchronized frame triplets for multi-view processing.

    Target temporal error: <16ms (assuming 60fps max)

    Usage:
        synchronizer = FrameSynchronizer(
            camera_ids=["realsense_d455i_0", "zed_2i_0", "zed_2i_1"],
            reference_camera="realsense_d455i_0",
            max_temporal_error_ms=16.0
        )

        # Add frames from each camera
        synchronizer.add_frame("realsense_d455i_0", frame_num, timestamp)
        synchronizer.add_frame("zed_2i_0", frame_num, timestamp)
        synchronizer.add_frame("zed_2i_1", frame_num, timestamp)

        # Get synchronized triplets
        triplets = synchronizer.get_frame_triplets()
    """

    def __init__(self,
                 camera_ids: List[str],
                 reference_camera: str,
                 max_temporal_error_ms: float = 16.0):
        """
        Initialize frame synchronizer.

        Args:
            camera_ids: List of camera IDs (should be 3 cameras)
            reference_camera: Reference camera for matching
            max_temporal_error_ms: Maximum allowed temporal error in milliseconds
        """
        if len(camera_ids) != 3:
            logger.warning(f"Expected 3 cameras, got {len(camera_ids)}")

        self.camera_ids = camera_ids
        self.reference_camera = reference_camera
        self.max_temporal_error_ms = max_temporal_error_ms
        self.max_temporal_error_sec = max_temporal_error_ms / 1000.0

        # Storage for frames per camera
        self.frames: Dict[str, List[FrameInfo]] = {
            camera_id: [] for camera_id in camera_ids
        }

        # Matched triplets
        self.triplets: List[FrameTriplet] = []

        # Unmatched frames (flagged for review)
        self.unmatched_frames: Dict[str, List[FrameInfo]] = {
            camera_id: [] for camera_id in camera_ids
        }

        logger.info(f"Initialized FrameSynchronizer with {len(camera_ids)} cameras")
        logger.info(f"Reference camera: {reference_camera}")
        logger.info(f"Max temporal error: {max_temporal_error_ms}ms")

    def add_frame(self,
                  camera_id: str,
                  frame_number: int,
                  timestamp: float,
                  metadata: Optional[Dict] = None):
        """
        Add frame from camera.

        Args:
            camera_id: Camera identifier
            frame_number: Frame number
            timestamp: Synchronized Unix epoch timestamp
            metadata: Optional metadata dictionary
        """
        if camera_id not in self.camera_ids:
            raise ValueError(f"Unknown camera_id: {camera_id}")

        frame_info = FrameInfo(
            camera_id=camera_id,
            frame_number=frame_number,
            timestamp=timestamp,
            metadata=metadata or {}
        )

        self.frames[camera_id].append(frame_info)

    def _find_nearest_frame(self,
                           target_timestamp: float,
                           frames: List[FrameInfo],
                           search_window_sec: float = 0.05) -> Optional[Tuple[FrameInfo, float]]:
        """
        Find frame with nearest timestamp to target.

        Args:
            target_timestamp: Target timestamp
            frames: List of frames to search
            search_window_sec: Search window in seconds (±50ms default)

        Returns:
            (nearest_frame, time_difference) or None if not found
        """
        if not frames:
            return None

        # Binary search for approximate position
        timestamps = np.array([f.timestamp for f in frames])
        idx = np.searchsorted(timestamps, target_timestamp)

        # Search in window around approximate position
        start_idx = max(0, idx - 10)
        end_idx = min(len(frames), idx + 10)

        best_frame = None
        best_diff = float('inf')

        for i in range(start_idx, end_idx):
            diff = abs(frames[i].timestamp - target_timestamp)

            if diff < best_diff and diff <= search_window_sec:
                best_diff = diff
                best_frame = frames[i]

        if best_frame is not None:
            return best_frame, best_diff
        else:
            return None

    def match_frames(self) -> List[FrameTriplet]:
        """
        Match frames across cameras to create synchronized triplets.

        Implements Scenario 5.6: Match frames by nearest timestamp

        Uses reference camera as anchor and finds nearest frames from
        other cameras within temporal window.

        Returns:
            List of FrameTriplet objects
        """
        logger.info("Matching frames across cameras...")

        ref_frames = self.frames[self.reference_camera]

        if not ref_frames:
            logger.warning(f"No frames from reference camera {self.reference_camera}")
            return []

        # Sort all frame lists by timestamp for efficient searching
        for camera_id in self.camera_ids:
            self.frames[camera_id].sort(key=lambda f: f.timestamp)

        # Keep track of which frames have been matched
        matched_indices: Dict[str, set] = {
            camera_id: set() for camera_id in self.camera_ids
        }

        triplets = []

        # For each reference frame, find nearest frames from other cameras
        for ref_frame in ref_frames:
            ref_timestamp = ref_frame.timestamp

            # Find matching frames from other cameras
            matched_frames = {self.reference_camera: ref_frame}
            temporal_errors = {self.reference_camera: 0.0}  # Reference has zero error
            max_error = 0.0

            all_matched = True

            for camera_id in self.camera_ids:
                if camera_id == self.reference_camera:
                    continue

                # Find nearest frame
                result = self._find_nearest_frame(
                    ref_timestamp,
                    self.frames[camera_id],
                    search_window_sec=self.max_temporal_error_sec
                )

                if result is None:
                    all_matched = False
                    break

                matched_frame, time_diff = result
                frame_idx = self.frames[camera_id].index(matched_frame)

                # Check if frame already matched (avoid duplicate matches)
                if frame_idx in matched_indices[camera_id]:
                    all_matched = False
                    break

                matched_frames[camera_id] = matched_frame
                temporal_errors[camera_id] = time_diff * 1000  # Convert to ms
                max_error = max(max_error, time_diff * 1000)
                matched_indices[camera_id].add(frame_idx)

            # Create triplet if all cameras matched
            if all_matched:
                # Mark reference frame as matched
                ref_idx = ref_frames.index(ref_frame)
                matched_indices[self.reference_camera].add(ref_idx)

                is_valid = max_error <= self.max_temporal_error_ms

                triplet = FrameTriplet(
                    timestamp=ref_timestamp,
                    frames=matched_frames,
                    temporal_errors=temporal_errors,
                    max_temporal_error=max_error,
                    is_valid=is_valid
                )

                triplets.append(triplet)

                if not is_valid:
                    logger.debug(
                        f"Triplet created but max error {max_error:.2f}ms "
                        f"> threshold {self.max_temporal_error_ms}ms"
                    )

        # Identify unmatched frames
        for camera_id in self.camera_ids:
            for idx, frame in enumerate(self.frames[camera_id]):
                if idx not in matched_indices[camera_id]:
                    self.unmatched_frames[camera_id].append(frame)

        logger.info(f"Matched {len(triplets)} frame triplets")

        for camera_id in self.camera_ids:
            num_unmatched = len(self.unmatched_frames[camera_id])
            if num_unmatched > 0:
                logger.info(
                    f"Camera {camera_id}: {num_unmatched} unmatched frames "
                    f"({100*num_unmatched/len(self.frames[camera_id]):.1f}%)"
                )

        self.triplets = triplets
        return triplets

    def get_matching_statistics(self) -> MatchingStatistics:
        """
        Compute statistics for frame matching.

        Returns:
            MatchingStatistics object
        """
        if not self.triplets:
            logger.warning("No triplets matched yet, call match_frames() first")

        total_frames = {
            camera_id: len(frames)
            for camera_id, frames in self.frames.items()
        }

        unmatched_counts = {
            camera_id: len(frames)
            for camera_id, frames in self.unmatched_frames.items()
        }

        # Calculate matching rate (based on reference camera)
        ref_total = total_frames.get(self.reference_camera, 0)
        matched_count = len(self.triplets)

        if ref_total > 0:
            matching_rate = (matched_count / ref_total) * 100
        else:
            matching_rate = 0.0

        # Calculate temporal error statistics
        if self.triplets:
            temporal_errors = [t.max_temporal_error for t in self.triplets]
            mean_error = np.mean(temporal_errors)
            max_error = np.max(temporal_errors)
        else:
            mean_error = 0.0
            max_error = 0.0

        # Quality assessment
        if matching_rate >= 95 and mean_error < 5.0:
            quality = "excellent"
        elif matching_rate >= 85 and mean_error < 10.0:
            quality = "good"
        elif matching_rate >= 70 and mean_error < 15.0:
            quality = "acceptable"
        else:
            quality = "poor"

        return MatchingStatistics(
            total_frames_per_camera=total_frames,
            matched_triplets=matched_count,
            unmatched_frames_per_camera=unmatched_counts,
            matching_rate=matching_rate,
            mean_temporal_error_ms=float(mean_error),
            max_temporal_error_ms=float(max_error),
            quality=quality
        )

    def get_frame_triplets(self,
                          only_valid: bool = True) -> List[FrameTriplet]:
        """
        Get synchronized frame triplets.

        Args:
            only_valid: If True, return only triplets with error below threshold

        Returns:
            List of FrameTriplet objects
        """
        if not self.triplets:
            self.match_frames()

        if only_valid:
            return [t for t in self.triplets if t.is_valid]
        else:
            return self.triplets

    def get_unmatched_frames(self, camera_id: str) -> List[FrameInfo]:
        """
        Get unmatched frames for specific camera.

        Args:
            camera_id: Camera identifier

        Returns:
            List of unmatched FrameInfo objects
        """
        return self.unmatched_frames.get(camera_id, [])

    def export_synchronized_dataset(self, output_dir: str):
        """
        Export synchronized frame triplets to directory structure.

        Creates structure:
        output_dir/
            triplet_0000/
                realsense_d455i_0.png
                zed_2i_0.png
                zed_2i_1.png
                metadata.json
            triplet_0001/
                ...

        Args:
            output_dir: Output directory path
        """
        from pathlib import Path
        import json

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        triplets = self.get_frame_triplets(only_valid=True)

        logger.info(f"Exporting {len(triplets)} synchronized frame triplets to {output_dir}")

        for i, triplet in enumerate(triplets):
            triplet_dir = output_path / f"triplet_{i:04d}"
            triplet_dir.mkdir(exist_ok=True)

            # Export metadata
            metadata = {
                "triplet_id": i,
                "timestamp": triplet.timestamp,
                "max_temporal_error_ms": triplet.max_temporal_error,
                "temporal_errors": triplet.temporal_errors,
                "frames": {
                    camera_id: {
                        "frame_number": frame.frame_number,
                        "timestamp": frame.timestamp,
                        "metadata": frame.metadata
                    }
                    for camera_id, frame in triplet.frames.items()
                }
            }

            metadata_path = triplet_dir / "metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)

        logger.info(f"Export complete: {len(triplets)} triplets")

    def clear(self):
        """Clear all stored frames and triplets"""
        for camera_id in self.camera_ids:
            self.frames[camera_id].clear()
            self.unmatched_frames[camera_id].clear()

        self.triplets.clear()

        logger.info("Cleared all frames and triplets")


# Convenience functions
def create_frame_triplets(frame_data: Dict[str, List[Tuple[int, float]]],
                         reference_camera: str,
                         max_error_ms: float = 16.0) -> List[FrameTriplet]:
    """
    Convenience function to create frame triplets from raw data.

    Args:
        frame_data: Dict mapping camera_id -> [(frame_num, timestamp), ...]
        reference_camera: Reference camera ID
        max_error_ms: Maximum temporal error in milliseconds

    Returns:
        List of FrameTriplet objects
    """
    camera_ids = list(frame_data.keys())
    synchronizer = FrameSynchronizer(camera_ids, reference_camera, max_error_ms)

    for camera_id, frames in frame_data.items():
        for frame_num, timestamp in frames:
            synchronizer.add_frame(camera_id, frame_num, timestamp)

    return synchronizer.match_frames()


def compute_matching_quality(triplets: List[FrameTriplet],
                             total_frames: int) -> Dict:
    """
    Compute matching quality metrics.

    Args:
        triplets: List of frame triplets
        total_frames: Total number of frames from reference camera

    Returns:
        Dict with quality metrics
    """
    if not triplets:
        return {
            "matching_rate": 0.0,
            "mean_error_ms": 0.0,
            "max_error_ms": 0.0,
            "quality": "poor"
        }

    matching_rate = (len(triplets) / total_frames) * 100 if total_frames > 0 else 0.0

    errors = [t.max_temporal_error for t in triplets]
    mean_error = np.mean(errors)
    max_error = np.max(errors)

    if matching_rate >= 95 and mean_error < 5.0:
        quality = "excellent"
    elif matching_rate >= 85 and mean_error < 10.0:
        quality = "good"
    elif matching_rate >= 70:
        quality = "acceptable"
    else:
        quality = "poor"

    return {
        "matching_rate": matching_rate,
        "mean_error_ms": float(mean_error),
        "max_error_ms": float(max_error),
        "quality": quality
    }
