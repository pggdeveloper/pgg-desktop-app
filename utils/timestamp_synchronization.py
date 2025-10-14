"""
Timestamp Synchronization for Multi-Vendor Camera Systems

This module provides software-based timestamp synchronization between:
- ZED 2i cameras (nanosecond timestamps)
- RealSense D455i cameras (millisecond timestamps)

Features:
- Timestamp conversion to common epoch
- Clock offset computation
- Clock drift detection and correction
- Temporal alignment quality metrics

Part of Scenario 13 (Multi-Vendor Multi-Camera Integration)

Implements:
- Scenario 5.3: Capture frame timestamps
- Scenario 5.4: Align timestamps across vendors
- Scenario 5.5: Handle clock drift
"""

import time
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from collections import deque
import logging

logger = logging.getLogger(__name__)


@dataclass
class TimestampRecord:
    """Single timestamp record from a camera"""
    camera_id: str
    frame_number: int
    native_timestamp: float  # Camera SDK timestamp (various units)
    system_timestamp: float  # System monotonic clock (seconds)
    synchronized_timestamp: Optional[float] = None  # Unix epoch after synchronization


@dataclass
class ClockOffsetEstimate:
    """Clock offset estimation between camera system and reference"""
    camera_id: str
    offset_seconds: float  # Offset to add to camera timestamp
    confidence: float  # Confidence in estimate (0-1)
    num_samples: int
    last_update: float  # System time of last update


@dataclass
class DriftStatistics:
    """Clock drift statistics over time"""
    camera_id: str
    drift_rate_us_per_sec: float  # Microseconds per second
    total_drift_ms: float  # Total accumulated drift in milliseconds
    measurement_duration_sec: float  # Duration of measurement
    resynchronization_count: int = 0


class TimestampSynchronizer:
    """
    Software-based timestamp synchronization for multi-vendor cameras.

    Handles:
    - Different timestamp formats (nanoseconds, milliseconds)
    - Clock offset estimation
    - Clock drift detection
    - Temporal alignment

    Target accuracy: <10ms for cattle monitoring (slow movement)

    Usage:
        synchronizer = TimestampSynchronizer(
            camera_ids=["realsense_d455i_0", "zed_2i_0", "zed_2i_1"],
            reference_timestamp=time.time()
        )

        # Add timestamps from each camera
        synchronizer.add_timestamp("zed_2i_0", frame_num, zed_timestamp_ns, system_time)
        synchronizer.add_timestamp("realsense_d455i_0", frame_num, rs_timestamp_ms, system_time)

        # Get synchronized timestamp
        sync_ts = synchronizer.get_synchronized_timestamp("zed_2i_0", frame_num)
    """

    def __init__(self,
                 camera_ids: List[str],
                 reference_timestamp: float,
                 drift_detection_interval: float = 300.0):  # 5 minutes
        """
        Initialize timestamp synchronizer.

        Args:
            camera_ids: List of camera IDs
            reference_timestamp: Reference Unix epoch timestamp (from orchestrator)
            drift_detection_interval: Interval for drift detection in seconds
        """
        self.camera_ids = camera_ids
        self.reference_timestamp = reference_timestamp
        self.drift_detection_interval = drift_detection_interval

        # Clock offset estimates per camera
        self.clock_offsets: Dict[str, ClockOffsetEstimate] = {}

        # Timestamp records per camera (deque for efficient FIFO)
        self.timestamp_records: Dict[str, deque] = {
            camera_id: deque(maxlen=1000)  # Keep last 1000 timestamps
            for camera_id in camera_ids
        }

        # Drift statistics per camera
        self.drift_stats: Dict[str, DriftStatistics] = {}

        # Initialize clock offsets
        self._initialize_clock_offsets()

        logger.info(f"Initialized TimestampSynchronizer with {len(camera_ids)} cameras")
        logger.info(f"Reference timestamp: {reference_timestamp}")

    def _initialize_clock_offsets(self):
        """Initialize clock offset estimates for all cameras"""
        for camera_id in self.camera_ids:
            self.clock_offsets[camera_id] = ClockOffsetEstimate(
                camera_id=camera_id,
                offset_seconds=0.0,
                confidence=0.0,
                num_samples=0,
                last_update=time.time()
            )

    def add_timestamp(self,
                     camera_id: str,
                     frame_number: int,
                     native_timestamp: float,
                     system_timestamp: Optional[float] = None,
                     timestamp_unit: str = "auto") -> TimestampRecord:
        """
        Add timestamp from camera.

        Args:
            camera_id: Camera identifier
            frame_number: Frame number
            native_timestamp: Native SDK timestamp
            system_timestamp: System monotonic clock (auto if None)
            timestamp_unit: Unit of native_timestamp ("ns", "ms", "s", or "auto")

        Returns:
            TimestampRecord with synchronized timestamp
        """
        if system_timestamp is None:
            system_timestamp = time.monotonic()

        # Convert native timestamp to seconds
        if timestamp_unit == "auto":
            # Auto-detect based on magnitude
            if native_timestamp > 1e15:  # Nanoseconds
                native_timestamp_sec = native_timestamp / 1e9
            elif native_timestamp > 1e9:  # Milliseconds
                native_timestamp_sec = native_timestamp / 1e3
            else:  # Seconds
                native_timestamp_sec = native_timestamp
        elif timestamp_unit == "ns":
            native_timestamp_sec = native_timestamp / 1e9
        elif timestamp_unit == "ms":
            native_timestamp_sec = native_timestamp / 1e3
        else:  # seconds
            native_timestamp_sec = native_timestamp

        # Create record
        record = TimestampRecord(
            camera_id=camera_id,
            frame_number=frame_number,
            native_timestamp=native_timestamp_sec,
            system_timestamp=system_timestamp
        )

        # Estimate and apply clock offset
        self._update_clock_offset(record)
        record.synchronized_timestamp = self._apply_synchronization(record)

        # Store record
        self.timestamp_records[camera_id].append(record)

        # Check for drift periodically
        if len(self.timestamp_records[camera_id]) % 100 == 0:
            self._check_clock_drift(camera_id)

        return record

    def _update_clock_offset(self, record: TimestampRecord):
        """
        Update clock offset estimate for camera.

        Computes offset as: offset = system_timestamp - native_timestamp
        """
        camera_id = record.camera_id
        offset_estimate = self.clock_offsets[camera_id]

        # Compute raw offset
        raw_offset = record.system_timestamp - record.native_timestamp

        # If first sample, initialize
        if offset_estimate.num_samples == 0:
            offset_estimate.offset_seconds = raw_offset
            offset_estimate.confidence = 0.5
            offset_estimate.num_samples = 1
        else:
            # Exponential moving average for offset
            alpha = 0.1  # Smoothing factor
            offset_estimate.offset_seconds = (
                alpha * raw_offset + (1 - alpha) * offset_estimate.offset_seconds
            )
            offset_estimate.num_samples += 1
            offset_estimate.confidence = min(1.0, offset_estimate.num_samples / 100)

        offset_estimate.last_update = time.time()

    def _apply_synchronization(self, record: TimestampRecord) -> float:
        """
        Apply synchronization to get Unix epoch timestamp.

        Args:
            record: Timestamp record

        Returns:
            Synchronized Unix epoch timestamp
        """
        camera_id = record.camera_id
        offset_estimate = self.clock_offsets[camera_id]

        # Convert to Unix epoch using reference and offset
        # synchronized_time = native_time + offset + reference
        synchronized_ts = (
            record.native_timestamp +
            offset_estimate.offset_seconds
        )

        return synchronized_ts

    def _check_clock_drift(self, camera_id: str):
        """
        Check for clock drift and trigger resynchronization if needed.

        Implements Scenario 5.5: Handle clock drift
        """
        records = list(self.timestamp_records[camera_id])

        if len(records) < 10:
            return

        # Compare timestamps from start and end of window
        start_records = records[:5]
        end_records = records[-5:]

        # Compute average offset at start and end
        start_offsets = [
            r.system_timestamp - r.native_timestamp for r in start_records
        ]
        end_offsets = [
            r.system_timestamp - r.native_timestamp for r in end_records
        ]

        avg_start_offset = np.mean(start_offsets)
        avg_end_offset = np.mean(end_offsets)

        # Compute drift
        drift = avg_end_offset - avg_start_offset
        duration = end_records[-1].system_timestamp - start_records[0].system_timestamp

        if duration > 0:
            drift_rate_us_per_sec = (drift * 1e6) / duration  # Microseconds per second
            total_drift_ms = drift * 1000  # Milliseconds

            # Update drift statistics
            if camera_id not in self.drift_stats:
                self.drift_stats[camera_id] = DriftStatistics(
                    camera_id=camera_id,
                    drift_rate_us_per_sec=drift_rate_us_per_sec,
                    total_drift_ms=total_drift_ms,
                    measurement_duration_sec=duration
                )
            else:
                stats = self.drift_stats[camera_id]
                stats.drift_rate_us_per_sec = drift_rate_us_per_sec
                stats.total_drift_ms = total_drift_ms
                stats.measurement_duration_sec = duration

            # Trigger resynchronization if drift > 5ms
            if abs(total_drift_ms) > 5.0:
                logger.warning(f"Camera {camera_id}: Clock drift detected: {total_drift_ms:.2f}ms")
                self._resynchronize(camera_id)

    def _resynchronize(self, camera_id: str):
        """
        Resynchronize clock offset for camera.

        Resets offset estimate to reduce accumulated drift.
        """
        logger.info(f"Resynchronizing camera {camera_id}")

        # Reset offset confidence (will be re-estimated)
        self.clock_offsets[camera_id].confidence = 0.5
        self.clock_offsets[camera_id].num_samples = 0

        # Update drift statistics
        if camera_id in self.drift_stats:
            self.drift_stats[camera_id].resynchronization_count += 1

    def get_synchronized_timestamp(self,
                                  camera_id: str,
                                  frame_number: int) -> Optional[float]:
        """
        Get synchronized timestamp for specific frame.

        Args:
            camera_id: Camera identifier
            frame_number: Frame number

        Returns:
            Synchronized Unix epoch timestamp or None if not found
        """
        for record in reversed(self.timestamp_records[camera_id]):
            if record.frame_number == frame_number:
                return record.synchronized_timestamp
        return None

    def get_temporal_alignment_error(self) -> Dict[str, float]:
        """
        Compute temporal alignment error across cameras.

        Returns:
            Dict mapping camera_id -> alignment_error_ms
        """
        errors = {}

        for camera_id in self.camera_ids:
            records = list(self.timestamp_records[camera_id])

            if len(records) < 10:
                errors[camera_id] = float('inf')
                continue

            # Compute alignment error as variance of clock offsets
            offsets = [
                r.system_timestamp - r.native_timestamp for r in records[-50:]
            ]

            if len(offsets) > 0:
                error_ms = np.std(offsets) * 1000  # Convert to milliseconds
                errors[camera_id] = error_ms
            else:
                errors[camera_id] = float('inf')

        return errors

    def get_synchronization_quality(self) -> Dict:
        """
        Get overall synchronization quality metrics.

        Returns:
            Dict with quality metrics
        """
        errors = self.get_temporal_alignment_error()

        mean_error = np.mean([e for e in errors.values() if e != float('inf')])
        max_error = np.max([e for e in errors.values() if e != float('inf')])

        # Quality assessment
        if mean_error < 5.0:
            quality = "excellent"
        elif mean_error < 10.0:
            quality = "good"
        elif mean_error < 20.0:
            quality = "acceptable"
        else:
            quality = "poor"

        return {
            "overall_quality": quality,
            "mean_alignment_error_ms": float(mean_error),
            "max_alignment_error_ms": float(max_error),
            "target_error_ms": 10.0,
            "errors_per_camera": errors,
            "clock_offsets": {
                camera_id: {
                    "offset_seconds": offset.offset_seconds,
                    "confidence": offset.confidence,
                    "num_samples": offset.num_samples
                }
                for camera_id, offset in self.clock_offsets.items()
            },
            "drift_statistics": {
                camera_id: {
                    "drift_rate_us_per_sec": stats.drift_rate_us_per_sec,
                    "total_drift_ms": stats.total_drift_ms,
                    "resynchronization_count": stats.resynchronization_count
                }
                for camera_id, stats in self.drift_stats.items()
            }
        }

    def get_latest_timestamps(self) -> Dict[str, Optional[float]]:
        """
        Get latest synchronized timestamp for each camera.

        Returns:
            Dict mapping camera_id -> latest_synchronized_timestamp
        """
        latest = {}

        for camera_id in self.camera_ids:
            records = self.timestamp_records[camera_id]
            if len(records) > 0:
                latest[camera_id] = records[-1].synchronized_timestamp
            else:
                latest[camera_id] = None

        return latest


# Convenience functions
def convert_timestamp_to_seconds(timestamp: float,
                                 unit: str = "auto") -> float:
    """
    Convert timestamp to seconds.

    Args:
        timestamp: Native timestamp
        unit: Unit ("ns", "ms", "s", or "auto")

    Returns:
        Timestamp in seconds
    """
    if unit == "auto":
        if timestamp > 1e15:  # Nanoseconds
            return timestamp / 1e9
        elif timestamp > 1e9:  # Milliseconds
            return timestamp / 1e3
        else:  # Seconds
            return timestamp
    elif unit == "ns":
        return timestamp / 1e9
    elif unit == "ms":
        return timestamp / 1e3
    else:
        return timestamp


def estimate_temporal_jitter(timestamps: List[float]) -> float:
    """
    Estimate temporal jitter in timestamps.

    Args:
        timestamps: List of timestamps in seconds

    Returns:
        Jitter in milliseconds (standard deviation of intervals)
    """
    if len(timestamps) < 2:
        return 0.0

    intervals = np.diff(timestamps)
    jitter_ms = np.std(intervals) * 1000
    return float(jitter_ms)
