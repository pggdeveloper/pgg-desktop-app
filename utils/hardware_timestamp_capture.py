"""
Hardware Timestamp Capture

This module provides hardware-level timestamp capture from camera SDKs
for precise multi-camera synchronization.

Part of Scenario 13 (Multi-Vendor Multi-Camera Integration)

Implements:
- Scenario 5.3: Capture frame timestamps from each camera independently

Features:
- ZED SDK hardware timestamp capture (nanoseconds)
- RealSense hardware timestamp capture (milliseconds)
- System monotonic clock fallback
- Frame metadata storage
"""

import time
from typing import Dict, Optional, Union
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class CameraVendor(Enum):
    """Camera vendor types"""
    ZED = "zed"
    REALSENSE = "realsense"
    UNKNOWN = "unknown"


@dataclass
class FrameTimestampMetadata:
    """
    Complete timestamp metadata for a single frame.

    Stores multiple timestamp sources for maximum precision
    and fallback capability.
    """
    # Hardware timestamps (vendor-specific)
    hardware_timestamp: Optional[float] = None  # SDK hardware timestamp
    hardware_timestamp_unit: str = "unknown"  # ns, ms, us, s

    # System timestamps
    system_monotonic: float = 0.0  # time.monotonic()
    system_epoch: float = 0.0  # time.time()

    # Synchronized timestamp (computed)
    synchronized_timestamp: Optional[float] = None  # Unix epoch

    # Metadata
    camera_id: str = ""
    camera_vendor: CameraVendor = CameraVendor.UNKNOWN
    frame_number: int = 0

    # Quality indicators
    timestamp_source: str = "unknown"  # "hardware", "system", "fallback"
    timestamp_confidence: float = 0.0  # 0.0-1.0


class HardwareTimestampCapture:
    """
    Capture hardware-level timestamps from camera SDKs.

    Implements Scenario 5.3: Capture frame timestamps from each camera independently

    This class provides vendor-specific timestamp capture methods that integrate
    with camera SDKs (pyzed, pyrealsense2) to extract hardware timestamps.
    """

    def __init__(self):
        """Initialize hardware timestamp capture."""
        self.zed_available = self._check_zed_availability()
        self.realsense_available = self._check_realsense_availability()

        logger.info(
            f"HardwareTimestampCapture initialized: "
            f"ZED={'available' if self.zed_available else 'unavailable'}, "
            f"RealSense={'available' if self.realsense_available else 'unavailable'}"
        )

    def _check_zed_availability(self) -> bool:
        """Check if ZED SDK is available."""
        try:
            import pyzed.sl as sl
            return True
        except ImportError:
            logger.warning("pyzed SDK not available - ZED hardware timestamps unavailable")
            return False

    def _check_realsense_availability(self) -> bool:
        """Check if RealSense SDK is available."""
        try:
            import pyrealsense2 as rs
            return True
        except ImportError:
            logger.warning("pyrealsense2 SDK not available - RealSense hardware timestamps unavailable")
            return False

    def capture_zed_timestamp(self,
                             zed_camera,
                             frame_number: int,
                             camera_id: str = "zed_2i_0") -> FrameTimestampMetadata:
        """
        Capture hardware timestamp from ZED camera.

        ZED SDK provides timestamp in nanoseconds since camera initialization.

        Args:
            zed_camera: ZED camera object (pyzed.sl.Camera)
            frame_number: Frame sequence number
            camera_id: Camera identifier

        Returns:
            FrameTimestampMetadata with ZED hardware timestamp

        Example:
            import pyzed.sl as sl
            zed = sl.Camera()
            # ... initialize and grab frame ...
            timestamp_meta = capture.capture_zed_timestamp(zed, frame_num, "zed_2i_0")
        """
        metadata = FrameTimestampMetadata(
            camera_id=camera_id,
            camera_vendor=CameraVendor.ZED,
            frame_number=frame_number,
            system_monotonic=time.monotonic(),
            system_epoch=time.time()
        )

        if not self.zed_available:
            metadata.timestamp_source = "system"
            metadata.timestamp_confidence = 0.5
            logger.warning(f"ZED SDK unavailable - using system timestamp for {camera_id}")
            return metadata

        try:
            import pyzed.sl as sl

            # Get ZED hardware timestamp
            # ZED provides timestamp in nanoseconds
            timestamp_ns = zed_camera.get_timestamp(sl.TIME_REFERENCE.IMAGE)

            # Convert to float (seconds)
            hardware_timestamp_s = timestamp_ns.get_nanoseconds() / 1e9

            metadata.hardware_timestamp = hardware_timestamp_s
            metadata.hardware_timestamp_unit = "ns"
            metadata.timestamp_source = "hardware"
            metadata.timestamp_confidence = 1.0

            logger.debug(
                f"Captured ZED hardware timestamp: {hardware_timestamp_s:.9f}s "
                f"({timestamp_ns.get_nanoseconds()}ns) for {camera_id} frame {frame_number}"
            )

        except Exception as e:
            logger.error(f"Failed to capture ZED hardware timestamp: {e}")
            metadata.timestamp_source = "fallback"
            metadata.timestamp_confidence = 0.3

        return metadata

    def capture_realsense_timestamp(self,
                                   realsense_frame,
                                   frame_number: int,
                                   camera_id: str = "realsense_d455i_0") -> FrameTimestampMetadata:
        """
        Capture hardware timestamp from RealSense camera.

        RealSense SDK provides timestamp in milliseconds.

        Args:
            realsense_frame: RealSense frame object (pyrealsense2.frame)
            frame_number: Frame sequence number
            camera_id: Camera identifier

        Returns:
            FrameTimestampMetadata with RealSense hardware timestamp

        Example:
            import pyrealsense2 as rs
            pipeline = rs.pipeline()
            # ... start pipeline and wait for frames ...
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            timestamp_meta = capture.capture_realsense_timestamp(color_frame, frame_num)
        """
        metadata = FrameTimestampMetadata(
            camera_id=camera_id,
            camera_vendor=CameraVendor.REALSENSE,
            frame_number=frame_number,
            system_monotonic=time.monotonic(),
            system_epoch=time.time()
        )

        if not self.realsense_available:
            metadata.timestamp_source = "system"
            metadata.timestamp_confidence = 0.5
            logger.warning(f"RealSense SDK unavailable - using system timestamp for {camera_id}")
            return metadata

        try:
            # Get RealSense hardware timestamp
            # RealSense provides timestamp in milliseconds
            timestamp_ms = realsense_frame.get_timestamp()

            # Convert to seconds
            hardware_timestamp_s = timestamp_ms / 1000.0

            metadata.hardware_timestamp = hardware_timestamp_s
            metadata.hardware_timestamp_unit = "ms"
            metadata.timestamp_source = "hardware"
            metadata.timestamp_confidence = 1.0

            logger.debug(
                f"Captured RealSense hardware timestamp: {hardware_timestamp_s:.3f}s "
                f"({timestamp_ms}ms) for {camera_id} frame {frame_number}"
            )

        except Exception as e:
            logger.error(f"Failed to capture RealSense hardware timestamp: {e}")
            metadata.timestamp_source = "fallback"
            metadata.timestamp_confidence = 0.3

        return metadata

    def capture_frame_timestamps(self,
                                camera_frames: Dict[str, tuple],
                                frame_number: int) -> Dict[str, FrameTimestampMetadata]:
        """
        Capture timestamps from all camera frames.

        Args:
            camera_frames: Dict of camera_id -> (camera_object, frame_object)
                          For ZED: (zed_camera, None) - timestamp from camera object
                          For RealSense: (None, rs_frame) - timestamp from frame object
            frame_number: Global frame sequence number

        Returns:
            Dict of camera_id -> FrameTimestampMetadata

        Example:
            camera_frames = {
                "zed_2i_0": (zed_camera_0, None),
                "zed_2i_1": (zed_camera_1, None),
                "realsense_d455i_0": (None, rs_color_frame)
            }
            timestamps = capture.capture_frame_timestamps(camera_frames, frame_num)
        """
        timestamps = {}

        for camera_id, (camera_obj, frame_obj) in camera_frames.items():
            # Determine vendor from camera_id
            if "zed" in camera_id.lower():
                if camera_obj is not None:
                    timestamps[camera_id] = self.capture_zed_timestamp(
                        camera_obj, frame_number, camera_id
                    )
                else:
                    logger.warning(f"ZED camera object missing for {camera_id}")
                    timestamps[camera_id] = self._create_fallback_timestamp(
                        camera_id, CameraVendor.ZED, frame_number
                    )

            elif "realsense" in camera_id.lower():
                if frame_obj is not None:
                    timestamps[camera_id] = self.capture_realsense_timestamp(
                        frame_obj, frame_number, camera_id
                    )
                else:
                    logger.warning(f"RealSense frame object missing for {camera_id}")
                    timestamps[camera_id] = self._create_fallback_timestamp(
                        camera_id, CameraVendor.REALSENSE, frame_number
                    )

            else:
                logger.warning(f"Unknown camera vendor for {camera_id}")
                timestamps[camera_id] = self._create_fallback_timestamp(
                    camera_id, CameraVendor.UNKNOWN, frame_number
                )

        return timestamps

    def _create_fallback_timestamp(self,
                                  camera_id: str,
                                  vendor: CameraVendor,
                                  frame_number: int) -> FrameTimestampMetadata:
        """
        Create fallback timestamp using system clock.

        Args:
            camera_id: Camera identifier
            vendor: Camera vendor
            frame_number: Frame sequence number

        Returns:
            FrameTimestampMetadata with system timestamps
        """
        return FrameTimestampMetadata(
            camera_id=camera_id,
            camera_vendor=vendor,
            frame_number=frame_number,
            system_monotonic=time.monotonic(),
            system_epoch=time.time(),
            timestamp_source="fallback",
            timestamp_confidence=0.3
        )

    def store_timestamps_in_metadata(self,
                                    timestamps: Dict[str, FrameTimestampMetadata],
                                    metadata_storage: Dict) -> None:
        """
        Store captured timestamps in frame metadata storage.

        Args:
            timestamps: Dict of camera_id -> FrameTimestampMetadata
            metadata_storage: Dict to store metadata (will be modified in place)

        Example:
            metadata = {}
            capture.store_timestamps_in_metadata(timestamps, metadata)
            # metadata now contains all timestamp information
        """
        if "frame_timestamps" not in metadata_storage:
            metadata_storage["frame_timestamps"] = {}

        for camera_id, timestamp_meta in timestamps.items():
            metadata_storage["frame_timestamps"][camera_id] = {
                "hardware_timestamp": timestamp_meta.hardware_timestamp,
                "hardware_timestamp_unit": timestamp_meta.hardware_timestamp_unit,
                "system_monotonic": timestamp_meta.system_monotonic,
                "system_epoch": timestamp_meta.system_epoch,
                "synchronized_timestamp": timestamp_meta.synchronized_timestamp,
                "frame_number": timestamp_meta.frame_number,
                "timestamp_source": timestamp_meta.timestamp_source,
                "timestamp_confidence": timestamp_meta.timestamp_confidence,
                "camera_vendor": timestamp_meta.camera_vendor.value
            }

        logger.info(
            f"Stored timestamps for {len(timestamps)} cameras in metadata"
        )


# Integration helper for camera recorders
class CameraRecorderTimestampMixin:
    """
    Mixin class for camera recorders to add hardware timestamp capture.

    Usage:
        class ZedCameraRecorder(CameraRecorderTimestampMixin):
            def __init__(self, ...):
                self.timestamp_capture = HardwareTimestampCapture()
                ...

            def capture_frame(self):
                # ... capture frame from ZED ...
                timestamp_meta = self.timestamp_capture.capture_zed_timestamp(
                    self.zed_camera, self.frame_number, self.camera_id
                )
                # Store in frame metadata
                self.frame_metadata["timestamp"] = timestamp_meta
    """
    pass


# Convenience function
def create_timestamp_capture_for_multi_camera_system(
    camera_ids: list
) -> tuple[HardwareTimestampCapture, Dict[str, CameraVendor]]:
    """
    Create timestamp capture system for multi-camera setup.

    Args:
        camera_ids: List of camera IDs (e.g., ["zed_2i_0", "zed_2i_1", "realsense_d455i_0"])

    Returns:
        (HardwareTimestampCapture instance, Dict of camera_id -> CameraVendor)

    Example:
        capture, vendors = create_timestamp_capture_for_multi_camera_system(
            ["zed_2i_0", "zed_2i_1", "realsense_d455i_0"]
        )
    """
    capture = HardwareTimestampCapture()

    vendors = {}
    for camera_id in camera_ids:
        if "zed" in camera_id.lower():
            vendors[camera_id] = CameraVendor.ZED
        elif "realsense" in camera_id.lower():
            vendors[camera_id] = CameraVendor.REALSENSE
        else:
            vendors[camera_id] = CameraVendor.UNKNOWN
            logger.warning(f"Unknown vendor for camera {camera_id}")

    logger.info(
        f"Created timestamp capture for {len(camera_ids)} cameras: "
        f"{len([v for v in vendors.values() if v == CameraVendor.ZED])} ZED, "
        f"{len([v for v in vendors.values() if v == CameraVendor.REALSENSE])} RealSense"
    )

    return capture, vendors
