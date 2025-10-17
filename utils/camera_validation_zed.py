"""
Stereolabs ZED camera validation using UVC (USB Video Class).

This module provides UVC-based validation for Stereolabs ZED cameras,
verifying stereo side-by-side format and extracting camera metadata
through standard USB Video Class protocol.

UPDATED (2025-10-17): With SDK-based exclusion strategy, ZED cameras are
now definitively identified in utils/camera_identification_sdk.py before
reaching this validation stage. This module now serves as CONFIRMATION
rather than primary identification.

Since the ZED SDK (pyzed) requires GPU/CUDA which is NOT available in this
system, this module uses OpenCV's UVC backend to validate ZED cameras and
verify they are outputting the expected stereo format.

Supported models:
- Stereolabs ZED 2i (primary)
- Stereolabs ZED 2 (secondary)

System requirements:
- OpenCV with DirectShow (Windows) / V4L2 (Linux) / AVFoundation (macOS)
- NO GPU required (validation only, no depth processing)
"""

import logging
import cv2
from typing import Optional, Tuple
from domain.camera_info import CameraInfo
from domain.camera_type import CameraType
from domain.camera_capabilities import CameraCapabilities
from config import DEBUG_MODE

logger = logging.getLogger(__name__)

# ZED stereo side-by-side resolutions (width x height)
# Each resolution has LEFT and RIGHT images side-by-side
ZED_STEREO_RESOLUTIONS = [
    (3840, 1080),  # 2.2K (1920x1080 per eye) - ZED 2i default
    (2560, 720),   # HD1080 (1280x720 per eye)
    (1344, 376),   # VGA (672x376 per eye)
]

# Expected aspect ratio for stereo side-by-side
# Stereo width is ~2x height (actually ~3.56 for 3840x1080)
STEREO_ASPECT_RATIO_MIN = 3.0
STEREO_ASPECT_RATIO_MAX = 4.0


def validate_zed_camera_uvc(camera: CameraInfo) -> Tuple[bool, Optional[dict]]:
    """
    Validate a ZED camera via UVC and verify stereo side-by-side format.

    UPDATED (2025-10-17): Camera already identified as ZED via SDK exclusion
    (utils/camera_identification_sdk.py). This function now serves as a
    SANITY CHECK to confirm the camera outputs proper stereo format.

    This function opens the camera using OpenCV, checks the output resolution,
    and verifies it matches the expected ZED stereo side-by-side format.

    Args:
        camera: CameraInfo object already identified as ZED_2i or ZED_2

    Returns:
        Tuple of (is_valid, metadata)
        - is_valid: True if camera outputs correct stereo format
        - metadata: Dict with resolution, fps, format info (None if invalid)

    Notes:
        - Camera was already confirmed as ZED (not RealSense) by SDK exclusion
        - This is a format verification, not primary identification
        - Opens camera briefly to check format (released immediately)
        - Does NOT require GPU (validation only, no depth processing)
        - Verifies aspect ratio matches stereo side-by-side (~3.56)
    """
    metadata: dict = {}

    try:
        # Open camera using the open_hint from enumeration
        from utils.utils import open_camera_hint
        cap = open_camera_hint(camera.open_hint)

        if not cap.isOpened():
            if DEBUG_MODE:
                print(f"[DEBUG] ZED validation failed: Cannot open camera at {camera.open_hint}")
            return False, None

        # Force ZED stereo resolution (DirectShow may default to 640x480)
        # Try highest resolution first (3840x1080)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)

        # Get camera properties after setting resolution
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        if DEBUG_MODE:
            print(f"[DEBUG] ZED camera {camera.index}: Resolution {width}x{height} @ {fps:.1f}fps")

        # Verify resolution matches ZED stereo side-by-side format
        is_stereo_resolution = (width, height) in ZED_STEREO_RESOLUTIONS

        # Calculate aspect ratio
        aspect_ratio = width / height if height > 0 else 0

        # Verify aspect ratio is in stereo range
        is_stereo_aspect = STEREO_ASPECT_RATIO_MIN <= aspect_ratio <= STEREO_ASPECT_RATIO_MAX

        # Check if camera outputs valid frames
        ret, frame = cap.read()
        has_valid_frame = ret and frame is not None

        # Release camera immediately
        cap.release()

        # Build metadata
        metadata = {
            "resolution": (width, height),
            "width": width,
            "height": height,
            "fps": fps,
            "aspect_ratio": aspect_ratio,
            "is_stereo_resolution": is_stereo_resolution,
            "is_stereo_aspect": is_stereo_aspect,
            "has_valid_frame": has_valid_frame,
        }

        # Validation criteria
        is_valid = is_stereo_resolution and is_stereo_aspect and has_valid_frame

        if DEBUG_MODE:
            print(f"[DEBUG] ZED validation result: {is_valid}")
            print(f"[DEBUG]   Stereo resolution: {is_stereo_resolution}")
            print(f"[DEBUG]   Stereo aspect ratio: {is_stereo_aspect} ({aspect_ratio:.2f})")
            print(f"[DEBUG]   Valid frame: {has_valid_frame}")

        return is_valid, metadata

    except Exception as e:
        logger.warning(f"Error validating ZED camera {camera.index}: {e}")
        if DEBUG_MODE:
            print(f"[DEBUG] ZED validation error: {e}")
        return False, None


def enhance_zed_detection_with_uvc(cameras: list[CameraInfo]) -> list[CameraInfo]:
    """
    Enhance ZED camera detection with UVC validation.

    UPDATED (2025-10-17): With SDK exclusion strategy, cameras reaching this
    function have already been definitively identified as ZED (not RealSense).
    This function now performs SANITY CHECK validation rather than primary
    identification.

    This function validates ZED cameras identified by SDK exclusion and updates
    their capabilities based on UVC validation results.

    For cameras identified as ZED (via SDK exclusion), this function:
    1. Validates stereo side-by-side format via UVC (sanity check)
    2. Updates capabilities with stereo=True if validated
    3. Marks cameras as GENERIC if validation fails (hardware issue)

    Args:
        cameras: List of CameraInfo from camera_identification_sdk

    Returns:
        Updated list with enhanced ZED detection

    Notes:
        - Only processes cameras with camera_type = ZED_2i or ZED_2
        - Cameras already confirmed as ZED by SDK exclusion
        - Validation failure indicates hardware/format issue, not misidentification
        - Does NOT require GPU (validation only)
    """
    enhanced: list[CameraInfo] = []

    for cam in cameras:
        # Only validate cameras identified as ZED
        if cam.camera_type not in (CameraType.ZED_2i, CameraType.ZED_2):
            enhanced.append(cam)
            continue

        if DEBUG_MODE:
            print(f"[DEBUG] Validating ZED camera at index {cam.index}")

        # Validate via UVC
        is_valid, metadata = validate_zed_camera_uvc(cam)

        if is_valid:
            # Update capabilities with validated stereo info
            updated_capabilities = CameraCapabilities(
                depth=True,  # ZED has depth via stereo
                imu=cam.camera_type == CameraType.ZED_2i,  # Only ZED 2i has IMU
                stereo=True,  # Verified via UVC
            )

            # Create updated CameraInfo with validated capabilities
            updated_cam = CameraInfo(
                index=cam.index,
                name=cam.name,
                backend=cam.backend,
                open_hint=cam.open_hint,
                camera_type=cam.camera_type,  # Keep detected type
                capabilities=updated_capabilities,
                usb_vid=cam.usb_vid,
                usb_pid=cam.usb_pid,
                os_id=cam.os_id,
                path=cam.path,
                serial_number=cam.serial_number,
                sdk_available=False,  # No SDK available (requires GPU)
                vendor=cam.vendor,
                model=cam.model,
            )

            enhanced.append(updated_cam)

            if DEBUG_MODE:
                print(f"[DEBUG] ZED validation passed: {cam.camera_type}")
                if metadata:
                    print(f"[DEBUG]   Resolution: {metadata.get('width')}x{metadata.get('height')}")

        else:
            # Validation failed - downgrade to GENERIC
            # Camera identified as ZED by SDK exclusion but format check failed
            # This indicates hardware issue, not misidentification
            if DEBUG_MODE:
                print(f"[DEBUG] ZED format validation failed - possible hardware issue")
                print(f"[DEBUG] Downgrading to GENERIC (camera not outputting stereo format)")

            updated_cam = CameraInfo(
                index=cam.index,
                name=cam.name,
                backend=cam.backend,
                open_hint=cam.open_hint,
                camera_type=CameraType.GENERIC,  # Downgrade
                capabilities=CameraCapabilities(),  # No special capabilities
                usb_vid=cam.usb_vid,
                usb_pid=cam.usb_pid,
                os_id=cam.os_id,
                path=cam.path,
                serial_number=cam.serial_number,
                sdk_available=False,
                vendor=cam.vendor,
                model=cam.model,
            )

            enhanced.append(updated_cam)

    return enhanced


def get_zed_supported_resolutions(camera: CameraInfo) -> list[Tuple[int, int]]:
    """
    Get list of supported stereo resolutions for a ZED camera.

    This function attempts to query which stereo resolutions are supported
    by the ZED camera by trying to set each known resolution.

    Args:
        camera: CameraInfo object for ZED camera

    Returns:
        List of supported (width, height) tuples

    Notes:
        - Opens camera multiple times (slow operation)
        - Use sparingly, prefer caching results
        - Returns empty list on error
    """
    supported: list[Tuple[int, int]] = []

    try:
        from utils.utils import open_camera_hint

        for width, height in ZED_STEREO_RESOLUTIONS:
            cap = open_camera_hint(camera.open_hint)

            if not cap.isOpened():
                continue

            # Try to set resolution
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

            # Read actual resolution
            actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            cap.release()

            # Check if resolution was applied
            if actual_width == width and actual_height == height:
                supported.append((width, height))

                if DEBUG_MODE:
                    print(f"[DEBUG] ZED camera {camera.index} supports {width}x{height}")

    except Exception as e:
        logger.warning(f"Error querying ZED resolutions: {e}")
        if DEBUG_MODE:
            print(f"[DEBUG] Error querying ZED resolutions: {e}")

    return supported
