"""
Camera validation and health checks.

This module provides functions to validate that cameras can be opened
and read frames successfully, filtering out non-functional cameras.

The validation is performed before cameras are added to the final list,
ensuring that only working cameras are presented to the user.
"""

import cv2
import logging
from typing import Optional
from domain.camera_info import CameraInfo

logger = logging.getLogger(__name__)


def validate_camera(camera: CameraInfo, timeout_seconds: float = 5.0) -> bool:
    """
    Validate that a camera can be opened and read.

    This function attempts to open the camera using its open_hint
    and backend, then tries to read a single frame to verify
    functionality.

    Args:
        camera: CameraInfo object to validate
        timeout_seconds: Timeout for validation (currently not enforced)

    Returns:
        True if camera can be opened and read, False otherwise

    Notes:
        - Opens camera briefly (released immediately after test)
        - Timeout parameter is for future implementation
        - Logs warnings/errors for failed validations
        - Non-blocking: failures return False instead of raising

    Example:
        cameras = enumerate_usb_external_cameras()
        valid_cameras = [cam for cam in cameras if validate_camera(cam)]
    """
    try:
        # Determine backend value
        if hasattr(camera.backend, 'value'):
            # If backend is an enum with opencv constant
            backend_value = camera.backend.value
        else:
            # Fallback to DSHOW for Windows by default
            from utils.utils import WINDOWS_DEFAULT_BACKEND
            backend_value = WINDOWS_DEFAULT_BACKEND

        # Try to open camera
        cap = cv2.VideoCapture(camera.open_hint, backend_value)

        if not cap.isOpened():
            logger.warning(f"Camera {camera.name} failed to open")
            return False

        # Try to read a frame
        ret, frame = cap.read()
        cap.release()

        if not ret or frame is None:
            logger.warning(f"Camera {camera.name} failed to read frame")
            return False

        logger.info(f"Camera {camera.name} validated successfully")
        return True

    except Exception as e:
        logger.error(f"Error validating camera {camera.name}: {e}")
        return False


def validate_camera_list(cameras: list[CameraInfo]) -> list[CameraInfo]:
    """
    Validate a list of cameras and return only valid ones.

    Iterates through each camera, validates it, and only includes
    cameras that pass validation in the returned list.

    Args:
        cameras: List of CameraInfo objects to validate

    Returns:
        List of validated CameraInfo objects (working cameras only)

    Notes:
        - Filters out cameras that cannot be opened
        - Logs warnings for each removed camera
        - Returns empty list if all cameras fail validation
        - Preserves original camera order

    Example:
        all_cameras = enumerate_usb_external_cameras()
        working_cameras = validate_camera_list(all_cameras)
        print(f"Working cameras: {len(working_cameras)}/{len(all_cameras)}")
    """
    valid_cameras: list[CameraInfo] = []
    removed_count = 0

    for camera in cameras:
        if validate_camera(camera):
            valid_cameras.append(camera)
        else:
            logger.warning(f"Removing invalid camera: {camera.name}")
            removed_count += 1

    if removed_count > 0:
        logger.info(f"Validation complete: {len(valid_cameras)} valid, {removed_count} removed")

    return valid_cameras


def quick_validate_camera(camera: CameraInfo) -> bool:
    """
    Quick validation without reading a frame.

    This function only checks if the camera can be opened,
    without attempting to read a frame. Faster than full
    validation but less thorough.

    Args:
        camera: CameraInfo object to validate

    Returns:
        True if camera can be opened, False otherwise

    Notes:
        - Faster than validate_camera() (no frame read)
        - Less reliable (camera might open but not produce frames)
        - Useful for quick sanity checks
        - Releases camera immediately

    Example:
        if quick_validate_camera(camera):
            # Camera can be opened, proceed with full validation
            if validate_camera(camera):
                # Camera is fully functional
                pass
    """
    try:
        # Determine backend value
        if hasattr(camera.backend, 'value'):
            backend_value = camera.backend.value
        else:
            from utils.utils import WINDOWS_DEFAULT_BACKEND
            backend_value = WINDOWS_DEFAULT_BACKEND

        # Try to open camera
        cap = cv2.VideoCapture(camera.open_hint, backend_value)
        is_opened = cap.isOpened()
        cap.release()

        return is_opened

    except Exception as e:
        logger.debug(f"Quick validation failed for {camera.name}: {e}")
        return False


def get_camera_resolution(camera: CameraInfo) -> Optional[tuple[int, int]]:
    """
    Get actual resolution of camera.

    Opens camera briefly to query its actual resolution.

    Args:
        camera: CameraInfo object

    Returns:
        Tuple of (width, height) if successful, None otherwise

    Notes:
        - Opens camera briefly (released immediately)
        - Returns actual resolution, not theoretical max
        - Returns None if camera cannot be opened
        - Useful for verifying expected resolutions

    Example:
        resolution = get_camera_resolution(camera)
        if resolution:
            width, height = resolution
            print(f"Camera resolution: {width}x{height}")
    """
    try:
        # Determine backend value
        if hasattr(camera.backend, 'value'):
            backend_value = camera.backend.value
        else:
            from utils.utils import WINDOWS_DEFAULT_BACKEND
            backend_value = WINDOWS_DEFAULT_BACKEND

        cap = cv2.VideoCapture(camera.open_hint, backend_value)

        if not cap.isOpened():
            return None

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        cap.release()

        return (width, height)

    except Exception as e:
        logger.debug(f"Failed to get resolution for {camera.name}: {e}")
        return None
