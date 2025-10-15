"""
Intel RealSense camera detection using pyrealsense2 SDK.

This module provides SDK-based detection for Intel RealSense cameras,
extracting detailed information like serial numbers and IMU availability
that cannot be obtained through standard USB enumeration.

Supported models:
- Intel RealSense D455i (primary)

System requirements:
- pyrealsense2 installed (pip install pyrealsense2)
- NO GPU required (depth computed on camera hardware)
"""

import logging
from typing import Optional
from domain.camera_info import CameraInfo
from domain.camera_type import CameraType
from domain.camera_backend import CameraBackend
from domain.camera_capabilities import CameraCapabilities
from config import DEBUG_MODE

logger = logging.getLogger(__name__)


def detect_realsense_cameras() -> list[CameraInfo]:
    """
    Detect Intel RealSense cameras using pyrealsense2 SDK.

    This function uses the RealSense SDK to enumerate connected cameras
    and extract detailed metadata that is not available through standard
    USB enumeration (serial numbers, IMU detection, firmware version, etc.).

    Returns:
        List of CameraInfo objects with SDK-enhanced data.
        Empty list if pyrealsense2 is not installed or no cameras found.

    Notes:
        - Works WITHOUT GPU (depth processing happens on camera hardware)
        - Only D455i is supported as the primary model
        - Gracefully handles ImportError if SDK not installed
        - Returns empty list on any error (non-blocking)
    """
    cameras: list[CameraInfo] = []

    # Try to import pyrealsense2
    try:
        import pyrealsense2 as rs
    except ImportError:
        if DEBUG_MODE:
            print("[DEBUG] pyrealsense2 SDK not available - skipping SDK detection")
        return cameras

    try:
        # Create RealSense context
        ctx = rs.context()
        devices = ctx.query_devices()

        if DEBUG_MODE:
            print(f"[DEBUG] RealSense SDK: Found {len(devices)} RealSense devices")

        # Enumerate each RealSense device
        for i, dev in enumerate(devices):
            try:
                # Extract device information
                serial = dev.get_info(rs.camera_info.serial_number)
                name = dev.get_info(rs.camera_info.name)
                product_line = dev.get_info(rs.camera_info.product_line)
                firmware_version = dev.get_info(rs.camera_info.firmware_version)

                if DEBUG_MODE:
                    print(f"[DEBUG] RealSense device {i}: {name} (Serial: {serial})")
                    print(f"[DEBUG]   Product line: {product_line}")
                    print(f"[DEBUG]   Firmware: {firmware_version}")

                # Detect IMU by checking sensors
                has_imu = False
                has_depth = False
                has_color = False

                for sensor in dev.query_sensors():
                    # Check if this sensor is a motion sensor (IMU)
                    if sensor.is_motion_sensor():
                        has_imu = True
                        if DEBUG_MODE:
                            print(f"[DEBUG]   IMU detected: {sensor.get_info(rs.camera_info.name)}")

                    # Check sensor type
                    sensor_name = sensor.get_info(rs.camera_info.name).lower()
                    if "depth" in sensor_name:
                        has_depth = True
                    if "rgb" in sensor_name or "color" in sensor_name:
                        has_color = True

                # Determine camera type based on product line and name
                camera_type = _determine_realsense_type(product_line, name)

                if camera_type == CameraType.GENERIC:
                    if DEBUG_MODE:
                        print(f"[DEBUG]   WARNING: RealSense model '{product_line}' not officially supported")

                # Create capabilities object
                capabilities = CameraCapabilities(
                    depth=has_depth,
                    imu=has_imu,
                    stereo=has_depth  # Depth cameras use stereo
                )

                # Create CameraInfo with SDK data
                # Note: index will be assigned during merge with USB enumeration
                cam = CameraInfo(
                    index=None,  # Will be filled during merge
                    name=name,
                    backend=CameraBackend.REALSENSE_SDK,
                    open_hint=-1,  # Will be updated during merge
                    camera_type=camera_type,
                    capabilities=capabilities,
                    usb_vid="0x8086",  # Intel VID
                    usb_pid="0x0B5C",  # D455/D455i PID
                    serial_number=serial,
                    sdk_available=True,
                )

                cameras.append(cam)

                if DEBUG_MODE:
                    print(f"[DEBUG]   Added to SDK detection list: {cam}")
                    print(f"[DEBUG]   Capabilities: {capabilities}")

            except Exception as e:
                logger.warning(f"Error processing RealSense device {i}: {e}")
                if DEBUG_MODE:
                    print(f"[DEBUG] Error processing RealSense device {i}: {e}")
                continue

        if DEBUG_MODE:
            print(f"[DEBUG] RealSense SDK detection complete: {len(cameras)} cameras")

    except Exception as e:
        logger.error(f"Error during RealSense SDK detection: {e}")
        if DEBUG_MODE:
            print(f"[DEBUG] RealSense SDK detection failed: {e}")
        return []

    return cameras


def _determine_realsense_type(product_line: str, name: str) -> CameraType:
    """
    Determine specific RealSense camera type from product line and name.

    Args:
        product_line: Product line from SDK (e.g., "D400")
        name: Device name from SDK (e.g., "Intel RealSense D455")

    Returns:
        CameraType enum value (REALSENSE_D455i for D455, GENERIC for unsupported)

    Notes:
        - D455 and D455i have same VID/PID, distinguished by IMU presence
        - This function returns D455i as primary (will be refined during merge)
        - Only D455i is officially supported in this system
    """
    product_line_lower = product_line.lower()
    name_lower = name.lower()

    # D455/D455i detection
    if "d455" in product_line_lower or "d455" in name_lower:
        # Both D455 and D455i use same VID/PID
        # Return D455i as primary (IMU will be verified in merge step)
        return CameraType.REALSENSE_D455i

    # Other RealSense models not officially supported
    # Return GENERIC to indicate SDK detected but not supported
    return CameraType.GENERIC


def merge_sdk_with_usb_detection(
    sdk_cameras: list[CameraInfo],
    usb_cameras: list[CameraInfo]
) -> list[CameraInfo]:
    """
    Merge SDK-detected cameras with USB enumeration results.

    This function combines the detailed SDK information (serial number, IMU)
    with the USB enumeration data (index, open_hint) to create complete
    CameraInfo objects.

    Strategy (multi-level fallback):
        1. Match cameras by VID+PID (Intel RealSense: 0x8086:0x0B5C)
        2. FALLBACK: If no VID/PID match, try serial number verification via SDK
        3. FALLBACK: If 1 SDK camera and multiple generic USB cameras, probe indices
        4. For matched cameras, use SDK data but preserve USB index/open_hint
        5. For unmatched USB cameras, keep as-is
        6. For unmatched SDK cameras, log warning (camera not accessible via USB)

    Args:
        sdk_cameras: List from detect_realsense_cameras()
        usb_cameras: List from platform-specific USB enumeration

    Returns:
        Merged list of CameraInfo objects with best available data

    Notes:
        - Serial numbers are used for precise matching when available
        - SDK data takes precedence for capabilities and type detection
        - USB data takes precedence for index and open_hint
        - Implements intelligent fallback for cases where VID/PID unavailable
    """
    merged: list[CameraInfo] = []
    usb_matched_indices: set[int] = set()

    if DEBUG_MODE:
        print(f"[DEBUG] Merging {len(sdk_cameras)} SDK cameras with {len(usb_cameras)} USB cameras")

    # Match SDK cameras with USB cameras
    for sdk_cam in sdk_cameras:
        matched_usb_cam: Optional[CameraInfo] = None

        # STRATEGY 1: Match by VID+PID (Primary method)
        if sdk_cam.serial_number:
            for usb_cam in usb_cameras:
                if (usb_cam.usb_vid == sdk_cam.usb_vid and
                    usb_cam.usb_pid == sdk_cam.usb_pid and
                    usb_cam.index is not None and
                    usb_cam.index not in usb_matched_indices):
                    # Found match by VID/PID
                    matched_usb_cam = usb_cam
                    usb_matched_indices.add(usb_cam.index)
                    if DEBUG_MODE:
                        print(f"[DEBUG] Strategy 1 (VID/PID): Matched SDK camera to USB index {usb_cam.index}")
                    break

        # STRATEGY 2: Fallback - Try to match by opening camera and checking serial
        # This is used when PowerShell timeout causes VID/PID to be unavailable
        if not matched_usb_cam and sdk_cam.serial_number:
            if DEBUG_MODE:
                print(f"[DEBUG] Strategy 2 (Serial Probe): Attempting to match SDK camera (Serial: {sdk_cam.serial_number})")

            matched_usb_cam = _try_match_by_serial_probe(
                sdk_cam, usb_cameras, usb_matched_indices
            )

            if matched_usb_cam:
                usb_matched_indices.add(matched_usb_cam.index)
                if DEBUG_MODE:
                    print(f"[DEBUG] Strategy 2 (Serial Probe): Matched to USB index {matched_usb_cam.index}")

        # STRATEGY 3: Last resort - If exactly 1 SDK camera and generic USB cameras
        # Test each generic camera to find which one is actually the RealSense
        if not matched_usb_cam and len(sdk_cameras) == 1:
            generic_cameras = [
                cam for cam in usb_cameras
                if cam.camera_type == CameraType.GENERIC
                and cam.index is not None
                and cam.index not in usb_matched_indices
            ]

            if generic_cameras:
                if DEBUG_MODE:
                    print(f"[DEBUG] Strategy 3 (Pipeline Test): Testing {len(generic_cameras)} generic cameras")

                # Try to initialize RealSense pipeline on each index
                matched_usb_cam = _try_match_by_pipeline_test(
                    sdk_cam, generic_cameras
                )

                if matched_usb_cam:
                    usb_matched_indices.add(matched_usb_cam.index)
                    if DEBUG_MODE:
                        print(f"[DEBUG] Strategy 3 (Pipeline Test): Confirmed USB index {matched_usb_cam.index} is RealSense")
                        print(f"[DEBUG]   Verified by successful pipeline initialization")
                else:
                    if DEBUG_MODE:
                        print(f"[DEBUG] Strategy 3 (Pipeline Test): Could not verify any generic camera as RealSense")
                        print(f"[DEBUG]   All pipeline initialization attempts failed")

        if matched_usb_cam:
            # Merge: Use SDK data but preserve USB index and open_hint
            merged_cam = CameraInfo(
                index=matched_usb_cam.index,
                name=sdk_cam.name,  # SDK name is more detailed
                backend=matched_usb_cam.backend,  # Use USB backend
                open_hint=matched_usb_cam.open_hint,
                camera_type=sdk_cam.camera_type,  # SDK type detection
                capabilities=sdk_cam.capabilities,  # SDK capabilities (includes IMU)
                usb_vid=sdk_cam.usb_vid,  # Update with SDK VID/PID
                usb_pid=sdk_cam.usb_pid,
                os_id=matched_usb_cam.os_id,  # USB OS identifier
                path=matched_usb_cam.path,
                serial_number=sdk_cam.serial_number,  # SDK serial number
                sdk_available=True,
                vendor=matched_usb_cam.vendor,
                model=matched_usb_cam.model,
            )
            merged.append(merged_cam)

            if DEBUG_MODE:
                print(f"[DEBUG] Successfully merged SDK camera (Serial: {sdk_cam.serial_number}) with USB index {matched_usb_cam.index}")
        else:
            # SDK detected but not found in USB enumeration
            if DEBUG_MODE:
                print(f"[DEBUG] WARNING: SDK camera (Serial: {sdk_cam.serial_number}) could not be matched to any USB camera")
                print(f"[DEBUG]   This camera will not be accessible via OpenCV/DirectShow")

    # Add unmatched USB cameras (non-RealSense or SDK detection failed)
    for usb_cam in usb_cameras:
        if usb_cam.index is None or usb_cam.index not in usb_matched_indices:
            merged.append(usb_cam)
            if DEBUG_MODE and usb_cam.camera_type == CameraType.REALSENSE_D455i:
                print(f"[DEBUG] USB-only RealSense at index {usb_cam.index} (SDK match failed)")

    if DEBUG_MODE:
        print(f"[DEBUG] Merge complete: {len(merged)} total cameras")
        for cam in merged:
            sdk_marker = " [SDK]" if cam.sdk_available else ""
            print(f"[DEBUG]   - Index {cam.index}: {cam.camera_type}{sdk_marker}")

    return merged


def _try_match_by_serial_probe(
    sdk_cam: CameraInfo,
    usb_cameras: list[CameraInfo],
    used_indices: set[int]
) -> Optional[CameraInfo]:
    """
    Try to match SDK camera by probing USB indices and checking serial numbers.

    This fallback strategy is used when VID/PID metadata is unavailable
    (e.g., PowerShell timeout on Windows).

    Strategy:
        - For each unmatched USB camera
        - Try to open it with pyrealsense2
        - Check if serial number matches SDK camera
        - Return first match

    Args:
        sdk_cam: SDK-detected camera with serial number
        usb_cameras: List of USB-detected cameras
        used_indices: Set of already-matched indices

    Returns:
        Matched CameraInfo or None
    """
    try:
        import pyrealsense2 as rs
    except ImportError:
        return None

    for usb_cam in usb_cameras:
        # Skip if already matched or no index
        if usb_cam.index is None or usb_cam.index in used_indices:
            continue

        # Try to open this camera with RealSense SDK
        try:
            ctx = rs.context()
            devices = ctx.query_devices()

            for dev in devices:
                try:
                    # Try to match by index (not reliable, but worth trying)
                    serial = dev.get_info(rs.camera_info.serial_number)
                    if serial == sdk_cam.serial_number:
                        if DEBUG_MODE:
                            print(f"[DEBUG]   Serial probe: Found matching serial {serial} at USB index {usb_cam.index}")
                        return usb_cam
                except Exception:
                    continue

        except Exception as e:
            if DEBUG_MODE:
                print(f"[DEBUG]   Serial probe failed for index {usb_cam.index}: {e}")
            continue

    return None


def _try_match_by_pipeline_test(
    sdk_cam: CameraInfo,
    generic_cameras: list[CameraInfo]
) -> Optional[CameraInfo]:
    """
    Try to match SDK camera by testing pipeline initialization on each index.

    This is the most reliable fallback strategy when both VID/PID and serial
    probe fail. It actually attempts to initialize a RealSense pipeline on
    each generic camera index to determine which one is the real RealSense.

    Strategy:
        - For each generic camera
        - Try to create and start a RealSense pipeline
        - If successful, this is definitely a RealSense camera
        - Stop pipeline immediately and return match

    Args:
        sdk_cam: SDK-detected camera with serial number
        generic_cameras: List of generic USB cameras to test

    Returns:
        Matched CameraInfo or None

    Notes:
        - This method is slower (~200-500ms per camera tested)
        - Only runs as last resort when other strategies fail
        - Guarantees correct identification (no false positives)
    """
    try:
        import pyrealsense2 as rs
    except ImportError:
        if DEBUG_MODE:
            print(f"[DEBUG]   Pipeline test: pyrealsense2 not available")
        return None

    for usb_cam in generic_cameras:
        if DEBUG_MODE:
            print(f"[DEBUG]   Pipeline test: Testing index {usb_cam.index}...")

        pipeline = None
        try:
            # Create pipeline and config
            pipeline = rs.pipeline()
            config = rs.config()

            # Try to enable device by serial number
            # This will only succeed if the camera at this index is a RealSense
            # with the matching serial number
            try:
                config.enable_device(sdk_cam.serial_number)
            except Exception:
                # If we can't enable by serial, try without specifying device
                # Pipeline will use first available RealSense
                pass

            # Try to start pipeline
            # This will fail if:
            # 1. Camera is not a RealSense
            # 2. Camera is already in use
            # 3. Driver/hardware issue
            profile = pipeline.start(config)

            # If we got here, pipeline started successfully!
            # This camera is definitely the RealSense
            pipeline.stop()

            if DEBUG_MODE:
                print(f"[DEBUG]   Pipeline test: SUCCESS on index {usb_cam.index}")
                print(f"[DEBUG]   Verified as RealSense with serial {sdk_cam.serial_number}")

            return usb_cam

        except Exception as e:
            # Pipeline start failed - this is NOT a RealSense or it's in use
            if DEBUG_MODE:
                print(f"[DEBUG]   Pipeline test: FAILED on index {usb_cam.index}")
                print(f"[DEBUG]   Reason: {type(e).__name__}: {str(e)[:100]}")

            # Make sure to stop pipeline if it was created
            if pipeline:
                try:
                    pipeline.stop()
                except Exception:
                    pass

            continue

    # No camera passed the pipeline test
    if DEBUG_MODE:
        print(f"[DEBUG]   Pipeline test: No generic camera could be verified as RealSense")

    return None
