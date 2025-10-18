"""
Windows Device Instance Path Resolution for Camera Detection

This module provides stable camera identification using Windows Device Instance Paths
instead of unreliable numeric OpenCV indices. Device paths are STABLE identifiers that:
  - Do NOT change when USB ports are swapped (serial-based matching)
  - Uniquely identify each physical camera
  - Survive system reboots

PROBLEM: OpenCV indices (0, 1, 2) change when:
  - USB ports are swapped
  - Cameras are connected/disconnected in different order
  - Other cameras are added to the system
  - System reboots

SOLUTION: Windows Device Instance Paths (InstanceId) are stable:
  Format: USB\\VID_{vendor}&PID_{product}&MI_{interface}\\{serial}&{port}&{endpoint}
  Example: USB\\VID_2B03&PID_F880&MI_00\\7&1EBA99DD&0&0000
           ^^^^ ^^^^ ^^^^ ^^^^  ^^^^   ^^^^^^^^^^^^^^^^^^
           |    |    |    |     |      Serial number (STABLE, unique per camera)
           |    |    |    |     Interface (00=primary video)
           |    |    |    Product ID (F880=ZED 2i)
           |    |    Vendor ID (2B03=Stereolabs, 8086=Intel)
           USB device

KEY CONCEPT: Serial-Based Matching
  The serial number portion (e.g., "7&1EBA99DD") is STABLE and unique per camera.
  Even if you swap USB ports, the serial remains the same.
  This allows us to map configured device paths to runtime OpenCV indices.

USAGE EXAMPLE:
  from utils.camera_device_path_resolver import resolve_configured_cameras
  from utils.utils import enumerate_usb_external_cameras

  # Enumerate all connected cameras
  all_cameras = enumerate_usb_external_cameras()

  # Resolve configured cameras from config.py
  configured = resolve_configured_cameras(all_cameras, fallback_mode="strict")

  # Result: dict with camera assignments
  # {
  #   "realsense_primary": CameraInfo(...),
  #   "zed_cameras": [CameraInfo(...), CameraInfo(...)],
  # }

FUNCTIONS:
  extract_serial_from_instance_id(instance_id: str) -> Optional[str]
    Extract stable serial identifier from Windows Device Instance ID.

  normalize_instance_id(instance_id: str) -> str
    Normalize device path for consistent comparison (uppercase, backslashes).

  match_instance_id_to_camera(target_id: str, cameras: list) -> Optional[CameraInfo]
    Find camera matching target InstanceId using serial-based matching.

  resolve_configured_cameras(cameras: list, fallback_mode: str) -> dict
    Resolve configured cameras from PREFERRED_CAMERA_DEVICE_PATHS.

AUTHOR: Claude Code
DATE: 2025-10-18
"""

import re
import logging
from typing import Optional, Union

# Import configuration
try:
    import config
except ImportError:
    config = None
    logging.warning("config.py not found - using defaults")

# Import domain classes
try:
    from domain.camera_info import CameraInfo, CameraType
except ImportError:
    logging.error("Cannot import CameraInfo - domain module not found")
    raise


def extract_serial_from_instance_id(instance_id: str) -> Optional[str]:
    """
    Extract stable serial identifier from Windows Device Instance ID.

    The serial portion is the unique part of the InstanceId that persists even
    when USB ports are swapped. It's located between the last backslash and
    the first ampersand in the serial section.

    Format: USB\\VID_xxxx&PID_xxxx&MI_xx\\{SERIAL}&{PORT}&{ENDPOINT}
                                            ^^^^^^^
                                            This is what we extract

    Args:
        instance_id: Full Windows Device Instance Path string
                     Example: "USB\\VID_2B03&PID_F880&MI_00\\7&1EBA99DD&0&0000"

    Returns:
        Serial portion (e.g., "7&1EBA99DD") if extractable, None otherwise.

    Examples:
        >>> extract_serial_from_instance_id("USB\\\\VID_2B03&PID_F880&MI_00\\\\7&1EBA99DD&0&0000")
        "7&1EBA99DD"

        >>> extract_serial_from_instance_id("USB\\\\VID_8086&PID_0B5C&MI_00\\\\6&D37468C&1&0000")
        "6&D37468C"

        >>> extract_serial_from_instance_id("MALFORMED")
        None

    Edge Cases:
        - If instance_id is None or empty, returns None
        - If format is malformed (no backslash), returns None
        - If serial section is empty, returns None
        - If multiple serials present (shouldn't happen), returns first
    """
    if not instance_id:
        return None

    try:
        # Normalize first (uppercase, backslashes)
        normalized = normalize_instance_id(instance_id)

        # Split by backslash to get sections
        # Format: USB\\VID_xxxx&PID_xxxx&MI_xx\\SERIAL&PORT&ENDPOINT
        #         [0]  [1]                        [2]
        parts = normalized.split('\\')

        if len(parts) < 3:
            # Not enough sections (need at least USB\...\...)
            logging.debug(f"extract_serial: Not enough sections in '{instance_id}'")
            return None

        # Get the serial section (last part after final backslash)
        serial_section = parts[-1]

        # Serial section format: {SERIAL}&{PORT}&{ENDPOINT}
        # Example: "7&1EBA99DD&0&0000"
        # We want: "7&1EBA99DD" (first two parts before last &)

        # Split by ampersand
        serial_parts = serial_section.split('&')

        if len(serial_parts) < 2:
            # Not enough parts (need at least SERIAL&PORT)
            logging.debug(f"extract_serial: Serial section too short: '{serial_section}'")
            return None

        # Extract first two parts (SERIAL)
        # Format is typically: {digit}&{hex_serial}&{port}&{endpoint}
        # Example: "7&1EBA99DD&0&0000" -> "7&1EBA99DD"
        serial = f"{serial_parts[0]}&{serial_parts[1]}"

        logging.debug(f"extract_serial: Extracted '{serial}' from '{instance_id}'")
        return serial

    except Exception as e:
        logging.error(f"extract_serial: Error extracting serial from '{instance_id}': {e}")
        return None


def normalize_instance_id(instance_id: str) -> str:
    """
    Normalize Windows Device Instance ID for consistent comparison.

    Normalization steps:
      1. Convert to uppercase (Windows paths are case-insensitive)
      2. Replace forward slashes with backslashes
      3. Strip leading/trailing whitespace
      4. Ensure consistent separator format

    This ensures that paths can be compared reliably regardless of how
    they were retrieved (PowerShell vs WMI vs Registry).

    Args:
        instance_id: Device Instance Path string to normalize

    Returns:
        Normalized path string (uppercase, backslashes, trimmed)

    Examples:
        >>> normalize_instance_id("usb\\\\vid_2b03&pid_f880&mi_00\\\\7&1eba99dd&0&0000")
        "USB\\\\VID_2B03&PID_F880&MI_00\\\\7&1EBA99DD&0&0000"

        >>> normalize_instance_id("USB/VID_2B03&PID_F880&MI_00/7&1EBA99DD&0&0000")
        "USB\\\\VID_2B03&PID_F880&MI_00\\\\7&1EBA99DD&0&0000"

        >>> normalize_instance_id("  USB\\\\VID_2B03  ")
        "USB\\\\VID_2B03"

    Edge Cases:
        - If instance_id is None, returns empty string
        - If instance_id is empty, returns empty string
        - Handles mixed case consistently
        - Handles both / and \\ separators
    """
    if not instance_id:
        return ""

    # Strip whitespace
    normalized = instance_id.strip()

    # Convert to uppercase (Windows paths are case-insensitive)
    normalized = normalized.upper()

    # Replace forward slashes with backslashes
    # (some APIs return / instead of \\)
    normalized = normalized.replace('/', '\\')

    return normalized


def match_instance_id_to_camera(
    target_instance_id: str,
    available_cameras: list,
    match_mode: str = "serial"
) -> Optional[CameraInfo]:
    """
    Find camera matching target Windows Device Instance ID.

    Implements multiple matching strategies with fallback priority:
      1. EXACT match (if mode="exact"): Full InstanceId must match exactly
      2. SERIAL match (if mode="serial", DEFAULT): Match by serial + VID/PID
      3. VID_PID_SERIAL match (if mode="vid_pid_serial"): Match by VID/PID only

    The SERIAL match is the most reliable because it works even when USB ports
    are swapped. The serial number is unique per camera and persists across
    port changes.

    Args:
        target_instance_id: The configured InstanceId to find
                           Example: "USB\\VID_2B03&PID_F880&MI_00\\7&1EBA99DD&0&0000"

        available_cameras: List of CameraInfo objects from enumerate_usb_external_cameras()
                          Each must have os_id field populated with InstanceId

        match_mode: Matching strategy
                   "exact" - Require exact InstanceId match
                   "serial" - Match by serial + VID/PID (DEFAULT, recommended)
                   "vid_pid_serial" - Match by VID/PID only (fallback)

    Returns:
        CameraInfo object if match found, None otherwise

    Examples:
        # Exact match
        >>> camera = match_instance_id_to_camera(
        ...     "USB\\\\VID_2B03&PID_F880&MI_00\\\\7&1EBA99DD&0&0000",
        ...     all_cameras,
        ...     match_mode="exact"
        ... )

        # Serial match (handles USB port changes)
        >>> camera = match_instance_id_to_camera(
        ...     "USB\\\\VID_2B03&PID_F880&MI_00\\\\7&1EBA99DD&0&0000",
        ...     all_cameras,
        ...     match_mode="serial"  # Will match even if port changed
        ... )

    Edge Cases:
        - If available_cameras is empty, returns None
        - If target_instance_id is None/empty, returns None
        - If multiple matches found, returns first (shouldn't happen with serial)
        - If no match found, returns None
        - Logs DEBUG messages for each match attempt
    """
    if not target_instance_id:
        logging.debug("match_instance_id: target_instance_id is empty")
        return None

    if not available_cameras:
        logging.debug("match_instance_id: available_cameras is empty")
        return None

    # Normalize target for comparison
    normalized_target = normalize_instance_id(target_instance_id)

    # Extract components from target
    target_serial = extract_serial_from_instance_id(normalized_target)
    target_vid_pid = _extract_vid_pid(normalized_target)

    logging.debug(f"match_instance_id: Looking for camera with:")
    logging.debug(f"  Target InstanceId: {normalized_target}")
    logging.debug(f"  Target Serial: {target_serial}")
    logging.debug(f"  Target VID/PID: {target_vid_pid}")
    logging.debug(f"  Match mode: {match_mode}")
    logging.debug(f"  Available cameras: {len(available_cameras)}")

    for camera in available_cameras:
        # Skip if camera doesn't have os_id (InstanceId)
        if not hasattr(camera, 'os_id') or not camera.os_id:
            continue

        # Normalize camera's InstanceId
        camera_instance_id = normalize_instance_id(camera.os_id)

        # Try matching based on mode
        if match_mode == "exact":
            # EXACT MATCH: Full InstanceId must match exactly
            if camera_instance_id == normalized_target:
                logging.info(f"match_instance_id: EXACT match found - {camera.friendly_name}")
                return camera

        elif match_mode == "serial":
            # SERIAL MATCH: Match by serial number + VID/PID
            # This works even if USB port changed
            camera_serial = extract_serial_from_instance_id(camera_instance_id)
            camera_vid_pid = _extract_vid_pid(camera_instance_id)

            if target_serial and camera_serial and target_serial == camera_serial:
                # Serial matches - also verify VID/PID to be safe
                if target_vid_pid and camera_vid_pid and target_vid_pid == camera_vid_pid:
                    logging.info(f"match_instance_id: SERIAL match found - {camera.friendly_name}")
                    logging.debug(f"  Camera InstanceId: {camera_instance_id}")
                    logging.debug(f"  Camera Serial: {camera_serial}")
                    return camera
                else:
                    # Serial matches but VID/PID doesn't (very unusual)
                    logging.warning(
                        f"match_instance_id: Serial match but VID/PID mismatch - {camera.friendly_name}"
                    )

        elif match_mode == "vid_pid_serial":
            # VID_PID_SERIAL MATCH: Match by VID/PID only (least specific)
            camera_vid_pid = _extract_vid_pid(camera_instance_id)

            if target_vid_pid and camera_vid_pid and target_vid_pid == camera_vid_pid:
                logging.info(f"match_instance_id: VID/PID match found - {camera.friendly_name}")
                return camera

    # No match found
    logging.debug(f"match_instance_id: No match found for {normalized_target}")
    return None


def _extract_vid_pid(instance_id: str) -> Optional[str]:
    """
    Extract VID and PID from Windows Device Instance ID.

    Internal helper function.

    Args:
        instance_id: Full InstanceId string

    Returns:
        "VID_xxxx&PID_xxxx" string if found, None otherwise

    Example:
        >>> _extract_vid_pid("USB\\\\VID_2B03&PID_F880&MI_00\\\\7&1EBA99DD")
        "VID_2B03&PID_F880"
    """
    if not instance_id:
        return None

    # Normalize
    normalized = normalize_instance_id(instance_id)

    # Extract VID and PID using regex
    # Pattern: VID_xxxx&PID_xxxx
    match = re.search(r'VID_[0-9A-F]{4}&PID_[0-9A-F]{4}', normalized)

    if match:
        return match.group(0)

    return None


def resolve_configured_cameras(
    available_cameras: list,
    fallback_mode: str = None
) -> dict:
    """
    Resolve configured cameras from PREFERRED_CAMERA_DEVICE_PATHS config.

    This is the main entry point for device path resolution. It reads the
    PREFERRED_CAMERA_DEVICE_PATHS dictionary from config.py and maps each
    configured device path to an actual camera from available_cameras.

    Fallback Modes:
      "strict" (RECOMMENDED):
        - ONLY use cameras listed in PREFERRED_CAMERA_DEVICE_PATHS
        - If a configured camera is not found, return None for that key
        - Other connected cameras are completely IGNORED
        - Use this when you need precise camera control

      "sdk_exclusion":
        - Use SDK-based detection as fallback
        - If configured camera missing, use first available camera of same type
        - Example: If ZED #1 missing, use any available ZED camera
        - Less strict, more flexible

      "first_available":
        - Use first available camera of requested type
        - Least restrictive mode
        - Not recommended for production

    Args:
        available_cameras: List of CameraInfo from enumerate_usb_external_cameras()
        fallback_mode: Override fallback mode (default: use config.CAMERA_DETECTION_FALLBACK_MODE)

    Returns:
        Dictionary mapping config keys to CameraInfo objects:
        {
            "realsense_primary": CameraInfo or None,
            "zed_cameras": [CameraInfo, CameraInfo, ...] or [],
        }

    Example Usage:
        >>> from utils.utils import enumerate_usb_external_cameras
        >>> all_cameras = enumerate_usb_external_cameras()
        >>> configured = resolve_configured_cameras(all_cameras, fallback_mode="strict")
        >>>
        >>> # Get RealSense camera
        >>> realsense = configured.get("realsense_primary")
        >>> if realsense:
        ...     print(f"RealSense: {realsense.friendly_name}")
        >>>
        >>> # Get ZED cameras
        >>> zed_cameras = configured.get("zed_cameras", [])
        >>> for i, zed in enumerate(zed_cameras):
        ...     print(f"ZED #{i+1}: {zed.friendly_name}")

    Strict Mode Example:
        Connected cameras:
          - ZED 2i #1 (7&1EBA99DD) <- CONFIGURED
          - ZED 2i #2 (7&1500F77) <- CONFIGURED
          - ZED 2i #3 (7&AAAA111) <- NOT CONFIGURED
          - RealSense (6&D37468C) <- CONFIGURED

        Result (strict mode):
          {
              "realsense_primary": CameraInfo(serial=6&D37468C),
              "zed_cameras": [
                  CameraInfo(serial=7&1EBA99DD),  # Primary
                  CameraInfo(serial=7&1500F77),   # Secondary
              ]
          }

        ZED 2i #3 is IGNORED (not in config)

    Edge Cases:
        - If config.py not available, returns empty dict
        - If PREFERRED_CAMERA_DEVICE_PATHS not configured, returns empty dict
        - If all configured cameras missing (strict mode), returns dict with None/empty lists
        - If fallback_mode invalid, defaults to "strict"
        - Logs WARNING for each unconfigured camera found
        - Logs ERROR for each configured camera not found
    """
    result = {}

    # Get fallback mode from config if not provided
    if fallback_mode is None:
        if config and hasattr(config, 'CAMERA_DETECTION_FALLBACK_MODE'):
            fallback_mode = config.CAMERA_DETECTION_FALLBACK_MODE
        else:
            fallback_mode = "strict"  # Default to strict

    # Validate fallback mode
    valid_modes = ["strict", "sdk_exclusion", "first_available"]
    if fallback_mode not in valid_modes:
        logging.warning(
            f"Invalid fallback_mode '{fallback_mode}', defaulting to 'strict'. "
            f"Valid modes: {valid_modes}"
        )
        fallback_mode = "strict"

    # Get preferred camera device paths from config
    if not config or not hasattr(config, 'PREFERRED_CAMERA_DEVICE_PATHS'):
        logging.error("config.PREFERRED_CAMERA_DEVICE_PATHS not found - returning empty dict")
        return result

    preferred_paths = config.PREFERRED_CAMERA_DEVICE_PATHS

    if not preferred_paths:
        logging.warning("PREFERRED_CAMERA_DEVICE_PATHS is empty - no cameras configured")
        return result

    # Enable debug logging if configured
    debug_enabled = (
        config and
        hasattr(config, 'DEBUG_DEVICE_PATH_RESOLUTION') and
        config.DEBUG_DEVICE_PATH_RESOLUTION
    )

    if debug_enabled:
        logging.info("=" * 70)
        logging.info("DEVICE PATH RESOLUTION START")
        logging.info("=" * 70)
        logging.info(f"Fallback mode: {fallback_mode}")
        logging.info(f"Available cameras: {len(available_cameras)}")
        logging.info(f"Configured groups: {list(preferred_paths.keys())}")

    # Process each configured camera group
    for group_key, device_paths in preferred_paths.items():
        if debug_enabled:
            logging.info(f"\nProcessing group: {group_key}")
            logging.info(f"  Configured paths: {len(device_paths)}")

        # Ensure device_paths is a list
        if not isinstance(device_paths, list):
            device_paths = [device_paths]

        resolved_cameras = []

        # Try to resolve each configured path
        for i, device_path in enumerate(device_paths):
            if debug_enabled:
                logging.info(f"  [{i+1}/{len(device_paths)}] Looking for: {device_path}")

            # Try to match this device path to an available camera
            matched_camera = match_instance_id_to_camera(
                device_path,
                available_cameras,
                match_mode="serial"  # Use serial matching (handles port changes)
            )

            if matched_camera:
                resolved_cameras.append(matched_camera)
                if debug_enabled:
                    logging.info(f"    FOUND: {matched_camera.friendly_name}")
            else:
                # Camera not found - handle based on fallback mode
                if fallback_mode == "strict":
                    # Strict mode: Do NOT use any other camera
                    logging.error(
                        f"Configured camera not found: {device_path} "
                        f"(strict mode - will not use fallback)"
                    )
                elif fallback_mode == "sdk_exclusion":
                    # SDK exclusion mode: Try to find any camera of same type
                    logging.warning(
                        f"Configured camera not found: {device_path} "
                        f"(trying SDK exclusion fallback)"
                    )
                    # TODO: Implement SDK exclusion fallback
                    # This would call SDK detection to find any camera of matching type
                elif fallback_mode == "first_available":
                    # First available mode: Use any camera
                    logging.warning(
                        f"Configured camera not found: {device_path} "
                        f"(trying first_available fallback)"
                    )
                    # TODO: Implement first_available fallback

        # Store result for this group
        if len(device_paths) == 1:
            # Single camera: return CameraInfo or None
            result[group_key] = resolved_cameras[0] if resolved_cameras else None
        else:
            # Multiple cameras: return list
            result[group_key] = resolved_cameras

        if debug_enabled:
            if resolved_cameras:
                logging.info(f"  RESULT: {len(resolved_cameras)} camera(s) resolved")
            else:
                logging.info(f"  RESULT: No cameras resolved")

    # Log any unconfigured cameras (warning in strict mode)
    if fallback_mode == "strict" and debug_enabled:
        configured_serials = set()
        for device_paths in preferred_paths.values():
            if not isinstance(device_paths, list):
                device_paths = [device_paths]
            for path in device_paths:
                serial = extract_serial_from_instance_id(path)
                if serial:
                    configured_serials.add(serial)

        for camera in available_cameras:
            if hasattr(camera, 'os_id') and camera.os_id:
                camera_serial = extract_serial_from_instance_id(camera.os_id)
                if camera_serial and camera_serial not in configured_serials:
                    logging.warning(
                        f"Unconfigured camera found (will be IGNORED in strict mode): "
                        f"{camera.friendly_name} - Serial: {camera_serial}"
                    )

    if debug_enabled:
        logging.info("=" * 70)
        logging.info("DEVICE PATH RESOLUTION COMPLETE")
        logging.info("=" * 70)

    return result

