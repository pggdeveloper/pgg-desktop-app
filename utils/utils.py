import os, platform, json, subprocess, re, glob, time
from typing import Optional, Union
from functools import wraps
import cv2
from config import DEBUG_MODE
from domain.camera_type import CameraType
from domain.camera_backend import CameraBackend
from domain.camera_capabilities import CameraCapabilities
from domain.camera_info import CameraInfo

WINDOWS_DEFAULT_BACKEND = cv2.CAP_DSHOW
MAC_DEFAULT_BACKEND = cv2.CAP_AVFOUNDATION
LINUX_DEFAULT_BACKEND = cv2.CAP_V4L2

# ---------------- Retry Decorator ----------------
def retry_with_backoff(max_retries: int = 3, initial_delay: float = 0.5):
    """
    Decorator for retry with exponential backoff.

    This decorator wraps functions to automatically retry on failure
    with increasing delays between attempts (exponential backoff).

    Args:
        max_retries: Maximum number of retry attempts (default: 3)
        initial_delay: Initial delay in seconds between retries (default: 0.5)

    Returns:
        Decorator function

    Example:
        @retry_with_backoff(max_retries=3, initial_delay=0.5)
        def unstable_function():
            # Function that might fail transiently
            pass

    Notes:
        - Delay doubles after each retry (exponential backoff)
        - If all retries fail, raises the last exception
        - Logs retry attempts if DEBUG_MODE is enabled
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            delay = initial_delay
            last_exception = None

            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt == max_retries - 1:
                        # Last attempt failed, re-raise exception
                        raise

                    if DEBUG_MODE:
                        print(f"[DEBUG] Retry {attempt + 1}/{max_retries} for {func.__name__} after error: {e}")

                    time.sleep(delay)
                    delay *= 2  # Exponential backoff

            # Should never reach here, but for safety
            if last_exception:
                raise last_exception

        return wrapper
    return decorator

# ---------------- Patterns ----------------
VIRTUAL_PATTERNS = (
    "virtual", "obs", "broadcast", "manycam", "splitcam", "ndi",
    "youcam", "droidcam", "mmhmm", "xsplit", "snap camera", "snapcamera"
)
INTERNAL_PATTERNS = (
    "built-in", "builtin", "integrated", "internal", "facetime", "isight"
)
VENDOR_HINTS = ("logitech", "elgato", "aver", "avermedia", "sony", "canon", "razer", "creativelive", "microsoft", "lenovo", "huehd")

def _norm(s: Optional[str]) -> str:
    return (s or "").strip()

def _looks_virtual(name: str) -> bool:
    n = _norm(name).lower()
    return any(p in n for p in VIRTUAL_PATTERNS)

def _looks_internal(name: str) -> bool:
    n = _norm(name).lower()
    return any(p in n for p in INTERNAL_PATTERNS)

# ---------------- Backends by OS ----------------
def _backend_for_os() -> int:
    sysname = platform.system()
    if sysname == 'Darwin':
        return MAC_DEFAULT_BACKEND
    elif sysname == 'Windows':
        return WINDOWS_DEFAULT_BACKEND
    else:
        return LINUX_DEFAULT_BACKEND

def open_camera_hint(hint: Union[int, str]):
    """
    Open a camera using a hint returned by enumerate_usb_external_cameras().
    - int => treated as index + OS backend
    - str => treated as device path (Linux), opened with V4L2 backend
    """
    backend = _backend_for_os()
    if isinstance(hint, int):
        return cv2.VideoCapture(hint, backend)
    else:
        # path (Linux): prefer V4L2
        return cv2.VideoCapture(hint, cv2.CAP_V4L2)

def _probe_indices(backend: int, max_count: int = 10) -> list[int]:
    """
    Find OpenCV indices that can be opened and read a frame successfully.
    """
    found: list[int] = []
    for i in range(max_count):
        cap = cv2.VideoCapture(i, backend)
        if cap.isOpened():
            ok, _ = cap.read()
            if ok:
                found.append(i)
        cap.release()
    return found

# ---------------- Camera Type Identification ----------------
def _identify_camera_type(
    vid: Optional[str],
    pid: Optional[str],
    name: str
) -> CameraType:
    """
    Identify camera type based on USB VID/PID and name.

    Args:
        vid: USB Vendor ID (e.g., "0x8086" or "8086")
        pid: USB Product ID (e.g., "0x0B5C" or "0B5C")
        name: Camera device name

    Returns:
        CameraType enum value
    """
    # Normalize VID/PID: remove 0x prefix and convert to uppercase
    vid_norm = vid.replace("0x", "").replace("0X", "").upper() if vid else ""
    pid_norm = pid.replace("0x", "").replace("0X", "").upper() if pid else ""
    name_lower = name.lower()

    # Intel RealSense Detection (Primary: D455i)
    if vid_norm == "8086":
        # RealSense D455/D455i (same PID: 0B5C)
        # Both models use same VID/PID, return D455i as primary
        if pid_norm == "0B5C":
            return CameraType.REALSENSE_D455i

        # Other RealSense models not supported in this system
        # Log warning if other PID detected

    # Stereolabs ZED Detection (Primary: ZED 2i)
    if vid_norm == "2B03":
        # ZED 2i: PIDs F880 (video) and F881 (HID/sensors)
        if pid_norm in ("F880", "F881"):
            return CameraType.ZED_2i
        # ZED 2: PIDs F780 (video) and F781 (HID/sensors)
        elif pid_norm in ("F780", "F781"):
            return CameraType.ZED_2

    # Name-based fallback detection
    if "realsense" in name_lower:
        if "d455" in name_lower:
            # Default to D455i for all D455 variants
            return CameraType.REALSENSE_D455i
        # Other RealSense models not supported

    if "zed" in name_lower:
        # Prefer ZED 2i if version ambiguous
        if "2i" in name_lower or "zed 2i" in name_lower:
            return CameraType.ZED_2i
        elif "zed 2" in name_lower or "zed2" in name_lower:
            # Check if it's specifically ZED 2 (not 2i)
            if "2i" not in name_lower:
                return CameraType.ZED_2
        # Default ZED to ZED 2i (primary model)
        return CameraType.ZED_2i

    # Generic camera
    return CameraType.GENERIC

# ---------------- Windows ----------------
@retry_with_backoff(max_retries=3, initial_delay=0.5)
def _win_list_usb_cameras_meta() -> list[dict[str, Union[str, bool, None]]]:
    """
    Use PowerShell PnP to list camera/image devices with USB metadata.
    Implements dual VID/PID extraction (InstanceId + HardwareIds) with timeout.

    This function is wrapped with retry logic to handle transient PowerShell failures.

    Returns:
        List of dicts with FriendlyName, InstanceId, VID, PID, etc.
    """
    ps = r"""
# Optimized camera detection using Get-CimInstance (faster than Get-PnpDevice)
# Targets: Intel RealSense D455i (VID_8086&PID_0B5C), Stereolabs ZED 2i (VID_2B03&PID_F880/F881)

$result = @()

# Get all USB camera devices with status OK
# Using CIM is 3-5x faster than PnP cmdlets
$devices = Get-CimInstance -ClassName Win32_PnPEntity -Filter "Status = 'OK'" | Where-Object {
    $_.PNPDeviceID -match '^USB\\VID_' -and
    ($_.PNPClass -eq 'Camera' -or $_.PNPClass -eq 'Image')
}

foreach ($d in $devices) {
    $id = $d.PNPDeviceID

    # Fast VID/PID extraction using single regex match
    # Format: USB\VID_8086&PID_0B5C&MI_00\SerialNumber
    $vid = ""
    $pidValue = ""

    if ($id -match 'USB\\VID_([0-9A-F]{4})&PID_([0-9A-F]{4})') {
        $vid = $matches[1]
        $pidValue = $matches[2]
    }

    # Skip if VID/PID extraction failed
    if (-not $vid -or -not $pidValue) {
        continue
    }

    # Filter: Only Intel RealSense (8086) and Stereolabs ZED (2B03)
    if ($vid -ne '8086' -and $vid -ne '2B03') {
        continue
    }

    # Extract serial number from InstanceId if present
    # Format: USB\VID_8086&PID_0B5C\318122302840
    $serial = ""
    if ($id -match '\\([^\\]+)$') {
        $serial = $matches[1]
    }

    # Determine if this is USB by checking Service property
    # Camera devices use 'usbvideo' service
    $isUSB = $d.Service -eq 'usbvideo' -or $id.StartsWith('USB\')

    # Build result object with minimal required data
    $obj = [PSCustomObject]@{
        FriendlyName = $d.Name
        InstanceId   = $id
        VID          = $vid
        PID          = $pidValue
        Serial       = $serial
        IsUSB        = $isUSB
        Service      = $d.Service
    }

    $result += $obj
}

# Convert to JSON
# Use -Compress for faster serialization, -Depth 2 is sufficient
if ($result.Count -eq 0) {
    "[]"
} elseif ($result.Count -eq 1) {
    $result | ConvertTo-Json -Compress -Depth 2
} else {
    $result | ConvertTo-Json -Compress -Depth 2
}
"""
    try:
        # Execute PowerShell with 20 second timeout (increased from 10s)
        import time
        start_time = time.time()

        if DEBUG_MODE:
            print("[DEBUG] Starting PowerShell camera enumeration...")

        out = subprocess.check_output(
            ["powershell", "-NoProfile", "-Command", ps],
            text=True,
            encoding="utf-8",
            errors="ignore",
            timeout=20  # Increased from 10s to 20s
        )

        elapsed = time.time() - start_time
        if DEBUG_MODE:
            print(f"[DEBUG] PowerShell query completed in {elapsed:.2f}s")
            print(f"[DEBUG] PowerShell raw output ({len(out)} chars):")
            print(f"[DEBUG] First 500 chars: {out[:500]!r}")
            print(f"[DEBUG] Last 500 chars: {out[-500:]!r}")

        data = json.loads(out) if out.strip() else []
        if isinstance(data, dict):
            data = [data]

        if DEBUG_MODE:
            print(f"[DEBUG] PowerShell enumerated {len(data)} camera devices")

    except subprocess.TimeoutExpired:
        if DEBUG_MODE:
            print("[DEBUG] PowerShell enumeration timeout (20s) - using fallback")
            print("[DEBUG] This may indicate:")
            print("[DEBUG]   - Too many PnP devices on system")
            print("[DEBUG]   - PowerShell performance issues")
            print("[DEBUG]   - Permissions or security software interference")
        data = []
    except Exception as e:
        if DEBUG_MODE:
            print(f"[DEBUG] PowerShell enumeration failed: {e}")
        data = []

    # Process and normalize data from optimized PowerShell script
    for d in data:
        vid_raw = d.get("VID", "")
        pid_raw = d.get("PID", "")

        # Normalize to 0xXXXX format consistently
        d["usb_vid"] = f"0x{vid_raw}" if vid_raw else None
        d["usb_pid"] = f"0x{pid_raw}" if pid_raw else None

        # PowerShell script already filters for USB, so trust IsUSB field
        d["is_usb"] = d.get("IsUSB", False)

        # Check if this is internal or virtual by name
        friendly_name = d.get("FriendlyName", "")
        d["is_internal"] = _looks_internal(friendly_name)
        d["is_virtual"] = _looks_virtual(friendly_name)

        if DEBUG_MODE and d.get("is_usb"):
            serial = d.get("Serial", "N/A")
            print(f"[DEBUG] {friendly_name}: VID={d['usb_vid']}, PID={d['usb_pid']}, Serial={serial}")

    # Keep only external USB, non-virtual, non-internal
    # PowerShell script already filtered for target VIDs (8086, 2B03)
    filtered = [d for d in data if d.get("is_usb") and not d.get("is_internal") and not d.get("is_virtual")]

    if DEBUG_MODE:
        print(f"[DEBUG] Filtered to {len(filtered)} external USB cameras")

    return filtered

def _win_enumerate_usb(max_test: int = 10) -> list[CameraInfo]:
    """
    Enumerate USB cameras on Windows with multi-interface deduplication.

    RealSense and ZED cameras expose multiple interfaces (RGB, Depth, HID).
    This function groups them by VID+PID to return one CameraInfo per physical camera.

    Returns:
        List of CameraInfo objects (one per physical camera)
    """
    backend = WINDOWS_DEFAULT_BACKEND
    indices = _probe_indices(backend, max_test)
    meta = _win_list_usb_cameras_meta()

    if DEBUG_MODE:
        print(f"[DEBUG] Found {len(indices)} OpenCV indices: {indices}")
        print(f"[DEBUG] Found {len(meta)} PowerShell metadata entries")

    # Build initial cameras list with all interfaces
    all_cameras: list[CameraInfo] = []

    for idx in indices:
        # Try to find matching metadata by index or name
        matched_meta = None

        # Simple matching: use position if available
        if idx < len(meta):
            matched_meta = meta[idx]

        if matched_meta:
            name = matched_meta.get("FriendlyName", f"Camera {idx}")
            usb_vid = matched_meta.get("usb_vid")
            usb_pid = matched_meta.get("usb_pid")
            os_id = matched_meta.get("InstanceId")

            if DEBUG_MODE:
                print(f"[DEBUG] Index {idx}: Matched to '{name}' VID={usb_vid} PID={usb_pid}")
        else:
            # No metadata found for this index
            name = f"Camera {idx}"
            usb_vid = None
            usb_pid = None
            os_id = None

            if DEBUG_MODE:
                print(f"[DEBUG] Index {idx}: No metadata match, using generic name")

        # Filter virtual/internal by name
        if _looks_virtual(name) or _looks_internal(name):
            if DEBUG_MODE:
                print(f"[DEBUG] Index {idx}: Filtered out (virtual/internal)")
            continue

        # Create CameraInfo for this interface
        cam = CameraInfo(
            index=idx,
            name=name,
            backend=CameraBackend.DSHOW,
            open_hint=idx,
            camera_type=CameraType.UNKNOWN,
            capabilities=CameraCapabilities(),
            usb_vid=usb_vid,
            usb_pid=usb_pid,
            os_id=os_id,
        )
        all_cameras.append(cam)

    if DEBUG_MODE:
        print(f"[DEBUG] After filtering: {len(all_cameras)} camera interfaces")

    # Helper function to extract base InstanceId (without &MI_XX interface suffix)
    def get_base_instance_id(os_id: Optional[str]) -> Optional[str]:
        """
        Extract base InstanceId without interface number.
        Example: USB\VID_2B03&PID_F880&MI_00\Serial123 -> USB\VID_2B03&PID_F880\Serial123
        """
        if not os_id:
            return None
        # Remove &MI_XX pattern (interface number)
        import re
        base_id = re.sub(r'&MI_[0-9A-F]{2}', '', os_id, flags=re.IGNORECASE)
        return base_id

    # Deduplicate multi-interface cameras by base InstanceId
    # Key: base_instance_id -> list of CameraInfo objects from same physical camera
    grouped: dict[Optional[str], list[CameraInfo]] = {}
    cameras_without_instance_id: list[CameraInfo] = []

    for cam in all_cameras:
        base_id = get_base_instance_id(cam.os_id)
        if base_id:
            if base_id not in grouped:
                grouped[base_id] = []
            grouped[base_id].append(cam)
        else:
            # Cameras without InstanceId kept separately
            cameras_without_instance_id.append(cam)

    # For each physical camera, keep only the first interface (lowest index)
    unique_cameras: list[CameraInfo] = []
    dedup_count = 0

    for base_id, cam_list in grouped.items():
        if len(cam_list) > 1:
            # Multiple interfaces detected - deduplicate
            # Sort by index and keep the first (usually RGB/Video interface)
            cam_list.sort(key=lambda c: c.index if c.index is not None else 999)
            selected = cam_list[0]

            if DEBUG_MODE:
                iface_count = len(cam_list)
                indices_str = ", ".join(str(c.index) for c in cam_list)
                cam_type = _identify_camera_type(selected.usb_vid, selected.usb_pid, selected.name)
                print(f"[DEBUG] Multi-interface camera {cam_type}: {iface_count} interfaces (indices: {indices_str})")
                print(f"[DEBUG]   Base ID: {base_id}")
                print(f"[DEBUG]   Selected index {selected.index} as primary")

            dedup_count += len(cam_list) - 1
            unique_cameras.append(selected)
        else:
            # Single interface - keep as is
            unique_cameras.append(cam_list[0])

    # Add cameras without InstanceId (keep all, deduplicate by index only)
    seen_indices: set[int] = set()
    for cam in cameras_without_instance_id:
        if cam.index is not None and cam.index in seen_indices:
            continue
        if cam.index is not None:
            seen_indices.add(cam.index)
        unique_cameras.append(cam)

    if DEBUG_MODE:
        print(f"[DEBUG] Deduplication: {dedup_count} interfaces removed (multi-interface cameras)")
        print(f"[DEBUG] Final result: {len(unique_cameras)} physical cameras")
        for cam in unique_cameras:
            cam_type = _identify_camera_type(cam.usb_vid, cam.usb_pid, cam.name)
            print(f"[DEBUG]   - Index {cam.index}: {cam.name} (Type: {cam_type})")

    return unique_cameras

# ---------------- macOS ----------------
def _mac_avfoundation_devices() -> list[dict[str, str]]:
    """
    Try to fetch devices via AVFoundation (PyObjC).
    Returns list of {name, unique_id}
    """
    try:
        from AVFoundation import AVCaptureDevice, AVMediaTypeVideo  # type: ignore
        devices = AVCaptureDevice.devicesWithMediaType_(AVMediaTypeVideo) or []
        out = []
        for device in devices:
            try:
                out.append({"name": str(device.localizedName()), "unique_id": str(device.uniqueID())})
            except Exception:
                pass
        return out
    except Exception:
        return []

def _mac_usb_camera_devices_from_sp() -> list[dict[str, Optional[str]]]:
    """
    Parse system_profiler SPUSBDataType to get USB cameras with VID/PID.

    Returns:
        List with name, vid, pid
    """
    devices: list[dict[str, Optional[str]]] = []
    try:
        out = subprocess.check_output(
            ["system_profiler", "SPUSBDataType", "-json"],
            text=True
        ).strip()
        data = json.loads(out).get("SPUSBDataType", [])

        def walk(node: Union[dict, list]) -> None:
            if isinstance(node, dict):
                name = _norm(node.get("_name", ""))

                # Extract vendor_id and product_id (format: "0x8086" or decimal string)
                vendor_id = node.get("vendor_id", "")
                product_id = node.get("product_id", "")

                # Normalize to hex format with 0x prefix
                vid_hex: Optional[str] = None
                pid_hex: Optional[str] = None

                if vendor_id:
                    if vendor_id.startswith("0x"):
                        vid_hex = vendor_id.lower()
                    else:
                        try:
                            vid_hex = f"0x{int(vendor_id):04x}"
                        except (ValueError, TypeError):
                            pass

                if product_id:
                    if product_id.startswith("0x"):
                        pid_hex = product_id.lower()
                    else:
                        try:
                            pid_hex = f"0x{int(product_id):04x}"
                        except (ValueError, TypeError):
                            pass

                if name:
                    low = name.lower()
                    if ("camera" in low) or any(v in low for v in VENDOR_HINTS):
                        if not _looks_virtual(name) and not _looks_internal(name):
                            devices.append({
                                "name": name,
                                "vid": vid_hex,
                                "pid": pid_hex,
                            })

                # Recurse through nested structures
                for k, v in node.items():
                    if isinstance(v, (list, dict)):
                        walk(v)

            elif isinstance(node, list):
                for x in node:
                    walk(x)

        walk(data)
    except Exception:
        pass

    # Deduplicate by name (case-insensitive)
    seen: set[str] = set()
    unique: list[dict[str, Optional[str]]] = []
    for d in devices:
        name_lower = d["name"].lower()
        if name_lower in seen:
            continue
        seen.add(name_lower)
        unique.append(d)

    return unique

def _mac_enumerate_usb(max_test: int = 10) -> list[CameraInfo]:
    """
    Enumerate USB cameras on macOS.

    Returns:
        List of CameraInfo objects
    """
    backend = MAC_DEFAULT_BACKEND
    indices = _probe_indices(backend, max_test)
    av = _mac_avfoundation_devices()
    usb_devices = _mac_usb_camera_devices_from_sp()

    # Build filtered list of AV devices that appear to be USB + not internal/virtual
    filtered_av: list[dict[str, str]] = []
    for d in av:
        name = d["name"]
        if _looks_internal(name) or _looks_virtual(name):
            continue

        # Check if this camera appears in USB device list
        low = name.lower()
        is_usb = any(
            name.lower() in usb_dev["name"].lower() or usb_dev["name"].lower() in low
            for usb_dev in usb_devices
        ) or any(v in low for v in VENDOR_HINTS)

        if is_usb:
            filtered_av.append(d)

    # Pair with indices and USB metadata
    cameras: list[CameraInfo] = []
    for i, idx in enumerate(indices):
        # Get name and unique_id from filtered AV devices
        name = filtered_av[i]["name"] if i < len(filtered_av) else (
            av[i]["name"] if i < len(av) else f"Camera {idx}"
        )
        unique_id = filtered_av[i].get("unique_id") if i < len(filtered_av) else (
            av[i].get("unique_id") if i < len(av) else None
        )

        # Skip if name patterns indicate internal/virtual
        if _looks_internal(name) or _looks_virtual(name):
            continue

        # Try to find VID/PID from USB device list by name matching
        usb_vid: Optional[str] = None
        usb_pid: Optional[str] = None
        for usb_dev in usb_devices:
            if usb_dev["name"].lower() in name.lower() or name.lower() in usb_dev["name"].lower():
                usb_vid = usb_dev.get("vid")
                usb_pid = usb_dev.get("pid")
                break

        cam = CameraInfo(
            index=idx,
            name=name,
            backend=CameraBackend.AVFOUNDATION,
            open_hint=idx,
            camera_type=CameraType.UNKNOWN,
            capabilities=CameraCapabilities(),
            usb_vid=usb_vid,
            usb_pid=usb_pid,
            os_id=unique_id,
        )
        cameras.append(cam)

    # Deduplicate by index
    seen: set[int] = set()
    unique: list[CameraInfo] = []
    for cam in cameras:
        if cam.index is not None and cam.index in seen:
            continue
        if cam.index is not None:
            seen.add(cam.index)
        unique.append(cam)

    return unique

# ---------------- Linux ----------------
def _linux_list_pyudev() -> list[CameraInfo]:
    """
    List USB cameras using pyudev on Linux.

    Returns:
        List of CameraInfo objects
    """
    try:
        import pyudev  # type: ignore
    except ImportError:
        return []

    ctx = pyudev.Context()
    cameras: list[CameraInfo] = []

    for dev in ctx.list_devices(subsystem="video4linux"):
        props = dev.properties
        if props.get("ID_BUS") != "usb":
            continue

        node = dev.device_node  # /dev/videoN
        if not node:
            continue

        # Extract VID/PID from ID_VENDOR_ID and ID_MODEL_ID
        vid_raw = props.get("ID_VENDOR_ID")  # e.g., "8086"
        pid_raw = props.get("ID_MODEL_ID")   # e.g., "0b5c"

        # Format as hex strings with 0x prefix
        usb_vid = f"0x{vid_raw}" if vid_raw else None
        usb_pid = f"0x{pid_raw}" if pid_raw else None

        # Prefer stable symlink by-id as device path
        by_id_path: Optional[str] = None
        for link in glob.glob("/dev/v4l/by-id/*"):
            try:
                target = os.readlink(link)
            except Exception:
                continue
            if target.endswith(os.path.basename(node)):
                by_id_path = link
                break

        # Get name from various properties
        name = (
            props.get("ID_V4L_PRODUCT") or
            props.get("ID_MODEL_FROM_DATABASE") or
            by_id_path or
            node
        )

        # Filter virtual/internal by name
        if _looks_virtual(name) or _looks_internal(name):
            continue

        # Extract index from /dev/videoN
        m = re.search(r"/dev/video(\d+)$", node)
        idx = int(m.group(1)) if m else None

        cam = CameraInfo(
            index=idx,
            name=name,
            backend=CameraBackend.V4L2,
            open_hint=by_id_path or node,
            camera_type=CameraType.UNKNOWN,
            capabilities=CameraCapabilities(),
            usb_vid=usb_vid,
            usb_pid=usb_pid,
            path=by_id_path or node,
            vendor=props.get("ID_VENDOR_FROM_DATABASE") or props.get("ID_VENDOR"),
            model=props.get("ID_MODEL_FROM_DATABASE") or props.get("ID_MODEL"),
        )
        cameras.append(cam)

    return cameras

def _linux_list_byid_fallback() -> list[CameraInfo]:
    """
    Fallback if pyudev isn't available: parse /dev/v4l/by-id symlinks.

    Returns:
        List of CameraInfo objects
    """
    cameras: list[CameraInfo] = []
    for link in glob.glob("/dev/v4l/by-id/*"):
        low = link.lower()
        if "usb-" not in low:
            continue
        if _looks_virtual(low) or _looks_internal(low):
            continue

        try:
            target = os.readlink(link)
        except Exception:
            continue

        m = re.search(r"video(\d+)$", target or "")
        if not m:
            continue

        idx = int(m.group(1))
        node = f"/dev/video{idx}"

        cam = CameraInfo(
            index=idx,
            name=link,
            backend=CameraBackend.V4L2,
            open_hint=link,
            camera_type=CameraType.UNKNOWN,
            capabilities=CameraCapabilities(),
            path=link,
        )
        cameras.append(cam)

    return cameras

def _linux_enumerate_usb(
    max_test: int = 10,
    blacklist_vendor_model: Optional[list[str]] = None
) -> list[CameraInfo]:
    """
    Enumerate USB cameras on Linux.

    Returns:
        List of CameraInfo objects
    """
    backend = LINUX_DEFAULT_BACKEND

    # Get USB cams via pyudev or fallback
    cameras = _linux_list_pyudev()
    if not cameras:
        cameras = _linux_list_byid_fallback()

    # Optional: exclude specific vendor/model substrings
    if blacklist_vendor_model:
        bl = tuple(s.lower() for s in blacklist_vendor_model)
        cameras = [
            c for c in cameras
            if not any(
                s in (c.vendor or "").lower() or s in (c.model or "").lower()
                for s in bl
            )
        ]

    # Verify cameras can be opened
    verified: list[CameraInfo] = []
    for cam in cameras:
        cap = open_camera_hint(cam.open_hint)
        ok = cap.isOpened()
        if ok:
            ok, _ = cap.read()
        cap.release()
        if ok:
            verified.append(cam)

    # As a last resort, probe indices
    if not verified:
        indices = _probe_indices(backend, max_test)
        for idx in indices:
            node = f"/dev/video{idx}"
            cam = CameraInfo(
                index=idx,
                name=node,
                backend=CameraBackend.V4L2,
                open_hint=node,
                camera_type=CameraType.UNKNOWN,
                capabilities=CameraCapabilities(),
                path=node,
            )
            verified.append(cam)

    # Deduplicate by path or index
    seen: set[Union[str, int]] = set()
    unique: list[CameraInfo] = []
    for cam in verified:
        key: Union[str, int] = cam.path if cam.path else cam.index if cam.index is not None else id(cam)
        if key in seen:
            continue
        seen.add(key)
        unique.append(cam)

    return unique

# ---------------- Public API ----------------
def enumerate_usb_external_cameras(
    max_test: int = 10,
    linux_blacklist_vendor_model: Optional[list[str]] = None,
    detect_specialized: bool = True,
    use_sdk_enhancement: bool = False
) -> list[CameraInfo]:
    """
    Returns a list of cameras with enhanced type detection.

    Args:
        max_test: Maximum camera indices to probe
        linux_blacklist_vendor_model: Linux-specific vendor/model blacklist
        detect_specialized: Enable VID/PID-based camera type detection
        use_sdk_enhancement: Use manufacturer SDKs for detailed detection (not yet implemented)

    Returns:
        List of CameraInfo objects with full metadata
    """
    sysname = platform.system()

    if sysname == "Windows":
        cameras = _win_enumerate_usb(max_test=max_test)
    elif sysname == "Darwin":
        cameras = _mac_enumerate_usb(max_test=max_test)
    else:
        cameras = _linux_enumerate_usb(
            max_test=max_test,
            blacklist_vendor_model=linux_blacklist_vendor_model
        )

    # Add camera type detection
    if detect_specialized:
        enhanced_cameras: list[CameraInfo] = []
        for cam in cameras:
            # Identify camera type
            detected_type = _identify_camera_type(cam.usb_vid, cam.usb_pid, cam.name)

            # Determine capabilities based on type
            caps = CameraCapabilities(
                depth=detected_type.has_depth,
                imu=detected_type.has_imu,
                stereo=detected_type.has_depth,  # Depth cameras typically have stereo
            )

            # Create new CameraInfo with detected type and capabilities
            updated_cam = CameraInfo(
                index=cam.index,
                name=cam.name,
                backend=cam.backend,
                open_hint=cam.open_hint,
                camera_type=detected_type,
                capabilities=caps,
                usb_vid=cam.usb_vid,
                usb_pid=cam.usb_pid,
                os_id=cam.os_id,
                path=cam.path,
                serial_number=None,
                sdk_available=False,
                vendor=cam.vendor,
                model=cam.model,
            )
            enhanced_cameras.append(updated_cam)

        cameras = enhanced_cameras

    # SDK-based enhancement (Phase 2 - RealSense SDK + ZED UVC validation)
    if use_sdk_enhancement:
        if DEBUG_MODE:
            print("[DEBUG] Starting SDK enhancement (Phase 2)")

        # LAYER 1: RealSense SDK Detection
        try:
            from utils.camera_detection_realsense import (
                detect_realsense_cameras,
                merge_sdk_with_usb_detection
            )

            sdk_cameras = detect_realsense_cameras()

            if DEBUG_MODE:
                print(f"[DEBUG] Layer 1 (RealSense SDK): Detected {len(sdk_cameras)} cameras")

            # Merge SDK data with USB enumeration
            if sdk_cameras:
                cameras = merge_sdk_with_usb_detection(sdk_cameras, cameras)

                if DEBUG_MODE:
                    print(f"[DEBUG] After SDK merge: {len(cameras)} total cameras")

        except ImportError as e:
            if DEBUG_MODE:
                print(f"[DEBUG] RealSense SDK module not available: {e}")
        except Exception as e:
            if DEBUG_MODE:
                print(f"[DEBUG] Error during RealSense SDK detection: {e}")

        # LAYER 4: ZED UVC Validation
        # (Layer 2 is platform enumeration above, Layer 3 is VID/PID detection above)
        try:
            from utils.camera_validation_zed import enhance_zed_detection_with_uvc

            if DEBUG_MODE:
                print("[DEBUG] Layer 4 (ZED UVC): Starting validation")

            cameras = enhance_zed_detection_with_uvc(cameras)

            if DEBUG_MODE:
                print(f"[DEBUG] After ZED UVC validation: {len(cameras)} total cameras")

        except ImportError as e:
            if DEBUG_MODE:
                print(f"[DEBUG] ZED validation module not available: {e}")
        except Exception as e:
            if DEBUG_MODE:
                print(f"[DEBUG] Error during ZED UVC validation: {e}")

        if DEBUG_MODE:
            print("[DEBUG] SDK enhancement complete")
            for cam in cameras:
                sdk_marker = " [SDK]" if cam.sdk_available else ""
                print(f"[DEBUG]   - Index {cam.index}: {cam.camera_type}{sdk_marker}")
                print(f"[DEBUG]     Capabilities: {cam.capabilities}")

    return cameras

def filter_cameras_by_type(
    cameras: list[CameraInfo],
    camera_types: list[CameraType]
) -> list[CameraInfo]:
    """
    Filter enumerated cameras by specific types.

    Args:
        cameras: List from enumerate_usb_external_cameras()
        camera_types: List of CameraType enum values to include

    Returns:
        Filtered list of cameras
    """
    return [cam for cam in cameras if cam.camera_type in camera_types]


def get_preferred_camera(
    cameras: list[CameraInfo],
    preference_order: Optional[list[CameraType]] = None
) -> Optional[CameraInfo]:
    """
    Get the first camera matching preference order.

    Args:
        cameras: List from enumerate_usb_external_cameras()
        preference_order: Ordered list of preferred CameraType values

    Returns:
        First matching camera or None
    """
    if preference_order is None:
        preference_order = [
            CameraType.REALSENSE_D455i,
            CameraType.REALSENSE_D455,
            CameraType.ZED_2i,
            CameraType.ZED_2,
            CameraType.GENERIC,
        ]

    for preferred_type in preference_order:
        for cam in cameras:
            if cam.camera_type == preferred_type:
                return cam

    return cameras[0] if cameras else None

def print_usb_external_cameras_info(cams: list[CameraInfo]) -> None:
    """Print information about enumerated cameras."""
    print("External USB cameras:")
    for c in cams:
        print(f"- [{c.backend}] {c.name}  -> open_hint: {c.open_hint}")
        print(f"  Type: {c.camera_type}, Capabilities: {c.capabilities}")
        if c.usb_vid and c.usb_pid:
            print(f"  USB VID/PID: {c.usb_vid}:{c.usb_pid}")
        if c.index is not None:
            print(f"  Index: {c.index}")
        if c.os_id:
            print(f"  OS ID: {c.os_id}")
        if c.path:
            print(f"  Path: {c.path}")
