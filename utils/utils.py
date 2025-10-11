import os, platform, json, subprocess, re, glob
from typing import Optional, Union
import cv2
from config import DEBUG_MODE
from domain.camera_type import CameraType
from domain.camera_backend import CameraBackend
from domain.camera_capabilities import CameraCapabilities
from domain.camera_info import CameraInfo

WINDOWS_DEFAULT_BACKEND = cv2.CAP_DSHOW
MAC_DEFAULT_BACKEND = cv2.CAP_AVFOUNDATION
LINUX_DEFAULT_BACKEND = cv2.CAP_V4L2

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
        vid: USB Vendor ID (e.g., "0x8086")
        pid: USB Product ID (e.g., "0x0B5C")
        name: Camera device name

    Returns:
        CameraType enum value
    """
    # Normalize VID/PID to lowercase for comparison
    vid_lower = vid.lower() if vid else ""
    pid_lower = pid.lower() if pid else ""
    name_lower = name.lower()

    # Intel RealSense Detection
    if vid_lower == "0x8086":
        # RealSense D455/D455i (same PID, need IMU check)
        if pid_lower == "0x0b5c":
            # Cannot distinguish D455 vs D455i from VID/PID alone
            # Need to check for IMU device or use SDK
            return CameraType.REALSENSE_D455  # Default to D455

        # Other RealSense models (for future expansion)
        # D435i: 0x0B3A, D435: 0x0B07, L515: 0x0B64, etc.

    # Stereolabs Zed Detection
    if vid_lower == "0x2b03":
        if pid_lower == "0xf882":
            return CameraType.ZED_2i
        elif pid_lower == "0xf780":
            return CameraType.ZED_2

    # Name-based fallback detection
    if "realsense" in name_lower:
        if "d455" in name_lower:
            return CameraType.REALSENSE_D455
        # Could add other model detection here

    if "zed" in name_lower:
        if "2i" in name_lower:
            return CameraType.ZED_2i
        return CameraType.ZED_2

    # Generic camera
    return CameraType.GENERIC

# ---------------- Windows ----------------
def _win_list_usb_cameras_meta() -> list[dict[str, Union[str, bool, None]]]:
    """
    Use PowerShell PnP to list camera/image devices with USB metadata.

    Returns:
        List of dicts with FriendlyName, InstanceId, VID, PID, etc.
    """
    ps = r"""
$devs = Get-PnpDevice -Class Camera,Image | Where-Object { $_.Status -eq 'OK' }
$result = @()
foreach ($d in $devs) {
  $id = $d.InstanceId
  $enum = (Get-PnpDeviceProperty -InstanceId $id -KeyName 'DEVPKEY_Device_EnumeratorName' -ErrorAction SilentlyContinue).Data
  $loc  = (Get-PnpDeviceProperty -InstanceId $id -KeyName 'DEVPKEY_Device_LocationInfo'   -ErrorAction SilentlyContinue).Data
  $bus  = (Get-PnpDeviceProperty -InstanceId $id -KeyName 'DEVPKEY_Device_BusReportedDeviceDesc' -ErrorAction SilentlyContinue).Data
  $hwid = (Get-PnpDeviceProperty -InstanceId $id -KeyName 'DEVPKEY_Device_HardwareIds' -ErrorAction SilentlyContinue).Data

  # Extract VID/PID from hardware ID (format: USB\VID_8086&PID_0B5C&...)
  $vid = ""
  $pid = ""
  if ($hwid -match "VID_([0-9A-F]{4})") { $vid = $matches[1] }
  if ($hwid -match "PID_([0-9A-F]{4})") { $pid = $matches[1] }

  $obj = [PSCustomObject]@{
    FriendlyName = $d.FriendlyName
    InstanceId   = $id
    Enumerator   = $enum
    Location     = $loc
    BusDesc      = $bus
    VID          = $vid
    PID          = $pid
  }
  $result += $obj
}
$result | ConvertTo-Json
"""
    try:
        out = subprocess.check_output(
            ["powershell", "-NoProfile", "-Command", ps],
            text=True,
            encoding="utf-8",
            errors="ignore"
        )
        data = json.loads(out) if out.strip() else []
        if isinstance(data, dict):
            data = [data]
    except Exception:
        data = []

    # Process and normalize data
    for d in data:
        iid = _norm(d.get("InstanceId", ""))
        vid_raw = d.get("VID", "")
        pid_raw = d.get("PID", "")

        d["usb_vid"] = f"0x{vid_raw}" if vid_raw else None
        d["usb_pid"] = f"0x{pid_raw}" if pid_raw else None
        d["is_usb"] = (d.get("Enumerator") == "USB") or iid.upper().startswith("USB\\VID_")
        d["is_internal"] = bool(_norm(d.get("Location", "")).lower().find("internal") >= 0) or _looks_internal(d.get("FriendlyName", ""))
        d["is_virtual"] = _looks_virtual(d.get("FriendlyName", ""))

    # Keep only external USB, non-virtual, non-internal
    filtered = [d for d in data if d.get("is_usb") and not d.get("is_internal") and not d.get("is_virtual")]
    return filtered

def _win_enumerate_usb(max_test: int = 10) -> list[CameraInfo]:
    """
    Enumerate USB cameras on Windows.

    Returns:
        List of CameraInfo objects
    """
    backend = WINDOWS_DEFAULT_BACKEND
    indices = _probe_indices(backend, max_test)
    meta = _win_list_usb_cameras_meta()

    cameras: list[CameraInfo] = []
    for i, idx in enumerate(indices):
        # Prefer a meta name if available
        name = meta[i]["FriendlyName"] if i < len(meta) else f"Camera {idx}"
        usb_vid = meta[i].get("usb_vid") if i < len(meta) else None
        usb_pid = meta[i].get("usb_pid") if i < len(meta) else None
        os_id = meta[i].get("InstanceId") if i < len(meta) else None

        # Create CameraInfo with default values (camera_type will be detected later)
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
        cameras.append(cam)

    # Final name-based filters (virtual/integrated patterns)
    cameras = [c for c in cameras if not _looks_virtual(c.name) and not _looks_internal(c.name)]

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

    # Optional SDK-based enhancement (Phase 3 - not implemented yet)
    if use_sdk_enhancement:
        if DEBUG_MODE:
            print("SDK enhancement not yet implemented (Phase 3)")

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
