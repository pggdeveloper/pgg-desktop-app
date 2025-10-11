"""Camera information data structure."""
from dataclasses import dataclass
from typing import Union, Optional
from domain.camera_type import CameraType
from domain.camera_backend import CameraBackend
from domain.camera_capabilities import CameraCapabilities


@dataclass
class CameraInfo:
    """
    Complete metadata for an enumerated USB camera.

    This is the unified return type for all platform-specific enumeration
    functions (_win_enumerate_usb, _mac_enumerate_usb, _linux_enumerate_usb).

    Attributes:
        index: OpenCV camera index (may be None for path-based access)
        name: Human-readable camera name
        backend: OpenCV backend used for this camera
        open_hint: Value to pass to open_camera_hint() - either int index or str path
        camera_type: Detected camera type (RealSense, Zed, Generic, etc.)
        capabilities: Hardware capabilities (depth, IMU, stereo)
        usb_vid: USB Vendor ID in hex format (e.g., "0x8086"), None if unavailable
        usb_pid: USB Product ID in hex format (e.g., "0x0B5C"), None if unavailable
        os_id: Platform-specific device identifier (Windows InstanceId, macOS uniqueID, etc.)
        path: Device path (primarily for Linux /dev/video* or by-id symlinks)
        serial_number: Camera serial number (only available with SDK enhancement)
        sdk_available: Whether manufacturer SDK successfully detected this camera
        vendor: Vendor name from device metadata (Linux only)
        model: Model name from device metadata (Linux only)
    """
    # Required fields
    index: Optional[int]
    name: str
    backend: CameraBackend
    open_hint: Union[int, str]
    camera_type: CameraType
    capabilities: CameraCapabilities

    # USB identifiers (VID/PID)
    usb_vid: Optional[str] = None
    usb_pid: Optional[str] = None

    # Platform-specific identifiers
    os_id: Optional[str] = None
    path: Optional[str] = None

    # SDK-enhanced fields
    serial_number: Optional[str] = None
    sdk_available: bool = False

    # Additional metadata (primarily Linux)
    vendor: Optional[str] = None
    model: Optional[str] = None

    def __str__(self) -> str:
        """Return human-readable camera description."""
        parts = [f"{self.name}"]
        if self.camera_type != CameraType.UNKNOWN:
            parts.append(f"[{self.camera_type}]")
        if self.usb_vid and self.usb_pid:
            parts.append(f"({self.usb_vid}:{self.usb_pid})")
        return " ".join(parts)

    def to_dict(self) -> dict[str, Union[int, str, bool, None]]:
        """
        Convert to dictionary for backward compatibility.

        Returns dict matching the old Dict[str, Any] format.
        """
        return {
            "index": self.index,
            "name": self.name,
            "backend": self.backend.value,
            "open_hint": self.open_hint,
            "camera_type": self.camera_type,
            "capabilities": {
                "depth": self.capabilities.depth,
                "imu": self.capabilities.imu,
                "stereo": self.capabilities.stereo,
            },
            "usb_vid": self.usb_vid,
            "usb_pid": self.usb_pid,
            "os_id": self.os_id,
            "path": self.path,
            "serial_number": self.serial_number,
            "sdk_available": self.sdk_available,
            "vendor": self.vendor,
            "model": self.model,
        }
