"""Camera type enumeration for specialized camera detection."""
from enum import Enum


class CameraType(Enum):
    """
    Enumeration of supported camera types.

    Used to identify specific camera models during enumeration.
    """
    REALSENSE_D455i = "realsense_d455i"
    REALSENSE_D455 = "realsense_d455"
    ZED_2i = "zed_2i"
    ZED_2 = "zed_2"
    GENERIC = "generic"
    UNKNOWN = "unknown"

    def __str__(self) -> str:
        """Return human-readable camera type name."""
        return self.value.replace('_', ' ').title()

    @property
    def is_realsense(self) -> bool:
        """Check if camera is any RealSense model."""
        return self in (CameraType.REALSENSE_D455, CameraType.REALSENSE_D455i)

    @property
    def is_zed(self) -> bool:
        """Check if camera is any Zed model."""
        return self in (CameraType.ZED_2, CameraType.ZED_2i)

    @property
    def has_imu(self) -> bool:
        """Check if this camera type typically includes IMU."""
        return self in (CameraType.REALSENSE_D455i, CameraType.ZED_2i)

    @property
    def has_depth(self) -> bool:
        """Check if this camera type supports depth sensing."""
        return self.is_realsense or self.is_zed
