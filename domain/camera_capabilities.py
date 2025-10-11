"""Camera capabilities data structure."""
from dataclasses import dataclass


@dataclass(frozen=True)
class CameraCapabilities:
    """
    Represents hardware capabilities of a camera.

    Attributes:
        depth: Whether camera supports depth sensing (stereo/ToF)
        imu: Whether camera includes IMU (gyroscope/accelerometer)
        stereo: Whether camera has stereo vision (dual cameras)
    """
    depth: bool = False
    imu: bool = False
    stereo: bool = False

    def __str__(self) -> str:
        """Return human-readable capability description."""
        caps = []
        if self.depth:
            caps.append("depth")
        if self.stereo:
            caps.append("stereo")
        if self.imu:
            caps.append("IMU")
        return f"[{', '.join(caps)}]" if caps else "[no special capabilities]"
