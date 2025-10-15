"""Camera backend enumeration for platform-specific video capture."""
from enum import Enum


class CameraBackend(Enum):
    """
    OpenCV backend types used for camera capture.

    Different platforms use different backends for optimal performance.
    """
    DSHOW = "DSHOW"           # Windows DirectShow
    AVFOUNDATION = "AVFOUNDATION"  # macOS AVFoundation
    V4L2 = "V4L2"             # Linux Video4Linux2
    REALSENSE_SDK = "REALSENSE_SDK"  # Intel RealSense SDK (pyrealsense2)

    def __str__(self) -> str:
        return self.value
