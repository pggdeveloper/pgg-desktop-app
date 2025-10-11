"""Recording session data model."""
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional
from domain.camera_info import CameraInfo
from config import VIDEO_SECS, START_RECORDING_DELAY, BASE_DIR


@dataclass
class RecordingSession:
    """
    Represents a recording session with multiple cameras.

    Attributes:
        session_id: Unique identifier (timestamp-based)
        base_dir: Base directory for all recordings
        cameras: List of cameras participating in this session
        start_time: Actual recording start time (after delay)
        end_time: Recording end time
        duration_secs: Requested recording duration
        recording_delay: Delay before starting recording
    """
    session_id: str
    base_dir: Path
    cameras: list[CameraInfo]
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration_secs: int = VIDEO_SECS
    recording_delay: int = START_RECORDING_DELAY

    @property
    def session_dir(self) -> Path:
        """Get the directory for this session's recordings."""
        return BASE_DIR / self.session_id

    @staticmethod
    def create_session_id() -> str:
        """Create a unique session ID based on timestamp."""
        return datetime.now().strftime("%Y%m%d_%H%M%S")
