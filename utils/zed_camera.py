"""Stereolabs Zed 2i camera wrapper for synchronized recording without GPU."""
import time
import threading
from pathlib import Path
from datetime import datetime
from typing import Optional
import numpy as np
import cv2
import json
from domain.camera_info import CameraInfo
from config import DEBUG_MODE, BASE_DIR, FPS, REQ_WIDTH, REQ_HEIGHT, VIDEO_SECS, START_RECORDING_DELAY


class ZedCameraRecorder:
    """
    Wrapper for Stereolabs Zed 2i camera without NVIDIA GPU.

    Handles:
    - Dual initialization strategy (Zed SDK first, OpenCV fallback)
    - Side-by-side stereo RGB recording (3840x1080)
    - Left and right camera stream separation
    - Optional IMU data recording via SDK

    Depth Processing:
    - This class ONLY captures stereo pairs (rgb_left, rgb_right, stereo_sbs)
    - For depth processing, use offline tools after capture:
      * Phase 1: ZedOfflineDepthProcessor (process all frames)
      * Phase 2: SelectiveDepthProcessor (process only frames with animals - RECOMMENDED)
    - See: utils/zed_phase_1_2_examples.py for complete workflow examples
    """

    def __init__(
        self,
        camera_info: CameraInfo,
        output_dir: Path = BASE_DIR,
        recording_duration: int = VIDEO_SECS,
        fps: int = FPS,
        width: int = REQ_WIDTH,
        height: int = REQ_HEIGHT,
        recording_delay: int = START_RECORDING_DELAY,
        enable_timestamp_logging: bool = True,
        save_metadata: bool = True,
    ):
        """
        Initialize Zed camera recorder.

        Args:
            camera_info: Camera metadata from enumeration
            output_dir: Directory to save recordings
            recording_duration: Duration in seconds
            fps: Target frame rate
            width: Stereo frame width (3840 for side-by-side)
            height: Frame height (1080)
            recording_delay: Delay before starting recording (seconds)
            enable_timestamp_logging: Enable detailed timestamp logging to CSV
            save_metadata: Save recording session metadata to JSON

        Note:
            This class only captures stereo pairs (rgb_left, rgb_right, stereo_sbs).
            For depth processing, use offline tools after capture:
            - Phase 1: ZedOfflineDepthProcessor (all frames)
            - Phase 2: SelectiveDepthProcessor (only frames with animals - RECOMMENDED)
        """
        self.camera_info = camera_info
        self.output_dir = Path(output_dir)
        self.recording_duration = recording_duration
        self.fps = fps
        self.width = width
        self.height = height
        self.single_width = width // 2  # Width of single camera (1920)
        self.recording_delay = recording_delay
        self.enable_timestamp_logging = enable_timestamp_logging
        self.save_metadata = save_metadata

        self.camera = None
        self.using_sdk = False
        self.is_recording = False
        self.recording_thread: Optional[threading.Thread] = None
        self.start_timestamp: Optional[float] = None
        self.frame_count = 0

        # Video writers for each stream
        self.left_writer = None
        self.right_writer = None
        self.stereo_writer = None

        # IMU data file (SDK only)
        self.imu_file = None

        # Timestamp log file
        self.timestamp_log_file = None

        # Session metadata
        self.session_metadata = {
            'start_time': None,
            'end_time': None,
            'total_frames': 0,
            'total_duration': 0,
            'settings': {},
        }

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def initialize(self) -> bool:
        """
        Initialize Zed camera with dual strategy.

        Try Zed SDK first (CPU mode), fallback to OpenCV if SDK fails.

        Returns:
            True if initialization successful, False otherwise
        """
        # Try SDK initialization first
        if self._initialize_with_sdk():
            self.using_sdk = True
            return True

        # Fallback to OpenCV
        if self._initialize_with_opencv():
            self.using_sdk = False
            return True

        return False

    def _initialize_with_sdk(self) -> bool:
        """
        Initialize Zed camera using Zed SDK (CPU mode).

        Returns:
            True if SDK initialization successful, False otherwise
        """
        try:
            import pyzed.sl as sl

            # Create camera object
            self.camera = sl.Camera()

            # Configure initialization parameters for CPU mode
            init_params = sl.InitParameters()
            init_params.camera_resolution = sl.RESOLUTION.HD1080
            init_params.camera_fps = self.fps
            init_params.coordinate_units = sl.UNIT.METER
            init_params.depth_mode = sl.DEPTH_MODE.NONE  # CRITICAL: No GPU depth

            # Set camera by serial number if available
            if self.camera_info.serial_number:
                try:
                    serial_int = int(self.camera_info.serial_number)
                    init_params.set_from_serial_number(serial_int)
                except (ValueError, AttributeError):
                    pass  # Use default camera if serial conversion fails

            # Open camera
            err = self.camera.open(init_params)

            if err != sl.ERROR_CODE.SUCCESS:
                if DEBUG_MODE:
                    print(f"Zed SDK initialization failed: {err}")
                return False

            # Disable GPU-dependent features
            self.camera.disable_positional_tracking()
            self.camera.disable_spatial_mapping()

            if DEBUG_MODE:
                print(f"Zed camera initialized with SDK (CPU mode): {self.camera_info.name}")

            return True

        except ImportError:
            if DEBUG_MODE:
                print("Zed SDK not available")
            return False
        except Exception as e:
            if DEBUG_MODE:
                print(f"Failed to initialize Zed with SDK: {e}")
            return False

    def _initialize_with_opencv(self) -> bool:
        """
        Initialize Zed camera using OpenCV fallback.

        Returns:
            True if OpenCV initialization successful, False otherwise
        """
        try:
            # Get camera index from OS ID or use default
            cam_index = 0
            if hasattr(self.camera_info, 'os_id') and self.camera_info.os_id:
                # Try to extract index from OS ID
                try:
                    cam_index = int(self.camera_info.os_id)
                except (ValueError, TypeError):
                    pass

            # Create VideoCapture with DirectShow backend on Windows
            self.camera = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)

            if not self.camera.isOpened():
                self.camera = cv2.VideoCapture(cam_index, cv2.CAP_MSMF)
                if not self.camera.isOpened():
                    if DEBUG_MODE:
                        print(f"Failed to open camera at index {cam_index}")
                    return False

            # Configure resolution and FPS
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.camera.set(cv2.CAP_PROP_FPS, self.fps)

            # Verify resolution was set correctly
            actual_width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # Update instance width/height to match actual camera output
            # DirectShow may not honor resolution requests
            if actual_width != self.width or actual_height != self.height:
                if DEBUG_MODE:
                    print(f"Warning: Camera returned {actual_width}x{actual_height}, adjusting from requested {self.width}x{self.height}")

                self.width = actual_width
                self.height = actual_height
                self.single_width = actual_width // 2  # Update single camera width

            if DEBUG_MODE:
                print(f"Zed camera initialized with OpenCV: {self.camera_info.name}")
                print(f"Zed camera resolution: {actual_width}x{actual_height}")

            return True

        except Exception as e:
            if DEBUG_MODE:
                print(f"Failed to initialize Zed with OpenCV: {e}")
            return False

    def prepare_for_recording(self, sync_timestamp: float) -> Optional[threading.Thread]:
        """
        Prepare for recording but don't start the thread yet.

        This allows the orchestrator to start all threads simultaneously
        for better synchronization. All slow operations (file creation,
        video writer initialization) are performed here, so thread.start()
        can be called rapidly for all cameras.

        Args:
            sync_timestamp: Synchronized start timestamp from orchestrator

        Returns:
            Thread object ready to start, or None if preparation failed
        """
        if self.is_recording:
            return None

        try:
            # Set recording state
            self.start_timestamp = sync_timestamp
            self.is_recording = True
            self.frame_count = 0

            # SLOW OPERATION: Create video writers
            self._create_video_writers()

            # SLOW OPERATION: Create IMU data file if using SDK
            if self.using_sdk and self.camera_info.capabilities.imu:
                self._create_imu_file()

            # SLOW OPERATION: Create timestamp log file if enabled
            if self.enable_timestamp_logging:
                self._create_timestamp_log_file()

            # Initialize session metadata
            self.session_metadata['start_time'] = datetime.fromtimestamp(sync_timestamp).isoformat()

            # Create thread but DON'T start it
            self.recording_thread = threading.Thread(
                target=self._recording_loop,
                name=f"Zed-{self.camera_info.index}",
                daemon=True
            )

            # Return thread for orchestrator to start later
            return self.recording_thread

        except Exception as e:
            if DEBUG_MODE:
                print(f"Error preparing Zed recording: {e}")
            self.is_recording = False
            return None

    def _create_video_writers(self):
        """Create OpenCV VideoWriter instances for each stream."""
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        # Left RGB stream
        left_filename = self._generate_filename("rgb_left", "mp4")
        self.left_writer = cv2.VideoWriter(
            str(self.output_dir / left_filename),
            fourcc,
            self.fps,
            (self.single_width, self.height)
        )

        # Right RGB stream
        right_filename = self._generate_filename("rgb_right", "mp4")
        self.right_writer = cv2.VideoWriter(
            str(self.output_dir / right_filename),
            fourcc,
            self.fps,
            (self.single_width, self.height)
        )

        # Stereo side-by-side stream
        stereo_filename = self._generate_filename("stereo_sbs", "mp4")
        self.stereo_writer = cv2.VideoWriter(
            str(self.output_dir / stereo_filename),
            fourcc,
            self.fps,
            (self.width, self.height)
        )

    def _create_imu_file(self):
        """Create CSV file for IMU data (SDK only)."""
        imu_filename = self._generate_filename("imu", "csv")
        self.imu_file = open(self.output_dir / imu_filename, 'w')
        # Write CSV header
        self.imu_file.write("timestamp,frame_number,sensor,x,y,z\n")

    def _create_timestamp_log_file(self):
        """Create CSV file for timestamp logging."""
        timestamp_filename = self._generate_filename("timestamps", "csv")
        self.timestamp_log_file = open(self.output_dir / timestamp_filename, 'w')
        # Write CSV header
        self.timestamp_log_file.write(
            "frame_number,system_timestamp,rgb_left_timestamp,rgb_right_timestamp,"
            "stereo_sbs_timestamp,imu_timestamp\n"
        )

    def _generate_filename(self, sensor: str, extension: str) -> str:
        """
        Generate filename following the required format.

        Format: YYYY-MM-dd-HH-mm-ss-sss-{CAMERA_INDEX}-{SENSOR}.ext
        """
        dt = datetime.fromtimestamp(self.start_timestamp)
        timestamp_str = dt.strftime("%Y-%m-%d-%H-%M-%S-%f")[:-3]  # milliseconds
        cam_idx = self.camera_info.index if self.camera_info.index is not None else 0
        return f"{timestamp_str}-{cam_idx}-{sensor}.{extension}"

    def _recording_loop(self):
        """
        Main recording loop (runs in separate thread).

        Waits for countdown delay before starting actual recording to ensure
        all cameras begin capturing frames at the same time.
        """
        try:
            # Calculate when to actually start recording (after countdown)
            recording_start_time = self.start_timestamp + self.recording_delay
            end_time = recording_start_time + self.recording_duration

            if DEBUG_MODE:
                print(f"Zed #{self.camera_info.index}: Thread started, waiting for countdown...")

            while self.is_recording and time.time() < end_time:
                current_time = time.time()

                # Wait for countdown to complete before starting actual recording
                if current_time < recording_start_time:
                    time.sleep(0.001)  # 1ms sleep to avoid busy-waiting
                    continue

                # Countdown complete - all cameras should be recording now
                if self.frame_count == 0 and DEBUG_MODE:
                    print(f"Zed #{self.camera_info.index}: Starting frame capture")

                # Capture frame based on initialization method
                if self.using_sdk:
                    stereo_frame = self._capture_frame_sdk()
                else:
                    stereo_frame = self._capture_frame_opencv()

                if stereo_frame is None:
                    continue

                # Validate frame dimensions
                if stereo_frame.shape != (self.height, self.width, 3):
                    if DEBUG_MODE:
                        print(f"Warning: Stereo frame has incorrect dimensions: {stereo_frame.shape}, expected ({self.height}, {self.width}, 3)")
                    continue

                # Split stereo frame into left and right
                left_frame = stereo_frame[:, :self.single_width]
                right_frame = stereo_frame[:, self.single_width:]

                # Write frames (stereo pairs only - depth processing is done offline)
                self.left_writer.write(left_frame)
                self.right_writer.write(right_frame)
                self.stereo_writer.write(stereo_frame)

                # Process IMU data if using SDK
                if self.using_sdk and self.imu_file:
                    self._process_imu_data_sdk(current_time)

                # Log timestamps if enabled
                if self.timestamp_log_file:
                    self.timestamp_log_file.write(
                        f"{self.frame_count},{current_time},{current_time},"
                        f"{current_time},{current_time},{current_time if self.imu_file else ''}\n"
                    )
                    self.timestamp_log_file.flush()

                self.frame_count += 1

        except Exception as e:
            if DEBUG_MODE:
                print(f"Error in Zed recording loop: {e}")
        finally:
            self._cleanup()

    def _capture_frame_sdk(self) -> Optional[np.ndarray]:
        """
        Capture frame using Zed SDK.

        Returns:
            Stereo frame as numpy array or None if capture fails
        """
        try:
            import pyzed.sl as sl

            # Grab frame
            if self.camera.grab() != sl.ERROR_CODE.SUCCESS:
                return None

            # Retrieve side-by-side image
            image = sl.Mat()
            self.camera.retrieve_image(image, sl.VIEW.SIDE_BY_SIDE)

            # Convert to numpy array
            frame = image.get_data()

            # Convert RGBA to BGR if necessary
            if frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)

            return frame

        except Exception as e:
            if DEBUG_MODE:
                print(f"Error capturing frame with SDK: {e}")
            return None

    def _capture_frame_opencv(self) -> Optional[np.ndarray]:
        """
        Capture frame using OpenCV.

        Returns:
            Stereo frame as numpy array or None if capture fails
        """
        try:
            ret, frame = self.camera.read()

            if not ret or frame is None:
                return None

            return frame

        except Exception as e:
            if DEBUG_MODE:
                print(f"Error capturing frame with OpenCV: {e}")
            return None

    def _process_imu_data_sdk(self, timestamp: float):
        """
        Process and save IMU data via SDK (if available).

        Args:
            timestamp: Current timestamp
        """
        try:
            import pyzed.sl as sl

            # Get sensor data
            sensors_data = sl.SensorsData()
            if self.camera.get_sensors_data(sensors_data, sl.TIME_REFERENCE.CURRENT) == sl.ERROR_CODE.SUCCESS:
                # Get IMU data
                imu_data = sensors_data.get_imu_data()

                # Write accelerometer data
                accel = imu_data.get_linear_acceleration()
                self.imu_file.write(
                    f"{timestamp},{self.frame_count},accel,"
                    f"{accel[0]},{accel[1]},{accel[2]}\n"
                )

                # Write gyroscope data
                gyro = imu_data.get_angular_velocity()
                self.imu_file.write(
                    f"{timestamp},{self.frame_count},gyro,"
                    f"{gyro[0]},{gyro[1]},{gyro[2]}\n"
                )

        except Exception as e:
            if DEBUG_MODE:
                print(f"Error processing IMU data: {e}")

    def stop_recording(self):
        """Stop recording and cleanup resources."""
        self.is_recording = False
        if self.recording_thread:
            self.recording_thread.join(timeout=5.0)

    def _save_session_metadata(self):
        """Save session metadata to JSON file."""
        if not self.save_metadata or not self.start_timestamp:
            return

        try:
            # Update session metadata with final values
            self.session_metadata['end_time'] = datetime.now().isoformat()
            self.session_metadata['total_frames'] = self.frame_count

            # Calculate total duration
            if self.start_timestamp:
                end_timestamp = time.time()
                self.session_metadata['total_duration'] = end_timestamp - self.start_timestamp

            # Add settings
            self.session_metadata['settings'] = {
                'fps': self.fps,
                'width': self.width,
                'height': self.height,
                'recording_duration': self.recording_duration,
                'recording_delay': self.recording_delay,
                'using_sdk': self.using_sdk,
            }

            # Add camera info
            self.session_metadata['camera_info'] = {
                'name': self.camera_info.name,
                'index': self.camera_info.index,
                'vendor': self.camera_info.vendor,
                'serial_number': getattr(self.camera_info, 'serial_number', None),
            }

            # Save to JSON file
            metadata_filename = self._generate_filename("session_metadata", "json")
            metadata_path = self.output_dir / metadata_filename

            with open(metadata_path, 'w') as f:
                json.dump(self.session_metadata, f, indent=2)

            if DEBUG_MODE:
                print(f"Saved session metadata to {metadata_path}")

        except Exception as e:
            if DEBUG_MODE:
                print(f"Error saving session metadata: {e}")

    def _cleanup(self):
        """Release all resources."""
        # Save session metadata before closing files
        self._save_session_metadata()

        # Release video writers
        if self.left_writer:
            self.left_writer.release()
        if self.right_writer:
            self.right_writer.release()
        if self.stereo_writer:
            self.stereo_writer.release()

        # Close timestamp log file
        if self.timestamp_log_file:
            self.timestamp_log_file.close()

        # Close IMU file
        if self.imu_file:
            self.imu_file.close()

        # Release camera
        if self.camera:
            if self.using_sdk:
                try:
                    self.camera.close()
                except:
                    pass
            else:
                self.camera.release()

    def __del__(self):
        """Destructor to ensure cleanup."""
        self._cleanup()
