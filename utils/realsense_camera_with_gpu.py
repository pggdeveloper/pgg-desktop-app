"""Intel RealSense camera wrapper for synchronized recording."""
import time
import threading
from pathlib import Path
from datetime import datetime
from typing import Optional
import numpy as np
import cv2
from domain.camera_info import CameraInfo
from config import DEBUG_MODE, BASE_DIR, FPS, REALSENSE_WIDTH, REALSENSE_HEIGHT, VIDEO_SECS, START_RECORDING_DELAY


class RealSenseCameraWithGpuRecorder:
    """
    Wrapper for Intel RealSense D455/D455i cameras.

    Handles:
    - RGB stream recording
    - Depth stream recording
    - Infrared stream recording (left/right)
    - Point cloud generation and saving
    - IMU data recording (D455i only)
    - Synchronized multi-stream capture
    """

    def __init__(
        self,
        camera_info: CameraInfo,
        output_dir: Path = BASE_DIR,
        recording_duration: int = VIDEO_SECS,
        fps: int = FPS,
        width: int = REALSENSE_WIDTH,
        height: int = REALSENSE_HEIGHT,
        recording_delay: int = START_RECORDING_DELAY,
    ):
        """
        Initialize RealSense camera recorder.

        Args:
            camera_info: Camera metadata from enumeration
            output_dir: Directory to save recordings
            recording_duration: Duration in seconds
            fps: Target frame rate
            width: RGB/Depth stream width
            height: RGB/Depth stream height
        """
        self.camera_info = camera_info
        self.output_dir = Path(output_dir)
        self.recording_duration = recording_duration
        self.fps = fps
        self.width = width
        self.height = height
        self.recording_delay = recording_delay

        self.pipeline = None
        self.config = None
        self.align = None
        self.is_recording = False
        self.recording_thread: Optional[threading.Thread] = None
        self.start_timestamp: Optional[float] = None
        self.frame_count = 0

        # Video writers for each stream
        self.rgb_writer = None
        self.depth_writer = None
        self.ir_left_writer = None
        self.ir_right_writer = None

        # IMU data file (D455i only)
        self.imu_file = None

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def initialize(self) -> bool:
        """
        Initialize RealSense pipeline and configure streams.

        Returns:
            True if initialization successful, False otherwise
        """
        try:
            import pyrealsense2 as rs

            # Create pipeline
            self.pipeline = rs.pipeline()
            self.config = rs.config()

            # Get device serial number from camera_info
            # This ensures we open the correct camera if multiple are connected
            if self.camera_info.serial_number:
                self.config.enable_device(self.camera_info.serial_number)
            elif self.camera_info.os_id:
                # Try to find by unique ID
                ctx = rs.context()
                devices = ctx.query_devices()
                for dev in devices:
                    if dev.get_info(rs.camera_info.serial_number) in str(self.camera_info.os_id):
                        self.config.enable_device(dev.get_info(rs.camera_info.serial_number))
                        break

            # Enable RGB stream
            self.config.enable_stream(
                rs.stream.color,
                self.width,
                self.height,
                rs.format.bgr8,
                self.fps
            )

            # Enable Depth stream
            self.config.enable_stream(
                rs.stream.depth,
                self.width,
                self.height,
                rs.format.z16,
                self.fps
            )

            # Enable Infrared streams (left and right)
            self.config.enable_stream(
                rs.stream.infrared,
                1,  # Left camera
                self.width,
                self.height,
                rs.format.y8,
                self.fps
            )

            self.config.enable_stream(
                rs.stream.infrared,
                2,  # Right camera
                self.width,
                self.height,
                rs.format.y8,
                self.fps
            )

            # Enable IMU streams if D455i
            if self.camera_info.capabilities.imu:
                try:
                    self.config.enable_stream(rs.stream.accel)
                    self.config.enable_stream(rs.stream.gyro)
                except Exception as e:
                    if DEBUG_MODE:
                        print(f"Warning: Could not enable IMU streams: {e}")

            # Start pipeline
            profile = self.pipeline.start(self.config)

            # Create align object to align depth to color
            align_to = rs.stream.color
            self.align = rs.align(align_to)

            # Warm up camera (discard first few frames)
            for _ in range(30):
                self.pipeline.wait_for_frames()

            if DEBUG_MODE:
                print(f"RealSense camera initialized: {self.camera_info.name}")

            return True

        except Exception as e:
            if DEBUG_MODE:
                print(f"Failed to initialize RealSense camera: {e}")
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

            # SLOW OPERATION: Create IMU data file if applicable
            if self.camera_info.capabilities.imu:
                self._create_imu_file()

            # Create thread but DON'T start it
            self.recording_thread = threading.Thread(
                target=self._recording_loop,
                name=f"RealSense-{self.camera_info.index}",
                daemon=True
            )

            # Return thread for orchestrator to start later
            return self.recording_thread

        except Exception as e:
            if DEBUG_MODE:
                print(f"Error preparing RealSense recording: {e}")
            self.is_recording = False
            return None

    def _create_video_writers(self):
        """Create OpenCV VideoWriter instances for each stream."""
        

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'XVID'

        # RGB stream
        rgb_filename = self._generate_filename("rgb", "mp4")
        self.rgb_writer = cv2.VideoWriter(
            str(self.output_dir / rgb_filename),
            fourcc,
            self.fps,
            (self.width, self.height)
        )

        # Depth stream (saved as grayscale visualization)
        depth_filename = self._generate_filename("depth", "mp4")
        self.depth_writer = cv2.VideoWriter(
            str(self.output_dir / depth_filename),
            fourcc,
            self.fps,
            (self.width, self.height),
            isColor=False
        )

        # Infrared left
        ir_left_filename = self._generate_filename("ir_left", "mp4")
        self.ir_left_writer = cv2.VideoWriter(
            str(self.output_dir / ir_left_filename),
            fourcc,
            self.fps,
            (self.width, self.height),
            isColor=False
        )

        # Infrared right
        ir_right_filename = self._generate_filename("ir_right", "mp4")
        self.ir_right_writer = cv2.VideoWriter(
            str(self.output_dir / ir_right_filename),
            fourcc,
            self.fps,
            (self.width, self.height),
            isColor=False
        )

    def _create_imu_file(self):
        """Create CSV file for IMU data."""
        imu_filename = self._generate_filename("imu", "csv")
        self.imu_file = open(self.output_dir / imu_filename, 'w')
        # Write CSV header
        self.imu_file.write("timestamp,frame_number,sensor,x,y,z\n")

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
        import pyrealsense2 as rs

        try:
            # Calculate when to actually start recording (after countdown)
            recording_start_time = self.start_timestamp + self.recording_delay
            end_time = recording_start_time + self.recording_duration

            if DEBUG_MODE:
                print(f"RealSense #{self.camera_info.index}: Thread started, waiting for countdown...")

            while self.is_recording and time.time() < end_time:
                current_time = time.time()

                # Wait for countdown to complete before starting actual recording
                if current_time < recording_start_time:
                    time.sleep(0.001)  # 1ms sleep to avoid busy-waiting
                    continue

                # Countdown complete - all cameras should be recording now
                if self.frame_count == 0 and DEBUG_MODE:
                    print(f"RealSense #{self.camera_info.index}: Starting frame capture")

                # Wait for frames with timeout
                frames = self.pipeline.wait_for_frames(timeout_ms=5000)

                # Align depth to color
                aligned_frames = self.align.process(frames)

                # Get individual streams
                color_frame = aligned_frames.get_color_frame()
                depth_frame = aligned_frames.get_depth_frame()
                ir_left_frame = frames.get_infrared_frame(1)
                ir_right_frame = frames.get_infrared_frame(2)

                if not color_frame or not depth_frame:
                    continue

                # Convert to numpy arrays
                color_image = np.asanyarray(color_frame.get_data())
                depth_image = np.asanyarray(depth_frame.get_data())

                # Write RGB frame
                self.rgb_writer.write(color_image)

                # Write depth frame (convert to 8-bit for visualization)
                depth_colormap = cv2.applyColorMap(
                    cv2.convertScaleAbs(depth_image, alpha=0.03),
                    cv2.COLORMAP_JET
                )
                depth_gray = cv2.cvtColor(depth_colormap, cv2.COLOR_BGR2GRAY)
                self.depth_writer.write(depth_gray)

                # Write infrared frames
                if ir_left_frame:
                    ir_left_image = np.asanyarray(ir_left_frame.get_data())
                    self.ir_left_writer.write(ir_left_image)

                if ir_right_frame:
                    ir_right_image = np.asanyarray(ir_right_frame.get_data())
                    self.ir_right_writer.write(ir_right_image)

                # Save point cloud periodically (e.g., every second)
                if self.frame_count % self.fps == 0:
                    self._save_point_cloud(depth_frame, color_frame, current_time)

                # Process IMU data if available
                if self.camera_info.capabilities.imu:
                    self._process_imu_data(frames, current_time)

                self.frame_count += 1

        except Exception as e:
            if DEBUG_MODE:
                print(f"Error in RealSense recording loop: {e}")
        finally:
            self._cleanup()

    def _save_point_cloud(self, depth_frame, color_frame, timestamp: float):
        """
        Save point cloud to PLY file.

        Args:
            depth_frame: RealSense depth frame
            color_frame: RealSense color frame
            timestamp: Current timestamp
        """
        try:
            import pyrealsense2 as rs

            # Create point cloud object
            pc = rs.pointcloud()
            points = pc.calculate(depth_frame)

            # Map color to points
            pc.map_to(color_frame)

            # Export to PLY
            ply_filename = self._generate_filename(
                f"pointcloud_{self.frame_count // self.fps}",
                "ply"
            )
            points.export_to_ply(str(self.output_dir / ply_filename), color_frame)

        except Exception as e:
            if DEBUG_MODE:
                print(f"Error saving point cloud: {e}")

    def _process_imu_data(self, frames, timestamp: float):
        """
        Process and save IMU data (accelerometer and gyroscope).

        Args:
            frames: RealSense frameset
            timestamp: Current timestamp
        """
        try:
            import pyrealsense2 as rs

            # Get motion frames
            for frame in frames:
                if frame.is_motion_frame():
                    motion = frame.as_motion_frame()

                    if motion.get_profile().stream_type() == rs.stream.accel:
                        accel_data = motion.get_motion_data()
                        self.imu_file.write(
                            f"{timestamp},{self.frame_count},accel,"
                            f"{accel_data.x},{accel_data.y},{accel_data.z}\n"
                        )

                    elif motion.get_profile().stream_type() == rs.stream.gyro:
                        gyro_data = motion.get_motion_data()
                        self.imu_file.write(
                            f"{timestamp},{self.frame_count},gyro,"
                            f"{gyro_data.x},{gyro_data.y},{gyro_data.z}\n"
                        )
        except Exception as e:
            if DEBUG_MODE:
                print(f"Error processing IMU data: {e}")

    def stop_recording(self):
        """Stop recording and cleanup resources."""
        self.is_recording = False
        if self.recording_thread:
            self.recording_thread.join(timeout=5.0)

    def _cleanup(self):
        """Release all resources."""
        # Release video writers
        if self.rgb_writer:
            self.rgb_writer.release()
        if self.depth_writer:
            self.depth_writer.release()
        if self.ir_left_writer:
            self.ir_left_writer.release()
        if self.ir_right_writer:
            self.ir_right_writer.release()

        # Close IMU file
        if self.imu_file:
            self.imu_file.close()

        # Stop pipeline
        if self.pipeline:
            self.pipeline.stop()

    def __del__(self):
        """Destructor to ensure cleanup."""
        self._cleanup()
