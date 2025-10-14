"""Intel RealSense camera wrapper for synchronized recording."""
import time
import threading
from pathlib import Path
from datetime import datetime
from typing import Optional, Literal, List, Tuple
from collections import deque
import numpy as np
import cv2
import json
import shutil
from domain.camera_info import CameraInfo
from config import DEBUG_MODE, BASE_DIR, FPS, REALSENSE_WIDTH, REALSENSE_HEIGHT, VIDEO_SECS, START_RECORDING_DELAY

# Import cow volume estimator for Scenario 7
try:
    from utils.cow_volume_and_dimensional_measurements import CowVolumeEstimator
    VOLUME_ESTIMATION_AVAILABLE = True
except ImportError:
    VOLUME_ESTIMATION_AVAILABLE = False
    if DEBUG_MODE:
        print("Warning: CowVolumeEstimator not available. Volume measurements disabled.")

# Import cow motion analyzer for Scenario 8
try:
    from utils.cow_motion_analysis import CowMotionAnalyzer
    MOTION_ANALYSIS_AVAILABLE = True
except ImportError:
    MOTION_ANALYSIS_AVAILABLE = False
    if DEBUG_MODE:
        print("Warning: CowMotionAnalyzer not available. Motion analysis disabled.")

# Import cow trajectory analyzer for Scenario 9
try:
    from utils.cow_trajectory_and_path_analysis import CowTrajectoryAnalyzer
    TRAJECTORY_ANALYSIS_AVAILABLE = True
except ImportError:
    TRAJECTORY_ANALYSIS_AVAILABLE = False
    if DEBUG_MODE:
        print("Warning: CowTrajectoryAnalyzer not available. Trajectory analysis disabled.")

# Import cow scene understanding for Scenario 10
try:
    from utils.cow_sceen_understanding import CowSceneUnderstanding
    SCENE_UNDERSTANDING_AVAILABLE = True
except ImportError:
    SCENE_UNDERSTANDING_AVAILABLE = False
    if DEBUG_MODE:
        print("Warning: CowSceneUnderstanding not available. Scene understanding disabled.")

# Import cow IMU integration for Scenario 11
try:
    from utils.cow_imu_integration import CowIMUIntegration
    IMU_INTEGRATION_AVAILABLE = True
except ImportError:
    IMU_INTEGRATION_AVAILABLE = False
    if DEBUG_MODE:
        print("Warning: CowIMUIntegration not available. IMU integration disabled.")

# Import cow classical CV for Adapted Scenario 12
try:
    from utils.cow_classical_cv import CowClassicalCV
    CLASSICAL_CV_AVAILABLE = True
except ImportError:
    CLASSICAL_CV_AVAILABLE = False
    if DEBUG_MODE:
        print("Warning: CowClassicalCV not available. Classical CV features disabled.")


class RealSenseCameraRecorder:
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
        rgb_format: Literal['bgr8', 'rgb8', 'rgba8'] = 'bgr8',
        enable_timestamp_logging: bool = True,
        enable_frame_skipping: bool = False,
        frame_skip_count: int = 2,
        enable_temporal_validation: bool = False,
        enable_hardware_sync: bool = True,
        # Scenario 4: Recording enhancements
        codec: Literal['mp4v', 'h264', 'h265', 'bag'] = 'mp4v',
        record_rgb: bool = True,
        record_depth: bool = True,
        record_ir: bool = True,
        record_imu: bool = True,
        trigger_mode: Literal['manual', 'timer', 'motion'] = 'manual',
        timer_interval: int = 10,
        enable_circular_buffer: bool = False,
        circular_buffer_seconds: int = 300,
        enable_pre_trigger: bool = False,
        pre_trigger_seconds: int = 10,
        split_mode: Optional[Literal['time', 'size']] = None,
        split_interval_minutes: int = 10,
        split_size_mb: int = 1024,
        enable_disk_monitoring: bool = True,
        disk_warning_gb: float = 5.0,
        disk_stop_gb: float = 1.0,
        enable_auto_cleanup: bool = False,
        auto_cleanup_threshold: float = 0.8,
        save_metadata: bool = True,
        enable_resume: bool = False,
        # Scenario 7: Volume and weight estimation for cows
        enable_volume_estimation: bool = False,
        volume_animal_type: Literal['cow', 'bull', 'calf', 'heifer', 'steer'] = 'cow',
        volume_bounding_box: Optional[Tuple[int, int, int, int]] = None,
        volume_measurement_interval: int = 30,  # Measure every 30 seconds
        # Scenario 8: Motion analysis for cows
        enable_motion_analysis: bool = False,
        motion_animal_type: Literal['cow', 'bull', 'calf', 'heifer', 'steer'] = 'cow',
        motion_analysis_interval: int = 1,  # Analyze every 1 second (every N seconds)
        # Scenario 9: Trajectory tracking and path analysis for cows
        enable_trajectory_analysis: bool = False,
        trajectory_animal_type: Literal['cow', 'bull', 'calf', 'heifer', 'steer'] = 'cow',
        trajectory_tracking_interval: int = 1,  # Record position every N seconds
        # Scenario 10: Scene understanding for cows
        enable_scene_understanding: bool = False,
        scene_environment_type: Literal['feedlot_closed', 'feedlot_outdoor', 'pasture_natural', 'pasture_fodder'] = 'feedlot_outdoor',
        scene_analysis_interval: int = 5,  # Analyze scene every N seconds
        # Scenario 11: IMU integration for cows
        enable_imu_integration: bool = False,
        imu_animal_type: Literal['cow', 'bull', 'calf', 'heifer', 'steer'] = 'cow',
        imu_environment_type: Literal['feedlot_closed', 'feedlot_outdoor', 'pasture_natural', 'pasture_fodder'] = 'feedlot_outdoor',
        imu_sampling_rate: float = 200.0,  # IMU sampling rate in Hz (D455i default: 200-400 Hz)
        # Adapted Scenario 12: Classical computer vision for cattle health & welfare
        enable_classical_cv: bool = False,
        classical_cv_animal_type: Literal['cow', 'bull', 'calf', 'heifer', 'steer'] = 'cow',
        classical_cv_environment_type: Literal['feedlot_closed', 'feedlot_outdoor', 'pasture_natural', 'pasture_fodder'] = 'feedlot_outdoor',
        classical_cv_analysis_interval: int = 10,  # Analyze every N seconds (BCS, health, behavior)
    ):
        """
        Initialize RealSense camera recorder.

        Args:
            camera_info: Camera metadata from enumeration
            output_dir: Directory to save recordings
            recording_duration: Duration in seconds
            fps: Target frame rate (15, 30, 60, or 90)
            width: RGB/Depth stream width (1920, 1280, 848, 640)
            height: RGB/Depth stream height (1080, 720, 480, 360)
            recording_delay: Delay before starting recording (seconds)
            rgb_format: RGB stream format ('bgr8', 'rgb8', or 'rgba8')
            enable_timestamp_logging: Enable detailed timestamp logging
            enable_frame_skipping: Enable frame skipping for performance
            frame_skip_count: Number of frames to skip (skip every Nth frame)
            enable_temporal_validation: Validate temporal consistency of frames
            enable_hardware_sync: Enable hardware timestamp synchronization

            # Scenario 4: Recording enhancements
            codec: Video codec ('mp4v', 'h264', 'h265', 'bag')
            record_rgb: Enable RGB stream recording
            record_depth: Enable depth stream recording
            record_ir: Enable infrared stream recording
            record_imu: Enable IMU data recording (D455i only)
            trigger_mode: Recording trigger mode ('manual', 'timer', 'motion')
            timer_interval: Timer interval in seconds (for timer mode)
            enable_circular_buffer: Enable circular buffer recording
            circular_buffer_seconds: Circular buffer duration (default 300s = 5 min)
            enable_pre_trigger: Enable pre-trigger recording
            pre_trigger_seconds: Pre-trigger buffer duration (default 10s)
            split_mode: Recording split mode ('time', 'size', or None)
            split_interval_minutes: Split interval in minutes (for time-based split)
            split_size_mb: Split size in MB (for size-based split)
            enable_disk_monitoring: Enable disk space monitoring
            disk_warning_gb: Disk space warning threshold in GB
            disk_stop_gb: Disk space stop threshold in GB
            enable_auto_cleanup: Enable automatic cleanup of old recordings
            auto_cleanup_threshold: Disk usage threshold for cleanup (0.8 = 80%)
            save_metadata: Save recording metadata to JSON
            enable_resume: Enable resume after interruption

            # Scenario 7: Volume and weight estimation
            enable_volume_estimation: Enable cow volume and weight estimation
            volume_animal_type: Type of cattle ('cow', 'bull', 'calf', 'heifer', 'steer')
            volume_bounding_box: Optional bounding box (x_min, x_max, y_min, y_max) for animal ROI
            volume_measurement_interval: Interval in seconds between volume measurements

            # Scenario 8: Motion analysis
            enable_motion_analysis: Enable cow motion analysis (optical flow, activity, gait)
            motion_animal_type: Type of cattle for motion analysis
            motion_analysis_interval: Interval in seconds between motion analysis reports

            # Scenario 9: Trajectory tracking and path analysis
            enable_trajectory_analysis: Enable trajectory tracking and path analysis
            trajectory_animal_type: Type of cattle for trajectory analysis
            trajectory_tracking_interval: Interval in seconds between trajectory point recording

            # Scenario 10: Scene understanding
            enable_scene_understanding: Enable scene understanding (planes, objects, environment)
            scene_environment_type: Type of environment ('feedlot_closed', 'feedlot_outdoor', 'pasture_natural', 'pasture_fodder')
            scene_analysis_interval: Interval in seconds between scene analysis reports

            # Scenario 11: IMU integration
            enable_imu_integration: Enable IMU data processing and sensor fusion
            imu_animal_type: Type of cattle for IMU integration
            imu_environment_type: Type of environment for IMU processing
            imu_sampling_rate: IMU sampling rate in Hz (typically 200-400 Hz for D455i)

            # Adapted Scenario 12: Classical computer vision
            enable_classical_cv: Enable classical CV analysis (BCS, health, behavior, identification)
            classical_cv_animal_type: Type of cattle for classical CV analysis
            classical_cv_environment_type: Type of environment for CV processing
            classical_cv_analysis_interval: Interval in seconds between classical CV analysis reports
        """
        # Basic configuration
        self.camera_info = camera_info
        self.output_dir = Path(output_dir)
        self.recording_duration = recording_duration
        self.fps = fps
        self.width = width
        self.height = height
        self.recording_delay = recording_delay
        self.rgb_format = rgb_format
        self.enable_timestamp_logging = enable_timestamp_logging
        self.enable_frame_skipping = enable_frame_skipping
        self.frame_skip_count = frame_skip_count
        self.enable_temporal_validation = enable_temporal_validation
        self.enable_hardware_sync = enable_hardware_sync

        # Scenario 4: Recording enhancements
        self.codec = codec
        self.record_rgb = record_rgb
        self.record_depth = record_depth
        self.record_ir = record_ir
        self.record_imu = record_imu and camera_info.capabilities.imu
        self.trigger_mode = trigger_mode
        self.timer_interval = timer_interval
        self.enable_circular_buffer = enable_circular_buffer
        self.circular_buffer_seconds = circular_buffer_seconds
        self.enable_pre_trigger = enable_pre_trigger
        self.pre_trigger_seconds = pre_trigger_seconds
        self.split_mode = split_mode
        self.split_interval_minutes = split_interval_minutes
        self.split_size_mb = split_size_mb
        self.enable_disk_monitoring = enable_disk_monitoring
        self.disk_warning_gb = disk_warning_gb
        self.disk_stop_gb = disk_stop_gb
        self.enable_auto_cleanup = enable_auto_cleanup
        self.auto_cleanup_threshold = auto_cleanup_threshold
        self.save_metadata = save_metadata
        self.enable_resume = enable_resume

        # Scenario 7: Volume and weight estimation
        self.enable_volume_estimation = enable_volume_estimation and VOLUME_ESTIMATION_AVAILABLE
        self.volume_animal_type = volume_animal_type
        self.volume_bounding_box = volume_bounding_box
        self.volume_measurement_interval = volume_measurement_interval
        self.last_volume_measurement_time = 0

        # Initialize volume estimator if enabled
        self.volume_estimator = None
        if self.enable_volume_estimation:
            try:
                self.volume_estimator = CowVolumeEstimator(animal_type=volume_animal_type)
                if DEBUG_MODE:
                    print(f"CowVolumeEstimator initialized for {volume_animal_type}")
            except Exception as e:
                if DEBUG_MODE:
                    print(f"Failed to initialize CowVolumeEstimator: {e}")
                self.enable_volume_estimation = False

        # Scenario 8: Motion analysis
        self.enable_motion_analysis = enable_motion_analysis and MOTION_ANALYSIS_AVAILABLE
        self.motion_animal_type = motion_animal_type
        self.motion_analysis_interval = motion_analysis_interval
        self.last_motion_analysis_time = 0

        # Initialize motion analyzer if enabled
        self.motion_analyzer = None
        if self.enable_motion_analysis:
            try:
                self.motion_analyzer = CowMotionAnalyzer(animal_type=motion_animal_type, fps=fps)
                if DEBUG_MODE:
                    print(f"CowMotionAnalyzer initialized for {motion_animal_type} at {fps} fps")
            except Exception as e:
                if DEBUG_MODE:
                    print(f"Failed to initialize CowMotionAnalyzer: {e}")
                self.enable_motion_analysis = False

        # Scenario 9: Trajectory tracking and path analysis
        self.enable_trajectory_analysis = enable_trajectory_analysis and TRAJECTORY_ANALYSIS_AVAILABLE
        self.trajectory_animal_type = trajectory_animal_type
        self.trajectory_tracking_interval = trajectory_tracking_interval
        self.last_trajectory_tracking_time = 0

        # Initialize trajectory analyzer if enabled
        self.trajectory_analyzer = None
        if self.enable_trajectory_analysis:
            try:
                self.trajectory_analyzer = CowTrajectoryAnalyzer(animal_type=trajectory_animal_type, fps=fps)
                if DEBUG_MODE:
                    print(f"CowTrajectoryAnalyzer initialized for {trajectory_animal_type} at {fps} fps")
            except Exception as e:
                if DEBUG_MODE:
                    print(f"Failed to initialize CowTrajectoryAnalyzer: {e}")
                self.enable_trajectory_analysis = False

        # Scenario 10: Scene understanding
        self.enable_scene_understanding = enable_scene_understanding and SCENE_UNDERSTANDING_AVAILABLE
        self.scene_environment_type = scene_environment_type
        self.scene_analysis_interval = scene_analysis_interval
        self.last_scene_analysis_time = 0

        # Initialize scene understanding if enabled
        self.scene_understanding = None
        if self.enable_scene_understanding:
            try:
                self.scene_understanding = CowSceneUnderstanding(animal_type='cow', environment_type=scene_environment_type)
                if DEBUG_MODE:
                    print(f"CowSceneUnderstanding initialized for {scene_environment_type} environment")
            except Exception as e:
                if DEBUG_MODE:
                    print(f"Failed to initialize CowSceneUnderstanding: {e}")
                self.enable_scene_understanding = False

        # Scenario 11: IMU integration
        self.enable_imu_integration = enable_imu_integration and IMU_INTEGRATION_AVAILABLE and camera_info.capabilities.imu
        self.imu_animal_type = imu_animal_type
        self.imu_environment_type = imu_environment_type
        self.imu_sampling_rate = imu_sampling_rate

        # Initialize IMU integration if enabled
        self.imu_integration = None
        if self.enable_imu_integration:
            try:
                self.imu_integration = CowIMUIntegration(
                    animal_type=imu_animal_type,
                    environment_type=imu_environment_type,
                    sampling_rate=imu_sampling_rate
                )
                if DEBUG_MODE:
                    print(f"CowIMUIntegration initialized for {imu_animal_type} in {imu_environment_type} at {imu_sampling_rate} Hz")
            except Exception as e:
                if DEBUG_MODE:
                    print(f"Failed to initialize CowIMUIntegration: {e}")
                self.enable_imu_integration = False

        # Adapted Scenario 12: Classical computer vision
        self.enable_classical_cv = enable_classical_cv and CLASSICAL_CV_AVAILABLE
        self.classical_cv_animal_type = classical_cv_animal_type
        self.classical_cv_environment_type = classical_cv_environment_type
        self.classical_cv_analysis_interval = classical_cv_analysis_interval
        self.last_classical_cv_analysis_time = 0

        # Initialize classical CV analyzer if enabled
        self.classical_cv_analyzer = None
        if self.enable_classical_cv:
            try:
                self.classical_cv_analyzer = CowClassicalCV(
                    animal_type=classical_cv_animal_type,
                    environment_type=classical_cv_environment_type,
                    enable_csv_export=True,
                    csv_output_dir=str(self.output_dir / "classical_cv_results")
                )
                if DEBUG_MODE:
                    print(f"CowClassicalCV initialized for {classical_cv_animal_type} in {classical_cv_environment_type}")
            except Exception as e:
                if DEBUG_MODE:
                    print(f"Failed to initialize CowClassicalCV: {e}")
                self.enable_classical_cv = False

        # Recording state
        self.pipeline = None
        self.config = None
        self.align = None
        self.bag_recorder = None  # For .bag format
        self.is_recording = False
        self.recording_thread: Optional[threading.Thread] = None
        self.start_timestamp: Optional[float] = None
        self.frame_count = 0
        self.processed_frame_count = 0
        self.current_file_index = 0

        # Video writers for each stream
        self.rgb_writer = None
        self.depth_writer = None
        self.ir_left_writer = None
        self.ir_right_writer = None

        # IMU data file (D455i only)
        self.imu_file = None

        # Timestamp tracking for validation
        self.timestamp_log_file = None
        self.last_timestamp: Optional[float] = None
        self.timestamp_violations = 0

        # Circular buffer for pre-trigger
        self.circular_buffer: deque = deque()
        self.pre_trigger_buffer: deque = deque()

        # File splitting
        self.current_file_size = 0
        self.last_split_time = 0

        # Metadata
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

            # Enable RGB stream with configurable format
            rgb_format_map = {
                'bgr8': rs.format.bgr8,
                'rgb8': rs.format.rgb8,
                'rgba8': rs.format.rgba8,
            }
            rs_format = rgb_format_map.get(self.rgb_format, rs.format.bgr8)

            self.config.enable_stream(
                rs.stream.color,
                self.width,
                self.height,
                rs_format,
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

            # Enable hardware timestamp synchronization if requested
            if self.enable_hardware_sync:
                try:
                    device = profile.get_device()
                    # Enable inter-stream synchronization
                    # This ensures all streams (RGB, depth, IR) are timestamped consistently
                    depth_sensor = device.first_depth_sensor()
                    if depth_sensor.supports(rs.option.global_time_enabled):
                        depth_sensor.set_option(rs.option.global_time_enabled, 1)

                    # Enable frames queue for synchronized capture
                    color_sensor = device.first_color_sensor()
                    if color_sensor.supports(rs.option.global_time_enabled):
                        color_sensor.set_option(rs.option.global_time_enabled, 1)

                    if DEBUG_MODE:
                        print(f"Hardware timestamp synchronization enabled")
                except Exception as e:
                    if DEBUG_MODE:
                        print(f"Warning: Could not enable hardware sync: {e}")

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

            # SLOW OPERATION: Create timestamp log file if enabled
            if self.enable_timestamp_logging:
                self._create_timestamp_log_file()

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

    def _get_codec_fourcc(self) -> int:
        """
        Get video codec FourCC based on codec selection.

        Returns:
            OpenCV FourCC code for the selected codec
        """
        codec_map = {
            'mp4v': cv2.VideoWriter_fourcc(*'mp4v'),
            'h264': cv2.VideoWriter_fourcc(*'H264'),  # or 'X264', 'avc1'
            'h265': cv2.VideoWriter_fourcc(*'HEVC'),  # or 'H265', 'hev1'
        }

        return codec_map.get(self.codec, cv2.VideoWriter_fourcc(*'mp4v'))

    def _create_video_writers(self):
        """Create OpenCV VideoWriter instances for each stream."""

        # Get codec
        if self.codec == 'bag':
            # For .bag format, we'll use pyrealsense2 recorder
            self._create_bag_recorder()
            return

        fourcc = self._get_codec_fourcc()

        # RGB stream - For RGBA, we'll convert to RGB before writing
        # since OpenCV VideoWriter doesn't support 4-channel RGBA directly
        if self.record_rgb:
            rgb_filename = self._generate_filename("rgb", "mp4", self.current_file_index)
            self.rgb_writer = cv2.VideoWriter(
                str(self.output_dir / rgb_filename),
                fourcc,
                self.fps,
                (self.width, self.height)
            )

        # Depth stream (saved as individual .npz frames - NO VIDEO WRITER NEEDED)
        # Depth frames will be saved as compressed NumPy arrays (.npz) with metadata
        # This provides: 40% smaller size, 3-5x faster loading, no data loss (uint16)
        # Each frame saved individually with timestamp, frame_number, depth_scale
        if self.record_depth:
            # No depth_writer needed for .npz format
            pass

        # Infrared left
        if self.record_ir:
            ir_left_filename = self._generate_filename("ir_left", "mp4", self.current_file_index)
            self.ir_left_writer = cv2.VideoWriter(
                str(self.output_dir / ir_left_filename),
                fourcc,
                self.fps,
                (self.width, self.height),
                isColor=False
            )

            # Infrared right
            ir_right_filename = self._generate_filename("ir_right", "mp4", self.current_file_index)
            self.ir_right_writer = cv2.VideoWriter(
                str(self.output_dir / ir_right_filename),
                fourcc,
                self.fps,
                (self.width, self.height),
                isColor=False
            )

    def _create_bag_recorder(self):
        """Create .bag recorder for lossless RAW recording."""
        import pyrealsense2 as rs

        bag_filename = self._generate_filename("recording", "bag", self.current_file_index)
        bag_path = str(self.output_dir / bag_filename)

        self.bag_recorder = rs.recorder(bag_path, self.pipeline.get_active_profile().get_device())

        if DEBUG_MODE:
            print(f"Created .bag recorder: {bag_path}")

    def _create_imu_file(self):
        """Create CSV file for IMU data."""
        imu_filename = self._generate_filename("imu", "csv")
        self.imu_file = open(self.output_dir / imu_filename, 'w')
        # Write CSV header
        self.imu_file.write("timestamp,frame_number,sensor,x,y,z\n")

    def _create_timestamp_log_file(self):
        """Create CSV file for detailed timestamp logging."""
        timestamp_filename = self._generate_filename("timestamps", "csv")
        self.timestamp_log_file = open(self.output_dir / timestamp_filename, 'w')
        # Write CSV header
        self.timestamp_log_file.write(
            "frame_number,system_timestamp,hardware_timestamp,"
            "rgb_timestamp,depth_timestamp,ir_left_timestamp,ir_right_timestamp,"
            "timestamp_delta,temporal_valid\n"
        )

    def _generate_filename(self, sensor: str, extension: str, file_index: int = 0) -> str:
        """
        Generate filename following the required format.

        Format: YYYY-MM-dd-HH-mm-ss-sss-{CAMERA_INDEX}-{SENSOR}-{INDEX}.ext

        Args:
            sensor: Sensor name (rgb, depth, ir_left, ir_right, imu, etc.)
            extension: File extension (mp4, csv, bag, etc.)
            file_index: File index for split recordings (default 0)

        Returns:
            Formatted filename
        """
        dt = datetime.fromtimestamp(self.start_timestamp)
        timestamp_str = dt.strftime("%Y-%m-%d-%H-%M-%S-%f")[:-3]  # milliseconds
        cam_idx = self.camera_info.index if self.camera_info.index is not None else 0

        if file_index > 0:
            return f"{timestamp_str}-{cam_idx}-{sensor}-{file_index:03d}.{extension}"
        else:
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

                # Increment frame counter
                self.frame_count += 1

                # Frame skipping logic
                if self.enable_frame_skipping:
                    if self.frame_count % self.frame_skip_count != 0:
                        continue  # Skip this frame

                # Increment processed frame counter
                self.processed_frame_count += 1

                # Get frame timestamps for logging and validation
                rgb_timestamp = color_frame.get_timestamp()
                depth_timestamp = depth_frame.get_timestamp()
                ir_left_timestamp = ir_left_frame.get_timestamp() if ir_left_frame else 0
                ir_right_timestamp = ir_right_frame.get_timestamp() if ir_right_frame else 0

                # Temporal consistency validation
                temporal_valid = True
                timestamp_delta = 0.0
                if self.enable_temporal_validation and self.last_timestamp is not None:
                    timestamp_delta = rgb_timestamp - self.last_timestamp
                    expected_delta = 1000.0 / self.fps  # Expected delta in milliseconds

                    # Check if timestamp is monotonically increasing
                    if timestamp_delta <= 0:
                        temporal_valid = False
                        self.timestamp_violations += 1
                        if DEBUG_MODE:
                            print(f"Warning: Timestamp inversion detected at frame {self.frame_count}")

                    # Check if delta matches expected frame rate (with 10% tolerance)
                    elif abs(timestamp_delta - expected_delta) > (expected_delta * 0.1):
                        if DEBUG_MODE:
                            print(f"Warning: Timestamp delta {timestamp_delta:.2f}ms "
                                  f"differs from expected {expected_delta:.2f}ms")

                self.last_timestamp = rgb_timestamp

                # Log timestamps if enabled
                if self.enable_timestamp_logging and self.timestamp_log_file:
                    self.timestamp_log_file.write(
                        f"{self.processed_frame_count},{current_time},{rgb_timestamp},"
                        f"{rgb_timestamp},{depth_timestamp},"
                        f"{ir_left_timestamp},{ir_right_timestamp},"
                        f"{timestamp_delta},{temporal_valid}\n"
                    )

                # Convert to numpy arrays
                color_image = np.asanyarray(color_frame.get_data())
                depth_image = np.asanyarray(depth_frame.get_data())

                # Handle RGBA format - convert to BGR for video writing
                if self.rgb_format == 'rgba8':
                    # RGBA has 4 channels, convert to BGR (3 channels) for video
                    color_image = cv2.cvtColor(color_image, cv2.COLOR_RGBA2BGR)
                elif self.rgb_format == 'rgb8':
                    # RGB to BGR conversion
                    color_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)
                # If bgr8, no conversion needed

                # Write RGB frame
                if self.record_rgb and self.rgb_writer:
                    self.rgb_writer.write(color_image)

                # Save depth frame as .npz (compressed NumPy format with metadata)
                if self.record_depth:
                    # Generate filename with frame number for individual depth frame
                    depth_filename = self._generate_filename(
                        f"depth_{self.processed_frame_count:06d}",
                        "npz",
                        self.current_file_index
                    )

                    # Save depth frame with metadata
                    np.savez_compressed(
                        self.output_dir / depth_filename,
                        depth=depth_image,  # uint16 raw depth values (millimeters)
                        timestamp=depth_timestamp,  # Hardware timestamp from camera
                        frame_number=self.processed_frame_count,
                        depth_scale=0.001,  # RealSense D455i: mm → m conversion factor
                        camera_index=self.camera_info.index
                    )

                # Write infrared frames
                if ir_left_frame:
                    ir_left_image = np.asanyarray(ir_left_frame.get_data())
                    self.ir_left_writer.write(ir_left_image)

                if ir_right_frame:
                    ir_right_image = np.asanyarray(ir_right_frame.get_data())
                    self.ir_right_writer.write(ir_right_image)

                # Perform volume and weight estimation if enabled
                if self.enable_volume_estimation:
                    self._process_volume_measurement(depth_frame, color_frame, current_time)

                # Perform motion analysis if enabled
                if self.enable_motion_analysis:
                    self._process_motion_analysis(color_image, depth_image, depth_frame, current_time)

                # Perform trajectory tracking if enabled
                if self.enable_trajectory_analysis:
                    self._process_trajectory_tracking(depth_frame, color_frame, current_time)

                # Perform scene understanding if enabled
                if self.enable_scene_understanding:
                    self._process_scene_understanding(depth_frame, color_frame, current_time)

                # Perform classical CV analysis if enabled
                if self.enable_classical_cv:
                    self._process_classical_cv(color_image, depth_image, depth_frame, current_time)

                # Process IMU data if available
                if self.camera_info.capabilities.imu:
                    self._process_imu_data(frames, current_time)

        except Exception as e:
            if DEBUG_MODE:
                print(f"Error in RealSense recording loop: {e}")
        finally:
            self._cleanup()

    def _process_imu_data(self, frames, timestamp: float):
        """
        Process and save IMU data (accelerometer and gyroscope).

        Enhanced with Scenario 11 IMU Integration:
        - Captures and buffers IMU data
        - Performs sensor fusion for orientation estimation
        - Analyzes vibration and motion blur
        - Estimates camera pose from IMU
        - Generates comprehensive IMU reports

        Args:
            frames: RealSense frameset
            timestamp: Current timestamp
        """
        try:
            import pyrealsense2 as rs

            # Track IMU data for this frame
            accel_captured = None
            gyro_captured = None
            current_dt = datetime.fromtimestamp(timestamp)

            # Get motion frames
            for frame in frames:
                if frame.is_motion_frame():
                    motion = frame.as_motion_frame()

                    if motion.get_profile().stream_type() == rs.stream.accel:
                        accel_data = motion.get_motion_data()

                        # Write raw data to IMU file
                        if self.imu_file:
                            self.imu_file.write(
                                f"{timestamp},{self.frame_count},accel,"
                                f"{accel_data.x},{accel_data.y},{accel_data.z}\n"
                            )

                        # Scenario 11: Capture accelerometer data with IMU integration
                        if self.enable_imu_integration and self.imu_integration:
                            accel_captured = self.imu_integration.capture_accelerometer_data(
                                accel_x=accel_data.x,
                                accel_y=accel_data.y,
                                accel_z=accel_data.z,
                                timestamp=current_dt,
                                frame_number=self.processed_frame_count
                            )

                    elif motion.get_profile().stream_type() == rs.stream.gyro:
                        gyro_data = motion.get_motion_data()

                        # Write raw data to IMU file
                        if self.imu_file:
                            self.imu_file.write(
                                f"{timestamp},{self.frame_count},gyro,"
                                f"{gyro_data.x},{gyro_data.y},{gyro_data.z}\n"
                            )

                        # Scenario 11: Capture gyroscope data with IMU integration
                        if self.enable_imu_integration and self.imu_integration:
                            gyro_captured = self.imu_integration.capture_gyroscope_data(
                                gyro_x=gyro_data.x,
                                gyro_y=gyro_data.y,
                                gyro_z=gyro_data.z,
                                timestamp=current_dt,
                                frame_number=self.processed_frame_count
                            )

            # Scenario 11: Process IMU integration if enabled and we have data
            if self.enable_imu_integration and self.imu_integration:
                # Synchronize IMU data with frame
                synchronized_data = self.imu_integration.synchronize_imu_with_frame(
                    frame_number=self.processed_frame_count,
                    frame_timestamp=current_dt
                )

                # Generate comprehensive IMU report if synchronized data available
                if synchronized_data:
                    # Get depth map for correction (if available)
                    depth_map = None
                    try:
                        for frame in frames:
                            if frame.is_depth_frame():
                                depth_map = np.asanyarray(frame.get_data())
                                break
                    except:
                        pass

                    # Generate IMU report with all analyses
                    imu_report = self.imu_integration.generate_imu_report(
                        frame_number=self.processed_frame_count,
                        timestamp=current_dt,
                        synchronized_data=synchronized_data,
                        depth_map=depth_map,
                        previous_orientation=self.imu_integration.current_orientation
                    )

                    # Save report to CSV
                    imu_csv = self._generate_filename("imu_report", "csv")
                    imu_path = str(self.output_dir / imu_csv)
                    self.imu_integration.save_imu_report_to_csv(imu_report, imu_path)

                    # Debug output
                    if DEBUG_MODE and self.processed_frame_count % (self.fps * 5) == 0:  # Every 5 seconds
                        print(f"\n{'='*60}")
                        print(f"IMU Report - Frame {self.processed_frame_count}")
                        print(f"{'='*60}")
                        print(f"Orientation:")
                        print(f"  Roll:  {np.degrees(imu_report.orientation.roll):.2f}°")
                        print(f"  Pitch: {np.degrees(imu_report.orientation.pitch):.2f}°")
                        print(f"  Yaw:   {np.degrees(imu_report.orientation.yaw):.2f}°")
                        print(f"  Confidence: {imu_report.orientation.confidence:.3f}")
                        print(f"Gravity:")
                        print(f"  Magnitude: {imu_report.gravity.magnitude:.2f} m/s²")
                        print(f"  Direction: ({imu_report.gravity.direction[0]:.3f}, "
                              f"{imu_report.gravity.direction[1]:.3f}, {imu_report.gravity.direction[2]:.3f})")
                        if imu_report.vibration:
                            print(f"Vibration:")
                            print(f"  Magnitude: {imu_report.vibration.vibration_magnitude:.2f} m/s²")
                            print(f"  Shaking: {imu_report.vibration.is_shaking}")
                            print(f"  Dominant Frequency: {imu_report.vibration.dominant_frequency:.1f} Hz")
                        if imu_report.motion_blur:
                            print(f"Motion Blur:")
                            print(f"  Angular Velocity: {imu_report.motion_blur.angular_velocity_magnitude:.3f} rad/s")
                            print(f"  Estimated Blur: {imu_report.motion_blur.estimated_blur_pixels:.2f} pixels")
                            print(f"  Blur Level: {imu_report.motion_blur.blur_level}")
                        print(f"Camera Pose:")
                        print(f"  Pointing: ({imu_report.camera_pose.pointing_direction[0]:.3f}, "
                              f"{imu_report.camera_pose.pointing_direction[1]:.3f}, "
                              f"{imu_report.camera_pose.pointing_direction[2]:.3f})")
                        print(f"Sync Offset: {synchronized_data.time_offset_ms:.2f} ms")
                        print(f"{'='*60}\n")

        except Exception as e:
            if DEBUG_MODE:
                print(f"Error processing IMU data: {e}")
                import traceback
                traceback.print_exc()

    def _process_volume_measurement(self, depth_frame, color_frame, timestamp: float):
        """
        Process volume and weight estimation for detected animal.

        This method integrates the CowVolumeEstimator from Scenario 7-Feature-1
        to calculate animal volume, body dimensions, and weight estimates from
        the point cloud data.

        Args:
            depth_frame: RealSense depth frame
            color_frame: RealSense color frame
            timestamp: Current timestamp
        """
        if not self.volume_estimator:
            return

        # Check if it's time to measure (based on interval)
        if timestamp - self.last_volume_measurement_time < self.volume_measurement_interval:
            return

        try:
            import pyrealsense2 as rs

            # Generate point cloud
            pc = rs.pointcloud()
            points_data = pc.calculate(depth_frame)
            pc.map_to(color_frame)

            # Extract vertices as NumPy array
            vertices = np.asanyarray(points_data.get_vertices())
            vertices_array = np.array([[v[0], v[1], v[2]] for v in vertices])

            # Filter invalid points
            valid_mask = ~np.isnan(vertices_array).any(axis=1) & \
                        ~np.isinf(vertices_array).any(axis=1) & \
                        (vertices_array != 0).any(axis=1)
            points = vertices_array[valid_mask]

            if len(points) < 100:  # Need minimum points for valid measurement
                if DEBUG_MODE:
                    print(f"Insufficient points for volume measurement: {len(points)}")
                return

            # Extract animal point cloud from bounding box if specified
            if self.volume_bounding_box:
                # Bounding box format: (x_min, x_max, y_min, y_max)
                x_min, x_max, y_min, y_max = self.volume_bounding_box
                # Filter points within bounding box (using X-Y plane)
                bbox_mask = (points[:, 0] >= x_min) & (points[:, 0] <= x_max) & \
                           (points[:, 1] >= y_min) & (points[:, 1] <= y_max)
                animal_points = points[bbox_mask]

                if len(animal_points) < 50:
                    if DEBUG_MODE:
                        print(f"Insufficient animal points in bounding box: {len(animal_points)}")
                    return
            else:
                # Use entire point cloud (no bounding box specified)
                animal_points = points

            # Remove ground plane
            animal_points_no_ground, plane_coeffs = self.volume_estimator.remove_ground_plane(
                animal_points,
                distance_threshold=0.02
            )

            if len(animal_points_no_ground) < 50:
                if DEBUG_MODE:
                    print(f"Insufficient points after ground removal: {len(animal_points_no_ground)}")
                return

            # Calculate volume using all methods
            volume_metrics = self.volume_estimator.calculate_all_volumes(animal_points_no_ground)

            # Calculate body dimensions
            body_dimensions = self.volume_estimator.calculate_bounding_box_dimensions(
                animal_points_no_ground
            )

            # Estimate weight
            weight_estimate = self.volume_estimator.estimate_weight_from_volume(
                volume_metrics.best_estimate,
                method='density'  # Use density-based by default (can be calibrated later)
            )

            # Generate unique animal ID (could be enhanced with actual tracking)
            animal_id = f"animal_{self.camera_info.index}_{int(timestamp)}"

            # Track volume over time
            self.volume_estimator.track_volume_over_time(
                animal_id=animal_id,
                volume=volume_metrics.best_estimate,
                timestamp=datetime.fromtimestamp(timestamp)
            )

            # Calculate growth rate if we have history
            growth_rate = self.volume_estimator.calculate_growth_rate(animal_id)

            # Generate comprehensive measurement report
            from utils.animal_volume_measurements import MeasurementReport
            report = MeasurementReport(
                animal_id=animal_id,
                animal_type=self.volume_animal_type,
                timestamp=datetime.fromtimestamp(timestamp),
                volume_metrics=volume_metrics,
                body_dimensions=body_dimensions,
                weight_estimate=weight_estimate,
                growth_rate_m3_per_day=growth_rate,
                previous_volume=None,  # Would need tracking to fill this
                volume_change=None,  # Would need tracking to fill this
                herd_average_comparison=None  # Would need multiple animals to calculate
            )

            # Save measurements to CSV
            csv_filename = self._generate_filename("volume_measurements", "csv")
            csv_path = self.output_dir / csv_filename
            self.volume_estimator.save_measurements_to_csv(report, csv_path)

            # Update last measurement time
            self.last_volume_measurement_time = timestamp

            if DEBUG_MODE:
                print(f"\n{'='*60}")
                print(f"Volume Measurement Report - {animal_id}")
                print(f"{'='*60}")
                print(f"Volume (best estimate): {volume_metrics.best_estimate:.4f} m³ "
                      f"(method: {volume_metrics.method_used})")
                print(f"Volume (convex hull):   {volume_metrics.convex_hull_volume:.4f} m³")
                print(f"Volume (voxel):         {volume_metrics.voxel_volume:.4f} m³")
                print(f"Body Length:            {body_dimensions.length:.3f} m")
                print(f"Body Height:            {body_dimensions.height:.3f} m")
                print(f"Body Width:             {body_dimensions.width:.3f} m")
                print(f"Estimated Weight:       {weight_estimate.estimated_weight_kg:.1f} kg "
                      f"(method: {weight_estimate.method})")
                print(f"Weight Confidence:      {weight_estimate.confidence}")
                print(f"Measurement Confidence: {volume_metrics.confidence:.2f}")
                if growth_rate:
                    print(f"Growth Rate:            {growth_rate:.6f} m³/day")
                print(f"Point Count:            {len(animal_points_no_ground)}")
                print(f"{'='*60}\n")

        except Exception as e:
            if DEBUG_MODE:
                print(f"Error processing volume measurement: {e}")
                import traceback
                traceback.print_exc()

    def _process_motion_analysis(
        self,
        color_image: np.ndarray,
        depth_image: np.ndarray,
        depth_frame,
        timestamp: float
    ):
        """
        Process motion analysis for detected animal.

        This method integrates the CowMotionAnalyzer from Scenario 8
        to calculate optical flow, activity level, speed, and gait metrics.

        Args:
            color_image: RGB frame (BGR format)
            depth_image: Depth image (raw depth values)
            depth_frame: RealSense depth frame object
            timestamp: Current timestamp
        """
        if not self.motion_analyzer:
            return

        # Check if it's time to analyze (based on interval)
        if timestamp - self.last_motion_analysis_time < self.motion_analysis_interval:
            return

        try:
            import pyrealsense2 as rs

            # Generate unique animal ID (could be enhanced with actual tracking)
            animal_id = f"animal_{self.camera_info.index}_{int(timestamp)}"

            # Get camera intrinsics for real-world speed calculation
            camera_intrinsics = None
            try:
                intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics
                camera_intrinsics = {
                    'fx': intrinsics.fx,
                    'fy': intrinsics.fy,
                    'cx': intrinsics.ppx,
                    'cy': intrinsics.ppy
                }
            except Exception as e:
                if DEBUG_MODE:
                    print(f"Warning: Could not get camera intrinsics: {e}")

            # Convert depth image to meters (assuming depth is in millimeters)
            depth_scale = 0.001  # RealSense D455i depth scale
            depth_frame_meters = depth_image.astype(np.float32) * depth_scale

            # Generate comprehensive motion report
            motion_report = self.motion_analyzer.generate_motion_report(
                animal_id=animal_id,
                frame=color_image,
                depth_frame=depth_frame_meters,
                camera_intrinsics=camera_intrinsics,
                frame_number=self.processed_frame_count,
                timestamp=datetime.fromtimestamp(timestamp)
            )

            # Save motion data to CSV
            csv_filename = self._generate_filename("motion_analysis", "csv")
            csv_path = str(self.output_dir / csv_filename)
            self.motion_analyzer.save_motion_data_to_csv(motion_report, csv_path)

            # Update last analysis time
            self.last_motion_analysis_time = timestamp

            # Print debug information
            if DEBUG_MODE and motion_report.optical_flow:
                print(f"\n{'='*60}")
                print(f"Motion Analysis Report - {animal_id}")
                print(f"{'='*60}")
                print(f"Frame: {motion_report.frame_number}")

                if motion_report.activity_metrics:
                    print(f"Activity Level:         {motion_report.activity_metrics.activity_level}")
                    print(f"Total Motion:           {motion_report.activity_metrics.total_motion:.2f}")
                    print(f"Motion Percentage:      {motion_report.activity_metrics.motion_percentage:.2f}%")

                if motion_report.speed_metrics:
                    print(f"Current Speed:          {motion_report.speed_metrics.speed_m_s:.3f} m/s")
                    print(f"Acceleration:           {motion_report.speed_metrics.acceleration_m_s2:.3f} m/s²")
                    print(f"Average Speed:          {motion_report.speed_metrics.average_speed_m_s:.3f} m/s")
                    print(f"Max Speed:              {motion_report.speed_metrics.max_speed_m_s:.3f} m/s")
                    print(f"Distance Traveled:      {motion_report.speed_metrics.distance_traveled_m:.3f} m")

                if motion_report.gait_metrics:
                    print(f"Step Frequency:         {motion_report.gait_metrics.step_frequency_hz:.3f} Hz")
                    print(f"Stride Length:          {motion_report.gait_metrics.stride_length_m:.3f} m")
                    print(f"Gait Cycle Duration:    {motion_report.gait_metrics.gait_cycle_duration_s:.3f} s")
                    print(f"Dominant Frequency:     {motion_report.gait_metrics.dominant_frequency_hz:.3f} Hz")
                    print(f"Periodicity Score:      {motion_report.gait_metrics.periodicity_score:.3f}")
                    print(f"Symmetry Score:         {motion_report.gait_metrics.symmetry_score:.3f}")

                # Optionally save visualization frames periodically
                if self.processed_frame_count % (self.fps * 5) == 0:  # Every 5 seconds
                    # Visualize optical flow
                    if motion_report.optical_flow:
                        flow_vis = self.motion_analyzer.visualize_flow_vectors(
                            color_image,
                            motion_report.optical_flow.flow,
                            step=16
                        )
                        flow_filename = self._generate_filename(
                            f"flow_{self.processed_frame_count // self.fps}",
                            "jpg"
                        )
                        cv2.imwrite(str(self.output_dir / flow_filename), flow_vis)

                print(f"{'='*60}\n")

        except Exception as e:
            if DEBUG_MODE:
                print(f"Error processing motion analysis: {e}")
                import traceback
                traceback.print_exc()

    def _process_trajectory_tracking(
        self,
        depth_frame,
        color_frame,
        timestamp: float
    ):
        """
        Process trajectory tracking for detected animal.

        This method integrates the CowTrajectoryAnalyzer from Scenario 9
        to record 3D trajectory, smooth paths, analyze turning patterns,
        and predict future positions.

        Args:
            depth_frame: RealSense depth frame object
            color_frame: RealSense color frame object
            timestamp: Current timestamp
        """
        if not self.trajectory_analyzer:
            return

        # Check if it's time to track (based on interval)
        if timestamp - self.last_trajectory_tracking_time < self.trajectory_tracking_interval:
            return

        try:
            import pyrealsense2 as rs

            # Generate unique animal ID (could be enhanced with actual tracking/detection)
            animal_id = f"animal_{self.camera_info.index}_{int(timestamp // 60)}"  # Group by minute

            # Generate point cloud to find animal centroid position
            pc = rs.pointcloud()
            points_data = pc.calculate(depth_frame)
            pc.map_to(color_frame)

            # Extract vertices as NumPy array
            vertices = np.asanyarray(points_data.get_vertices())
            vertices_array = np.array([[v[0], v[1], v[2]] for v in vertices])

            # Filter invalid points
            valid_mask = ~np.isnan(vertices_array).any(axis=1) & \
                        ~np.isinf(vertices_array).any(axis=1) & \
                        (vertices_array != 0).any(axis=1)
            points = vertices_array[valid_mask]

            if len(points) < 100:
                return

            # Calculate centroid as animal position
            # In production, this would be enhanced with actual object detection
            centroid = np.mean(points, axis=0)

            # Record trajectory point
            self.trajectory_analyzer.record_trajectory_point(
                animal_id=animal_id,
                position=centroid,
                timestamp=datetime.fromtimestamp(timestamp),
                frame_number=self.processed_frame_count
            )

            # Update last tracking time
            self.last_trajectory_tracking_time = timestamp

            # Periodically generate and save trajectory reports (every 10 seconds)
            if self.processed_frame_count % (self.fps * 10) == 0:
                trajectory_report = self.trajectory_analyzer.generate_trajectory_report(animal_id)

                if trajectory_report:
                    # Save trajectory points to CSV
                    trajectory_csv = self._generate_filename("trajectory_points", "csv")
                    trajectory_path = str(self.output_dir / trajectory_csv)
                    self.trajectory_analyzer.save_trajectory_to_csv(
                        trajectory_report.trajectory,
                        trajectory_path
                    )

                    # Save trajectory report to CSV
                    report_csv = self._generate_filename("trajectory_reports", "csv")
                    report_path = str(self.output_dir / report_csv)
                    self.trajectory_analyzer.save_trajectory_report_to_csv(
                        trajectory_report,
                        report_path
                    )

                    if DEBUG_MODE:
                        print(f"\n{'='*60}")
                        print(f"Trajectory Report - {animal_id}")
                        print(f"{'='*60}")
                        print(f"Total Points:           {len(trajectory_report.trajectory.points)}")
                        print(f"Path Length:            {trajectory_report.path_metrics.total_length_m:.3f} m")
                        print(f"Average Speed:          {trajectory_report.path_metrics.average_speed_m_s:.3f} m/s")
                        print(f"Max Speed:              {trajectory_report.path_metrics.max_speed_m_s:.3f} m/s")
                        print(f"Duration:               {trajectory_report.path_metrics.total_duration_s:.1f} s")
                        print(f"Sharp Turns:            {trajectory_report.path_metrics.sharp_turns_count}")

                        if trajectory_report.predicted_positions:
                            print(f"Predicted Future Positions (next 5 seconds):")
                            for i, pred_pos in enumerate(trajectory_report.predicted_positions, 1):
                                print(f"  +{i}s: ({pred_pos[0]:.3f}, {pred_pos[1]:.3f}, {pred_pos[2]:.3f})")

                        if trajectory_report.cluster_assignment is not None:
                            print(f"Cluster Assignment:     {trajectory_report.cluster_assignment}")

                        print(f"{'='*60}\n")

                        # Optionally smooth trajectory and compare
                        if len(trajectory_report.trajectory.points) >= 10:
                            smoothed_traj = self.trajectory_analyzer.smooth_trajectory_moving_average(
                                trajectory_report.trajectory,
                                window_size=5
                            )
                            print(f"Smoothed Trajectory:")
                            print(f"  Original Length: {trajectory_report.trajectory.total_length:.3f} m")
                            print(f"  Smoothed Length: {smoothed_traj.total_length:.3f} m")
                            print(f"  Difference:      {abs(trajectory_report.trajectory.total_length - smoothed_traj.total_length):.3f} m")
                            print()

        except Exception as e:
            if DEBUG_MODE:
                print(f"Error processing trajectory tracking: {e}")
                import traceback
                traceback.print_exc()

    def _process_scene_understanding(
        self,
        depth_frame,
        color_frame,
        timestamp: float
    ):
        """
        Process scene understanding for environment analysis.

        This method integrates the CowSceneUnderstanding from Scenario 10
        to detect planes (floor, walls, ceiling), identify objects (feed troughs,
        water sources, equipment), and analyze environmental conditions (lighting,
        shadows, wetness/mud).

        Args:
            depth_frame: RealSense depth frame object
            color_frame: RealSense color frame object
            timestamp: Current timestamp
        """
        if not self.scene_understanding:
            return

        # Check if it's time to analyze (based on interval)
        if timestamp - self.last_scene_analysis_time < self.scene_analysis_interval:
            return

        try:
            import pyrealsense2 as rs

            # Generate point cloud for plane detection
            pc = rs.pointcloud()
            points_data = pc.calculate(depth_frame)
            pc.map_to(color_frame)

            # Extract vertices as NumPy array
            vertices = np.asanyarray(points_data.get_vertices())
            vertices_array = np.array([[v[0], v[1], v[2]] for v in vertices])

            # Filter invalid points
            valid_mask = ~np.isnan(vertices_array).any(axis=1) & \
                        ~np.isinf(vertices_array).any(axis=1) & \
                        (vertices_array != 0).any(axis=1)
            points = vertices_array[valid_mask]

            if len(points) < 100:
                return

            # Get RGB and depth images
            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())

            # Convert depth to meters
            depth_scale = self.pipeline.get_active_profile().get_device().first_depth_sensor().get_depth_scale()
            depth_map_meters = depth_image.astype(np.float32) * depth_scale

            # Generate comprehensive scene report
            scene_report = self.scene_understanding.generate_scene_report(
                points=points,
                rgb_image=color_image,
                depth_map=depth_map_meters,
                frame_number=self.processed_frame_count,
                timestamp=datetime.fromtimestamp(timestamp)
            )

            # Update last analysis time
            self.last_scene_analysis_time = timestamp

            # Save scene report to CSV
            scene_csv = self._generate_filename("scene_report", "csv")
            scene_path = str(self.output_dir / scene_csv)
            self.scene_understanding.save_scene_report_to_csv(scene_report, scene_path)

            if DEBUG_MODE:
                print(f"\n{'='*60}")
                print(f"Scene Understanding Report - Frame {self.processed_frame_count}")
                print(f"{'='*60}")
                print(f"Environment Type:       {scene_report.environment_type}")
                print(f"Total Planes:           {len(scene_report.planes)}")

                # Display plane information
                for plane in scene_report.planes:
                    print(f"  - {plane.plane_type.capitalize()}: {plane.inlier_count} inliers, confidence {plane.confidence:.2f}")

                print(f"Total Objects:          {len(scene_report.objects)}")

                # Display object information
                for obj in scene_report.objects:
                    print(f"  - {obj.object_type}: pos ({obj.position[0]:.2f}, {obj.position[1]:.2f}, {obj.position[2]:.2f}), confidence {obj.confidence:.2f}")

                # Display lighting analysis
                print(f"\nLighting Analysis:")
                print(f"  Average Brightness:   {scene_report.lighting.average_brightness:.1f}")
                print(f"  Brightness StdDev:    {scene_report.lighting.std_brightness:.1f}")
                print(f"  Uniformity Score:     {scene_report.lighting.uniformity_score:.3f}")
                print(f"  Lighting Quality:     {scene_report.lighting.lighting_quality}")

                # Display shadow information if available
                if scene_report.shadows:
                    print(f"\nShadow Detection:")
                    print(f"  Shadow Coverage:      {scene_report.shadows.shadow_percentage:.1f}%")
                    print(f"  Shadow Regions:       {len(scene_report.shadows.shadow_regions)}")

                # Display wetness information if available
                if scene_report.wetness:
                    print(f"\nWetness/Mud Detection:")
                    print(f"  Wet Area Coverage:    {scene_report.wetness.wetness_percentage:.1f}%")
                    print(f"  Wet Regions:          {len(scene_report.wetness.wet_regions)}")

                print(f"{'='*60}\n")

        except Exception as e:
            if DEBUG_MODE:
                print(f"Error processing scene understanding: {e}")
                import traceback
                traceback.print_exc()

    def _process_classical_cv(
        self,
        color_image: np.ndarray,
        depth_image: np.ndarray,
        depth_frame,
        timestamp: float
    ):
        """
        Process classical computer vision analysis for cattle health & welfare.

        This method integrates the CowClassicalCV from Adapted Scenario 12
        to perform comprehensive health monitoring including:
        - Body Condition Scoring (BCS) through anatomical feature detection
        - Weight estimation support via body dimension measurements
        - Health monitoring (coat texture, lesions, anemia, posture)
        - Behavioral analysis (lameness, activity, feeding, social interaction)
        - Individual identification (coat patterns, facial features, ear tags)
        - Environmental assessment (pen cleanliness, water availability, feed distribution)
        - Image enhancement for better analysis
        - Integrated health scoring and welfare reporting

        Args:
            color_image: RGB image as NumPy array
            depth_image: Depth image as NumPy array
            depth_frame: RealSense depth frame object
            timestamp: Current timestamp
        """
        if not self.classical_cv_analyzer:
            return

        # Check if it's time to analyze (based on interval)
        if timestamp - self.last_classical_cv_analysis_time < self.classical_cv_analysis_interval:
            return

        try:
            import pyrealsense2 as rs
            from datetime import datetime

            # Convert depth to meters
            depth_scale = self.pipeline.get_active_profile().get_device().first_depth_sensor().get_depth_scale()
            depth_map_meters = depth_image.astype(np.float32) * depth_scale

            # Update last analysis time
            self.last_classical_cv_analysis_time = timestamp

            # ================================================================
            # BCS Analysis (Scenarios 1.1-1.5)
            # ================================================================
            # Scenario 1.1: Spine keypoints detection
            spine_analysis = self.classical_cv_analyzer.detect_spine_keypoints(
                rgb_image=color_image,
                depth_map=depth_map_meters
            )
            self.classical_cv_analyzer.save_analysis_to_csv(spine_analysis, "bcs_spine_analysis.csv")

            # Scenario 1.2: Rib cage prominence
            rib_analysis = self.classical_cv_analyzer.detect_rib_cage_prominence(
                rgb_image=color_image
            )
            self.classical_cv_analyzer.save_analysis_to_csv(rib_analysis, "bcs_rib_analysis.csv")

            # Scenario 1.3: Hip bone prominence
            hip_analysis = self.classical_cv_analyzer.detect_hip_bone_prominence(
                rgb_image=color_image,
                depth_map=depth_map_meters
            )
            self.classical_cv_analyzer.save_analysis_to_csv(hip_analysis, "bcs_hip_analysis.csv")

            # Scenario 1.4: Body contour extraction
            contour_analysis = self.classical_cv_analyzer.extract_body_contour(
                rgb_image=color_image,
                foreground_mask=None
            )
            self.classical_cv_analyzer.save_analysis_to_csv(contour_analysis, "bcs_contour_analysis.csv")

            # Scenario 1.5: Body symmetry analysis
            symmetry_analysis = self.classical_cv_analyzer.analyze_body_symmetry(
                rgb_image=color_image
            )
            self.classical_cv_analyzer.save_analysis_to_csv(symmetry_analysis, "body_symmetry_analysis.csv")

            # ================================================================
            # Weight Estimation (Scenario 2.2)
            # ================================================================
            # Generate point cloud for comprehensive measurements
            pc = rs.pointcloud()
            points_data = pc.calculate(depth_frame)
            vertices = np.asanyarray(points_data.get_vertices())
            vertices_array = np.array([[v[0], v[1], v[2]] for v in vertices])
            valid_mask = ~np.isnan(vertices_array).any(axis=1) & \
                        ~np.isinf(vertices_array).any(axis=1) & \
                        (vertices_array != 0).any(axis=1)
            point_cloud = vertices_array[valid_mask]

            # Measure comprehensive body dimensions
            dimensions = self.classical_cv_analyzer.measure_comprehensive_dimensions(
                rgb_image=color_image,
                depth_map=depth_map_meters,
                point_cloud=point_cloud
            )
            self.classical_cv_analyzer.save_analysis_to_csv(dimensions, "body_dimensions.csv")

            # Estimate weight from dimensions
            weight_estimate = self.classical_cv_analyzer.estimate_weight_from_dimensions(dimensions)
            self.classical_cv_analyzer.save_analysis_to_csv(weight_estimate, "weight_estimate.csv")

            # ================================================================
            # Health Monitoring (Scenarios 3.1-3.5)
            # ================================================================
            # Scenario 3.1: Coat texture analysis
            coat_analysis = self.classical_cv_analyzer.analyze_coat_texture(
                rgb_image=color_image
            )
            self.classical_cv_analyzer.save_analysis_to_csv(coat_analysis, "coat_analysis.csv")

            # Scenario 3.2: Skin lesion detection
            lesion_detection = self.classical_cv_analyzer.detect_skin_lesions(
                rgb_image=color_image
            )
            self.classical_cv_analyzer.save_analysis_to_csv(lesion_detection, "lesion_detection.csv")

            # Scenario 3.3: Eye color analysis for anemia
            anemia_detection = self.classical_cv_analyzer.analyze_eye_color_for_anemia(
                rgb_image=color_image
            )
            self.classical_cv_analyzer.save_analysis_to_csv(anemia_detection, "anemia_detection.csv")

            # Scenario 3.4: Nasal discharge detection
            respiratory_assessment = self.classical_cv_analyzer.detect_nasal_discharge(
                rgb_image=color_image
            )
            self.classical_cv_analyzer.save_analysis_to_csv(respiratory_assessment, "respiratory_assessment.csv")

            # Scenario 3.5: Posture analysis
            posture_analysis = self.classical_cv_analyzer.analyze_posture(
                rgb_image=color_image,
                depth_map=depth_map_meters
            )
            self.classical_cv_analyzer.save_analysis_to_csv(posture_analysis, "posture_analysis.csv")

            # ================================================================
            # Environmental Analysis (Scenarios 6.1-6.3)
            # ================================================================
            # Scenario 6.1: Pen cleanliness assessment
            cleanliness = self.classical_cv_analyzer.assess_pen_cleanliness(
                rgb_image=color_image
            )
            self.classical_cv_analyzer.save_analysis_to_csv(cleanliness, "pen_cleanliness.csv")

            # Scenario 6.2: Water availability detection
            water_availability = self.classical_cv_analyzer.detect_water_availability(
                rgb_image=color_image,
                depth_map=depth_map_meters
            )
            self.classical_cv_analyzer.save_analysis_to_csv(water_availability, "water_availability.csv")

            # Scenario 6.3: Feed distribution analysis
            feed_distribution = self.classical_cv_analyzer.analyze_feed_distribution(
                rgb_image=color_image,
                depth_map=depth_map_meters
            )
            self.classical_cv_analyzer.save_analysis_to_csv(feed_distribution, "feed_distribution.csv")

            # ================================================================
            # Integrated Health Scoring (Scenario 8.1)
            # ================================================================
            # Calculate comprehensive health score from all analyses
            health_score = self.classical_cv_analyzer.calculate_comprehensive_health_score(
                bcs_analysis=spine_analysis,
                posture_analysis=posture_analysis,
                coat_analysis=coat_analysis,
                lesion_detection=lesion_detection,
                lameness_detection=None,  # Would need video frames
                activity_analysis=None     # Would need video frames
            )
            self.classical_cv_analyzer.save_analysis_to_csv(health_score, "comprehensive_health_score.csv")

            # Debug output (every analysis interval)
            if DEBUG_MODE:
                print(f"\n{'='*70}")
                print(f"Classical CV Analysis Report - Frame {self.processed_frame_count}")
                print(f"{'='*70}")

                print(f"\n[Body Condition Scoring]")
                print(f"  Spine BCS:       {spine_analysis.bcs_spine_score.value} (prominence: {spine_analysis.prominence_score:.2f})")
                print(f"  Rib BCS:         {rib_analysis.bcs_rib_score.value} (visible ribs: {rib_analysis.visible_rib_count})")
                print(f"  Hip BCS:         {hip_analysis.bcs_hip_score.value} (prominence: {hip_analysis.hip_bone_prominence:.2f})")
                print(f"  Shape BCS:       {contour_analysis.bcs_shape_score.value} (smoothness: {contour_analysis.smoothness:.2f})")

                print(f"\n[Weight Estimation]")
                print(f"  Body Length:     {dimensions.body_length_m:.2f} m")
                print(f"  Heart Girth:     {dimensions.heart_girth_m:.2f} m")
                print(f"  Height at Withers: {dimensions.height_at_withers_m:.2f} m")
                print(f"  Estimated Weight: {weight_estimate.estimated_weight_kg:.1f} kg (confidence: {weight_estimate.confidence:.2f})")

                print(f"\n[Health Monitoring]")
                print(f"  Coat Condition:  {coat_analysis.coat_condition_score}/5 (smoothness: {coat_analysis.smoothness_score:.2f})")
                print(f"  Lesions Detected: {lesion_detection.total_lesion_count} (vet attention: {lesion_detection.requires_vet_attention})")
                print(f"  Anemia Risk:     {anemia_detection.anemia_risk_score:.2f} (conjunctiva: {anemia_detection.conjunctiva_color})")
                print(f"  Respiratory Risk: {respiratory_assessment.respiratory_illness_risk:.2f} (discharge: {respiratory_assessment.discharge_present})")
                print(f"  Posture:         {posture_analysis.abnormal_posture_type or 'Normal'} (illness likelihood: {posture_analysis.illness_likelihood:.2f})")

                if symmetry_analysis.asymmetry_index > 0.15:
                    print(f"  Body Asymmetry: {symmetry_analysis.asymmetry_index:.2f} (lameness risk: {symmetry_analysis.lameness_risk_score:.2f})")

                print(f"\n[Environmental Conditions]")
                print(f"  Pen Cleanliness: {cleanliness.cleanliness_score:.2f} (soiled: {cleanliness.soiled_area_percentage:.1f}%)")
                print(f"  Water Available: {water_availability.water_present} (level: {water_availability.water_level_cm:.1f} cm)")
                print(f"  Feed Uniformity: {feed_distribution.uniformity_score:.2f} (empty sections: {feed_distribution.empty_sections_count})")

                print(f"\n[Comprehensive Health Score]")
                print(f"  Overall Score:   {health_score.overall_score:.1f}/100 ({health_score.health_status.value})")
                print(f"  Vet Priority:    {health_score.veterinary_priority}")
                print(f"  Trend:           {health_score.trend}")
                if health_score.primary_concerns:
                    print(f"  Primary Concerns: {', '.join(health_score.primary_concerns)}")

                print(f"{'='*70}\n")

        except Exception as e:
            if DEBUG_MODE:
                print(f"Error processing classical CV analysis: {e}")
                import traceback
                traceback.print_exc()

    def _check_disk_space(self) -> Tuple[float, bool]:
        """
        Check available disk space.

        Returns:
            Tuple of (available_gb, should_stop)
        """
        disk_stat = shutil.disk_usage(self.output_dir)
        available_gb = disk_stat.free / (1024 ** 3)

        if available_gb < self.disk_stop_gb:
            if DEBUG_MODE:
                print(f"Disk space critically low: {available_gb:.2f} GB. Stopping recording.")
            return available_gb, True
        elif available_gb < self.disk_warning_gb:
            if DEBUG_MODE:
                print(f"Disk space warning: {available_gb:.2f} GB remaining")

        return available_gb, False

    def _cleanup_old_recordings(self):
        """
        Automatically cleanup old recordings when disk usage exceeds threshold.
        """
        disk_stat = shutil.disk_usage(self.output_dir)
        disk_usage = disk_stat.used / disk_stat.total

        if disk_usage > self.auto_cleanup_threshold:
            # Find all recording files
            video_files = sorted(
                self.output_dir.glob("*.mp4") if self.codec != 'bag' else self.output_dir.glob("*.bag"),
                key=lambda p: p.stat().st_mtime
            )

            # Delete oldest files until usage is below threshold
            for old_file in video_files:
                if disk_usage <= self.auto_cleanup_threshold:
                    break

                file_size = old_file.stat().st_size
                old_file.unlink()

                if DEBUG_MODE:
                    print(f"Deleted old recording: {old_file.name}")

                # Recalculate disk usage
                disk_stat = shutil.disk_usage(self.output_dir)
                disk_usage = disk_stat.used / disk_stat.total

    def _save_session_metadata(self):
        """Save recording session metadata to JSON file."""
        if not self.save_metadata:
            return

        self.session_metadata.update({
            'end_time': datetime.now().isoformat(),
            'total_frames': self.processed_frame_count,
            'total_duration': time.time() - self.start_timestamp if self.start_timestamp else 0,
            'settings': {
                'fps': self.fps,
                'width': self.width,
                'height': self.height,
                'codec': self.codec,
                'record_rgb': self.record_rgb,
                'record_depth': self.record_depth,
                'record_ir': self.record_ir,
                'record_imu': self.record_imu,
                'camera_info': {
                    'name': self.camera_info.name,
                    'index': self.camera_info.index,
                    'serial_number': self.camera_info.serial_number,
                },
            },
        })

        metadata_filename = self._generate_filename("metadata", "json")
        metadata_path = self.output_dir / metadata_filename

        with open(metadata_path, 'w') as f:
            json.dump(self.session_metadata, f, indent=2)

        if DEBUG_MODE:
            print(f"Saved metadata: {metadata_path}")

    def _should_split_file(self) -> bool:
        """
        Check if file should be split based on split mode.

        Returns:
            Boolean indicating if split is needed
        """
        if self.split_mode is None:
            return False

        current_time = time.time()

        if self.split_mode == 'time':
            # Time-based splitting
            if self.last_split_time == 0:
                self.last_split_time = current_time
                return False

            elapsed_minutes = (current_time - self.last_split_time) / 60
            if elapsed_minutes >= self.split_interval_minutes:
                self.last_split_time = current_time
                return True

        elif self.split_mode == 'size':
            # Size-based splitting (approximate)
            # Estimate file size based on frame count and codec
            estimated_size_mb = self.current_file_size / (1024 * 1024)
            if estimated_size_mb >= self.split_size_mb:
                return True

        return False

    def _split_recording(self):
        """
        Split recording to a new file.
        Closes current writers and creates new ones with incremented index.
        """
        if DEBUG_MODE:
            print(f"Splitting recording to new file (index {self.current_file_index + 1})")

        # Close current writers
        if self.rgb_writer:
            self.rgb_writer.release()
        # No depth_writer to release (using .npz individual frames)
        if self.ir_left_writer:
            self.ir_left_writer.release()
        if self.ir_right_writer:
            self.ir_right_writer.release()

        # Increment file index
        self.current_file_index += 1
        self.current_file_size = 0

        # Create new writers
        self._create_video_writers()

    def stop_recording(self):
        """Stop recording and cleanup resources."""
        self.is_recording = False
        if self.recording_thread:
            self.recording_thread.join(timeout=5.0)

    def _cleanup(self):
        """Release all resources."""
        # Save metadata before cleanup
        self._save_session_metadata()

        # Release video writers
        if self.rgb_writer:
            self.rgb_writer.release()
        # No depth_writer to release (using .npz individual frames)
        if self.ir_left_writer:
            self.ir_left_writer.release()
        if self.ir_right_writer:
            self.ir_right_writer.release()

        # Close .bag recorder
        if self.bag_recorder:
            self.bag_recorder = None

        # Close IMU file
        if self.imu_file:
            self.imu_file.close()

        # Close timestamp log file
        if self.timestamp_log_file:
            self.timestamp_log_file.close()

        # Print temporal validation summary if enabled
        if self.enable_temporal_validation and DEBUG_MODE:
            print(f"RealSense #{self.camera_info.index} Temporal Validation Summary:")
            print(f"  Total frames captured: {self.frame_count}")
            print(f"  Processed frames: {self.processed_frame_count}")
            if self.enable_frame_skipping:
                print(f"  Frames skipped: {self.frame_count - self.processed_frame_count}")
            print(f"  Timestamp violations: {self.timestamp_violations}")
            if self.timestamp_violations == 0:
                print(f"  All timestamps monotonically increasing")

        # Print recording summary
        if DEBUG_MODE:
            print(f"RealSense #{self.camera_info.index} Recording Summary:")
            print(f"  Codec: {self.codec}")
            print(f"  Total files created: {self.current_file_index + 1}")
            if self.split_mode:
                print(f"  Split mode: {self.split_mode}")

        # Stop pipeline
        if self.pipeline:
            self.pipeline.stop()

    def __del__(self):
        """Destructor to ensure cleanup."""
        self._cleanup()
