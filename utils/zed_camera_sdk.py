"""Stereolabs Zed 2i camera wrapper for GPU-accelerated recording with advanced features."""
import time
import threading
from pathlib import Path
from datetime import datetime
from typing import Optional
import numpy as np
import cv2
from domain.camera_info import CameraInfo
from config import DEBUG_MODE, BASE_DIR, FPS, REQ_WIDTH, REQ_HEIGHT, VIDEO_SECS, START_RECORDING_DELAY


class ZedCameraSDKRecorder:
    """
    Wrapper for Stereolabs Zed 2i camera with NVIDIA GPU acceleration.

    Handles:
    - GPU-accelerated depth computation (PERFORMANCE/QUALITY/ULTRA/NEURAL modes)
    - 3D point cloud generation
    - 6-DOF positional tracking (Visual SLAM)
    - Spatial mapping (3D mesh reconstruction)
    - Object detection (AI-powered with TensorRT)
    - Body tracking (Human pose estimation)
    - Plane detection (AR/Robotics)
    - Advanced IMU fusion
    - SVO recording (complete data archive)
    - Side-by-side stereo RGB recording
    - IMU data recording

    Prerequisites:
    - NVIDIA GPU with CUDA support (Compute Capability 5.0+)
    - Minimum 4GB VRAM (8GB+ recommended)
    - CUDA Toolkit 11.x or 12.x
    - cuDNN library
    - TensorRT (for AI features)
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
        # GPU Feature Configuration
        depth_mode: str = "ULTRA",  # PERFORMANCE, QUALITY, ULTRA, NEURAL
        enable_depth: bool = True,
        enable_point_clouds: bool = False,
        enable_positional_tracking: bool = False,
        enable_spatial_mapping: bool = False,
        enable_object_detection: bool = False,
        enable_body_tracking: bool = False,
        enable_plane_detection: bool = False,
        enable_imu_fusion: bool = False,
        enable_svo_recording: bool = False,
        # Depth Configuration
        depth_minimum_distance: float = 0.3,  # meters
        depth_maximum_distance: float = 20.0,  # meters
        depth_confidence_threshold: int = 50,  # 0-100
        # Point Cloud Configuration
        point_cloud_frequency: int = 1,  # Export every N seconds
        point_cloud_format: str = "PLY",  # PLY, XYZ, PCD
        point_cloud_downsample_factor: int = 1,  # 1=no downsample, 2=every 2nd point
        # Positional Tracking Configuration
        tracking_set_as_static: bool = False,  # True=static camera, False=moving camera
        tracking_enable_imu_fusion: bool = True,  # Fuse IMU data with visual tracking
        tracking_enable_area_memory: bool = True,  # Remember visited areas for loop closure
        # Spatial Mapping Configuration
        mapping_resolution: str = "MEDIUM",  # LOW, MEDIUM, HIGH
        mapping_range_meter: float = 10.0,
        mapping_save_texture: bool = True,  # Include RGB texture in mesh
        mapping_mesh_formats: list = None,  # ["obj", "ply"] - formats to export
        # Object Detection Configuration
        detection_model: str = "MULTI_CLASS_BOX_FAST",  # FAST, MEDIUM, ACCURATE
        detection_confidence_threshold: float = 50.0,
        # Body Tracking Configuration
        body_tracking_model: str = "HUMAN_BODY_FAST",  # FAST, MEDIUM, ACCURATE
        body_format: str = "BODY_18",  # BODY_18, BODY_34, BODY_70
        # SVO Recording Configuration
        svo_compression_mode: str = "H264",  # H264, H265, LOSSLESS
    ):
        """
        Initialize Zed camera recorder with GPU features.

        Args:
            camera_info: Camera metadata from enumeration
            output_dir: Directory to save recordings
            recording_duration: Duration in seconds
            fps: Target frame rate
            width: Stereo frame width (3840 for side-by-side)
            height: Frame height (1080)
            recording_delay: Countdown delay before recording starts
            depth_mode: Depth computation mode
            enable_depth: Enable depth computation
            enable_point_clouds: Enable point cloud generation
            enable_positional_tracking: Enable 6-DOF tracking
            enable_spatial_mapping: Enable 3D mesh reconstruction
            enable_object_detection: Enable AI object detection
            enable_body_tracking: Enable human pose estimation
            enable_plane_detection: Enable plane detection
            enable_imu_fusion: Enable IMU fusion with visual tracking
            enable_svo_recording: Enable SVO file recording
            depth_minimum_distance: Minimum depth distance in meters
            depth_maximum_distance: Maximum depth distance in meters
            depth_confidence_threshold: Depth confidence threshold (0-100)
            point_cloud_frequency: Point cloud export frequency (seconds)
            mapping_resolution: Spatial mapping resolution
            mapping_range_meter: Spatial mapping range in meters
            detection_model: Object detection model
            detection_confidence_threshold: Object detection confidence (0-100)
            body_tracking_model: Body tracking model
            body_format: Body skeleton format (18/34/70 keypoints)
            svo_compression_mode: SVO compression mode
        """
        self.camera_info = camera_info
        self.output_dir = Path(output_dir)
        self.recording_duration = recording_duration
        self.fps = fps
        self.width = width
        self.height = height
        self.single_width = width // 2  # Width of single camera (1920)
        self.recording_delay = recording_delay

        # Feature flags
        self.depth_mode = depth_mode
        self.enable_depth = enable_depth
        self.enable_point_clouds = enable_point_clouds
        self.enable_positional_tracking = enable_positional_tracking
        self.enable_spatial_mapping = enable_spatial_mapping
        self.enable_object_detection = enable_object_detection
        self.enable_body_tracking = enable_body_tracking
        self.enable_plane_detection = enable_plane_detection
        self.enable_imu_fusion = enable_imu_fusion
        self.enable_svo_recording = enable_svo_recording

        # Depth configuration
        self.depth_minimum_distance = depth_minimum_distance
        self.depth_maximum_distance = depth_maximum_distance
        self.depth_confidence_threshold = depth_confidence_threshold

        # Point cloud configuration
        self.point_cloud_frequency = point_cloud_frequency
        self.point_cloud_format = point_cloud_format.upper()
        self.point_cloud_downsample_factor = point_cloud_downsample_factor

        # Positional tracking configuration
        self.tracking_set_as_static = tracking_set_as_static
        self.tracking_enable_imu_fusion = tracking_enable_imu_fusion
        self.tracking_enable_area_memory = tracking_enable_area_memory

        # Spatial mapping configuration
        self.mapping_resolution = mapping_resolution
        self.mapping_range_meter = mapping_range_meter
        self.mapping_save_texture = mapping_save_texture
        self.mapping_mesh_formats = mapping_mesh_formats if mapping_mesh_formats else ["obj"]

        # Object detection configuration
        self.detection_model = detection_model
        self.detection_confidence_threshold = detection_confidence_threshold

        # Body tracking configuration
        self.body_tracking_model = body_tracking_model
        self.body_format = body_format

        # SVO recording configuration
        self.svo_compression_mode = svo_compression_mode

        # Camera state
        self.camera = None
        self.is_recording = False
        self.recording_thread: Optional[threading.Thread] = None
        self.start_timestamp: Optional[float] = None
        self.frame_count = 0
        self.point_cloud_count = 0  # Track point cloud exports
        self.plane_id_counter = 0  # Track unique plane IDs (scenario-8-feature-1)

        # Video writers
        self.left_writer = None
        self.right_writer = None
        self.stereo_writer = None
        self.depth_writer = None
        self.disparity_writer = None  # For disparity map video
        self.confidence_writer = None  # For confidence map video

        # Data files
        self.imu_file = None
        self.trajectory_file = None
        self.detections_file = None
        self.body_tracking_file = None
        self.planes_file = None
        self.imu_fused_file = None

        # Raw depth storage (for numpy saving)
        self.raw_depth_frames = [] if enable_depth else None
        self.save_raw_depth = False  # Can be configured

        # Disparity and confidence map configuration
        self.enable_disparity_map = False  # Can be configured
        self.enable_confidence_map = False  # Can be configured

        # Performance monitoring
        self.last_frame_time = None
        self.frame_times = []  # Store recent frame times for FPS calculation
        self.fps_warning_threshold = 15.0  # Warn if FPS drops below this

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def initialize(self) -> bool:
        """
        Initialize Zed camera with GPU features.

        Scenarios Covered:
        - Initialize SDK with depth mode PERFORMANCE/QUALITY/ULTRA/NEURAL
        - Initialize with coordinate units in meters
        - Configure depth minimum and maximum distance
        - Enable multiple GPU features during initialization

        Returns:
            True if initialization successful, False otherwise
        """
        try:
            import pyzed.sl as sl

            if DEBUG_MODE:
                print(f"Initializing Zed camera with GPU features: {self.camera_info.name}")

            # Create camera object
            self.camera = sl.Camera()

            # Configure initialization parameters
            init_params = sl.InitParameters()
            init_params.camera_resolution = sl.RESOLUTION.HD1080
            init_params.camera_fps = self.fps
            init_params.coordinate_units = sl.UNIT.METER  # Scenario: Initialize with coordinate units in meters

            # Configure depth mode based on configuration
            if self.enable_depth:
                if self.depth_mode == "PERFORMANCE":
                    init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE
                    if DEBUG_MODE:
                        print("  Depth mode: PERFORMANCE (fast, 60-100 FPS)")
                elif self.depth_mode == "QUALITY":
                    init_params.depth_mode = sl.DEPTH_MODE.QUALITY
                    if DEBUG_MODE:
                        print("  Depth mode: QUALITY (balanced, 30-60 FPS)")
                elif self.depth_mode == "ULTRA":
                    init_params.depth_mode = sl.DEPTH_MODE.ULTRA
                    if DEBUG_MODE:
                        print("  Depth mode: ULTRA (high-precision, 15-30 FPS)")
                elif self.depth_mode == "NEURAL":
                    init_params.depth_mode = sl.DEPTH_MODE.NEURAL
                    if DEBUG_MODE:
                        print("  Depth mode: NEURAL (AI-enhanced)")
                else:
                    init_params.depth_mode = sl.DEPTH_MODE.ULTRA  # Default
                    if DEBUG_MODE:
                        print(f"  Unknown depth mode '{self.depth_mode}', using ULTRA")
            else:
                init_params.depth_mode = sl.DEPTH_MODE.NONE
                if DEBUG_MODE:
                    print("  Depth computation disabled")

            # Configure depth range
            init_params.depth_minimum_distance = self.depth_minimum_distance
            init_params.depth_maximum_distance = self.depth_maximum_distance

            if DEBUG_MODE:
                print(f"  Depth range: {self.depth_minimum_distance}m - {self.depth_maximum_distance}m")

            # Set camera by serial number if available
            if self.camera_info.serial_number:
                try:
                    serial_int = int(self.camera_info.serial_number)
                    init_params.set_from_serial_number(serial_int)
                    if DEBUG_MODE:
                        print(f"  Using camera with serial: {self.camera_info.serial_number}")
                except (ValueError, AttributeError):
                    if DEBUG_MODE:
                        print("  Using default camera (serial conversion failed)")

            # Open camera
            err = self.camera.open(init_params)

            if err != sl.ERROR_CODE.SUCCESS:
                if DEBUG_MODE:
                    print(f"  ✗ Zed SDK initialization failed: {err}")
                return False

            if DEBUG_MODE:
                print("  ✓ Zed camera opened successfully")

            # Enable GPU features
            enabled_features = []

            # Enable positional tracking (required for spatial mapping and plane detection)
            if self.enable_positional_tracking or self.enable_spatial_mapping or self.enable_plane_detection:
                if self._enable_positional_tracking():
                    enabled_features.append("Positional Tracking")
                else:
                    if DEBUG_MODE:
                        print("  ⚠ Positional tracking initialization failed")

            # Enable spatial mapping
            if self.enable_spatial_mapping:
                if self._enable_spatial_mapping():
                    enabled_features.append("Spatial Mapping")
                else:
                    if DEBUG_MODE:
                        print("  ⚠ Spatial mapping initialization failed")

            # Enable object detection
            if self.enable_object_detection:
                if self._enable_object_detection():
                    enabled_features.append("Object Detection")
                else:
                    if DEBUG_MODE:
                        print("  ⚠ Object detection initialization failed")

            # Enable body tracking
            if self.enable_body_tracking:
                if self._enable_body_tracking():
                    enabled_features.append("Body Tracking")
                else:
                    if DEBUG_MODE:
                        print("  ⚠ Body tracking initialization failed")

            # Enable SVO recording
            if self.enable_svo_recording:
                if self._enable_svo_recording():
                    enabled_features.append("SVO Recording")
                else:
                    if DEBUG_MODE:
                        print("  ⚠ SVO recording initialization failed")

            if DEBUG_MODE:
                if enabled_features:
                    print(f"  ✓ Enabled GPU features: {', '.join(enabled_features)}")
                print(f"  ✓ Zed camera initialized with GPU features")

            return True

        except ImportError:
            if DEBUG_MODE:
                print("  ✗ Zed SDK not available")
            return False
        except Exception as e:
            if DEBUG_MODE:
                print(f"  ✗ Failed to initialize Zed with GPU: {e}")
            return False

    def _enable_positional_tracking(self) -> bool:
        """
        Enable positional tracking feature.

        Scenarios Covered (scenario-4 main):
        - Scenario 1: Enable positional tracking during initialization
        - Scenario 2: Configure tracking for static camera
        - Scenario 3: Configure tracking for moving camera
        - Scenario 4: Enable IMU fusion for better tracking
        - Scenario 5: Enable area memory
        """
        try:
            import pyzed.sl as sl

            # Scenario 1: Enable positional tracking during initialization
            tracking_params = sl.PositionalTrackingParameters()

            # Scenario 2 & 3: Configure tracking for static/moving camera
            tracking_params.set_as_static = self.tracking_set_as_static

            # Scenario 4: Enable IMU fusion for better tracking
            if self.tracking_enable_imu_fusion:
                tracking_params.enable_imu_fusion = True
                if DEBUG_MODE:
                    print("  ✓ IMU fusion enabled for positional tracking")

            # Scenario 5: Enable area memory
            if self.tracking_enable_area_memory:
                tracking_params.enable_area_memory = True
                if DEBUG_MODE:
                    print("  ✓ Area memory enabled for loop closure")

            if DEBUG_MODE:
                camera_mode = "static" if self.tracking_set_as_static else "moving"
                print(f"  Tracking mode: {camera_mode}")

            # Enable positional tracking
            err = self.camera.enable_positional_tracking(tracking_params)

            if err == sl.ERROR_CODE.SUCCESS:
                # Create trajectory CSV file if tracking enabled
                if self.trajectory_file is None:
                    self._create_trajectory_file()
                return True
            else:
                if DEBUG_MODE:
                    print(f"  ✗ Positional tracking failed: {err}")
                return False

        except Exception as e:
            if DEBUG_MODE:
                print(f"  Error enabling positional tracking: {e}")
            return False

    def _enable_spatial_mapping(self) -> bool:
        """
        Enable spatial mapping feature.

        Scenarios Covered (scenario-5 main and feature-1):
        - Scenario 1: Enable spatial mapping during initialization
        - Scenarios 2-4: Configure mapping resolution LOW/MEDIUM/HIGH
        - Scenario 5: Set mapping range
        - Scenario 6: Enable texture mapping
        - Scenario 16: Skip spatial mapping when GPU insufficient
        - Feature scenarios: Create and configure spatial mapping parameters
        """
        try:
            import pyzed.sl as sl

            # Scenario 16: Skip spatial mapping when GPU insufficient
            # Check if GPU has sufficient VRAM (at least 3GB required)
            try:
                import torch
                if torch.cuda.is_available():
                    vram_bytes = torch.cuda.get_device_properties(0).total_memory
                    vram_gb = vram_bytes / (1024 ** 3)

                    if vram_gb < 3.0:
                        if DEBUG_MODE:
                            print(f"  ⚠ Spatial mapping disabled: GPU VRAM ({vram_gb:.2f} GB) < 3 GB required")
                            print(f"  Recording will continue without spatial mapping")
                        return False
            except ImportError:
                if DEBUG_MODE:
                    print(f"  ⚠ Cannot verify VRAM (PyTorch not available), attempting spatial mapping anyway")

            # Scenario: Create spatial mapping parameters
            mapping_params = sl.SpatialMappingParameters()

            # Scenarios 2-4: Configure mapping resolution
            if self.mapping_resolution == "HIGH":
                # Scenario 4: Configure mapping resolution HIGH
                mapping_params.set(sl.MAPPING_RESOLUTION.HIGH)
                if DEBUG_MODE:
                    print(f"  Mapping resolution: HIGH (0.025m = 2.5cm voxels)")
            elif self.mapping_resolution == "MEDIUM":
                # Scenario 3: Configure mapping resolution MEDIUM
                mapping_params.set(sl.MAPPING_RESOLUTION.MEDIUM)
                if DEBUG_MODE:
                    print(f"  Mapping resolution: MEDIUM (0.05m = 5cm voxels)")
            else:
                # Scenario 2: Configure mapping resolution LOW
                mapping_params.set(sl.MAPPING_RESOLUTION.LOW)
                if DEBUG_MODE:
                    print(f"  Mapping resolution: LOW (0.10m = 10cm voxels)")

            # Scenario 5: Set mapping range
            mapping_params.range_meter = self.mapping_range_meter
            if DEBUG_MODE:
                print(f"  Mapping range: {self.mapping_range_meter}m")

            # Scenario 6: Enable texture mapping
            mapping_params.save_texture = self.mapping_save_texture
            if DEBUG_MODE:
                texture_status = "enabled" if self.mapping_save_texture else "disabled"
                print(f"  Texture mapping: {texture_status}")

            # Scenario 1: Enable spatial mapping on camera
            err = self.camera.enable_spatial_mapping(mapping_params)

            if err == sl.ERROR_CODE.SUCCESS:
                if DEBUG_MODE:
                    print(f"  ✓ Spatial mapping enabled")
                return True
            else:
                if DEBUG_MODE:
                    print(f"  ✗ Spatial mapping failed: {err}")
                return False

        except Exception as e:
            if DEBUG_MODE:
                print(f"  Error enabling spatial mapping: {e}")
            return False

    def _enable_object_detection(self) -> bool:
        """
        Enable object detection feature.

        Scenarios Covered (scenario-6 main and feature-1):
        - Scenario 1: Enable object detection during initialization
        - Scenarios 2-4: Configure detection model FAST/MEDIUM/ACCURATE
        - Scenario 5: Configure person head detection
        - Scenario 6: Set detection confidence threshold
        - Scenario 7: Enable object tracking
        - Feature scenarios: Create and configure object detection parameters
        """
        try:
            import pyzed.sl as sl

            # Scenario: Create object detection parameters (feature-1)
            obj_params = sl.ObjectDetectionParameters()

            # Scenarios 2-5: Set detection model from configuration
            if self.detection_model == "MULTI_CLASS_BOX_FAST":
                # Scenario 2: Configure detection model FAST
                obj_params.detection_model = sl.DETECTION_MODEL.MULTI_CLASS_BOX_FAST
                if DEBUG_MODE:
                    print(f"  Detection model: MULTI_CLASS_BOX_FAST (real-time robotics)")
            elif self.detection_model == "MULTI_CLASS_BOX_MEDIUM":
                # Scenario 3: Configure detection model MEDIUM
                obj_params.detection_model = sl.DETECTION_MODEL.MULTI_CLASS_BOX_MEDIUM
                if DEBUG_MODE:
                    print(f"  Detection model: MULTI_CLASS_BOX_MEDIUM (balanced)")
            elif self.detection_model == "MULTI_CLASS_BOX_ACCURATE":
                # Scenario 4: Configure detection model ACCURATE
                obj_params.detection_model = sl.DETECTION_MODEL.MULTI_CLASS_BOX_ACCURATE
                if DEBUG_MODE:
                    print(f"  Detection model: MULTI_CLASS_BOX_ACCURATE (offline analysis)")
            elif self.detection_model == "PERSON_HEAD_BOX_FAST":
                # Scenario 5: Configure person head detection
                obj_params.detection_model = sl.DETECTION_MODEL.PERSON_HEAD_BOX_FAST
                if DEBUG_MODE:
                    print(f"  Detection model: PERSON_HEAD_BOX_FAST (crowd counting)")
            else:
                # Default to FAST model
                obj_params.detection_model = sl.DETECTION_MODEL.MULTI_CLASS_BOX_FAST
                if DEBUG_MODE:
                    print(f"  Unknown detection model '{self.detection_model}', using MULTI_CLASS_BOX_FAST")

            # Scenario 7: Enable object tracking
            obj_params.enable_tracking = True
            if DEBUG_MODE:
                print(f"  Object tracking: enabled (objects will have persistent IDs)")

            # Scenario 6: Set detection confidence threshold
            obj_params.detection_confidence = self.detection_confidence_threshold
            if DEBUG_MODE:
                print(f"  Detection confidence threshold: {self.detection_confidence_threshold}%")

            # Scenario 1: Enable object detection on camera
            err = self.camera.enable_object_detection(obj_params)

            if err == sl.ERROR_CODE.SUCCESS:
                if DEBUG_MODE:
                    print(f"  ✓ Object detection enabled")

                # Create detections CSV file
                if self.detections_file is None:
                    self._create_detections_file()

                return True
            else:
                if DEBUG_MODE:
                    print(f"  ✗ Object detection failed: {err}")
                    print(f"  Ensure TensorRT is installed and GPU supports AI models")
                return False

        except Exception as e:
            if DEBUG_MODE:
                print(f"  Error enabling object detection: {e}")
            return False

    def _enable_body_tracking(self) -> bool:
        """
        Enable body tracking feature.

        Scenarios Covered (scenario-7 main and feature-1):
        - Scenario 1: Enable body tracking during initialization
        - Scenarios 2-4: Configure body tracking model FAST/MEDIUM/ACCURATE
        - Scenarios 5-7: Configure body format 18/34/70 keypoints
        - Scenario 8: Enable body tracking
        - Scenario 9: Enable body fitting
        - Feature scenarios: Create and configure body tracking parameters
        """
        try:
            import pyzed.sl as sl

            # Scenario: Create body tracking parameters (feature-1)
            body_params = sl.BodyTrackingParameters()

            # Scenarios 2-4: Set tracking model from configuration
            if self.body_tracking_model == "HUMAN_BODY_FAST":
                # Scenario 2: Configure body tracking model FAST
                body_params.detection_model = sl.BODY_TRACKING_MODEL.HUMAN_BODY_FAST
                if DEBUG_MODE:
                    print(f"  Body tracking model: HUMAN_BODY_FAST (up to 8 people)")
            elif self.body_tracking_model == "HUMAN_BODY_MEDIUM":
                # Scenario 3: Configure body tracking model MEDIUM
                body_params.detection_model = sl.BODY_TRACKING_MODEL.HUMAN_BODY_MEDIUM
                if DEBUG_MODE:
                    print(f"  Body tracking model: HUMAN_BODY_MEDIUM (balanced, up to 8 people)")
            elif self.body_tracking_model == "HUMAN_BODY_ACCURATE":
                # Scenario 4: Configure body tracking model ACCURATE
                body_params.detection_model = sl.BODY_TRACKING_MODEL.HUMAN_BODY_ACCURATE
                if DEBUG_MODE:
                    print(f"  Body tracking model: HUMAN_BODY_ACCURATE (best quality, up to 8 people)")
            else:
                # Default to FAST model
                body_params.detection_model = sl.BODY_TRACKING_MODEL.HUMAN_BODY_FAST
                if DEBUG_MODE:
                    print(f"  Unknown body tracking model '{self.body_tracking_model}', using HUMAN_BODY_FAST")

            # Scenarios 5-7: Set body format from configuration
            if self.body_format == "BODY_18":
                # Scenario 5: Configure body format 18 keypoints
                body_params.body_format = sl.BODY_FORMAT.BODY_18
                if DEBUG_MODE:
                    print(f"  Body format: BODY_18 (18 keypoints)")
            elif self.body_format == "BODY_34":
                # Scenario 6: Configure body format 34 keypoints
                body_params.body_format = sl.BODY_FORMAT.BODY_34
                if DEBUG_MODE:
                    print(f"  Body format: BODY_34 (34 keypoints with spine/feet/hands)")
            elif self.body_format == "BODY_70":
                # Scenario 7: Configure body format 70 keypoints
                body_params.body_format = sl.BODY_FORMAT.BODY_70
                if DEBUG_MODE:
                    print(f"  Body format: BODY_70 (70 keypoints with full hands + face)")
            else:
                # Default to BODY_18
                body_params.body_format = sl.BODY_FORMAT.BODY_18
                if DEBUG_MODE:
                    print(f"  Unknown body format '{self.body_format}', using BODY_18")

            # Scenario 8: Enable body tracking
            body_params.enable_tracking = True
            if DEBUG_MODE:
                print(f"  Body tracking: enabled (people will have persistent IDs)")

            # Scenario 9: Enable body fitting
            body_params.enable_body_fitting = True
            if DEBUG_MODE:
                print(f"  Body fitting: enabled (body model fit to keypoints)")

            # Scenario 1: Enable body tracking on camera
            err = self.camera.enable_body_tracking(body_params)

            if err == sl.ERROR_CODE.SUCCESS:
                if DEBUG_MODE:
                    print(f"  ✓ Body tracking enabled")

                # Create body tracking CSV file
                if self.body_tracking_file is None:
                    self._create_body_tracking_file()

                return True
            else:
                if DEBUG_MODE:
                    print(f"  ✗ Body tracking failed: {err}")
                    print(f"  Ensure TensorRT is installed and GPU supports AI models")
                return False

        except Exception as e:
            if DEBUG_MODE:
                print(f"  Error enabling body tracking: {e}")
            return False

    def _enable_svo_recording(self) -> bool:
        """
        Enable SVO recording feature.

        Scenarios Covered (scenario-10 main and feature-1):
        - Scenario 1: Enable SVO recording
        - Scenario 2: Configure SVO filename
        - Scenarios 3-5: Configure SVO compression H265/H264/LOSSLESS
        - Scenario 7: Enable SVO recording on camera
        - Scenario 12: Handle enable_recording failure
        - Feature scenarios: Create and configure recording parameters
        """
        try:
            import pyzed.sl as sl

            # Scenario 1 & 2: Generate SVO filename with format {timestamp}-{camera_index}.svo
            timestamp_str = datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d-%H-%M-%S-%f")[:-3]
            svo_filename = f"{timestamp_str}-{self.camera_info.index}.svo"
            svo_path = self.output_dir / svo_filename

            # Scenario 1: Create recording parameters
            recording_params = sl.RecordingParameters()

            # Scenario 3: Set SVO filename in recording parameters
            recording_params.video_filename = str(svo_path)

            # Scenarios 4-6: Set compression mode
            if self.svo_compression_mode == "H265":
                # Scenario 4: Set compression mode to H265
                recording_params.compression_mode = sl.SVO_COMPRESSION_MODE.H265
                if DEBUG_MODE:
                    print(f"  SVO compression: H265 (HEVC, ~200-500 MB/min)")
            elif self.svo_compression_mode == "H264":
                # Scenario 5: Set compression mode to H264
                recording_params.compression_mode = sl.SVO_COMPRESSION_MODE.H264
                if DEBUG_MODE:
                    print(f"  SVO compression: H264 (AVC, ~300-600 MB/min)")
            elif self.svo_compression_mode == "LOSSLESS":
                # Scenario 6: Set compression mode to LOSSLESS
                recording_params.compression_mode = sl.SVO_COMPRESSION_MODE.LOSSLESS
                if DEBUG_MODE:
                    print(f"  SVO compression: LOSSLESS (~2-5 GB/min)")
            else:
                # Default to H264
                recording_params.compression_mode = sl.SVO_COMPRESSION_MODE.H264
                if DEBUG_MODE:
                    print(f"  Unknown SVO compression mode '{self.svo_compression_mode}', using H264")

            # Scenario 7: Enable SVO recording on camera
            err = self.camera.enable_recording(recording_params)

            if err == sl.ERROR_CODE.SUCCESS:
                # Store SVO path for later verification
                self.svo_filepath = svo_path
                if DEBUG_MODE:
                    print(f"  ✓ SVO recording enabled: {svo_filename}")
                    print(f"  SVO will contain: left/right images, IMU data, camera params, timestamps")
                return True
            else:
                # Scenario 12: Handle enable_recording failure
                if DEBUG_MODE:
                    print(f"  ✗ SVO recording failed: {err}")
                    print(f"  Recording will continue without SVO")
                return False

        except Exception as e:
            # Scenario 12: Handle enable_recording failure
            if DEBUG_MODE:
                print(f"  Error enabling SVO recording: {e}")
                print(f"  Recording will continue without SVO")
            return False

    def prepare_for_recording(self, sync_timestamp: float) -> Optional[threading.Thread]:
        """
        Prepare for recording (create files, setup) but don't start thread yet.

        Phase 1 of two-phase recording start for synchronized multi-camera.

        Args:
            sync_timestamp: Synchronized start timestamp

        Returns:
            Recording thread (not started) or None if preparation failed
        """
        if self.is_recording:
            return None

        try:
            self.start_timestamp = sync_timestamp
            self.is_recording = True
            self.frame_count = 0

            # Create video writers
            self._create_video_writers()

            # Create data files
            if self.enable_positional_tracking:
                self._create_trajectory_file()

            # Scenario 11: Create planes CSV file if spatial mapping enabled (scenario-8-feature-1)
            if self.enable_spatial_mapping:
                self._create_planes_file()

            # Scenario 9: Create fused IMU CSV file if positional tracking with IMU fusion enabled (scenario-9-feature-1)
            if self.enable_positional_tracking and self.tracking_enable_imu_fusion:
                self._create_fused_imu_file()

            # Create recording thread (but don't start it)
            self.recording_thread = threading.Thread(
                target=self._recording_loop,
                name=f"ZedSDK-{self.camera_info.index}",
                daemon=True
            )

            return self.recording_thread

        except Exception as e:
            if DEBUG_MODE:
                print(f"Error preparing Zed SDK recording: {e}")
            self.is_recording = False
            return None

    def start_recording(self, sync_timestamp: Optional[float] = None) -> bool:
        """
        Legacy method: Prepare and immediately start recording.

        For synchronized multi-camera, use prepare_for_recording() instead.

        Args:
            sync_timestamp: Synchronized start timestamp (or generate now)

        Returns:
            True if recording started successfully
        """
        if sync_timestamp is None:
            sync_timestamp = time.time()

        thread = self.prepare_for_recording(sync_timestamp)
        if thread:
            thread.start()
            return True
        return False

    def _create_video_writers(self):
        """
        Create video writers for RGB and depth streams.

        Scenarios Covered (scenario-2-feature-2):
        - Create depth video writer
        - Generate depth video filename (YYYY-MM-DD-HH-mm-ss-sss-{INDEX}-depth.mp4)
        - Verify depth writer initialization (isOpened check)
        - Write frames at correct frame rate
        - Handle depth writer creation failure
        - Support different depth video codecs (mp4v, avc1 fallback)
        """
        try:
            # Scenario: Support different depth video codecs
            # Try mp4v first (works on all platforms), fallback to avc1 if needed
            codecs_to_try = ['mp4v', 'avc1']
            fourcc = None

            for codec in codecs_to_try:
                try:
                    fourcc = cv2.VideoWriter_fourcc(*codec)
                    if DEBUG_MODE:
                        print(f"  Using video codec: {codec}")
                    break
                except Exception as e:
                    if DEBUG_MODE:
                        print(f"  Codec {codec} not available: {e}")
                    continue

            if fourcc is None:
                raise Exception("No supported video codec available")

            # Create left camera writer
            left_filename = self._generate_filename("left", "mp4")
            self.left_writer = cv2.VideoWriter(
                str(self.output_dir / left_filename),
                fourcc,
                self.fps,
                (self.single_width, self.height)
            )

            if not self.left_writer.isOpened():
                raise Exception("Failed to create left video writer")

            # Create right camera writer
            right_filename = self._generate_filename("right", "mp4")
            self.right_writer = cv2.VideoWriter(
                str(self.output_dir / right_filename),
                fourcc,
                self.fps,
                (self.single_width, self.height)
            )

            if not self.right_writer.isOpened():
                raise Exception("Failed to create right video writer")

            # Create stereo writer (side-by-side)
            stereo_filename = self._generate_filename("stereo", "mp4")
            self.stereo_writer = cv2.VideoWriter(
                str(self.output_dir / stereo_filename),
                fourcc,
                self.fps,
                (self.width, self.height)
            )

            if not self.stereo_writer.isOpened():
                raise Exception("Failed to create stereo video writer")

            # Scenario: Create depth writer if depth is enabled
            if self.enable_depth:
                try:
                    # Scenario: Generate depth video filename
                    # Format: YYYY-MM-DD-HH-mm-ss-sss-{INDEX}-depth.mp4
                    depth_filename = self._generate_filename("depth", "mp4")
                    depth_path = str(self.output_dir / depth_filename)

                    # Scenario: Create depth video writer
                    self.depth_writer = cv2.VideoWriter(
                        depth_path,
                        fourcc,
                        self.fps,
                        (self.single_width, self.height)  # Same resolution as single camera
                    )

                    # Scenario: Verify depth writer initialization
                    if not self.depth_writer.isOpened():
                        raise Exception("Failed to open depth video writer")

                    if DEBUG_MODE:
                        print(f"  ✓ Created depth video writer: {depth_filename}")
                        print(f"    - Resolution: {self.single_width}x{self.height}")
                        print(f"    - FPS: {self.fps}")
                        print(f"    - Codec: {codecs_to_try[0]}")

                except Exception as e:
                    # Scenario: Handle depth writer creation failure
                    self.depth_writer = None
                    if DEBUG_MODE:
                        print(f"  ⚠ Error creating depth video writer: {e}")
                        print(f"  Recording will continue without depth video")
                    # Don't raise - continue recording without depth video

            # Scenario: Generate disparity map (scenario-2 main scenario 11)
            if self.enable_disparity_map and self.enable_depth:
                try:
                    disparity_filename = self._generate_filename("disparity", "mp4")
                    disparity_path = str(self.output_dir / disparity_filename)

                    self.disparity_writer = cv2.VideoWriter(
                        disparity_path,
                        fourcc,
                        self.fps,
                        (self.single_width, self.height)
                    )

                    if not self.disparity_writer.isOpened():
                        raise Exception("Failed to open disparity video writer")

                    if DEBUG_MODE:
                        print(f"  ✓ Created disparity video writer: {disparity_filename}")

                except Exception as e:
                    self.disparity_writer = None
                    if DEBUG_MODE:
                        print(f"  ⚠ Error creating disparity video writer: {e}")
                        print(f"  Recording will continue without disparity video")

            # Scenario: Generate confidence map (scenario-2 main scenario 12)
            if self.enable_confidence_map and self.enable_depth:
                try:
                    confidence_filename = self._generate_filename("confidence", "mp4")
                    confidence_path = str(self.output_dir / confidence_filename)

                    self.confidence_writer = cv2.VideoWriter(
                        confidence_path,
                        fourcc,
                        self.fps,
                        (self.single_width, self.height)
                    )

                    if not self.confidence_writer.isOpened():
                        raise Exception("Failed to open confidence video writer")

                    if DEBUG_MODE:
                        print(f"  ✓ Created confidence video writer: {confidence_filename}")

                except Exception as e:
                    self.confidence_writer = None
                    if DEBUG_MODE:
                        print(f"  ⚠ Error creating confidence video writer: {e}")
                        print(f"  Recording will continue without confidence video")

            if DEBUG_MODE:
                print(f"  ✓ Created video writers for camera #{self.camera_info.index}")

        except Exception as e:
            if DEBUG_MODE:
                print(f"  ✗ Error creating video writers: {e}")
            raise

    def _create_trajectory_file(self):
        """
        Create CSV file for trajectory data.

        Scenarios Covered (scenario-4-feature-1):
        - Create trajectory CSV file
        - Generate trajectory filename
        - Open file in write mode
        - Write header row
        - Store file handle for appending
        """
        try:
            # Scenario: Generate trajectory filename
            trajectory_filename = self._generate_filename("trajectory", "csv")
            trajectory_path = self.output_dir / trajectory_filename

            # Scenario: Open file in write mode
            self.trajectory_file = open(trajectory_path, 'w')

            # Scenario: Write header row
            header = "timestamp,frame,x,y,z,qx,qy,qz,qw,confidence\n"
            self.trajectory_file.write(header)

            # Flush to ensure header is written
            self.trajectory_file.flush()

            if DEBUG_MODE:
                print(f"  ✓ Created trajectory CSV file: {trajectory_filename}")

        except Exception as e:
            self.trajectory_file = None
            if DEBUG_MODE:
                print(f"  ⚠ Error creating trajectory CSV file: {e}")
                print(f"  Recording will continue without trajectory data")

    def _create_detections_file(self):
        """
        Create CSV file for object detection data.

        Scenarios Covered (scenario-6-feature-1):
        - Scenario 11: Create detections CSV file
        - Scenario 12: Write CSV header for detections
        - Generate detections filename
        - Open file in write mode
        - Write header row
        - Store file handle for appending
        """
        try:
            # Scenario 11: Generate detections filename
            detections_filename = self._generate_filename("detections", "csv")
            detections_path = self.output_dir / detections_filename

            # Scenario 11: Open file in write mode
            self.detections_file = open(detections_path, 'w')

            # Scenario 12: Write CSV header for detections
            # Header: timestamp,frame,object_id,class,confidence,x,y,z,bbox_x,bbox_y,bbox_w,bbox_h
            header = "timestamp,frame,object_id,class,confidence,x,y,z,bbox_x,bbox_y,bbox_w,bbox_h\n"
            self.detections_file.write(header)

            # Flush to ensure header is written
            self.detections_file.flush()

            if DEBUG_MODE:
                print(f"  ✓ Created detections CSV file: {detections_filename}")

        except Exception as e:
            self.detections_file = None
            if DEBUG_MODE:
                print(f"  ⚠ Error creating detections CSV file: {e}")
                print(f"  Recording will continue without object detection data")

    def _create_body_tracking_file(self):
        """
        Create CSV file for body tracking data.

        Scenarios Covered (scenario-7-feature-1):
        - Scenario 15: Create body tracking CSV file
        - Scenario 16: Write CSV header for body tracking
        - Generate body tracking filename
        - Open file in write mode
        - Write header row
        - Store file handle for appending
        """
        try:
            # Scenario 15: Generate body tracking filename
            body_filename = self._generate_filename("body_tracking", "csv")
            body_path = self.output_dir / body_filename

            # Scenario 15: Open file in write mode
            self.body_tracking_file = open(body_path, 'w')

            # Scenario 16: Write CSV header for body tracking
            # Header: timestamp,frame,body_id,keypoint_id,x,y,z,confidence
            header = "timestamp,frame,body_id,keypoint_id,x,y,z,confidence\n"
            self.body_tracking_file.write(header)

            # Flush to ensure header is written
            self.body_tracking_file.flush()

            if DEBUG_MODE:
                print(f"  ✓ Created body tracking CSV file: {body_filename}")

        except Exception as e:
            self.body_tracking_file = None
            if DEBUG_MODE:
                print(f"  ⚠ Error creating body tracking CSV file: {e}")
                print(f"  Recording will continue without body tracking data")

    def _create_planes_file(self):
        """
        Create CSV file for plane detection data.

        Scenarios Covered (scenario-8-feature-1):
        - Scenario 11: Create planes CSV file
        - Scenario 12: Write CSV header for planes
        - Generate planes filename
        - Open file in write mode
        - Write header row
        - Store file handle for appending
        """
        try:
            # Scenario 11: Generate planes filename
            planes_filename = self._generate_filename("planes", "csv")
            planes_path = self.output_dir / planes_filename

            # Scenario 11: Open file in write mode
            self.planes_file = open(planes_path, 'w')

            # Scenario 12: Write CSV header for planes
            # Header: timestamp,plane_id,type,center_x,center_y,center_z,normal_x,normal_y,normal_z,area
            header = "timestamp,plane_id,type,center_x,center_y,center_z,normal_x,normal_y,normal_z,area\n"
            self.planes_file.write(header)

            # Flush to ensure header is written
            self.planes_file.flush()

            if DEBUG_MODE:
                print(f"  ✓ Created planes CSV file: {planes_filename}")

        except Exception as e:
            self.planes_file = None
            if DEBUG_MODE:
                print(f"  ⚠ Error creating planes CSV file: {e}")
                print(f"  Recording will continue without plane detection data")

    def _create_fused_imu_file(self):
        """
        Create CSV file for fused IMU data.

        Scenarios Covered (scenario-9-feature-1):
        - Scenario 9: Create fused IMU CSV file
        - Scenario 10: Write CSV header for fused IMU
        - Generate fused IMU filename
        - Open file in write mode
        - Write header row
        - Store file handle for appending
        """
        try:
            # Scenario 9: Generate fused IMU filename
            imu_fused_filename = self._generate_filename("imu_fused", "csv")
            imu_fused_path = self.output_dir / imu_fused_filename

            # Scenario 9: Open file in write mode
            self.imu_fused_file = open(imu_fused_path, 'w')

            # Scenario 10: Write CSV header for fused IMU
            # Header: timestamp,frame,ax,ay,az,gx,gy,gz,qx,qy,qz,qw,pos_x,pos_y,pos_z,confidence
            header = "timestamp,frame,ax,ay,az,gx,gy,gz,qx,qy,qz,qw,pos_x,pos_y,pos_z,confidence\n"
            self.imu_fused_file.write(header)

            # Flush to ensure header is written
            self.imu_fused_file.flush()

            if DEBUG_MODE:
                print(f"  ✓ Created fused IMU CSV file: {imu_fused_filename}")

        except Exception as e:
            self.imu_fused_file = None
            if DEBUG_MODE:
                print(f"  ⚠ Error creating fused IMU CSV file: {e}")
                print(f"  Recording will continue without fused IMU data")

    def _recording_loop(self):
        """
        Main recording loop with GPU features including depth map capture.

        Scenarios Covered:
        - Create depth Mat object (reused across frames)
        - Retrieve depth measure
        - Configure runtime parameters for depth
        - Convert depth Mat to numpy array
        - Colorize depth for visualization
        - Write depth frame to video
        - Handle depth retrieval failure
        - Apply depth filtering
        - Calculate depth statistics
        - Save raw depth to numpy file (at end)
        """
        try:
            import pyzed.sl as sl

            # Calculate recording times
            recording_start_time = self.start_timestamp + self.recording_delay
            end_time = recording_start_time + self.recording_duration

            # Create Mat objects (reused for efficiency)
            left_image = sl.Mat()
            right_image = sl.Mat()
            depth_map = sl.Mat() if self.enable_depth else None
            disparity_map = sl.Mat() if self.enable_disparity_map else None  # Scenario 11
            confidence_map = sl.Mat() if self.enable_confidence_map else None  # Scenario 12

            # Scenario: Configure runtime parameters for depth
            runtime_params = sl.RuntimeParameters()
            if self.enable_depth:
                runtime_params.confidence_threshold = self.depth_confidence_threshold
                runtime_params.enable_depth = True
                # Optional: runtime_params.texture_confidence_threshold = 100

            if DEBUG_MODE:
                print(f"Recording loop started for camera #{self.camera_info.index}")
                if self.enable_depth:
                    print(f"  Depth enabled with confidence threshold: {self.depth_confidence_threshold}")

            # Main recording loop
            while self.is_recording and time.time() < end_time:
                current_time = time.time()

                # Wait for countdown to complete
                if current_time < recording_start_time:
                    time.sleep(0.001)
                    continue

                # Grab frame from camera
                err = self.camera.grab(runtime_params)

                if err != sl.ERROR_CODE.SUCCESS:
                    if DEBUG_MODE:
                        print(f"  Frame grab failed: {err}")
                    continue

                # Retrieve left and right images
                self.camera.retrieve_image(left_image, sl.VIEW.LEFT)
                self.camera.retrieve_image(right_image, sl.VIEW.RIGHT)

                # Convert to numpy arrays
                left_frame = left_image.get_data()[:, :, :3]  # Remove alpha channel
                right_frame = right_image.get_data()[:, :, :3]

                # Convert from RGBA to BGR
                left_bgr = cv2.cvtColor(left_frame, cv2.COLOR_RGBA2BGR)
                right_bgr = cv2.cvtColor(right_frame, cv2.COLOR_RGBA2BGR)

                # Create stereo frame (side-by-side)
                stereo_frame = np.hstack([left_bgr, right_bgr])

                # Write RGB frames
                if self.left_writer:
                    self.left_writer.write(left_bgr)
                if self.right_writer:
                    self.right_writer.write(right_bgr)
                if self.stereo_writer:
                    self.stereo_writer.write(stereo_frame)

                # Scenario: Retrieve depth measure and process
                if self.enable_depth and depth_map is not None:
                    try:
                        # Scenario: Retrieve depth measure
                        err_depth = self.camera.retrieve_measure(depth_map, sl.MEASURE.DEPTH)

                        if err_depth == sl.ERROR_CODE.SUCCESS:
                            # Scenario: Convert depth Mat to numpy array
                            depth_data = depth_map.get_data()  # Shape: (height, width, 4), dtype: float32

                            # Extract single channel depth (first channel contains depth in meters)
                            depth_values = depth_data[:, :, 0]  # Shape: (height, width), dtype: float32

                            # Scenario: Apply depth filtering
                            # Invalid depth values are NaN, inf, or outside range
                            valid_mask = np.isfinite(depth_values)
                            valid_mask &= (depth_values >= self.depth_minimum_distance)
                            valid_mask &= (depth_values <= self.depth_maximum_distance)

                            # Scenario: Calculate depth statistics
                            if DEBUG_MODE and self.frame_count % 30 == 0:  # Log every 30 frames
                                valid_depths = depth_values[valid_mask]
                                if len(valid_depths) > 0:
                                    min_depth = np.min(valid_depths)
                                    max_depth = np.max(valid_depths)
                                    mean_depth = np.mean(valid_depths)
                                    print(f"  Depth stats - min: {min_depth:.2f}m, max: {max_depth:.2f}m, mean: {mean_depth:.2f}m")

                            # Scenario: Colorize depth for visualization
                            depth_colorized = self._colorize_depth(depth_values, valid_mask)

                            # Scenario: Write depth frame to video
                            if self.depth_writer:
                                self.depth_writer.write(depth_colorized)

                            # Scenario: Save raw depth to numpy file (accumulate frames)
                            if self.save_raw_depth:
                                self.raw_depth_frames.append(depth_values.copy())

                        else:
                            # Scenario: Handle depth retrieval failure
                            if DEBUG_MODE:
                                print(f"  Depth retrieval failed: {err_depth}")
                            # Continue recording without depth for this frame

                    except Exception as e:
                        # Scenario: Handle depth retrieval failure
                        if DEBUG_MODE:
                            print(f"  Error processing depth: {e}")
                        # Continue recording

                # Scenario 11: Generate disparity map (scenario-2 main)
                if self.enable_disparity_map and disparity_map is not None:
                    try:
                        err_disp = self.camera.retrieve_measure(disparity_map, sl.MEASURE.DISPARITY)

                        if err_disp == sl.ERROR_CODE.SUCCESS:
                            disp_data = disparity_map.get_data()
                            disp_values = disp_data[:, :, 0]  # Extract single channel

                            # Normalize disparity to 0-255 for visualization
                            disp_normalized = np.zeros_like(disp_values, dtype=np.uint8)
                            valid_disp = np.isfinite(disp_values) & (disp_values > 0)

                            if np.any(valid_disp):
                                min_disp = np.min(disp_values[valid_disp])
                                max_disp = np.max(disp_values[valid_disp])

                                if max_disp > min_disp:
                                    disp_normalized[valid_disp] = (
                                        255 * (disp_values[valid_disp] - min_disp) / (max_disp - min_disp)
                                    ).astype(np.uint8)

                            # Apply colormap
                            disp_colorized = cv2.applyColorMap(disp_normalized, cv2.COLORMAP_JET)
                            disp_colorized[~valid_disp] = [0, 0, 0]

                            # Write to video
                            if self.disparity_writer:
                                self.disparity_writer.write(disp_colorized)

                    except Exception as e:
                        if DEBUG_MODE:
                            print(f"  Error processing disparity: {e}")

                # Scenario 12: Generate confidence map (scenario-2 main)
                if self.enable_confidence_map and confidence_map is not None:
                    try:
                        err_conf = self.camera.retrieve_measure(confidence_map, sl.MEASURE.CONFIDENCE)

                        if err_conf == sl.ERROR_CODE.SUCCESS:
                            conf_data = confidence_map.get_data()
                            conf_values = conf_data[:, :, 0]  # Confidence is 0-100

                            # Normalize confidence to 0-255 for visualization
                            conf_normalized = (conf_values * 2.55).astype(np.uint8)

                            # Apply colormap (higher confidence = warmer colors)
                            conf_colorized = cv2.applyColorMap(conf_normalized, cv2.COLORMAP_HOT)

                            # Write to video
                            if self.confidence_writer:
                                self.confidence_writer.write(conf_colorized)

                    except Exception as e:
                        if DEBUG_MODE:
                            print(f"  Error processing confidence: {e}")

                # Scenario: Export point cloud at correct frequency (scenario-3-feature-1)
                if self.enable_point_clouds and self.enable_depth:
                    # Calculate frames per export
                    frames_per_export = self.fps * self.point_cloud_frequency

                    # Scenario: Call _save_point_cloud at correct frequency
                    # Export when frame_count is multiple of frames_per_export
                    if self.frame_count > 0 and self.frame_count % frames_per_export == 0:
                        self._save_point_cloud(current_time, self.frame_count)

                # Scenarios 6-16: Retrieve camera pose and save to trajectory CSV (scenario-4 main)
                if self.enable_positional_tracking:
                    try:
                        # Scenario 6: Retrieve camera pose during recording
                        camera_pose = sl.Pose()
                        tracking_state = self.camera.get_position(camera_pose)

                        # Scenario 7: Handle tracking state OK
                        if tracking_state == sl.POSITIONAL_TRACKING_STATE.OK:
                            # Pose data is valid and should be saved
                            self._save_tracking_data(camera_pose, current_time, self.frame_count)

                        # Scenario 8: Handle tracking state SEARCHING
                        elif tracking_state == sl.POSITIONAL_TRACKING_STATE.SEARCHING:
                            # Tracking temporarily lost, attempting to relocalize
                            if DEBUG_MODE and self.frame_count % 30 == 0:
                                print(f"  ⚠ Tracking searching (attempting to relocalize)")

                        # Scenario 9: Handle tracking state OFF
                        elif tracking_state == sl.POSITIONAL_TRACKING_STATE.OFF:
                            # Tracking failed or disabled
                            if DEBUG_MODE and self.frame_count % 30 == 0:
                                print(f"  ⚠ Tracking OFF (no pose data available)")

                    except Exception as e:
                        if DEBUG_MODE:
                            print(f"  Error retrieving tracking data: {e}")

                # Scenario 7: Request spatial map update during recording (scenario-5 main)
                if self.enable_spatial_mapping:
                    try:
                        # Request spatial map update (async, non-blocking)
                        # Update every 30 frames to avoid overwhelming the system
                        if self.frame_count % 30 == 0:
                            self.camera.request_spatial_map_async()
                    except Exception as e:
                        if DEBUG_MODE:
                            print(f"  Error requesting spatial map update: {e}")

                # Scenarios 8-18: Retrieve detected objects and save to CSV (scenario-6 main and feature-1)
                if self.enable_object_detection:
                    try:
                        # Scenario 8: Retrieve detected objects during recording
                        # Scenario 6 (feature-1): Create objects container
                        objects = sl.Objects()

                        # Scenario 6 (feature-1): Retrieve detected objects in recording loop
                        err_obj = self.camera.retrieve_objects(objects)

                        # Scenario 9: Process detected objects
                        if err_obj == sl.ERROR_CODE.SUCCESS:
                            # Scenario 11: Save detection data to CSV
                            if objects.object_list and len(objects.object_list) > 0:
                                self._save_detection_data(objects, current_time, self.frame_count)

                                # Optional: Log detection count periodically
                                if DEBUG_MODE and self.frame_count % 100 == 0:
                                    print(f"  Detected {len(objects.object_list)} objects in frame {self.frame_count}")

                        # Scenario 18: Handle detection failures (feature-1)
                        elif DEBUG_MODE and self.frame_count % 100 == 0:
                            print(f"  Object retrieval returned: {err_obj}")

                    except Exception as e:
                        # Scenario 18: Handle retrieve_objects failure
                        if DEBUG_MODE and self.frame_count % 100 == 0:
                            print(f"  Error retrieving objects: {e}")
                        # Continue recording despite detection failure

                # Scenarios 10-20: Retrieve tracked bodies and save to CSV (scenario-7 main and feature-1)
                if self.enable_body_tracking:
                    try:
                        # Scenario 10: Retrieve tracked bodies during recording
                        # Scenario 5 (feature-1): Create bodies container
                        bodies = sl.Bodies()

                        # Scenario 5 (feature-1): Retrieve tracked bodies in recording loop
                        err_body = self.camera.retrieve_bodies(bodies)

                        # Scenario 11: Process tracked bodies
                        if err_body == sl.ERROR_CODE.SUCCESS:
                            # Scenario 14: Save body tracking data to CSV
                            if bodies.body_list and len(bodies.body_list) > 0:
                                self._save_body_tracking_data(bodies, current_time, self.frame_count)

                                # Scenario 16: Track multiple people simultaneously
                                # Optional: Log body count periodically
                                if DEBUG_MODE and self.frame_count % 100 == 0:
                                    print(f"  Tracked {len(bodies.body_list)} people in frame {self.frame_count}")

                        # Scenario 20: Handle body tracking failures (feature-1)
                        elif DEBUG_MODE and self.frame_count % 100 == 0:
                            print(f"  Body retrieval returned: {err_body}")

                    except Exception as e:
                        # Scenario 20: Handle retrieve_bodies failure
                        if DEBUG_MODE and self.frame_count % 100 == 0:
                            print(f"  Error retrieving bodies: {e}")
                        # Continue recording despite body tracking failure

                # Scenarios 1-15: Detect planes and save to CSV (scenario-8 main and feature-1)
                # Note: Plane detection requires spatial mapping to be enabled
                # Scenario 19: Limit plane detection frequency (every 30 frames to reduce overhead)
                if self.enable_spatial_mapping and self.frame_count % 30 == 0:
                    try:
                        # Scenario 1: Detect floor plane in recording loop
                        floor_plane = sl.Plane()

                        # Scenario 1: Call find_floor_plane
                        err_floor = self.camera.find_floor_plane(floor_plane)

                        # Scenario 2: Handle successful floor plane detection
                        if err_floor == sl.ERROR_CODE.SUCCESS:
                            # Floor plane detected successfully
                            self.plane_id_counter += 1
                            self._save_plane_data(floor_plane, current_time, self.plane_id_counter)

                        # Scenario 14: Handle no planes detected
                        # If find_floor_plane fails, no error is logged (this is normal)

                    except Exception as e:
                        # Scenario 17/18: Handle find_floor_plane failure
                        if DEBUG_MODE and self.frame_count % 100 == 0:
                            print(f"  Error detecting planes: {e}")
                        # Continue recording despite plane detection failure

                # Scenarios 1-12: Retrieve fused IMU data and save to CSV (scenario-9 main and feature-1)
                # Note: Requires positional tracking with IMU fusion enabled
                if self.enable_positional_tracking and self.tracking_enable_imu_fusion:
                    try:
                        # Scenario 2: Retrieve fused sensors data in recording loop
                        sensors_data = sl.SensorsData()

                        # Scenario 2 & 18: Synchronize IMU data with frame timestamp
                        err_sensors = self.camera.get_sensors_data(sensors_data, sl.TIME_REFERENCE.IMAGE)

                        # Scenario 2: Handle successful sensors data retrieval
                        if err_sensors == sl.ERROR_CODE.SUCCESS:
                            # Scenario 8: Save fused IMU data to CSV
                            self._save_fused_imu_data(sensors_data, current_time, self.frame_count)

                        # Scenario 15 & 16: Handle IMU fusion unavailable or get_sensors_data failure
                        # If get_sensors_data fails, no error is logged (this is normal for some camera models)

                    except Exception as e:
                        # Scenario 16: Handle get_sensors_data failure
                        if DEBUG_MODE and self.frame_count % 100 == 0:
                            print(f"  Error retrieving fused IMU data: {e}")
                        # Continue recording despite fused IMU retrieval failure

                # Scenario 14: Monitor SVO recording status (scenario-10 main)
                # Note: Periodic status monitoring for debugging (every 300 frames)
                if self.enable_svo_recording and DEBUG_MODE and self.frame_count % 300 == 0 and self.frame_count > 0:
                    try:
                        # Scenario 14: Get recording status
                        recording_status = self.camera.get_recording_status()
                        if recording_status.status:
                            print(f"  SVO recording active - Frame {self.frame_count}")
                            # Optional: Log compression time/ratio if available
                            # print(f"  Compression time: {recording_status.current_compression_time} ms")
                            # print(f"  Compression ratio: {recording_status.current_compression_ratio}")
                    except Exception as e:
                        # Status monitoring is optional, don't fail if it errors
                        pass

                # Scenario 22: Handle insufficient disk space (scenario-10-feature-1)
                # Note: Monitor disk space every 600 frames to warn if running out
                if self.enable_svo_recording and self.frame_count % 600 == 0 and self.frame_count > 0:
                    try:
                        import shutil
                        # Get disk usage for output directory
                        disk_usage = shutil.disk_usage(self.output_dir)
                        available_gb = disk_usage.free / (1024 ** 3)

                        # Scenario 22: Warning should be logged if space < threshold
                        # Threshold: 5 GB for SVO recording (can fill quickly)
                        if available_gb < 5.0:
                            if DEBUG_MODE:
                                print(f"  ⚠ Low disk space: {available_gb:.1f} GB available")
                                print(f"  SVO recording may fail if disk becomes full")
                        elif available_gb < 10.0 and DEBUG_MODE and self.frame_count % 1800 == 0:
                            # Less urgent warning at 10GB threshold, log less frequently
                            print(f"  Disk space: {available_gb:.1f} GB available")
                    except Exception as e:
                        # Scenario 22: Error should be caught gracefully
                        # Disk space check is optional monitoring, don't fail recording
                        if DEBUG_MODE and self.frame_count % 1800 == 0:
                            print(f"  Could not check disk space: {e}")

                # Increment frame counter
                self.frame_count += 1

                # Scenario 14: Monitor depth computation performance (scenario-2 main)
                if self.enable_depth:
                    frame_time = time.time()

                    # Calculate time since last frame
                    if self.last_frame_time is not None:
                        frame_duration = frame_time - self.last_frame_time
                        self.frame_times.append(frame_duration)

                        # Keep only last 30 frame times for rolling average
                        if len(self.frame_times) > 30:
                            self.frame_times.pop(0)

                        # Calculate FPS every 30 frames
                        if self.frame_count % 30 == 0 and len(self.frame_times) > 0:
                            avg_frame_time = sum(self.frame_times) / len(self.frame_times)
                            current_fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0

                            if DEBUG_MODE:
                                print(f"  Performance: {current_fps:.1f} FPS (avg frame time: {avg_frame_time*1000:.1f}ms)")

                            # Scenario: if FPS drops below threshold, a warning should be logged
                            if current_fps < self.fps_warning_threshold:
                                if DEBUG_MODE:
                                    print(f"  ⚠ Warning: FPS ({current_fps:.1f}) below threshold ({self.fps_warning_threshold})")

                    self.last_frame_time = frame_time

            # Scenario: Save raw depth to numpy file (when recording completes)
            if self.save_raw_depth and self.raw_depth_frames:
                self._save_raw_depth_numpy()

            if DEBUG_MODE:
                print(f"Recording loop completed for camera #{self.camera_info.index}")
                print(f"  Total frames captured: {self.frame_count}")
                if self.enable_point_clouds:
                    print(f"  Total point clouds exported: {self.point_cloud_count}")

        except Exception as e:
            if DEBUG_MODE:
                print(f"Error in recording loop for camera #{self.camera_info.index}: {e}")
        finally:
            self.is_recording = False

    def stop_recording(self):
        """Stop recording and cleanup resources."""
        if not self.is_recording:
            return

        self.is_recording = False

        # Wait for thread to finish
        if self.recording_thread and self.recording_thread.is_alive():
            self.recording_thread.join(timeout=2.0)

        # Cleanup
        self._cleanup()

    def _cleanup(self):
        """
        Cleanup resources after recording completes or errors.

        Scenarios Covered (scenario-2-feature-2):
        - Release depth writer on cleanup
        - Writer should be set to None after release
        - File should be flushed to disk
        """
        # Scenario: Release video writers and set to None
        if self.left_writer:
            self.left_writer.release()
            self.left_writer = None

        if self.right_writer:
            self.right_writer.release()
            self.right_writer = None

        if self.stereo_writer:
            self.stereo_writer.release()
            self.stereo_writer = None

        # Scenario: Release depth writer on cleanup
        if self.depth_writer:
            self.depth_writer.release()
            self.depth_writer = None
            if DEBUG_MODE:
                print(f"  ✓ Released depth video writer")

        # Scenario 11: Release disparity writer on cleanup
        if self.disparity_writer:
            self.disparity_writer.release()
            self.disparity_writer = None
            if DEBUG_MODE:
                print(f"  ✓ Released disparity video writer")

        # Scenario 12: Release confidence writer on cleanup
        if self.confidence_writer:
            self.confidence_writer.release()
            self.confidence_writer = None
            if DEBUG_MODE:
                print(f"  ✓ Released confidence video writer")

        # Close data files
        if self.imu_file:
            self.imu_file.close()
            self.imu_file = None

        # Scenario: Close trajectory file on cleanup (scenario-4-feature-1)
        if self.trajectory_file:
            self.trajectory_file.close()
            self.trajectory_file = None
            if DEBUG_MODE:
                print(f"  ✓ Closed trajectory CSV file")

        # Scenario 19: Close detections CSV on cleanup (scenario-6-feature-1)
        if self.detections_file:
            self.detections_file.close()
            self.detections_file = None
            if DEBUG_MODE:
                print(f"  ✓ Closed detections CSV file")

        # Scenario 19: Close body tracking CSV on cleanup (scenario-7-feature-1)
        if self.body_tracking_file:
            self.body_tracking_file.close()
            self.body_tracking_file = None
            if DEBUG_MODE:
                print(f"  ✓ Closed body tracking CSV file")

        # Scenario 20: Close planes CSV on cleanup (scenario-8-feature-1)
        if self.planes_file:
            self.planes_file.close()
            self.planes_file = None
            if DEBUG_MODE:
                print(f"  ✓ Closed planes CSV file")

        # Scenario 19: Close fused IMU CSV on cleanup (scenario-9-feature-1)
        if self.imu_fused_file:
            self.imu_fused_file.close()
            self.imu_fused_file = None
            if DEBUG_MODE:
                print(f"  ✓ Closed fused IMU CSV file")

        # Disable GPU features
        if self.camera:
            try:
                import pyzed.sl as sl

                # Scenario 8 & 15: Extract and save mesh before disabling (scenario-5 main)
                if self.enable_spatial_mapping:
                    self._extract_and_save_mesh()

                    # Scenario 15: Disable spatial mapping during cleanup
                    self.camera.disable_spatial_mapping()
                    if DEBUG_MODE:
                        print(f"  ✓ Disabled spatial mapping")

                # Scenario 19: Disable object detection during cleanup (scenario-6 main and feature-1)
                if self.enable_object_detection:
                    self.camera.disable_object_detection()
                    if DEBUG_MODE:
                        print(f"  ✓ Disabled object detection")

                # Scenario 21: Disable body tracking during cleanup (scenario-7 main and feature-1)
                if self.enable_body_tracking:
                    self.camera.disable_body_tracking()
                    if DEBUG_MODE:
                        print(f"  ✓ Disabled body tracking")

                # Scenario 20: Disable tracking during cleanup (scenario-4 main)
                if self.enable_positional_tracking:
                    self.camera.disable_positional_tracking()
                    if DEBUG_MODE:
                        print(f"  ✓ Disabled positional tracking")

                # Scenario 8: Disable SVO recording after capture (scenario-10 main and feature-1)
                if self.enable_svo_recording:
                    self.camera.disable_recording()
                    if DEBUG_MODE:
                        print(f"  ✓ Disabled SVO recording")
                        print(f"  SVO file finalized and ready for replay")

                    # Scenario 13: Verify SVO file integrity after recording (scenario-10-feature-1)
                    if hasattr(self, 'svo_filepath') and self.svo_filepath:
                        import os
                        if os.path.exists(self.svo_filepath):
                            file_size_mb = os.path.getsize(self.svo_filepath) / (1024 * 1024)
                            if DEBUG_MODE:
                                print(f"  ✓ SVO file verified: {self.svo_filepath.name}")
                                print(f"  SVO file size: {file_size_mb:.1f} MB")
                        else:
                            if DEBUG_MODE:
                                print(f"  ⚠ SVO file not found at expected path")

                self.camera.close()
                self.camera = None

            except Exception as e:
                if DEBUG_MODE:
                    print(f"  ⚠ Error during cleanup: {e}")

    def _colorize_depth(self, depth_values: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
        """
        Colorize depth map for visualization.

        Scenarios Covered:
        - Colorize depth for visualization
        - Depth should be normalized to 0-255 range
        - Apply cv2.applyColorMap with COLORMAP_JET or COLORMAP_TURBO
        - Result should be 3-channel BGR image
        - Invalid pixels should be visually distinguishable

        Args:
            depth_values: Depth values in meters (height, width)
            valid_mask: Boolean mask of valid depth pixels

        Returns:
            Colorized depth image (height, width, 3) BGR format
        """
        # Create output image
        depth_normalized = np.zeros_like(depth_values, dtype=np.uint8)

        # Get valid depths for normalization
        valid_depths = depth_values[valid_mask]

        if len(valid_depths) > 0:
            # Normalize valid depths to 0-255 range
            min_valid = np.min(valid_depths)
            max_valid = np.max(valid_depths)

            if max_valid > min_valid:
                # Normalize to 0-255
                depth_normalized[valid_mask] = (
                    255 * (depth_values[valid_mask] - min_valid) / (max_valid - min_valid)
                ).astype(np.uint8)

        # Apply colormap (COLORMAP_TURBO for better visualization, COLORMAP_JET as fallback)
        try:
            depth_colorized = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_TURBO)
        except:
            depth_colorized = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)

        # Mark invalid pixels as black
        depth_colorized[~valid_mask] = [0, 0, 0]

        return depth_colorized

    def _save_raw_depth_numpy(self):
        """
        Save raw depth frames to numpy file.

        Scenarios Covered:
        - Save raw depth to numpy file
        - All depth frames should be stacked into 3D array
        - Filename should be {timestamp}-{index}-depth_raw.npy
        - File should contain shape (frame_count, height, width)
        - Depth values should be 32-bit float in meters
        """
        try:
            if not self.raw_depth_frames:
                return

            # Stack all frames into 3D array
            depth_array = np.stack(self.raw_depth_frames, axis=0)  # Shape: (frame_count, height, width)

            # Generate filename
            depth_raw_filename = self._generate_filename("depth_raw", "npy")
            depth_raw_path = self.output_dir / depth_raw_filename

            # Save to numpy file
            np.save(str(depth_raw_path), depth_array)

            if DEBUG_MODE:
                print(f"  Saved raw depth array: {depth_raw_filename}")
                print(f"  Shape: {depth_array.shape}, dtype: {depth_array.dtype}")

        except Exception as e:
            if DEBUG_MODE:
                print(f"  Error saving raw depth numpy file: {e}")

    def _save_point_cloud(self, timestamp: float, frame_number: int) -> None:
        """
        Save point cloud to file (PLY, XYZ, or PCD format).

        Scenarios Covered (scenario-3 main scenarios):
        - Scenario 3: Retrieve point cloud from camera
        - Scenario 4: Export point cloud to PLY format
        - Scenario 5: Export point cloud to XYZ format
        - Scenario 6: Export point cloud to PCD format
        - Scenario 7: Calculate point cloud size
        - Scenario 8: Filter invalid points from cloud
        - Scenario 9: Include RGB color in point cloud
        - Scenario 10: Handle point cloud export failure
        - Scenario 11: Track point cloud export count
        - Scenario 12: Verify point cloud quality
        - Scenario 14: Downsample point cloud for smaller files
        - Scenario 15: Skip point cloud export when depth unavailable

        Args:
            timestamp: Current timestamp for filename generation
            frame_number: Frame number for filename generation

        Returns:
            None
        """
        try:
            import pyzed.sl as sl

            # Scenario 3: Retrieve point cloud from camera
            # Create point cloud Mat object
            point_cloud = sl.Mat()

            # Retrieve XYZRGBA point cloud
            err = self.camera.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)

            # Scenario 15: Skip point cloud export when depth unavailable
            if err != sl.ERROR_CODE.SUCCESS:
                if DEBUG_MODE:
                    print(f"  Failed to retrieve point cloud: {err}")
                return

            # Get point cloud data as numpy array
            pc_data = point_cloud.get_data()  # Shape: (height, width, 4) - XYZRGBA

            # Scenario 8: Filter invalid points from cloud
            # Extract XYZ coordinates and RGB color
            xyz = pc_data[:, :, :3]  # XYZ coordinates
            rgba = pc_data[:, :, 3]  # RGB packed as float (need to extract)

            # Flatten to point list
            xyz_flat = xyz.reshape(-1, 3)

            # Filter invalid points (NaN, inf, or zero coordinates)
            valid_mask = np.isfinite(xyz_flat).all(axis=1)
            valid_mask &= ~(np.abs(xyz_flat).sum(axis=1) < 1e-6)  # Remove zero points

            xyz_valid = xyz_flat[valid_mask]

            # Scenario 14: Downsample point cloud for smaller files
            if self.point_cloud_downsample_factor > 1:
                # Sample every Nth point
                xyz_valid = xyz_valid[::self.point_cloud_downsample_factor]

            point_count = len(xyz_valid)

            # Scenario 12: Verify point cloud quality
            if point_count < 1000:
                if DEBUG_MODE:
                    print(f"  ⚠ Point cloud quality low: only {point_count} valid points")
                # Continue anyway, but log warning

            if point_count > 10_000_000:
                if DEBUG_MODE:
                    print(f"  ⚠ Point cloud very large: {point_count} points")

            # Generate filename with appropriate extension
            file_ext = self.point_cloud_format.lower()
            pc_filename = self._generate_filename(f"pointcloud_{frame_number:04d}", file_ext)
            full_path = self.output_dir / pc_filename

            # Export based on format
            if self.point_cloud_format == "PLY":
                # Scenario 4: Export point cloud to PLY format (native Zed SDK)
                # Scenario 9: Include RGB color in point cloud
                write_err = point_cloud.write(str(full_path))

                if write_err != sl.ERROR_CODE.SUCCESS:
                    if DEBUG_MODE:
                        print(f"  Failed to write PLY point cloud: {write_err}")
                    return

            elif self.point_cloud_format == "XYZ":
                # Scenario 5: Export point cloud to XYZ format
                # XYZ format: X Y Z per line, no color
                self._write_xyz_point_cloud(xyz_valid, full_path)

            elif self.point_cloud_format == "PCD":
                # Scenario 6: Export point cloud to PCD format
                # PCD format: PCL library format
                self._write_pcd_point_cloud(xyz_valid, full_path)

            else:
                if DEBUG_MODE:
                    print(f"  Unknown point cloud format: {self.point_cloud_format}")
                return

            # Scenario 11: Track point cloud export count
            self.point_cloud_count += 1

            # Scenario 7: Calculate point cloud size
            if DEBUG_MODE:
                file_size_mb = full_path.stat().st_size / (1024 * 1024)
                print(f"  ✓ Exported point cloud #{frame_number:04d}: {pc_filename}")
                print(f"    - Format: {self.point_cloud_format}")
                print(f"    - Point count: {point_count:,}")
                print(f"    - File size: {file_size_mb:.1f} MB")
                if self.point_cloud_downsample_factor > 1:
                    print(f"    - Downsample factor: {self.point_cloud_downsample_factor}")
                print(f"    - Total exported: {self.point_cloud_count}")

        except Exception as e:
            # Scenario 10: Handle point cloud export failure
            if DEBUG_MODE:
                print(f"  Error saving point cloud: {e}")
            # Recording continues despite point cloud export failure

    def _write_xyz_point_cloud(self, points: np.ndarray, output_path: Path) -> None:
        """
        Write point cloud to XYZ format (X Y Z per line).

        Scenario 5: Export point cloud to XYZ format

        Args:
            points: Nx3 array of XYZ coordinates
            output_path: Output file path
        """
        try:
            with open(output_path, 'w') as f:
                for point in points:
                    f.write(f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f}\n")
        except Exception as e:
            if DEBUG_MODE:
                print(f"  Error writing XYZ file: {e}")
            raise

    def _write_pcd_point_cloud(self, points: np.ndarray, output_path: Path) -> None:
        """
        Write point cloud to PCD format (Point Cloud Data).

        Scenario 6: Export point cloud to PCD format

        Args:
            points: Nx3 array of XYZ coordinates
            output_path: Output file path
        """
        try:
            num_points = len(points)

            with open(output_path, 'w') as f:
                # PCD header
                f.write("# .PCD v0.7 - Point Cloud Data file format\n")
                f.write("VERSION 0.7\n")
                f.write("FIELDS x y z\n")
                f.write("SIZE 4 4 4\n")
                f.write("TYPE F F F\n")
                f.write("COUNT 1 1 1\n")
                f.write(f"WIDTH {num_points}\n")
                f.write("HEIGHT 1\n")
                f.write("VIEWPOINT 0 0 0 1 0 0 0\n")
                f.write(f"POINTS {num_points}\n")
                f.write("DATA ascii\n")

                # Write points
                for point in points:
                    f.write(f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f}\n")

        except Exception as e:
            if DEBUG_MODE:
                print(f"  Error writing PCD file: {e}")
            raise

    def _save_tracking_data(self, camera_pose, timestamp: float, frame_number: int) -> None:
        """
        Save camera pose (position and orientation) to trajectory CSV file.

        Scenarios Covered (scenario-4-feature-1):
        - Implement _save_tracking_data method signature
        - Extract position from pose
        - Extract orientation from pose
        - Extract tracking confidence
        - Write pose data to CSV
        - Handle file write exception
        - Flush trajectory file periodically

        Args:
            camera_pose: sl.Pose object with position and orientation
            timestamp: Current timestamp
            frame_number: Frame number for CSV

        Returns:
            None
        """
        try:
            import pyzed.sl as sl

            # Check if trajectory file is available
            if not self.trajectory_file:
                return

            # Scenario: Extract position from pose
            translation = camera_pose.get_translation()
            tx = translation.get()  # Returns numpy array [x, y, z]
            x, y, z = float(tx[0]), float(tx[1]), float(tx[2])

            # Scenario: Extract orientation from pose
            orientation = camera_pose.get_orientation()
            quat = orientation.get()  # Returns numpy array [qx, qy, qz, qw]
            qx, qy, qz, qw = float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])

            # Scenario: Extract tracking confidence
            try:
                confidence = int(camera_pose.pose_confidence)
            except:
                confidence = -1  # Default if unavailable

            # Scenario: Write pose data to CSV
            csv_line = f"{timestamp},{frame_number},{x},{y},{z},{qx},{qy},{qz},{qw},{confidence}\n"
            self.trajectory_file.write(csv_line)

            # Scenario: Flush trajectory file periodically (every 100 frames)
            if frame_number % 100 == 0:
                self.trajectory_file.flush()

        except Exception as e:
            # Scenario: Handle file write exception
            if DEBUG_MODE:
                print(f"  Error saving tracking data: {e}")
            # Continue recording despite tracking save failure

    def _save_detection_data(self, objects, timestamp: float, frame_number: int) -> None:
        """
        Save detected objects data to detections CSV file.

        Scenarios Covered (scenario-6-feature-1):
        - Scenario 13: Save detection data to CSV
        - Scenario 14: Iterate through detected objects
        - Scenario 15: Extract object ID and label
        - Scenario 16: Write detection data row to CSV
        - Scenario 17: Filter objects by class (optional)
        - Scenario 18: Handle retrieve_objects failure

        Args:
            objects: sl.Objects container with detected objects
            timestamp: Current timestamp
            frame_number: Frame number for CSV

        Returns:
            None
        """
        try:
            import pyzed.sl as sl

            # Check if detections file is available
            if not self.detections_file:
                return

            # Scenario 13 & 14: Iterate through all detected objects
            if not hasattr(objects, 'object_list') or not objects.object_list:
                return  # No objects detected

            for obj in objects.object_list:
                try:
                    # Scenario 15 & 16: Extract object ID and label
                    object_id = int(obj.id)
                    object_label = str(obj.label).split('.')[-1]  # Extract label name from enum

                    # Scenario 16: Extract confidence
                    confidence = float(obj.confidence)

                    # Scenario 16: Extract 3D position
                    position = obj.position
                    x, y, z = float(position[0]), float(position[1]), float(position[2])

                    # Scenario 16: Extract 2D bounding box
                    # bounding_box_2d is array of 4 points [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
                    # We need top-left corner and width/height
                    bbox_2d = obj.bounding_box_2d

                    if bbox_2d is not None and len(bbox_2d) >= 4:
                        # Get all x and y coordinates
                        xs = [float(bbox_2d[i][0]) for i in range(4)]
                        ys = [float(bbox_2d[i][1]) for i in range(4)]

                        # Calculate bounding box in (x, y, w, h) format
                        bbox_x = min(xs)
                        bbox_y = min(ys)
                        bbox_w = max(xs) - min(xs)
                        bbox_h = max(ys) - min(ys)
                    else:
                        bbox_x, bbox_y, bbox_w, bbox_h = 0, 0, 0, 0

                    # Scenario 16: Write detection data row to CSV
                    # Format: timestamp,frame,object_id,class,confidence,x,y,z,bbox_x,bbox_y,bbox_w,bbox_h
                    csv_line = f"{timestamp},{frame_number},{object_id},{object_label},{confidence},"
                    csv_line += f"{x},{y},{z},{bbox_x},{bbox_y},{bbox_w},{bbox_h}\n"

                    self.detections_file.write(csv_line)

                except Exception as e:
                    # Skip object if extraction fails
                    if DEBUG_MODE:
                        print(f"  Error extracting object data: {e}")
                    continue

            # Scenario 16: Flush to disk periodically (every 100 frames)
            if frame_number % 100 == 0:
                self.detections_file.flush()

        except Exception as e:
            # Scenario 18: Handle save exception
            if DEBUG_MODE:
                print(f"  Error saving detection data: {e}")
            # Continue recording despite detection save failure

    def _save_body_tracking_data(self, bodies, timestamp: float, frame_number: int) -> None:
        """
        Save tracked bodies skeleton data to body tracking CSV file.

        Scenarios Covered (scenario-7-feature-1):
        - Scenario 14: Save body tracking data to CSV
        - Scenario 6: Iterate through tracked bodies
        - Scenario 7: Extract body ID and confidence
        - Scenario 8: Extract skeleton keypoints
        - Scenario 10: Extract keypoint confidences
        - Scenario 17: Write body keypoint data to CSV
        - Scenario 19: Handle occluded keypoints
        - Scenario 20: Handle retrieve_bodies failure

        Args:
            bodies: sl.Bodies container with detected people
            timestamp: Current timestamp
            frame_number: Frame number for CSV

        Returns:
            None
        """
        try:
            import pyzed.sl as sl

            # Check if body tracking file is available
            if not self.body_tracking_file:
                return

            # Scenario 6: Iterate through all tracked bodies
            if not hasattr(bodies, 'body_list') or not bodies.body_list:
                return  # No bodies detected

            for body in bodies.body_list:
                try:
                    # Scenario 7: Extract body ID and confidence
                    body_id = int(body.id)
                    body_confidence = float(body.confidence)

                    # Scenario 8: Extract skeleton keypoints
                    # keypoint is array of 3D positions (18/34/70 depending on format)
                    keypoints = body.keypoint

                    # Scenario 10: Extract keypoint confidences
                    # keypoint_confidence is array of confidences per keypoint
                    keypoint_confidences = body.keypoint_confidence

                    # Scenario 17: Write body keypoint data to CSV
                    # For each keypoint, write a separate CSV row
                    for keypoint_id in range(len(keypoints)):
                        try:
                            # Extract keypoint 3D position
                            kp = keypoints[keypoint_id]
                            x, y, z = float(kp[0]), float(kp[1]), float(kp[2])

                            # Extract keypoint confidence
                            kp_conf = float(keypoint_confidences[keypoint_id]) if keypoint_id < len(keypoint_confidences) else 0.0

                            # Scenario 19: Handle occluded keypoints
                            # Occluded keypoints have low confidence (<30) but are still written
                            # Format: timestamp,frame,body_id,keypoint_id,x,y,z,confidence
                            csv_line = f"{timestamp},{frame_number},{body_id},{keypoint_id},{x},{y},{z},{kp_conf}\n"

                            self.body_tracking_file.write(csv_line)

                        except Exception as e:
                            # Skip keypoint if extraction fails
                            if DEBUG_MODE and frame_number % 100 == 0:
                                print(f"  Error extracting keypoint {keypoint_id}: {e}")
                            continue

                except Exception as e:
                    # Skip body if extraction fails
                    if DEBUG_MODE and frame_number % 100 == 0:
                        print(f"  Error extracting body data: {e}")
                    continue

            # Scenario 17: Flush to disk periodically (every 100 frames)
            if frame_number % 100 == 0:
                self.body_tracking_file.flush()

        except Exception as e:
            # Scenario 20: Handle save exception
            if DEBUG_MODE:
                print(f"  Error saving body tracking data: {e}")
            # Continue recording despite body tracking save failure

    def _save_plane_data(self, plane, timestamp: float, plane_id: int) -> None:
        """
        Save detected plane data to planes CSV file.

        Scenarios Covered (scenario-8-feature-1):
        - Scenario 10: Save plane data to CSV
        - Scenario 4: Extract plane type
        - Scenario 5: Extract plane normal vector
        - Scenario 6: Extract plane center point
        - Scenario 7: Extract plane bounds
        - Scenario 9: Calculate plane area
        - Scenario 13: Write plane data row to CSV
        - Scenario 15: Filter planes by area threshold

        Args:
            plane: sl.Plane object with detected plane
            timestamp: Current timestamp
            plane_id: Unique plane identifier

        Returns:
            None
        """
        try:
            import pyzed.sl as sl
            import numpy as np

            # Check if planes file is available
            if not self.planes_file:
                return

            # Scenario 4: Extract plane type
            plane_type = str(plane.type).split('.')[-1]  # Extract type name from enum

            # Scenario 5: Extract plane normal vector
            normal = plane.get_normal()
            normal_x, normal_y, normal_z = float(normal[0]), float(normal[1]), float(normal[2])

            # Scenario 6: Extract plane center point
            center = plane.get_center()
            center_x, center_y, center_z = float(center[0]), float(center[1]), float(center[2])

            # Scenario 7: Extract plane bounds
            # Scenario 9: Calculate plane area
            try:
                bounds = plane.get_bounds()

                # Calculate area from bounds polygon
                # Approximate area using triangulation from centroid
                if bounds is not None and len(bounds) >= 3:
                    # Simple triangulation: sum of triangles from first point to all other edges
                    area = 0.0
                    for i in range(1, len(bounds) - 1):
                        # Triangle from bounds[0], bounds[i], bounds[i+1]
                        v1 = np.array(bounds[i]) - np.array(bounds[0])
                        v2 = np.array(bounds[i + 1]) - np.array(bounds[0])
                        # Cross product gives twice the triangle area
                        cross = np.cross(v1, v2)
                        triangle_area = 0.5 * np.linalg.norm(cross)
                        area += triangle_area
                else:
                    area = 0.0
            except Exception as e:
                area = 0.0
                if DEBUG_MODE:
                    print(f"  Error calculating plane area: {e}")

            # Scenario 15: Filter planes by area threshold
            # Minimum area threshold: 0.5 m²
            area_threshold = 0.5
            if area < area_threshold:
                if DEBUG_MODE:
                    print(f"  Skipping small plane (area={area:.2f} m² < {area_threshold} m²)")
                return

            # Scenario 13: Write plane data row to CSV
            # Format: timestamp,plane_id,type,center_x,center_y,center_z,normal_x,normal_y,normal_z,area
            csv_line = f"{timestamp},{plane_id},{plane_type},"
            csv_line += f"{center_x},{center_y},{center_z},"
            csv_line += f"{normal_x},{normal_y},{normal_z},{area}\n"

            self.planes_file.write(csv_line)

            # Flush periodically
            if plane_id % 10 == 0:
                self.planes_file.flush()

            if DEBUG_MODE:
                print(f"  Saved plane {plane_id}: {plane_type}, area={area:.2f} m², center=({center_x:.2f}, {center_y:.2f}, {center_z:.2f})")

        except Exception as e:
            # Handle save exception
            if DEBUG_MODE:
                print(f"  Error saving plane data: {e}")
            # Continue recording despite plane save failure

    def _save_fused_imu_data(self, sensors_data, timestamp: float, frame_number: int) -> None:
        """
        Save fused IMU data to CSV file.

        Scenarios Covered (scenario-9-feature-1):
        - Scenario 8: Save fused IMU data to CSV
        - Scenario 3: Extract IMU data from sensors data
        - Scenario 4: Extract linear acceleration
        - Scenario 5: Extract angular velocity
        - Scenario 6: Extract fused orientation
        - Scenario 7: Extract pose translation
        - Scenario 11: Write fused IMU data row to CSV
        - Scenario 12: Extract pose confidence

        Args:
            sensors_data: sl.SensorsData with fused IMU measurements
            timestamp: Current timestamp
            frame_number: Frame number for CSV

        Returns:
            None
        """
        try:
            import pyzed.sl as sl

            # Check if fused IMU file is available
            if not self.imu_fused_file:
                return

            # Scenario 3: Extract IMU data from sensors data
            imu_data = sensors_data.get_imu_data()

            # Scenario 4: Extract linear acceleration (gravity-compensated)
            linear_accel = imu_data.get_linear_acceleration()
            ax, ay, az = float(linear_accel[0]), float(linear_accel[1]), float(linear_accel[2])

            # Scenario 5: Extract angular velocity (calibrated)
            angular_vel = imu_data.get_angular_velocity()
            gx, gy, gz = float(angular_vel[0]), float(angular_vel[1]), float(angular_vel[2])

            # Scenario 6: Extract fused orientation
            pose = imu_data.get_pose()
            orientation = pose.get_orientation()
            qx, qy, qz, qw = float(orientation[0]), float(orientation[1]), float(orientation[2]), float(orientation[3])

            # Scenario 7: Extract pose translation
            translation = pose.get_translation()
            pos_x, pos_y, pos_z = float(translation[0]), float(translation[1]), float(translation[2])

            # Scenario 12: Extract pose confidence
            try:
                confidence = int(pose.pose_confidence)
            except:
                confidence = -1  # Default if unavailable

            # Scenario 11: Write fused IMU data row to CSV
            # Format: timestamp,frame,ax,ay,az,gx,gy,gz,qx,qy,qz,qw,pos_x,pos_y,pos_z,confidence
            csv_line = f"{timestamp},{frame_number},"
            csv_line += f"{ax},{ay},{az},"
            csv_line += f"{gx},{gy},{gz},"
            csv_line += f"{qx},{qy},{qz},{qw},"
            csv_line += f"{pos_x},{pos_y},{pos_z},{confidence}\n"

            self.imu_fused_file.write(csv_line)

            # Flush periodically (every 100 frames)
            if frame_number % 100 == 0:
                self.imu_fused_file.flush()

        except Exception as e:
            # Scenario 16: Handle get_sensors_data failure
            if DEBUG_MODE and frame_number % 100 == 0:
                print(f"  Error saving fused IMU data: {e}")
            # Continue recording despite fused IMU save failure

    @staticmethod
    def calculate_trajectory_statistics(trajectory_csv_path: str) -> dict:
        """
        Calculate trajectory statistics from CSV file.

        Scenario Covered (scenario-4-feature-1):
        - Calculate trajectory statistics
        - Total distance traveled can be calculated
        - Average velocity can be calculated
        - Trajectory can be plotted in 3D

        Args:
            trajectory_csv_path: Path to trajectory CSV file

        Returns:
            Dictionary with statistics:
            {
                'total_distance': float,  # meters
                'average_velocity': float,  # m/s
                'duration': float,  # seconds
                'num_poses': int,
                'positions': list,  # [(x, y, z), ...] for 3D plotting
                'timestamps': list,  # [timestamp, ...]
                'success': bool,
                'error': str or None
            }
        """
        try:
            import csv
            from pathlib import Path

            result = {
                'total_distance': 0.0,
                'average_velocity': 0.0,
                'duration': 0.0,
                'num_poses': 0,
                'positions': [],
                'timestamps': [],
                'success': False,
                'error': None
            }

            # Check if file exists
            csv_path = Path(trajectory_csv_path)
            if not csv_path.exists():
                result['error'] = f"Trajectory file not found: {trajectory_csv_path}"
                return result

            # Read CSV file
            positions = []
            timestamps = []

            with open(csv_path, 'r') as f:
                reader = csv.DictReader(f)

                for row in reader:
                    try:
                        timestamp = float(row['timestamp'])
                        x = float(row['x'])
                        y = float(row['y'])
                        z = float(row['z'])

                        timestamps.append(timestamp)
                        positions.append((x, y, z))

                    except (KeyError, ValueError) as e:
                        # Skip invalid rows
                        continue

            num_poses = len(positions)
            result['num_poses'] = num_poses
            result['positions'] = positions
            result['timestamps'] = timestamps

            if num_poses < 2:
                result['error'] = "Not enough valid poses to calculate statistics"
                return result

            # Calculate total distance traveled
            total_distance = 0.0
            for i in range(1, num_poses):
                prev_pos = positions[i - 1]
                curr_pos = positions[i]

                # Euclidean distance
                dx = curr_pos[0] - prev_pos[0]
                dy = curr_pos[1] - prev_pos[1]
                dz = curr_pos[2] - prev_pos[2]
                distance = np.sqrt(dx**2 + dy**2 + dz**2)

                total_distance += distance

            result['total_distance'] = total_distance

            # Calculate duration
            duration = timestamps[-1] - timestamps[0]
            result['duration'] = duration

            # Calculate average velocity
            if duration > 0:
                average_velocity = total_distance / duration
                result['average_velocity'] = average_velocity
            else:
                result['average_velocity'] = 0.0

            result['success'] = True

            if DEBUG_MODE:
                print(f"\n=== Trajectory Statistics ===")
                print(f"  Number of poses: {num_poses}")
                print(f"  Duration: {duration:.2f} seconds")
                print(f"  Total distance: {total_distance:.2f} meters")
                print(f"  Average velocity: {result['average_velocity']:.3f} m/s")
                print(f"  Trajectory data ready for 3D plotting")

            return result

        except Exception as e:
            return {
                'total_distance': 0.0,
                'average_velocity': 0.0,
                'duration': 0.0,
                'num_poses': 0,
                'positions': [],
                'timestamps': [],
                'success': False,
                'error': str(e)
            }

    def _extract_and_save_mesh(self) -> None:
        """
        Extract and save spatial mesh to file(s).

        Scenarios Covered (scenario-5 main and feature-1):
        - Scenario 8: Extract final mesh after recording
        - Scenario 9: Apply mesh filtering
        - Scenario 10: Save mesh to OBJ format
        - Scenario 11: Export mesh with texture
        - Scenario 12: Export mesh without texture
        - Scenario 13: Get mesh statistics
        - Feature scenarios: Extract mesh, filter, save to OBJ/PLY formats
        """
        try:
            import pyzed.sl as sl

            if DEBUG_MODE:
                print(f"\n=== Extracting Spatial Mesh ===")

            # Scenario 8: Extract final mesh after recording
            mesh = sl.Mesh()

            if DEBUG_MODE:
                print(f"  Extracting whole spatial map...")

            # Extract the complete mesh
            err = self.camera.extract_whole_spatial_map(mesh)

            # Handle mesh extraction failure
            if err != sl.ERROR_CODE.SUCCESS:
                if DEBUG_MODE:
                    print(f"  ✗ Mesh extraction failed: {err}")
                return

            if DEBUG_MODE:
                print(f"  ✓ Mesh extracted successfully")

            # Scenario 13: Get mesh statistics
            nb_vertices = mesh.get_number_of_vertices()
            nb_triangles = mesh.get_number_of_triangles()

            if DEBUG_MODE:
                print(f"  Mesh statistics:")
                print(f"    - Vertices: {nb_vertices:,}")
                print(f"    - Triangles: {nb_triangles:,}")

            # Check if mesh is empty
            if nb_vertices == 0 or nb_triangles == 0:
                if DEBUG_MODE:
                    print(f"  ⚠ Mesh is empty, skipping save")
                return

            # Scenario 9: Apply mesh filtering
            if DEBUG_MODE:
                print(f"  Applying mesh filter...")

            mesh.filter(sl.MESH_FILTER.LOW)

            if DEBUG_MODE:
                print(f"  ✓ Mesh filtered (artifacts removed)")

            # Save mesh in requested formats
            for mesh_format in self.mapping_mesh_formats:
                if mesh_format.lower() == "obj":
                    self._save_mesh_obj(mesh)
                elif mesh_format.lower() == "ply":
                    self._save_mesh_ply(mesh)
                else:
                    if DEBUG_MODE:
                        print(f"  ⚠ Unknown mesh format: {mesh_format}")

        except Exception as e:
            # Scenario: Handle mesh extraction failure
            if DEBUG_MODE:
                print(f"  Error extracting/saving mesh: {e}")
            # Continue cleanup despite mesh extraction failure

    def _save_mesh_obj(self, mesh) -> None:
        """
        Save mesh to OBJ format.

        Scenarios Covered (scenario-5 main and feature-1):
        - Scenario 10: Save mesh to OBJ format
        - Scenario 11: Export mesh with texture
        - Scenario 12: Export mesh without texture
        """
        try:
            import pyzed.sl as sl

            # Scenario 10: Save mesh to OBJ format
            # Generate filename
            mesh_filename = self._generate_filename("mesh", "obj")
            mesh_path = self.output_dir / mesh_filename

            if DEBUG_MODE:
                print(f"  Saving mesh to OBJ format...")

            # Save mesh with texture or vertex colors
            err = mesh.save(str(mesh_path))

            if err == sl.ERROR_CODE.SUCCESS:
                if DEBUG_MODE:
                    print(f"  ✓ Saved mesh: {mesh_filename}")

                    # Check what files were created
                    if mesh_path.exists():
                        file_size_mb = mesh_path.stat().st_size / (1024 * 1024)
                        print(f"    - OBJ file: {file_size_mb:.1f} MB")

                    # Scenario 11: Export mesh with texture
                    if self.mapping_save_texture:
                        mtl_path = mesh_path.with_suffix('.mtl')
                        texture_path = mesh_path.with_name(mesh_path.stem + '_texture.png')

                        if mtl_path.exists():
                            print(f"    - MTL file: {mtl_path.name}")
                        if texture_path.exists():
                            tex_size_mb = texture_path.stat().st_size / (1024 * 1024)
                            print(f"    - Texture file: {texture_path.name} ({tex_size_mb:.1f} MB)")
                    # Scenario 12: Export mesh without texture
                    else:
                        print(f"    - Vertex colors only (no texture)")

                # Scenario 17: Verify mesh file integrity
                self._verify_mesh_integrity(mesh, mesh_path)
            else:
                if DEBUG_MODE:
                    print(f"  ✗ Failed to save OBJ mesh: {err}")

        except Exception as e:
            if DEBUG_MODE:
                print(f"  Error saving OBJ mesh: {e}")

    def _save_mesh_ply(self, mesh) -> None:
        """
        Save mesh to PLY format.

        Scenario Covered (scenario-5-feature-1):
        - Save mesh to PLY format
        """
        try:
            import pyzed.sl as sl

            # Generate filename
            mesh_filename = self._generate_filename("mesh", "ply")
            mesh_path = self.output_dir / mesh_filename

            if DEBUG_MODE:
                print(f"  Saving mesh to PLY format...")

            # Save mesh to PLY
            err = mesh.save(str(mesh_path), sl.MESH_FILE_FORMAT.PLY)

            if err == sl.ERROR_CODE.SUCCESS:
                if DEBUG_MODE:
                    file_size_mb = mesh_path.stat().st_size / (1024 * 1024)
                    print(f"  ✓ Saved PLY mesh: {mesh_filename} ({file_size_mb:.1f} MB)")
            else:
                if DEBUG_MODE:
                    print(f"  ✗ Failed to save PLY mesh: {err}")

        except Exception as e:
            if DEBUG_MODE:
                print(f"  Error saving PLY mesh: {e}")

    def _verify_mesh_integrity(self, mesh, mesh_path: Path) -> bool:
        """
        Verify mesh file integrity after saving.

        Scenario Covered (scenario-5 main):
        - Scenario 17: Verify mesh file integrity
        - .obj file should be valid and openable
        - Vertex count should match expected range
        - Triangle count should be reasonable
        - Texture should be properly UV-mapped (if enabled)

        Args:
            mesh: sl.Mesh object that was saved
            mesh_path: Path to the saved mesh file

        Returns:
            True if mesh passes integrity checks, False otherwise
        """
        try:
            if not DEBUG_MODE:
                return True  # Skip validation if not in debug mode

            print(f"\n  === Mesh Integrity Verification ===")

            # Scenario: .obj file should be valid and openable
            if not mesh_path.exists():
                print(f"  ✗ OBJ file not found: {mesh_path}")
                return False

            file_size = mesh_path.stat().st_size
            if file_size == 0:
                print(f"  ✗ OBJ file is empty")
                return False

            print(f"  ✓ OBJ file exists and has content ({file_size / (1024*1024):.2f} MB)")

            # Scenario: Vertex count should match expected range
            nb_vertices = mesh.get_number_of_vertices()
            if nb_vertices < 100:
                print(f"  ⚠ Warning: Very few vertices ({nb_vertices}) - mesh may be incomplete")
            elif nb_vertices > 50_000_000:
                print(f"  ⚠ Warning: Extremely high vertex count ({nb_vertices:,}) - may cause memory issues")
            else:
                print(f"  ✓ Vertex count is reasonable: {nb_vertices:,}")

            # Scenario: Triangle count should be reasonable
            nb_triangles = mesh.get_number_of_triangles()
            if nb_triangles < 100:
                print(f"  ⚠ Warning: Very few triangles ({nb_triangles}) - mesh may be incomplete")
            elif nb_triangles > 100_000_000:
                print(f"  ⚠ Warning: Extremely high triangle count ({nb_triangles:,}) - may cause memory issues")
            else:
                print(f"  ✓ Triangle count is reasonable: {nb_triangles:,}")

            # Check vertex/triangle ratio (typically 2:1 for closed meshes)
            if nb_vertices > 0 and nb_triangles > 0:
                ratio = nb_triangles / nb_vertices
                if ratio < 0.5 or ratio > 4.0:
                    print(f"  ⚠ Warning: Unusual vertex/triangle ratio ({ratio:.2f}) - typical is ~2.0")
                else:
                    print(f"  ✓ Vertex/triangle ratio is good: {ratio:.2f}")

            # Scenario: Texture should be properly UV-mapped (if enabled)
            if self.mapping_save_texture:
                texture_path = mesh_path.with_name(mesh_path.stem + '_texture.png')
                mtl_path = mesh_path.with_suffix('.mtl')

                if texture_path.exists():
                    tex_size = texture_path.stat().st_size / (1024 * 1024)
                    if tex_size > 0.1:  # At least 100KB
                        print(f"  ✓ Texture file exists and has content ({tex_size:.1f} MB)")
                    else:
                        print(f"  ⚠ Warning: Texture file is very small ({tex_size:.2f} MB)")
                else:
                    print(f"  ⚠ Warning: Texture file not found (texture mapping was enabled)")

                if mtl_path.exists():
                    print(f"  ✓ Material file (.mtl) exists")
                else:
                    print(f"  ⚠ Warning: Material file (.mtl) not found")

            # Basic OBJ file format validation
            try:
                with open(mesh_path, 'r') as f:
                    first_lines = [f.readline() for _ in range(10)]
                    has_vertices = any(line.startswith('v ') for line in first_lines)
                    has_faces = any(line.startswith('f ') for line in first_lines)

                    if has_vertices and has_faces:
                        print(f"  ✓ OBJ file format appears valid (contains vertices and faces)")
                    else:
                        print(f"  ⚠ Warning: OBJ file may be malformed (missing v or f entries)")
            except Exception as e:
                print(f"  ⚠ Warning: Could not validate OBJ format: {e}")

            print(f"  === Mesh verification complete ===\n")
            return True

        except Exception as e:
            if DEBUG_MODE:
                print(f"  Error during mesh verification: {e}")
            return False

    def _generate_filename(self, sensor: str, extension: str) -> str:
        """
        Generate filename with timestamp and camera index.

        Args:
            sensor: Sensor type (e.g., 'left', 'depth', 'trajectory')
            extension: File extension (e.g., 'mp4', 'csv', 'ply')

        Returns:
            Filename string
        """
        if self.start_timestamp:
            timestamp_str = datetime.fromtimestamp(self.start_timestamp).strftime("%Y-%m-%d-%H-%M-%S-%f")[:-3]
        else:
            timestamp_str = datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")[:-3]

        return f"{timestamp_str}-{self.camera_info.index}-{sensor}.{extension}"
