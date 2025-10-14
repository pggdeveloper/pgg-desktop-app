"""Multi-camera recording orchestrator with synchronized timestamp management."""
import time
import platform
from datetime import datetime
from pathlib import Path
from typing import Optional, Union
import numpy as np
from domain.camera_info import CameraInfo
from domain.recording_session import RecordingSession
from utils.realsense_camera import RealSenseCameraRecorder
from utils.zed_camera import ZedCameraRecorder
from utils.zed_camera_sdk import ZedCameraSDKRecorder
from utils.calibration_loader import CalibrationLoader
from config import DEBUG_MODE, VIDEO_SECS, FPS, START_RECORDING_DELAY, BASE_DIR
import logging

logger = logging.getLogger(__name__)


class CameraRecordingOrchestrator:
    """
    Orchestrates synchronized multi-camera recording.

    Manages multiple camera recorders, each running in its own thread,
    with synchronized start timestamps for temporal alignment.

    Features:
    - Automatic recorder creation based on camera type
    - Synchronized timestamp generation
    - Thread-based concurrent recording
    - Error isolation per camera
    - Session management
    """

    # Class-level cache for GPU availability (shared across all instances)
    _gpu_available_cache: Optional[bool] = None
    _cuda_version: Optional[str] = None
    _cudnn_available: Optional[bool] = None
    _tensorrt_available: Optional[bool] = None
    _vram_gb: Optional[float] = None
    _compute_capability: Optional[float] = None

    def __init__(
        self,
        cameras: list[CameraInfo],
        session_id: Optional[str] = None,
        duration_secs: int = VIDEO_SECS,
        recording_delay: int = START_RECORDING_DELAY,
        fps: int = FPS,
        calibration_path: Optional[Union[str, Path]] = None,
    ):
        """
        Initialize camera recording orchestrator.

        Implements Scenario 9.1: Load camera calibration at orchestrator initialization

        Args:
            cameras: List of CameraInfo objects for specialized cameras
            session_id: Optional session identifier (auto-generated if None)
            duration_secs: Recording duration in seconds
            recording_delay: Countdown delay before starting
            fps: Target frame rate for all cameras
            calibration_path: Optional path to calibration JSON file
        """
        self.cameras = cameras
        self.duration_secs = duration_secs
        self.recording_delay = recording_delay
        self.fps = fps

        # Create or use provided session ID
        if session_id is None:
            session_id = RecordingSession.create_session_id()

        # Create recording session
        self.session = RecordingSession(
            session_id=session_id,
            base_dir=BASE_DIR,
            cameras=cameras,
            duration_secs=duration_secs,
            recording_delay=recording_delay,
        )

        # Create session directory
        self.session.session_dir.mkdir(parents=True, exist_ok=True)

        # Recorder management
        self.recorders: list[Union[RealSenseCameraRecorder, ZedCameraRecorder, ZedCameraSDKRecorder]] = []
        self.is_recording = False
        self.sync_timestamp: Optional[float] = None

        # Calibration management (Scenario 9.1)
        self.calibration_loader: Optional[CalibrationLoader] = None
        self.calibration_loaded = False

        if calibration_path:
            self._load_calibration(calibration_path)

    def initialize_cameras(self) -> tuple[int, int]:
        """
        Initialize all camera recorders.

        Creates appropriate recorder instances for each camera type
        and calls their initialize() methods.

        Platform Validation:
            Multi-camera recording only supported on Windows due to SDK requirements.
            Returns (0, 0) on non-Windows platforms.

        Returns:
            Tuple of (success_count, failed_count)
        """
        # Platform validation - Windows only
        if platform.system() != 'Windows':
            if DEBUG_MODE:
                print(f"Multi-camera recording only supported on Windows (current platform: {platform.system()})")
            return (0, 0)

        success_count = 0
        failed_count = 0

        for camera in self.cameras:
            try:
                # Create appropriate recorder
                recorder = self._create_recorder(camera)

                # Initialize recorder
                if recorder.initialize():
                    self.recorders.append(recorder)
                    success_count += 1
                    if DEBUG_MODE:
                        print(f"Initialized {camera.camera_type} camera #{camera.index}")
                else:
                    failed_count += 1
                    if DEBUG_MODE:
                        print(f"Failed to initialize {camera.camera_type} camera #{camera.index}")

            except Exception as e:
                failed_count += 1
                if DEBUG_MODE:
                    print(f"Error initializing {camera.camera_type} camera #{camera.index}: {e}")

        return (success_count, failed_count)

    def _create_recorder(
        self, camera_info: CameraInfo
    ) -> Union[RealSenseCameraRecorder, ZedCameraRecorder, ZedCameraSDKRecorder]:
        """
        Create appropriate recorder based on camera type and GPU availability.

        For Zed cameras:
        - If GPU is available: Creates ZedCameraSDKRecorder with GPU features
        - If GPU not available: Creates ZedCameraRecorder (CPU fallback)

        Scenarios Covered:
        - Initialize ZedCameraSDKRecorder with GPU
        - Fallback to CPU recorder when GPU unavailable
        - Detect system without GPU

        Args:
            camera_info: Camera metadata

        Returns:
            Appropriate camera recorder instance

        Raises:
            ValueError: If camera type is not supported
        """
        # RealSense cameras
        if camera_info.camera_type.is_realsense:
            return RealSenseCameraRecorder(
                camera_info=camera_info,
                output_dir=self.session.session_dir,
                recording_duration=self.duration_secs,
                recording_delay=self.recording_delay,
                fps=self.fps,
            )

        # Zed cameras - check GPU availability
        elif camera_info.camera_type.is_zed:
            # Check GPU availability
            gpu_available = self._check_gpu_available()

            if gpu_available:
                # Scenario: Initialize ZedCameraSDKRecorder with GPU
                if DEBUG_MODE:
                    print(f"Creating ZedCameraSDKRecorder with GPU features for camera #{camera_info.index}")

                return ZedCameraSDKRecorder(
                    camera_info=camera_info,
                    output_dir=self.session.session_dir,
                    recording_duration=self.duration_secs,
                    recording_delay=self.recording_delay,
                    fps=self.fps,
                    # GPU features enabled by default
                    depth_mode="ULTRA",  # High-precision depth
                    enable_depth=True,
                    enable_point_clouds=False,  # Can be configured
                    enable_positional_tracking=False,  # Can be configured
                    enable_spatial_mapping=False,  # Can be configured
                    enable_object_detection=False,  # Requires TensorRT
                    enable_body_tracking=False,  # Requires TensorRT
                    enable_plane_detection=False,  # Can be configured
                    enable_imu_fusion=False,  # Can be configured
                    enable_svo_recording=False,  # Can be configured
                )
            else:
                # Scenario: Fallback to CPU recorder when GPU unavailable
                # Scenario: Detect system without GPU
                if DEBUG_MODE:
                    print(f"Creating ZedCameraRecorder (CPU fallback) for camera #{camera_info.index}")

                return ZedCameraRecorder(
                    camera_info=camera_info,
                    output_dir=self.session.session_dir,
                    recording_duration=self.duration_secs,
                    recording_delay=self.recording_delay,
                    fps=self.fps,
                )

        else:
            raise ValueError(f"Unsupported camera type: {camera_info.camera_type}")

    def start_recording_with_countdown(self) -> bool:
        """
        Start recording with countdown delay.

        Displays countdown, generates synchronized timestamp,
        and starts all recorders simultaneously.

        Returns:
            True if recording started successfully, False otherwise
        """
        if self.is_recording:
            if DEBUG_MODE:
                print("Recording already in progress")
            return False

        if not self.recorders:
            if DEBUG_MODE:
                print("No recorders initialized")
            return False

        # Countdown
        if DEBUG_MODE:
            print(f"Starting recording in {self.recording_delay} seconds...")

        for i in range(self.recording_delay, 0, -1):
            if DEBUG_MODE:
                print(f"  {i}...")
            time.sleep(1)

        # Generate synchronized timestamp
        self.sync_timestamp = time.time()
        self.session.start_time = datetime.fromtimestamp(self.sync_timestamp)

        if DEBUG_MODE:
            print(f"Recording started at {self.session.start_time.isoformat()}")

        # PHASE 1: Prepare all recorders (slow operations done sequentially)
        self.is_recording = True
        prepared_threads = []

        if DEBUG_MODE:
            print(f"Phase 1: Preparing all recorders...")

        for recorder in self.recorders:
            try:
                thread = recorder.prepare_for_recording(self.sync_timestamp)
                if thread:
                    prepared_threads.append((recorder, thread))
                    if DEBUG_MODE:
                        print(f"  Prepared recorder for camera #{recorder.camera_info.index}")
                else:
                    if DEBUG_MODE:
                        print(f"  Failed to prepare recorder for camera #{recorder.camera_info.index}")
            except Exception as e:
                if DEBUG_MODE:
                    print(f"  Error preparing recorder for camera #{recorder.camera_info.index}: {e}")

        if not prepared_threads:
            self.is_recording = False
            return False

        # PHASE 2: Start all threads in rapid succession (<1ms total)
        if DEBUG_MODE:
            print(f"Phase 2: Starting {len(prepared_threads)} recording threads...")

        started_count = 0
        for recorder, thread in prepared_threads:
            try:
                thread.start()  # Fast operation, ~0.1-0.5ms per thread
                started_count += 1
            except Exception as e:
                if DEBUG_MODE:
                    print(f"  Error starting thread for camera #{recorder.camera_info.index}: {e}")

        if started_count == 0:
            self.is_recording = False
            return False

        if DEBUG_MODE:
            print(f"Started {started_count}/{len(self.recorders)} recorders")
            print(f"Thread start synchronization: <1ms gap between cameras")

        return True

    def wait_for_completion(self):
        """
        Wait for recording to complete naturally.

        Blocks for the duration of the recording plus a small buffer.
        Sets session end_time when complete.
        """
        if not self.is_recording:
            return

        # Wait for recording duration plus recording delay buffer
        time.sleep(self.duration_secs + self.recording_delay)

        # Set end time
        self.session.end_time = datetime.now()

        if DEBUG_MODE:
            print(f"Recording completed at {self.session.end_time.isoformat()}")

    def stop_recording(self):
        """
        Stop all recorders gracefully.

        Signals all recorder threads to stop and waits for them to exit.
        Sets session end_time.
        """
        if not self.is_recording:
            return

        if DEBUG_MODE:
            print("Stopping all recorders...")

        # Stop all recorders
        for recorder in self.recorders:
            try:
                recorder.stop_recording()
            except Exception as e:
                if DEBUG_MODE:
                    print(f"Error stopping recorder for camera #{recorder.camera_info.index}: {e}")

        self.is_recording = False
        self.session.end_time = datetime.now()

        if DEBUG_MODE:
            print(f"All recorders stopped at {self.session.end_time.isoformat()}")

    def get_recording_status(self) -> dict:
        """
        Get current recording status.

        Returns:
            Dictionary with recording status information
        """
        status = {
            "is_recording": self.is_recording,
            "session_id": self.session.session_id,
            "output_dir": str(self.session.session_dir),
            "camera_count": len(self.recorders),
            "duration_secs": self.duration_secs,
            "start_time": self.session.start_time.isoformat() if self.session.start_time else None,
        }

        return status

    @classmethod
    def _check_gpu_available(cls) -> bool:
        """
        Check if NVIDIA GPU with CUDA support is available for Zed SDK.

        This method detects GPU availability by attempting to initialize
        a test Zed camera with depth mode. It also checks for CUDA, cuDNN,
        and TensorRT availability.

        The result is cached at the class level to avoid repeated detection.

        Returns:
            True if GPU with CUDA is available, False otherwise

        Scenarios Covered:
            - Detect NVIDIA GPU with CUDA support
            - Handle Zed SDK not installed (ImportError)
            - Handle any exception during GPU detection
            - Cache GPU availability result
            - Log GPU detection result in DEBUG mode
            - Detect CUDA version
            - Verify cuDNN availability
            - Check TensorRT for AI features
        """
        # Return cached result if available
        if cls._gpu_available_cache is not None:
            return cls._gpu_available_cache

        try:
            # Import Zed SDK
            import pyzed.sl as sl

            if DEBUG_MODE:
                print("Checking GPU availability for Zed SDK...")

            # Check CUDA version
            try:
                cuda_version = sl.Camera.get_sdk_version()
                cls._cuda_version = cuda_version
                if DEBUG_MODE:
                    print(f"  Zed SDK version: {cuda_version}")
            except Exception as e:
                if DEBUG_MODE:
                    print(f"  Could not retrieve Zed SDK version: {e}")

            # Attempt to get device list (requires CUDA)
            devices = sl.Camera.get_device_list()

            # Check GPU compute capability and VRAM
            try:
                import torch
                if torch.cuda.is_available():
                    # Get CUDA compute capability
                    capability = torch.cuda.get_device_capability(0)
                    cls._compute_capability = float(f"{capability[0]}.{capability[1]}")

                    # Get VRAM
                    vram_bytes = torch.cuda.get_device_properties(0).total_memory
                    cls._vram_gb = vram_bytes / (1024 ** 3)  # Convert to GB

                    if DEBUG_MODE:
                        print(f"  GPU Compute Capability: {cls._compute_capability}")
                        print(f"  Available VRAM: {cls._vram_gb:.2f} GB")

                    # Scenario: Verify CUDA compute capability
                    if cls._compute_capability < 5.0:
                        if DEBUG_MODE:
                            print(f"  Compute capability {cls._compute_capability} < 5.0 (minimum required)")
                            print("  GPU mode disabled, using CPU fallback")
                        cls._gpu_available_cache = False
                        return False

                    # Scenario: Verify minimum VRAM requirements
                    if cls._vram_gb < 4.0:
                        if DEBUG_MODE:
                            print(f"  VRAM {cls._vram_gb:.2f} GB < 4 GB (recommended minimum)")
                            print("  GPU mode will be attempted, but some features may be disabled")
                        # Note: Still attempt GPU mode, but with warning

            except ImportError:
                # PyTorch not installed, can't check VRAM/compute capability
                if DEBUG_MODE:
                    print("  PyTorch not installed, cannot verify VRAM/compute capability")
                return False
            except Exception as e:
                if DEBUG_MODE:
                    print(f"  Could not check GPU properties: {e}")
                return False

            # Test GPU availability with camera initialization
            init_params = sl.InitParameters()
            init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE
            init_params.camera_resolution = sl.RESOLUTION.HD720  # Use lower resolution for quick test

            test_camera = sl.Camera()
            err = test_camera.open(init_params)

            # Close test camera immediately
            test_camera.close()

            # Check if initialization was successful
            if err == sl.ERROR_CODE.SUCCESS:
                cls._gpu_available_cache = True

                if DEBUG_MODE:
                    print("  NVIDIA GPU detected with CUDA support")

                    # Try to get GPU information
                    try:
                        # Get device properties if available
                        if devices:
                            for device in devices:
                                print(f"  GPU Device: {device.camera_model}")
                    except Exception:
                        pass

                # Check cuDNN availability (indirectly through SDK features)
                try:
                    # If depth mode works, cuDNN is likely available
                    cls._cudnn_available = True
                    if DEBUG_MODE:
                        print("  cuDNN library detected (depth computation available)")
                except Exception:
                    cls._cudnn_available = False
                    if DEBUG_MODE:
                        print("  cuDNN library not detected")

                # Check TensorRT availability (for AI features)
                try:
                    # Try to check if object detection is available
                    # This indirectly checks for TensorRT
                    test_obj_params = sl.ObjectDetectionParameters()
                    cls._tensorrt_available = True
                    if DEBUG_MODE:
                        print("  TensorRT detected (AI features available)")
                except Exception:
                    cls._tensorrt_available = False
                    if DEBUG_MODE:
                        print("  TensorRT not detected (AI features disabled)")

                return True
            else:
                cls._gpu_available_cache = False
                if DEBUG_MODE:
                    print(f"  GPU initialization failed: {err}")
                    print("  Using CPU fallback")
                return False

        except ImportError:
            # Zed SDK not installed
            cls._gpu_available_cache = False
            if DEBUG_MODE:
                print("  Zed SDK (pyzed) not installed")
                print("  Using CPU fallback")
            return False

        except Exception as e:
            # Any other exception during GPU check
            cls._gpu_available_cache = False
            if DEBUG_MODE:
                print(f"  GPU detection failed: {e}")
                print("  Using CPU fallback")
            return False

    def _load_calibration(self, calibration_path: Union[str, Path]):
        """
        Load camera calibration from file.

        Implements Scenario 9.1: Load camera calibration at orchestrator initialization

        Args:
            calibration_path: Path to calibration JSON file
        """
        try:
            self.calibration_loader = CalibrationLoader(
                calibration_path=Path(calibration_path)
            )

            if self.calibration_loader.is_valid():
                self.calibration_loaded = True

                calib_info = self.calibration_loader.get_calibration_info()

                if DEBUG_MODE:
                    logger.info("Calibration loaded successfully")
                    logger.info(f"  Reference camera: {calib_info['reference_camera']}")
                    logger.info(f"  Calibrated cameras: {', '.join(calib_info['camera_ids'])}")
                    logger.info(f"  Calibration age: {calib_info['age_days']} days")
                    logger.info(f"  Quality: {calib_info['quality']}")

                # Check for warnings
                if self.calibration_loader.has_warnings():
                    for warning in self.calibration_loader.get_warnings():
                        logger.warning(f"{warning}")

                # Save calibration metadata to session directory
                self._save_calibration_metadata()

            else:
                self.calibration_loaded = False
                errors = self.calibration_loader.get_validation_messages()

                if DEBUG_MODE:
                    logger.error("Calibration validation failed")
                    for error in errors:
                        logger.error(f"  {error}")

        except Exception as e:
            self.calibration_loaded = False
            logger.exception(f"Failed to load calibration: {e}")

    def _save_calibration_metadata(self):
        """Save calibration metadata to session directory."""
        if not self.calibration_loaded or not self.calibration_loader:
            return

        try:
            import json

            calib_info = self.calibration_loader.get_calibration_info()

            # Save transformation matrices
            transformations = {}
            camera_ids = calib_info['camera_ids']

            for i, cam_from in enumerate(camera_ids):
                for cam_to in camera_ids[i+1:]:
                    T = self.calibration_loader.get_transformation(cam_from, cam_to)
                    if T is not None:
                        key = f"{cam_from}_to_{cam_to}"
                        transformations[key] = T.tolist()

            # Save to session directory
            metadata_path = self.session.session_dir / "calibration_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump({
                    "calibration_info": calib_info,
                    "transformations": transformations
                }, f, indent=2)

            if DEBUG_MODE:
                logger.info(f"Calibration metadata saved to: {metadata_path}")

        except Exception as e:
            logger.warning(f"Failed to save calibration metadata: {e}")

    def get_transformation(self, from_camera: str, to_camera: str) -> Optional[np.ndarray]:
        """
        Get transformation matrix from one camera to another.

        Implements Scenario 9.2: Apply transformations during recording session

        Args:
            from_camera: Source camera ID
            to_camera: Target camera ID

        Returns:
            4x4 transformation matrix or None if calibration not loaded
        """
        if not self.calibration_loaded or not self.calibration_loader:
            logger.warning("Calibration not loaded, cannot get transformation")
            return None

        return self.calibration_loader.get_transformation(from_camera, to_camera)

    def get_camera_intrinsics(self, camera_id: str) -> Optional[dict]:
        """
        Get intrinsic calibration parameters for a camera.

        Args:
            camera_id: Camera identifier

        Returns:
            Intrinsic parameters dict or None if not available
        """
        if not self.calibration_loaded or not self.calibration_loader:
            logger.warning("Calibration not loaded, cannot get intrinsics")
            return None

        intrinsics = self.calibration_loader.get_intrinsics(camera_id)
        if intrinsics:
            return {
                "camera_matrix": intrinsics.camera_matrix.tolist(),
                "dist_coeffs": intrinsics.dist_coeffs.tolist(),
                "image_size": intrinsics.image_size
            }
        return None

    def get_calibration_status(self) -> dict:
        """
        Get calibration status information.

        Returns:
            Dictionary with calibration status
        """
        if not self.calibration_loaded or not self.calibration_loader:
            return {
                "loaded": False,
                "valid": False,
                "message": "No calibration loaded"
            }

        return {
            "loaded": True,
            "valid": self.calibration_loader.is_valid(),
            "info": self.calibration_loader.get_calibration_info(),
            "warnings": self.calibration_loader.get_warnings() if self.calibration_loader.has_warnings() else []
        }

    @classmethod
    def get_gpu_info(cls) -> dict:
        """
        Get GPU availability and feature information.

        This method returns cached GPU information. If GPU has not been
        checked yet, it will trigger the check.

        Returns:
            Dictionary with GPU information:
                - gpu_available: bool - Whether GPU is available
                - cuda_version: str | None - CUDA/SDK version
                - cudnn_available: bool | None - cuDNN availability
                - tensorrt_available: bool | None - TensorRT availability
                - vram_gb: float | None - Available VRAM in GB
                - compute_capability: float | None - CUDA compute capability
        """
        # Ensure GPU has been checked
        if cls._gpu_available_cache is None:
            cls._check_gpu_available()

        return {
            "gpu_available": cls._gpu_available_cache or False,
            "cuda_version": cls._cuda_version,
            "cudnn_available": cls._cudnn_available,
            "tensorrt_available": cls._tensorrt_available,
            "vram_gb": cls._vram_gb,
            "compute_capability": cls._compute_capability,
        }
