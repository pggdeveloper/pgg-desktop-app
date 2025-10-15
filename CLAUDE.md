# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A PyQt5 desktop application for **multi-camera cattle monitoring** using Intel RealSense D455/D455i and Stereolabs Zed 2/2i cameras. The application captures synchronized video streams, depth data, and performs real-time cattle health analysis including body condition scoring, volume estimation, motion tracking, and behavioral analysis.

**Target Platforms:** Windows 10/11 (primary), macOS 12+, Ubuntu 20.04+
**Python Version:** 3.11+
**Key Technologies:** PyQt5, OpenCV, Intel RealSense SDK, Zed SDK, Open3D

## Setup and Development Commands

### Environment Setup
```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Application
```bash
# Run the main application
python app.py
```

### Building for Windows
```bash
# Automated build (recommended)
build_windows_exe.bat

# Manual build
python build_spec_generator.py
pyinstaller --clean --noconfirm pgg_app.spec
```

Note: Building requires separate installation of Intel RealSense SDK and Zed SDK on the target system.

## Architecture Overview

### Application Entry Point

**app.py**: Initializes PyQt5 with high-DPI support, applies global Fusion style and dark theme from `THEME["qss"]["base"]`, and creates MainWindow.

### Screen Management

**screens/main_screen.py**: MainWindow orchestrates screen navigation using QStackedWidget with crossfade animations. Currently displays VideoScreen directly (LoginScreen available but commented out). Session data flows through `set_session_data()` slots.

**screens/video_screen.py**: Multi-camera recording interface for synchronized capture. Detects RealSense and Zed cameras, initializes CameraRecordingOrchestrator, and runs recording workflow in background thread with Qt signals for thread-safe UI updates.

**screens/capture_screen.py**: Simple single-camera photo capture (legacy feature, not primary focus).

**screens/login_screen.py**: Authentication screen (currently disabled) with POST requests to API endpoint.

### Camera Architecture

The application has a sophisticated multi-camera system with three main layers:

#### 1. Camera Detection & Enumeration (`utils/utils.py`)

**enumerate_usb_external_cameras(detect_specialized=True)**: Cross-platform USB camera detection with specialized camera identification.

- **Windows**: PowerShell PnP Device enumeration + DirectShow backend
- **macOS**: AVFoundation API (via PyObjC) + system_profiler for USB detection
- **Linux**: pyudev or /dev/v4l/by-id symlinks + V4L2 backend

Returns list of `CameraInfo` objects with:
- Camera type detection (RealSense D455/D455i, Zed 2/2i, Generic)
- USB VID/PID identification
- Capabilities (depth, IMU, stereo)
- SDK availability detection
- Backend-specific open hints

**filter_cameras_by_type(cameras, types)**: Filters camera list by specified types.

**open_camera_hint(hint)**: Opens camera using appropriate backend based on hint (index or device path).

#### 2. Camera Domain Models

**domain/camera_info.py**: Complete camera metadata structure returned by enumeration.

**domain/camera_type.py**: Enum defining supported camera types with helper properties:
- `is_realsense`, `is_zed` - Type checking
- `has_imu`, `has_depth` - Capability checking

**domain/camera_backend.py**: Enum for OpenCV backends (DirectShow, AVFoundation, V4L2).

**domain/camera_capabilities.py**: Data structure for camera capabilities (depth, IMU, stereo).

**domain/recording_session.py**: Session management with timestamp-based directory structure.

#### 3. Camera Recorders

**utils/realsense_camera.py** - `RealSenseCameraRecorder`:
- Multi-stream capture: RGB, Depth, Infrared (left/right), IMU
- Depth saved as individual .npz files (compressed NumPy) with metadata
- Hardware timestamp synchronization
- Frame alignment (depth to color)
- Advanced features: Volume estimation, motion analysis, scene understanding, IMU integration, classical CV health analysis

**utils/zed_camera.py** - `ZedCameraRecorder`:
- Dual initialization strategy: Zed SDK (CPU mode) first, OpenCV fallback
- Side-by-side stereo capture (3840x1080): left, right, and combined streams
- IMU data recording (SDK only)
- Depth processing done offline after capture (see `utils/zed_offline_depth_processor.py`)

**utils/zed_camera_sdk.py** - `ZedCameraSDKRecorder`:
- GPU-accelerated recording using Zed SDK
- High-precision depth modes (ULTRA)
- Advanced SDK features: Positional tracking, spatial mapping, object detection, body tracking
- Requires NVIDIA GPU with CUDA support

#### 4. Multi-Camera Orchestration

**utils/camera_orchestrator.py** - `CameraRecordingOrchestrator`:

Core responsibilities:
- Automatic recorder creation based on camera type and GPU availability
- Two-phase synchronized recording start:
  - **Phase 1**: Prepare all recorders (slow operations: file creation, video writers)
  - **Phase 2**: Start all threads in rapid succession (<1ms gap)
- Synchronized timestamp generation
- Thread-based concurrent recording with isolated error handling
- Session management with output directory structure
- Calibration loading and transformation matrix management

**GPU Detection Logic**:
```python
# Checks GPU availability and creates appropriate recorder
_check_gpu_available()  # Class-level caching
_create_recorder(camera_info)  # Returns ZedCameraSDKRecorder or ZedCameraRecorder
```

**Recording Flow**:
```python
1. initialize_cameras() → Returns (success_count, failed_count)
2. start_recording_with_countdown() → Countdown + Phase 1 (prepare) + Phase 2 (start threads)
3. wait_for_completion() → Blocks for recording duration
4. stop_recording() → Signals all recorders to stop
```

### Configuration & Theme System

**config.py**: Application-wide configuration with all runtime parameters:
- API endpoints (BASE_API_URL, LOGIN_ENDPOINT_URL)
- Debug flags (DEBUG_MODE, PHOTO_PREVIEW)
- Recording parameters (VIDEO_SECS, FPS, START_RECORDING_DELAY)
- Resolution settings (REQ_WIDTH, REQ_HEIGHT, REALSENSE_WIDTH, REALSENSE_HEIGHT)
- File formats (SAVE_JPEG)
- Directory structure (BASE_DIR)

**constants.py**: Centralized theme system with dark color palette:
- `APP`: Window dimensions, title, border radius
- `THEME`: Complete design system
  - `colors`: Semantic color names (bg, primary, text_muted, etc.)
  - `sizes`: Button heights, input dimensions, content widths
  - `anim`: Hover/press/fade durations, easing curves, opacity values
  - `qss`: Global QSS stylesheet for all UI components

**Important**: All UI styling goes through THEME. Never hardcode colors or sizes in components.

### Components

**components/animated_button.py**: Custom QPushButton with QGraphicsOpacityEffect for hover/press animations. Uses THEME for animation timing. Note: Only one QGraphicsEffect per widget; use wrapper for drop shadows.

**components/password_field.py**: QLineEdit with embedded QToolButton toggle for show/hide password. Respects global QSS styling.

### Camera Calibration System

**utils/calibration_loader.py**: Loads multi-camera calibration JSON with transformation matrices and intrinsics.

**scripts/calibrate_multi_camera.py**: ChArUco board-based multi-camera calibration workflow.

**utils/multi_camera_calibration.py**: Core calibration algorithms.

### Cattle Analysis Modules

**Volume Estimation** (`utils/cow_volume_and_dimensional_measurements.py`):
- Point cloud-based volume calculation (convex hull, voxel grid, alpha shapes)
- Body dimension measurements (length, height, width, girth)
- Weight estimation from volume and dimensions
- Temporal tracking of volume changes

**Motion Analysis** (`utils/cow_motion_analysis.py`):
- Optical flow calculation (dense and sparse)
- Activity level classification
- Real-world speed and acceleration measurement
- Gait analysis (step frequency, stride length, symmetry)

**Scene Understanding** (`utils/cow_scene_understanding.py`):
- Plane detection (floor, walls, ceiling)
- Object identification (feed troughs, water sources)
- Environmental analysis (lighting, shadows, wetness/mud)

**IMU Integration** (`utils/cow_imu_integration.py`):
- Sensor fusion for orientation estimation (Madgwick/Mahony filters)
- Vibration and motion blur analysis
- Camera pose estimation from IMU
- IMU-frame synchronization

**Classical CV Health Analysis** (`utils/cow_classical_cv.py`):
- Body Condition Scoring (BCS) via anatomical feature detection
- Health monitoring (coat texture, lesions, anemia, respiratory issues)
- Behavioral analysis (lameness, posture, activity)
- Individual identification (coat patterns, facial features)
- Environmental welfare assessment

### Point Cloud Processing

**utils/realsense_point_cloud_generator.py**: Generate point clouds from RealSense depth data offline.

**utils/zed_point_cloud_generator.py**: Generate point clouds from Zed depth data offline.

**utils/multi_camera_point_cloud_fusion_workflow.py**: Fuse point clouds from multiple cameras using calibration.

**scripts/fuse_multi_camera_point_clouds.py**: Command-line tool for point cloud fusion.

### Depth Processing

**Note**: Depth processing is done **offline** after capture, not during recording.

**utils/zed_offline_depth_processor.py**: Processes all stereo frames to generate depth maps (Phase 1).

**utils/selective_depth_processor.py**: Processes only frames containing animals using YOLO detection (Phase 2 - RECOMMENDED).

## Important Patterns and Conventions

### Thread-Safe UI Updates

Always use Qt signals for cross-thread communication:
```python
# In __init__
self._update_feedback_signal = QtCore.pyqtSignal(str, str)
self._update_feedback_signal.connect(self._update_feedback)

# In background thread
self._update_feedback_signal.emit("Message", "Style")

# In main thread
@QtCore.pyqtSlot(str, str)
def _update_feedback(self, message: str, style: str):
    self.feedback.setText(message)
```

### Camera Recording Workflow

```python
# 1. Enumerate cameras with specialized detection
cameras = enumerate_usb_external_cameras(detect_specialized=True)

# 2. Filter for specialized cameras
specialized = filter_cameras_by_type(cameras, [
    CameraType.REALSENSE_D455,
    CameraType.REALSENSE_D455i,
    CameraType.ZED_2,
    CameraType.ZED_2i
])

# 3. Create orchestrator
orchestrator = CameraRecordingOrchestrator(cameras=specialized)

# 4. Initialize cameras
success_count, failed_count = orchestrator.initialize_cameras()

# 5. Start recording in background thread
workflow_thread = threading.Thread(
    target=self._recording_workflow,
    args=(orchestrator,),
    daemon=True
)
workflow_thread.start()

# In workflow thread:
orchestrator.start_recording_with_countdown()
orchestrator.wait_for_completion()
orchestrator.stop_recording()
```

### Theme Application

Global QSS applied once in app.py:
```python
app.setStyleSheet(THEME["qss"]["base"])
```

Widgets use `objectName` for specific styling:
```python
widget.setObjectName("Card")      # Card styling
label.setObjectName("H1")         # Header styling
label.setObjectName("Muted")      # Muted text styling
button.setObjectName("PrimaryButton")  # Primary button styling
```

### Screen Transitions

```python
self._crossfade_to(target_widget)
# Fade out current → setCurrentWidget → Fade in target
# Animations stored in self._anim_guard to prevent GC
```

### File Naming Convention

All recorded files follow the format:
```
YYYY-MM-dd-HH-mm-ss-sss-{CAMERA_INDEX}-{SENSOR}.{ext}

Examples:
2025-10-15-14-30-25-123-0-rgb.mp4
2025-10-15-14-30-25-123-0-depth_000001.npz
2025-10-15-14-30-25-123-0-ir_left.mp4
2025-10-15-14-30-25-123-0-imu.csv
2025-10-15-14-30-25-123-0-timestamps.csv
```

### Session Directory Structure

```
./data/
└── YYYY-MM-dd-HH-mm-ss-{SESSION_ID}/
    ├── 2025-10-15-14-30-25-123-0-rgb.mp4
    ├── 2025-10-15-14-30-25-123-0-depth_000001.npz
    ├── 2025-10-15-14-30-25-123-0-depth_000002.npz
    ├── ...
    ├── 2025-10-15-14-30-25-123-0-ir_left.mp4
    ├── 2025-10-15-14-30-25-123-0-ir_right.mp4
    ├── 2025-10-15-14-30-25-123-0-imu.csv
    ├── 2025-10-15-14-30-25-123-0-timestamps.csv
    ├── 2025-10-15-14-30-25-123-0-metadata.json
    ├── 2025-10-15-14-30-25-123-1-rgb.mp4  (second camera)
    └── calibration_metadata.json
```

### Platform-Specific Considerations

**Windows**:
- Multi-camera recording fully supported
- RealSense SDK and Zed SDK must be installed separately
- DirectShow backend for cameras
- Visual C++ Redistributables 2015-2022 required

**macOS**:
- Camera enumeration uses AVFoundation + pyobjc
- Camera permissions must be granted in System Preferences
- Multi-camera recording limited (test thoroughly)

**Linux**:
- Camera enumeration uses pyudev + V4L2
- RealSense SDK: `sudo apt-get install librealsense2-dev`
- Multi-camera recording limited (test thoroughly)

### Depth Data Storage

**RealSense depth**: Saved as individual .npz files (compressed NumPy format):
- Advantages: 40% smaller size, 3-5x faster loading, no data loss (uint16 preserved)
- Format: `depth_NNNNNN.npz` with metadata (timestamp, frame_number, depth_scale, camera_index)
- Load with: `data = np.load(filename); depth = data['depth']; timestamp = data['timestamp']`

**Zed depth**: Generated offline after capture:
- Phase 1: `ZedOfflineDepthProcessor` - processes all frames (comprehensive but slow)
- Phase 2: `SelectiveDepthProcessor` - processes only frames with animals using YOLO (RECOMMENDED)

## Key Files to Review

- `utils/camera_orchestrator.py` - Multi-camera synchronization logic
- `utils/realsense_camera.py` - RealSense recording implementation
- `utils/zed_camera.py` - Zed recording implementation (CPU mode)
- `screens/video_screen.py` - Main recording UI
- `config.py` - All configuration parameters
- `constants.py` - Theme system and styling
- `docs/dev_handover.md` - Detailed checklist for development handover

## Common Tasks

### Adding a New Analysis Module

1. Create module in `utils/` following existing patterns (e.g., `cow_volume_and_dimensional_measurements.py`)
2. Add configuration flags to `config.py` if needed
3. Import and initialize in `RealSenseCameraRecorder.__init__()` with feature flag
4. Add processing method called from `_recording_loop()` at appropriate interval
5. Export results to CSV using consistent filename patterns

### Modifying Recording Parameters

Edit `config.py`:
- `VIDEO_SECS`: Recording duration
- `FPS`: Frame rate (15, 30, 60, 90 for RealSense)
- `START_RECORDING_DELAY`: Countdown before recording starts
- `REALSENSE_WIDTH` / `REALSENSE_HEIGHT`: Resolution

### Adding New Camera Type

1. Add enum value to `domain/camera_type.py`
2. Update detection logic in `utils/utils.py` platform-specific functions
3. Add VID/PID mapping if known
4. Create recorder class if needed or extend existing
5. Update `CameraRecordingOrchestrator._create_recorder()` to handle new type

### Debugging Multi-Camera Issues

1. Set `DEBUG_MODE = True` in `config.py`
2. Check console output for camera enumeration results
3. Verify USB VID/PID matches expected values
4. Check SDK installation (RealSense SDK / Zed SDK)
5. Review timestamp logs in session directory: `*-timestamps.csv`
6. Check temporal validation summary at end of recording

## Testing

Manual testing checklist:
- Camera detection on each platform
- Multi-camera synchronized recording
- Frame timestamp consistency (check timestamp CSV)
- Output file creation and naming
- Metadata JSON generation
- Session directory structure
- Calibration loading (if available)
- Analysis module outputs (volume, motion, etc.)
- UI responsiveness during recording
- Thread cleanup after recording completion

## Build System

Uses PyInstaller for Windows .exe generation:
- `build_spec_generator.py` - Generates .spec file dynamically
- `build_windows_exe.bat` - Automated build script
- `pgg_app.spec` - Generated PyInstaller specification
- See `BUILD_INSTRUCTIONS.md` for detailed build documentation
