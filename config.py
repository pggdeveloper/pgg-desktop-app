from pathlib import Path

BASE_API_URL        = "http://localhost:3001/api/v1/"
LOGIN_ENDPOINT_URL  = "auth/login"
DEBUG_MODE          = True
PHOTO_PREVIEW       = False
BASE_DIR            = Path("./data")    # carpeta base de datasets
CAM_INDEX           = 0                 # índice del dispositivo (0,1,2,…)
FPS                 = 30                # objetivo de FPS
PHOTO_COUNT         = 15                # nº de fotos por ojo
VIDEO_SECS          = 60                # duración del vídeo
# Si tu cámara UVC soporta 3840x1080 (1080p por ojo), ponlos aquí:
REQ_WIDTH           = 2560              # 2 * 1280       3840             # 2 * 1920 (SBS)
REQ_HEIGHT          = 720               # 1080
REALSENSE_WIDTH     = 1280              # 1920             # por ojo
REALSENSE_HEIGHT    = 720               # 1080             # por ojo
SAVE_JPEG           = False             # True=JPEG (más liviano), False=PNG (sin pérdida)
PHOTO_DELAY         = 3                 # segundos entre fotos
SAVE_PHOTOS         = True              # True=guardar fotos, False=no guardar fotos
START_RECORDING_DELAY = 5               # segundos antes de empezar a grabar

# ==============================================================================
# CAMERA DEVICE PATH DETECTION (Windows-specific, stable identification)
# ==============================================================================
#
# This configuration enables stable camera identification using Windows Device
# Instance Paths instead of unreliable numeric OpenCV indices.
#
# PROBLEM: OpenCV indices (0, 1, 2) change when:
#   - USB ports are swapped
#   - Cameras are connected/disconnected in different order
#   - Other cameras are added to the system
#   - System reboots
#
# SOLUTION: Windows Device Instance Paths are STABLE identifiers that:
#   - Do NOT change when USB ports are swapped (serial-based matching)
#   - Uniquely identify each physical camera
#   - Survive system reboots
#
# HOW TO FIND DEVICE PATHS:
#   1. Connect all your cameras
#   2. Run: python scripts/diagnose_camera_indices.py
#   3. Look for "OS ID" field in the output for each camera
#   4. Copy the full InstanceId string (format: USB\VID_xxxx&PID_xxxx&MI_xx\...)
#   5. Paste it into PREFERRED_CAMERA_DEVICE_PATHS below
#
# DEVICE PATH FORMAT:
#   USB\VID_{vendor}&PID_{product}&MI_{interface}\{serial}&{port}&{endpoint}
#
#   Example: USB\VID_2B03&PID_F880&MI_00\7&1EBA99DD&0&0000
#            ^^^^ ^^^^ ^^^^ ^^^^  ^^^^   ^^^^^^^^^^^
#            |    |    |    |     |      Serial number (STABLE, unique per camera)
#            |    |    |    |     Interface (00=primary video)
#            |    |    |    Product ID (F880=ZED 2i)
#            |    |    Vendor ID (2B03=Stereolabs, 8086=Intel)
#            USB device
#
# MULTI-CAMERA SUPPORT:
#   - List multiple paths for same camera type
#   - Order matters: first = primary, second = secondary, etc.
#   - System will use cameras in the order specified
#
# MULTI-INTERFACE CAMERAS (e.g., RealSense):
#   - Only specify ONE interface (MI_00 for depth or MI_03 for RGB)
#   - System automatically detects all interfaces from the same camera
#   - All interfaces share the same base serial number
#
# ==============================================================================

PREFERRED_CAMERA_DEVICE_PATHS = {
    # RealSense D455i primary camera
    # NOTE: Only specify ONE interface (MI_00 recommended)
    # The system will automatically find all other interfaces (RGB, IR, IMU)
    "realsense_primary": [
        "USB\\VID_8086&PID_0B5C&MI_00\\6&D37468C&1&0000",  # Primary RealSense D455i
    ],

    # ZED 2i cameras (supports multiple units)
    # List in order of preference: first = primary, second = secondary
    # Each ZED 2i has a unique serial number in the path
    "zed_cameras": [
        "USB\\VID_2B03&PID_F880&MI_00\\7&1EBA99DD&0&0000",  # ZED 2i #1 (primary)
        "USB\\VID_2B03&PID_F880&MI_00\\7&1500F77&0&0000",   # ZED 2i #2 (secondary)
    ],
}

# ==============================================================================
# FALLBACK BEHAVIOR (when configured cameras are not found)
# ==============================================================================
#
# Controls what happens when a configured camera is missing or not found.
#
# OPTIONS:
#
#   "strict" (RECOMMENDED):
#       - ONLY use cameras listed in PREFERRED_CAMERA_DEVICE_PATHS
#       - If a configured camera is not found, log ERROR but DO NOT use another camera
#       - Other connected cameras are completely IGNORED
#       - This ensures you ONLY record with the exact cameras you specify
#       - Use this when you need precise camera control
#
#   "sdk_exclusion":
#       - Use SDK-based detection as fallback
#       - If configured camera missing, use first available camera of same type
#       - Example: If ZED #1 missing, use any available ZED camera
#       - Less strict, more flexible
#
#   "first_available":
#       - Use first available camera of requested type
#       - Least restrictive mode
#       - Not recommended for production
#
# EXAMPLE BEHAVIOR (strict mode):
#
#   Connected cameras:
#     - ZED 2i #1 (7&1EBA99DD) <- CONFIGURED
#     - ZED 2i #2 (7&1500F77) <- CONFIGURED
#     - ZED 2i #3 (7&AAAA111) <- NOT CONFIGURED
#     - RealSense (6&D37468C) <- CONFIGURED
#
#   Cameras used for recording: 3 cameras
#     - ZED 2i #1 (7&1EBA99DD)
#     - ZED 2i #2 (7&1500F77)
#     - RealSense (6&D37468C)
#
#   ZED 2i #3 is IGNORED (not in config)
#
# ==============================================================================

CAMERA_DETECTION_FALLBACK_MODE = "strict"  # Options: "strict", "sdk_exclusion", "first_available"

# ==============================================================================
# ENABLE DEVICE PATH DETECTION
# ==============================================================================
#
# Master switch for device path-based camera detection.
#
# True:  Enable device path detection (Windows only, auto-disabled on other platforms)
# False: Use traditional OpenCV index-based detection only
#
# NOTE: Even when enabled, the system falls back to traditional detection if:
#   - Platform is not Windows
#   - PREFERRED_CAMERA_DEVICE_PATHS is empty or not configured
#   - Device path resolution fails
#
# ==============================================================================

ENABLE_DEVICE_PATH_DETECTION = True

# ==============================================================================
# DEBUG OUTPUT FOR DEVICE PATH RESOLUTION
# ==============================================================================
#
# Enable verbose logging for device path resolution process.
#
# True:  Show detailed logs about:
#        - Device path parsing
#        - Serial number extraction
#        - Camera matching attempts
#        - Resolution results
#        - Fallback decisions
#
# False: Minimal logging (production mode)
#
# RECOMMENDATION: Set to True during development/debugging, False in production
#
# ==============================================================================

DEBUG_DEVICE_PATH_RESOLUTION = True  # Set to False in production
