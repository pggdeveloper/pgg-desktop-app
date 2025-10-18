"""
Diagnostic script to test all camera indices and check resolutions.
Identifies which index provides ZED 2i stereo side-by-side format.

UPDATED (2025-10-18): Now shows Windows Device Instance Paths (InstanceId)
and generates configuration for stable camera identification.

This script tests all camera indices from 0 to 10 and checks:
- Which indices can be opened
- Default resolution for each camera
- Aspect ratio (to detect stereo side-by-side)
- Which resolutions each camera supports
- SDK-based identification (RealSense vs ZED)
- Windows Device Instance Paths (InstanceId) for stable identification
- Serial numbers extracted from device paths

Critical for debugging ZED 2i detection issues and camera configuration.
"""
import cv2
import sys
import os
import platform

# Add parent directory to path to import SDK exclusion functions
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

TESTED_RESOLUTIONS = [
    (3840, 1080, "Full HD Stereo (ZED 2i native)"),
    (2560, 720, "HD Stereo (ZED 2i HD)"),
    (1920, 1080, "Full HD Single Camera"),
    (1280, 720, "HD Single Camera"),
    (1344, 376, "VGA Stereo (ZED 2i VGA)"),
    (640, 480, "VGA"),
]

def get_camera_metadata(index: int):
    """
    Get camera metadata including InstanceId (Windows only).

    Returns:
        Dict with camera metadata or None if not available
    """
    if platform.system() != "Windows":
        return None

    try:
        from utils.utils import enumerate_usb_external_cameras
        cameras = enumerate_usb_external_cameras(use_device_path_resolution=False)

        # Find camera with matching index
        for cam in cameras:
            if cam.index == index:
                return {
                    "name": cam.name,
                    "os_id": cam.os_id,
                    "usb_vid": cam.usb_vid,
                    "usb_pid": cam.usb_pid,
                    "camera_type": str(cam.camera_type),
                }
        return None
    except Exception:
        return None


def extract_and_display_serial(instance_id: str) -> str:
    """
    Extract serial number from Windows Device Instance ID.

    Args:
        instance_id: Full InstanceId string

    Returns:
        Extracted serial or "N/A" if not extractable
    """
    if not instance_id:
        return "N/A"

    try:
        from utils.camera_device_path_resolver import extract_serial_from_instance_id
        serial = extract_serial_from_instance_id(instance_id)
        return serial if serial else "N/A"
    except ImportError:
        return "N/A"


def test_camera_index(index: int) -> dict:
    """
    Test a camera index with DirectShow backend.

    Args:
        index: Camera index to test (0-10)

    Returns:
        Dict with test results including:
        - openable: bool
        - default_resolution: tuple (width, height, fps)
        - supported_modes: list of tuples (width, height, description)
        - is_stereo_candidate: bool (aspect ratio 3.0-4.0)
        - sdk_is_realsense: bool or None (None if SDK not available)
        - sdk_status: str or None
        - metadata: dict or None (Windows InstanceId, VID/PID, etc.)
    """
    results = {
        "index": index,
        "openable": False,
        "default_resolution": None,
        "supported_modes": [],
        "is_stereo_candidate": False,
        "sdk_is_realsense": None,
        "sdk_status": None,
        "metadata": None,
    }

    # Import SDK exclusion function
    try:
        from utils.camera_identification_sdk import test_index_is_realsense_via_sdk_robust
        sdk_available = True
    except ImportError:
        sdk_available = False

    try:
        # Open with DirectShow (Windows)
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)

        if not cap.isOpened():
            print(f" Index {index}: Cannot open")
            return results

        results["openable"] = True

        # Get default resolution WITHOUT setting anything
        # This is what the camera defaults to when opened
        default_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        default_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        default_fps = cap.get(cv2.CAP_PROP_FPS)

        results["default_resolution"] = (default_width, default_height, default_fps)

        print(f"\n Index {index}: OPENED")
        print(f"   Default: {default_width}x{default_height} @ {default_fps:.1f}fps")

        # Calculate aspect ratio
        aspect = default_width / default_height if default_height > 0 else 0
        print(f"   Aspect ratio: {aspect:.2f}")

        # Check if stereo aspect ratio (3.0 to 4.0)
        # ZED stereo side-by-side has aspect ratio ~3.56 (2560/720 or 3840/1080)
        if 3.0 <= aspect <= 4.0:
            results["is_stereo_candidate"] = True
            print(f"    STEREO CANDIDATE (aspect {aspect:.2f}) ")

        # Try to read a frame to verify camera actually works
        ret, frame = cap.read()
        if ret and frame is not None:
            print(f"   Frame read:  ({frame.shape[1]}x{frame.shape[0]} HxWxC: {frame.shape})")
        else:
            print(f"   Frame read:  FAILED")

        # Test each known resolution to see what camera supports
        print(f"   Testing standard resolutions:")
        for width, height, desc in TESTED_RESOLUTIONS:
            # Request resolution
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

            # Check what camera actually set
            actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            if actual_width == width and actual_height == height:
                print(f"       {width}x{height} ({desc})")
                results["supported_modes"].append((width, height, desc))
            else:
                # Camera doesn't support this resolution
                pass
                # Uncomment for verbose output:
                # print(f"       {width}x{height} ({desc}) - got {actual_width}x{actual_height}")

        cap.release()

        # Get camera metadata (Windows only)
        metadata = get_camera_metadata(index)
        if metadata:
            results["metadata"] = metadata
            print(f"   Camera metadata:")
            print(f"      Name: {metadata.get('name', 'N/A')}")
            print(f"      Type: {metadata.get('camera_type', 'N/A')}")

            # Display InstanceId (Device Path)
            instance_id = metadata.get('os_id')
            if instance_id:
                print(f"      OS ID (InstanceId): {instance_id}")

                # Extract and display serial
                serial = extract_and_display_serial(instance_id)
                print(f"      Serial Number: {serial}")

            # Display VID/PID
            vid = metadata.get('usb_vid', 'N/A')
            pid = metadata.get('usb_pid', 'N/A')
            print(f"      USB VID/PID: {vid}:{pid}")

        # SDK-based identification (run after closing OpenCV to avoid conflicts)
        if sdk_available:
            print(f"   SDK identification:")
            is_realsense, status = test_index_is_realsense_via_sdk_robust(index, 'DSHOW')
            results["sdk_is_realsense"] = is_realsense
            results["sdk_status"] = status

            if is_realsense:
                print(f"      REALSENSE confirmed ({status})")
            else:
                print(f"      NOT RealSense ({status})")

                # If stereo + not RealSense, likely ZED
                if results["is_stereo_candidate"]:
                    print(f"      Combined result: Likely ZED 2i (stereo + not RealSense)")
        else:
            print(f"   SDK identification: Not available (install pyrealsense2)")

    except Exception as e:
        print(f" Index {index}: Error - {e}")
        import traceback
        traceback.print_exc()

    return results

def main():
    """Test all camera indices from 0 to 10."""
    print("=" * 70)
    print("ZED 2i Camera Index Diagnostic")
    print("=" * 70)
    print("\nTesting camera indices 0-10...")
    print("Looking for stereo side-by-side format (aspect ratio ~3.56)")
    print("Expected for ZED 2i: 2560x720 or 3840x1080")
    print("\n")

    all_results = []

    # Test each index
    for index in range(11):
        result = test_camera_index(index)
        if result["openable"]:
            all_results.append(result)

    # Summary
    print("\n" + "=" * 70)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 70)

    # Find stereo candidates
    stereo_candidates = [r for r in all_results if r["is_stereo_candidate"]]

    if stereo_candidates:
        print(f"\n Found {len(stereo_candidates)} STEREO CANDIDATE(S):")
        for r in stereo_candidates:
            width, height, fps = r["default_resolution"]
            aspect = width / height
            print(f"\n    Index {r['index']}: {width}x{height} @ {fps:.1f}fps")
            print(f"      Aspect ratio: {aspect:.2f} (stereo side-by-side format)")
            print(f"      Supported stereo modes: {len(r['supported_modes'])}")
            for mode in r['supported_modes']:
                print(f"         - {mode[0]}x{mode[1]} ({mode[2]})")

            # Show SDK identification
            if r["sdk_is_realsense"] is not None:
                if r["sdk_is_realsense"]:
                    print(f"      SDK: REALSENSE ({r['sdk_status']})")
                else:
                    print(f"      SDK: NOT RealSense ({r['sdk_status']})")
                    print(f"      CONCLUSION: This is the ZED 2i stereo interface")
            else:
                print(f"      SDK: Not available")

        print("\n" + "=" * 70)
        print(" RECOMMENDATION:")
        print("=" * 70)
        print(f"\nThe ZED 2i stereo interface is likely at index {stereo_candidates[0]['index']}")
        print(f"Current code uses index 2, but stereo interface is at index {stereo_candidates[0]['index']}")
        print("\nACTION REQUIRED:")
        print(f"1. Update camera enumeration to use index {stereo_candidates[0]['index']} for ZED 2i")
        print(f"2. Or add logic to scan multiple indices and select the one with stereo format")

    else:
        print("\n  NO STEREO CANDIDATES FOUND!")
        print("    This suggests:")
        print("    1. ZED 2i may not be properly connected (check USB 3.0 cable)")
        print("    2. ZED 2i firmware/driver may not be installed correctly")
        print("    3. DirectShow may not support ZED stereo mode (try ZED SDK)")
        print("    4. Camera is in the wrong UVC mode (needs reconfiguration)")

    print("\n" + "=" * 70)
    print(" ALL OPENABLE CAMERAS:")
    print("=" * 70)
    for r in all_results:
        width, height, fps = r["default_resolution"]
        aspect = width / height if height > 0 else 0
        stereo_marker = " STEREO" if r["is_stereo_candidate"] else ""

        # SDK identification marker
        sdk_marker = ""
        if r["sdk_is_realsense"] is not None:
            if r["sdk_is_realsense"]:
                sdk_marker = " [RealSense]"
            else:
                sdk_marker = " [NOT RealSense]"

        print(f"   Index {r['index']}: {width}x{height} @ {fps:.1f}fps (aspect {aspect:.2f}){stereo_marker}{sdk_marker}")

    print("\n" + "=" * 70)
    print(f"Total cameras tested: 11")
    print(f"Openable cameras: {len(all_results)}")
    print(f"Stereo candidates: {len(stereo_candidates)}")
    print("=" * 70)

    # Generate configuration (Windows only)
    if platform.system() == "Windows" and all_results:
        generate_configuration(all_results)


def generate_configuration(results: list):
    """
    Generate PREFERRED_CAMERA_DEVICE_PATHS configuration for config.py.

    Args:
        results: List of test results with metadata
    """
    print("\n" + "=" * 70)
    print("CONFIGURATION GENERATOR")
    print("=" * 70)
    print("\nCopy this to your config.py:\n")

    # Separate cameras by type
    realsense_cameras = []
    zed_cameras = []
    generic_cameras = []

    for r in results:
        if not r.get("metadata"):
            continue

        metadata = r["metadata"]
        instance_id = metadata.get("os_id")
        camera_type = metadata.get("camera_type", "")

        if not instance_id:
            continue

        if "REALSENSE" in camera_type.upper():
            realsense_cameras.append((r["index"], instance_id, metadata.get("name", "Unknown")))
        elif "ZED" in camera_type.upper():
            zed_cameras.append((r["index"], instance_id, metadata.get("name", "Unknown")))
        else:
            generic_cameras.append((r["index"], instance_id, metadata.get("name", "Unknown")))

    # Generate config
    print("PREFERRED_CAMERA_DEVICE_PATHS = {")

    # RealSense cameras
    if realsense_cameras:
        print("    # RealSense D455i camera(s)")
        print("    # NOTE: Only specify ONE interface (MI_00 recommended)")
        print("    \"realsense_primary\": [")
        for idx, instance_id, name in realsense_cameras[:1]:  # Only first RealSense
            # Escape backslashes for Python string
            escaped_path = instance_id.replace("\\", "\\\\")
            print(f"        \"{escaped_path}\",  # Index {idx}: {name}")
        print("    ],")
        print("")

    # ZED cameras
    if zed_cameras:
        print("    # ZED 2i camera(s)")
        print("    # List in order of preference: first = primary, second = secondary")
        print("    \"zed_cameras\": [")
        for idx, instance_id, name in zed_cameras:
            # Escape backslashes for Python string
            escaped_path = instance_id.replace("\\", "\\\\")
            serial = extract_and_display_serial(instance_id)
            print(f"        \"{escaped_path}\",  # Index {idx}: {name} (Serial: {serial})")
        print("    ],")

    print("}")

    print("\n" + "=" * 70)
    print("CONFIGURATION NOTES")
    print("=" * 70)
    print("\n1. Copy the PREFERRED_CAMERA_DEVICE_PATHS dictionary above to your config.py")
    print("2. Device paths are STABLE - they persist across USB port changes")
    print("3. Serial numbers uniquely identify each physical camera")
    print("4. For RealSense, only specify ONE interface (MI_00 for depth)")
    print("5. For multiple ZED cameras, order in list determines primary/secondary")
    print("\n6. Set fallback mode in config.py:")
    print("   CAMERA_DETECTION_FALLBACK_MODE = \"strict\"  # Recommended")
    print("   - \"strict\": ONLY use configured cameras (recommended)")
    print("   - \"sdk_exclusion\": Use SDK detection as fallback")
    print("   - \"first_available\": Use any available camera")
    print("\n7. Enable device path detection:")
    print("   ENABLE_DEVICE_PATH_DETECTION = True")
    print("   DEBUG_DEVICE_PATH_RESOLUTION = True  # Set False in production")

    # Test device path resolution if module available
    print("\n" + "=" * 70)
    print("DEVICE PATH RESOLUTION TEST")
    print("=" * 70)

    try:
        from utils.camera_device_path_resolver import resolve_configured_cameras
        from utils.utils import enumerate_usb_external_cameras

        print("\nTesting device path resolution with generated config...")

        # Build config dict from detected cameras
        test_config = {}
        if realsense_cameras:
            test_config["realsense_primary"] = [realsense_cameras[0][1]]
        if zed_cameras:
            test_config["zed_cameras"] = [cam[1] for cam in zed_cameras]

        # Get all cameras
        all_cameras = enumerate_usb_external_cameras(use_device_path_resolution=False)

        # Mock config for testing
        import config as original_config
        original_paths = getattr(original_config, 'PREFERRED_CAMERA_DEVICE_PATHS', {})
        original_config.PREFERRED_CAMERA_DEVICE_PATHS = test_config

        # Resolve cameras
        configured = resolve_configured_cameras(all_cameras, fallback_mode="strict")

        # Restore original config
        original_config.PREFERRED_CAMERA_DEVICE_PATHS = original_paths

        print("\nResolution results:")
        for key, value in configured.items():
            if isinstance(value, list):
                print(f"  {key}: {len(value)} camera(s)")
                for i, cam in enumerate(value):
                    if cam:
                        print(f"    [{i}] Index {cam.index}: {cam.name}")
            elif value:
                print(f"  {key}: Index {value.index}: {value.name}")
            else:
                print(f"  {key}: None (not found)")

        print("\nDevice path resolution: WORKING")

    except ImportError as e:
        print(f"\nDevice path resolution module not available: {e}")
    except Exception as e:
        print(f"\nDevice path resolution test failed: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
