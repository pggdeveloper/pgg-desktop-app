"""
Quick test for camera index 4 with SDK-based identification.

UPDATED (2025-10-18): Now shows Windows Device Instance Paths (InstanceId)
for stable camera identification across USB port changes.

Based on console output, index 4 exists but has no PowerShell metadata.
This script quickly tests if index 4 is the ZED 2i stereo interface.

Run this first for a quick answer to the mystery.
"""
import cv2
import sys
import os
import platform

# Add parent directory to path to import SDK exclusion functions
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def get_camera_device_path(index: int):
    """
    Get Windows Device Instance Path for a camera index.

    Returns:
        Tuple of (instance_id, serial) or (None, None)
    """
    if platform.system() != "Windows":
        return None, None

    try:
        from utils.utils import enumerate_usb_external_cameras
        from utils.camera_device_path_resolver import extract_serial_from_instance_id

        cameras = enumerate_usb_external_cameras(use_device_path_resolution=False)

        for cam in cameras:
            if cam.index == index:
                instance_id = cam.os_id
                serial = extract_serial_from_instance_id(instance_id) if instance_id else None
                return instance_id, serial

        return None, None
    except Exception:
        return None, None


def test_index_4():
    """Quick test of index 4."""
    print("=" * 70)
    print("Quick Test: Camera Index 4")
    print("=" * 70)
    print("\nHypothesis: Index 4 might be the ZED 2i stereo interface")
    print("(Index 2 was detected as ZED 2i but failed stereo validation)")
    print()

    # Import SDK exclusion function
    try:
        from utils.camera_identification_sdk import test_index_is_realsense_via_sdk_robust
        sdk_available = True
    except ImportError:
        print("WARNING: SDK exclusion module not available")
        sdk_available = False

    # Try to open index 4
    print("Opening camera at index 4 with DirectShow...")
    cap = cv2.VideoCapture(4, cv2.CAP_DSHOW)

    if not cap.isOpened():
        print("FAILED: Cannot open camera at index 4")
        print("\nPossible reasons:")
        print("  - Index 4 doesn't exist")
        print("  - Camera is in use by another application")
        print("  - DirectShow backend issue")
        return

    print("SUCCESS: Camera opened at index 4\n")

    # Get default resolution (without setting anything)
    default_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    default_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    default_fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"Default resolution: {default_w}x{default_h} @ {default_fps:.1f}fps")

    # Calculate aspect ratio
    aspect = default_w / default_h if default_h > 0 else 0
    print(f"Aspect ratio: {aspect:.2f}")

    # Check if stereo
    if 3.0 <= aspect <= 4.0:
        print("\n" + "-" * 35)
        print(" INDEX 4 IS A STEREO CAMERA! ")
        print("-" * 35)
        print(f"\nAspect ratio {aspect:.2f} indicates STEREO SIDE-BY-SIDE format")
        print("\nSOLUTION FOUND:")
        print("   The ZED 2i stereo interface is at INDEX 4, not INDEX 2!")
        print("\n   Next steps:")
        print("   1. Update camera enumeration to use index 4 for ZED 2i")
        print("   2. Modify camera_validation_zed.py to scan adjacent indices")
        print("   3. Update Windows enumeration to handle multi-interface cameras")
    else:
        print(f"\nIndex 4 is NOT stereo (aspect ratio {aspect:.2f})")
        print("   Expected aspect ratio: ~3.56 for stereo side-by-side")
        print("   Got: 1.78 (16:9 standard) or other non-stereo ratio")

    print()

    # Try to read a frame
    print("Testing frame capture...")
    ret, frame = cap.read()

    if ret and frame is not None:
        print(f"Frame captured successfully")
        print(f"   Frame shape: {frame.shape}")
        print(f"   Dimensions: {frame.shape[1]}x{frame.shape[0]} (width x height)")

        # Verify frame aspect ratio
        frame_aspect = frame.shape[1] / frame.shape[0]
        print(f"   Frame aspect ratio: {frame_aspect:.2f}")

        if 3.0 <= frame_aspect <= 4.0:
            print("\n   CONFIRMED: Frame has stereo aspect ratio!")
    else:
        print("Failed to read frame")

    cap.release()

    # Get device path (Windows only)
    if platform.system() == "Windows":
        print("\n" + "=" * 70)
        print("DEVICE PATH INFORMATION (Windows)")
        print("=" * 70)

        instance_id, serial = get_camera_device_path(4)

        if instance_id:
            print(f"\nOS ID (InstanceId): {instance_id}")
            print(f"Serial Number: {serial if serial else 'N/A'}")
            print("\nThis device path can be used in config.py for stable identification:")
            print("\nPREFERRED_CAMERA_DEVICE_PATHS = {")
            print("    \"zed_cameras\": [")
            escaped_path = instance_id.replace("\\", "\\\\")
            print(f"        \"{escaped_path}\",  # Index 4 (Serial: {serial if serial else 'N/A'})")
            print("    ],")
            print("}")
        else:
            print("\nDevice path not available for index 4")
            print("(Camera may not be enumerated via PowerShell)")

    # SDK-based identification
    print("\n" + "=" * 70)
    print("SDK-BASED IDENTIFICATION (Definitive)")
    print("=" * 70)

    if sdk_available:
        print("\nTesting Index 4 with RealSense SDK...")
        is_realsense, status = test_index_is_realsense_via_sdk_robust(4, 'DSHOW')

        print(f"Result: {status}")
        print(f"Is RealSense: {is_realsense}")

        if is_realsense:
            print("\n" + "-" * 70)
            print(" INDEX 4 IS A REALSENSE CAMERA")
            print("-" * 70)
            print("\nSDK confirmed this is a RealSense D455i or similar model")
        else:
            print("\n" + "-" * 70)
            print(" INDEX 4 IS NOT A REALSENSE CAMERA")
            print("-" * 70)
            print("\nSDK exclusion indicates this is likely a ZED 2i or generic camera")

            # If stereo aspect ratio detected, likely ZED
            if 3.0 <= aspect <= 4.0:
                print("\nCombined conclusion:")
                print("  - NOT RealSense (SDK exclusion)")
                print("  - HAS stereo aspect ratio")
                print("  - CONCLUSION: This is the ZED 2i stereo interface")
    else:
        print("\nSDK testing not available - install pyrealsense2")

    # Re-open for resolution testing
    print("\n" + "=" * 70)
    print("RESOLUTION COMPATIBILITY TESTING")
    print("=" * 70)
    cap = cv2.VideoCapture(4, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("\nCannot re-open camera for resolution testing")
        return

    # Try to set ZED stereo resolutions
    print("\nTesting ZED stereo resolutions:")
    zed_resolutions = [
        (2560, 720, "HD Stereo"),
        (3840, 1080, "Full HD Stereo"),
        (1344, 376, "VGA Stereo"),
    ]

    supported_stereo = []

    for width, height, desc in zed_resolutions:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if actual_w == width and actual_h == height:
            print(f"   {width}x{height} - {desc} SUPPORTED")
            supported_stereo.append((width, height, desc))
        else:
            print(f"   {width}x{height} - {desc} NOT SUPPORTED (got {actual_w}x{actual_h})")

    if supported_stereo:
        print(f"\nIndex 4 supports {len(supported_stereo)} ZED stereo format(s)!")
        print("   This CONFIRMS index 4 is the ZED 2i stereo interface")
    else:
        print("\n  Index 4 does NOT support any ZED stereo formats")

    cap.release()

    print("\n" + "=" * 70)
    print("Test complete")
    print("=" * 70)

def compare_index_2_and_4():
    """Compare index 2 and index 4 side-by-side."""
    print("\n" + "=" * 70)
    print("Comparison: Index 2 vs Index 4")
    print("=" * 70)

    # Import SDK exclusion function
    try:
        from utils.camera_identification_sdk import test_index_is_realsense_via_sdk_robust
        sdk_available = True
    except ImportError:
        sdk_available = False

    for idx in [2, 4]:
        print(f"\n--- Index {idx} ---")
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)

        if not cap.isOpened():
            print(f"Cannot open index {idx}")
            continue

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        aspect = w / h if h > 0 else 0

        print(f"Resolution: {w}x{h} @ {fps:.1f}fps")
        print(f"Aspect: {aspect:.2f}")

        if 3.0 <= aspect <= 4.0:
            print("Aspect ratio: STEREO")
        else:
            print("Aspect ratio: Single camera (non-stereo)")

        cap.release()

        # Device path (Windows only)
        if platform.system() == "Windows":
            instance_id, serial = get_camera_device_path(idx)
            if instance_id:
                print(f"OS ID: {instance_id}")
                print(f"Serial: {serial if serial else 'N/A'}")
            else:
                print("Device path: Not available")

        # SDK identification
        if sdk_available:
            is_realsense, status = test_index_is_realsense_via_sdk_robust(idx, 'DSHOW')
            if is_realsense:
                print(f"SDK result: REALSENSE ({status})")
            else:
                print(f"SDK result: NOT RealSense ({status})")
        else:
            print("SDK result: Not available")

if __name__ == "__main__":
    test_index_4()
    compare_index_2_and_4()
