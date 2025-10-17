"""
Enumerate supported video formats for each camera index.
Shows all resolutions, pixel formats, and frame rates supported by DirectShow.

UPDATED (2025-10-17): Now uses SDK exclusion strategy to definitively
identify if cameras are RealSense or ZED.

This script provides detailed format information for each camera, helping to
identify which camera supports which resolutions and formats.
"""
import cv2
import sys
import os

# Add parent directory to path to import SDK exclusion functions
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def enumerate_formats(index: int):
    """
    Enumerate all supported formats for a camera index.

    Args:
        index: Camera index to test

    Tests multiple common resolutions and reports which ones are supported.
    Includes SDK-based camera identification (RealSense vs ZED).
    """
    print(f"\n{'='*70}")
    print(f"Camera Index {index} - Format Enumeration")
    print(f"{'='*70}")

    # Import SDK exclusion function
    try:
        from utils.camera_identification_sdk import test_index_is_realsense_via_sdk_robust
        sdk_available = True
    except ImportError:
        sdk_available = False

    try:
        # Try DirectShow first (primary backend on Windows)
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)

        if not cap.isOpened():
            print(" Cannot open camera with DirectShow")

            # Try MSMF as fallback
            cap = cv2.VideoCapture(index, cv2.CAP_MSMF)
            if not cap.isOpened():
                print(" Cannot open camera with MSMF either")
                return
            else:
                print(" Camera opened with MSMF (fallback)")
        else:
            print(" Camera opened with DirectShow")

        # Get backend name
        backend = cap.getBackendName()
        print(f"Backend: {backend}")

        # Get default properties
        default_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        default_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        default_fps = cap.get(cv2.CAP_PROP_FPS)

        print(f"Default format: {default_w}x{default_h} @ {default_fps:.1f}fps")

        # Common resolutions to test (including ZED stereo formats)
        resolutions = [
            # ZED stereo formats
            (3840, 1080, "ZED Full HD Stereo"),
            (2560, 720, "ZED HD Stereo"),
            (1344, 376, "ZED VGA Stereo"),

            # Standard single camera formats
            (1920, 1080, "Full HD"),
            (1280, 720, "HD"),
            (640, 480, "VGA"),
            (320, 240, "QVGA"),

            # RealSense formats
            (1280, 800, "RealSense Native"),
            (848, 480, "RealSense VGA"),
        ]

        supported = []
        stereo_formats = []

        print("\nTesting resolutions:")
        for width, height, desc in resolutions:
            # Try to set resolution
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

            # Check what was actually set
            actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            if actual_w == width and actual_h == height:
                fps = cap.get(cv2.CAP_PROP_FPS)
                aspect = width / height
                supported.append((width, height, fps, aspect, desc))

                # Check if stereo format (aspect ratio 3.0-4.0)
                if 3.0 <= aspect <= 4.0:
                    stereo_formats.append((width, height, fps, aspect, desc))

        # Display results
        if supported:
            print(f"\n Supported resolutions ({len(supported)}):")
            for w, h, fps, aspect, desc in supported:
                stereo_marker = " STEREO" if 3.0 <= aspect <= 4.0 else ""
                print(f"   {w:4d}x{h:4d} @ {fps:5.1f}fps (aspect {aspect:.2f}) - {desc:20s} {stereo_marker}")
        else:
            print("\n  No standard resolutions supported")
            print("    Camera may use proprietary or non-standard formats")

        # Highlight stereo formats
        if stereo_formats:
            print(f"\n STEREO FORMATS DETECTED ({len(stereo_formats)}):")
            for w, h, fps, aspect, desc in stereo_formats:
                print(f"   {w}x{h} @ {fps:.1f}fps - {desc}")

        # Try to read a test frame
        print("\nTesting frame capture:")
        ret, frame = cap.read()
        if ret and frame is not None:
            print(f" Test frame captured: {frame.shape[1]}x{frame.shape[0]} (HxWxC: {frame.shape})")

            # Check frame aspect ratio
            frame_aspect = frame.shape[1] / frame.shape[0]
            if 3.0 <= frame_aspect <= 4.0:
                print(f"    Frame has STEREO aspect ratio ({frame_aspect:.2f})")
        else:
            print(" Failed to capture frame")

        cap.release()

        # SDK-based identification (run after closing OpenCV to avoid conflicts)
        print("\nSDK-based identification:")
        if sdk_available:
            is_realsense, status = test_index_is_realsense_via_sdk_robust(index, 'DSHOW')

            if is_realsense:
                print(f" REALSENSE camera detected ({status})")
            else:
                print(f" NOT RealSense camera ({status})")

                # If stereo formats detected + not RealSense, likely ZED
                if stereo_formats:
                    print(f" CONCLUSION: This is likely a ZED 2i camera")
                    print(f"   - NOT RealSense (SDK exclusion)")
                    print(f"   - HAS stereo format(s): {len(stereo_formats)}")
                else:
                    print(f" CONCLUSION: Generic camera (not RealSense, not stereo)")
        else:
            print(" SDK not available (install pyrealsense2 for identification)")

    except Exception as e:
        print(f" Error: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Test all indices 0-10."""
    print("DirectShow Format Enumeration")
    print("="*70)
    print("\nThis script tests each camera index to determine supported formats")
    print("Focus: Identifying which index provides ZED 2i stereo side-by-side\n")

    # Test all indices
    for i in range(11):
        enumerate_formats(i)

    print("\n" + "="*70)
    print("Format enumeration complete")
    print("="*70)
    print("\nLook for cameras with:")
    print("  - Aspect ratio ~3.56 (stereo side-by-side)")
    print("  - Resolutions like 2560x720 or 3840x1080")
    print("  - Multiple stereo format options")

if __name__ == "__main__":
    main()
