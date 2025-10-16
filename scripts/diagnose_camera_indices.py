"""
Diagnostic script to test all camera indices and check resolutions.
Identifies which index provides ZED 2i stereo side-by-side format.

This script tests all camera indices from 0 to 10 and checks:
- Which indices can be opened
- Default resolution for each camera
- Aspect ratio (to detect stereo side-by-side)
- Which resolutions each camera supports

Critical for debugging ZED 2i detection issues.
"""
import cv2
import sys

TESTED_RESOLUTIONS = [
    (3840, 1080, "Full HD Stereo (ZED 2i native)"),
    (2560, 720, "HD Stereo (ZED 2i HD)"),
    (1920, 1080, "Full HD Single Camera"),
    (1280, 720, "HD Single Camera"),
    (1344, 376, "VGA Stereo (ZED 2i VGA)"),
    (640, 480, "VGA"),
]

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
    """
    results = {
        "index": index,
        "openable": False,
        "default_resolution": None,
        "supported_modes": [],
        "is_stereo_candidate": False,
    }

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
        print(f"   Index {r['index']}: {width}x{height} @ {fps:.1f}fps (aspect {aspect:.2f}) {stereo_marker}")

    print("\n" + "=" * 70)
    print(f"Total cameras tested: 11")
    print(f"Openable cameras: {len(all_results)}")
    print(f"Stereo candidates: {len(stereo_candidates)}")
    print("=" * 70)

if __name__ == "__main__":
    main()
