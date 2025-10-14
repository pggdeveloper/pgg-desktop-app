#!/usr/bin/env python3
"""
Multi-Camera Calibration Script

Interactive script to calibrate multi-vendor camera system:
- 2x ZED 2i cameras
- 1x RealSense D455i camera

Usage:
    python scripts/calibrate_multi_camera.py --output calibration.json

Requirements:
- Chessboard calibration pattern (9×6 internal corners, 30mm squares)
- All 3 cameras connected and functional
- Good lighting conditions
- 20-30 calibration image captures from different angles/positions
"""

import sys
import os
import argparse
import time
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.multi_camera_calibration import MultiCameraCalibrator
from utils.calibration_pattern_detector import CalibrationPatternDetector
from utils.stereo_pair_calibration import StereoPairCalibrator
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Multi-camera calibration for ZED 2i + RealSense D455i'
    )

    parser.add_argument(
        '--output', '-o',
        type=str,
        default='calibration_result.json',
        help='Output calibration file (default: calibration_result.json)'
    )

    parser.add_argument(
        '--pattern-width', '-w',
        type=int,
        default=9,
        help='Chessboard pattern width (internal corners, default: 9)'
    )

    parser.add_argument(
        '--pattern-height', '-h',
        type=int,
        default=6,
        help='Chessboard pattern height (internal corners, default: 6)'
    )

    parser.add_argument(
        '--square-size',
        type=float,
        default=30.0,
        help='Square size in millimeters (default: 30.0)'
    )

    parser.add_argument(
        '--min-images',
        type=int,
        default=20,
        help='Minimum number of calibration images (default: 20)'
    )

    parser.add_argument(
        '--reference-camera',
        type=str,
        default='realsense_d455i_0',
        help='Reference camera ID (default: realsense_d455i_0)'
    )

    parser.add_argument(
        '--calibrate-stereo',
        action='store_true',
        help='Also calibrate ZED 2i pair as wide-baseline stereo'
    )

    return parser.parse_args()


def print_banner():
    """Print welcome banner"""
    print("=" * 70)
    print("  MULTI-CAMERA CALIBRATION")
    print("  Multi-Vendor System: 2x ZED 2i + 1x RealSense D455i")
    print("=" * 70)
    print()


def print_instructions(args):
    """Print calibration instructions"""
    print("📋 CALIBRATION INSTRUCTIONS:")
    print()
    print(f"1. Pattern Configuration:")
    print(f"   - Type: Chessboard")
    print(f"   - Size: {args.pattern_width}×{args.pattern_height} internal corners")
    print(f"   - Square size: {args.square_size}mm")
    print()
    print(f"2. Target: Capture at least {args.min_images} calibration images")
    print()
    print("3. Best Practices:")
    print("   - Ensure pattern is visible from ALL 3 cameras simultaneously")
    print("   - Capture from different angles (tilted left/right, up/down)")
    print("   - Capture from different distances (0.5m - 3m)")
    print("   - Cover the entire field of view")
    print("   - Ensure good lighting (avoid shadows on pattern)")
    print()
    print("4. Camera Setup:")
    print(f"   - Reference camera: {args.reference_camera} (origin of coordinates)")
    print("   - All cameras must be stable (fixed positions during calibration)")
    print()
    print("=" * 70)
    print()


def capture_calibration_images_interactive(calibrator, min_images):
    """
    Interactive calibration image capture.

    This is a placeholder - in real implementation, this would:
    1. Initialize camera connections
    2. Display live preview from all cameras
    3. Wait for user to press spacebar to capture
    4. Show detection results
    5. Repeat until enough images captured

    Args:
        calibrator: MultiCameraCalibrator instance
        min_images: Minimum number of images to capture

    Returns:
        True if successful, False otherwise
    """
    print("📸 CALIBRATION IMAGE CAPTURE")
    print()
    print("NOTE: This is a simulation mode.")
    print("In production, this would:")
    print("  1. Show live preview from all 3 cameras")
    print("  2. Detect chessboard pattern in real-time")
    print("  3. Capture images when you press SPACE")
    print("  4. Continue until you have enough images")
    print()
    print("For now, we'll simulate the calibration process.")
    print()

    # In real implementation, replace this with actual camera capture
    logger.warning("Calibration image capture not implemented - using simulation mode")
    logger.info("To use real cameras, integrate with:")
    logger.info("  - pyzed for ZED 2i cameras")
    logger.info("  - pyrealsense2 for RealSense D455i")

    return False


def main():
    """Main calibration workflow"""
    args = parse_args()

    print_banner()
    print_instructions(args)

    # Confirm to proceed
    response = input("Ready to start calibration? (y/n): ").strip().lower()
    if response != 'y':
        print("Calibration cancelled.")
        return

    print()

    # Initialize calibrator
    camera_ids = ["realsense_d455i_0", "zed_2i_0", "zed_2i_1"]

    calibrator = MultiCameraCalibrator(
        camera_ids=camera_ids,
        reference_camera=args.reference_camera,
        pattern_size=(args.pattern_width, args.pattern_height),
        square_size_mm=args.square_size
    )

    logger.info("Initialized MultiCameraCalibrator")

    # Capture calibration images
    print("⏳ Starting calibration image capture...")
    print()

    success = capture_calibration_images_interactive(calibrator, args.min_images)

    if not success:
        logger.error("Calibration image capture failed or not implemented")
        print()
        print("CALIBRATION FAILED")
        print()
        print("To complete calibration, you need to:")
        print("1. Implement camera capture integration")
        print("2. Or provide pre-captured calibration images")
        print()
        print("Example code structure:")
        print("```python")
        print("# Initialize cameras")
        print("# For each calibration position:")
        print("#   images = {")
        print("#       'realsense_d455i_0': capture_from_realsense(),")
        print("#       'zed_2i_0': capture_from_zed_left(0),")
        print("#       'zed_2i_1': capture_from_zed_left(1)")
        print("#   }")
        print("#   calibrator.add_calibration_images(images)")
        print("```")
        print()
        return

    # Check if enough images
    progress = calibrator.get_calibration_progress()
    num_images = progress['num_images_captured']

    if num_images < args.min_images:
        logger.error(f"Only {num_images} images captured, need {args.min_images}")
        print(f"Insufficient calibration images: {num_images}/{args.min_images}")
        return

    print(f"Captured {num_images} calibration images")
    print()

    # Perform calibration
    print("⏳ Performing multi-camera calibration...")
    print("   This may take a few minutes...")
    print()

    try:
        result = calibrator.calibrate()

        print("Multi-camera calibration complete!")
        print()

        # Print quality metrics
        print("📊 CALIBRATION QUALITY:")
        print(f"   Overall quality: {result.quality_metrics['overall_quality']}")
        print(f"   Mean reprojection error: {result.quality_metrics['mean_reprojection_error']:.4f} pixels")
        print(f"   Max reprojection error: {result.quality_metrics['max_reprojection_error']:.4f} pixels")
        print()

        # Print camera poses
        print("📍 CAMERA POSES (in global coordinates):")
        for camera_id, calib in result.intrinsic_calibrations.items():
            if camera_id == args.reference_camera:
                print(f"   {camera_id}: [REFERENCE - Origin (0, 0, 0)]")
            else:
                pose = result.extrinsic_calibration.camera_poses[camera_id]
                print(f"   {camera_id}:")
                print(f"      Position: ({pose.position[0]:.3f}, {pose.position[1]:.3f}, {pose.position[2]:.3f}) m")
                euler_deg = tuple(np.degrees(e) for e in pose.euler_angles)
                print(f"      Orientation: ({euler_deg[0]:.1f}°, {euler_deg[1]:.1f}°, {euler_deg[2]:.1f}°)")
        print()

        # Save calibration
        output_path = Path(args.output)
        result.save(output_path)

        print(f"Calibration saved to: {output_path}")
        print()

        # Calibrate stereo pair if requested
        if args.calibrate_stereo:
            print("⏳ Calibrating ZED 2i stereo pair...")

            stereo_calibrator = StereoPairCalibrator()
            stereo_calib = stereo_calibrator.calibrate_from_multi_camera(
                result,
                camera_left_id="zed_2i_0",
                camera_right_id="zed_2i_1"
            )

            print("Stereo pair calibration complete!")
            print(f"   Baseline: {stereo_calib.baseline_meters:.3f} meters")
            print(f"   Disparity range: [{stereo_calib.disparity_range[0]:.1f}, {stereo_calib.disparity_range[1]:.1f}] pixels")
            print()

            # Save stereo calibration
            stereo_output = output_path.parent / f"{output_path.stem}_stereo.json"
            import json
            with open(stereo_output, 'w') as f:
                json.dump(stereo_calib.to_dict(), f, indent=2)

            print(f"Stereo calibration saved to: {stereo_output}")
            print()

        print("=" * 70)
        print("🎉 CALIBRATION COMPLETE!")
        print("=" * 70)
        print()
        print("Next steps:")
        print(f"1. Verify calibration quality (target: <1.0 pixel reprojection error)")
        print(f"2. Test with known 3D objects to validate accuracy")
        print(f"3. Load calibration in CameraRecordingOrchestrator")
        print(f"4. Start multi-camera cattle monitoring!")
        print()

    except Exception as e:
        logger.exception("Calibration failed")
        print(f"CALIBRATION FAILED: {e}")
        return


if __name__ == '__main__':
    # Import numpy here to avoid circular import in logging
    import numpy as np
    main()
