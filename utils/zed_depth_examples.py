"""
ZED 2i Depth Processing - Usage Examples

This module demonstrates different ways to use depth computation with ZED 2i:
1. Real-time depth during capture (CPU-intensive)
2. Offline depth processing after capture (RECOMMENDED)
3. Batch processing of recorded sessions

Created: 2025-10-14
"""

from pathlib import Path
from utils.zed_camera import ZedCameraRecorder
from utils.zed_offline_depth_processor import (
    ZedOfflineDepthProcessor,
    StereoCalibration,
    create_zed2i_default_calibration
)
from domain.camera_info import CameraInfo, CameraCapabilities


def example_1_realtime_depth_capture():
    """
    Example 1: Real-time depth computation during capture.

    WARNING: This is CPU-intensive and will reduce achievable FPS.
    Only recommended if you have a powerful CPU and low target FPS.
    """
    print("="*80)
    print("Example 1: Real-time Depth Computation During Capture")
    print("="*80)
    print("WARNING: CPU-intensive, may reduce FPS!")
    print()

    # Create mock camera info (replace with real enumeration)
    camera_info = CameraInfo(
        name="ZED 2i",
        index=0,
        vendor="Stereolabs",
        capabilities=CameraCapabilities(rgb=True, depth=True, imu=True)
    )

    # NOTE: Real-time depth computation has been REMOVED from ZedCameraRecorder
    # This example is now DEPRECATED. Use Phase 2 (SelectiveDepthProcessor) instead!
    # See: utils/zed_phase_1_2_examples.py

    print("WARNING: Real-time depth computation is no longer supported!")
    print("   Please use offline processing (Phase 1 or Phase 2)")
    print("   See: utils/zed_phase_1_2_examples.py")
    print()
    return

    # OLD CODE (DEPRECATED):
    # zed_recorder = ZedCameraRecorder(
    #     camera_info=camera_info,
    #     output_dir=Path("./recordings"),
    #     recording_duration=10,
    #     fps=30  # Capture stereo only
    # )

    # Initialize camera
    if zed_recorder.initialize():
        print("Camera initialized")
        print(f"   Depth computation: {zed_recorder.enable_depth_computation}")
        print(f"   Algorithm: {zed_recorder.depth_algorithm}")
        print(f"   Quality: {zed_recorder.depth_quality}")
        print()

        # Prepare for recording
        recording_thread = zed_recorder.prepare_for_recording(sync_timestamp=0.0)

        if recording_thread:
            print("Starting recording with real-time depth...")
            recording_thread.start()
            recording_thread.join()  # Wait for completion
            print("Recording complete!")
            print()
            print("Output files:")
            print("  - rgb_left.mp4")
            print("  - rgb_right.mp4")
            print("  - stereo_sbs.mp4")
            print("  - depth_XXXXXX.npz (individual depth frames)")
    else:
        print("Failed to initialize camera")


def example_2_offline_depth_processing():
    """
    Example 2: Offline depth processing after capture (RECOMMENDED).

    This is the recommended approach:
    1. Capture stereo pairs during recording (fast, no frame drops)
    2. Process depth offline afterwards (no time pressure)
    """
    print("="*80)
    print("Example 2: Offline Depth Processing (RECOMMENDED)")
    print("="*80)
    print("Best approach: Capture first, process later")
    print()

    # Step 1: Capture stereo pairs WITHOUT depth processing
    print("Step 1: Capturing stereo pairs...")
    print("-" * 40)

    camera_info = CameraInfo(
        name="ZED 2i",
        index=0,
        vendor="Stereolabs",
        capabilities=CameraCapabilities(rgb=True, depth=True, imu=True)
    )

    zed_recorder = ZedCameraRecorder(
        camera_info=camera_info,
        output_dir=Path("./recordings"),
        recording_duration=10,
        fps=30  # Full speed capture (stereo pairs only)
    )

    # ... (capture code same as before)
    print("Stereo pairs captured at 30 FPS")
    print()

    # Step 2: Process depth offline
    print("Step 2: Processing depth offline...")
    print("-" * 40)

    # Create calibration
    calibration = create_zed2i_default_calibration(1920, 1080)

    # Initialize offline processor
    processor = ZedOfflineDepthProcessor(
        calibration=calibration,
        algorithm='sgbm',  # Best quality
        quality='balanced'  # Good balance
    )

    # Process recorded videos
    processed_count = processor.process_video_pair_offline(
        left_video_path=Path("./recordings/rgb_left.mp4"),
        right_video_path=Path("./recordings/rgb_right.mp4"),
        output_dir=Path("./recordings/depth_frames"),
        max_frames=None,  # Process all frames
        frame_skip=1  # Process every frame
    )

    print(f"Generated {processed_count} depth frames")
    print()
    print("Advantages of offline processing:")
    print("  No FPS reduction during capture")
    print("  Better depth quality (can use 'quality' preset)")
    print("  Can retry with different parameters")
    print("  Process multiple sessions in batch")


def example_3_batch_processing():
    """
    Example 3: Batch process multiple recording sessions.
    """
    print("="*80)
    print("Example 3: Batch Processing Multiple Sessions")
    print("="*80)
    print()

    # Define sessions to process
    sessions = [
        {
            'left': './recordings/session1/rgb_left.mp4',
            'right': './recordings/session1/rgb_right.mp4',
            'output': './recordings/session1/depth_frames'
        },
        {
            'left': './recordings/session2/rgb_left.mp4',
            'right': './recordings/session2/rgb_right.mp4',
            'output': './recordings/session2/depth_frames'
        },
    ]

    # Create calibration once
    calibration = create_zed2i_default_calibration(1920, 1080)

    # Create processor once
    processor = ZedOfflineDepthProcessor(
        calibration=calibration,
        algorithm='sgbm',
        quality='balanced'
    )

    # Process all sessions
    total_frames = 0
    for i, session in enumerate(sessions, 1):
        print(f"Processing session {i}/{len(sessions)}...")
        print(f"  Input: {session['left']}")

        try:
            count = processor.process_video_pair_offline(
                left_video_path=Path(session['left']),
                right_video_path=Path(session['right']),
                output_dir=Path(session['output']),
                max_frames=None,
                frame_skip=1
            )
            total_frames += count
            print(f"  Generated {count} depth frames")
        except Exception as e:
            print(f"  Error: {e}")

        print()

    print(f"Batch processing complete: {total_frames} total depth frames")


def example_4_load_and_visualize_depth():
    """
    Example 4: Load depth .npz file and visualize.
    """
    print("="*80)
    print("Example 4: Load and Visualize Depth Frames")
    print("="*80)
    print()

    import numpy as np
    import cv2

    # Load depth frame
    depth_path = "./recordings/depth_frames/depth_000001.npz"
    print(f"Loading: {depth_path}")

    data = np.load(depth_path)

    print("\nMetadata:")
    for key in data.files:
        if key != 'depth':
            print(f"  {key}: {data[key]}")

    # Get depth map
    depth_mm = data['depth']  # uint16 in millimeters
    depth_scale = data['depth_scale']  # 0.001 (mm → m)

    print(f"\nDepth map shape: {depth_mm.shape}")
    print(f"Depth range: {depth_mm.min()} - {depth_mm.max()} mm")
    print(f"Valid pixels: {np.count_nonzero(depth_mm > 0)} / {depth_mm.size}")

    # Convert to meters
    depth_m = depth_mm.astype(np.float32) * depth_scale

    # Visualize
    depth_colormap = cv2.applyColorMap(
        cv2.convertScaleAbs(depth_mm, alpha=0.03),
        cv2.COLORMAP_JET
    )

    print("\nVisualization:")
    print("  - Press any key to close")
    cv2.imshow("Depth Visualization", depth_colormap)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print("\nDepth frame loaded and visualized")


def comparison_realtime_vs_offline():
    """
    Comparison: Real-time vs Offline depth processing.

    NOTE: Real-time depth is DEPRECATED and no longer supported.
    This comparison is for reference only.
    """
    print("="*80)
    print("Comparison: Real-time vs Offline Depth Processing")
    print("="*80)
    print("NOTE: Real-time depth is DEPRECATED")
    print("   See Phase 1 & Phase 2 examples for current recommendations")
    print()

    print("REAL-TIME DEPTH (during capture):")
    print("-" * 40)
    print("Advantages:")
    print("  - Immediate depth availability")
    print("  - No post-processing needed")
    print()
    print("Disadvantages:")
    print("  - Reduces achievable FPS (expect ~5-15 FPS)")
    print("  - CPU-intensive (may overheat)")
    print("  - Lower quality (must use 'fast' preset)")
    print("  - Frame drops possible")
    print()
    print("Use when: Immediate depth needed, low FPS acceptable")
    print()

    print("OFFLINE DEPTH (after capture):")
    print("-" * 40)
    print("Advantages:")
    print("  - No FPS reduction (capture at full 30 FPS)")
    print("  - Better depth quality (can use 'quality' preset)")
    print("  - No frame drops")
    print("  - Can retry with different parameters")
    print("  - Process in batch overnight")
    print()
    print("Disadvantages:")
    print("  - Requires post-processing step")
    print("  - Depth not immediately available")
    print()
    print("Use when: Best quality needed, have time for processing")
    print()

    print("RECOMMENDATION FOR CATTLE MONITORING:")
    print("-" * 40)
    print("OFFLINE PROCESSING")
    print("   Reasons:")
    print("   1. Capture full resolution stereo at 30 FPS")
    print("   2. Process depth overnight in batch")
    print("   3. Use 'quality' preset for best results")
    print("   4. Compatible with RealSense pipeline")


if __name__ == "__main__":
    print("\n" + "="*80)
    print("ZED 2i DEPTH PROCESSING - EXAMPLES")
    print("="*80)
    print()

    print("Available examples:")
    print("  1. Real-time depth during capture (CPU-intensive)")
    print("  2. Offline depth processing (RECOMMENDED)")
    print("  3. Batch processing multiple sessions")
    print("  4. Load and visualize depth frames")
    print("  5. Comparison: Real-time vs Offline")
    print()

    # Show comparison by default
    comparison_realtime_vs_offline()

    print("\n" + "="*80)
    print("To run specific examples, uncomment the function calls below:")
    print("="*80)
    # example_1_realtime_depth_capture()
    # example_2_offline_depth_processing()
    # example_3_batch_processing()
    # example_4_load_and_visualize_depth()
