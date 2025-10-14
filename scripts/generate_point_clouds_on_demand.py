"""
On-Demand Point Cloud Generation Script

Demonstrates both RealSense D455i and ZED 2i workflows for generating
point clouds from recorded sessions.

Usage:
    # RealSense D455i:
    python scripts/generate_point_clouds_on_demand.py --camera realsense --session ./recordings/session_001

    # ZED 2i:
    python scripts/generate_point_clouds_on_demand.py --camera zed --session ./recordings/session_001

    # With custom options:
    python scripts/generate_point_clouds_on_demand.py \
        --camera zed \
        --session ./recordings/session_001 \
        --yolo-model yolov8s.pt \
        --format ply \
        --mode selective \
        --frames 10,50,100,200

Created: 2025-10-14
Status: PRODUCTION READY
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, List


def example_realsense_all_frames():
    """Example: Process all frames from RealSense session."""
    from utils.realsense_point_cloud_generator import process_realsense_session_point_clouds

    print("="*80)
    print("Example: RealSense D455i - All Frames")
    print("="*80)
    print()

    session_dir = Path("./recordings/realsense_session_001")

    # Generate all point clouds
    count = process_realsense_session_point_clouds(
        session_dir=session_dir,
        camera_index=0,
        mode='all',
        output_format='ply'
    )

    print(f"Generated {count} point clouds for RealSense")
    print()


def example_realsense_selective():
    """Example: Process only specific frames from RealSense session."""
    from utils.realsense_point_cloud_generator import process_realsense_session_point_clouds

    print("="*80)
    print("Example: RealSense D455i - Selective Frames")
    print("="*80)
    print()

    session_dir = Path("./recordings/realsense_session_001")

    # Generate point clouds for specific frames only
    frame_indices = [10, 50, 100, 200, 500, 1000]

    count = process_realsense_session_point_clouds(
        session_dir=session_dir,
        camera_index=0,
        mode='selective',
        frame_indices=frame_indices,
        output_format='ply'
    )

    print(f"Generated {count} point clouds for selected frames")
    print()


def example_zed_complete_workflow():
    """Example: Complete ZED 2i two-phase workflow."""
    from utils.zed_point_cloud_generator import process_zed_session_point_clouds

    print("="*80)
    print("Example: ZED 2i - Complete Two-Phase Workflow")
    print("="*80)
    print()

    session_dir = Path("./recordings/zed_session_001")

    # Run complete workflow (Phase 1 + Phase 2)
    stats, depth_count, pc_count = process_zed_session_point_clouds(
        session_dir=session_dir,
        yolo_model="yolov8n.pt",
        target_classes=[21],  # COCO class 21 = cow
        output_format='ply'
    )

    print(f"Complete workflow finished!")
    print(f"  Depth frames: {depth_count}")
    print(f"  Point clouds: {pc_count}")
    print(f"  Time saved: {stats.processing_time_saved/60:.1f} minutes")
    print()


def example_zed_phase1_only():
    """Example: Generate depth frames only (Phase 1)."""
    from utils.zed_point_cloud_generator import ZedPointCloudWorkflow

    print("="*80)
    print("Example: ZED 2i - Phase 1 Only (Depth Generation)")
    print("="*80)
    print()

    session_dir = Path("./recordings/zed_session_001")

    workflow = ZedPointCloudWorkflow(session_dir, camera_index=0)

    # Phase 1: Generate depth frames with YOLO detection
    stats, depth_count = workflow.generate_selective_depth_frames(
        yolo_model="yolov8n.pt",
        target_classes=[21]  # Cows only
    )

    print(f"Phase 1 complete!")
    print(f"  Depth frames generated: {depth_count}")
    print(f"  Detection ratio: {stats.detection_ratio*100:.1f}%")
    print()


def example_zed_phase2_only():
    """Example: Generate point clouds from existing depth (Phase 2)."""
    from utils.zed_point_cloud_generator import ZedPointCloudWorkflow

    print("="*80)
    print("Example: ZED 2i - Phase 2 Only (Point Cloud Generation)")
    print("="*80)
    print()

    session_dir = Path("./recordings/zed_session_001")

    workflow = ZedPointCloudWorkflow(session_dir, camera_index=0)

    # Phase 2: Generate point clouds from existing depth frames
    pc_count = workflow.generate_point_clouds_from_depth(
        output_format='ply'
    )

    print(f"Phase 2 complete!")
    print(f"  Point clouds generated: {pc_count}")
    print()


def run_realsense_session(
    session_dir: Path,
    camera_index: int = 0,
    output_format: str = 'ply',
    mode: str = 'all',
    frame_indices: Optional[List[int]] = None
):
    """Run RealSense point cloud generation."""
    from utils.realsense_point_cloud_generator import process_realsense_session_point_clouds

    print(f"\n{'='*80}")
    print(f"RealSense D455i Point Cloud Generation")
    print(f"{'='*80}")
    print(f"Session: {session_dir}")
    print(f"Camera Index: {camera_index}")
    print(f"Mode: {mode}")
    print(f"Format: {output_format}")
    if frame_indices:
        print(f"Frame Indices: {frame_indices}")
    print()

    try:
        count = process_realsense_session_point_clouds(
            session_dir=session_dir,
            camera_index=camera_index,
            mode=mode,
            frame_indices=frame_indices,
            output_format=output_format
        )

        print(f"\n{'='*80}")
        print(f"RealSense Processing Complete")
        print(f"{'='*80}")
        print(f"Generated: {count} point clouds")
        print(f"{'='*80}\n")

        return count

    except Exception as e:
        print(f"\nError processing RealSense session: {e}")
        import traceback
        traceback.print_exc()
        return 0


def run_zed_session(
    session_dir: Path,
    camera_index: int = 0,
    yolo_model: str = "yolov8n.pt",
    target_classes: Optional[List[int]] = None,
    output_format: str = 'ply'
):
    """Run ZED 2i complete two-phase workflow."""
    from utils.zed_point_cloud_generator import process_zed_session_point_clouds

    if target_classes is None:
        target_classes = [21]  # Cow detection by default

    print(f"\n{'='*80}")
    print(f"ZED 2i Two-Phase Point Cloud Generation")
    print(f"{'='*80}")
    print(f"Session: {session_dir}")
    print(f"Camera Index: {camera_index}")
    print(f"YOLO Model: {yolo_model}")
    print(f"Target Classes: {target_classes}")
    print(f"Format: {output_format}")
    print()

    try:
        stats, depth_count, pc_count = process_zed_session_point_clouds(
            session_dir=session_dir,
            camera_index=camera_index,
            yolo_model=yolo_model,
            target_classes=target_classes,
            output_format=output_format
        )

        print(f"\n{'='*80}")
        print(f"ZED Processing Complete")
        print(f"{'='*80}")
        print(f"Depth Frames: {depth_count}")
        print(f"Point Clouds: {pc_count}")
        print(f"Detection Ratio: {stats.detection_ratio*100:.1f}%")
        print(f"Time Saved: {stats.processing_time_saved/60:.1f} minutes")
        print(f"{'='*80}\n")

        return pc_count

    except Exception as e:
        print(f"\nError processing ZED session: {e}")
        import traceback
        traceback.print_exc()
        return 0


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="On-Demand Point Cloud Generation for RealSense D455i and ZED 2i",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # RealSense - all frames
  python scripts/generate_point_clouds_on_demand.py --camera realsense --session ./recordings/session_001

  # RealSense - selective frames
  python scripts/generate_point_clouds_on_demand.py --camera realsense --session ./recordings/session_001 \
      --mode selective --frames 10,50,100,200

  # ZED 2i - complete workflow
  python scripts/generate_point_clouds_on_demand.py --camera zed --session ./recordings/session_001

  # ZED 2i - custom YOLO model
  python scripts/generate_point_clouds_on_demand.py --camera zed --session ./recordings/session_001 \
      --yolo-model yolov8s.pt --target-classes 21,19,20
        """
    )

    parser.add_argument(
        '--camera',
        type=str,
        choices=['realsense', 'zed'],
        required=True,
        help='Camera type (realsense or zed)'
    )

    parser.add_argument(
        '--session',
        type=str,
        required=True,
        help='Path to recording session directory'
    )

    parser.add_argument(
        '--camera-index',
        type=int,
        default=0,
        help='Camera index for multi-camera setups (default: 0)'
    )

    parser.add_argument(
        '--format',
        type=str,
        choices=['ply', 'pcd', 'xyz', 'npy'],
        default='ply',
        help='Output format (default: ply)'
    )

    # RealSense-specific options
    parser.add_argument(
        '--mode',
        type=str,
        choices=['all', 'selective'],
        default='all',
        help='Processing mode for RealSense (default: all)'
    )

    parser.add_argument(
        '--frames',
        type=str,
        help='Comma-separated frame indices for selective mode (e.g., 10,50,100,200)'
    )

    # ZED-specific options
    parser.add_argument(
        '--yolo-model',
        type=str,
        default='yolov8n.pt',
        help='YOLO model for ZED detection (default: yolov8n.pt)'
    )

    parser.add_argument(
        '--target-classes',
        type=str,
        default='21',
        help='Comma-separated COCO class IDs for ZED detection (default: 21 for cow)'
    )

    args = parser.parse_args()

    # Validate session directory
    session_dir = Path(args.session)
    if not session_dir.exists():
        print(f"Error: Session directory not found: {session_dir}")
        return 1

    # Process based on camera type
    if args.camera == 'realsense':
        # Parse frame indices if provided
        frame_indices = None
        if args.mode == 'selective':
            if not args.frames:
                print("Error: --frames required for selective mode")
                return 1
            frame_indices = [int(x.strip()) for x in args.frames.split(',')]

        run_realsense_session(
            session_dir=session_dir,
            camera_index=args.camera_index,
            output_format=args.format,
            mode=args.mode,
            frame_indices=frame_indices
        )

    elif args.camera == 'zed':
        # Parse target classes
        target_classes = [int(x.strip()) for x in args.target_classes.split(',')]

        run_zed_session(
            session_dir=session_dir,
            camera_index=args.camera_index,
            yolo_model=args.yolo_model,
            target_classes=target_classes,
            output_format=args.format
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
