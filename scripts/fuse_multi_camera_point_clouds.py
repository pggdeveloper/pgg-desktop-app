"""
Multi-Camera Point Cloud Fusion CLI

Command-line interface for fusing point clouds from multi-camera sessions.

Features:
- Automatic point cloud generation if missing
- Complete or selective frame processing
- Quality metrics and validation
- Multiple export formats

Usage:
    # Fuse all frames
    python scripts/fuse_multi_camera_point_clouds.py \
        --session ./recordings/session_001

    # Fuse specific frames only
    python scripts/fuse_multi_camera_point_clouds.py \
        --session ./recordings/session_001 \
        --frames 10,50,100,200

    # Custom voxel size and format
    python scripts/fuse_multi_camera_point_clouds.py \
        --session ./recordings/session_001 \
        --voxel-size 0.02 \
        --format pcd

Created: 2025-10-14
Status: PRODUCTION READY
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, List


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Multi-Camera Point Cloud Fusion for Cattle Monitoring",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fuse all frames from session
  python scripts/fuse_multi_camera_point_clouds.py --session ./recordings/session_001

  # Fuse specific frames only
  python scripts/fuse_multi_camera_point_clouds.py --session ./recordings/session_001 \
      --frames 10,50,100,200

  # High quality fusion (smaller voxels)
  python scripts/fuse_multi_camera_point_clouds.py --session ./recordings/session_001 \
      --voxel-size 0.005

  # Disable outlier removal (faster, lower quality)
  python scripts/fuse_multi_camera_point_clouds.py --session ./recordings/session_001 \
      --no-outlier-removal
        """
    )

    parser.add_argument(
        '--session',
        type=str,
        required=True,
        help='Path to recording session directory'
    )

    parser.add_argument(
        '--frames',
        type=str,
        help='Comma-separated frame indices for selective processing (e.g., 10,50,100,200)'
    )

    parser.add_argument(
        '--format',
        type=str,
        choices=['ply', 'pcd'],
        default='ply',
        help='Output format for point clouds (default: ply)'
    )

    parser.add_argument(
        '--voxel-size',
        type=float,
        default=0.01,
        help='Voxel size for downsampling in meters (default: 0.01 = 1cm)'
    )

    parser.add_argument(
        '--no-outlier-removal',
        action='store_true',
        help='Disable outlier removal (faster but lower quality)'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )

    args = parser.parse_args()

    # Configure logging
    if args.verbose:
        import logging
        logging.basicConfig(level=logging.INFO)

    # Parse frame indices
    frame_indices = None
    if args.frames:
        frame_indices = [int(x.strip()) for x in args.frames.split(',')]

    # Validate session directory
    session_dir = Path(args.session)
    if not session_dir.exists():
        print(f"Error: Session directory not found: {session_dir}")
        return 1

    # Run fusion workflow
    from utils.multi_camera_point_cloud_fusion_workflow import fuse_session_point_clouds

    print(f"\n{'='*80}")
    print("Multi-Camera Point Cloud Fusion")
    print(f"{'='*80}")
    print(f"Session: {session_dir}")
    print(f"Format: {args.format}")
    print(f"Voxel size: {args.voxel_size}m")
    print(f"Outlier removal: {not args.no_outlier_removal}")
    if frame_indices:
        print(f"Frames: {len(frame_indices)} selected")
    else:
        print("Frames: ALL")
    print()

    try:
        result = fuse_session_point_clouds(
            session_dir=session_dir,
            output_format=args.format,
            voxel_size=args.voxel_size,
            remove_outliers=not args.no_outlier_removal,
            frame_indices=frame_indices
        )

        print(f"\n{'='*80}")
        print("Fusion Complete!")
        print(f"{'='*80}")
        print(f"Frames processed: {result.num_frames_processed}")
        print(f"Frames fused: {result.num_point_clouds_fused}")
        print(f"Average quality: {result.average_coverage_quality}")
        print(f"Processing time: {result.total_processing_time/60:.1f} minutes")
        print(f"Output directory: {result.output_directory}")
        print(f"{'='*80}\n")

        return 0

    except Exception as e:
        print(f"\nError during fusion: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
