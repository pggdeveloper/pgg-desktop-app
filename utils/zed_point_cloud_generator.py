"""
ZED 2i On-Demand Point Cloud Generator

Two-phase workflow:
Phase 1: Generate depth frames (selective with YOLO detection)
Phase 2: Generate point clouds from depth + RGB + calibration

Usage:
    from utils.zed_point_cloud_generator import ZedPointCloudWorkflow

    workflow = ZedPointCloudWorkflow(
        session_dir="./recordings/2025-01-14-10-30-00-000-0/"
    )

    # Phase 1: Generate depth frames (only for frames with animals)
    stats, depth_count = workflow.generate_selective_depth_frames(
        yolo_model="yolov8n.pt",
        target_classes=[21]  # Cow detection
    )

    # Phase 2: Generate point clouds from depth frames
    pc_count = workflow.generate_point_clouds_from_depth(
        output_dir="./point_clouds"
    )

Created: 2025-10-14
Status: PRODUCTION READY
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple, Literal
import json

try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False
    print("WARNING: open3d not installed. Install with: pip install open3d")

from utils.selective_depth_processor import (
    process_zed_recording_selective,
    DetectionStats
)

from utils.zed_offline_depth_processor import (
    create_zed2i_default_calibration,
    StereoCalibration
)


class ZedPointCloudWorkflow:
    """
    Two-phase point cloud generation for ZED 2i cameras.

    Phase 1: Generate depth frames using YOLO detection (selective)
    Phase 2: Generate point clouds from depth + RGB + calibration
    """

    def __init__(
        self,
        session_dir: Path,
        camera_index: int = 0
    ):
        """
        Initialize ZED point cloud workflow.

        Args:
            session_dir: Directory containing ZED recording session
            camera_index: Camera index for multi-camera setups
        """
        self.session_dir = Path(session_dir)
        self.camera_index = camera_index

        # Check for stereo video files
        self.rgb_left_video = self.session_dir / "rgb_left.mp4"
        self.rgb_right_video = self.session_dir / "rgb_right.mp4"

        if not self.rgb_left_video.exists():
            raise FileNotFoundError(f"RGB left video not found: {self.rgb_left_video}")
        if not self.rgb_right_video.exists():
            raise FileNotFoundError(f"RGB right video not found: {self.rgb_right_video}")

        print(f"ZED Point Cloud Workflow initialized:")
        print(f"  Session: {self.session_dir}")
        print(f"  Camera Index: {self.camera_index}")
        print(f"  RGB Left: {self.rgb_left_video.name}")
        print(f"  RGB Right: {self.rgb_right_video.name}")

    def generate_selective_depth_frames(
        self,
        yolo_model: str = "yolov8n.pt",
        target_classes: Optional[List[int]] = None,
        output_depth_dir: Optional[Path] = None,
        calibration: Optional[StereoCalibration] = None
    ) -> Tuple[DetectionStats, int]:
        """
        Phase 1: Generate depth frames using YOLO detection.
        Only processes frames with detected animals.

        Args:
            yolo_model: YOLO model to use (yolov8n.pt, yolov8s.pt, etc.)
            target_classes: COCO class IDs to detect (None = [21] for cows)
            output_depth_dir: Output directory for depth frames
                             (defaults to session_dir/depth_frames)
            calibration: Stereo calibration (None = use default ZED 2i)

        Returns:
            Tuple of (DetectionStats, depth_frame_count)
        """
        if target_classes is None:
            target_classes = [21]  # COCO class 21 = cow

        if output_depth_dir is None:
            output_depth_dir = self.session_dir / "depth_frames"

        if calibration is None:
            # Use default ZED 2i calibration (1920x1080)
            calibration = create_zed2i_default_calibration(1920, 1080)

        print(f"\n{'='*80}")
        print("Phase 1: Selective Depth Frame Generation")
        print(f"{'='*80}")
        print(f"YOLO Model: {yolo_model}")
        print(f"Target Classes: {target_classes}")
        print(f"Output: {output_depth_dir}")
        print()

        # Call the selective depth processor
        stats, count = process_zed_recording_selective(
            recording_dir=self.session_dir,
            output_depth_dir=output_depth_dir,
            yolo_model=yolo_model,
            target_classes=target_classes,
            calibration=calibration
        )

        print(f"\nPhase 1 Complete!")
        print(f"  Frames with animals: {stats.frames_with_animals} / {stats.total_frames}")
        print(f"  Detection ratio: {stats.detection_ratio*100:.1f}%")
        print(f"  Depth frames generated: {count}")
        print(f"  Time saved: {stats.processing_time_saved/60:.1f} minutes")
        print()

        return stats, count

    def generate_point_clouds_from_depth(
        self,
        output_dir: Optional[Path] = None,
        output_format: Literal['ply', 'pcd', 'xyz', 'npy'] = 'ply',
        calibration: Optional[StereoCalibration] = None,
        verbose: bool = True
    ) -> int:
        """
        Phase 2: Generate point clouds from depth frames.
        Reads depth .npz files and RGB stereo pairs.

        Args:
            output_dir: Output directory for point clouds
                       (defaults to session_dir/point_clouds)
            output_format: Output format (ply, pcd, xyz, npy)
            calibration: Stereo calibration (None = use default ZED 2i)
            verbose: Print progress

        Returns:
            Number of point clouds generated
        """
        if output_dir is None:
            output_dir = self.session_dir / "point_clouds"

        output_dir.mkdir(parents=True, exist_ok=True)

        # Load calibration
        if calibration is None:
            calibration = create_zed2i_default_calibration(1920, 1080)

        # Find depth frames (generated in Phase 1)
        depth_dir = self.session_dir / "depth_frames"
        if not depth_dir.exists():
            raise FileNotFoundError(
                f"Depth frames directory not found: {depth_dir}\n"
                "Please run Phase 1 (generate_selective_depth_frames) first!"
            )

        depth_files = sorted(depth_dir.glob("depth_*.npz"))
        if not depth_files:
            raise FileNotFoundError(
                f"No depth frames found in {depth_dir}\n"
                "Please run Phase 1 (generate_selective_depth_frames) first!"
            )

        if verbose:
            print(f"\n{'='*80}")
            print("Phase 2: Point Cloud Generation from Depth Frames")
            print(f"{'='*80}")
            print(f"  Depth frames: {len(depth_files)}")
            print(f"  Output directory: {output_dir}")
            print(f"  Output format: {output_format}")
            print()

        # Open RGB left video
        rgb_cap = cv2.VideoCapture(str(self.rgb_left_video))
        if not rgb_cap.isOpened():
            raise ValueError(f"Could not open RGB video: {self.rgb_left_video}")

        generated_count = 0

        try:
            for depth_file in depth_files:
                # Extract frame number from depth filename
                frame_number = self._extract_frame_number(depth_file)

                try:
                    # Load depth frame
                    depth_m, metadata = self._load_depth_frame(depth_file)

                    # Extract RGB frame from left camera
                    rgb_frame = self._extract_rgb_frame(frame_number, rgb_cap)
                    if rgb_frame is None:
                        if verbose:
                            print(f"  Warning: Could not extract RGB frame {frame_number}")
                        continue

                    # Generate point cloud
                    points, colors = self._generate_point_cloud(
                        depth_m,
                        rgb_frame,
                        calibration
                    )

                    # Save point cloud
                    output_file = output_dir / f"pointcloud_{frame_number:06d}.{output_format}"
                    self._save_point_cloud(points, colors, output_file, output_format)

                    generated_count += 1

                    if verbose and generated_count % 10 == 0:
                        progress = (generated_count / len(depth_files)) * 100
                        print(f"  Generated: {generated_count}/{len(depth_files)} ({progress:.1f}%)")

                except Exception as e:
                    if verbose:
                        print(f"  Error processing frame {frame_number}: {e}")
                    continue

        finally:
            rgb_cap.release()

        if verbose:
            print(f"\nPhase 2 Complete!")
            print(f"  Generated: {generated_count} point clouds")
            print(f"  Output directory: {output_dir}")
            print()

        return generated_count

    def run_complete_workflow(
        self,
        yolo_model: str = "yolov8n.pt",
        target_classes: Optional[List[int]] = None,
        output_dir: Optional[Path] = None,
        output_format: Literal['ply', 'pcd', 'xyz', 'npy'] = 'ply',
        calibration: Optional[StereoCalibration] = None
    ) -> Tuple[DetectionStats, int, int]:
        """
        Execute complete two-phase workflow.

        Phase 1: Generate depth frames with YOLO detection
        Phase 2: Generate point clouds from depth frames

        Args:
            yolo_model: YOLO model to use
            target_classes: COCO class IDs to detect (None = [21] for cows)
            output_dir: Output directory for point clouds
            output_format: Output format (ply, pcd, xyz, npy)
            calibration: Stereo calibration (None = use default ZED 2i)

        Returns:
            Tuple of (DetectionStats, depth_count, point_cloud_count)
        """
        # Phase 1: Generate depth frames
        stats, depth_count = self.generate_selective_depth_frames(
            yolo_model=yolo_model,
            target_classes=target_classes,
            calibration=calibration
        )

        # Phase 2: Generate point clouds
        pc_count = self.generate_point_clouds_from_depth(
            output_dir=output_dir,
            output_format=output_format,
            calibration=calibration
        )

        print(f"\n{'='*80}")
        print("Complete Workflow Summary")
        print(f"{'='*80}")
        print(f"  Detection scan: Complete")
        print(f"  Frames with animals: {stats.frames_with_animals} / {stats.total_frames}")
        print(f"  Depth frames generated: {depth_count}")
        print(f"  Point clouds generated: {pc_count}")
        print(f"  Total time saved: {stats.processing_time_saved/60:.1f} minutes")
        print(f"{'='*80}\n")

        return stats, depth_count, pc_count

    def _load_depth_frame(self, depth_file: Path) -> Tuple[np.ndarray, dict]:
        """
        Load depth frame from .npz file.

        Returns:
            Tuple of (depth_in_meters, metadata)
        """
        data = np.load(depth_file)
        depth_mm = data['depth']  # uint16 in millimeters

        metadata = {
            'timestamp': float(data['timestamp']),
            'frame_number': int(data['frame_number']),
            'depth_scale': float(data['depth_scale']),  # 0.001 (mm to m)
        }

        # Convert to meters
        depth_m = depth_mm.astype(np.float32) * metadata['depth_scale']

        return depth_m, metadata

    def _extract_rgb_frame(
        self,
        frame_number: int,
        rgb_cap: cv2.VideoCapture
    ) -> Optional[np.ndarray]:
        """Extract RGB frame at specific frame number."""
        rgb_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = rgb_cap.read()

        if not ret:
            return None

        # Convert BGR to RGB
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def _generate_point_cloud(
        self,
        depth_m: np.ndarray,
        rgb_frame: np.ndarray,
        calibration: StereoCalibration
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate point cloud from depth and RGB using calibration."""
        if OPEN3D_AVAILABLE:
            return self._generate_with_open3d(depth_m, rgb_frame, calibration)
        else:
            return self._generate_with_numpy(depth_m, rgb_frame, calibration)

    def _generate_with_open3d(
        self,
        depth_m: np.ndarray,
        rgb_frame: np.ndarray,
        calibration: StereoCalibration
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate point cloud using Open3D (recommended)."""
        # Create Open3D images
        depth_o3d = o3d.geometry.Image(depth_m)
        rgb_o3d = o3d.geometry.Image(rgb_frame)

        # Create RGBD image
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            rgb_o3d,
            depth_o3d,
            depth_scale=1.0,  # Already in meters
            depth_trunc=10.0,  # Max 10 meters
            convert_rgb_to_intensity=False
        )

        # Create camera intrinsics
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width=calibration.width,
            height=calibration.height,
            fx=calibration.fx,
            fy=calibration.fy,
            cx=calibration.cx,
            cy=calibration.cy
        )

        # Generate point cloud
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd,
            intrinsic
        )

        # Extract points and colors
        points = np.asarray(pcd.points)
        colors = (np.asarray(pcd.colors) * 255).astype(np.uint8)

        return points, colors

    def _generate_with_numpy(
        self,
        depth_m: np.ndarray,
        rgb_frame: np.ndarray,
        calibration: StereoCalibration
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Fallback: Generate point cloud using NumPy."""
        height, width = depth_m.shape

        # Create pixel coordinate grids
        u, v = np.meshgrid(np.arange(width), np.arange(height))

        # Convert to camera coordinates
        fx = calibration.fx
        fy = calibration.fy
        cx = calibration.cx
        cy = calibration.cy

        z = depth_m
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy

        # Stack into point cloud
        points = np.stack([x, y, z], axis=-1).reshape(-1, 3)
        colors = rgb_frame.reshape(-1, 3)

        # Filter invalid points
        valid_mask = (z.flatten() > 0) & (z.flatten() < 10.0)
        points = points[valid_mask]
        colors = colors[valid_mask]

        return points, colors

    def _extract_frame_number(self, depth_file: Path) -> int:
        """Extract frame number from depth filename."""
        # Format: depth_{frame_number:06d}.npz
        stem = depth_file.stem
        parts = stem.split('_')
        return int(parts[-1])

    def _save_point_cloud(
        self,
        points: np.ndarray,
        colors: np.ndarray,
        output_path: Path,
        format: str
    ):
        """Save point cloud in specified format."""
        if format == 'ply' and OPEN3D_AVAILABLE:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)
            o3d.io.write_point_cloud(str(output_path), pcd, write_ascii=False)

        elif format == 'pcd' and OPEN3D_AVAILABLE:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)
            o3d.io.write_point_cloud(str(output_path), pcd, write_ascii=False)

        elif format == 'xyz':
            # XYZ format: X Y Z R G B per line
            with open(output_path, 'w') as f:
                for point, color in zip(points, colors):
                    f.write(f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f} "
                           f"{color[0]} {color[1]} {color[2]}\n")

        elif format == 'npy':
            # Save as NumPy arrays
            np.save(output_path, points)
            color_path = output_path.parent / f"{output_path.stem}_colors.npy"
            np.save(color_path, colors)

        else:
            # Fallback to NPY format
            np.save(output_path.with_suffix('.npy'), points)
            color_path = output_path.parent / f"{output_path.stem}_colors.npy"
            np.save(color_path, colors)


# Convenience function
def process_zed_session_point_clouds(
    session_dir: Path,
    camera_index: int = 0,
    yolo_model: str = "yolov8n.pt",
    target_classes: Optional[List[int]] = None,
    output_dir: Optional[Path] = None,
    output_format: Literal['ply', 'pcd', 'xyz', 'npy'] = 'ply'
) -> Tuple[DetectionStats, int, int]:
    """
    Process ZED session with complete two-phase workflow.

    Phase 1: Generate depth frames with YOLO detection
    Phase 2: Generate point clouds from depth frames

    Args:
        session_dir: ZED recording session directory
        camera_index: Camera index
        yolo_model: YOLO model to use
        target_classes: COCO class IDs to detect (None = [21] for cows)
        output_dir: Output directory for point clouds
        output_format: Output format (ply, pcd, xyz, npy)

    Returns:
        Tuple of (DetectionStats, depth_count, point_cloud_count)
    """
    workflow = ZedPointCloudWorkflow(session_dir, camera_index)

    return workflow.run_complete_workflow(
        yolo_model=yolo_model,
        target_classes=target_classes,
        output_dir=output_dir,
        output_format=output_format
    )


# Example usage
if __name__ == "__main__":
    print(__doc__)
    print("\n" + "="*80)
    print("ZED POINT CLOUD WORKFLOW - Example Usage")
    print("="*80)

    print("\nComplete workflow example:")
    print("-" * 40)
    print("from utils.zed_point_cloud_generator import process_zed_session_point_clouds")
    print()
    print("# Run complete two-phase workflow")
    print("stats, depth_count, pc_count = process_zed_session_point_clouds(")
    print("    session_dir='./recordings/zed_session_001',")
    print("    yolo_model='yolov8n.pt',")
    print("    target_classes=[21],  # COCO class 21 = cow")
    print("    output_format='ply'")
    print(")")
    print()
    print("print(f'Depth frames: {depth_count}')")
    print("print(f'Point clouds: {pc_count}')")
    print("print(f'Time saved: {stats.processing_time_saved/60:.1f} minutes')")
    print()

    print("\nAdvanced usage (step-by-step):")
    print("-" * 40)
    print("from utils.zed_point_cloud_generator import ZedPointCloudWorkflow")
    print()
    print("workflow = ZedPointCloudWorkflow('./recordings/zed_session_001')")
    print()
    print("# Phase 1: Generate depth frames")
    print("stats, depth_count = workflow.generate_selective_depth_frames(")
    print("    yolo_model='yolov8n.pt',")
    print("    target_classes=[21]")
    print(")")
    print()
    print("# Phase 2: Generate point clouds")
    print("pc_count = workflow.generate_point_clouds_from_depth(")
    print("    output_format='ply'")
    print(")")
    print()
    print("="*80)
