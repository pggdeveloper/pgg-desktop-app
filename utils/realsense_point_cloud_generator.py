"""
RealSense D455i On-Demand Point Cloud Generator

Generates point clouds from recorded session files:
- Depth frames (.npz files)
- RGB video (.mp4 file)
- Calibration metadata (.json file)

Usage:
    from utils.realsense_point_cloud_generator import RealSensePointCloudGenerator

    generator = RealSensePointCloudGenerator(
        session_dir="./recordings/2025-01-14-10-30-00-000-0/"
    )

    # Generate all point clouds
    generator.generate_all_point_clouds(output_dir="./point_clouds")

    # Or selective generation (frames with animals detected)
    generator.generate_selective_point_clouds(
        frame_indices=[10, 50, 100, 200],
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

try:
    import pyrealsense2 as rs
    PYREALSENSE_AVAILABLE = True
except ImportError:
    PYREALSENSE_AVAILABLE = False


class RealSensePointCloudGenerator:
    """
    Generate point clouds on-demand from RealSense D455i recordings.

    This class processes pre-recorded session data to generate colored
    point clouds without real-time constraints.
    """

    def __init__(
        self,
        session_dir: Path,
        camera_index: int = 0
    ):
        """
        Initialize point cloud generator.

        Args:
            session_dir: Directory containing recording session
            camera_index: Camera index for multi-camera setups
        """
        self.session_dir = Path(session_dir)
        self.camera_index = camera_index

        # Load calibration
        self.calibration = self._load_calibration()

        # Find depth and RGB files
        self.depth_files = sorted(self.session_dir.glob(f"*-{camera_index}-depth_*.npz"))
        self.rgb_video = self._find_rgb_video()

        if not self.depth_files:
            raise FileNotFoundError(f"No depth frames found in {session_dir}")
        if not self.rgb_video:
            raise FileNotFoundError(f"No RGB video found in {session_dir}")

        print(f"RealSense Point Cloud Generator initialized:")
        print(f"  Session: {self.session_dir}")
        print(f"  Camera Index: {self.camera_index}")
        print(f"  Depth frames: {len(self.depth_files)}")
        print(f"  RGB video: {self.rgb_video.name}")

    def _load_calibration(self) -> dict:
        """Load camera calibration from session directory."""
        calib_file = self.session_dir / "calibration_metadata.json"
        if not calib_file.exists():
            raise FileNotFoundError(f"Calibration file not found: {calib_file}")

        with open(calib_file, 'r') as f:
            calib_data = json.load(f)

        # Extract intrinsics for this camera
        camera_key = f"camera_{self.camera_index}"
        if camera_key not in calib_data:
            raise ValueError(f"Camera {self.camera_index} not found in calibration")

        return calib_data[camera_key]['intrinsics']

    def _find_rgb_video(self) -> Optional[Path]:
        """Find RGB video file for this camera."""
        rgb_pattern = f"*-{self.camera_index}-rgb.mp4"
        rgb_files = list(self.session_dir.glob(rgb_pattern))
        return rgb_files[0] if rgb_files else None

    def _load_depth_frame(self, depth_file: Path) -> Tuple[np.ndarray, dict]:
        """
        Load depth frame from .npz file.

        Returns:
            Tuple of (depth_array, metadata)
        """
        data = np.load(depth_file)
        depth = data['depth']  # uint16 in millimeters

        metadata = {
            'timestamp': float(data['timestamp']),
            'frame_number': int(data['frame_number']),
            'depth_scale': float(data['depth_scale']),  # 0.001 (mm to m)
            'camera_index': int(data['camera_index'])
        }

        return depth, metadata

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

    def generate_point_cloud_from_files(
        self,
        depth_file: Path,
        frame_number: int,
        rgb_cap: cv2.VideoCapture
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate point cloud from depth and RGB files.

        Returns:
            Tuple of (points [N, 3], colors [N, 3])
        """
        # Load depth
        depth_mm, metadata = self._load_depth_frame(depth_file)
        depth_m = depth_mm.astype(np.float32) * metadata['depth_scale']

        # Extract RGB frame
        rgb_frame = self._extract_rgb_frame(frame_number, rgb_cap)
        if rgb_frame is None:
            raise ValueError(f"Could not extract RGB frame {frame_number}")

        # Generate point cloud using Open3D
        if OPEN3D_AVAILABLE:
            return self._generate_with_open3d(depth_m, rgb_frame)
        else:
            return self._generate_with_numpy(depth_m, rgb_frame)

    def _generate_with_open3d(
        self,
        depth_m: np.ndarray,
        rgb_frame: np.ndarray
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
            width=int(self.calibration['width']),
            height=int(self.calibration['height']),
            fx=self.calibration['fx'],
            fy=self.calibration['fy'],
            cx=self.calibration['cx'],
            cy=self.calibration['cy']
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
        rgb_frame: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Fallback: Generate point cloud using NumPy."""
        height, width = depth_m.shape

        # Create pixel coordinate grids
        u, v = np.meshgrid(np.arange(width), np.arange(height))

        # Convert to camera coordinates
        fx = self.calibration['fx']
        fy = self.calibration['fy']
        cx = self.calibration['cx']
        cy = self.calibration['cy']

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

    def generate_all_point_clouds(
        self,
        output_dir: Path,
        output_format: Literal['ply', 'pcd', 'xyz', 'npy'] = 'ply',
        verbose: bool = True
    ) -> int:
        """
        Generate point clouds for all frames.

        Args:
            output_dir: Directory to save point clouds
            output_format: Output format (ply, pcd, xyz, npy)
            verbose: Print progress

        Returns:
            Number of point clouds generated
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Open RGB video
        rgb_cap = cv2.VideoCapture(str(self.rgb_video))

        if not rgb_cap.isOpened():
            raise ValueError(f"Could not open RGB video: {self.rgb_video}")

        generated_count = 0

        if verbose:
            print(f"\nGenerating point clouds for all frames...")
            print(f"  Total depth frames: {len(self.depth_files)}")
            print(f"  Output directory: {output_dir}")
            print(f"  Output format: {output_format}")
            print()

        try:
            for depth_file in self.depth_files:
                # Extract frame number from filename
                frame_number = self._extract_frame_number(depth_file)

                try:
                    # Generate point cloud
                    points, colors = self.generate_point_cloud_from_files(
                        depth_file,
                        frame_number,
                        rgb_cap
                    )

                    # Save point cloud
                    output_file = output_dir / f"pointcloud_{frame_number:06d}.{output_format}"
                    self._save_point_cloud(points, colors, output_file, output_format)

                    generated_count += 1

                    if verbose and generated_count % 30 == 0:
                        progress = (generated_count / len(self.depth_files)) * 100
                        print(f"  Generated: {generated_count}/{len(self.depth_files)} ({progress:.1f}%)")

                except Exception as e:
                    print(f"  Error processing frame {frame_number}: {e}")
                    continue

        finally:
            rgb_cap.release()

        if verbose:
            print(f"\nPoint cloud generation complete!")
            print(f"  Generated: {generated_count} point clouds")
            print(f"  Output directory: {output_dir}")
            print()

        return generated_count

    def generate_selective_point_clouds(
        self,
        frame_indices: List[int],
        output_dir: Path,
        output_format: Literal['ply', 'pcd', 'xyz', 'npy'] = 'ply',
        verbose: bool = True
    ) -> int:
        """
        Generate point clouds for specific frames only.

        Args:
            frame_indices: List of frame numbers to process
            output_dir: Directory to save point clouds
            output_format: Output format
            verbose: Print progress

        Returns:
            Number of point clouds generated
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create frame index set for O(1) lookup
        frame_set = set(frame_indices)

        # Open RGB video
        rgb_cap = cv2.VideoCapture(str(self.rgb_video))

        if not rgb_cap.isOpened():
            raise ValueError(f"Could not open RGB video: {self.rgb_video}")

        generated_count = 0

        if verbose:
            print(f"\nGenerating point clouds for selected frames...")
            print(f"  Total frames to process: {len(frame_indices)}")
            print(f"  Output directory: {output_dir}")
            print(f"  Output format: {output_format}")
            print()

        try:
            for depth_file in self.depth_files:
                frame_number = self._extract_frame_number(depth_file)

                # Skip if not in selective list
                if frame_number not in frame_set:
                    continue

                try:
                    # Generate point cloud
                    points, colors = self.generate_point_cloud_from_files(
                        depth_file,
                        frame_number,
                        rgb_cap
                    )

                    # Save point cloud
                    output_file = output_dir / f"pointcloud_{frame_number:06d}.{output_format}"
                    self._save_point_cloud(points, colors, output_file, output_format)

                    generated_count += 1

                    if verbose:
                        progress = (generated_count / len(frame_indices)) * 100
                        print(f"  Generated: {generated_count}/{len(frame_indices)} ({progress:.1f}%)")

                except Exception as e:
                    print(f"  Error processing frame {frame_number}: {e}")
                    continue

        finally:
            rgb_cap.release()

        if verbose:
            print(f"\nSelective point cloud generation complete!")
            print(f"  Generated: {generated_count} point clouds")
            print(f"  Output directory: {output_dir}")
            print()

        return generated_count

    def _extract_frame_number(self, depth_file: Path) -> int:
        """Extract frame number from depth filename."""
        # Format: *-{camera_index}-depth_{frame_number:06d}.npz
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
def process_realsense_session_point_clouds(
    session_dir: Path,
    camera_index: int = 0,
    output_dir: Optional[Path] = None,
    mode: Literal['all', 'selective'] = 'all',
    frame_indices: Optional[List[int]] = None,
    output_format: Literal['ply', 'pcd', 'xyz', 'npy'] = 'ply'
) -> int:
    """
    Process RealSense session to generate point clouds on-demand.

    Args:
        session_dir: Recording session directory
        camera_index: Camera index
        output_dir: Output directory (defaults to session_dir/point_clouds)
        mode: 'all' or 'selective'
        frame_indices: Frame indices for selective mode
        output_format: Output format (ply, pcd, xyz, npy)

    Returns:
        Number of point clouds generated
    """
    if output_dir is None:
        output_dir = Path(session_dir) / "point_clouds"

    generator = RealSensePointCloudGenerator(session_dir, camera_index)

    if mode == 'all':
        return generator.generate_all_point_clouds(output_dir, output_format=output_format)
    else:
        if frame_indices is None:
            raise ValueError("frame_indices required for selective mode")
        return generator.generate_selective_point_clouds(
            frame_indices,
            output_dir,
            output_format=output_format
        )


# Example usage
if __name__ == "__main__":
    print(__doc__)
    print("\n" + "="*80)
    print("REALSENSE POINT CLOUD GENERATOR - Example Usage")
    print("="*80)

    print("\nQuick example:")
    print("-" * 40)
    print("from utils.realsense_point_cloud_generator import process_realsense_session_point_clouds")
    print()
    print("# Generate all point clouds")
    print("count = process_realsense_session_point_clouds(")
    print("    session_dir='./recordings/session_001',")
    print("    camera_index=0,")
    print("    mode='all',")
    print("    output_format='ply'")
    print(")")
    print()
    print("# Or selective generation (only specific frames)")
    print("count = process_realsense_session_point_clouds(")
    print("    session_dir='./recordings/session_001',")
    print("    camera_index=0,")
    print("    mode='selective',")
    print("    frame_indices=[10, 50, 100, 200, 500],")
    print("    output_format='ply'")
    print(")")
    print()
    print("="*80)
