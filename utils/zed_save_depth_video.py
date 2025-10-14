"""
Legacy Stereolabs ZED 2i Depth Video Saving - MP4 Format (TEMPLATE FOR FUTURE USE)

This module contains template code for depth frame saving to MP4 video format.
Currently, ZED cameras run in CPU mode without GPU depth computation.

STATUS: NOT IMPLEMENTED - Template only

When GPU depth computation is enabled, this code can be used as reference.
However, it is STRONGLY RECOMMENDED to use .npz format instead for the same
reasons as RealSense:
- 40% smaller file size
- 3-5x faster loading
- Metadata preservation (timestamp, frame_number, depth_scale)
- Direct compatibility with pyzed SDK
- No data loss (uint16 vs uint8)

Current ZED Configuration (utils/zed_camera.py):
- Depth mode: sl.DEPTH_MODE.NONE (CPU mode, no GPU)
- Streams saved: RGB left, RGB right, stereo side-by-side
- IMU data: Saved to CSV if using SDK

Created: 2025-10-14
Status: TEMPLATE - Not currently used
"""

import cv2
import numpy as np


def create_depth_video_writer_mp4_zed(output_dir, filename, fourcc, fps, width, height):
    """
    Create OpenCV VideoWriter for ZED depth stream (grayscale visualization).

    NOT CURRENTLY USED: ZED cameras run in CPU mode without depth computation.

    When GPU depth is enabled, this function would create a video writer
    for depth frames similar to RealSense implementation.

    Args:
        output_dir: Output directory path
        filename: Depth video filename
        fourcc: Video codec FourCC
        fps: Frame rate
        width: Frame width (1920 for single camera)
        height: Frame height (1080)

    Returns:
        cv2.VideoWriter instance

    Example (TEMPLATE - NOT USED):
        >>> fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        >>> depth_writer = create_depth_video_writer_mp4_zed(
        ...     output_dir,
        ...     "depth_left.mp4",
        ...     fourcc,
        ...     30,
        ...     1920,
        ...     1080
        ... )
    """
    depth_writer = cv2.VideoWriter(
        str(output_dir / filename),
        fourcc,
        fps,
        (width, height),
        isColor=False
    )
    return depth_writer


def save_depth_frame_to_video_mp4_zed(depth_writer, depth_map):
    """
    Save ZED depth frame to MP4 video (converts to 8-bit grayscale visualization).

    NOT CURRENTLY USED: Use save_depth_frame_npz() instead when depth is enabled.

    This method would:
    1. Convert float32 depth to uint16 (millimeters)
    2. Convert uint16 to 8-bit using scaling
    3. Apply JET colormap for visualization
    4. Convert to grayscale
    5. Write to video writer

    PROBLEMS with this approach (same as RealSense):
    - Loses precision (float32 → uint8 = massive data loss)
    - Not suitable for reconstruction (only visualization)
    - Larger file size than .npz
    - Slower to load than .npz
    - No metadata preservation

    Args:
        depth_writer: cv2.VideoWriter instance
        depth_map: Depth map as numpy array (float32 in meters from ZED SDK)

    Example (TEMPLATE - NOT USED):
        >>> # depth_map is float32 in meters from sl.Mat
        >>> depth_mm = (depth_map * 1000).astype(np.uint16)  # Convert to millimeters
        >>> depth_colormap = cv2.applyColorMap(
        ...     cv2.convertScaleAbs(depth_mm, alpha=0.03),
        ...     cv2.COLORMAP_JET
        ... )
        >>> depth_gray = cv2.cvtColor(depth_colormap, cv2.COLOR_BGR2GRAY)
        >>> save_depth_frame_to_video_mp4_zed(depth_writer, depth_map)
    """
    # Convert depth from meters (float32) to millimeters (uint16)
    depth_mm = (depth_map * 1000).astype(np.uint16)

    # Clip extreme values for better visualization
    depth_mm = np.clip(depth_mm, 0, 10000)  # 0-10 meters

    # Convert to 8-bit colormap for visualization
    depth_colormap = cv2.applyColorMap(
        cv2.convertScaleAbs(depth_mm, alpha=0.03),
        cv2.COLORMAP_JET
    )

    # Convert to grayscale
    depth_gray = cv2.cvtColor(depth_colormap, cv2.COLOR_BGR2GRAY)

    # Write to video
    depth_writer.write(depth_gray)


def example_usage_zed_recording_loop_with_depth():
    """
    Example of how depth saving would be integrated in ZED recording loop.

    TEMPLATE ONLY - NOT CURRENTLY IMPLEMENTED.

    To enable depth computation in ZED:
    1. Change depth_mode in _initialize_with_sdk():
       init_params.depth_mode = sl.DEPTH_MODE.NEURAL  # or ULTRA, QUALITY, PERFORMANCE

    2. Add depth video writers in _create_video_writers()

    3. Retrieve depth maps in _capture_frame_sdk()

    4. Save frames (preferably as .npz, not MP4)

    For the NEW implementation, see recommendations in:
    - docs/files_needed/cattle_monitoring_progress.md
    """

    # TEMPLATE CODE FOR zed_camera.py (would be added if depth enabled):
    # =====================================================================

    # In _initialize_with_sdk() (line 112):
    # -------------------------------------
    # # Enable depth computation (requires NVIDIA GPU)
    # init_params.depth_mode = sl.DEPTH_MODE.NEURAL  # Best quality
    # # OR
    # init_params.depth_mode = sl.DEPTH_MODE.ULTRA   # Balanced
    # # OR
    # init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE  # Fastest

    # In _create_video_writers():
    # ---------------------------
    # # Left depth stream
    # left_depth_filename = self._generate_filename("depth_left", "mp4")
    # self.left_depth_writer = cv2.VideoWriter(
    #     str(self.output_dir / left_depth_filename),
    #     fourcc,
    #     self.fps,
    #     (self.single_width, self.height),
    #     isColor=False
    # )

    # In _capture_frame_sdk():
    # ------------------------
    # # Retrieve left depth map
    # depth_left = sl.Mat()
    # self.camera.retrieve_measure(depth_left, sl.MEASURE.DEPTH)
    # depth_left_data = depth_left.get_data()  # float32 in meters

    # # Save depth frame (OLD MP4 way - NOT RECOMMENDED)
    # save_depth_frame_to_video_mp4_zed(self.left_depth_writer, depth_left_data)

    # # OR BETTER: Save as .npz (RECOMMENDED)
    # depth_filename = self._generate_filename(f"depth_left_{self.frame_count:06d}", "npz")
    # np.savez_compressed(
    #     self.output_dir / depth_filename,
    #     depth=depth_left_data,  # float32 in meters (or convert to uint16 mm)
    #     timestamp=timestamp,
    #     frame_number=self.frame_count,
    #     depth_unit="meters",  # or "millimeters" if converted
    #     camera_index=self.camera_info.index,
    #     camera_side="left"
    # )

    pass


def comparison_zed_vs_realsense_depth():
    """
    Comparison: ZED depth vs RealSense depth characteristics

    ZED 2i Depth (Stereo-based):
    -----------------------------
    - Technology: Passive stereo matching
    - Range: 0.3m - 20m (configurable)
    - Output: float32 depth maps in meters
    - Requires: GPU for neural depth (or CPU for basic stereo)
    - Accuracy: Sub-centimeter at close range
    - Field of View: ~110° (H) x ~70° (V)

    RealSense D455i Depth (Active stereo):
    --------------------------------------
    - Technology: Active IR stereo
    - Range: 0.6m - 6m (optimal)
    - Output: uint16 depth values (depth_scale = 0.001 for mm)
    - Requires: No GPU needed
    - Accuracy: <2% at 4m
    - Field of View: ~87° (H) x ~58° (V)

    DEPTH SAVING RECOMMENDATION (Both cameras):
    -------------------------------------------
    Use .npz format with:
    - ZED: float32 meters (or uint16 millimeters after conversion)
    - RealSense: uint16 millimeters (as provided by SDK)
    - Metadata: timestamp, frame_number, depth_unit, camera_index
    - Advantages: 40% smaller, 3-5x faster, no data loss
    """
    pass


def gpu_requirements_note():
    """
    GPU Requirements for ZED Depth Computation

    CURRENT STATUS: CPU mode only (no depth computation)

    To enable depth computation, system needs:
    - NVIDIA GPU (GTX 1050 or better)
    - CUDA Toolkit 11.x or 12.x
    - ZED SDK 4.x with CUDA support
    - Sufficient VRAM (2GB minimum, 4GB recommended)

    Available depth modes (GPU required):
    - PERFORMANCE: Fastest, lowest quality (~90 FPS)
    - QUALITY: Balanced quality/speed (~60 FPS)
    - ULTRA: High quality (~45 FPS)
    - NEURAL: Best quality, slowest (~30 FPS), requires RTX GPU

    Current workaround:
    - Save stereo RGB pairs (left + right)
    - Perform offline depth reconstruction using:
      * OpenCV stereo matching
      * Dense stereo reconstruction utilities
      * Multi-camera fusion for better accuracy

    See:
    - utils/dense_stereo_reconstruction.py
    - utils/depth_fusion.py
    """
    pass


if __name__ == "__main__":
    print(__doc__)
    print("\n" + "="*80)
    print("ℹ️  INFORMATION: This module is a TEMPLATE")
    print("="*80)
    print("\nZED cameras currently run in CPU mode without depth computation.")
    print("This file contains template code for when GPU depth is enabled.")
    print("\nFor current ZED implementation, see:")
    print("  - utils/zed_camera.py")
    print("  - docs/files_needed/cattle_monitoring_progress.md")
    print("\nFor depth reconstruction without GPU, see:")
    print("  - utils/dense_stereo_reconstruction.py")
    print("  - utils/depth_fusion.py")
    print("\n" + "="*80)
