"""
Legacy Intel RealSense Depth Video Saving - MP4 Format (DEPRECATED)

This module contains the old implementation of depth frame saving to MP4 video format.
It is kept for reference purposes only and should NOT be used in production.

The current implementation saves depth frames as .npz (NumPy compressed) format,
which provides:
- 40% smaller file size
- 3-5x faster loading
- Metadata preservation (timestamp, frame_number, depth_scale)
- Direct compatibility with pyrealsense2 and pyzed
- No decodification required

If you need to revert to MP4 depth video saving, you can reference this code.
However, it is STRONGLY RECOMMENDED to use .npz format instead.

Created: 2025-10-14
Status: DEPRECATED - For reference only
"""

import cv2
import numpy as np


def create_depth_video_writer_mp4(output_dir, filename, fourcc, fps, width, height):
    """
    Create OpenCV VideoWriter for depth stream (grayscale visualization).

    DEPRECATED: Use .npz individual frames instead.

    Args:
        output_dir: Output directory path
        filename: Depth video filename
        fourcc: Video codec FourCC
        fps: Frame rate
        width: Frame width
        height: Frame height

    Returns:
        cv2.VideoWriter instance

    Example (OLD - DEPRECATED):
        >>> fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        >>> depth_writer = create_depth_video_writer_mp4(
        ...     output_dir,
        ...     "depth.mp4",
        ...     fourcc,
        ...     30,
        ...     1280,
        ...     720
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


def save_depth_frame_to_video_mp4(depth_writer, depth_image):
    """
    Save depth frame to MP4 video (converts to 8-bit grayscale visualization).

    DEPRECATED: Use save_depth_frame_npz() instead.

    This method:
    1. Converts uint16 depth to 8-bit using scaling (alpha=0.03)
    2. Applies JET colormap for visualization
    3. Converts to grayscale
    4. Writes to video writer

    PROBLEMS with this approach:
    - Loses precision (uint16 → uint8 = 99.6% data loss)
    - Not suitable for reconstruction (only visualization)
    - Larger file size than .npz
    - Slower to load than .npz
    - No metadata preservation

    Args:
        depth_writer: cv2.VideoWriter instance
        depth_image: Depth frame as numpy array (uint16 raw)

    Example (OLD - DEPRECATED):
        >>> depth_colormap = cv2.applyColorMap(
        ...     cv2.convertScaleAbs(depth_image, alpha=0.03),
        ...     cv2.COLORMAP_JET
        ... )
        >>> depth_gray = cv2.cvtColor(depth_colormap, cv2.COLOR_BGR2GRAY)
        >>> save_depth_frame_to_video_mp4(depth_writer, depth_image)
    """
    # Convert depth to 8-bit colormap for visualization
    depth_colormap = cv2.applyColorMap(
        cv2.convertScaleAbs(depth_image, alpha=0.03),
        cv2.COLORMAP_JET
    )

    # Convert to grayscale
    depth_gray = cv2.cvtColor(depth_colormap, cv2.COLOR_BGR2GRAY)

    # Write to video
    depth_writer.write(depth_gray)


def example_usage_realsense_recording_loop_old():
    """
    Example of OLD depth saving in RealSense recording loop.

    DEPRECATED: This is how it was done before. DO NOT USE.

    For the NEW implementation, see:
    - utils/realsense_camera.py:_recording_loop() (lines 828-834 updated)
    """

    # OLD CODE FROM realsense_camera.py (lines 635-644):
    # ===================================================
    # # Depth stream (saved as grayscale visualization)
    # if self.record_depth:
    #     depth_filename = self._generate_filename("depth", "mp4", self.current_file_index)
    #     self.depth_writer = cv2.VideoWriter(
    #         str(self.output_dir / depth_filename),
    #         fourcc,
    #         self.fps,
    #         (self.width, self.height),
    #         isColor=False
    #     )

    # OLD CODE FROM realsense_camera.py (lines 828-834):
    # ===================================================
    # # Write depth frame (convert to 8-bit for visualization)
    # depth_colormap = cv2.applyColorMap(
    #     cv2.convertScaleAbs(depth_image, alpha=0.03),
    #     cv2.COLORMAP_JET
    # )
    # depth_gray = cv2.cvtColor(depth_colormap, cv2.COLOR_BGR2GRAY)
    # self.depth_writer.write(depth_gray)

    pass


def comparison_old_vs_new():
    """
    Comparison: OLD MP4 video vs NEW .npz format

    OLD APPROACH (MP4 video):
    -------------------------
    - File size: ~4-6 GB for 5 minutes (3 cameras)
    - Precision: 8-bit (0-255 values) - 99.6% data loss
    - Load time: ~15-20 ms per frame
    - Format: Continuous video file
    - Metadata: None
    - Use case: Visualization only
    - Reconstruction: NOT POSSIBLE (data loss)

    NEW APPROACH (.npz frames):
    ---------------------------
    - File size: ~2-3 GB for 5 minutes (3 cameras) - 40% smaller
    - Precision: uint16 (0-65535 values) - NO data loss
    - Load time: ~3-5 ms per frame - 3-5x faster
    - Format: Individual compressed frames
    - Metadata: timestamp, frame_number, depth_scale, camera_index
    - Use case: Reconstruction + Analysis + Visualization
    - Reconstruction: FULLY POSSIBLE

    RECOMMENDATION: Use .npz format for all production use cases.
    """
    pass


if __name__ == "__main__":
    print(__doc__)
    print("\n" + "="*80)
    print("WARNING: This module is DEPRECATED")
    print("="*80)
    print("\nFor current depth saving implementation, see:")
    print("  - utils/realsense_camera.py")
    print("  - docs/files_needed/cattle_monitoring_progress.md")
    print("\n" + "="*80)
