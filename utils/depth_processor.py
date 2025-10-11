"""RealSense depth processing and filtering utilities (CPU-only)."""
from typing import Optional, Tuple, List, Literal
from pathlib import Path
import numpy as np
import cv2


class DepthProcessor:
    """
    Advanced depth processing and filtering for RealSense cameras.

    Provides CPU-only depth enhancement including:
    - Temporal filtering (noise reduction across frames)
    - Spatial filtering (edge-preserving smoothing)
    - Hole filling (invalid pixel interpolation)
    - Decimation (resolution reduction)
    - Disparity transforms (improved near-object precision)
    - Threshold filtering (depth range limiting)
    - Statistics and analysis
    - Multi-scale depth maps
    - Colorization and visualization
    """

    def __init__(self):
        """Initialize depth processor with filter objects."""
        try:
            import pyrealsense2 as rs
            self.rs = rs

            # Initialize filter objects
            self.temporal_filter: Optional[rs.temporal_filter] = None
            self.spatial_filter: Optional[rs.spatial_filter] = None
            self.hole_filling_filter: Optional[rs.hole_filling_filter] = None
            self.decimation_filter: Optional[rs.decimation_filter] = None
            self.threshold_filter: Optional[rs.threshold_filter] = None
            self.disparity_to_depth: Optional[rs.disparity_transform] = None
            self.depth_to_disparity: Optional[rs.disparity_transform] = None

            # Filter pipeline
            self.filter_pipeline: List = []

            # Statistics tracking
            self.last_frame_stats: Optional[dict] = None

        except ImportError:
            raise ImportError("pyrealsense2 is required for DepthProcessor")

    # ============================================================================
    # TEMPORAL FILTERING
    # ============================================================================

    def initialize_temporal_filter(
        self,
        smooth_alpha: float = 0.4,
        smooth_delta: float = 20.0,
        persistence_control: int = 3
    ) -> 'rs.temporal_filter':
        """
        Initialize temporal filter for noise reduction across frames.

        Args:
            smooth_alpha: Smoothing strength [0.0-1.0]. 0=no smoothing, 1=max smoothing
            smooth_delta: Outlier threshold. Depth changes > delta are outliers
            persistence_control: Persistence mode (0-8)
                0 = disabled
                1 = valid-in-8-of-8
                2 = valid-in-2-of-last-3
                3 = valid-in-2-of-last-4
                4 = valid-in-2-of-8
                5 = valid-in-1-of-last-2
                6 = valid-in-1-of-last-5
                7 = valid-in-1-of-last-8
                8 = persist-indefinitely

        Returns:
            Configured temporal filter object
        """
        self.temporal_filter = self.rs.temporal_filter()

        # Configure parameters
        self.temporal_filter.set_option(
            self.rs.option.filter_smooth_alpha, smooth_alpha
        )
        self.temporal_filter.set_option(
            self.rs.option.filter_smooth_delta, smooth_delta
        )
        self.temporal_filter.set_option(
            self.rs.option.holes_fill, persistence_control
        )

        return self.temporal_filter

    def apply_temporal_filter(self, depth_frame) -> 'rs.depth_frame':
        """
        Apply temporal filter to depth frame.

        Args:
            depth_frame: RealSense depth frame

        Returns:
            Temporally filtered depth frame
        """
        if self.temporal_filter is None:
            self.initialize_temporal_filter()

        return self.temporal_filter.process(depth_frame)

    # ============================================================================
    # SPATIAL FILTERING
    # ============================================================================

    def initialize_spatial_filter(
        self,
        magnitude: int = 2,
        smooth_alpha: float = 0.5,
        smooth_delta: float = 20.0,
        hole_fill: int = 0
    ) -> 'rs.spatial_filter':
        """
        Initialize spatial edge-preserving filter.

        Args:
            magnitude: Number of filter iterations [1-5]. Higher = stronger smoothing
            smooth_alpha: Edge preservation strength [0.25-1.0]
            smooth_delta: Edge detection threshold. Changes > delta are edges
            hole_fill: Hole filling mode (0=disabled, 1-5=iterations)

        Returns:
            Configured spatial filter object
        """
        self.spatial_filter = self.rs.spatial_filter()

        self.spatial_filter.set_option(
            self.rs.option.filter_magnitude, magnitude
        )
        self.spatial_filter.set_option(
            self.rs.option.filter_smooth_alpha, smooth_alpha
        )
        self.spatial_filter.set_option(
            self.rs.option.filter_smooth_delta, smooth_delta
        )
        self.spatial_filter.set_option(
            self.rs.option.holes_fill, hole_fill
        )

        return self.spatial_filter

    def apply_spatial_filter(self, depth_frame) -> 'rs.depth_frame':
        """
        Apply spatial edge-preserving filter to depth frame.

        Args:
            depth_frame: RealSense depth frame

        Returns:
            Spatially filtered depth frame
        """
        if self.spatial_filter is None:
            self.initialize_spatial_filter()

        return self.spatial_filter.process(depth_frame)

    # ============================================================================
    # HOLE FILLING
    # ============================================================================

    def initialize_hole_filling_filter(
        self,
        mode: Literal['farest', 'nearest'] = 'farest'
    ) -> 'rs.hole_filling_filter':
        """
        Initialize hole filling filter for invalid pixel interpolation.

        Args:
            mode: Filling mode
                'farest' (mode=1): Fill with farthest valid neighbor (conservative)
                'nearest' (mode=2): Fill with nearest valid neighbor (aggressive)

        Returns:
            Configured hole filling filter object
        """
        self.hole_filling_filter = self.rs.hole_filling_filter()

        mode_map = {
            'farest': 1,
            'nearest': 2
        }

        self.hole_filling_filter.set_option(
            self.rs.option.holes_fill, mode_map[mode]
        )

        return self.hole_filling_filter

    def apply_hole_filling_filter(self, depth_frame) -> 'rs.depth_frame':
        """
        Apply hole filling filter to depth frame.

        Args:
            depth_frame: RealSense depth frame

        Returns:
            Hole-filled depth frame
        """
        if self.hole_filling_filter is None:
            self.initialize_hole_filling_filter()

        return self.hole_filling_filter.process(depth_frame)

    # ============================================================================
    # DECIMATION FILTERING
    # ============================================================================

    def initialize_decimation_filter(
        self,
        scale: int = 2
    ) -> 'rs.decimation_filter':
        """
        Initialize decimation filter for resolution reduction.

        Args:
            scale: Downsampling factor [2-8]. Resolution reduced by 1/scale
                scale=2: 1280x720 → 640x360
                scale=4: 1280x720 → 320x180

        Returns:
            Configured decimation filter object
        """
        self.decimation_filter = self.rs.decimation_filter()

        self.decimation_filter.set_option(
            self.rs.option.filter_magnitude, scale
        )

        return self.decimation_filter

    def apply_decimation_filter(self, depth_frame) -> 'rs.depth_frame':
        """
        Apply decimation filter to reduce depth resolution.

        Args:
            depth_frame: RealSense depth frame

        Returns:
            Decimated depth frame (lower resolution)
        """
        if self.decimation_filter is None:
            self.initialize_decimation_filter()

        return self.decimation_filter.process(depth_frame)

    # ============================================================================
    # DISPARITY TRANSFORM
    # ============================================================================

    def initialize_disparity_transforms(self):
        """Initialize depth-to-disparity and disparity-to-depth transforms."""
        self.depth_to_disparity = self.rs.disparity_transform(True)
        self.disparity_to_depth = self.rs.disparity_transform(False)

    def apply_depth_to_disparity(self, depth_frame) -> 'rs.disparity_frame':
        """
        Convert depth to disparity for better near-object precision.

        Disparity = baseline × focal_length / depth

        Args:
            depth_frame: RealSense depth frame

        Returns:
            Disparity frame
        """
        if self.depth_to_disparity is None:
            self.initialize_disparity_transforms()

        return self.depth_to_disparity.process(depth_frame)

    def apply_disparity_to_depth(self, disparity_frame) -> 'rs.depth_frame':
        """
        Convert disparity back to depth.

        Args:
            disparity_frame: RealSense disparity frame

        Returns:
            Depth frame
        """
        if self.disparity_to_depth is None:
            self.initialize_disparity_transforms()

        return self.disparity_to_depth.process(disparity_frame)

    # ============================================================================
    # THRESHOLD FILTERING
    # ============================================================================

    def initialize_threshold_filter(
        self,
        min_distance: float = 0.1,
        max_distance: float = 10.0
    ) -> 'rs.threshold_filter':
        """
        Initialize threshold filter for depth range limiting.

        Args:
            min_distance: Minimum valid depth in meters
            max_distance: Maximum valid depth in meters

        Returns:
            Configured threshold filter object
        """
        self.threshold_filter = self.rs.threshold_filter()

        self.threshold_filter.set_option(
            self.rs.option.min_distance, min_distance
        )
        self.threshold_filter.set_option(
            self.rs.option.max_distance, max_distance
        )

        return self.threshold_filter

    def apply_threshold_filter(self, depth_frame) -> 'rs.depth_frame':
        """
        Apply threshold filter to limit depth range.

        Args:
            depth_frame: RealSense depth frame

        Returns:
            Thresholded depth frame
        """
        if self.threshold_filter is None:
            self.initialize_threshold_filter()

        return self.threshold_filter.process(depth_frame)

    # ============================================================================
    # FILTER PIPELINE
    # ============================================================================

    def build_filter_pipeline(
        self,
        use_decimation: bool = False,
        decimation_scale: int = 2,
        use_disparity: bool = True,
        use_spatial: bool = True,
        spatial_iterations: int = 2,
        use_temporal: bool = True,
        use_hole_filling: bool = True,
        hole_fill_mode: Literal['farest', 'nearest'] = 'farest',
        use_threshold: bool = True,
        min_distance: float = 0.1,
        max_distance: float = 10.0
    ):
        """
        Build complete filtering pipeline with optimal ordering.

        Recommended order for quality and performance:
        1. Decimation (optional, for performance)
        2. Depth→Disparity (for near-object precision)
        3. Spatial filter (edge-preserving smoothing)
        4. Temporal filter (noise reduction)
        5. Disparity→Depth (convert back)
        6. Hole filling (fill invalid pixels)
        7. Threshold (limit depth range)

        Args:
            use_decimation: Enable decimation filter
            decimation_scale: Decimation scale factor
            use_disparity: Enable disparity transform
            use_spatial: Enable spatial filter
            spatial_iterations: Spatial filter iterations
            use_temporal: Enable temporal filter
            use_hole_filling: Enable hole filling
            hole_fill_mode: Hole filling mode ('farest' or 'nearest')
            use_threshold: Enable threshold filter
            min_distance: Minimum depth threshold
            max_distance: Maximum depth threshold
        """
        self.filter_pipeline = []

        # 1. Decimation (optional)
        if use_decimation:
            self.initialize_decimation_filter(decimation_scale)
            self.filter_pipeline.append(('decimation', self.decimation_filter))

        # 2. Depth to Disparity
        if use_disparity:
            self.initialize_disparity_transforms()
            self.filter_pipeline.append(('depth_to_disparity', self.depth_to_disparity))

        # 3. Spatial filter
        if use_spatial:
            self.initialize_spatial_filter(magnitude=spatial_iterations)
            self.filter_pipeline.append(('spatial', self.spatial_filter))

        # 4. Temporal filter
        if use_temporal:
            self.initialize_temporal_filter()
            self.filter_pipeline.append(('temporal', self.temporal_filter))

        # 5. Disparity to Depth
        if use_disparity:
            self.filter_pipeline.append(('disparity_to_depth', self.disparity_to_depth))

        # 6. Hole filling
        if use_hole_filling:
            self.initialize_hole_filling_filter(mode=hole_fill_mode)
            self.filter_pipeline.append(('hole_filling', self.hole_filling_filter))

        # 7. Threshold
        if use_threshold:
            self.initialize_threshold_filter(min_distance, max_distance)
            self.filter_pipeline.append(('threshold', self.threshold_filter))

    def apply_filter_pipeline(self, depth_frame) -> 'rs.depth_frame':
        """
        Apply complete filter pipeline to depth frame.

        Args:
            depth_frame: Raw RealSense depth frame

        Returns:
            Fully filtered depth frame
        """
        filtered_frame = depth_frame

        for name, filter_obj in self.filter_pipeline:
            filtered_frame = filter_obj.process(filtered_frame)

        return filtered_frame

    # ============================================================================
    # DEPTH EXPORT
    # ============================================================================

    def export_depth_as_png(
        self,
        depth_frame,
        output_path: Path,
        depth_scale: float = 0.001
    ):
        """
        Export depth frame as 16-bit PNG (lossless).

        Args:
            depth_frame: RealSense depth frame
            output_path: Output file path (.png)
            depth_scale: Depth scale factor (default 0.001 = mm to m)
        """
        depth_image = np.asanyarray(depth_frame.get_data())

        # Save as 16-bit PNG (lossless)
        cv2.imwrite(str(output_path), depth_image)

    def export_depth_as_numpy(
        self,
        depth_frame,
        output_path: Path,
        dtype: str = 'uint16'
    ):
        """
        Export depth frame as NumPy array (.npy).

        Args:
            depth_frame: RealSense depth frame
            output_path: Output file path (.npy)
            dtype: Data type ('uint16' or 'float32')
        """
        depth_image = np.asanyarray(depth_frame.get_data())

        if dtype == 'float32':
            depth_image = depth_image.astype(np.float32)

        np.save(output_path, depth_image)

    # ============================================================================
    # DEPTH STATISTICS & ANALYSIS
    # ============================================================================

    def calculate_depth_statistics(
        self,
        depth_frame,
        depth_scale: float = 0.001
    ) -> dict:
        """
        Calculate comprehensive depth statistics.

        Args:
            depth_frame: RealSense depth frame
            depth_scale: Depth scale factor (default 0.001 = mm to m)

        Returns:
            Dictionary with statistics:
                - min_depth: Minimum depth in meters
                - max_depth: Maximum depth in meters
                - mean_depth: Mean depth in meters
                - median_depth: Median depth in meters
                - std_depth: Standard deviation in meters
                - valid_pixels: Number of valid depth pixels
                - invalid_pixels: Number of invalid (zero) pixels
                - validity_percentage: Percentage of valid pixels
        """
        depth_image = np.asanyarray(depth_frame.get_data())

        # Get valid depth pixels (non-zero)
        valid_depth = depth_image[depth_image > 0]

        if len(valid_depth) == 0:
            return {
                'min_depth': 0.0,
                'max_depth': 0.0,
                'mean_depth': 0.0,
                'median_depth': 0.0,
                'std_depth': 0.0,
                'valid_pixels': 0,
                'invalid_pixels': depth_image.size,
                'validity_percentage': 0.0
            }

        # Convert to meters
        valid_depth_m = valid_depth * depth_scale

        stats = {
            'min_depth': float(np.min(valid_depth_m)),
            'max_depth': float(np.max(valid_depth_m)),
            'mean_depth': float(np.mean(valid_depth_m)),
            'median_depth': float(np.median(valid_depth_m)),
            'std_depth': float(np.std(valid_depth_m)),
            'valid_pixels': int(len(valid_depth)),
            'invalid_pixels': int(depth_image.size - len(valid_depth)),
            'validity_percentage': float(len(valid_depth) / depth_image.size * 100)
        }

        self.last_frame_stats = stats
        return stats

    def generate_depth_histogram(
        self,
        depth_frame,
        depth_scale: float = 0.001,
        num_bins: int = 100,
        depth_range: Optional[Tuple[float, float]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate depth distribution histogram.

        Args:
            depth_frame: RealSense depth frame
            depth_scale: Depth scale factor
            num_bins: Number of histogram bins
            depth_range: Depth range (min, max) in meters. None = auto

        Returns:
            Tuple of (histogram, bin_edges)
        """
        depth_image = np.asanyarray(depth_frame.get_data())
        valid_depth = depth_image[depth_image > 0] * depth_scale

        if depth_range is None:
            depth_range = (valid_depth.min(), valid_depth.max())

        histogram, bin_edges = np.histogram(
            valid_depth,
            bins=num_bins,
            range=depth_range
        )

        return histogram, bin_edges

    def create_depth_validity_map(self, depth_frame) -> np.ndarray:
        """
        Create binary validity map (1=valid, 0=invalid).

        Args:
            depth_frame: RealSense depth frame

        Returns:
            Binary validity map (uint8)
        """
        depth_image = np.asanyarray(depth_frame.get_data())
        validity_map = (depth_image > 0).astype(np.uint8)
        return validity_map

    # ============================================================================
    # DEPTH COLORIZATION & VISUALIZATION
    # ============================================================================

    def apply_depth_colorization(
        self,
        depth_frame,
        colormap: int = cv2.COLORMAP_JET,
        min_distance: float = 0.0,
        max_distance: float = 10.0
    ) -> np.ndarray:
        """
        Apply colorization to depth frame for visualization.

        Args:
            depth_frame: RealSense depth frame
            colormap: OpenCV colormap
                cv2.COLORMAP_JET: Red (near) → Blue (far)
                cv2.COLORMAP_RAINBOW: Rainbow colors
                cv2.COLORMAP_BONE: Grayscale
                cv2.COLORMAP_HOT: Hot colors
            min_distance: Minimum distance for color mapping (meters)
            max_distance: Maximum distance for color mapping (meters)

        Returns:
            Colorized depth image (BGR, uint8)
        """
        depth_image = np.asanyarray(depth_frame.get_data())

        # Normalize depth to 0-255 range
        depth_normalized = cv2.normalize(
            depth_image,
            None,
            0, 255,
            cv2.NORM_MINMAX,
            dtype=cv2.CV_8U
        )

        # Apply colormap
        depth_colorized = cv2.applyColorMap(depth_normalized, colormap)

        return depth_colorized

    # ============================================================================
    # MULTI-SCALE DEPTH MAPS
    # ============================================================================

    def generate_multiscale_depth_pyramid(
        self,
        depth_frame,
        num_levels: int = 4
    ) -> List[np.ndarray]:
        """
        Generate multi-scale depth pyramid.

        Creates depth maps at multiple resolutions:
        Level 0: Full resolution
        Level 1: 1/2 resolution
        Level 2: 1/4 resolution
        Level 3: 1/8 resolution

        Args:
            depth_frame: RealSense depth frame
            num_levels: Number of pyramid levels

        Returns:
            List of depth images at different scales
        """
        depth_image = np.asanyarray(depth_frame.get_data())
        pyramid = [depth_image]

        current = depth_image
        for i in range(1, num_levels):
            # Downsample by factor of 2
            downsampled = cv2.resize(
                current,
                (current.shape[1] // 2, current.shape[0] // 2),
                interpolation=cv2.INTER_NEAREST
            )
            pyramid.append(downsampled)
            current = downsampled

        return pyramid

    # ============================================================================
    # UTILITY METHODS
    # ============================================================================

    def get_pipeline_description(self) -> List[str]:
        """
        Get description of current filter pipeline.

        Returns:
            List of filter names in pipeline order
        """
        return [name for name, _ in self.filter_pipeline]

    def clear_pipeline(self):
        """Clear the filter pipeline."""
        self.filter_pipeline = []

    def reset_all_filters(self):
        """Reset all filter objects to None."""
        self.temporal_filter = None
        self.spatial_filter = None
        self.hole_filling_filter = None
        self.decimation_filter = None
        self.threshold_filter = None
        self.disparity_to_depth = None
        self.depth_to_disparity = None
        self.filter_pipeline = []
