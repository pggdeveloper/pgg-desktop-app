"""
Abstract Base Class for Animal Scene Understanding

This module provides an abstract interface for animal-specific scene understanding
implementations. Enables future extensibility for different animal types and
agricultural environments (cows, pigs, sheep, etc.) while maintaining consistent API.

Author: PGG Desktop App Team
Date: 2025-10-10
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Literal
from dataclasses import dataclass
from datetime import datetime


@dataclass
class PlaneModel:
    """Plane model with equation ax + by + cz + d = 0."""
    coefficients: np.ndarray  # [a, b, c, d]
    inliers: np.ndarray  # Inlier point indices
    inlier_count: int
    plane_type: Literal['floor', 'wall', 'ceiling', 'table', 'unknown']
    confidence: float  # 0-1


@dataclass
class ObjectDetection:
    """Detected object in scene."""
    object_type: Literal['feed_trough', 'water_source', 'equipment', 'obstacle', 'unknown']
    position: np.ndarray  # 3D position (x, y, z)
    dimensions: Optional[np.ndarray]  # (width, height, depth) if available
    confidence: float
    bounding_box: Optional[Tuple[int, int, int, int]]  # (x_min, y_min, x_max, y_max)


@dataclass
class LightingAnalysis:
    """Lighting condition analysis."""
    average_brightness: float  # 0-255
    std_brightness: float  # Standard deviation
    histogram: np.ndarray  # Brightness histogram
    uniformity_score: float  # 0-1 (1 = perfectly uniform)
    lighting_quality: Literal['excellent', 'good', 'fair', 'poor']
    timestamp: datetime


@dataclass
class ShadowDetection:
    """Shadow detection result."""
    shadow_mask: np.ndarray  # Binary mask of shadows
    shadow_percentage: float  # Percentage of frame with shadows
    shadow_regions: List[np.ndarray]  # List of shadow contours
    lighting_direction: Optional[np.ndarray]  # Inferred light direction vector
    timestamp: datetime


@dataclass
class WetnessDetection:
    """Wetness/moisture detection result."""
    wetness_mask: np.ndarray  # Binary mask of wet regions
    wetness_percentage: float  # Percentage of area that's wet
    reflectance_map: np.ndarray  # Reflectance intensity map
    wet_regions: List[np.ndarray]  # List of wet region contours
    timestamp: datetime


@dataclass
class SceneReport:
    """Comprehensive scene understanding report."""
    environment_type: Literal['feedlot_closed', 'feedlot_outdoor', 'pasture_natural', 'pasture_fodder']
    timestamp: datetime
    planes: List[PlaneModel]  # Detected planes
    objects: List[ObjectDetection]  # Detected objects
    lighting: LightingAnalysis  # Lighting analysis
    shadows: Optional[ShadowDetection]  # Shadow detection
    wetness: Optional[WetnessDetection]  # Wetness detection
    frame_number: int


class AnimalSceneUnderstanding(ABC):
    """
    Abstract base class for animal scene understanding.

    Provides consistent interface for different animal types and agricultural
    environments while allowing species-specific implementations for plane
    detection, object recognition, and environmental analysis.
    """

    def __init__(
        self,
        animal_type: str,
        environment_type: Literal['feedlot_closed', 'feedlot_outdoor', 'pasture_natural', 'pasture_fodder'] = 'feedlot_outdoor'
    ):
        """
        Initialize animal scene understanding.

        Args:
            animal_type: Type of animal ('cow', 'pig', 'sheep', etc.)
            environment_type: Type of agricultural environment
        """
        self.animal_type = animal_type
        self.environment_type = environment_type

        # Scene history
        self.scene_reports: List[SceneReport] = []
        self.detected_planes: List[PlaneModel] = []
        self.detected_objects: Dict[str, List[ObjectDetection]] = {}

    # ========================================================================
    # PLANE DETECTION (Abstract Methods)
    # ========================================================================

    @abstractmethod
    def detect_floor_plane(
        self,
        points: np.ndarray,
        distance_threshold: float = 0.02
    ) -> Optional[PlaneModel]:
        """
        Detect floor plane using RANSAC.

        Args:
            points: Point cloud (N, 3)
            distance_threshold: RANSAC distance threshold

        Returns:
            PlaneModel for floor or None
        """
        pass

    @abstractmethod
    def detect_wall_planes(
        self,
        points: np.ndarray,
        max_planes: int = 4
    ) -> List[PlaneModel]:
        """
        Detect vertical wall planes.

        Args:
            points: Point cloud (N, 3)
            max_planes: Maximum number of wall planes to detect

        Returns:
            List of wall PlaneModels
        """
        pass

    @abstractmethod
    def detect_ceiling_plane(
        self,
        points: np.ndarray
    ) -> Optional[PlaneModel]:
        """
        Detect ceiling plane (horizontal plane above).

        Args:
            points: Point cloud (N, 3)

        Returns:
            PlaneModel for ceiling or None
        """
        pass

    @abstractmethod
    def detect_table_or_platform(
        self,
        points: np.ndarray,
        floor_plane: Optional[PlaneModel] = None
    ) -> List[PlaneModel]:
        """
        Detect elevated horizontal planes (tables/platforms).

        Args:
            points: Point cloud (N, 3)
            floor_plane: Optional floor plane for reference

        Returns:
            List of elevated plane PlaneModels
        """
        pass

    @abstractmethod
    def multi_plane_segmentation(
        self,
        points: np.ndarray,
        max_planes: int = 10
    ) -> List[PlaneModel]:
        """
        Perform multi-plane RANSAC segmentation.

        Args:
            points: Point cloud (N, 3)
            max_planes: Maximum number of planes to detect

        Returns:
            List of all detected PlaneModels
        """
        pass

    # ========================================================================
    # OBJECT DETECTION (Abstract Methods)
    # ========================================================================

    @abstractmethod
    def detect_feed_trough(
        self,
        points: np.ndarray,
        rgb_image: Optional[np.ndarray] = None
    ) -> List[ObjectDetection]:
        """
        Detect feed troughs using geometric features.

        Args:
            points: Point cloud (N, 3)
            rgb_image: Optional RGB image

        Returns:
            List of feed trough detections
        """
        pass

    @abstractmethod
    def identify_water_source(
        self,
        points: np.ndarray,
        rgb_image: Optional[np.ndarray] = None
    ) -> List[ObjectDetection]:
        """
        Identify water sources (bowls/tanks).

        Args:
            points: Point cloud (N, 3)
            rgb_image: Optional RGB image

        Returns:
            List of water source detections
        """
        pass

    @abstractmethod
    def recognize_equipment(
        self,
        rgb_image: np.ndarray,
        template_features: Optional[Dict[str, Any]] = None
    ) -> List[ObjectDetection]:
        """
        Recognize equipment using feature-based matching.

        Args:
            rgb_image: RGB image
            template_features: Optional pre-computed template features

        Returns:
            List of equipment detections
        """
        pass

    @abstractmethod
    def detect_obstacles(
        self,
        depth_map: np.ndarray,
        floor_plane: Optional[PlaneModel] = None,
        height_threshold: float = 0.1
    ) -> List[ObjectDetection]:
        """
        Detect obstacles protruding from ground.

        Args:
            depth_map: Depth map
            floor_plane: Optional floor plane for reference
            height_threshold: Minimum height above floor

        Returns:
            List of obstacle detections
        """
        pass

    # ========================================================================
    # ENVIRONMENTAL ANALYSIS (Abstract Methods)
    # ========================================================================

    @abstractmethod
    def analyze_lighting(
        self,
        rgb_image: np.ndarray
    ) -> LightingAnalysis:
        """
        Analyze lighting conditions.

        Args:
            rgb_image: RGB image

        Returns:
            LightingAnalysis with brightness metrics
        """
        pass

    @abstractmethod
    def detect_shadows(
        self,
        rgb_image: np.ndarray
    ) -> ShadowDetection:
        """
        Detect shadows in the scene.

        Args:
            rgb_image: RGB image

        Returns:
            ShadowDetection with shadow mask and regions
        """
        pass

    @abstractmethod
    def detect_wetness(
        self,
        rgb_image: np.ndarray,
        depth_map: Optional[np.ndarray] = None
    ) -> WetnessDetection:
        """
        Detect wetness/moisture using reflectance analysis.

        Args:
            rgb_image: RGB image
            depth_map: Optional depth map

        Returns:
            WetnessDetection with wet regions
        """
        pass

    # ========================================================================
    # REPORT GENERATION (Abstract Methods)
    # ========================================================================

    @abstractmethod
    def generate_scene_report(
        self,
        points: np.ndarray,
        rgb_image: np.ndarray,
        depth_map: np.ndarray,
        frame_number: int,
        timestamp: Optional[datetime] = None
    ) -> SceneReport:
        """
        Generate comprehensive scene understanding report.

        Args:
            points: Point cloud (N, 3)
            rgb_image: RGB image
            depth_map: Depth map
            frame_number: Frame number
            timestamp: Optional timestamp

        Returns:
            SceneReport with all scene understanding results
        """
        pass

    @abstractmethod
    def save_scene_report_to_csv(
        self,
        report: SceneReport,
        csv_path: str
    ):
        """
        Save scene report to CSV file.

        Args:
            report: SceneReport to save
            csv_path: Path to CSV file
        """
        pass
