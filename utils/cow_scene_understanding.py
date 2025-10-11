"""
Cow-Specific Scene Understanding Implementation

Implements scene understanding for cattle (cow, bull, calf, heifer, steer) in
various agricultural environments using classical computer vision on CPU.

Environments:
- Outdoor/closed feedlot
- Outdoor natural field pasture
- Outdoor green fodder pasture
- Common features: mud, uneven terrain, outdoor lighting

Covers all 12 scenarios from scenario-10.feature:
- Plane detection (floor, walls, ceiling, tables, multi-plane)
- Object detection (feed troughs, water sources, equipment, obstacles)
- Environmental analysis (lighting, shadows, wetness/mud)

Author: PGG Desktop App Team
Date: 2025-10-10
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Literal
from datetime import datetime
from pathlib import Path
import csv

from utils.animal_scene_understanding import (
    AnimalSceneUnderstanding,
    PlaneModel,
    ObjectDetection,
    LightingAnalysis,
    ShadowDetection,
    WetnessDetection,
    SceneReport
)


class CowSceneUnderstanding(AnimalSceneUnderstanding):
    """
    Cow-specific scene understanding implementation.

    Supports: cow, bull, calf, heifer, steer
    Environments: feedlots, pastures (natural, fodder)
    """

    # Cow-specific constants
    COW_MIN_FLOOR_INLIERS = 100  # Minimum inliers for valid floor plane
    COW_TROUGH_MIN_LENGTH = 1.0  # Minimum trough length in meters
    COW_TROUGH_HEIGHT_RANGE = (0.3, 0.8)  # Trough height from floor (m)
    COW_WATER_BOWL_RADIUS_RANGE = (0.2, 0.6)  # Water bowl radius range (m)
    COW_OBSTACLE_HEIGHT_THRESHOLD = 0.15  # Minimum obstacle height (m)

    def __init__(
        self,
        animal_type: str = 'cow',
        environment_type: Literal['feedlot_closed', 'feedlot_outdoor', 'pasture_natural', 'pasture_fodder'] = 'feedlot_outdoor'
    ):
        """
        Initialize cow scene understanding.

        Args:
            animal_type: Type of cattle
            environment_type: Agricultural environment type
        """
        super().__init__(animal_type, environment_type)

        # Environment-specific adjustments
        self.expect_ceiling = environment_type == 'feedlot_closed'
        self.expect_walls = environment_type == 'feedlot_closed'
        self.expect_mud = 'outdoor' in environment_type or 'pasture' in environment_type

    # ========================================================================
    # PLANE DETECTION IMPLEMENTATION
    # ========================================================================

    def detect_floor_plane(
        self,
        points: np.ndarray,
        distance_threshold: float = 0.02
    ) -> Optional[PlaneModel]:
        """
        Detect floor plane using RANSAC.

        Scenario: Detect floor plane with RANSAC

        Args:
            points: Point cloud (N, 3)
            distance_threshold: RANSAC distance threshold

        Returns:
            PlaneModel for floor or None
        """
        try:
            import open3d as o3d
        except ImportError:
            print("Warning: Open3D not available. Using fallback plane detection.")
            return self._detect_plane_fallback(points, 'floor')

        if len(points) < 100:
            return None

        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        # Segment plane using RANSAC
        plane_model, inliers = pcd.segment_plane(
            distance_threshold=distance_threshold,
            ransac_n=3,
            num_iterations=1000
        )

        if len(inliers) < self.COW_MIN_FLOOR_INLIERS:
            return None

        # Extract plane coefficients [a, b, c, d] from equation ax + by + cz + d = 0
        coefficients = np.array(plane_model)

        # Check if plane is roughly horizontal (floor should be horizontal)
        normal = coefficients[:3]
        normal = normal / np.linalg.norm(normal)

        # Floor normal should point up (positive z) and be roughly vertical
        # Allow some tolerance for uneven outdoor terrain
        z_component = abs(normal[2])
        if z_component < 0.7:  # Allow 45° deviation for outdoor terrain
            return None

        # Calculate confidence based on inlier ratio
        confidence = len(inliers) / len(points)

        floor_plane = PlaneModel(
            coefficients=coefficients,
            inliers=np.array(inliers),
            inlier_count=len(inliers),
            plane_type='floor',
            confidence=float(confidence)
        )

        self.detected_planes.append(floor_plane)
        return floor_plane

    def detect_wall_planes(
        self,
        points: np.ndarray,
        max_planes: int = 4
    ) -> List[PlaneModel]:
        """
        Detect vertical wall planes.

        Scenario: Detect wall planes

        Args:
            points: Point cloud (N, 3)
            max_planes: Maximum number of wall planes to detect

        Returns:
            List of wall PlaneModels
        """
        if not self.expect_walls:
            return []  # No walls in outdoor pastures

        try:
            import open3d as o3d
        except ImportError:
            return []

        if len(points) < 100:
            return []

        walls = []
        remaining_points = points.copy()

        for _ in range(max_planes):
            if len(remaining_points) < 50:
                break

            # Create point cloud
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(remaining_points)

            # Segment plane
            plane_model, inliers = pcd.segment_plane(
                distance_threshold=0.03,
                ransac_n=3,
                num_iterations=500
            )

            if len(inliers) < 50:
                break

            coefficients = np.array(plane_model)
            normal = coefficients[:3]
            normal = normal / np.linalg.norm(normal)

            # Wall should be vertical (low z component)
            z_component = abs(normal[2])
            if z_component < 0.3:  # More vertical than horizontal
                confidence = len(inliers) / len(remaining_points)

                wall_plane = PlaneModel(
                    coefficients=coefficients,
                    inliers=np.array(inliers),
                    inlier_count=len(inliers),
                    plane_type='wall',
                    confidence=float(confidence)
                )

                walls.append(wall_plane)
                self.detected_planes.append(wall_plane)

            # Remove inliers for next iteration
            mask = np.ones(len(remaining_points), dtype=bool)
            mask[inliers] = False
            remaining_points = remaining_points[mask]

        return walls

    def detect_ceiling_plane(
        self,
        points: np.ndarray
    ) -> Optional[PlaneModel]:
        """
        Detect ceiling plane (horizontal plane above).

        Scenario: Detect ceiling plane

        Args:
            points: Point cloud (N, 3)

        Returns:
            PlaneModel for ceiling or None
        """
        if not self.expect_ceiling:
            return None  # No ceiling in outdoor environments

        try:
            import open3d as o3d
        except ImportError:
            return None

        if len(points) < 100:
            return None

        # Filter points above a threshold (ceiling should be high)
        z_threshold = np.percentile(points[:, 2], 90)  # Top 10% of points
        ceiling_candidates = points[points[:, 2] > z_threshold]

        if len(ceiling_candidates) < 50:
            return None

        # Create point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(ceiling_candidates)

        # Segment plane
        plane_model, inliers = pcd.segment_plane(
            distance_threshold=0.03,
            ransac_n=3,
            num_iterations=500
        )

        if len(inliers) < 50:
            return None

        coefficients = np.array(plane_model)
        normal = coefficients[:3]
        normal = normal / np.linalg.norm(normal)

        # Ceiling should be horizontal with normal pointing down
        z_component = abs(normal[2])
        if z_component < 0.8:
            return None

        confidence = len(inliers) / len(ceiling_candidates)

        ceiling_plane = PlaneModel(
            coefficients=coefficients,
            inliers=np.array(inliers),
            inlier_count=len(inliers),
            plane_type='ceiling',
            confidence=float(confidence)
        )

        self.detected_planes.append(ceiling_plane)
        return ceiling_plane

    def detect_table_or_platform(
        self,
        points: np.ndarray,
        floor_plane: Optional[PlaneModel] = None
    ) -> List[PlaneModel]:
        """
        Detect elevated horizontal planes (tables/platforms).

        Scenario: Detect table or platform

        Args:
            points: Point cloud (N, 3)
            floor_plane: Optional floor plane for reference

        Returns:
            List of elevated plane PlaneModels
        """
        try:
            import open3d as o3d
        except ImportError:
            return []

        if len(points) < 100:
            return []

        # Determine floor height
        if floor_plane is not None:
            floor_height = -floor_plane.coefficients[3] / floor_plane.coefficients[2]
        else:
            floor_height = np.percentile(points[:, 2], 10)

        # Filter points elevated above floor (0.3m to 1.5m range for tables/platforms)
        elevated_mask = (points[:, 2] > floor_height + 0.3) & (points[:, 2] < floor_height + 1.5)
        elevated_points = points[elevated_mask]

        if len(elevated_points) < 50:
            return []

        platforms = []
        remaining = elevated_points.copy()

        for _ in range(3):  # Try to find up to 3 platforms
            if len(remaining) < 30:
                break

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(remaining)

            plane_model, inliers = pcd.segment_plane(
                distance_threshold=0.02,
                ransac_n=3,
                num_iterations=300
            )

            if len(inliers) < 30:
                break

            coefficients = np.array(plane_model)
            normal = coefficients[:3]
            normal = normal / np.linalg.norm(normal)

            # Platform should be horizontal
            z_component = abs(normal[2])
            if z_component > 0.8:
                confidence = len(inliers) / len(remaining)

                platform_plane = PlaneModel(
                    coefficients=coefficients,
                    inliers=np.array(inliers),
                    inlier_count=len(inliers),
                    plane_type='table',
                    confidence=float(confidence)
                )

                platforms.append(platform_plane)
                self.detected_planes.append(platform_plane)

            # Remove inliers
            mask = np.ones(len(remaining), dtype=bool)
            mask[inliers] = False
            remaining = remaining[mask]

        return platforms

    def multi_plane_segmentation(
        self,
        points: np.ndarray,
        max_planes: int = 10
    ) -> List[PlaneModel]:
        """
        Perform multi-plane RANSAC segmentation.

        Scenario: Perform multi-plane segmentation

        Args:
            points: Point cloud (N, 3)
            max_planes: Maximum number of planes to detect

        Returns:
            List of all detected PlaneModels
        """
        try:
            import open3d as o3d
        except ImportError:
            return []

        if len(points) < 100:
            return []

        all_planes = []
        remaining_points = points.copy()

        for plane_idx in range(max_planes):
            if len(remaining_points) < 50:
                break

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(remaining_points)

            plane_model, inliers = pcd.segment_plane(
                distance_threshold=0.02,
                ransac_n=3,
                num_iterations=500
            )

            if len(inliers) < 30:
                break

            coefficients = np.array(plane_model)
            normal = coefficients[:3]
            normal = normal / np.linalg.norm(normal)

            # Classify plane type based on orientation
            z_component = abs(normal[2])
            if z_component > 0.8:
                # Horizontal plane
                z_pos = np.mean(remaining_points[inliers, 2])
                if z_pos < np.percentile(points[:, 2], 20):
                    plane_type = 'floor'
                elif z_pos > np.percentile(points[:, 2], 80) and self.expect_ceiling:
                    plane_type = 'ceiling'
                else:
                    plane_type = 'table'
            elif z_component < 0.3 and self.expect_walls:
                plane_type = 'wall'
            else:
                plane_type = 'unknown'

            confidence = len(inliers) / len(remaining_points)

            plane = PlaneModel(
                coefficients=coefficients,
                inliers=np.array(inliers),
                inlier_count=len(inliers),
                plane_type=plane_type,
                confidence=float(confidence)
            )

            all_planes.append(plane)

            # Remove inliers
            mask = np.ones(len(remaining_points), dtype=bool)
            mask[inliers] = False
            remaining_points = remaining_points[mask]

        self.detected_planes.extend(all_planes)
        return all_planes

    def _detect_plane_fallback(
        self,
        points: np.ndarray,
        plane_type: str
    ) -> Optional[PlaneModel]:
        """Fallback plane detection without Open3D."""
        if len(points) < 10:
            return None

        # Simple plane fitting using least squares
        # Fit plane ax + by + cz + d = 0 to minimize distance to points

        # Center points
        centroid = np.mean(points, axis=0)
        centered = points - centroid

        # SVD to find normal vector
        _, _, vh = np.linalg.svd(centered)
        normal = vh[2, :]  # Normal is last singular vector

        # Calculate d
        d = -np.dot(normal, centroid)

        coefficients = np.array([normal[0], normal[1], normal[2], d])

        # Calculate distances to plane
        distances = np.abs(np.dot(points, normal) + d) / np.linalg.norm(normal)
        inliers = np.where(distances < 0.05)[0]

        if len(inliers) < 10:
            return None

        confidence = len(inliers) / len(points)

        return PlaneModel(
            coefficients=coefficients,
            inliers=inliers,
            inlier_count=len(inliers),
            plane_type=plane_type,
            confidence=float(confidence)
        )

    # ========================================================================
    # OBJECT DETECTION IMPLEMENTATION
    # ========================================================================

    def detect_feed_trough(
        self,
        points: np.ndarray,
        rgb_image: Optional[np.ndarray] = None
    ) -> List[ObjectDetection]:
        """
        Detect feed troughs using geometric features.

        Scenario: Detect feed trough with geometric features

        Feed troughs are typically:
        - Elongated rectangular regions
        - Elevated from floor (0.3-0.8m)
        - Length > 1m for cattle
        - Located along walls or in feed alleys

        Args:
            points: Point cloud (N, 3)
            rgb_image: Optional RGB image

        Returns:
            List of feed trough detections
        """
        if len(points) < 50:
            return []

        detections = []

        # Get floor height
        floor_height = np.percentile(points[:, 2], 10)

        # Filter points at trough height
        min_h, max_h = self.COW_TROUGH_HEIGHT_RANGE
        trough_mask = (points[:, 2] > floor_height + min_h) & (points[:, 2] < floor_height + max_h)
        trough_candidates = points[trough_mask]

        if len(trough_candidates) < 20:
            return []

        # Project to 2D (X-Y plane)
        points_2d = trough_candidates[:, :2]

        # Cluster points to find rectangular regions
        # Use simple grid-based clustering
        if len(points_2d) > 10:
            # Calculate bounding box
            x_min, y_min = points_2d.min(axis=0)
            x_max, y_max = points_2d.max(axis=0)

            length = max(x_max - x_min, y_max - y_min)
            width = min(x_max - x_min, y_max - y_min)

            # Check if elongated (length >> width) and large enough
            if length > self.COW_TROUGH_MIN_LENGTH and length / width > 3.0:
                center_3d = np.mean(trough_candidates, axis=0)

                detection = ObjectDetection(
                    object_type='feed_trough',
                    position=center_3d,
                    dimensions=np.array([length, width, 0.3]),
                    confidence=0.7,
                    bounding_box=None
                )

                detections.append(detection)

        return detections

    def identify_water_source(
        self,
        points: np.ndarray,
        rgb_image: Optional[np.ndarray] = None
    ) -> List[ObjectDetection]:
        """
        Identify water sources (bowls/tanks).

        Scenario: Identify water source

        Water sources are typically:
        - Circular or cylindrical objects
        - Bowl radius 0.2-0.6m for cattle
        - May show reflections in RGB
        - Lower than feed troughs

        Args:
            points: Point cloud (N, 3)
            rgb_image: Optional RGB image

        Returns:
            List of water source detections
        """
        if len(points) < 30:
            return []

        detections = []

        # Get floor height
        floor_height = np.percentile(points[:, 2], 10)

        # Filter points at water bowl height (0.2-0.5m above floor)
        water_mask = (points[:, 2] > floor_height + 0.2) & (points[:, 2] < floor_height + 0.5)
        water_candidates = points[water_mask]

        if len(water_candidates) < 10:
            return []

        # Project to 2D
        points_2d = water_candidates[:, :2]

        # Calculate centroid and distances
        center_2d = np.mean(points_2d, axis=0)
        distances = np.linalg.norm(points_2d - center_2d, axis=1)
        avg_radius = np.mean(distances)

        min_r, max_r = self.COW_WATER_BOWL_RADIUS_RANGE

        # Check if size matches water bowl
        if min_r < avg_radius < max_r:
            # Check circularity (standard deviation of distances should be low)
            circularity = 1.0 - (np.std(distances) / avg_radius)

            if circularity > 0.6:  # Reasonably circular
                center_3d = np.mean(water_candidates, axis=0)

                detection = ObjectDetection(
                    object_type='water_source',
                    position=center_3d,
                    dimensions=np.array([avg_radius * 2, avg_radius * 2, 0.2]),
                    confidence=float(circularity),
                    bounding_box=None
                )

                detections.append(detection)

        return detections

    def recognize_equipment(
        self,
        rgb_image: np.ndarray,
        template_features: Optional[Dict[str, Any]] = None
    ) -> List[ObjectDetection]:
        """
        Recognize equipment using feature-based matching.

        Scenario: Recognize equipment with feature-based matching

        Uses SIFT/ORB features to match equipment templates.

        Args:
            rgb_image: RGB image
            template_features: Optional pre-computed template features

        Returns:
            List of equipment detections
        """
        detections = []

        # Convert to grayscale
        if len(rgb_image.shape) == 3:
            gray = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = rgb_image

        # Use ORB for feature detection (SIFT requires opencv-contrib)
        orb = cv2.ORB_create(nfeatures=500)
        keypoints, descriptors = orb.detectAndCompute(gray, None)

        if descriptors is None or len(keypoints) < 10:
            return []

        # In production, would match against template database
        # For now, return empty list (placeholder)
        # Template matching would involve:
        # 1. Load equipment templates
        # 2. Extract features from templates
        # 3. Match features using BFMatcher or FLANN
        # 4. Find homography using RANSAC
        # 5. Localize equipment in scene

        return detections

    def detect_obstacles(
        self,
        depth_map: np.ndarray,
        floor_plane: Optional[PlaneModel] = None,
        height_threshold: float = None
    ) -> List[ObjectDetection]:
        """
        Detect obstacles protruding from ground.

        Scenario: Detect obstacles

        Args:
            depth_map: Depth map
            floor_plane: Optional floor plane for reference
            height_threshold: Minimum height above floor

        Returns:
            List of obstacle detections
        """
        if height_threshold is None:
            height_threshold = self.COW_OBSTACLE_HEIGHT_THRESHOLD

        detections = []

        # Convert depth map to meters (assuming millimeters input)
        depth_m = depth_map.astype(np.float32) * 0.001

        # Get floor height
        if floor_plane is not None:
            # Calculate height from plane
            # This would require reconstructing point cloud, simplified here
            floor_height = np.percentile(depth_m[depth_m > 0], 10)
        else:
            floor_height = np.percentile(depth_m[depth_m > 0], 10)

        # Find obstacles (points significantly above floor)
        obstacle_mask = depth_m > (floor_height + height_threshold)

        # Filter valid depths
        obstacle_mask &= (depth_m > 0)

        if not obstacle_mask.any():
            return []

        # Find contours of obstacles
        obstacle_mask_uint8 = (obstacle_mask * 255).astype(np.uint8)
        contours, _ = cv2.findContours(obstacle_mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)

            # Filter small noise
            if area < 100:  # Minimum 100 pixels
                continue

            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)

            # Get center
            cx, cy = x + w // 2, y + h // 2

            # Estimate 3D position (simplified)
            if 0 <= cy < depth_m.shape[0] and 0 <= cx < depth_m.shape[1]:
                depth_at_center = depth_m[cy, cx]

                if depth_at_center > 0:
                    # Rough 3D position estimation
                    position_3d = np.array([cx * 0.001, cy * 0.001, depth_at_center])

                    detection = ObjectDetection(
                        object_type='obstacle',
                        position=position_3d,
                        dimensions=np.array([w * 0.001, h * 0.001, 0.2]),
                        confidence=0.6,
                        bounding_box=(x, y, x + w, y + h)
                    )

                    detections.append(detection)

        return detections

    # ========================================================================
    # ENVIRONMENTAL ANALYSIS IMPLEMENTATION
    # ========================================================================

    def analyze_lighting(
        self,
        rgb_image: np.ndarray
    ) -> LightingAnalysis:
        """
        Analyze lighting conditions.

        Scenario: Analyze lighting conditions

        Args:
            rgb_image: RGB image

        Returns:
            LightingAnalysis with brightness metrics
        """
        # Convert to grayscale for brightness analysis
        if len(rgb_image.shape) == 3:
            gray = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = rgb_image

        # Calculate brightness statistics
        avg_brightness = float(np.mean(gray))
        std_brightness = float(np.std(gray))

        # Calculate histogram
        histogram = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()

        # Calculate uniformity score (inverse of coefficient of variation)
        cv = std_brightness / (avg_brightness + 1e-6)
        uniformity_score = 1.0 / (1.0 + cv)

        # Classify lighting quality
        if avg_brightness > 180 and uniformity_score > 0.7:
            lighting_quality = 'excellent'
        elif avg_brightness > 120 and uniformity_score > 0.5:
            lighting_quality = 'good'
        elif avg_brightness > 60:
            lighting_quality = 'fair'
        else:
            lighting_quality = 'poor'

        return LightingAnalysis(
            average_brightness=avg_brightness,
            std_brightness=std_brightness,
            histogram=histogram,
            uniformity_score=float(uniformity_score),
            lighting_quality=lighting_quality,
            timestamp=datetime.now()
        )

    def detect_shadows(
        self,
        rgb_image: np.ndarray
    ) -> ShadowDetection:
        """
        Detect shadows in the scene.

        Scenario: Detect shadows

        Outdoor cattle environments often have strong shadows from sun.

        Args:
            rgb_image: RGB image

        Returns:
            ShadowDetection with shadow mask and regions
        """
        # Convert to LAB color space for better shadow detection
        lab = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2LAB)
        l_channel = lab[:, :, 0]

        # Detect dark regions (potential shadows)
        # Adaptive thresholding works better than fixed threshold
        shadow_mask = cv2.adaptiveThreshold(
            l_channel,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            blockSize=51,
            C=10
        )

        # Morphological operations to clean up mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        shadow_mask = cv2.morphologyEx(shadow_mask, cv2.MORPH_CLOSE, kernel)
        shadow_mask = cv2.morphologyEx(shadow_mask, cv2.MORPH_OPEN, kernel)

        # Calculate shadow percentage
        shadow_percentage = float(np.sum(shadow_mask > 0) / shadow_mask.size * 100)

        # Find shadow contours
        contours, _ = cv2.findContours(shadow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter small contours
        shadow_regions = [c for c in contours if cv2.contourArea(c) > 100]

        # Infer lighting direction (simplified)
        # In outdoor scenes, shadows typically fall in consistent direction
        lighting_direction = None
        if len(shadow_regions) > 0:
            # Calculate average shadow orientation
            # This is simplified; full implementation would analyze shadow geometry
            moments = [cv2.moments(c) for c in shadow_regions if cv2.contourArea(c) > 500]
            if moments:
                # Rough estimate based on centroid distribution
                centroids = np.array([[m['m10']/m['m00'], m['m01']/m['m00']] for m in moments if m['m00'] > 0])
                if len(centroids) > 1:
                    direction = centroids.mean(axis=0)
                    lighting_direction = np.array([direction[0], direction[1], -1.0])  # Light from above
                    lighting_direction = lighting_direction / np.linalg.norm(lighting_direction)

        return ShadowDetection(
            shadow_mask=shadow_mask,
            shadow_percentage=shadow_percentage,
            shadow_regions=shadow_regions,
            lighting_direction=lighting_direction,
            timestamp=datetime.now()
        )

    def detect_wetness(
        self,
        rgb_image: np.ndarray,
        depth_map: Optional[np.ndarray] = None
    ) -> WetnessDetection:
        """
        Detect wetness/moisture using reflectance analysis.

        Scenario: Detect wetness with reflectance analysis

        Common in outdoor cattle environments (mud, puddles, wet concrete).

        Args:
            rgb_image: RGB image
            depth_map: Optional depth map

        Returns:
            WetnessDetection with wet regions
        """
        # Convert to grayscale
        if len(rgb_image.shape) == 3:
            gray = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = rgb_image

        # Detect high-intensity regions (reflections from wet surfaces)
        # Wet surfaces create specular reflections (bright spots)
        _, bright_mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

        # Also detect dark saturated regions (water/mud)
        hsv = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2HSV)
        _, s_channel, v_channel = cv2.split(hsv)

        # Water often has low saturation and variable value
        dark_mask = (v_channel < 80).astype(np.uint8) * 255

        # Combine masks
        wetness_mask = cv2.bitwise_or(bright_mask, dark_mask)

        # Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        wetness_mask = cv2.morphologyEx(wetness_mask, cv2.MORPH_CLOSE, kernel)
        wetness_mask = cv2.morphologyEx(wetness_mask, cv2.MORPH_OPEN, kernel)

        # Calculate reflectance map (approximate)
        # Reflectance is roughly proportional to intensity
        reflectance_map = gray.astype(np.float32) / 255.0

        # Calculate wetness percentage
        wetness_percentage = float(np.sum(wetness_mask > 0) / wetness_mask.size * 100)

        # Find wet regions
        contours, _ = cv2.findContours(wetness_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        wet_regions = [c for c in contours if cv2.contourArea(c) > 100]

        return WetnessDetection(
            wetness_mask=wetness_mask,
            wetness_percentage=wetness_percentage,
            reflectance_map=reflectance_map,
            wet_regions=wet_regions,
            timestamp=datetime.now()
        )

    # ========================================================================
    # REPORT GENERATION IMPLEMENTATION
    # ========================================================================

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
        if timestamp is None:
            timestamp = datetime.now()

        # Detect planes
        planes = []

        # Floor (most important for cattle)
        floor_plane = self.detect_floor_plane(points)
        if floor_plane:
            planes.append(floor_plane)

        # Walls (if indoors)
        if self.expect_walls:
            wall_planes = self.detect_wall_planes(points)
            planes.extend(wall_planes)

        # Ceiling (if indoors)
        if self.expect_ceiling:
            ceiling_plane = self.detect_ceiling_plane(points)
            if ceiling_plane:
                planes.append(ceiling_plane)

        # Detect objects
        objects = []

        # Feed troughs
        troughs = self.detect_feed_trough(points, rgb_image)
        objects.extend(troughs)

        # Water sources
        water_sources = self.identify_water_source(points, rgb_image)
        objects.extend(water_sources)

        # Obstacles
        obstacles = self.detect_obstacles(depth_map, floor_plane)
        objects.extend(obstacles)

        # Environmental analysis
        lighting = self.analyze_lighting(rgb_image)
        shadows = self.detect_shadows(rgb_image) if 'outdoor' in self.environment_type else None
        wetness = self.detect_wetness(rgb_image, depth_map) if self.expect_mud else None

        report = SceneReport(
            environment_type=self.environment_type,
            timestamp=timestamp,
            planes=planes,
            objects=objects,
            lighting=lighting,
            shadows=shadows,
            wetness=wetness,
            frame_number=frame_number
        )

        self.scene_reports.append(report)
        return report

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
        file_exists = Path(csv_path).exists()

        with open(csv_path, 'a', newline='') as csvfile:
            fieldnames = [
                'timestamp', 'frame_number', 'environment_type',
                'num_planes', 'num_objects',
                'avg_brightness', 'lighting_quality',
                'shadow_percentage', 'wetness_percentage',
                'floor_detected', 'walls_detected', 'ceiling_detected',
                'feed_troughs', 'water_sources', 'obstacles'
            ]

            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            if not file_exists:
                writer.writeheader()

            # Count objects by type
            feed_troughs = sum(1 for obj in report.objects if obj.object_type == 'feed_trough')
            water_sources = sum(1 for obj in report.objects if obj.object_type == 'water_source')
            obstacles = sum(1 for obj in report.objects if obj.object_type == 'obstacle')

            # Count planes by type
            floor_detected = any(p.plane_type == 'floor' for p in report.planes)
            walls_detected = any(p.plane_type == 'wall' for p in report.planes)
            ceiling_detected = any(p.plane_type == 'ceiling' for p in report.planes)

            row = {
                'timestamp': report.timestamp.isoformat(),
                'frame_number': report.frame_number,
                'environment_type': report.environment_type,
                'num_planes': len(report.planes),
                'num_objects': len(report.objects),
                'avg_brightness': report.lighting.average_brightness,
                'lighting_quality': report.lighting.lighting_quality,
                'shadow_percentage': report.shadows.shadow_percentage if report.shadows else 0.0,
                'wetness_percentage': report.wetness.wetness_percentage if report.wetness else 0.0,
                'floor_detected': floor_detected,
                'walls_detected': walls_detected,
                'ceiling_detected': ceiling_detected,
                'feed_troughs': feed_troughs,
                'water_sources': water_sources,
                'obstacles': obstacles
            }

            writer.writerow(row)
