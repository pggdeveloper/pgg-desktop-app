"""
Abstract base class for classical computer vision methods for animal health monitoring.

This module provides the base interface for applying classical CV techniques (edge detection,
corner detection, contour analysis, feature matching, etc.) to assess animal health, welfare,
and production metrics without GPU acceleration.

Key capabilities:
- Body Condition Scoring (BCS) through anatomical feature detection
- Weight estimation support via body dimension measurements
- Health monitoring through texture, color, and lesion detection
- Behavioral analysis via gait and activity tracking
- Individual identification through pattern matching
- Environmental assessment of pen/facility conditions
- Image quality enhancement for better analysis
- Integrated health scoring and reporting

Designed for extensibility to support multiple animal types (cattle, pigs, sheep, etc.)
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Literal
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


# ========================================================================
# ENUMS AND TYPE DEFINITIONS
# ========================================================================

class BCSScore(Enum):
    """Body Condition Score (1-5 scale)"""
    VERY_THIN = 1  # Emaciated, bones very prominent
    THIN = 2  # Under-conditioned, bones visible
    IDEAL = 3  # Optimal condition
    FAT = 4  # Over-conditioned, bones barely visible
    VERY_FAT = 5  # Obese, bones not visible


class HealthStatus(Enum):
    """Overall health status categories"""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    CRITICAL = "critical"


class LamenessScore(Enum):
    """Lameness severity scoring (0-5)"""
    SOUND = 0  # Normal gait
    MILD = 1  # Slightly abnormal gait
    MODERATE = 2  # Affected limb identifiable
    LAME = 3  # Reluctance to bear weight
    SEVERE = 4  # Minimal weight bearing
    NON_WEIGHT_BEARING = 5  # Unable to bear weight


# ========================================================================
# BODY CONDITION SCORING (BCS) DATA STRUCTURES
# ========================================================================

@dataclass
class SpineAnalysis:
    """Results from spine keypoint detection and prominence analysis."""
    keypoints: np.ndarray  # (N, 2) spine keypoints from neck to tail
    prominence_score: float  # 0-1, higher = more prominent
    vertebrae_visibility: int  # 1-5 scale
    curvature: float  # Spine curvature in radians
    posture_abnormal: bool
    bcs_spine_score: BCSScore
    timestamp: datetime
    confidence: float = 0.0


@dataclass
class RibCageAnalysis:
    """Results from rib cage prominence detection."""
    visible_rib_count: int
    rib_spacing_mm: float
    rib_sharpness: float  # 0-1, higher = sharper edges
    bcs_rib_score: BCSScore
    rib_locations: Optional[np.ndarray] = None  # (N, 4) line segments
    timestamp: Optional[datetime] = None
    confidence: float = 0.0


@dataclass
class HipBoneAnalysis:
    """Results from hip bone and tail head prominence detection."""
    hip_bone_prominence: float  # Depth profile measurement
    tail_head_depression_depth: float  # mm
    pelvic_bone_sharpness: float  # 0-1
    bcs_hip_score: BCSScore
    hip_keypoints: Optional[np.ndarray] = None  # (2, 2) left/right hip
    timestamp: Optional[datetime] = None
    confidence: float = 0.0


@dataclass
class BodyContourAnalysis:
    """Results from body contour extraction and shape analysis."""
    contour: np.ndarray  # (N, 2) body outline points
    smoothness: float  # Higher = more fat/rounded
    shape_rectangularity: float  # 0-1, rectangular vs rounded
    bcs_shape_score: BCSScore
    area_pixels: int
    perimeter_pixels: float
    timestamp: Optional[datetime] = None
    confidence: float = 0.0


@dataclass
class BodySymmetryAnalysis:
    """Results from body symmetry assessment."""
    asymmetry_index: float  # 0-1, 0 = perfect symmetry
    abnormal_swelling_detected: bool
    atrophy_detected: bool
    lameness_risk_score: float  # 0-1
    affected_side: Optional[Literal['left', 'right']] = None
    timestamp: Optional[datetime] = None
    confidence: float = 0.0


# ========================================================================
# WEIGHT ESTIMATION SUPPORT DATA STRUCTURES
# ========================================================================

@dataclass
class BodyDimensions:
    """Comprehensive body dimension measurements."""
    body_length_m: float  # Shoulder to hip
    heart_girth_m: float  # Chest circumference
    height_at_withers_m: float  # Floor to withers
    chest_depth_m: float  # Front-to-back
    chest_width_m: float  # Left-to-right
    length_to_height_ratio: float
    timestamp: Optional[datetime] = None
    confidence: float = 0.0


@dataclass
class WeightEstimate:
    """Weight estimation from body dimensions and volume."""
    estimated_weight_kg: float
    confidence: float
    method: Literal['dimensions', 'volume', 'combined']
    body_dimensions: Optional[BodyDimensions] = None
    body_volume_m3: Optional[float] = None
    timestamp: Optional[datetime] = None


# ========================================================================
# HEALTH MONITORING DATA STRUCTURES
# ========================================================================

@dataclass
class CoatAnalysis:
    """Results from coat texture and condition analysis."""
    smoothness_score: float  # 0-1, higher = smoother
    texture_roughness: float  # 0-1, higher = rougher
    abnormal_regions: List[Tuple[int, int, int, int]]  # (x, y, w, h) bounding boxes
    coat_condition_score: int  # 1-5 scale
    health_concerns: List[str]  # e.g., ["matted_hair", "alopecia"]
    timestamp: Optional[datetime] = None
    confidence: float = 0.0


@dataclass
class LesionDetection:
    """Results from skin lesion and wound detection."""
    lesion_locations: List[Tuple[int, int]]  # (x, y) centers
    lesion_sizes: List[float]  # Areas in pixels
    lesion_types: List[str]  # "abrasion", "laceration", "abscess"
    severity_scores: List[float]  # 0-1 per lesion
    total_lesion_count: int
    requires_vet_attention: bool
    timestamp: Optional[datetime] = None
    confidence: float = 0.0


@dataclass
class AnemiaDetection:
    """Results from eye/mucous membrane color analysis."""
    conjunctiva_color: str  # "pink", "pale", "yellow"
    anemia_risk_score: float  # 0-1
    color_histogram: Optional[np.ndarray] = None
    requires_health_check: bool = False
    timestamp: Optional[datetime] = None
    confidence: float = 0.0


@dataclass
class RespiratoryAssessment:
    """Results from nasal discharge detection."""
    discharge_present: bool
    discharge_color: Optional[str] = None  # "clear", "yellow", "green"
    respiratory_illness_risk: float = 0.0  # 0-1
    pneumonia_alert: bool = False
    timestamp: Optional[datetime] = None
    confidence: float = 0.0


@dataclass
class PostureAnalysis:
    """Results from posture analysis for pain/illness indicators."""
    body_angle_degrees: float
    head_droop_degrees: float
    back_arch_detected: bool
    abnormal_posture_type: Optional[str] = None  # "hunched", "head_down", "extended_neck"
    pain_indicators: List[str] = field(default_factory=list)
    illness_likelihood: float = 0.0  # 0-1
    timestamp: Optional[datetime] = None
    confidence: float = 0.0


# ========================================================================
# BEHAVIORAL ANALYSIS DATA STRUCTURES
# ========================================================================

@dataclass
class LamenessDetection:
    """Results from gait asymmetry and lameness analysis."""
    lameness_score: LamenessScore
    gait_symmetry_index: float  # 0-1, 1 = perfect symmetry
    head_bobbing_detected: bool
    stride_lengths: Dict[str, float]  # {"left_front": 1.2, "right_front": 1.1, ...}
    affected_limb: Optional[str] = None  # "left_front", "right_rear", etc.
    requires_examination: bool = False
    timestamp: Optional[datetime] = None
    confidence: float = 0.0


@dataclass
class ActivityAnalysis:
    """Results from standing/lying behavior tracking."""
    standing_time_percentage: float
    lying_duration_hours: float
    transition_count: int
    activity_level: Literal['very_low', 'low', 'normal', 'high']
    excessive_lying_detected: bool
    illness_risk: float = 0.0  # 0-1
    timestamp: Optional[datetime] = None


@dataclass
class FeedingBehavior:
    """Results from feeding behavior analysis."""
    feeding_duration_minutes: float
    feeding_frequency_per_day: int
    head_at_trough_percentage: float
    competition_detected: bool
    reduced_feeding_alert: bool
    timestamp: Optional[datetime] = None
    confidence: float = 0.0


@dataclass
class SocialInteraction:
    """Results from social interaction pattern analysis."""
    social_distance_m: float  # Average distance to nearest neighbor
    isolation_detected: bool
    aggressive_interaction_count: int
    social_stress_score: float  # 0-1
    herd_position: Literal['central', 'peripheral', 'isolated']
    timestamp: Optional[datetime] = None


# ========================================================================
# INDIVIDUAL IDENTIFICATION DATA STRUCTURES
# ========================================================================

@dataclass
class IndividualIdentification:
    """Results from coat pattern or facial recognition."""
    animal_id: Optional[str]
    confidence: float
    method: Literal['coat_pattern', 'facial', 'ear_tag', 'combined']
    feature_matches: int
    historical_records_available: bool
    timestamp: Optional[datetime] = None


@dataclass
class EarTagReading:
    """Results from ear tag OCR."""
    tag_number: Optional[str]
    confidence: float
    characters_detected: List[str]
    tag_location: Optional[Tuple[int, int, int, int]] = None  # (x, y, w, h)
    timestamp: Optional[datetime] = None


# ========================================================================
# ENVIRONMENTAL ANALYSIS DATA STRUCTURES
# ========================================================================

@dataclass
class PenCleanlinessAssessment:
    """Results from pen floor cleanliness analysis."""
    cleanliness_score: float  # 0-1
    soiled_area_percentage: float
    requires_cleaning: bool
    manure_detected: bool
    mud_detected: bool
    timestamp: Optional[datetime] = None


@dataclass
class WaterAvailability:
    """Results from water trough analysis."""
    water_present: bool
    water_level_cm: float
    trough_empty: bool
    requires_refill: bool
    timestamp: Optional[datetime] = None


@dataclass
class FeedDistribution:
    """Results from feed bunk uniformity analysis."""
    uniformity_score: float  # 0-1, higher = more uniform
    empty_sections_count: int
    feed_height_std_dev: float
    wastage_estimate_kg: float
    requires_pushup: bool
    timestamp: Optional[datetime] = None


# ========================================================================
# IMAGE ENHANCEMENT DATA STRUCTURES
# ========================================================================

@dataclass
class ImageEnhancement:
    """Results and metadata from image enhancement."""
    enhanced_image: np.ndarray
    enhancement_method: str
    quality_improvement_score: float  # 0-1
    timestamp: Optional[datetime] = None


# ========================================================================
# INTEGRATED HEALTH SCORING DATA STRUCTURES
# ========================================================================

@dataclass
class ComprehensiveHealthScore:
    """Integrated health assessment from multiple CV features."""
    overall_score: float  # 0-100
    health_status: HealthStatus
    bcs_component: float
    posture_component: float
    coat_component: float
    gait_component: float
    activity_component: float
    primary_concerns: List[str]
    veterinary_priority: Literal['none', 'routine', 'soon', 'urgent', 'emergency']
    trend: Literal['improving', 'stable', 'declining']
    timestamp: datetime


@dataclass
class WelfareReport:
    """Comprehensive welfare report with visual evidence."""
    report_period_days: int
    animal_id: Optional[str]
    health_score: ComprehensiveHealthScore
    bcs_trend: List[float]
    activity_trend: List[float]
    lesion_history: List[LesionDetection]
    gait_analysis: List[LamenessDetection]
    annotated_images: List[np.ndarray]
    recommendations: List[str]
    compliance_status: bool
    timestamp: datetime


# ========================================================================
# ABSTRACT BASE CLASS
# ========================================================================

class AnimalClassicalCV(ABC):
    """
    Abstract base class for classical computer vision analysis of animals.

    Provides interface for:
    - Body condition scoring through anatomical feature detection
    - Weight estimation support via body measurements
    - Health monitoring through texture, color, lesion detection
    - Behavioral analysis via gait and activity tracking
    - Individual identification through pattern matching
    - Environmental assessment
    - Image enhancement
    - Integrated health scoring
    """

    def __init__(
        self,
        animal_type: Literal['cow', 'bull', 'calf', 'heifer', 'steer', 'pig', 'sheep'],
        environment_type: Literal['feedlot_closed', 'feedlot_outdoor', 'pasture_natural', 'pasture_fodder'],
        enable_csv_export: bool = True,
        csv_output_dir: str = "./classical_cv_output"
    ):
        """
        Initialize classical CV analyzer.

        Args:
            animal_type: Type of animal being monitored
            environment_type: Environment setting (affects thresholds and expectations)
            enable_csv_export: Whether to export results to CSV
            csv_output_dir: Directory for CSV output files
        """
        self.animal_type = animal_type
        self.environment_type = environment_type
        self.enable_csv_export = enable_csv_export
        self.csv_output_dir = csv_output_dir

        # History tracking for trend analysis
        self.bcs_history: List[float] = []
        self.health_score_history: List[ComprehensiveHealthScore] = []
        self.lameness_history: List[LamenessDetection] = []
        self.lesion_history: List[LesionDetection] = []

    # ====================================================================
    # BODY CONDITION SCORING (BCS) - Scenarios 1.1-1.5
    # ====================================================================

    @abstractmethod
    def detect_spine_keypoints(
        self,
        rgb_image: np.ndarray,
        depth_map: np.ndarray
    ) -> SpineAnalysis:
        """
        Scenario 1.1: Detect cattle spine keypoints for BCS assessment.

        Uses edge detection to find spine ridge, extracts keypoints along spine
        from neck to tail base, measures prominence and vertebrae visibility.

        Args:
            rgb_image: RGB frame of dorsal (top-down) view
            depth_map: Corresponding depth frame

        Returns:
            SpineAnalysis with keypoints, prominence, visibility score, BCS score
        """
        pass

    @abstractmethod
    def detect_rib_cage_prominence(
        self,
        rgb_image: np.ndarray
    ) -> RibCageAnalysis:
        """
        Scenario 1.2: Detect rib cage prominence for BCS assessment.

        Uses Canny edge detection and horizontal line detection to identify
        individual ribs, quantify visibility, measure spacing and sharpness.

        Args:
            rgb_image: RGB frame of side view

        Returns:
            RibCageAnalysis with rib count, spacing, sharpness, BCS score
        """
        pass

    @abstractmethod
    def detect_hip_bone_prominence(
        self,
        rgb_image: np.ndarray,
        depth_map: np.ndarray
    ) -> HipBoneAnalysis:
        """
        Scenario 1.3: Detect hip bone and tail head prominence for BCS.

        Uses Harris/Shi-Tomasi corner detection for hip bones, blob detection
        for tail head region, measures prominence from depth profile.

        Args:
            rgb_image: RGB frame of rear view
            depth_map: Corresponding depth frame

        Returns:
            HipBoneAnalysis with prominence measurements and BCS score
        """
        pass

    @abstractmethod
    def extract_body_contour(
        self,
        rgb_image: np.ndarray,
        foreground_mask: Optional[np.ndarray] = None
    ) -> BodyContourAnalysis:
        """
        Scenario 1.4: Extract body contour for shape analysis.

        Uses edge detection and cv2.findContours to extract body outline,
        calculates smoothness (indicates body fat) and rectangularity.

        Args:
            rgb_image: RGB frame with animal segmented from background
            foreground_mask: Optional binary mask of animal (if None, will segment)

        Returns:
            BodyContourAnalysis with contour, smoothness, shape score, BCS score
        """
        pass

    @abstractmethod
    def analyze_body_symmetry(
        self,
        rgb_image: np.ndarray
    ) -> BodySymmetryAnalysis:
        """
        Scenario 1.5: Analyze body symmetry for health assessment.

        Detects body midline, compares left/right halves, identifies abnormal
        swelling or atrophy that may indicate lameness or health issues.

        Args:
            rgb_image: RGB frame of dorsal or rear view

        Returns:
            BodySymmetryAnalysis with asymmetry index and health alerts
        """
        pass

    # ====================================================================
    # WEIGHT ESTIMATION SUPPORT - Scenarios 2.1-2.4
    # ====================================================================

    @abstractmethod
    def estimate_heart_girth(
        self,
        rgb_image: np.ndarray,
        depth_map: np.ndarray
    ) -> float:
        """
        Scenario 2.2: Estimate heart girth circumference from depth profile.

        Segments chest region, extracts depth contour at chest level,
        measures chest depth and width, approximates circumference.

        Args:
            rgb_image: RGB frame of side view
            depth_map: Corresponding depth frame

        Returns:
            Estimated heart girth in meters
        """
        pass

    @abstractmethod
    def measure_comprehensive_dimensions(
        self,
        rgb_image: np.ndarray,
        depth_map: np.ndarray,
        point_cloud: Optional[np.ndarray] = None
    ) -> BodyDimensions:
        """
        Measure all body dimensions for weight estimation.

        Combines multiple measurements: body length, heart girth, height at withers.

        Args:
            rgb_image: RGB frame
            depth_map: Depth frame
            point_cloud: Optional 3D point cloud

        Returns:
            BodyDimensions with all measurements
        """
        pass

    @abstractmethod
    def estimate_weight_from_dimensions(
        self,
        dimensions: BodyDimensions
    ) -> WeightEstimate:
        """
        Estimate weight using allometric formula.

        Weight ≈ k × length^a × girth^b × height^c

        Args:
            dimensions: Body dimension measurements

        Returns:
            WeightEstimate with estimated weight and confidence
        """
        pass

    # ====================================================================
    # HEALTH MONITORING - Scenarios 3.1-3.5
    # ====================================================================

    @abstractmethod
    def analyze_coat_texture(
        self,
        rgb_image: np.ndarray
    ) -> CoatAnalysis:
        """
        Scenario 3.1: Analyze coat texture for health indicators.

        Uses CLAHE for texture enhancement, ORB keypoints for pattern detection,
        quantifies smoothness vs roughness, detects abnormal regions.

        Args:
            rgb_image: RGB frame of coat region

        Returns:
            CoatAnalysis with texture metrics and health alerts
        """
        pass

    @abstractmethod
    def detect_skin_lesions(
        self,
        rgb_image: np.ndarray
    ) -> LesionDetection:
        """
        Scenario 3.2: Detect skin lesions and wounds using blob detection.

        Uses color thresholding to isolate abnormal colors (red, dark),
        blob detection (LoG) for circular lesions, edge detection for wounds.

        Args:
            rgb_image: RGB frame of skin surface

        Returns:
            LesionDetection with lesion locations, sizes, types, severity
        """
        pass

    @abstractmethod
    def analyze_eye_color_for_anemia(
        self,
        rgb_image: np.ndarray
    ) -> AnemiaDetection:
        """
        Scenario 3.3: Analyze eye and mucous membrane color for anemia.

        Detects face region, isolates eye/conjunctiva, analyzes color
        (pink=healthy, pale=anemia), compares to baseline.

        Args:
            rgb_image: RGB frame with face visible

        Returns:
            AnemiaDetection with color analysis and anemia risk score
        """
        pass

    @abstractmethod
    def detect_nasal_discharge(
        self,
        rgb_image: np.ndarray
    ) -> RespiratoryAssessment:
        """
        Scenario 3.4: Detect nasal discharge for respiratory illness.

        Segments nose region using color/edge detection, analyzes texture
        for wetness/discharge, assesses color (clear vs infected).

        Args:
            rgb_image: RGB frame of muzzle/nose

        Returns:
            RespiratoryAssessment with discharge detection and illness risk
        """
        pass

    @abstractmethod
    def analyze_posture(
        self,
        rgb_image: np.ndarray,
        depth_map: np.ndarray
    ) -> PostureAnalysis:
        """
        Scenario 3.5: Analyze posture for pain or illness indicators.

        Detects keypoints (head, shoulders, back, hips, legs), calculates
        body angles, identifies abnormal postures (hunched, head down, arched back).

        Args:
            rgb_image: RGB frame of full body
            depth_map: Corresponding depth frame

        Returns:
            PostureAnalysis with angles, abnormal posture detection, pain indicators
        """
        pass

    # ====================================================================
    # BEHAVIORAL ANALYSIS - Scenarios 4.1-4.4
    # ====================================================================

    @abstractmethod
    def detect_lameness_from_gait(
        self,
        video_frames: List[np.ndarray],
        timestamps: List[float]
    ) -> LamenessDetection:
        """
        Scenario 4.1: Detect lameness from gait asymmetry.

        Uses optical flow to track leg movement over frames, measures stride
        length per leg, calculates symmetry index, detects head bobbing.

        Args:
            video_frames: Sequence of RGB frames showing walking
            timestamps: Frame timestamps

        Returns:
            LamenessDetection with lameness score, symmetry, affected limb
        """
        pass

    @abstractmethod
    def analyze_standing_lying_behavior(
        self,
        video_frames: List[np.ndarray],
        timestamps: List[float]
    ) -> ActivityAnalysis:
        """
        Scenario 4.2: Analyze standing vs lying behavior.

        Detects body orientation (vertical=standing, horizontal=lying),
        tracks lying duration, counts transitions, assesses activity level.

        Args:
            video_frames: Time-series of RGB frames
            timestamps: Frame timestamps

        Returns:
            ActivityAnalysis with standing/lying times, activity level, illness risk
        """
        pass

    @abstractmethod
    def analyze_feeding_behavior(
        self,
        video_frames: List[np.ndarray],
        feed_trough_region: Tuple[int, int, int, int]
    ) -> FeedingBehavior:
        """
        Scenario 4.3: Detect feeding behavior from head position.

        Tracks head region using template matching, measures time at feed trough,
        counts feeding frequency, detects competition/displacement.

        Args:
            video_frames: RGB frames of animal at feed bunk
            feed_trough_region: (x, y, w, h) bounding box of feed trough

        Returns:
            FeedingBehavior with duration, frequency, competition detection
        """
        pass

    @abstractmethod
    def analyze_social_interactions(
        self,
        rgb_image: np.ndarray
    ) -> SocialInteraction:
        """
        Scenario 4.4: Analyze social interaction patterns.

        Detects and tracks multiple animals, measures proximity, identifies
        isolation behavior, counts aggressive interactions.

        Args:
            rgb_image: RGB frame with multiple animals

        Returns:
            SocialInteraction with distance patterns, isolation detection
        """
        pass

    # ====================================================================
    # INDIVIDUAL IDENTIFICATION - Scenarios 5.1-5.3
    # ====================================================================

    @abstractmethod
    def identify_by_coat_pattern(
        self,
        rgb_image: np.ndarray,
        database_path: Optional[str] = None
    ) -> IndividualIdentification:
        """
        Scenario 5.1: Identify individual using coat pattern matching.

        Extracts coat pattern features using ORB/AKAZE descriptors,
        matches to database using FLANN, rejects outliers with RANSAC.

        Args:
            rgb_image: RGB frame of side view with unique markings
            database_path: Path to database of enrolled animals

        Returns:
            IndividualIdentification with animal ID and confidence
        """
        pass

    @abstractmethod
    def identify_by_facial_features(
        self,
        rgb_image: np.ndarray,
        database_path: Optional[str] = None
    ) -> IndividualIdentification:
        """
        Scenario 5.2: Recognize facial features for identification.

        Detects face keypoints (eyes, nose, muzzle), extracts SIFT descriptors,
        matches to enrolled database.

        Args:
            rgb_image: RGB frame of face
            database_path: Path to database of enrolled animals

        Returns:
            IndividualIdentification with animal ID and confidence
        """
        pass

    @abstractmethod
    def read_ear_tag(
        self,
        rgb_image: np.ndarray
    ) -> EarTagReading:
        """
        Scenario 5.3: Track ear tag numbers using OCR-like techniques.

        Uses edge detection to outline ear tag, applies perspective correction,
        segments and decodes individual characters.

        Args:
            rgb_image: RGB frame with visible ear tag

        Returns:
            EarTagReading with tag number and confidence
        """
        pass

    # ====================================================================
    # ENVIRONMENTAL ANALYSIS - Scenarios 6.1-6.3
    # ====================================================================

    @abstractmethod
    def assess_pen_cleanliness(
        self,
        rgb_image: np.ndarray
    ) -> PenCleanlinessAssessment:
        """
        Scenario 6.1: Assess pen cleanliness using texture and color analysis.

        Uses color segmentation to distinguish clean vs soiled areas,
        texture analysis to detect manure or mud.

        Args:
            rgb_image: RGB frame of pen floor

        Returns:
            PenCleanlinessAssessment with cleanliness score and alerts
        """
        pass

    @abstractmethod
    def detect_water_availability(
        self,
        rgb_image: np.ndarray,
        depth_map: np.ndarray
    ) -> WaterAvailability:
        """
        Scenario 6.2: Detect water availability in troughs.

        Uses edge detection for water surface, depth profile to confirm
        water presence, measures water level.

        Args:
            rgb_image: RGB frame of water trough
            depth_map: Corresponding depth frame

        Returns:
            WaterAvailability with water level and refill alerts
        """
        pass

    @abstractmethod
    def analyze_feed_distribution(
        self,
        rgb_image: np.ndarray,
        depth_map: np.ndarray
    ) -> FeedDistribution:
        """
        Scenario 6.3: Analyze feed distribution uniformity in bunks.

        Uses depth profile to measure feed height along bunk length,
        edge detection for feed pile boundaries, calculates uniformity.

        Args:
            rgb_image: RGB frame of feed bunk
            depth_map: Corresponding depth frame

        Returns:
            FeedDistribution with uniformity score and alerts
        """
        pass

    # ====================================================================
    # IMAGE QUALITY ENHANCEMENT - Scenarios 7.1-7.5
    # ====================================================================

    @abstractmethod
    def enhance_low_light_image(
        self,
        rgb_image: np.ndarray
    ) -> ImageEnhancement:
        """
        Scenario 7.1: Enhance low-light images for night monitoring.

        Applies CLAHE (Contrast Limited Adaptive Histogram Equalization)
        to enhance local contrast without over-amplification.

        Args:
            rgb_image: Low-light RGB frame

        Returns:
            ImageEnhancement with enhanced image
        """
        pass

    @abstractmethod
    def remove_motion_blur(
        self,
        rgb_image: np.ndarray
    ) -> ImageEnhancement:
        """
        Scenario 7.2: Remove motion blur from moving animal images.

        Applies image sharpening kernel (unsharp masking) to enhance edges
        and reduce motion blur.

        Args:
            rgb_image: Motion-blurred RGB frame

        Returns:
            ImageEnhancement with sharpened image
        """
        pass

    @abstractmethod
    def balance_outdoor_lighting(
        self,
        rgb_image: np.ndarray
    ) -> ImageEnhancement:
        """
        Scenario 7.3: Reduce outdoor lighting variations.

        Applies adaptive histogram equalization to balance illumination,
        brighten shadows, tone down overexposed regions.

        Args:
            rgb_image: RGB frame with harsh shadows/highlights

        Returns:
            ImageEnhancement with balanced illumination
        """
        pass

    @abstractmethod
    def denoise_infrared_frame(
        self,
        infrared_image: np.ndarray
    ) -> ImageEnhancement:
        """
        Scenario 7.4: Denoise infrared frames for night vision.

        Applies bilateral filter to reduce noise while preserving edges.

        Args:
            infrared_image: Infrared frame with thermal noise

        Returns:
            ImageEnhancement with denoised image
        """
        pass

    @abstractmethod
    def remove_weather_artifacts(
        self,
        rgb_image: np.ndarray
    ) -> ImageEnhancement:
        """
        Scenario 7.5: Remove rain/snow artifacts from outdoor images.

        Uses median filter for impulse noise, morphological opening for
        small artifacts (rain streaks, snow particles).

        Args:
            rgb_image: RGB frame with rain/snow artifacts

        Returns:
            ImageEnhancement with artifacts removed
        """
        pass

    # ====================================================================
    # INTEGRATED HEALTH SCORING - Scenarios 8.1-8.2
    # ====================================================================

    @abstractmethod
    def calculate_comprehensive_health_score(
        self,
        bcs_analysis: Optional[SpineAnalysis] = None,
        posture_analysis: Optional[PostureAnalysis] = None,
        coat_analysis: Optional[CoatAnalysis] = None,
        lesion_detection: Optional[LesionDetection] = None,
        lameness_detection: Optional[LamenessDetection] = None,
        activity_analysis: Optional[ActivityAnalysis] = None
    ) -> ComprehensiveHealthScore:
        """
        Scenario 8.1: Calculate comprehensive health score from multiple CV features.

        Applies weighted scoring to combine multiple health indicators,
        categorizes overall health status, identifies primary concerns,
        assigns veterinary priority.

        Args:
            bcs_analysis: Body condition score results
            posture_analysis: Posture assessment results
            coat_analysis: Coat condition results
            lesion_detection: Lesion detection results
            lameness_detection: Lameness assessment results
            activity_analysis: Activity level results

        Returns:
            ComprehensiveHealthScore with overall score, status, priorities
        """
        pass

    @abstractmethod
    def generate_welfare_report(
        self,
        animal_id: Optional[str],
        report_period_days: int,
        include_images: bool = True
    ) -> WelfareReport:
        """
        Scenario 8.2: Generate welfare report with visual evidence.

        Aggregates health metrics over time period, highlights abnormal
        findings with image snapshots, provides recommendations.

        Args:
            animal_id: Unique identifier for animal
            report_period_days: Number of days to include in report
            include_images: Whether to include annotated images

        Returns:
            WelfareReport with trends, images, recommendations, compliance status
        """
        pass

    # ====================================================================
    # CSV EXPORT AND UTILITY METHODS
    # ====================================================================

    @abstractmethod
    def save_analysis_to_csv(
        self,
        analysis_result: Any,
        csv_filename: str
    ) -> bool:
        """
        Save any analysis result to CSV file.

        Args:
            analysis_result: Any dataclass result from analysis
            csv_filename: Output CSV filename

        Returns:
            True if successful, False otherwise
        """
        pass

    @abstractmethod
    def export_batch_results_to_csv(
        self,
        results: List[Any],
        csv_filename: str
    ) -> bool:
        """
        Export batch of results to CSV.

        Args:
            results: List of analysis results
            csv_filename: Output CSV filename

        Returns:
            True if successful, False otherwise
        """
        pass
