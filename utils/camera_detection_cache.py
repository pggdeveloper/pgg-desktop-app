"""
Camera detection cache to avoid repeated SDK queries.

This module provides caching functionality for camera detection results
to avoid expensive SDK queries and platform-specific enumeration on every call.

The cache has a configurable TTL (time-to-live) and can be cleared manually
when needed (e.g., when cameras are connected/disconnected).
"""

import time
from typing import Optional
from dataclasses import dataclass
from domain.camera_info import CameraInfo


@dataclass
class CameraDetectionCache:
    """
    Cache for camera detection results.

    Attributes:
        cameras: List of detected CameraInfo objects
        timestamp: Unix timestamp when cache was created
        ttl_seconds: Time-to-live in seconds (cache validity period)
    """
    cameras: list[CameraInfo]
    timestamp: float
    ttl_seconds: float = 30.0

    def is_valid(self) -> bool:
        """
        Check if cache is still valid based on TTL.

        Returns:
            True if cache has not expired, False otherwise
        """
        elapsed = time.time() - self.timestamp
        return elapsed < self.ttl_seconds


# Global cache instance
_detection_cache: Optional[CameraDetectionCache] = None


def get_cached_cameras() -> Optional[list[CameraInfo]]:
    """
    Get cameras from cache if valid.

    Returns:
        List of CameraInfo objects if cache is valid, None otherwise

    Example:
        cameras = get_cached_cameras()
        if cameras is None:
            cameras = enumerate_usb_external_cameras()
            set_cached_cameras(cameras)
    """
    global _detection_cache
    if _detection_cache and _detection_cache.is_valid():
        return _detection_cache.cameras
    return None


def set_cached_cameras(cameras: list[CameraInfo], ttl_seconds: float = 30.0):
    """
    Store cameras in cache.

    Args:
        cameras: List of CameraInfo objects to cache
        ttl_seconds: Time-to-live in seconds (default: 30 seconds)

    Note:
        Calling this function updates the global cache. Previous cache
        (if any) is replaced.
    """
    global _detection_cache
    _detection_cache = CameraDetectionCache(
        cameras=cameras,
        timestamp=time.time(),
        ttl_seconds=ttl_seconds
    )


def clear_cache():
    """
    Clear camera detection cache.

    Use this function when:
    - A camera is connected/disconnected
    - Camera configuration changes
    - Forcing a fresh detection is needed

    Example:
        clear_cache()
        cameras = enumerate_usb_external_cameras()
    """
    global _detection_cache
    _detection_cache = None


def get_cache_age() -> Optional[float]:
    """
    Get age of current cache in seconds.

    Returns:
        Age in seconds if cache exists, None otherwise
    """
    global _detection_cache
    if _detection_cache:
        return time.time() - _detection_cache.timestamp
    return None


def get_cache_remaining_ttl() -> Optional[float]:
    """
    Get remaining TTL of current cache in seconds.

    Returns:
        Remaining seconds until cache expires, None if no cache or expired
    """
    global _detection_cache
    if _detection_cache and _detection_cache.is_valid():
        return _detection_cache.ttl_seconds - (time.time() - _detection_cache.timestamp)
    return None
