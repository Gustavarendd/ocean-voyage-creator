"""Distance calculation utilities."""

import math

def nm_distance(lat1, lon1, lat2, lon2):
    """Calculate distance in nautical miles between two lat/lon points."""
    R = 3440.065  # Earth's radius in nautical miles
    return math.acos(
        math.sin(math.radians(lat1)) * math.sin(math.radians(lat2)) +
        math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
        math.cos(math.radians(lon2 - lon1))
    ) * R
