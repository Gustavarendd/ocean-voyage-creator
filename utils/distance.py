"""Distance calculation utilities."""

import math

def nm_distance(lat1, lon1, lat2, lon2):
    """Calculate distance in nautical miles between two lat/lon points."""
    R = 3440.065  # Earth's radius in nautical miles
    
    # Handle dateline crossing by using the shorter of two possible paths
    lon_diff = abs(lon2 - lon1)
    if lon_diff > 180:
        # If the longitude difference is greater than 180°, it's shorter to go the other way
        lon_diff = 360 - lon_diff
        if lon1 < lon2:
            lon2 = lon2 - 360
        else:
            lon2 = lon2 + 360
    
    return math.acos(
        math.sin(math.radians(lat1)) * math.sin(math.radians(lat2)) +
        math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
        math.cos(math.radians(lon2 - lon1))
    ) * R
