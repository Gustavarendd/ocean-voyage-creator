"""Coordinate conversion and validation utilities."""

from config import LAT_MIN, LAT_MAX, LON_MIN, LON_MAX, IMAGE_WIDTH, IMAGE_HEIGHT
try:
    from core.initialization import get_active_bounds, get_active_dimensions
except Exception:
    def get_active_bounds():  # fallback
        return LAT_MIN, LAT_MAX, LON_MIN, LON_MAX
    def get_active_dimensions():  # fallback
        return IMAGE_WIDTH, IMAGE_HEIGHT
    


def validate_coordinates(lat, lon, warn=True):
    """Validate coordinates against the available data range."""
    lat_min, lat_max, lon_min, lon_max = get_active_bounds()

    if not (lat_min <= lat <= lat_max):
        if warn:
            print(f"Warning: Latitude {lat}° is outside the current data range ({lat_min}° to {lat_max}°)")
        return False
    if not (lon_min <= lon <= lon_max):
        if warn:
            print(f"Warning: Longitude {lon}° is outside the current data range ({lon_min}° to {lon_max}°)")
        return False
    return True

def latlon_to_pixel(lat, lon, warn=True):
    """Convert latitude/longitude to pixel coordinates.

    Args:
        lat, lon: geographic coordinates
        warn: if False, suppress out-of-range warnings (raises ValueError still)
    """
    if not validate_coordinates(lat, lon, warn=warn):
        raise ValueError(f"Coordinates ({lat}, {lon}) are outside the valid range")
    
    # For longitude, ensure we handle dateline crossing by normalizing the longitude
    lat_min, lat_max, lon_min, lon_max = get_active_bounds()
    width, height = get_active_dimensions()

    normalized_lon = lon
    # Keep simple normalization relative to global +-180 for safety
    if normalized_lon > 180:
        normalized_lon -= 360
    elif normalized_lon < -180:
        normalized_lon += 360

    x = int((normalized_lon - lon_min) / (lon_max - lon_min) * width)
    y = int((lat_max - lat) / (lat_max - lat_min) * height)
    return x, y

def pixel_to_latlon(x, y):
    """Convert pixel coordinates to latitude/longitude."""
    lat_min, lat_max, lon_min, lon_max = get_active_bounds()
    width, height = get_active_dimensions()

    lon = lon_min + x / width * (lon_max - lon_min)
    lat = lat_max - y / height * (lat_max - lat_min)
    
    # Normalize longitude to be in the range [-180, 180]
    if lon > 180:
        lon -= 360
    elif lon < -180:
        lon += 360
    
    return lat, lon


