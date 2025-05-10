"""Coordinate conversion and validation utilities."""

from config import LAT_MIN, LAT_MAX, LON_MIN, LON_MAX, IMAGE_WIDTH, IMAGE_HEIGHT

def validate_coordinates(lat, lon, warn=True):
    """Validate coordinates against the available data range."""
    if not (LAT_MIN <= lat <= LAT_MAX):
        if warn:
            print(f"Warning: Latitude {lat}° is outside the current data range ({LAT_MIN}° to {LAT_MAX}°)")
        return False
    if not (LON_MIN <= lon <= LON_MAX):
        if warn:
            print(f"Warning: Longitude {lon}° is outside the current data range ({LON_MIN}° to {LON_MAX}°)")
        return False
    return True

def latlon_to_pixel(lat, lon):
    """Convert latitude/longitude to pixel coordinates."""
    if not validate_coordinates(lat, lon):
        raise ValueError(f"Coordinates ({lat}, {lon}) are outside the valid range")
    
    x = int((lon - LON_MIN) / (LON_MAX - LON_MIN) * IMAGE_WIDTH)
    y = int((LAT_MAX - lat) / (LAT_MAX - LAT_MIN) * IMAGE_HEIGHT)
    return x, y

def pixel_to_latlon(x, y):
    """Convert pixel coordinates to latitude/longitude."""
    lon = LON_MIN + x / IMAGE_WIDTH * (LON_MAX - LON_MIN)
    lat = LAT_MAX - y / IMAGE_HEIGHT * (LAT_MAX - LAT_MIN)
    return lat, lon
