"""Core initialization and image loading functions."""

from PIL import Image, ImageFile
import numpy as np
from config import *

# Globals to communicate active cropped bounds
ACTIVE_LAT_MIN = LAT_MIN
ACTIVE_LAT_MAX = LAT_MAX
ACTIVE_LON_MIN = LON_MIN
ACTIVE_LON_MAX = LON_MAX

def load_and_process_images(land_mask_path, max_lat, min_lat, max_lon, min_lon):
    Image.MAX_IMAGE_PIXELS = None  # just above 233,280,000; or set to None to disable
    ImageFile.LOAD_TRUNCATED_IMAGES = True  # optional: avoids errors on slightly corrupted files
    # Load and process land mask
    land_mask = Image.open(land_mask_path).convert("L")
    land_mask_np = np.array(land_mask)
    
    # Calculate crop indices for land mask
    full_height, full_width = land_mask_np.shape
    
     # Clamp input bounds to valid ranges
    max_lat = min(max_lat, 90)
    min_lat = max(min_lat, -90)
    max_lon = min(max_lon, 180)
    min_lon = max(min_lon, -180)

    # Convert lat/lon bounds to pixel indices
    north_limit_px = int((90 - max_lat) / 180 * full_height)
    south_limit_px = int((90 - min_lat) / 180 * full_height)
    west_limit_px  = int((min_lon + 180) / 360 * full_width)
    east_limit_px  = int((max_lon + 180) / 360 * full_width)
    
    # Crop and resize land mask
    land_mask_cropped = land_mask_np[north_limit_px:south_limit_px, west_limit_px:east_limit_px]
    
    # Resize to standard working resolution
    land_mask_resized = Image.fromarray(land_mask_cropped).resize(
        (IMAGE_WIDTH, IMAGE_HEIGHT),
        Image.Resampling.BILINEAR
    )
    land_mask_np = np.array(land_mask_resized)
    
    # Create water mask
    is_water = land_mask_np < 20  # True for water, False for land

    # Update active bounds globals
    global ACTIVE_LAT_MIN, ACTIVE_LAT_MAX, ACTIVE_LON_MIN, ACTIVE_LON_MAX
    ACTIVE_LAT_MIN = min_lat
    ACTIVE_LAT_MAX = max_lat
    ACTIVE_LON_MIN = min_lon
    ACTIVE_LON_MAX = max_lon

    return is_water

def get_active_bounds():
    """Return the current active crop bounds (lat_min, lat_max, lon_min, lon_max)."""
    return ACTIVE_LAT_MIN, ACTIVE_LAT_MAX, ACTIVE_LON_MIN, ACTIVE_LON_MAX

