"""Core initialization and image loading functions."""

from PIL import Image, ImageFile
import numpy as np
import os
import re
from config import *

# Globals to communicate active cropped bounds and dimensions
ACTIVE_LAT_MIN = LAT_MIN
ACTIVE_LAT_MAX = LAT_MAX
ACTIVE_LON_MIN = LON_MIN
ACTIVE_LON_MAX = LON_MAX
ACTIVE_WIDTH = IMAGE_WIDTH
ACTIVE_HEIGHT = IMAGE_HEIGHT

def set_active_bounds(lat_min, lat_max, lon_min, lon_max, width, height):
    """Set the active bounds and dimensions for coordinate conversions."""
    global ACTIVE_LAT_MIN, ACTIVE_LAT_MAX, ACTIVE_LON_MIN, ACTIVE_LON_MAX, ACTIVE_WIDTH, ACTIVE_HEIGHT
    ACTIVE_LAT_MIN = lat_min
    ACTIVE_LAT_MAX = lat_max
    ACTIVE_LON_MIN = lon_min
    ACTIVE_LON_MAX = lon_max
    ACTIVE_WIDTH = width
    ACTIVE_HEIGHT = height

def get_active_dimensions():
    """Return the active image dimensions."""
    return ACTIVE_WIDTH, ACTIVE_HEIGHT

def load_and_process_image(max_lat, min_lat, max_lon, min_lon):
    land_mask_path = "./images/land_mask_90N_90S_21600x10800.png"
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


def load_and_process_divided_image(max_lat, min_lat, max_lon, min_lon, tiles_dir="./images/landmask_divided", threshold_world_fraction=0.6):
    """Load and process the high-resolution divided land mask tiles.

    This function mirrors the behavior of `load_and_process_image`, but instead of
    using the single full-world PNG it works with 8 higher-resolution tiles laid
    out in a 2 (rows) x 4 (columns) grid:

        Columns (west -> east): A, B, C, D
        Rows (north -> south):  1 (top), 2 (bottom)

    Each tile file name pattern: world.oceanmask.21600x21600.<COL><ROW>.png

    The combined mosaic dimensions (high-res) are:
        width  = 4 * 21600 = 86400
        height = 2 * 21600 = 43200
    Which is exactly a 4x linear upscale of the base 21600x10800 land mask.

    We crop only the requested geographic window from the mosaic without
    assembling the entire 3.7+ GB array in memory. Then we downsample to the
    standard working resolution (IMAGE_WIDTH x IMAGE_HEIGHT) and convert to a
    boolean water mask consistent with `load_and_process_image`.

    To avoid accidental huge allocations, if the requested window exceeds
    `threshold_world_fraction` fraction of total world pixels at high-res, we
    fallback to the lower-resolution single-image path (`load_and_process_image`).

    Parameters
    ----------
    max_lat, min_lat, max_lon, min_lon : float
        Geographic bounding box to load (will be clamped to valid ranges).
    tiles_dir : str
        Directory containing the 8 tile PNGs.
    threshold_world_fraction : float (0-1)
        Fraction of world area (in pixels at high-res) beyond which we fallback.

    Returns
    -------
    is_water : np.ndarray[bool]
        Boolean array (IMAGE_HEIGHT, IMAGE_WIDTH) True for water, False for land.
    """
    # Clamp inputs
    max_lat = min(max_lat, 90)
    min_lat = max(min_lat, -90)
    max_lon = min(max_lon, 180)
    min_lon = max(min_lon, -180)

    # High-res mosaic geometry
    tile_w = 21600
    tile_h = 21600
    cols = ['A', 'B', 'C', 'D']  # west -> east
    rows = ['1', '2']            # north -> south
    full_width_hi = tile_w * len(cols)   # 86400
    full_height_hi = tile_h * len(rows)  # 43200

    # Compute high-res crop indices
    north_px = int((90 - max_lat) / 180 * full_height_hi)
    south_px = int((90 - min_lat) / 180 * full_height_hi)
    west_px  = int((min_lon + 180) / 360 * full_width_hi)
    east_px  = int((max_lon + 180) / 360 * full_width_hi)

    # Bounds safety
    north_px = max(0, min(full_height_hi, north_px))
    south_px = max(0, min(full_height_hi, south_px))
    west_px  = max(0, min(full_width_hi,  west_px))
    east_px  = max(0, min(full_width_hi,  east_px))

    crop_h = south_px - north_px
    crop_w = east_px  - west_px
    if crop_h <= 0 or crop_w <= 0:
        raise ValueError("Invalid crop window results in zero-sized region.")

    # Fallback if too large (avoid multi-GB allocation)
    world_pixels_hi = full_width_hi * full_height_hi
    crop_pixels = crop_w * crop_h
    # if crop_pixels / world_pixels_hi > threshold_world_fraction:
    #     # Use existing low-res path for memory efficiency
    #     return load_and_process_image(max_lat, min_lat, max_lon, min_lon)

    # Preallocate target crop buffer (uint8) then fill from intersecting tiles
    crop_buffer = np.empty((crop_h, crop_w), dtype=np.uint8)

    # Compile regex for tile parsing (optional future validation)
    tile_re = re.compile(r"world\.oceanmask\.21600x21600\.([A-D])([12])\.png$")

    Image.MAX_IMAGE_PIXELS = None
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    # Iterate tiles and copy intersecting slices
    for fname in os.listdir(tiles_dir):
        m = tile_re.match(fname)
        if not m:
            continue
        col_letter, row_number = m.group(1), m.group(2)
        c_idx = cols.index(col_letter)
        r_idx = rows.index(row_number)
        tile_x0 = c_idx * tile_w
        tile_y0 = r_idx * tile_h
        tile_x1 = tile_x0 + tile_w
        tile_y1 = tile_y0 + tile_h

        # Compute intersection with desired crop
        inter_x0 = max(west_px, tile_x0)
        inter_y0 = max(north_px, tile_y0)
        inter_x1 = min(east_px, tile_x1)
        inter_y1 = min(south_px, tile_y1)
        if inter_x0 >= inter_x1 or inter_y0 >= inter_y1:
            continue  # no overlap

        # Open tile only when needed
        tile_path = os.path.join(tiles_dir, fname)
        with Image.open(tile_path).convert("L") as tile_img:
            tile_np = np.array(tile_img)
            # Local coordinates inside tile
            local_x0 = inter_x0 - tile_x0
            local_x1 = inter_x1 - tile_x0
            local_y0 = inter_y0 - tile_y0
            local_y1 = inter_y1 - tile_y0
            tile_slice = tile_np[local_y0:local_y1, local_x0:local_x1]

        # Destination coordinates inside crop buffer
        dest_x0 = inter_x0 - west_px
        dest_x1 = inter_x1 - west_px
        dest_y0 = inter_y0 - north_px
        dest_y1 = inter_y1 - north_px
        crop_buffer[dest_y0:dest_y1, dest_x0:dest_x1] = tile_slice

    # Downsample to working resolution
    crop_img = Image.fromarray(crop_buffer)
    resized = crop_img.resize((IMAGE_WIDTH, IMAGE_HEIGHT), Image.Resampling.BILINEAR)
    land_mask_np = np.array(resized)
    is_water = land_mask_np < 20

    # Update globals
    global ACTIVE_LAT_MIN, ACTIVE_LAT_MAX, ACTIVE_LON_MIN, ACTIVE_LON_MAX
    ACTIVE_LAT_MIN = min_lat
    ACTIVE_LAT_MAX = max_lat
    ACTIVE_LON_MIN = min_lon
    ACTIVE_LON_MAX = max_lon

    return is_water

