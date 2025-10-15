"""Dynamic resolution image loading - crops and scales high-res images for specific routes."""

import numpy as np
from PIL import Image
import math
import os

# Disable decompression bomb check for large tiles
Image.MAX_IMAGE_PIXELS = None


def calculate_route_bounds(route_coords, padding_degrees=10):
    """Calculate bounding box for a route with padding.
    
    Args:
        route_coords: List of (lat, lon) tuples
        padding_degrees: Padding to add around route (default 10 degrees)
        
    Returns:
        Tuple of (min_lat, max_lat, min_lon, max_lon)
    """
    lats = [lat for lat, lon in route_coords]
    lons = [lon for lat, lon in route_coords]
    
    min_lat = max(-90, min(lats) - padding_degrees)
    max_lat = min(90, max(lats) + padding_degrees)
    min_lon = max(-180, min(lons) - padding_degrees)
    max_lon = min(180, max(lons) + padding_degrees)
    
    return min_lat, max_lat, min_lon, max_lon


def calculate_target_dimensions(min_lat, max_lat, min_lon, max_lon, pixels_per_nm=2.0):
    """Calculate target image dimensions for desired resolution.

    IMPORTANT: Keep equirectangular aspect (linear degrees in both axes)
    so that lat/lon_to_pixel mapping aligns with imagery and TSS rasters.

    We choose resolution based on latitude degrees only (1° lat ≈ 60 nm),
    then derive width to maintain pixels-per-degree equal in both axes.

    Args:
        min_lat, max_lat, min_lon, max_lon: Bounding box in degrees
        pixels_per_nm: Target resolution along latitude (px per nautical mile)

    Returns:
        (target_width, target_height)
    """
    lat_span = max(1e-6, max_lat - min_lat)
    lon_span = max(1e-6, max_lon - min_lon)

    # Use latitude to set vertical resolution (1° lat ≈ 60 nm)
    height_nm = lat_span * 60.0
    target_height = int(round(height_nm * pixels_per_nm))

    # Enforce reasonable bounds
    target_height = max(100, min(50000, target_height))

    # Maintain equirectangular: same pixels-per-degree in both axes
    ppd = target_height / lat_span
    target_width = int(round(lon_span * ppd))
    target_width = max(100, min(50000, target_width))

    print(f"Route coverage: {lat_span:.1f}° lat × {lon_span:.1f}° lon")
    print(f"Vertical distance: {height_nm:.0f} nm (lat). Using {pixels_per_nm:.1f} px/nm")
    print(f"Target dimensions (equirectangular): {target_width} × {target_height} pixels")

    return target_width, target_height


def load_high_res_crop(min_lat, max_lat, min_lon, max_lon,
                       target_width, target_height,
                       image_dir="images/landmask_divided"):
    """Load and crop high-resolution world mask for the specified bounds.

    Uses the same mosaic-based math as core.initialization.load_and_process_divided_image
    to ensure the land mask aligns with coordinate conversions and TSS.

    Args:
        min_lat, max_lat, min_lon, max_lon: Bounding box in degrees
        target_width, target_height: Target output dimensions
        image_dir: Directory containing high-res tile images

    Returns:
        numpy array of shape (target_height, target_width) with True=water, False=land
    """
    # Clamp inputs to valid bounds
    min_lat = max(-90, min_lat)
    max_lat = min(90, max_lat)
    min_lon = max(-180, min_lon)
    max_lon = min(180, max_lon)

    # High-res mosaic geometry (2 rows x 4 cols tiles, each 21600x21600)
    tile_w = 21600
    tile_h = 21600
    cols = ['A', 'B', 'C', 'D']  # west -> east
    rows = ['1', '2']            # north -> south
    full_width_hi = tile_w * len(cols)   # 86400
    full_height_hi = tile_h * len(rows)  # 43200

    # Compute mosaic crop indices from lat/lon (top=90N, bottom=-90)
    north_px = int((90 - max_lat) / 180 * full_height_hi)
    south_px = int((90 - min_lat) / 180 * full_height_hi)
    west_px  = int((min_lon + 180) / 360 * full_width_hi)
    east_px  = int((max_lon + 180) / 360 * full_width_hi)

    # Clamp and validate
    north_px = max(0, min(full_height_hi, north_px))
    south_px = max(0, min(full_height_hi, south_px))
    west_px  = max(0, min(full_width_hi,  west_px))
    east_px  = max(0, min(full_width_hi,  east_px))
    crop_h = south_px - north_px
    crop_w = east_px  - west_px
    if crop_h <= 0 or crop_w <= 0:
        raise ValueError("Invalid crop window results in zero-sized region.")

    # Preallocate crop buffer
    crop_buffer = np.empty((crop_h, crop_w), dtype=np.uint8)

    # Iterate tiles and copy intersecting slices
    # Filenames: world.oceanmask.21600x21600.<COL><ROW>.png
    for col_idx, col in enumerate(cols):
        for row_idx, row in enumerate(rows):
            tile_name = f"{col}{row}"
            tile_path = os.path.join(image_dir, f"world.oceanmask.21600x21600.{tile_name}.png")
            if not os.path.exists(tile_path):
                # Skip silently; we'll only copy existing overlaps
                continue

            tile_x0 = col_idx * tile_w
            tile_y0 = row_idx * tile_h
            tile_x1 = tile_x0 + tile_w
            tile_y1 = tile_y0 + tile_h

            # Intersection with desired crop in mosaic space
            inter_x0 = max(west_px, tile_x0)
            inter_y0 = max(north_px, tile_y0)
            inter_x1 = min(east_px, tile_x1)
            inter_y1 = min(south_px, tile_y1)
            if inter_x0 >= inter_x1 or inter_y0 >= inter_y1:
                continue

            # Load tile lazily only when overlapping
            img = Image.open(tile_path).convert("L")
            tile_np = np.array(img)

            # Local coords inside tile
            local_x0 = inter_x0 - tile_x0
            local_x1 = inter_x1 - tile_x0
            local_y0 = inter_y0 - tile_y0
            local_y1 = inter_y1 - tile_y0
            tile_slice = tile_np[local_y0:local_y1, local_x0:local_x1]

            # Dest coords inside crop buffer
            dest_x0 = inter_x0 - west_px
            dest_x1 = inter_x1 - west_px
            dest_y0 = inter_y0 - north_px
            dest_y1 = inter_y1 - north_px
            crop_buffer[dest_y0:dest_y1, dest_x0:dest_x1] = tile_slice

    # Resize to requested working resolution
    print(f"Scaling from {(crop_h, crop_w)} to ({target_height}, {target_width})...")
    resized = Image.fromarray(crop_buffer).resize((target_width, target_height), Image.Resampling.LANCZOS)
    scaled_array = np.array(resized)

    # Convert to boolean (water=True, land=False). 0-20≈water in these tiles.
    is_water = scaled_array < 20

    print(f"Final mask: {is_water.shape}, water coverage: {is_water.sum() / is_water.size * 100:.1f}%")

    return is_water


def load_route_optimized_mask(route_coords, pixels_per_nm=2.0, padding_degrees=10):
    """Load a route-optimized mask with specified resolution.
    
    This is the main entry point for dynamic resolution loading.
    
    Args:
        route_coords: List of (lat, lon) tuples for the route
        pixels_per_nm: Target resolution (default 2.0 = 1px per 0.5nm)
        padding_degrees: Padding around route (default 10 degrees)
        
    Returns:
        Tuple of (is_water_mask, min_lat, max_lat, min_lon, max_lon, width, height)
    """
    print("\n" + "="*60)
    print("DYNAMIC RESOLUTION IMAGE LOADING")
    print("="*60)
    
    # Calculate route bounds
    min_lat, max_lat, min_lon, max_lon = calculate_route_bounds(route_coords, padding_degrees)
    print(f"\nRoute bounds: {min_lat:.1f}° to {max_lat:.1f}°, {min_lon:.1f}° to {max_lon:.1f}°")
    
    # Calculate target dimensions
    target_width, target_height = calculate_target_dimensions(
        min_lat, max_lat, min_lon, max_lon, pixels_per_nm
    )
    
    # Load and crop high-res images
    print(f"\nLoading high-resolution imagery...")
    is_water = load_high_res_crop(min_lat, max_lat, min_lon, max_lon, 
                                   target_width, target_height)
    
    print("="*60 + "\n")
    
    return is_water, min_lat, max_lat, min_lon, max_lon, target_width, target_height
