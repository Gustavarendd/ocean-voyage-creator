"""Water mask creation and processing functions.

Enhancements:
        - Optional integration of TSS (Traffic Separation Scheme)    if force_recompute:
        print("Force recompute requested - computing buffered water mask...")
    else:
        print("Computing buffered water mask (this may take a moment)...")

    import time
    start_time = time.time()

    try:
        from core.initialization import get_active_bounds as _gab
        lat_min, lat_max, lon_min, lon_max = _gab()
    except Exception:
        lat_min, lat_max, lon_min, lon_max = LAT_MIN, LAT_MAX, LON_MIN, LON_MAX
    
    # Calculate buffer size in pixels (base grid), then scale if supersampling
    base_pixels_per_nm = (IMAGE_WIDTH / (lon_max - lon_min)) / 60
    buffer_pixels = int(buffer_nm * base_pixels_per_nm * 2)
    print(f"Buffer size: {buffer_pixels} pixels for {buffer_nm} nm")           GeoJSON file (e.g. `separation_lanes_with_direction.geojson`).
            Any features whose `seamark:type` property matches one of the provided
            lane types (default: separation_zone, separation_lane) are rasterized
            onto the water mask and converted to land so that routing will avoid
            crossing them.
"""

import numpy as np
import os
import hashlib
import time
from scipy import ndimage
from config import IMAGE_WIDTH, IMAGE_HEIGHT, CRITICAL_REGIONS, LON_MIN, LON_MAX, LAT_MIN, LAT_MAX
import json

from utils.coordinates import get_active_bounds

try:
    from shapely.geometry import LineString, Polygon, Point  # type: ignore
except Exception:  # Shapely should be available; keep graceful degradation
    LineString = None
    Polygon = None
    Point = None


try:
    from core.initialization import ACTIVE_LON_MIN, ACTIVE_LON_MAX
except Exception:
    # Fallback to full range so behavior matches original if initialization not run
    ACTIVE_LON_MIN, ACTIVE_LON_MAX = -180.0, 180.0


def _create_cache_key(is_water, buffer_nm):
    """Create a unique cache key based on input parameters."""
    # Create hash of the water mask and buffer parameters
    water_hash = hashlib.md5(is_water.tobytes()).hexdigest()[:8]
    critical_hash = hashlib.md5(str(CRITICAL_REGIONS).encode()).hexdigest()[:8]
    buffer_str = f"{buffer_nm:.1f}"
    
    return f"{water_hash}_{critical_hash}_{buffer_str}"

def clear_buffered_water_cache():
    """Clear all cached buffered water masks."""
    cache_dir = "cache"
    if not os.path.exists(cache_dir):
        return
    
    import glob
    cache_files = glob.glob(os.path.join(cache_dir, "buffered_water_*.np*"))  # Match both .npy and .npz
    
    for cache_file in cache_files:
        try:
            os.remove(cache_file)
            print(f"Removed cache file: {cache_file}")
        except Exception as e:
            print(f"Error removing cache file {cache_file}: {e}")
    
    if not cache_files:
        print("No cache files found to remove.")

def create_buffered_water_mask(
    is_water,
    buffer_nm,
    force_recompute: bool = False,
    tss_geojson_path: str | None = None,
    land_lane_types: list[str] | None = None,
    water_lane_types: list[str] | None = None,
    lane_pixel_width: int = 10,
    apply_tss_before_buffer: bool = True,
    tss_lanes: np.ndarray | None = None,

):
    """Create a water mask with a coastal buffer zone while preserving small islands and narrow channels.
    Uses caching to avoid recomputation when the same parameters are used.
    
    Args:
        is_water: Boolean array indicating water (True) vs land (False)
        buffer_nm: Buffer distance in nautical miles
        force_recompute: If True, bypass cache and recompute the mask
        tss_geojson_path: Optional path to GeoJSON file containing TSS lanes/zones.
        land_lane_types: List of seamark:type values to convert to land. Defaults to
                    ["separation_zone", "separation_lane"].
        lane_pixel_width: Thickness (in pixels) of the rasterized TSS feature when
                          converting to land.
    apply_tss_before_buffer: If True (default) treat TSS features as land prior
                 to applying the coastline buffer (so buffer will
                 also expand them). If False, they are carved out
                 after buffering (narrower restriction).
   
    """
    # Create cache directory if it doesn't exist
    cache_dir = "cache"
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    
    # Create a hash of the input parameters to use as cache key
    cache_key = _create_cache_key(is_water, buffer_nm)
    
    # Incorporate TSS modifications into cache key if provided
    tss_hash = None
    if tss_geojson_path and os.path.exists(tss_geojson_path):
        try:
            with open(tss_geojson_path, "rb") as f:
                import hashlib as _hl
                tss_hash = _hl.md5(f.read()).hexdigest()[:8]
            cache_key += f"_tss{tss_hash}_w{lane_pixel_width}_pre{int(apply_tss_before_buffer)}"
        except Exception as e:
            print(f"Warning: Could not hash TSS file for cache key ({e}); not caching TSS variant uniquely.")
    
    # Add tss_lanes hash to cache key if provided
    if tss_lanes is not None:
        tss_lanes_hash = hashlib.md5(tss_lanes.tobytes()).hexdigest()[:8]
        cache_key += f"_tssl{tss_lanes_hash}"
    
    cache_file = os.path.join(cache_dir, f"buffered_water_{cache_key}.npz")
    
    # Check if cached version exists and force_recompute is False
    if not force_recompute and os.path.exists(cache_file):
        print(f"Loading cached buffered water mask from {cache_file}")
        try:
            cached_data = np.load(cache_file)
            cached_mask = cached_data['mask']
            print(f"Successfully loaded cached mask with shape: {cached_mask.shape}")
            return cached_mask
        except Exception as e:
            print(f"Error loading cached mask: {e}. Recomputing...")
    
    if force_recompute:
        print("Force recompute requested - computing buffered water mask...")
    else:
        print("Computing buffered water mask (this may take a moment)...")

    start_time = time.time()

    try:
        from core.initialization import get_active_bounds as _gab
        lat_min, lat_max, lon_min, lon_max = _gab()
    except Exception:
        lat_min, lat_max, lon_min, lon_max = LAT_MIN, LAT_MAX, LON_MIN, LON_MAX
    
    # Calculate buffer size in pixels (base grid), then scale if supersampling
    base_pixels_per_nm = (IMAGE_WIDTH / (lon_max - lon_min)) / 60
    buffer_pixels = int(buffer_nm * base_pixels_per_nm * 2)
    
    # use tss_lanes if provided
    tss_start = time.time()
    if tss_lanes is not None:
        if tss_lanes.shape != is_water.shape:
            print("Warning: Provided TSS lanes mask has different shape than water mask; ignoring TSS lanes.")
        else:
            print("Applying provided TSS lanes mask to water mask.")
            # Make sure all lanes are marked as water in the mask (OR operation)
            is_water = is_water | tss_lanes
            print(f"  ✓ TSS lanes applied in {time.time() - tss_start:.2f}s")

    # Optionally apply TSS (turn specified lanes/zones into land) either before buffering
    # or after, depending on parameter.
    if tss_geojson_path and apply_tss_before_buffer:
        try:
            geojson_start = time.time()
            is_water = _apply_tss_to_mask(
                is_water,
                tss_geojson_path,
                land_lane_types=land_lane_types,
                water_lane_types=water_lane_types,
                lane_pixel_width=lane_pixel_width,
            )
            print(f"Applied TSS lanes/zones to mask (pre-buffer) in {time.time() - geojson_start:.2f}s")
        except Exception as e:
            print(f"Warning: Failed to apply TSS lanes before buffering: {e}")

    # is_water_corrected = preserve_critical_channels(is_water, critical_areas)
  
    # Try to use OpenCV for faster morphological operations
    try:
        import cv2
        use_opencv = True
        print("Using OpenCV for morphological operations (faster)")
    except ImportError:
        use_opencv = False
        print("OpenCV not available, using scipy (slower)")
    
    # Create land mask (invert is_water so True = land, False = water)
    land_mask = ~is_water

    if use_opencv:
        # OpenCV-based fast implementation
        morph_start = time.time()
        # Convert to uint8 for OpenCV
        land_uint8 = land_mask.astype(np.uint8)
        
        # Define threshold for small islands
        small_island_threshold = max(buffer_pixels / 5, 5)
        
        # Label connected components using OpenCV (much faster than scipy)
        label_start = time.time()
        num_features, labeled_lands = cv2.connectedComponents(land_uint8, connectivity=8)
        print(f"  ✓ Connected components labeled in {time.time() - label_start:.2f}s ({num_features} features)")
        
        # Calculate sizes
        land_sizes = np.bincount(labeled_lands.ravel())
        
        # Process large and small landmasses separately
        classify_start = time.time()
        
        # Fully vectorized classification (NO LOOPS - uses numpy broadcasting)
        # Create lookup table: for each label, is it large (255) or small (0)?
        # NOTE: Label 0 is background (water), so we exclude it from classification
        is_large = (land_sizes > small_island_threshold).astype(np.uint8) * 255
        is_small = ((land_sizes > 0) & (land_sizes <= small_island_threshold)).astype(np.uint8) * 255
        
        # CRITICAL FIX: Set label 0 (background/water) to 0 in both lookup tables
        is_large[0] = 0
        is_small[0] = 0
        
        # Use the labeled array as indices into lookup table (vectorized!)
        large_lands = is_large[labeled_lands]
        small_lands = is_small[labeled_lands]
        
        large_count = np.sum(land_sizes[1:] > small_island_threshold)  # Exclude label 0
        small_count = np.sum((land_sizes[1:] > 0) & (land_sizes[1:] <= small_island_threshold))
        print(f"  ✓ Land masses classified in {time.time() - classify_start:.2f}s ({large_count} large, {small_count} small)")
        
        # Skip dilation if buffer_pixels is 0 or too small
        if buffer_pixels < 3:
            print(f"  ⚠ Buffer size too small ({buffer_pixels} pixels), skipping dilation")
            final_mask = is_water.copy()
            print(f"Total morphological operations: {time.time() - morph_start:.2f}s")
        else:
            # Apply different buffer sizes using OpenCV morphological operations
            # Create circular kernels (more accurate than square)
            kernel_start = time.time()
            # Ensure kernel size is odd and at least 3
            main_size = max(3, buffer_pixels if buffer_pixels % 2 == 1 else buffer_pixels + 1)
            small_size = max(3, buffer_pixels // 2 if (buffer_pixels // 2) % 2 == 1 else (buffer_pixels // 2) + 1)
            
            main_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (main_size, main_size))
            small_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (small_size, small_size))
            print(f"  ✓ Kernels created in {time.time() - kernel_start:.2f}s (main: {main_size}x{main_size}, small: {small_size}x{small_size})")
            
            dilate_start = time.time()
            dilated_large = cv2.dilate(large_lands, main_kernel, iterations=1)
            dilated_small = cv2.dilate(small_lands, small_kernel, iterations=1)
            print(f"  ✓ Dilation completed in {time.time() - dilate_start:.2f}s")
            
            # Combine results
            combined_lands = (dilated_large > 0) | (dilated_small > 0)
            final_mask = ~combined_lands
            print(f"Total morphological operations: {time.time() - morph_start:.2f}s")
    else:
        # Fallback to scipy implementation
        labeled_lands, num_features = ndimage.label(land_mask)
        land_sizes = np.bincount(labeled_lands.ravel())
        
        # Define threshold for small islands
        small_island_threshold = max(buffer_pixels / 5, 5)
        
        # Process large and small landmasses separately
        large_lands = np.zeros_like(land_mask)
        small_lands = np.zeros_like(land_mask)
        
        for i in range(1, num_features + 1):
            if land_sizes[i] > small_island_threshold:
                large_lands |= (labeled_lands == i)
            else:
                small_lands |= (labeled_lands == i)
        
        # Create structuring elements
        main_structure = ndimage.iterate_structure(
            ndimage.generate_binary_structure(2, 1),
            buffer_pixels // 2
        )
        
        small_structure = ndimage.iterate_structure(
            ndimage.generate_binary_structure(2, 1),
            max(1, buffer_pixels // 4)
        )
        
        # Apply different buffer sizes
        dilated_large = ndimage.binary_dilation(large_lands, structure=main_structure)
        dilated_small = ndimage.binary_dilation(small_lands, structure=small_structure)
        
        # Combine results
        combined_lands = dilated_large | dilated_small
        final_mask = ~combined_lands

    # Apply TSS after buffering if requested so they are carved out without extra dilation
    if tss_geojson_path and not apply_tss_before_buffer:
        try:
            post_tss_start = time.time()
            final_mask = _apply_tss_to_mask(
                final_mask,
                tss_geojson_path,
                land_lane_types=land_lane_types,
                water_lane_types=water_lane_types,
                lane_pixel_width=lane_pixel_width,
                make_land=True,
            )
            print(f"Applied TSS lanes/zones to mask (post-buffer) in {time.time() - post_tss_start:.2f}s")
        except Exception as e:
            print(f"Warning: Failed to apply TSS lanes after buffering: {e}")
    
    total_time = time.time() - start_time
    print(f"✓ Buffered water mask created in {total_time:.2f}s")

    # Save to cache with compression
    try:
        save_start = time.time()
        np.savez_compressed(cache_file, mask=final_mask)
        print(f"Cached buffered water mask saved to {cache_file} in {time.time() - save_start:.2f}s")
    except Exception as e:
        print(f"Warning: Could not save cache file: {e}")
    
    return final_mask  # Return the water mask


def preserve_critical_channels(is_water, critical_regions):
    """
    Ensure critical water channels remain open in the water mask.
    
    Args:
        is_water (np.ndarray): The original water mask.
        critical_regions (list of tuples): List of regions to preserve, each defined as
                                            (x_min, x_max, y_min, y_max).
    
    Returns:
        np.ndarray: Updated water mask with critical regions preserved.
    """
    for x_min, x_max, y_min, y_max in critical_regions:
        is_water[y_min:y_max, x_min:x_max] = True  # Mark region as water
    return is_water


def _apply_tss_to_mask(
    is_water: np.ndarray,
    geojson_path: str,
    land_lane_types: list[str] | None = None,
    water_lane_types: list[str] | None = None,
    lane_pixel_width: int = 3,
    make_land: bool = True,
):
    """Overlay TSS features (lanes/zones) onto the water mask, converting them to land.

    Args:
        is_water: Boolean 2D array (True=water, False=land)
        geojson_path: Path to GeoJSON file containing features.
        land_lane_types: seamark:type values to target; defaults if None.
        lane_pixel_width: Thickness in pixels of line rasterization (roughly radius).
        make_land: If True, targeted features become land (False in mask). If False,
                   they become water (inverse operation, kept for flexibility).
    Returns:
        Modified mask (same array object mutated and returned for performance).
    """
  
    

    try:
        with open(geojson_path, "r", encoding="utf-8") as f:
            gj = json.load(f)
    except FileNotFoundError:
        print(f"TSS GeoJSON not found: {geojson_path}")
        return is_water
    except Exception as e:
        print(f"Error reading TSS GeoJSON {geojson_path}: {e}")
        return is_water

    features = gj.get("features", [])
    if not features:
        print("TSS GeoJSON has no features.")
        return is_water

    # Normalize type lists (avoid None checks later)
    land_lane_types = land_lane_types or []
    water_lane_types = water_lane_types or []

    # Active (possibly cropped) bounds for correct raster alignment
    try:
        from core.initialization import get_active_bounds as _gab
        lat_min, lat_max, lon_min, lon_max = _gab()
    except Exception:
        lat_min, lat_max, lon_min, lon_max = LAT_MIN, LAT_MAX, LON_MIN, LON_MAX
    lon_span = (lon_max - lon_min) or 1e-6
    lat_span = (lat_max - lat_min) or 1e-6

    grid_h, grid_w = is_water.shape

    def to_pixel(lon: float, lat: float):
        # Normalize into [-180,180]
        if lon > 180:
            lon -= 360
        elif lon < -180:
            lon += 360
        # Direct linear mapping into cropped grid
        x = int((lon - lon_min) / lon_span * grid_w)
        y = int((lat_max - lat) / lat_span * grid_h)
        return x, y

    drawn = 0
    skipped = 0

    # Pre-classify: collect land-first, then water overrides so that water can reopen.
    land_features: list[tuple[dict, bool]] = []  # (feature, feature_make_land)
    water_features: list[tuple[dict, bool]] = []

    debug_counts: dict[str, int] = {}

    for feat in features:
        props = feat.get("properties", {})
        seamark_type = None
        parsed = props.get("parsed_other_tags") or {}
        if isinstance(parsed, dict):
            seamark_type = parsed.get("seamark:type")
        # Fallbacks: direct key, common alternative key, or already extracted tag
        if not seamark_type:
            seamark_type = props.get("seamark:type") or props.get("seamark_type")
        if not seamark_type:
            raw = props.get("other_tags", "")
            if "seamark:type" in raw:
                try:
                    part = raw.split("seamark:type", 1)[1]
                    seamark_type = part.split("\"", 3)[2]
                except Exception:
                    seamark_type = None

        # Evaluate classification
        if seamark_type in land_lane_types:
            land_features.append((feat, make_land))  # usually True
            debug_counts[seamark_type] = debug_counts.get(seamark_type, 0) + 1
        elif seamark_type in water_lane_types:
            # Always force to water regardless of global make_land
            water_features.append((feat, False))
            debug_counts[seamark_type] = debug_counts.get(seamark_type, 0) + 1
        else:
            skipped += 1

    def rasterize_feature(feat: dict, feature_make_land: bool):
        nonlocal drawn, skipped
        geom = feat.get("geometry", {})
        gtype = geom.get("type")
        coords = geom.get("coordinates")
        if not coords:
            skipped += 1
            return
        try:
            if gtype == "LineString":
                closed_loop = False
                if coords and len(coords) >= 4:
                    (lon0, lat0) = coords[0]
                    (lone, late) = coords[-1]
                    if abs(lon0 - lone) < 1e-6 and abs(lat0 - late) < 1e-6:
                        closed_loop = True
                if closed_loop:
                    try:
                        from PIL import Image, ImageDraw
                        grid_h, grid_w = is_water.shape
                        img = Image.new("1", (grid_w, grid_h), 0)
                        draw = ImageDraw.Draw(img, "1")
                        poly_xy = [to_pixel(lon, lat) for lon, lat in coords]
                        draw.polygon(poly_xy, outline=1, fill=1)
                        poly_mask = np.array(img, dtype=bool)
                        if feature_make_land:
                            is_water[poly_mask] = False
                        else:
                            is_water[poly_mask] = True
                        drawn += 1
                    except Exception:
                        # Fallback: draw boundary and attempt manual fill if shapely available
                        _rasterize_line(is_water, coords, to_pixel, lane_pixel_width, feature_make_land, closed=True)
                        if Polygon is not None and Point is not None:
                            try:
                                poly = Polygon(coords)
                                if poly.is_valid and not poly.is_empty:
                                    # Compute bounding box in pixel space
                                    min_lon, min_lat, max_lon, max_lat = poly.bounds[0], poly.bounds[1], poly.bounds[2], poly.bounds[3]
                                    # Map bounds to pixels (clip)
                                    x0, y1 = to_pixel(min_lon, min_lat)  # note y is inverted
                                    x1, y0 = to_pixel(max_lon, max_lat)
                                    x_min = max(0, min(x0, x1))
                                    x_max = min(grid_w - 1, max(x0, x1))
                                    y_min = max(0, min(y0, y1))
                                    y_max = min(grid_h - 1, max(y0, y1))
                                    # Precompute inverse mapping factors
                                    for py in range(y_min, y_max + 1):
                                        # Pixel center to lat
                                        lat = lat_max - (py + 0.5) / grid_h * lat_span
                                        for px in range(x_min, x_max + 1):
                                            lon = lon_min + (px + 0.5) / grid_w * lon_span
                                            if poly.contains(Point(lon, lat)):
                                                if feature_make_land:
                                                    is_water[py, px] = False
                                                else:
                                                    is_water[py, px] = True
                            except Exception:
                                pass
                        drawn += 1
                else:
                    _rasterize_line(is_water, coords, to_pixel, lane_pixel_width, feature_make_land)
                    drawn += 1
            elif gtype == "MultiLineString":
                for line in coords:
                    _rasterize_line(is_water, line, to_pixel, lane_pixel_width, feature_make_land)
                drawn += 1
            elif gtype in ("Polygon", "MultiPolygon"):
                try:
                    from PIL import Image, ImageDraw
                    grid_h, grid_w = is_water.shape
                    poly_img = Image.new("1", (grid_w, grid_h), 0)
                    draw = ImageDraw.Draw(poly_img, "1")

                    def draw_one(poly):
                        if not poly:
                            return
                        exterior = poly[0]
                        ext_xy = [to_pixel(lon, lat) for lon, lat in exterior]
                        draw.polygon(ext_xy, outline=1, fill=1)
                        for hole in poly[1:]:
                            hole_xy = [to_pixel(lon, lat) for lon, lat in hole]
                            draw.polygon(hole_xy, outline=0, fill=0)

                    if gtype == "Polygon":
                        draw_one(coords)
                    else:  # MultiPolygon
                        for poly in coords:
                            draw_one(poly)

                    poly_mask = np.array(poly_img, dtype=bool)
                    if feature_make_land:
                        is_water[poly_mask] = False
                    else:
                        is_water[poly_mask] = True
                    drawn += 1
                except Exception:
                    if gtype == "Polygon":
                        rings = [coords[0]] if coords else []
                    else:
                        rings = [poly[0] for poly in coords if poly]
                    for ring in rings:
                        _rasterize_line(is_water, ring, to_pixel, lane_pixel_width, feature_make_land, closed=True)
                    drawn += 1
            else:
                skipped += 1
        except Exception as e:
            print(f"Error rasterizing TSS feature: {e}")
            skipped += 1

    # Pass 1: land features
    for feat, feature_make_land in land_features:
        rasterize_feature(feat, feature_make_land)
    # Pass 2: water overrides (re-open / retain water)
    for feat, feature_make_land in water_features:
        rasterize_feature(feat, feature_make_land)

    if debug_counts:
        summary = ", ".join(f"{k}:{v}" for k, v in sorted(debug_counts.items()))
    else:
        summary = "none"
    print(
        f"TSS overlay: drew {drawn} targeted features (land first, water overrides last); classified counts [{summary}]; skipped {skipped} others."
    )
    return is_water


def _rasterize_line(mask: np.ndarray, line_coords, to_pixel_fn, width: int, make_land: bool, closed: bool = False):
    """Rasterize a geodesic LineString into the boolean mask.

    Simple pixel connection using linear interpolation between successive points.
    Width acts as a square (Chebyshev) radius. For routing purposes this is
    sufficient and fast; can be replaced with more accurate shapely buffering
    later if needed.
    """
    h, w = mask.shape
    if width < 1:
        width = 1
    pts = line_coords
    if closed and pts and pts[0] != pts[-1]:
        pts = pts + [pts[0]]
    last_xy = None
    for lon, lat in pts:
        x, y = to_pixel_fn(lon, lat)
        if 0 <= x < w and 0 <= y < h:
            _draw_disk(mask, x, y, width, make_land)
        if last_xy is not None:
            lx, ly = last_xy
            dx = x - lx
            dy = y - ly
            steps = max(abs(dx), abs(dy), 1)
            for i in range(1, steps + 1):
                xi = lx + int(round(dx * i / steps))
                yi = ly + int(round(dy * i / steps))
                if 0 <= xi < w and 0 <= yi < h:
                    _draw_disk(mask, xi, yi, width, make_land)
        last_xy = (x, y)


def _draw_disk(mask: np.ndarray, x: int, y: int, radius: int, make_land: bool):
    r = radius
    x_min = max(0, x - r)
    x_max = min(mask.shape[1] - 1, x + r)
    y_min = max(0, y - r)
    y_max = min(mask.shape[0] - 1, y + r)
    if make_land:
        mask[y_min:y_max+1, x_min:x_max+1] = False
    else:
        mask[y_min:y_max+1, x_min:x_max+1] = True