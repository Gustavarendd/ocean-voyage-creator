"""Water mask creation and processing functions.

Enhancements:
        - Optional integration of TSS (Traffic Separation Scheme) data from a
            GeoJSON file (e.g. `separation_lanes_with_direction.geojson`).
            Any features whose `seamark:type` property matches one of the provided
            lane types (default: separation_zone, separation_lane) are rasterized
            onto the water mask and converted to land so that routing will avoid
            crossing them.
"""

import numpy as np
import os
import hashlib
from scipy import ndimage
from config import IMAGE_WIDTH, IMAGE_HEIGHT, CRITICAL_REGIONS, LON_MIN, LON_MAX, LAT_MIN, LAT_MAX
import json

from utils.coordinates import get_active_bounds

try:
    from shapely.geometry import LineString
except Exception:  # Shapely is already a dependency (used in config) but guard anyway
    LineString = None


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
    cache_files = glob.glob(os.path.join(cache_dir, "buffered_water_*.npy"))
    
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
    supersample_factor: int = 1,
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
    supersample_factor: Temporarily upscale the mask by this integer factor
                before buffering to achieve a smoother, higher
                resolution coastline buffer. Down-sampled back to
                original resolution at the end. Memory cost grows
                roughly with factor^2. Typical values: 1 (off), 2 or 3.
    """
    # Create cache directory if it doesn't exist
    cache_dir = "cache"
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    
    # Create a hash of the input parameters to use as cache key
    cache_key = _create_cache_key(is_water, buffer_nm)
    if supersample_factor > 1:
        cache_key += f"_ss{supersample_factor}"
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
    cache_file = os.path.join(cache_dir, f"buffered_water_{cache_key}.npy")
    
    # Check if cached version exists and force_recompute is False
    if not force_recompute and os.path.exists(cache_file):
        print(f"Loading cached buffered water mask from {cache_file}")
        try:
            cached_mask = np.load(cache_file)
            print(f"Successfully loaded cached mask with shape: {cached_mask.shape}")
            return cached_mask
        except Exception as e:
            print(f"Error loading cached mask: {e}. Recomputing...")
    
    if force_recompute:
        print("Force recompute requested - computing buffered water mask...")
    else:
        print("Computing buffered water mask (this may take a moment)...")

    try:
        from core.initialization import get_active_bounds as _gab
        lat_min, lat_max, lon_min, lon_max = _gab()
    except Exception:
        lat_min, lat_max, lon_min, lon_max = LAT_MIN, LAT_MAX, LON_MIN, LON_MAX
    
    # Calculate buffer size in pixels (base grid), then scale if supersampling
    base_pixels_per_nm = (IMAGE_WIDTH / (lon_max - lon_min)) / 60
    buffer_pixels = int(buffer_nm * base_pixels_per_nm * 2)
    if supersample_factor > 1:
        buffer_pixels *= supersample_factor

    # Supersample (nearest neighbor) for higher resolution buffering
    if supersample_factor > 1:
        is_water = np.repeat(np.repeat(is_water, supersample_factor, axis=0), supersample_factor, axis=1)
        # Scale critical regions for high-res operations
        scaled_critical = [
            (
                x_min * supersample_factor,
                x_max * supersample_factor,
                y_min * supersample_factor,
                y_max * supersample_factor,
            )
            for (x_min, x_max, y_min, y_max) in CRITICAL_REGIONS
        ]
    else:
        scaled_critical = CRITICAL_REGIONS

    # Optionally apply TSS (turn specified lanes/zones into land) either before buffering
    # or after, depending on parameter.
    if tss_geojson_path and apply_tss_before_buffer:
        try:
            is_water = _apply_tss_to_mask(
                is_water,
                tss_geojson_path,
                land_lane_types=land_lane_types,
                water_lane_types=water_lane_types,
                lane_pixel_width=lane_pixel_width,
            )
            print("Applied TSS lanes/zones to mask (pre-buffer).")
        except Exception as e:
            print(f"Warning: Failed to apply TSS lanes before buffering: {e}")

    is_water_corrected = preserve_critical_channels(is_water, scaled_critical)
    
    # Create structuring elements
    main_structure = ndimage.iterate_structure(
        ndimage.generate_binary_structure(2, 1),
        buffer_pixels // 2
    )
    
    # Create land mask
    land_mask = ~is_water_corrected
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
    
    # Identify narrow water channels
    water_structure = ndimage.iterate_structure(
        ndimage.generate_binary_structure(2, 1),
        max(1, buffer_pixels // 8)
    )
    narrow_water = ndimage.binary_erosion(is_water_corrected, structure=water_structure)
    preserved_channels = is_water_corrected & narrow_water  # Preserve narrow water channels
    
    # Apply different buffer sizes
    dilated_large = ndimage.binary_dilation(large_lands, structure=main_structure)
    
    small_structure = ndimage.iterate_structure(
        ndimage.generate_binary_structure(2, 1),
        max(1, buffer_pixels // 4)
    )
    dilated_small = ndimage.binary_dilation(small_lands, structure=small_structure)
    
    # Combine results
    combined_lands = dilated_large | dilated_small
    
    # Ensure preserved channels remain open
    final_mask = ~combined_lands | preserved_channels

    # Apply TSS after buffering if requested so they are carved out without extra dilation
    if tss_geojson_path and not apply_tss_before_buffer:
        try:
            final_mask = _apply_tss_to_mask(
                final_mask,
                tss_geojson_path,
                land_lane_types=land_lane_types,
                water_lane_types=water_lane_types,
                lane_pixel_width=lane_pixel_width,
                make_land=True,
            )
            print("Applied TSS lanes/zones to mask (post-buffer).")
        except Exception as e:
            print(f"Warning: Failed to apply TSS lanes after buffering: {e}")
    
    # Downsample back if supersampled
    if supersample_factor > 1:
        H_hr, W_hr = final_mask.shape
        H = H_hr // supersample_factor
        W = W_hr // supersample_factor
        # Majority pooling to retain navigable water where >50% of high-res pixels are water
        final_mask = (
            final_mask.reshape(H, supersample_factor, W, supersample_factor)
            .mean(axis=(1, 3))
            > 0.5
        )

    # Save to cache
    try:
        np.save(cache_file, final_mask)
        print(f"Cached buffered water mask saved to {cache_file}")
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

    for feat in features:
        props = feat.get("properties", {})
        seamark_type = None
        parsed = props.get("parsed_other_tags") or {}
        if isinstance(parsed, dict):
            seamark_type = parsed.get("seamark:type")
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
        elif seamark_type in water_lane_types:
            # Always force to water regardless of global make_land
            water_features.append((feat, False))
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
                        _rasterize_line(is_water, coords, to_pixel, lane_pixel_width, feature_make_land, closed=True)
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

    print(
        f"TSS overlay: drew {drawn} targeted features (land first, water overrides last); skipped {skipped} others."
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