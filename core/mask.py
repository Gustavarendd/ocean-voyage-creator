"""Water mask creation and processing functions.

Includes optional integration of Traffic Separation Scheme (TSS) features and a
coastal buffer in nautical miles, correctly calibrated to pixel radius based on
the current active map bounds and resolution.
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

# Try importing OpenCV globally to avoid unbound warnings in certain analyzers
try:  # type: ignore
    import cv2  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    cv2 = None  # type: ignore


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
    tss_lanes: np.ndarray | None = None,
):
    """Create a water mask with a coastal buffer zone.

    Calibrates buffer_nm to an exact pixel radius using the current active
    geographic bounds and output resolution.
    """
    # Cache setup
    cache_dir = "cache"
    os.makedirs(cache_dir, exist_ok=True)
    cache_key = _create_cache_key(is_water, buffer_nm)

    if tss_geojson_path and os.path.exists(tss_geojson_path):
        try:
            with open(tss_geojson_path, "rb") as f:
                import hashlib as _hl
                tss_hash = _hl.md5(f.read()).hexdigest()[:8]
            cache_key += f"_tss{tss_hash}_w{lane_pixel_width}_pre{int(apply_tss_before_buffer)}"
        except Exception:
            pass
    if tss_lanes is not None:
        tss_lanes_hash = hashlib.md5(tss_lanes.tobytes()).hexdigest()[:8]
        cache_key += f"_tssl{tss_lanes_hash}"

    cache_file = os.path.join(cache_dir, f"buffered_water_{cache_key}.npy")
    if not force_recompute and os.path.exists(cache_file):
        print(f"Loading cached buffered water mask from {cache_file}")
        try:
            return np.load(cache_file)
        except Exception:
            print("Cache load failed; recomputing...")

    print("Force recompute requested - computing buffered water mask..." if force_recompute else "Computing buffered water mask (this may take a moment)...")
    start_time = time.time()

    # Active bounds and pixel scale
    try:
        from core.initialization import get_active_bounds as _gab
        lat_min, lat_max, lon_min, lon_max = _gab()
    except Exception:
        lat_min, lat_max, lon_min, lon_max = LAT_MIN, LAT_MAX, LON_MIN, LON_MAX
    width_px = is_water.shape[1]
    pixels_per_degree = width_px / (lon_max - lon_min)
    pixels_per_nm = pixels_per_degree / 60.0
    buffer_radius_px = int(round(buffer_nm * pixels_per_nm))
    print(f"Coastal buffer: {buffer_nm} nm -> {buffer_radius_px} px (px/°={pixels_per_degree:.2f}, px/nm={pixels_per_nm:.2f})")

    # Integrate TSS lanes (as water) if provided
    if tss_lanes is not None and tss_lanes.shape == is_water.shape:
        t0 = time.time()
        is_water = is_water | tss_lanes
        print(f"  ✓ TSS lanes applied in {time.time() - t0:.2f}s")
    elif tss_lanes is not None:
        print("Warning: Provided TSS lanes mask has different shape than water mask; ignoring TSS lanes.")

    # Apply TSS (make land/water) before buffering if requested
    if tss_geojson_path and apply_tss_before_buffer:
        try:
            t0 = time.time()
            is_water = _apply_tss_to_mask(
                is_water,
                tss_geojson_path,
                land_lane_types=land_lane_types,
                water_lane_types=water_lane_types,
                lane_pixel_width=lane_pixel_width,
            )
            print(f"Applied TSS (pre-buffer) in {time.time() - t0:.2f}s")
        except Exception as e:
            print(f"Warning: Failed to apply TSS lanes before buffering: {e}")

    # Invert mask for dilation (land=True)
    land_mask = ~is_water

    # Apply coastal buffer by dilating land by a radius of buffer_radius_px pixels
    if cv2 is not None:
        print("Using OpenCV for morphological operations (faster)")
    else:
        print("OpenCV not available, using scipy (slower)")

    if buffer_radius_px < 1:
        print("  ⚠ Buffer radius < 1 px, skipping dilation")
        final_mask = is_water.copy()
    else:
        if cv2 is not None:
            t0 = time.time()
            land_uint8 = land_mask.astype(np.uint8)
            ksize = 2 * buffer_radius_px + 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
            dilated = cv2.dilate(land_uint8, kernel, iterations=1)
            final_mask = ~ (dilated.astype(bool))
            print(f"  ✓ Dilation completed in {time.time() - t0:.2f}s (kernel {ksize}x{ksize})")
        else:
            # Build a circular (disk) footprint for SciPy dilation
            r = int(buffer_radius_px)
            yy, xx = np.ogrid[-r:r+1, -r:r+1]
            disk = (xx*xx + yy*yy) <= r*r
            dilated = ndimage.binary_dilation(land_mask, structure=disk)
            final_mask = ~dilated

    # Apply TSS after buffering (optional carve-out)
    if tss_geojson_path and not apply_tss_before_buffer:
        try:
            t0 = time.time()
            final_mask = _apply_tss_to_mask(
                final_mask,
                tss_geojson_path,
                land_lane_types=land_lane_types,
                water_lane_types=water_lane_types,
                lane_pixel_width=lane_pixel_width,
                make_land=True,
            )
            print(f"Applied TSS (post-buffer) in {time.time() - t0:.2f}s")
        except Exception as e:
            print(f"Warning: Failed to apply TSS lanes after buffering: {e}")

    print(f"✓ Buffered water mask created in {time.time() - start_time:.2f}s")
    try:
        np.save(cache_file, final_mask)
        print(f"Cached buffered water mask saved to {cache_file}")
    except Exception as e:
        print(f"Warning: Could not save cache file: {e}")
    return final_mask


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
        # Ensure grid dimensions are available for all branches
        grid_h, grid_w = is_water.shape
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