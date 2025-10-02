"""Precompute a pixel mask for Traffic Separation Scheme (TSS) lanes.

This speeds up TSS preference during A* search by avoiding per-node GeoJSON
distance checks. Each pixel that lies on (or close to) a TSS LineString is
flagged True. The mask can then be used for a simple O(1) membership test.

Implementation notes:
 - We iterate through all GeoJSON files under TSS_by_direction.
 - Only LineString geometries are considered.
 - Coordinates are converted to pixel (x, y) using existing converters.
 - For every segment we interpolate along the dominant axis so the line is
   continuous (simple Bresenham-style sampling) and mark pixels True.
 - Optional dilation_radius lets us slightly widen the lane to make route
   attraction stronger without requiring pixel-perfect alignment.
"""

from __future__ import annotations

import json
import os
from typing import Iterable, Tuple
import hashlib
from datetime import datetime

import numpy as np
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

from utils.coordinates import latlon_to_pixel, validate_coordinates
try:
    from core.initialization import (
        ACTIVE_LAT_MIN, ACTIVE_LAT_MAX, ACTIVE_LON_MIN, ACTIVE_LON_MAX
    )
except Exception:
    ACTIVE_LAT_MIN, ACTIVE_LAT_MAX = -90.0, 90.0
    ACTIVE_LON_MIN, ACTIVE_LON_MAX = -180.0, 180.0


def _iter_tss_files(root_dir: str) -> Iterable[str]:
    by_dir = os.path.join(root_dir, "TSS_by_direction")
    if not os.path.isdir(by_dir):
        return []
    for name in os.listdir(by_dir):
        if name.endswith('.geojson') and name.startswith('TSS_Lanes_'):
            yield os.path.join(by_dir, name)


def _rasterize_linestring(pixel_points: list[Tuple[int, int]], 
                          mask: np.ndarray, 
                          vecs: np.ndarray):
    """Rasterize polyline into mask + vector field.

    For each interpolated pixel, mark mask=True and store unit direction vector.
    """
    h, w = mask.shape
    for (x1, y1), (x2, y2) in zip(pixel_points[:-1], pixel_points[1:]):
        dx = x2 - x1
        dy = y2 - y1
        steps = max(abs(dx), abs(dy))
        if steps == 0:
            if 0 <= y1 < h and 0 <= x1 < w:
                mask[y1, x1] = True
                # undefined direction → leave as (0,0)
            continue

        length = (dx**2 + dy**2) ** 0.5
        if length == 0:
            continue
        u, v = dx / length, dy / length

        for i in range(steps + 1):
            x = int(round(x1 + dx * i / steps))
            y = int(round(y1 + dy * i / steps))
            if 0 <= y < h and 0 <= x < w:
                mask[y, x] = True
                vecs[y, x] = (u, v)


def _dilate(mask: np.ndarray, vecs: np.ndarray, radius: int) -> Tuple[np.ndarray, np.ndarray]:
    """Dilate mask by radius, propagating nearest vector values."""
    if radius <= 0:
        return mask, vecs
    h, w = mask.shape
    dilated = mask.copy()
    vecs_dilated = vecs.copy()
    yy, xx = np.nonzero(mask)
    for y, x in zip(yy, xx):
        u, v = vecs[y, x]
        y_min = max(0, y - radius)
        y_max = min(h - 1, y + radius)
        x_min = max(0, x - radius)
        x_max = min(w - 1, x + radius)
        for yy2 in range(y_min, y_max + 1):
            for xx2 in range(x_min, x_max + 1):
                if (yy2 - y) ** 2 + (xx2 - x) ** 2 <= radius ** 2:
                    dilated[yy2, xx2] = True
                    # assign direction if not already set
                    if vecs_dilated[yy2, xx2, 0] == 0 and vecs_dilated[yy2, xx2, 1] == 0:
                        vecs_dilated[yy2, xx2] = (u, v)
    return dilated, vecs_dilated


def _is_closed_polygon(pixel_points: list[Tuple[int, int]], tolerance: int = 2) -> bool:
    """Check if a sequence of pixel points forms a closed polygon.
    
    Args:
        pixel_points: List of (x, y) pixel coordinates
        tolerance: Maximum pixel distance to consider start/end as "closed"
    
    Returns:
        True if the first and last points are within tolerance distance
    """
    if len(pixel_points) < 3:
        return False
    
    start = pixel_points[0]
    end = pixel_points[-1]
    
    # Check if start and end are the same or very close
    dx = abs(start[0] - end[0])
    dy = abs(start[1] - end[1])
    
    return dx <= tolerance and dy <= tolerance


def _fill_polygon(pixel_points: list[Tuple[int, int]], mask: np.ndarray) -> None:
    """Fill a closed polygon in the mask.
    
    Uses OpenCV's fillPoly if available, otherwise uses a simple scan-line fill.
    
    Args:
        pixel_points: List of (x, y) pixel coordinates forming a closed polygon
        mask: Boolean mask to fill
    """
    if len(pixel_points) < 3:
        return
    
    h, w = mask.shape
    
    if HAS_CV2:
        # Use OpenCV for efficient polygon filling
        # Create a temporary uint8 array for cv2
        import cv2 as cv2_local
        temp_mask = np.zeros((h, w), dtype=np.uint8)
        pts = np.array(pixel_points, dtype=np.int32).reshape((-1, 1, 2))
        cv2_local.fillPoly(temp_mask, [pts], (1,))
        # Merge with existing mask
        mask |= temp_mask.astype(bool)
    else:
        # Fallback: simple scan-line fill algorithm
        # This is a basic implementation - not as efficient as cv2
        from PIL import Image, ImageDraw
        
        # Create a temporary PIL image
        img = Image.new('L', (w, h), 0)
        draw = ImageDraw.Draw(img)
        
        # Draw filled polygon
        draw.polygon(pixel_points, fill=1, outline=1)
        
        # Convert back to numpy and update mask
        arr = np.array(img, dtype=bool)
        mask |= arr


def _downsample_mask(mask_hi: np.ndarray, factor: int) -> np.ndarray:
    """Downsample a boolean mask using max pooling.
    
    Any True value in a factor x factor block results in True in output.
    
    Args:
        mask_hi: High-resolution boolean mask
        factor: Downsampling factor
    
    Returns:
        Downsampled boolean mask
    """
    h, w = mask_hi.shape
    h_out = h // factor
    w_out = w // factor
    
    # Trim to multiple of factor
    mask_trimmed = mask_hi[:h_out * factor, :w_out * factor]
    
    # Reshape and apply max pooling
    mask_reshaped = mask_trimmed.reshape(h_out, factor, w_out, factor)
    mask_out = mask_reshaped.any(axis=(1, 3))
    
    return mask_out


def _downsample_vectors(vecs_hi: np.ndarray, factor: int) -> np.ndarray:
    """Downsample a vector field using average pooling.
    
    Averages vectors in each factor x factor block, preserving direction.
    
    Args:
        vecs_hi: High-resolution vector field (H, W, 2)
        factor: Downsampling factor
    
    Returns:
        Downsampled vector field
    """
    h, w, _ = vecs_hi.shape
    h_out = h // factor
    w_out = w // factor
    
    # Trim to multiple of factor
    vecs_trimmed = vecs_hi[:h_out * factor, :w_out * factor, :]
    
    # Reshape and apply mean pooling
    vecs_reshaped = vecs_trimmed.reshape(h_out, factor, w_out, factor, 2)
    vecs_out = vecs_reshaped.mean(axis=(1, 3))
    
    # Renormalize non-zero vectors
    magnitudes = np.linalg.norm(vecs_out, axis=2, keepdims=True)
    mask_nonzero = magnitudes[..., 0] > 1e-6
    vecs_out[mask_nonzero] /= magnitudes[mask_nonzero]
    
    return vecs_out.astype(np.float32)


def _hash_tss_sources(files: list[str], dilation_radius: int, width: int, height: int) -> str:
    """Create a stable short hash representing all inputs that affect the mask.

    Hash components:
      - File paths + modified timestamps + file size
      - MD5 of concatenated file contents (efficient enough for typical file sizes)
      - Image dimensions & dilation radius
      - Active geographic bounds (cropping)
    """
    h = hashlib.md5()
    meta_parts: list[str] = []
    for p in sorted(files):
        try:
            stat = os.stat(p)
            meta_parts.append(f"{p}:{int(stat.st_mtime)}:{stat.st_size}")
        except OSError:
            meta_parts.append(f"{p}:missing:0")
    meta_parts.append(f"W{width}H{height}D{dilation_radius}")
    meta_parts.append(f"LAT{ACTIVE_LAT_MIN:.3f}_{ACTIVE_LAT_MAX:.3f}_LON{ACTIVE_LON_MIN:.3f}_{ACTIVE_LON_MAX:.3f}")
    meta_blob = "|".join(meta_parts).encode()
    h.update(meta_blob)
    # Content hash (optional, only if few files & total size reasonable)
    for p in sorted(files):
        try:
            with open(p, "rb") as f:
                h.update(f.read())
        except Exception:
            continue
    return h.hexdigest()[:16]


def build_tss_mask(
    image_width: int,
    image_height: int,
    root_dir: str = '.',
    dilation_radius: int = 2,
    use_cache: bool = True,
    cache_dir: str = "cache",
    force_recompute: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build (or load cached) boolean mask + vector field for TSS lanes.

    Args:
        image_width / image_height: Grid dimensions.
        root_dir: Project root containing TSS_by_direction.
        dilation_radius: Pixel radius to dilate lanes.
        use_cache: Enable disk cache (default True).
        cache_dir: Directory to store cache .npz.
        force_recompute: Ignore cache and rebuild.
    Returns:
        (mask, vecs)
    """
    files = list(_iter_tss_files(root_dir))
    if not files:
        print("TSS mask: no TSS_by_direction geojson files found")
        return (np.zeros((image_height, image_width), dtype=bool),
                np.zeros((image_height, image_width, 2), dtype=np.float32))

    os.makedirs(cache_dir, exist_ok=True)
    hash_id = _hash_tss_sources(files, dilation_radius, image_width, image_height)
    cache_path = os.path.join(cache_dir, f"tss_mask_{hash_id}.npz")

    if use_cache and not force_recompute and os.path.isfile(cache_path):
        try:
            data = np.load(cache_path)
            mask = data['mask']
            vecs = data['vecs']
            if mask.shape == (image_height, image_width) and vecs.shape == (image_height, image_width, 2):
                print(f"TSS mask: loaded from cache {os.path.basename(cache_path)}")
                return mask, vecs
            else:
                print("TSS mask: cache shape mismatch; recomputing...")
        except Exception as e:
            print(f"TSS mask: failed to load cache ({e}); recomputing...")

    # Build from scratch
    mask = np.zeros((image_height, image_width), dtype=bool)
    vecs = np.zeros((image_height, image_width, 2), dtype=np.float32)
    count_features = 0
    for path in files:
        try:
            with open(path, 'r') as f:
                data = json.load(f)
        except Exception as e:
            print(f"TSS mask: failed to load {path}: {e}")
            continue
        for feature in data.get('features', []):
            geom = feature.get('geometry', {})
            if geom.get('type') != 'LineString':
                continue
            coords = geom.get('coordinates') or []
            pixel_coords = []
            for lon, lat in coords:
                if not (ACTIVE_LAT_MIN <= lat <= ACTIVE_LAT_MAX and ACTIVE_LON_MIN <= lon <= ACTIVE_LON_MAX):
                    continue
                try:
                    x, y = latlon_to_pixel(lat, lon, warn=False)
                except Exception:
                    continue
                if 0 <= x < image_width and 0 <= y < image_height:
                    pixel_coords.append((x, y))
            if len(pixel_coords) >= 2:
                _rasterize_linestring(pixel_coords, mask, vecs)
                count_features += 1

    if dilation_radius > 0:
        mask, vecs = _dilate(mask, vecs, dilation_radius)

    print(f"TSS mask: rasterized {count_features} lane features across {len(files)} files")

    if use_cache:
        try:
            np.savez_compressed(cache_path, mask=mask, vecs=vecs, built=str(datetime.utcnow()))
            print(f"TSS mask: cached to {cache_path}")
        except Exception as e:
            print(f"TSS mask: warning could not write cache ({e})")
    return mask, vecs


def build_tss_combined_mask(
    image_width: int,
    image_height: int,
    root_dir: str = '.',
    dilation_radius: int = 2,
    no_go_dilation: int = 3,
    supersample_factor: int = 1,
    use_cache: bool = True,
    cache_dir: str = "cache",
    force_recompute: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build mask combining separation lanes (with vectors) and no-go areas.

    This function processes separation_lanes_with_direction.geojson and creates:
    - Separation lanes: marked with direction vectors (navigable, preferred)
    - All other features: marked as no-go areas (obstacles for A* to avoid)

    Args:
        image_width / image_height: Grid dimensions (output size).
        root_dir: Project root containing TSS directory.
        dilation_radius: Pixel radius to dilate separation lanes (in supersampled space).
        no_go_dilation: Pixel radius to dilate no-go areas (in supersampled space).
        supersample_factor: Resolution multiplier (e.g., 2 = 2x resolution, 4 = 4x).
        use_cache: Enable disk cache (default True).
        cache_dir: Directory to store cache .npz.
        force_recompute: Ignore cache and rebuild.

    Returns:
        Tuple of (lanes_mask, lanes_vecs, no_go_mask):
        - lanes_mask: Boolean array marking separation lane pixels
        - lanes_vecs: Float array (H, W, 2) with direction vectors for lanes
        - no_go_mask: Boolean array marking areas to avoid (obstacles)
    """
    geojson_path = os.path.join(root_dir, "TSS", "separation_lanes_with_direction.geojson")
    
    if not os.path.isfile(geojson_path):
        print(f"TSS combined mask: file not found: {geojson_path}")
        return (
            np.zeros((image_height, image_width), dtype=bool),
            np.zeros((image_height, image_width, 2), dtype=np.float32),
            np.zeros((image_height, image_width), dtype=bool),
        )

    os.makedirs(cache_dir, exist_ok=True)
    
    # Create hash based on file content and parameters
    try:
        stat = os.stat(geojson_path)
        file_meta = f"{geojson_path}:{int(stat.st_mtime)}:{stat.st_size}"
    except OSError:
        file_meta = f"{geojson_path}:missing:0"
    
    h = hashlib.md5()
    meta_parts = [
        file_meta,
        f"W{image_width}H{image_height}D{dilation_radius}NGD{no_go_dilation}SS{supersample_factor}",
        f"LAT{ACTIVE_LAT_MIN:.3f}_{ACTIVE_LAT_MAX:.3f}_LON{ACTIVE_LON_MIN:.3f}_{ACTIVE_LON_MAX:.3f}",
        "v3_supersample"  # Version marker for cache invalidation
    ]
    h.update("|".join(meta_parts).encode())
    try:
        with open(geojson_path, "rb") as f:
            h.update(f.read())
    except Exception:
        pass
    hash_id = h.hexdigest()[:16]
    
    cache_path = os.path.join(cache_dir, f"tss_combined_{hash_id}.npz")

    # Try loading from cache
    if use_cache and not force_recompute and os.path.isfile(cache_path):
        try:
            data = np.load(cache_path)
            lanes_mask = data['lanes_mask']
            lanes_vecs = data['lanes_vecs']
            no_go_mask = data['no_go_mask']
            if (lanes_mask.shape == (image_height, image_width) and 
                lanes_vecs.shape == (image_height, image_width, 2) and
                no_go_mask.shape == (image_height, image_width)):
                print(f"TSS combined mask: loaded from cache {os.path.basename(cache_path)}")
                return lanes_mask, lanes_vecs, no_go_mask
            else:
                print("TSS combined mask: cache shape mismatch; recomputing...")
        except Exception as e:
            print(f"TSS combined mask: failed to load cache ({e}); recomputing...")

    # Build from scratch with supersampling
    supersample_factor = max(1, int(supersample_factor))  # Ensure at least 1
    
    # Create high-resolution masks
    hi_width = image_width * supersample_factor
    hi_height = image_height * supersample_factor
    
    print(f"TSS combined mask: building at {supersample_factor}x resolution ({hi_width}x{hi_height})")
    
    lanes_mask_hi = np.zeros((hi_height, hi_width), dtype=bool)
    lanes_vecs_hi = np.zeros((hi_height, hi_width, 2), dtype=np.float32)
    no_go_mask_hi = np.zeros((hi_height, hi_width), dtype=bool)
    
    # Temporarily scale the coordinate conversion for higher resolution
    from utils.coordinates import latlon_to_pixel
    import utils.coordinates as coords_module
    
    # Store original dimensions
    original_width = getattr(coords_module, 'IMAGE_WIDTH', image_width)
    original_height = getattr(coords_module, 'IMAGE_HEIGHT', image_height)
    
    # Temporarily override for supersampled space
    coords_module.IMAGE_WIDTH = hi_width
    coords_module.IMAGE_HEIGHT = hi_height
    
    try:
        with open(geojson_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"TSS combined mask: failed to load {geojson_path}: {e}")
        # Restore original dimensions
        coords_module.IMAGE_WIDTH = original_width
        coords_module.IMAGE_HEIGHT = original_height
        return (
            np.zeros((image_height, image_width), dtype=bool),
            np.zeros((image_height, image_width, 2), dtype=np.float32),
            np.zeros((image_height, image_width), dtype=bool),
        )

    count_lanes = 0
    count_no_go = 0

    for feature in data.get('features', []):
        geom = feature.get('geometry', {})
        props = feature.get('properties', {})
        
        # Determine feature type
        seamark_type = None
        
        # Check in direct properties first
        if 'seamark:type' in props:
            seamark_type = props['seamark:type']
        
        # Check in parsed_other_tags
        if seamark_type is None and 'parsed_other_tags' in props:
            parsed = props['parsed_other_tags']
            if isinstance(parsed, dict) and 'seamark:type' in parsed:
                seamark_type = parsed['seamark:type']
        
        # Check in other_tags string
        if seamark_type is None and 'other_tags' in props:
            other_tags = props['other_tags']
            if isinstance(other_tags, str) and 'seamark:type' in other_tags:
                # Parse "seamark:type"=>"value"
                import re
                match = re.search(r'"seamark:type"=>"([^"]+)"', other_tags)
                if match:
                    seamark_type = match.group(1)
        
        if seamark_type is None:
            # Default to no-go if type is unclear
            seamark_type = "unknown"
        
        # Process geometry based on type
        is_separation_lane = seamark_type == "separation_lane"
        
        if geom.get('type') not in ['LineString', 'MultiLineString', 'Polygon', 'MultiPolygon']:
            continue
        
        # Extract coordinate sequences
        coord_sequences = []
        if geom['type'] == 'LineString':
            coord_sequences = [geom.get('coordinates', [])]
        elif geom['type'] == 'MultiLineString':
            coord_sequences = geom.get('coordinates', [])
        elif geom['type'] == 'Polygon':
            coord_sequences = geom.get('coordinates', [])  # Include all rings
        elif geom['type'] == 'MultiPolygon':
            for polygon in geom.get('coordinates', []):
                coord_sequences.extend(polygon)
        
        for coords in coord_sequences:
            if not coords:
                continue
                
            pixel_coords = []
            for lon, lat in coords:
                if not (ACTIVE_LAT_MIN <= lat <= ACTIVE_LAT_MAX and 
                       ACTIVE_LON_MIN <= lon <= ACTIVE_LON_MAX):
                    continue
                try:
                    x, y = latlon_to_pixel(lat, lon, warn=False)
                except Exception:
                    continue
                if 0 <= x < hi_width and 0 <= y < hi_height:
                    pixel_coords.append((x, y))
            
            if len(pixel_coords) >= 2:
                if is_separation_lane:
                    _rasterize_linestring(pixel_coords, lanes_mask_hi, lanes_vecs_hi)
                    count_lanes += 1
                else:
                    # Check if this is a closed polygon (start and end are the same)
                    if _is_closed_polygon(pixel_coords, tolerance=2 * supersample_factor):
                        # Fill the entire enclosed area as no-go
                        _fill_polygon(pixel_coords, no_go_mask_hi)
                        count_no_go += 1
                    else:
                        # Rasterize as no-go line/boundary (no direction needed)
                        temp_vecs = np.zeros((hi_height, hi_width, 2), dtype=np.float32)
                        _rasterize_linestring(pixel_coords, no_go_mask_hi, temp_vecs)
                        count_no_go += 1

    # Restore original dimensions
    coords_module.IMAGE_WIDTH = original_width
    coords_module.IMAGE_HEIGHT = original_height

    # Apply dilation at high resolution
    if dilation_radius > 0:
        lanes_mask_hi, lanes_vecs_hi = _dilate(lanes_mask_hi, lanes_vecs_hi, dilation_radius)
    
    if no_go_dilation > 0:
        temp_vecs = np.zeros((hi_height, hi_width, 2), dtype=np.float32)
        no_go_mask_hi, _ = _dilate(no_go_mask_hi, temp_vecs, no_go_dilation)

    print(f"TSS combined mask: processed {count_lanes} separation lanes, {count_no_go} no-go features")

    # Downsample to target resolution
    if supersample_factor > 1:
        print(f"TSS combined mask: downsampling from {hi_width}x{hi_height} to {image_width}x{image_height}")
        
        # Downsample masks using max pooling (any True in block -> True)
        lanes_mask = _downsample_mask(lanes_mask_hi, supersample_factor)
        no_go_mask = _downsample_mask(no_go_mask_hi, supersample_factor)
        
        # Downsample vectors using average pooling
        lanes_vecs = _downsample_vectors(lanes_vecs_hi, supersample_factor)
    else:
        lanes_mask = lanes_mask_hi
        lanes_vecs = lanes_vecs_hi
        no_go_mask = no_go_mask_hi

    print(f"TSS combined mask: processed {count_lanes} separation lanes, {count_no_go} no-go features")

    # Save to cache
    if use_cache:
        try:
            np.savez_compressed(
                cache_path, 
                lanes_mask=lanes_mask, 
                lanes_vecs=lanes_vecs, 
                no_go_mask=no_go_mask,
                built=str(datetime.utcnow())
            )
            print(f"TSS combined mask: cached to {cache_path}")
        except Exception as e:
            print(f"TSS combined mask: warning could not write cache ({e})")
    
    return lanes_mask, lanes_vecs, no_go_mask


__all__ = ["build_tss_mask", "build_tss_combined_mask"]
