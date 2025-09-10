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


__all__ = ["build_tss_mask"]
