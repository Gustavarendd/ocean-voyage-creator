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


def build_tss_mask(image_width: int, image_height: int, root_dir: str = '.', dilation_radius: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    """Build a boolean mask + vector field for TSS lanes.

    Returns:
        mask: np.ndarray[bool] (H,W) → True if pixel lies on/near lane
        vecs: np.ndarray[float32] (H,W,2) → (u,v) unit vector, (0,0) if none
    """
    mask = np.zeros((image_height, image_width), dtype=bool)
    vecs = np.zeros((image_height, image_width, 2), dtype=np.float32)

    files = list(_iter_tss_files(root_dir))
    if not files:
        print("TSS mask: no TSS_by_direction geojson files found")
        return mask, vecs

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
            coords = geom.get('coordinates') or []  # list of [lon, lat]
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
    return mask, vecs


__all__ = ["build_tss_mask"]
