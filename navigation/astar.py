"""A* pathfinding implementation for ocean routing - shortest distance only.

Performance optimizations (2025-09):
 - Numpy arrays for g_score & closed set (O(1) boolean lookup) instead of dict/set of tuples.
 - Parent pointers stored in int32 arrays (parent_x/parent_y) to avoid large dict for came_from.
 - Ring offset caching: expensive sin/cos per expansion radius removed; offsets computed once per radius.
 - Optional heuristic inflation (>1) strongly recommended for speed (slight sub‑optimality acceptable).
 - Fast TSS mask lookup path retained; on-demand geo fallback unchanged.

Public interface preserved (find_path returns list of (x,y)).
"""

import heapq
import math
from config import LAT_MIN, LAT_MAX, RADIUS
import numpy as np
try:
    from core.initialization import ACTIVE_LON_MIN, ACTIVE_LON_MAX
except Exception:
    # Fallback to full range so behavior matches original if initialization not run
    ACTIVE_LON_MIN, ACTIVE_LON_MAX = -180.0, 180.0
from utils.coordinates import pixel_to_latlon
from .tss import get_tss_waypoints_near_position

class AStar:
    def __init__(self, buffered_water, tss_preference=True, tss_cost_factor=0.1, tss_search_radius_m= 25000, tss_mask=None, tss_vecs=None,
                 pixel_radius: int | None = None, exploration_angles: int | None = None, max_expansions: int | None = None,
                 heuristic_weight: float = 1.2, disable_pruning: bool = True, use_numpy_core: bool = True):
        # Core inputs & preferences
        self.buffered_water = buffered_water
        self.height, self.width = buffered_water.shape
        self.tss_cache = {}
        self.tss_preference = tss_preference
        self.tss_cost_factor = tss_cost_factor
        self.tss_search_radius_m = tss_search_radius_m
        self.tss_mask = tss_mask
        self.tss_vecs = tss_vecs
        self.user_pixel_radius = pixel_radius
        self.user_exploration_angles = exploration_angles
        self.max_expansions = max_expansions
        self.heuristic_weight = max(0.1, heuristic_weight)
        self.disable_pruning = disable_pruning
        self.use_numpy_core = use_numpy_core

        # Active geographic bounds (may be cropped)
        try:  # import inside to honor runtime cropping
            from core.initialization import (
                ACTIVE_LAT_MIN, ACTIVE_LAT_MAX, ACTIVE_LON_MIN, ACTIVE_LON_MAX
            )
        except Exception:
            from config import LAT_MIN as ACTIVE_LAT_MIN, LAT_MAX as ACTIVE_LAT_MAX, LON_MIN as ACTIVE_LON_MIN, LON_MAX as ACTIVE_LON_MAX
        self.ACTIVE_LON_MIN = ACTIVE_LON_MIN
        self.ACTIVE_LON_MAX = ACTIVE_LON_MAX
        self.ACTIVE_LAT_MIN = ACTIVE_LAT_MIN
        self.ACTIVE_LAT_MAX = ACTIVE_LAT_MAX

        self.wrap_longitude = (self.ACTIVE_LON_MAX - self.ACTIVE_LON_MIN) >= 359.0

        # Degrees per pixel for current grid
        self.dlat_per_px = (self.ACTIVE_LAT_MAX - self.ACTIVE_LAT_MIN) / self.height
        self.dlon_per_px = (self.ACTIVE_LON_MAX - self.ACTIVE_LON_MIN) / self.width
        self.mid_lat_rad = math.radians((self.ACTIVE_LAT_MIN + self.ACTIVE_LAT_MAX) / 2.0)

        # Scale exploration radius if grid cropped (so physical distance similar)
        lon_scale = 360.0 / max(1e-6, (self.ACTIVE_LON_MAX - self.ACTIVE_LON_MIN))
        lat_scale = (LAT_MAX - LAT_MIN) / max(1e-6, (self.ACTIVE_LAT_MAX - self.ACTIVE_LAT_MIN))
        scale_factor = max(lon_scale, lat_scale)
        scaled_radius = max(1, int(round(RADIUS * scale_factor)))
        if self.user_pixel_radius is not None:
            scaled_radius = max(1, self.user_pixel_radius)
        self.pixel_radius = scaled_radius

        self.num_directions = self.user_exploration_angles
        

        # Caches
        self._ring_cache = {}
        self._dxdy_unit_cache = {}

        # Numpy structures (optional)
        if self.use_numpy_core:
            self._g_score = np.full((self.height, self.width), np.inf, dtype=np.float64)
            self._closed = np.zeros((self.height, self.width), dtype=bool)
            self._parent_x = np.full((self.height, self.width), -1, dtype=np.int32)
            self._parent_y = np.full((self.height, self.width), -1, dtype=np.int32)
        else:
            self._g_score = None

    def find_path(self, start, goal):
        """Find shortest distance path between start and goal points."""
        import time
        t0 = time.time()

        if not self.use_numpy_core:
            # Fallback to previous dict/set implementation if needed
            open_set = []
            heapq.heappush(open_set, (0, start))
            came_from = {}
            g_score = {start: 0.0}
            closed = set()
            expansions = 0
            while open_set:
                _, current = heapq.heappop(open_set)
                if current in closed:
                    continue
                if current == goal:
                    dt = time.time() - t0
                    print(f"A*: reached goal. expansions={expansions} elapsed={dt:.2f}s")
                    return self._reconstruct_path_dict(came_from, start, goal)
                closed.add(current)
                expansions += 1
                if self.max_expansions and expansions > self.max_expansions:
                    print(f"A*: max_expansions {self.max_expansions} reached (fallback core)")
                    return self._reconstruct_path_dict(came_from, start, current)
                for neighbor in self._get_neighbors(current, goal):
                    if neighbor in closed:
                        continue
                    dy_px = neighbor[1] - current[1]
                    if self.wrap_longitude:
                        dx1 = neighbor[0] - current[0]
                        dx2 = neighbor[0] - current[0] - self.width
                        dx3 = neighbor[0] - current[0] + self.width
                        dx_px = min([dx1, dx2, dx3], key=abs)
                    else:
                        dx_px = neighbor[0] - current[0]
                    dlat_deg = dy_px * self.dlat_per_px
                    dlon_deg = dx_px * self.dlon_per_px
                    base_cost = math.hypot(dlat_deg, dlon_deg * math.cos(self.mid_lat_rad))
                    cost = self._apply_tss_cost_modifier(current, neighbor, base_cost)
                    tentative_g = g_score[current] + cost
                    if neighbor not in g_score or tentative_g < g_score[neighbor]:
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g
                        f = tentative_g + self.heuristic_weight * self._heuristic(neighbor, goal)
                        heapq.heappush(open_set, (f, neighbor))
            return None

        # Optimized numpy core
        # Reset arrays (cheap) instead of reallocating
        self._g_score.fill(np.inf)
        self._closed.fill(False)
        self._parent_x.fill(-1)
        self._parent_y.fill(-1)

        sx, sy = start
        gx, gy = goal
        if not (0 <= sx < self.width and 0 <= sy < self.height):
            return None
        if not (0 <= gx < self.width and 0 <= gy < self.height):
            return None
        self._g_score[sy, sx] = 0.0

        open_set = []  # entries: (f, x, y)
        heapq.heappush(open_set, (0.0, sx, sy))

        expansions = 0
        cos_mid = math.cos(self.mid_lat_rad)

        while open_set:
            _, cx, cy = heapq.heappop(open_set)
            if self._closed[cy, cx]:
                continue
            if (cx, cy) == (gx, gy):
                dt = time.time() - t0
                print(f"A*: reached goal. expansions={expansions} elapsed={dt:.2f}s")
                return self._reconstruct_path_arrays(start, goal)
            self._closed[cy, cx] = True
            expansions += 1
            if self.max_expansions and expansions > self.max_expansions:
                print(f"A*: max_expansions {self.max_expansions} reached, partial path returned")
                return self._reconstruct_path_arrays(start, (cx, cy))

            for nx, ny in self._get_neighbors((cx, cy), goal):
                if self._closed[ny, nx]:
                    continue
                # Delta respecting wrap
                if self.wrap_longitude:
                    dx1 = nx - cx
                    dx2 = nx - cx - self.width
                    dx3 = nx - cx + self.width
                    dx_px = min([dx1, dx2, dx3], key=abs)
                else:
                    dx_px = nx - cx
                dy_px = ny - cy
                dlat_deg = dy_px * self.dlat_per_px
                dlon_deg = dx_px * self.dlon_per_px
                base_cost = math.hypot(dlat_deg, dlon_deg * cos_mid)
                cost = self._apply_tss_cost_modifier((cx, cy), (nx, ny), base_cost, goal)
                tentative_g = self._g_score[cy, cx] + cost
                if tentative_g < self._g_score[ny, nx]:
                    self._g_score[ny, nx] = tentative_g
                    self._parent_x[ny, nx] = cx
                    self._parent_y[ny, nx] = cy
                    f = tentative_g + self.heuristic_weight * self._heuristic((nx, ny), goal)
                    heapq.heappush(open_set, (f, nx, ny))

        return None  # no path

    def _apply_tss_cost_modifier(self, current, neighbor, base_cost, goal=None):
        """Apply cost modifier for TSS lanes to prefer routing through them."""
        # Skip if disabled
        if not self.tss_preference:
            return base_cost

        # Fast path: precomputed mask + vector lookup
        if self.tss_mask is not None and self.tss_vecs is not None:
            nx, ny = neighbor
            if 0 <= ny < self.height and 0 <= nx < self.width and self.tss_mask[ny, nx]:
                lane_vec = self.tss_vecs[ny, nx]
                if not np.allclose(lane_vec, (0.0, 0.0)):
                    # Ship step vector (pixel space)
                    dx = nx - current[0]
                    dy = ny - current[1]
                    step_len = (dx**2 + dy**2) ** 0.5
                    if step_len > 0:
                        step_vec = np.array([dx/step_len, dy/step_len], dtype=np.float32)

                        # Heading toward goal (pixel space)
                        if goal is not None:
                            gdx = goal[0] - current[0]
                            gdy = goal[1] - current[1]
                            goal_len = (gdx**2 + gdy**2) ** 0.5
                            if goal_len > 0:
                                goal_vec = np.array([gdx/goal_len, gdy/goal_len], dtype=np.float32)
                               


                        # Alignment = cosine similarity between lane dir and step dir and goal dir                        
                        # -1 = opposite, 0 = perpendicular, +1 = same direction
                        align = float(np.dot(step_vec, lane_vec))  # -1..1
                        if goal is not None and goal_len > 0:
                            align = (align + float(np.dot(goal_vec, lane_vec))) / 2.0
                      

                        # If align > 0, we’re going with the lane; < 0 = against
                        if align > 0.9:      # ~ ≤ 25° difference
                            return base_cost * self.tss_cost_factor * 0.5
                        elif align > 0.75:      # ~ ≤ 40° difference
                            return base_cost* self.tss_cost_factor * 0.5
                        elif align > 0.5:      # ~ ≤ 60° difference
                            return base_cost* self.tss_cost_factor * 0.5
                        elif align > 0.0:    # ~ ≤ 90° difference
                            return base_cost* self.tss_cost_factor * 0.5
                            #return base_cost* self.tss_cost_factor * (1.0 - 0.5*(1-align))  # small bonus
                        elif align == 0.0:
                            return base_cost
                        elif align > -0.5:     # ~ ≤ 120° difference
                            return base_cost * 2.0
                        else:                  # > 120° difference
                            return base_cost * 10  # heavily penalize going against lane

            return base_cost


    def _calculate_bearing(self, lat1, lon1, lat2, lon2):
        """Calculate the bearing between two lat/lon points in degrees."""
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        delta_lon_rad = math.radians(lon2 - lon1)
        
        y = math.sin(delta_lon_rad) * math.cos(lat2_rad)
        x = (math.cos(lat1_rad) * math.sin(lat2_rad) - 
             math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(delta_lon_rad))
        
        bearing_rad = math.atan2(y, x)
        bearing_deg = (math.degrees(bearing_rad) + 360) % 360
        
        return bearing_deg
    
    

    def _get_neighbors(self, point, goal=None, max_radius=15, oversample=1.0):
        """
        Return neighbor pixels on the *smallest* radius ring that contains any valid neighbor.
        Grows radius outward (1 px at a time) until at least one neighbor is found,
        then returns only those neighbors from that ring.

        Args:
            point: (x, y) pixel
            goal: optional (x, y) to enable mild directional pruning
            max_radius: optional hard cap (defaults to image diagonal)
            oversample: multiplier for angular sampling density as radius grows
        """
        width, height = self.width, self.height
        start_radius = max(1, int(round(self.pixel_radius)))

        # Safety cap so we can't loop forever
        if max_radius is None:
            max_radius = int(math.ceil(math.hypot(width, height)))

        # Pruning setup (unchanged from your logic)
        uxg = uyg = None
        dir_cos_min = None
        if (not self.disable_pruning) and goal is not None:
            gx, gy = goal
            dxg = gx - point[0]
            dyg = gy - point[1]
            if self.wrap_longitude:
                options = [dxg, dxg - width, dxg + width]
                dxg = min(options, key=abs)
            distg = math.hypot(dxg, dyg)
            if distg > 1e-6:
                uxg = dxg / distg
                uyg = dyg / distg
                max_angle = 120 if distg < 80 else 80
                dir_cos_min = math.cos(math.radians(max_angle))

        r = start_radius
       
        # Retrieve or build ring offsets
        if r not in self._ring_cache:
            if self.num_directions is None:
                num_dirs = int(2 * math.pi * r * oversample)
            else:
                num_dirs = self.num_directions
            offsets = []
            seen = set()
            # Precompute integer ring with rounding; duplicates removed
            cos_vals = [math.cos(2 * math.pi * i / num_dirs) for i in range(num_dirs)]
            sin_vals = [math.sin(2 * math.pi * i / num_dirs) for i in range(num_dirs)]
            for c, s in zip(cos_vals, sin_vals):
                dx = int(round(r * c))
                dy = int(round(r * s))
                if (dx, dy) == (0, 0):
                    continue
                if (dx, dy) in seen:
                    continue
                seen.add((dx, dy))
                offsets.append((dx, dy))
            self._ring_cache[r] = offsets
            print(f"[A*] r={r} requested_dirs={self.num_directions} generated={num_dirs} unique={len(offsets)}")
        offsets = self._ring_cache[r]

        neighbors: list[tuple[int,int]] = []
        for dx, dy in offsets:
            if dir_cos_min is not None:
                key = (dx, dy)
                unit = self._dxdy_unit_cache.get(key)
                if unit is None:
                    mag = math.hypot(dx, dy) or 1.0
                    unit = (dx / mag, dy / mag)
                    self._dxdy_unit_cache[key] = unit
                ux, uy = unit
                if ux * uxg + uy * uyg < dir_cos_min:
                    continue
            nx = point[0] + dx
            ny = point[1] + dy
            if self.wrap_longitude:
                nx = self._wrap_x(nx)
            if 0 <= nx < width and 0 <= ny < height and self.buffered_water[ny, nx]:
                neighbors.append((nx, ny))
        if neighbors:
            return neighbors
            
        return []


    def _wrap_x(self, x):
        """Wrap x coordinate around the dateline (only if wrap_longitude)."""
        if not self.wrap_longitude:
            return x  # caller should bounds-check
        while x < 0:
            x += self.width
        while x >= self.width:
            x -= self.width
        return x

    def _heuristic(self, a, b):
        """Calculate heuristic distance between two points."""
        dy_px = b[1] - a[1]
        if self.wrap_longitude:
            dx1 = b[0] - a[0]
            dx2 = b[0] - a[0] - self.width
            dx3 = b[0] - a[0] + self.width
            dx_px = min([dx1, dx2, dx3], key=abs)
        else:
            dx_px = b[0] - a[0]
        dlat_deg = dy_px * self.dlat_per_px
        dlon_deg = dx_px * self.dlon_per_px
        return math.hypot(dlat_deg, dlon_deg * math.cos(self.mid_lat_rad))

    # --- Path reconstruction helpers ---
    def _reconstruct_path_dict(self, came_from, start, goal):
        path = []
        cur = goal
        while cur in came_from:
            path.append(cur)
            cur = came_from[cur]
        path.append(start)
        return path[::-1]

    def _reconstruct_path_arrays(self, start, goal):
        path = []
        gx, gy = goal
        if not (0 <= gx < self.width and 0 <= gy < self.height):
            return []
        cx, cy = gx, gy
        path.append((cx, cy))
        # Follow parents until start or -1
        while not (cx == start[0] and cy == start[1]):
            px = self._parent_x[cy, cx]
            py = self._parent_y[cy, cx]
            if px < 0 or py < 0:
                break  # disconnected
            path.append((px, py))
            cx, cy = px, py
        path.reverse()
        return path