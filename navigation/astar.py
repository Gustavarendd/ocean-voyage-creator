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
    def __init__(self, buffered_water, tss_preference=True, tss_cost_factor=1.0, tss_mask=None, tss_vecs=None,
                 no_go_mask=None, pixel_radius: int | None = None, exploration_angles: int | None = None, max_expansions: int | None = None,
                 heuristic_weight: float = 1.2, disable_pruning: bool = True, use_numpy_core: bool = True,
                 tss_lane_index=None, tss_snap_radius_km: float = 10.0, tss_bearing_tolerance: float = 45.0,
                 tss_lane_snap_enabled: bool = False):
        # Core inputs & preferences
        self.buffered_water = buffered_water
        self.height, self.width = buffered_water.shape
        self.tss_cache = {}
        self.tss_preference = tss_preference
        self.tss_cost_factor = tss_cost_factor
      
        self.tss_mask = tss_mask
        self.tss_vecs = tss_vecs
        self.no_go_mask = no_go_mask  # Mask for areas to avoid (obstacles)
        self.user_pixel_radius = pixel_radius
        self.user_exploration_angles = exploration_angles
        self.max_expansions = max_expansions
        self.heuristic_weight = max(0.1, heuristic_weight)
        self.disable_pruning = disable_pruning
        self.use_numpy_core = use_numpy_core

        # TSS lane snapping configuration
        self.tss_lane_snap_enabled = tss_lane_snap_enabled
        if tss_lane_index is not None and len(tss_lane_index) == 3:
            self.tss_entry_tree = tss_lane_index[0]
            self.tss_exit_tree = tss_lane_index[1]
            self.tss_lane_data = tss_lane_index[2]
        else:
            self.tss_entry_tree = None
            self.tss_exit_tree = None
            self.tss_lane_data = []
        
        self.tss_snap_radius_km = tss_snap_radius_km
        self.tss_bearing_tolerance = tss_bearing_tolerance
        self.tss_snap_cache = {}  # Cache for lane queries at each position
        self.tss_lane_entry_map = {}  # Map (x, y) -> lane_idx for lane entry points
        self.tss_in_lane_nodes = set()  # Track nodes that are part of an active lane

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

    def find_path(self, start, goal, step_length = None, tss_cost_factor = None):
        """Find shortest distance path between start and goal points."""
        import time
        t0 = time.time()
        
        # Store goal for lane waypoint injection
        self.current_goal = goal
        
        # Clear lane tracking state
        self.tss_in_lane_nodes.clear()
        self.tss_lane_entry_map.clear()
        self.tss_snap_cache.clear()

        if step_length is not None:
            if step_length < 1:
                step_length = 1
            elif step_length > self.pixel_radius:
                step_length = self.pixel_radius
            self.pixel_radius = step_length
        
        if tss_cost_factor is not None:
            if tss_cost_factor < 0.01:
                tss_cost_factor = 0.01
            elif tss_cost_factor > 1.0:
                tss_cost_factor = 1.0
            self.tss_cost_factor = tss_cost_factor

        if not self.use_numpy_core:
            # Fallback to previous dict/set implementation if needed
            # Check if start or goal are in no-go areas
            if self.no_go_mask is not None:
                sx, sy = start
                gx, gy = goal
                if self.no_go_mask[sy, sx] and self.tss_mask is not None and not self.tss_mask[sy, sx]:
                    print(f"A*: WARNING - start point {start} is in a no-go area!")
                    return None
                if self.no_go_mask[gy, gx] and self.tss_mask is not None and not self.tss_mask[gy, gx]:
                    print(f"A*: WARNING - goal point {goal} is in a no-go area!")
                    return None
            
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
        
        # Check if start or goal are in no-go areas
        if self.no_go_mask is not None:
            if self.no_go_mask[sy, sx] and self.tss_mask is not None and not self.tss_mask[sy, sx]:
                print(f"A*: WARNING - start point ({sx}, {sy}) is in a no-go area!")
                return None
            if self.no_go_mask[gy, gx] and self.tss_mask is not None and not self.tss_mask[gy, gx]:
                print(f"A*: WARNING - goal point ({gx}, {gy}) is in a no-go area!")
                return None
        
        self._g_score[sy, sx] = 0.0

        open_set = []  # entries: (f, x, y)
        heapq.heappush(open_set, (0.0, sx, sy))

        expansions = 0
        cos_mid = math.cos(self.mid_lat_rad)
        start_wp = (sx, sy)  

        while open_set:
            _, cx, cy = heapq.heappop(open_set)

          
            
            

            # Already processed
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

            # Check if this node is a TSS lane entry point - if so, inject waypoints
            if (cx, cy) in self.tss_lane_entry_map:
                lane_idx = self.tss_lane_entry_map[(cx, cy)]
                injected = self._inject_lane_waypoints(lane_idx, (cx, cy), open_set, cos_mid)
                # Allow both lane waypoints and regular neighbors for flexibility

            # Get neighbors
            neighbors = list(self._get_neighbors((cx, cy), goal, prev_wp=start_wp))
            
            for nx, ny in neighbors:
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
        """Apply cost modifier for TSS lanes to prefer routing through them.
        
        This method handles:
        1. Injected lane waypoints (marked as in-lane) get heavy discount
        2. Lane entry points get heavy discount to encourage A* to choose them
        3. Regular TSS mask-based routing with alignment checks
        4. Transitions into/out of lanes
        """
        # Skip if disabled
        if not self.tss_preference:
            return base_cost

        cx, cy = current
        nx, ny = neighbor

        
        # Priority 1: Check if neighbor is a lane entry point - heavily incentivize!
        if (nx, ny) in self.tss_lane_entry_map:
            return base_cost * 0.1  # 90% discount to strongly prefer lane entries
        
        # Priority 2: Check if neighbor is an injected in-lane waypoint
        # These get the best cost since they're pre-validated lane waypoints
        if (nx, ny) in self.tss_in_lane_nodes:
            # Moving along injected TSS lane waypoint
            return base_cost * 0.3  # 70% discount for verified lane waypoints
        
        # Priority 3: Check if we're exiting from a lane entry (should be rare)
        if (cx, cy) in self.tss_lane_entry_map:
            return base_cost * 0.5  # Still give some discount

        # Priority 4: Original mask-based TSS routing (for areas not in injected lanes)
        if self.tss_mask is not None and self.tss_vecs is not None:
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
                        goal_vec = None
                        goal_len = 0
                        if goal is not None:
                            gdx = goal[0] - current[0]
                            gdy = goal[1] - current[1]
                            goal_len = (gdx**2 + gdy**2) ** 0.5
                            if goal_len > 0:
                                goal_vec = np.array([gdx/goal_len, gdy/goal_len], dtype=np.float32)
                               


                        # Alignment = cosine similarity between lane dir and step dir and goal dir                        
                        # -1 = opposite, 0 = perpendicular, +1 = same direction
                        align = float(np.dot(step_vec, lane_vec))  # -1..1
                        if goal_vec is not None and goal_len > 0:
                            align = (align + float(np.dot(goal_vec, lane_vec))) / 2.0
                      

                        # If align > 0, we’re going with the lane; < 0 = against
                        if align > 0.9 and self.tss_mask[cy, cx]:      # ~ ≤ 25° difference
                            return base_cost * self.tss_cost_factor * 0.6
                        elif align > 0.75 and self.tss_mask[cy, cx]:      # ~ ≤ 40° difference
                            return base_cost* self.tss_cost_factor * 0.7
                        elif align > 0.5 and self.tss_mask[cy, cx]:      # ~ ≤ 60° difference
                            return base_cost* self.tss_cost_factor * 0.7
                        elif align > 0.0 and self.tss_mask[cy, cx]:    # ~ ≤ 90° difference
                            return base_cost* self.tss_cost_factor * 0.85
                            #return base_cost* self.tss_cost_factor * (1.0 - 0.5*(1-align))  # small bonus
                        elif align == 0.0 and self.tss_mask[cy, cx]:   # perpendicular
                            return base_cost * 0.9
                        elif align > -0.2:     # ~ ≤ 100° difference
                            return base_cost * 1
                        elif align > -0.5:     # ~ ≤ 120° difference
                            return base_cost * 1.5
                        else:                  # > 120° difference
                            return base_cost * 10  # heavily penalize going against lane


            return base_cost  # No discount if not in TSS mask


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
    
    def _calculate_goal_bearing(self, current, goal):
        """Calculate bearing from current position to goal (in degrees 0-360).

        Args:
            current: (x, y) pixel coordinates
            goal: (x, y) pixel coordinates

        Returns:
            Bearing in degrees (0 = North, 90 = East) or None if invalid
        """
        try:
            # Convert pixel coords to lat/lon
            from utils.coordinates import pixel_to_latlon
            
            cx, cy = current
            gx, gy = goal
            
            # Simple bounds check
            if not (0 <= cx < self.width and 0 <= cy < self.height):
                return None
            if not (0 <= gx < self.width and 0 <= gy < self.height):
                return None
            
            c_lat, c_lon = pixel_to_latlon(cx, cy)
            g_lat, g_lon = pixel_to_latlon(gx, gy)
            
            return self._calculate_bearing(c_lat, c_lon, g_lat, g_lon)
        except Exception:
            return None
    
    def _is_bearing_aligned(self, bearing1, bearing2, tolerance_deg=None):
        """Check if two bearings are aligned within tolerance.

        Args:
            bearing1: First bearing in degrees (0-360)
            bearing2: Second bearing in degrees (0-360)
            tolerance_deg: Maximum difference in degrees (default: use self.tss_bearing_tolerance)

        Returns:
            True if bearings are within tolerance, False otherwise
        """
        if bearing1 is None or bearing2 is None:
            return False
        
        if tolerance_deg is None:
            tolerance_deg = self.tss_bearing_tolerance
        
        # Calculate angular difference, accounting for 360/0 wrap
        diff = abs(bearing1 - bearing2)
        if diff > 180:
            diff = 360 - diff
        
        return diff <= tolerance_deg
    
    def _find_nearby_lane_entries(self, position, goal, prev_point=None):
        """Find TSS lane entries near current position that align with goal direction.

        Args:
            position: (x, y) pixel coordinates
            goal: (x, y) pixel coordinates for goal

        Returns:
            List of (lane_index, distance_km) tuples for aligned lanes, sorted by distance
        """
        if not self.tss_lane_snap_enabled or self.tss_entry_tree is None:
            return []
        
        # Check cache first
        cache_key = (position, goal)
        if cache_key in self.tss_snap_cache:
            return self.tss_snap_cache[cache_key]
        
        try:
            from utils.coordinates import pixel_to_latlon
            from utils.distance import nm_distance
            
            px, py = position
            if not (0 <= px < self.width and 0 <= py < self.height):
                return []
            
            # Convert position to lat/lon
            pos_lat, pos_lon = pixel_to_latlon(px, py)
            
            # Calculate goal bearing
            goal_bearing = self._calculate_goal_bearing(position, goal)
            if goal_bearing is None:
                return []
            
            # Calculate previous bearing if available
            if prev_point is not None:
                prev_bearing = self._calculate_goal_bearing(prev_point, position)
                if prev_bearing is not None:
                    # Average with goal bearing for smoother transitions
                    
                    goal_bearing = self._mean_bearing(goal_bearing, prev_bearing)
            
            # Query KD-tree for nearby lane entries
            # Convert km to degrees (approximate: 1 deg ~ 111 km)
            search_radius_deg = self.tss_snap_radius_km / 111.0
            
            # Query returns indices and distances
            indices = self.tss_entry_tree.query_ball_point(
                [pos_lat, pos_lon], 
                r=search_radius_deg
            )
            
            if not indices:
                self.tss_snap_cache[cache_key] = []
                return []
            
            # Filter by bearing alignment and calculate actual distances
            aligned_lanes = []
            for idx in indices:
                lane = self.tss_lane_data[idx]
                lane_bearing = lane['bearing']
                
                is_aligned = self._is_bearing_aligned(goal_bearing, lane_bearing)
                
                if is_aligned:
                    # Calculate actual distance
                    entry_lat, entry_lon = lane['entry']
                    dist_nm = nm_distance(pos_lat, pos_lon, entry_lat, entry_lon)
                    dist_km = dist_nm * 1.852
                    
                    if dist_km <= self.tss_snap_radius_km:
                    
                        aligned_lanes.append((idx, dist_km))
            
            # Sort by distance
            aligned_lanes.sort(key=lambda x: x[1])
            
            # Cache result
            self.tss_snap_cache[cache_key] = aligned_lanes
            
            return aligned_lanes
            
        except Exception as e:
            # Silently fail - don't break A* if lane query fails
            return []
    
    def _inject_lane_waypoints(self, lane_idx, current_pos, open_set, cos_mid):
        """Inject TSS lane waypoints into the A* search.

        When a lane entry is detected, this method adds all lane waypoints
        to the open set with low cost, creating a fast path through the lane.

        Args:
            lane_idx: Index of the lane in self.tss_lane_data
            current_pos: (x, y) current pixel position (lane entry point)
            open_set: Priority queue (heapq) to inject waypoints into
            cos_mid: Cosine of mid-latitude for distance calculations

        Returns:
            Number of waypoints successfully injected
        """
        if not self.use_numpy_core:
            return 0  # Only support numpy core for now
        
        try:
            from utils.coordinates import latlon_to_pixel
            
            lane = self.tss_lane_data[lane_idx]
            waypoints = lane['waypoints']  # List of [lat, lon]
            
            if len(waypoints) < 2:
                return 0
            
            cx, cy = current_pos
            current_g = self._g_score[cy, cx]
            
            injected = 0
            prev_x, prev_y = cx, cy
            
            # Inject waypoints sequentially along the lane
            for i, (wp_lat, wp_lon) in enumerate(waypoints[1:], 1):  # Skip first (entry)
                try:
                    wp_x, wp_y = latlon_to_pixel(wp_lat, wp_lon, warn=False)
                    
                    # Validate waypoint
                    if not (0 <= wp_x < self.width and 0 <= wp_y < self.height):
                        continue
                    if not self.buffered_water[wp_y, wp_x]:
                        continue
                    if self.no_go_mask is not None and self.no_go_mask[wp_y, wp_x] and self.tss_mask is not None and not self.tss_mask[wp_y, wp_x]:
                        continue
                    
                    # Calculate cost from previous waypoint
                    dx_px = wp_x - prev_x
                    dy_px = wp_y - prev_y
                    if self.wrap_longitude:
                        dx1 = wp_x - prev_x
                        dx2 = wp_x - prev_x - self.width
                        dx3 = wp_x - prev_x + self.width
                        dx_px = min([dx1, dx2, dx3], key=abs)
                    
                    dlat_deg = dy_px * self.dlat_per_px
                    dlon_deg = dx_px * self.dlon_per_px
                    segment_dist = math.hypot(dlat_deg, dlon_deg * cos_mid)
                    
                    # Apply heavy discount for in-lane movement
                    lane_cost_factor = 0.3  # 70% discount for using TSS lane
                    segment_cost = segment_dist * lane_cost_factor
                    
                    tentative_g = current_g + segment_cost * (i)  # Accumulate cost
                    
                    # Only inject if this is a better path
                    if tentative_g < self._g_score[wp_y, wp_x]:
                        self._g_score[wp_y, wp_x] = tentative_g
                        self._parent_x[wp_y, wp_x] = prev_x
                        self._parent_y[wp_y, wp_x] = prev_y
                        
                        # Mark as in-lane node
                        self.tss_in_lane_nodes.add((wp_x, wp_y))
                        
                        # Add to open set with heuristic
                        f = tentative_g + self.heuristic_weight * self._heuristic((wp_x, wp_y), self.current_goal)
                        heapq.heappush(open_set, (f, wp_x, wp_y))
                        
                        injected += 1
                        prev_x, prev_y = wp_x, wp_y
                    
                except Exception:
                    continue
            
            return injected
            
        except Exception as e:
            return 0
    
    

    def _get_neighbors(self, point, goal=None, max_radius=15, oversample=1.0, prev_wp=None):
        """
        Return neighbor pixels on the *smallest* radius ring that contains any valid neighbor.
        Grows radius outward (1 px at a time) until at least one neighbor is found,
        then returns only those neighbors from that ring.

        TSS Lane Snapping: If enabled, checks for nearby aligned TSS lane entries.
        When a suitable lane is found, returns lane entry point as a high-priority neighbor.

        Args:
            point: (x, y) pixel
            goal: optional (x, y) to enable mild directional pruning
            max_radius: optional hard cap (defaults to image diagonal)
            oversample: multiplier for angular sampling density as radius grows
        """
        # Debug lane snapping status ONCE (optional - can be removed in production)
        if not hasattr(self, '_logged_snap_status'):
            self._logged_snap_status = True
            if self.tss_lane_snap_enabled:
                print(f"TSS lane snapping enabled: {len(self.tss_lane_data) if self.tss_lane_data else 0} lanes indexed")
        
        # Check for TSS lane snapping opportunity
        if self.tss_lane_snap_enabled and goal is not None:

            # get previously wp and calculate bearing to current point


            nearby_lanes = self._find_nearby_lane_entries(point, goal, prev_point=prev_wp)
            if nearby_lanes:
                # Found aligned lane(s) - mark entry point and let main loop handle injection
                lane_neighbors = []
                for lane_idx, dist_km in nearby_lanes[:1]:  # Use closest lane
                    lane = self.tss_lane_data[lane_idx]
                    # Convert lane entry to pixel coordinates
                    try:
                        from utils.coordinates import latlon_to_pixel
                        entry_lat, entry_lon = lane['entry']
                        entry_x, entry_y = latlon_to_pixel(entry_lat, entry_lon, warn=False)
                        
                        # Verify entry point is valid (water, in bounds, not no-go)
                        # Also skip if entry point is the current position (already there!)
                        if (entry_x, entry_y) == point:
                            continue
                        
                        # Skip if entry has already been visited (in closed set)
                        if self._closed[entry_y, entry_x]:
                            continue
                        
                        if (0 <= entry_x < self.width and 0 <= entry_y < self.height and
                            self.buffered_water[entry_y, entry_x]):
                            
                            if (self.no_go_mask is None or not self.no_go_mask[entry_y, entry_x]) or (self.tss_mask is None or self.tss_mask[entry_y, entry_x]):
                                # Store lane index for this entry point
                                self.tss_lane_entry_map[(entry_x, entry_y)] = lane_idx
                                lane_neighbors.append((entry_x, entry_y))
                    except Exception as e:
                        pass
                
                if lane_neighbors:
                    # Return both lane entry and regular neighbors
                    # This gives A* a choice but biases toward the lane
                    return lane_neighbors
        
        width, height = self.width, self.height
        start_radius = max(1, int(round(self.pixel_radius)))

        close_to_land, pixel_dist = self.position_close_to_land(point, check_radius=start_radius)


        if close_to_land and pixel_dist is not None and pixel_dist < start_radius:
            start_radius = max(1, pixel_dist - 1)

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
                max_angle = 120 if distg < 80 else 100
                dir_cos_min = math.cos(math.radians(max_angle))

        px, py = point
        if self.tss_mask is not None and self.tss_mask[py, px]:
            # In TSS lane - use full radius for max flexibility
            r = min(2, start_radius)
        else:
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
            # Check bounds, water availability, and no-go areas unless in a tss lane

            if 0 <= nx < width and 0 <= ny < height and (self.buffered_water[ny, nx] or (self.tss_mask is not None and self.tss_mask[ny, nx])):
                # Skip if this pixel is marked as a no-go area
                if self.no_go_mask is not None and self.no_go_mask[ny, nx] and (self.tss_mask is None or not self.tss_mask[ny, nx]):
                    continue
                if self.check_between_wps(point, (nx, ny)):
                    continue  # land in between
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
        print(f"A*: path length {len(path)}")
        return path
    

    def check_between_wps(self, start, end):
        """Check if a straight line between start and end crosses any land pixels.

        Args:
            start: (x, y) pixel
            end: (x, y) pixel
        Returns:
            True if line crosses land, False if clear water
        """
        x0, y0 = start
        x1, y1 = end
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy

        while True:
            if not (0 <= x0 < self.width and 0 <= y0 < self.height):
                return True  # out of bounds treated as land
            if not self.buffered_water[y0, x0]:
                return True  # hit land
            # if self.no_go_mask is not None and self.no_go_mask[y0, x0]:
            #     return True  # hit no-go area
            if (x0, y0) == (x1, y1):
                break
            err2 = err * 2
            if err2 > -dy:
                err -= dy
                x0 += sx
            if err2 < dx:
                err += dx
                y0 += sy

        return False  # clear water between points
    

    def get_tss_waypoints_near(self, position, radius_nm=50):
        """Get cached or compute TSS waypoints near a given pixel position.

        Args:
            position: (x, y) pixel
            radius_nm: search radius in nautical miles
        Returns:
            List of (lat, lon) waypoints within radius
        """
        if position in self.tss_cache:
            return self.tss_cache[position]
        lat, lon = pixel_to_latlon(position[0], position[1], self.width, self.height,
                                  self.ACTIVE_LAT_MIN, self.ACTIVE_LAT_MAX,
                                  self.ACTIVE_LON_MIN, self.ACTIVE_LON_MAX)
        waypoints = get_tss_waypoints_near_position(lat, lon, radius_nm)
        self.tss_cache[position] = waypoints
        return waypoints
    

    def position_close_to_land(self, position, check_radius=20):
        """Check if there is land within a square of given radius around position.

        Args:
            position: (x, y) pixel
            check_radius: radius in pixels to check around position
        Returns:
            True if any land pixel found, False if all water,
            and pixel distance to nearest land
        """
        x, y = position
        x_min = max(0, x - check_radius)
        x_max = min(self.width - 1, x + check_radius)
        y_min = max(0, y - check_radius)
        y_max = min(self.height - 1, y + check_radius)
        sub_area = self.buffered_water[y_min:y_max+1, x_min:x_max+1]

        close_to_land = not np.all(sub_area)  # True if any land (False in water-only area)
        pixel_dist = None
        if close_to_land:
            # Compute distance to nearest land pixel
            land_indices = np.argwhere(~sub_area)  # indices of land pixels in sub_area
            if land_indices.size > 0:
                # Convert to absolute pixel coordinates
                land_coords = land_indices + np.array([y_min, x_min])
                dists = np.hypot(land_coords[:,1] - x, land_coords[:,0] - y)
                pixel_dist = int(np.min(dists))
            return close_to_land, pixel_dist
        return close_to_land, None
       

    def is_position_close_to_tss(self, position):
        """Check if a pixel position is within a TSS lane.

        Args:
            position: (x, y) pixel
        Returns:
            True if in TSS lane, False otherwise.
            pixel distance to nearest TSS lane (None if not found)
        """
        x, y = position
        if self.tss_mask is None:
            return False, None
        if 0 <= x < self.width and 0 <= y < self.height:
            if self.tss_mask[y, x]:
                return True, 0  # in TSS lane
            # Compute distance to nearest TSS lane pixel
            check_radius = 20
            x_min = max(0, x - check_radius)
            x_max = min(self.width - 1, x + check_radius)
            y_min = max(0, y - check_radius)
            y_max = min(self.height - 1, y + check_radius)
            sub_area = self.tss_mask[y_min:y_max+1, x_min:x_max+1]
            tss_indices = np.argwhere(sub_area)  # indices of TSS pixels in sub_area
            if tss_indices.size > 0:
                # Convert to absolute pixel coordinates
                tss_coords = tss_indices + np.array([y_min, x_min])
                dists = np.hypot(tss_coords[:,1] - x, tss_coords[:,0] - y)
                pixel_dist = int(np.min(dists))
                return False, pixel_dist
        return False, None
    
    

    def _mean_bearing(self, a, b):
        # both in degrees
        a_rad = math.radians(a)
        b_rad = math.radians(b)
        x = math.cos(a_rad) + math.cos(b_rad)
        y = math.sin(a_rad) + math.sin(b_rad)
        mean = math.degrees(math.atan2(y, x)) % 360
        return mean