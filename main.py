"""Main script for ocean route optimization - simplified for distance-only routing."""

from config import ROUTE_COORDS, COASTAL_BUFFER_NM, IMAGE_WIDTH, IMAGE_HEIGHT
from core.initialization import load_and_process_divided_image, load_and_process_image, set_active_bounds
from core.mask import create_buffered_water_mask
from core.dynamic_resolution import load_route_optimized_mask
from navigation.astar import AStar
from navigation.tss_index import build_tss_lane_index, build_tss_mask, build_tss_combined_mask


from navigation.route import RouteCalculator
from utils.coordinates import latlon_to_pixel, validate_coordinates, pixel_to_latlon
from visualization.export import export_path_to_csv
from visualization.plotting import plot_route, plot_route_with_tss
import time


def main():

    FORCE_RECOMPUTE = True  # Set to True to force recompute of cached data
    # Enable dynamic resolution based on route
    USE_DYNAMIC_RESOLUTION = True  # Set to False to use old method
    PIXELS_PER_NM = 1  # 1 pixel = 4 nautical miles (lower res = faster A*)
    ROUTE_PADDING_DEGREES = 10  # Padding around route
    
    if USE_DYNAMIC_RESOLUTION:
        # Load route-optimized mask with dynamic resolution
        is_water, min_lat, max_lat, min_lon, max_lon, IMAGE_WIDTH, IMAGE_HEIGHT = \
            load_route_optimized_mask(ROUTE_COORDS, 
                                     pixels_per_nm=PIXELS_PER_NM,
                                     padding_degrees=ROUTE_PADDING_DEGREES)
        
        # Set active bounds for coordinate conversions
        set_active_bounds(min_lat, max_lat, min_lon, max_lon, IMAGE_WIDTH, IMAGE_HEIGHT)
    else:
        # Use old static resolution method
        min_lat = -80
        max_lat = 80
        min_lon = -170
        max_lon = 170
        
        is_water = load_and_process_image(
             max_lat=max_lat + 10, min_lat=min_lat - 10, 
             max_lon=max_lon + 10, min_lon=min_lon - 10
        )
        IMAGE_WIDTH = is_water.shape[1]
        IMAGE_HEIGHT = is_water.shape[0]
    

    # Precompute TSS combined mask (separation lanes + no-go areas)
    print("\nBuilding TSS combined mask (cached)...")
    EXTEND_LANES_PX = 2  # Set >0 to extend lanes at both ends by N pixels (no change to dilation)
    lanes_mask, lanes_vecs, no_go_mask = build_tss_combined_mask(
        IMAGE_WIDTH,
        IMAGE_HEIGHT,
        root_dir='.',
        dilation_radius=0,        # Widen separation lanes slightly
        no_go_dilation=0,         # Widen no-go areas for safety
        supersample_factor=0,     # 2x higher resolution for TSS features
        extend_pixels=EXTEND_LANES_PX,
        use_cache=True,
        cache_dir='cache',
        force_recompute=FORCE_RECOMPUTE,
        
    )

    print("\n[2/5] Building TSS lane spatial index...")
    t0 = time.time()
    lane_index = build_tss_lane_index(
        root_dir='.',
        use_cache=True,
        cache_dir="cache",
        force_recompute=FORCE_RECOMPUTE  # Use cache now that it's fixed
    )
    dt = time.time() - t0
    
    if lane_index[0] is not None:
        num_lanes = len(lane_index[2])
        print(f"      Indexed {num_lanes} separation lanes in {dt:.2f}s")
    else:
        print("      Warning: No lane index built (missing GeoJSON?)")
        lane_index = None

    # Create water mask with TSS lanes integrated
    # PERFORMANCE FIX: Use pre-computed tss_lanes instead of slow GeoJSON processing
    # Set force_recompute=True ONLY the first time, then change to False to use cache
    print("\nCreating buffered water mask with TSS lanes (cached)...")
    buffered_water = create_buffered_water_mask(
        is_water,
        COASTAL_BUFFER_NM,
        force_recompute=FORCE_RECOMPUTE,    # ← Changed to False (use cache after first run)
        tss_lanes=lanes_mask,     # ← UNCOMMENTED: Use pre-computed mask (fast!)
        # Commented out slow GeoJSON processing (no longer needed with tss_lanes):
        # tss_geojson_path="./TSS/separation_lanes_with_direction.geojson",
        # water_lane_types=["separation_lane"],
        # lane_pixel_width=1,
        # apply_tss_before_buffer=False,
        
    )
   
    # Convert waypoints to pixels
    pixel_waypoints = []
    for i, (lat, lon) in enumerate(ROUTE_COORDS):
        if not validate_coordinates(lat, lon):
            print(f"Error: Waypoint {i} coordinates are outside the available data range")
            return
        
        x, y = latlon_to_pixel(lat, lon)
        pixel_waypoints.append((x, y))
        
        print(f"\nWaypoint {i} Debug:")
        print(f"Coords (lat, lon): {lat}, {lon}")
        print(f"Pixels (x, y): {x}, {y}")
        print(f"Is navigable water:", buffered_water[y, x])

    # Nudge waypoints that ended up on land to nearest water pixel
    def nudge_to_water(pt, mask, max_search=200):  # Increased from 50 to 200 pixels
        x0, y0 = pt
        if 0 <= y0 < mask.shape[0] and 0 <= x0 < mask.shape[1] and mask[y0, x0]:
            return pt
        
        print(f"    Nudging ({x0}, {y0}) to water (searching up to {max_search} pixels)...", end=" ")
        for r in range(1, max_search+1):
            for dx in range(-r, r+1):
                dy = r
                for dy in (r, -r):
                    x = x0 + dx; y = y0 + dy
                    if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1] and mask[y, x]:
                        print(f"found water at ({x}, {y}), distance={r} pixels")
                        return (x, y)
            for dy in range(-r+1, r):
                for dx in (r, -r):
                    x = x0 + dx; y = y0 + dy
                    if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1] and mask[y, x]:
                        print(f"found water at ({x}, {y}), distance={r} pixels")
                        return (x, y)
        print(f"no water found within {max_search} pixels!")
        return pt  # give up
    
    print("\nNudging waypoints to water if needed...")
    pixel_waypoints = [nudge_to_water(p, buffered_water) for p in pixel_waypoints]
    
    

    # Set pixel search radius for A* (in pixels) that equals about 5 nm
    pixel_radius = int(5 * 1852 / ((40075000 / 360) * abs(((max_lat + 10) - (min_lat - 10)) / buffered_water.shape[0]))) 

    print(f"Using pixel search radius: {pixel_radius} pixels")

    # Initialize TSS-aware A* pathfinding using mask for fast cost adjustments
    # Now includes no_go_mask to avoid restricted areas
    astar = AStar(
        buffered_water, # Use buffered water mask for navigation
        tss_preference=True,    # Enable TSS lane preference
        tss_cost_factor=0.95,       # Lower cost to favor TSS lanes more
        tss_mask=lanes_mask,      # Use separation lanes mask
        tss_vecs=lanes_vecs,      # Use direction vectors for lanes
        no_go_mask=no_go_mask,    # Block areas to avoid
        pixel_radius=pixel_radius * 10, # Search radius in pixels
        exploration_angles=30,   # Wider search angles for better pathfinding
        heuristic_weight=1.0,       # Standard A* heuristic less makes better paths but slower
        max_expansions=None,       # No limit on expansions (can be set for performance)
        tss_snap_radius_km=10,
        tss_lane_index=lane_index,    # Lane spatial index for snapping
        tss_bearing_tolerance=90.0,  # Wider tolerance for matching lane direction

        tss_lane_snap_enabled=True
    )
    
    # Initialize simplified route calculator
    route_calculator = RouteCalculator(astar)
    
    # Calculate route
    print("\nCalculating route...")
    complete_path, total_distance, in_tss_lane = route_calculator.optimize_route(pixel_waypoints)

    if complete_path is None:
        print("No complete path found")
        complete_path = []
        total_distance = 0.0
        in_tss_lane = []

    print(f"\nRoute calculated successfully!")
    print(f"Total distance: {total_distance:.2f} nautical miles")
    export_path_to_csv(complete_path, "./exports/direct_route.csv")
    
    # Print TSS lane statistics
    if in_tss_lane:
        tss_count = sum(in_tss_lane)
        tss_percentage = (tss_count / len(in_tss_lane)) * 100
        print(f"TSS lane usage: {tss_count}/{len(in_tss_lane)} waypoints ({tss_percentage:.1f}%)")

    # Post-process: If the route leaves a TSS and later joins another one, and a straight line
    # between those TSS waypoints doesn't cross land (or no-go areas), remove the intermediate waypoints.
    def is_clear_straight_path(p0, p1, mask, blocked_mask=None, tss_mask=None):
        """Return True if all pixels on the straight line from p0 to p1 are navigable in mask
        and not blocked in blocked_mask. Uses Bresenham's line algorithm.

        p0, p1: (x, y) integer pixel coordinates
        mask: 2D boolean array where True indicates navigable water
        blocked_mask: optional 2D boolean array where True indicates blocked (e.g., restricted areas)
        """
        x0, y0 = p0
        x1, y1 = p1
        dx = abs(x1 - x0)
        dy = -abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx + dy  # error value e_xy

        width = mask.shape[1]
        height = mask.shape[0]

        while True:
            # Bounds check
            if not (0 <= x0 < width and 0 <= y0 < height):
                return False
            # Navigable water check
            if not mask[y0, x0]:
                return False
            # Blocked area check (if provided)
            if blocked_mask is not None and blocked_mask[y0, x0] and tss_mask is not None and not tss_mask[y0, x0]:
                return False
            if x0 == x1 and y0 == y1:
                return True
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                x0 += sx
            if e2 <= dx:
                err += dx
                y0 += sy

    # Only proceed if we have TSS flags
    if in_tss_lane:
        print("\nSimplifying segments between TSS waypoints when straight water path exists...")

        # Find indices of waypoints that are in TSS
        tss_indices = [i for i, f in enumerate(in_tss_lane) if f]

        # Build keep mask for waypoints; default keep all
        keep_wp = [True] * len(complete_path)
        simplified_segments = 0

        for i in range(len(tss_indices) - 1):
            start_idx = tss_indices[i]
            end_idx = tss_indices[i + 1]

            # Only consider if there are non-TSS points in between
            if end_idx - start_idx <= 1:
                continue
            has_non_tss = any(not in_tss_lane[k] for k in range(start_idx + 1, end_idx))
            if not has_non_tss:
                continue

            start_pt = complete_path[start_idx]
            end_pt = complete_path[end_idx]

            if is_clear_straight_path(start_pt, end_pt, buffered_water, blocked_mask=no_go_mask, tss_mask=lanes_mask):
                # Remove intermediate points (keep endpoints)
                for k in range(start_idx + 1, end_idx):
                    keep_wp[k] = False
                simplified_segments += 1

        if simplified_segments > 0:
            original_len = len(complete_path)
            complete_path = [p for p, keep in zip(complete_path, keep_wp) if keep]

            # Re-evaluate TSS lane flags for the new path
            try:
                in_tss_lane = route_calculator._check_tss_lane_status(complete_path)
            except Exception:
                # Fallback: map filtered flags by keep mask lengths if private method unavailable
                in_tss_lane = [f for f, keep in zip(in_tss_lane, keep_wp) if keep]

            # Recalculate distance and export
            new_distance = route_calculator.calculate_route_distance(complete_path)
            print(f"  Simplified {simplified_segments} segment(s); waypoints {original_len} -> {len(complete_path)}")
            # Ensure robust arithmetic even if analyzer considers total_distance optional
            prev_distance = float(total_distance if total_distance is not None else 0.0)
            delta_nm = new_distance - prev_distance
            print(f"  Distance: {prev_distance:.2f} nm -> {new_distance:.2f} nm (Δ {delta_nm:+.2f} nm)")

            # Update totals
            total_distance = new_distance
            export_path_to_csv(complete_path, "./exports/direct_route2.csv")

            # Updated TSS stats after simplification
            if in_tss_lane:
                tss_count = sum(in_tss_lane)
                tss_percentage = (tss_count / len(in_tss_lane)) * 100
                print(f"  Updated TSS usage: {tss_count}/{len(in_tss_lane)} waypoints ({tss_percentage:.1f}%)")
        else:
            print("  No eligible TSS-to-TSS segments could be simplified with a straight water path.")

    # Additional simplification: simplify stretches outside of TSS
    if in_tss_lane and complete_path:
        print("\nSimplifying stretches outside TSS where straight water path exists...")

        n = len(complete_path)
        keep_wp2 = [True] * n
        simplified_ntss_segments = 0
        removed_points_ntss = 0

        # Find contiguous ranges where in_tss_lane is False
        idx = 0
        while idx < n:
            if in_tss_lane[idx]:
                idx += 1
                continue
            # Start of a non-TSS run
            run_start = idx
            while idx < n and not in_tss_lane[idx]:
                idx += 1
            run_end = idx - 1  # inclusive

            # Only simplify if at least 3 points in this run
            if run_end - run_start + 1 >= 2:
                i = run_start
                # Greedy skipping: from i, jump to farthest j where straight path is clear
                while i < run_end:
                    # Always keep current i
                    # Find farthest j >= i+2 that is clear
                    best_j = None
                    j = min(run_end, i + 25)  # optional cap to limit checks
                    # Try farthest first to maximize skip
                    while j >= i + 2:
                        if is_clear_straight_path(
                            complete_path[i],
                            complete_path[j],
                            buffered_water,
                            blocked_mask=no_go_mask,
                            tss_mask=lanes_mask,
                        ):
                            best_j = j
                            break
                        j -= 1
                    if best_j is not None:
                        # Remove intermediates (i+1 .. best_j-1)
                        for k in range(i + 1, best_j):
                            if keep_wp2[k]:
                                keep_wp2[k] = False
                                removed_points_ntss += 1
                        simplified_ntss_segments += 1
                        i = best_j
                    else:
                        i += 1

        if removed_points_ntss > 0:
            original_len = len(complete_path)
            complete_path = [p for p, keep in zip(complete_path, keep_wp2) if keep]

            # Recompute flags and distance
            try:
                in_tss_lane = route_calculator._check_tss_lane_status(complete_path)
            except Exception:
                in_tss_lane = [f for f, keep in zip(in_tss_lane, keep_wp2) if keep]

            new_distance = route_calculator.calculate_route_distance(complete_path)
            prev_distance = float(total_distance if total_distance is not None else 0.0)
            delta_nm = new_distance - prev_distance
            print(
                f"  Simplified {simplified_ntss_segments} non-TSS stretch(es); "
                f"waypoints {original_len} -> {len(complete_path)} (−{removed_points_ntss})"
            )
            print(f"  Distance: {prev_distance:.2f} nm -> {new_distance:.2f} nm (Δ {delta_nm:+.2f} nm)")

            total_distance = new_distance
            export_path_to_csv(complete_path, "./exports/direct_route2.csv")

            if in_tss_lane:
                tss_count = sum(in_tss_lane)
                tss_percentage = (tss_count / len(in_tss_lane)) * 100
                print(f"  Updated TSS usage after non-TSS simplification: {tss_percentage:.1f}%")
        else:
            print("  No non-TSS stretches could be simplified with a straight water path.")
   

    # Plot results with TSS lanes
    # print("\nGenerating visualization with TSS lanes...")
    tss_geojson_path = "./TSS/separation_lanes_with_direction.geojson"  # Define for plotting
    plot_route_with_tss(
        buffered_water,
        complete_path,
        tss_geojson_path,
        pixel_waypoints,
        tss_lanes_mask=lanes_mask,
        no_go_mask=no_go_mask,
    )

if __name__ == "__main__":
    main()



