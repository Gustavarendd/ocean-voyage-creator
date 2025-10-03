"""Main script for ocean route optimization - simplified for distance-only routing."""

import math
from config import ROUTE_COORDS, COASTAL_BUFFER_NM, IMAGE_WIDTH, IMAGE_HEIGHT
from core.initialization import load_and_process_divided_image, load_and_process_image
from core.mask import create_buffered_water_mask
from navigation.astar import AStar
from navigation.tss_index import build_tss_mask, build_tss_combined_mask


from navigation.route import RouteCalculator
from utils.coordinates import latlon_to_pixel, validate_coordinates, pixel_to_latlon
from visualization.export import export_path_to_csv
from visualization.plotting import plot_route, plot_route_with_tss
from analysis.tss_analysis import export_tss_analysis, print_tss_segments

def main():

    # Calculate bounding box from route waypoints with ±10° padding
    print("\n" + "="*80)
    print("ROUTE REGION CALCULATION")
    print("="*80)
    
    lats = [lat for lat, lon in ROUTE_COORDS]
    lons = [lon for lat, lon in ROUTE_COORDS]
    
    # Find bounds with safety margin
    PADDING_DEGREES = 10
    min_lat = max(-90, min(lats) - PADDING_DEGREES)
    max_lat = min(90, max(lats) + PADDING_DEGREES)
    min_lon = max(-180, min(lons) - PADDING_DEGREES)
    max_lon = min(180, max(lons) + PADDING_DEGREES)
    
    # Calculate region statistics
    lat_span = max_lat - min_lat
    lon_span = max_lon - min_lon
    mid_lat = (max_lat + min_lat) / 2.0
    
    print(f"\nRoute waypoints: {len(ROUTE_COORDS)}")
    print(f"  Start: ({lats[0]:.2f}°, {lons[0]:.2f}°)")
    print(f"  End:   ({lats[-1]:.2f}°, {lons[-1]:.2f}°)")
    
    print(f"\nRegion bounds (with ±{PADDING_DEGREES}° padding):")
    print(f"  Latitude:  {min_lat:.2f}° to {max_lat:.2f}° (span: {lat_span:.2f}°)")
    print(f"  Longitude: {min_lon:.2f}° to {max_lon:.2f}° (span: {lon_span:.2f}°)")
    print(f"  Center:    ({mid_lat:.2f}°, {(min_lon+max_lon)/2:.2f}°)")
    
    # Calculate approximate coverage
    world_lat_coverage = (lat_span / 180.0) * 100
    world_lon_coverage = (lon_span / 360.0) * 100
    world_area_coverage = (world_lat_coverage / 100) * (world_lon_coverage / 100) * 100
    
    print(f"\nRegion coverage:")
    print(f"  Latitude:  {world_lat_coverage:.1f}% of world")
    print(f"  Longitude: {world_lon_coverage:.1f}% of world")
    print(f"  Area:      {world_area_coverage:.1f}% of world (estimated)")
    
    if world_area_coverage < 10:
        print(f"  → Regional optimization: ~{100/max(world_area_coverage, 0.1):.0f}× faster loading!")
    
    print("="*80 + "\n")

    # Load and process images for the calculated region
    print(f"Loading land mask for region: {min_lat:.1f}° to {max_lat:.1f}°, {min_lon:.1f}° to {max_lon:.1f}°")
    is_water = load_and_process_image(
         max_lat=max_lat, min_lat=min_lat, max_lon=max_lon, min_lon=min_lon
    )
    

    # Precompute TSS combined mask (separation lanes + no-go areas)
    print("\nBuilding TSS combined mask (cached)...")
    lanes_mask, lanes_vecs, no_go_mask = build_tss_combined_mask(
        IMAGE_WIDTH,
        IMAGE_HEIGHT,
        root_dir='.',
        dilation_radius=1,        # Widen separation lanes slightly for easier capture
        no_go_dilation=2,         # Safety margin around restricted areas (IMPROVED)
        supersample_factor=1,     # 2x higher resolution for TSS features
        use_cache=True,
        cache_dir='cache',
        force_recompute=False,
    )

    # Create water mask with TSS lanes integrated
    # PERFORMANCE FIX: Use pre-computed tss_lanes instead of slow GeoJSON processing
    # Set force_recompute=True ONLY the first time, then change to False to use cache
    print("\nCreating buffered water mask with TSS lanes (cached)...")
    buffered_water = create_buffered_water_mask(
        is_water,
        COASTAL_BUFFER_NM,
        force_recompute=False,    # ← Changed to False (use cache after first run)
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

    # Add intermediate waypoints for long segments to speed up A* search
    # Break segments longer than 1000 pixels into smaller chunks
    def add_intermediate_waypoints(waypoints, max_segment_pixels=1000):
        result = [waypoints[0]]
        for i in range(len(waypoints) - 1):
            x1, y1 = waypoints[i]
            x2, y2 = waypoints[i + 1]
            dx = x2 - x1
            dy = y2 - y1
            dist_pixels = math.hypot(dx, dy)
            
            if dist_pixels > max_segment_pixels:
                # Add intermediate waypoints
                num_segments = int(math.ceil(dist_pixels / max_segment_pixels))
                print(f"\nSegment {i}->{i+1} is {dist_pixels:.0f} pixels, breaking into {num_segments} segments")
                for j in range(1, num_segments):
                    t = j / num_segments
                    inter_x = int(x1 + t * dx)
                    inter_y = int(y1 + t * dy)
                    result.append((inter_x, inter_y))
            
            result.append((x2, y2))
        return result
    
    print("\nChecking for long segments...")
    original_waypoint_count = len(pixel_waypoints)
    pixel_waypoints = add_intermediate_waypoints(pixel_waypoints, max_segment_pixels=1000)
    if len(pixel_waypoints) > original_waypoint_count:
        print(f"Added {len(pixel_waypoints) - original_waypoint_count} intermediate waypoints for faster routing")


    # Nudge waypoints that ended up on land to nearest water pixel
    def nudge_to_water(pt, mask, max_search=50):
        x0, y0 = pt
        if 0 <= y0 < mask.shape[0] and 0 <= x0 < mask.shape[1] and mask[y0, x0]:
            return pt
        for r in range(1, max_search+1):
            for dx in range(-r, r+1):
                dy = r
                for dy in (r, -r):
                    x = x0 + dx; y = y0 + dy
                    if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1] and mask[y, x]:
                        return (x, y)
            for dy in range(-r+1, r):
                for dx in (r, -r):
                    x = x0 + dx; y = y0 + dy
                    if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1] and mask[y, x]:
                        return (x, y)
        return pt  # give up
    pixel_waypoints = [nudge_to_water(p, buffered_water) for p in pixel_waypoints]
    
    

    # Set pixel search radius for A* (in pixels) that equals about 10 nm for long routes
    # For 43200 pixels width = 360 degrees, so 120 pixels per degree
    # At mid-latitude, ~60nm per degree
    # So roughly: 10nm / 60nm = 0.167 degrees = 0.167 * 120 = ~20 pixels
    pixel_radius = 20  # Larger steps for long-distance routes (faster search)

    # Calculate optimal exploration angles based on radius
    # For long routes, fewer angles = faster search
    min_angles = 36  # Reduced to 36 for 10° increments (much faster)
    max_angles = 72  # Reduced to 72 for 5° increments (still good coverage)
    exploration_angles = max(min_angles, min(max_angles, int(2 * math.pi * pixel_radius * 0.5)))



    print(f"Using pixel search radius: {pixel_radius} pixels")
    print(f"Using exploration angles: {exploration_angles} (≈{exploration_angles/(2*math.pi*pixel_radius):.1f} angles per pixel)")

    # Initialize TSS-aware A* pathfinding using mask for fast cost adjustments
    # Now includes no_go_mask to avoid restricted areas
    astar = AStar(
        buffered_water,
        tss_preference=True,
        tss_cost_factor=0.6,      # Strong preference for TSS lanes (IMPROVED from 1.0)
        tss_mask=lanes_mask,      # Use separation lanes mask
        tss_vecs=lanes_vecs,      # Use direction vectors for lanes
        no_go_mask=no_go_mask,    # Block areas to avoid
        pixel_radius=pixel_radius,
        exploration_angles=exploration_angles,  # Dynamic based on radius (IMPROVED from 60)
        heuristic_weight=1.5,     # Higher for long routes (IMPROVED: 1.5 for speed)
        max_expansions=10_000_000, # Limit search to prevent infinite loops (10M nodes ≈ 60-90s)
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

        # DISABLED: Re-optimization can be very slow for long routes
        # TODO: Implement smarter re-optimization with max_expansions limit
        segments_to_reoptimize = []
        
        """
        # Find segments that go from TSS (True) to TSS (True) with non-TSS waypoints in between
        print("\nSearching for segments between TSS waypoints with non-TSS points in between...")
        
         # check if start is in TSS, if not, force include it
        if not in_tss_lane[0]:
            in_tss_lane[0] = True

        # check if end is in TSS, if not, force include it
        if not in_tss_lane[-1]:
            in_tss_lane[-1] = True

        # Find all TSS waypoint indices
        tss_indices = [i for i, in_lane in enumerate(in_tss_lane) if in_lane]
        
        if len(tss_indices) >= 2:
           

            for i in range(len(tss_indices) - 1):
                start_idx = tss_indices[i]
                end_idx = tss_indices[i + 1]
                
                # Check if there are waypoints in between
                if end_idx - start_idx > 1:
                    # There are waypoints in between, check if any are non-TSS
                    between_indices = range(start_idx + 1, end_idx)
                    has_non_tss = any(not in_tss_lane[idx] for idx in between_indices)
                    
                    if has_non_tss:
                        segments_to_reoptimize.append((start_idx, end_idx))
                        print(f"  Found segment: waypoint {start_idx} (TSS) -> {end_idx} (TSS) "
                              f"with {end_idx - start_idx - 1} waypoints in between")
        """
        
        """
        # RE-OPTIMIZATION DISABLED - Can be very slow
        if segments_to_reoptimize:
            print(f"\nRe-optimizing {len(segments_to_reoptimize)} segment(s) to stay in TSS lanes...")
            
            # Build new path by replacing segments
            new_path = complete_path.copy()
            offset = 0  # Track index shifts from replacements
            
            for start_idx, end_idx in segments_to_reoptimize:
                # Adjust indices for previous replacements
                adj_start = start_idx + offset
                adj_end = end_idx + offset
                
                start_wp = new_path[adj_start]
                end_wp = new_path[adj_end]

                # Calculate new path for this segment with higher TSS preference
                # Use default step_length (pixel_radius) for reasonable performance
                segment_path = astar.find_path(start_wp, end_wp, tss_cost_factor=0.4)
                
                if segment_path and len(segment_path) > 0:
                    # Replace the segment (keep start, replace middle, keep end)
                    old_segment_len = adj_end - adj_start + 1
                    new_segment_len = len(segment_path)
                    
                    # Remove old segment and insert new one
                    new_path[adj_start:adj_end + 1] = segment_path
                    
                    # Update offset for next iteration
                    offset += (new_segment_len - old_segment_len)
                    
                    print(f"  Replaced segment {start_idx}->{end_idx}: "
                          f"{old_segment_len} waypoints -> {new_segment_len} waypoints")
                else:
                    print(f"  Could not find alternative path for segment {start_idx}->{end_idx}")
            
            # Simplify and recalculate
            print(f"\nOriginal path after segment replacement: {len(new_path)} points")
            simplified_new_path = route_calculator.simplify_straight_lines(new_path)
            print(f"Simplified path: {len(simplified_new_path)} points")
            
            new_distance = route_calculator.calculate_route_distance(simplified_new_path)
            new_in_tss_lane = route_calculator._check_tss_lane_status(simplified_new_path)
            
            new_tss_count = sum(new_in_tss_lane)
            new_tss_percentage = (new_tss_count / len(new_in_tss_lane)) * 100 if new_in_tss_lane else 0
            
            print(f"\nRe-optimized route results:")
            print(f"  Distance: {total_distance:.2f} nm -> {new_distance:.2f} nm "
                  f"(Δ {new_distance - total_distance:+.2f} nm)")
            print(f"  TSS usage: {tss_percentage:.1f}% -> {new_tss_percentage:.1f}% "
                  f"(Δ {new_tss_percentage - tss_percentage:+.1f}%)")
            
            # Update to use new path
            complete_path = simplified_new_path
            total_distance = new_distance
            in_tss_lane = new_in_tss_lane
        else:
            print("No segments found that need re-optimization.")
        """
        
        # Print detailed segment analysis
        print_tss_segments(complete_path, in_tss_lane)
        
        # Export TSS analysis to CSV
        export_tss_analysis(complete_path, in_tss_lane, "exports/tss_analysis.csv")
    else:
        print("TSS lane information not available")

    # Export routes
    export_path_to_csv(complete_path, "./exports/direct_route2.csv")
   

    # Plot results with TSS lanes
    # print("\nGenerating visualization with TSS lanes...")
    tss_geojson_path = "./TSS/separation_lanes_with_direction.geojson"  # Define for plotting
    # plot_route_with_tss(buffered_water, complete_path, tss_geojson_path, pixel_waypoints)

if __name__ == "__main__":
    main()



