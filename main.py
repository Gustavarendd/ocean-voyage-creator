"""Main script for ocean route optimization - simplified for distance-only routing."""

from config import ROUTE_COORDS, COASTAL_BUFFER_NM, IMAGE_WIDTH, IMAGE_HEIGHT
from core.initialization import load_and_process_divided_image, load_and_process_image
from core.mask import create_buffered_water_mask
from navigation.astar import AStar
from navigation.tss_index import build_tss_mask, build_tss_combined_mask


from navigation.route import RouteCalculator
from utils.coordinates import latlon_to_pixel, validate_coordinates, pixel_to_latlon
from visualization.export import export_path_to_csv
from visualization.plotting import plot_route, plot_route_with_tss


def main():

    # find min/max lat/lon from ROUTE_COORDS
    # lats = [lat for lat, lon in ROUTE_COORDS]
    # lons = [lon for lat, lon in ROUTE_COORDS]
    # min_lat = min(lats)
    # max_lat = max(lats)
    # min_lon = min(lons)
    # max_lon = max(lons)


    # Derive working image dimensions aiming for ~1 px per nautical mile.
    # 1 degree latitude ~ 60 nm; 1 degree longitude ~ 60 * cos(mid_lat) nm.
    # We already pad by +/-10 deg in load_and_process_images call; include that here.
    # lat_span_deg = (max_lat - min_lat) + 20
    # lon_span_deg = (max_lon - min_lon) + 20
    # mid_lat_rad = math.radians((max_lat + min_lat) / 2.0)
    # nm_height = lat_span_deg * 60.0 * 4
    # nm_width = lon_span_deg * 60.0 * 4 * max(0.1, math.cos(mid_lat_rad))  # guard near poles
    # IMAGE_HEIGHT = int(max(1, round(nm_height)))
    # IMAGE_WIDTH = int(max(1, round(nm_width)))
    # print(f"Using image dimensions: {IMAGE_WIDTH} x {IMAGE_HEIGHT}")



    min_lat = -80
    max_lat = 80
    min_lon = -170
    max_lon = 170

    # Load and process images
    # is_water = load_and_process_image(
    #      max_lat=max_lat + 10, min_lat=min_lat - 10, max_lon=max_lon + 10, min_lon=min_lon - 10
    # )
    is_water = load_and_process_image(
         max_lat=max_lat + 10, min_lat=min_lat - 10, max_lon=max_lon + 10, min_lon=min_lon - 10
    )
    

    # Precompute TSS combined mask (separation lanes + no-go areas)
    print("\nBuilding TSS combined mask (cached)...")
    lanes_mask, lanes_vecs, no_go_mask = build_tss_combined_mask(
        IMAGE_WIDTH,
        IMAGE_HEIGHT,
        root_dir='.',
        dilation_radius=0,        # Widen separation lanes slightly
        no_go_dilation=0,         # Widen no-go areas for safety
        supersample_factor=2,     # 2x higher resolution for TSS features
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
        pixel_radius=pixel_radius * 5, # Search radius in pixels
        exploration_angles=60,   # Wider search angles for better pathfinding
        heuristic_weight=1.0,       # Standard A* heuristic less makes better paths but slower
        max_expansions=None,       # No limit on expansions (can be set for performance)
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
    # if in_tss_lane:
    #     tss_count = sum(in_tss_lane)
    #     tss_percentage = (tss_count / len(in_tss_lane)) * 100
    #     print(f"TSS lane usage: {tss_count}/{len(in_tss_lane)} waypoints ({tss_percentage:.1f}%)")

    #     # Find segments that go from TSS (True) to TSS (True) with non-TSS waypoints in between
    #     print("\nSearching for segments between TSS waypoints with non-TSS points in between...")
    #     segments_to_reoptimize = []
        
    #      # check if start is in TSS, if not, force include it
    #     if not in_tss_lane[0]:
    #         in_tss_lane[0] = True

    #     # check if end is in TSS, if not, force include it
    #     if not in_tss_lane[-1]:
    #         in_tss_lane[-1] = True

    #     # Find all TSS waypoint indices
    #     tss_indices = [i for i, in_lane in enumerate(in_tss_lane) if in_lane]
        
    #     if len(tss_indices) >= 2:
           

    #         for i in range(len(tss_indices) - 1):
    #             start_idx = tss_indices[i]
    #             end_idx = tss_indices[i + 1]
                
    #             # Check if there are waypoints in between
    #             if end_idx - start_idx > 1:
    #                 # There are waypoints in between, check if any are non-TSS
    #                 between_indices = range(start_idx + 1, end_idx)
    #                 has_non_tss = any(not in_tss_lane[idx] for idx in between_indices)
                    
    #                 if has_non_tss:
    #                     segments_to_reoptimize.append((start_idx, end_idx))
    #                     print(f"  Found segment: waypoint {start_idx} (TSS) -> {end_idx} (TSS) "
    #                           f"with {end_idx - start_idx - 1} waypoints in between")
        
    #     if segments_to_reoptimize:
    #         print(f"\nRe-optimizing {len(segments_to_reoptimize)} segment(s) to stay in TSS lanes...")
            
    #         # Build new path by replacing segments
    #         new_path = complete_path.copy()
    #         offset = 0  # Track index shifts from replacements
            
    #         for start_idx, end_idx in segments_to_reoptimize:
    #             # Adjust indices for previous replacements
    #             adj_start = start_idx + offset
    #             adj_end = end_idx + offset
                
    #             start_wp = new_path[adj_start]
    #             end_wp = new_path[adj_end]


                
    #             # Calculate new path for this segment
    #             segment_path = astar.find_path(start_wp, end_wp, step_length = pixel_radius * 3, tss_cost_factor=0.5)
                
    #             if segment_path and len(segment_path) > 0:
    #                 # Replace the segment (keep start, replace middle, keep end)
    #                 old_segment_len = adj_end - adj_start + 1
    #                 new_segment_len = len(segment_path)
                    
    #                 # Remove old segment and insert new one
    #                 new_path[adj_start:adj_end + 1] = segment_path
                    
    #                 # Update offset for next iteration
    #                 offset += (new_segment_len - old_segment_len)
                    
    #                 print(f"  Replaced segment {start_idx}->{end_idx}: "
    #                       f"{old_segment_len} waypoints -> {new_segment_len} waypoints")
    #             else:
    #                 print(f"  Could not find alternative path for segment {start_idx}->{end_idx}")
            
    #         # Simplify and recalculate
    #         print(f"\nOriginal path after segment replacement: {len(new_path)} points")
    #         simplified_new_path = route_calculator.simplify_straight_lines(new_path)
    #         print(f"Simplified path: {len(simplified_new_path)} points")
            
    #         new_distance = route_calculator.calculate_route_distance(simplified_new_path)
    #         new_in_tss_lane = route_calculator._check_tss_lane_status(simplified_new_path)
            
    #         new_tss_count = sum(new_in_tss_lane)
    #         new_tss_percentage = (new_tss_count / len(new_in_tss_lane)) * 100 if new_in_tss_lane else 0
            
    #         print(f"\nRe-optimized route results:")
    #         print(f"  Distance: {total_distance:.2f} nm -> {new_distance:.2f} nm "
    #               f"(Δ {new_distance - total_distance:+.2f} nm)")
    #         print(f"  TSS usage: {tss_percentage:.1f}% -> {new_tss_percentage:.1f}% "
    #               f"(Δ {new_tss_percentage - tss_percentage:+.1f}%)")
            
    #         # Update to use new path
    #         complete_path = simplified_new_path
    #         total_distance = new_distance
    #         in_tss_lane = new_in_tss_lane
    #     else:
    #         print("No segments found that need re-optimization.")
        
    #     # Print detailed segment analysis
    #     print_tss_segments(complete_path, in_tss_lane)
        
    #     # Export TSS analysis to CSV
    #     export_tss_analysis(complete_path, in_tss_lane, "exports/tss_analysis.csv")
    # else:
    #     print("TSS lane information not available")

    # Export routes
    # export_path_to_csv(complete_path, "./exports/direct_route2.csv")
   

    # Plot results with TSS lanes
    # print("\nGenerating visualization with TSS lanes...")
    tss_geojson_path = "./TSS/separation_lanes_with_direction.geojson"  # Define for plotting
    plot_route_with_tss(buffered_water, complete_path, tss_geojson_path, pixel_waypoints)

if __name__ == "__main__":
    main()



