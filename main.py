"""Main script for ocean route optimization - simplified for distance-only routing."""

from config import ROUTE_COORDS, COASTAL_BUFFER_NM, IMAGE_WIDTH, IMAGE_HEIGHT
from core.initialization import load_and_process_images
from core.mask import create_buffered_water_mask
from navigation.astar import AStar
from navigation.tss_index import build_tss_mask


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
    is_water = load_and_process_images(
        f"./images/land_mask_90N_90S_21600x10800.png", max_lat=max_lat + 10, min_lat=min_lat - 10, max_lon=max_lon + 10, min_lon=min_lon - 10
    )
    
    # Create water mask, converting separation zones into land so routing avoids them.
    # Set force_recompute=True the first time after introducing this feature so cache updates.
    tss_geojson_path = "./TSS/separation_lanes_with_direction.geojson"
    buffered_water = create_buffered_water_mask(
        is_water,
        COASTAL_BUFFER_NM,
        force_recompute=True,
        tss_geojson_path=tss_geojson_path,
        land_lane_types=["separation_zone", "inshore_traffic_zone", "separation_line", "separation_boundary", "area_to_avoid"],
        water_lane_types=["separation_lane"],
        lane_pixel_width=3,
        apply_tss_before_buffer=False,
        supersample_factor=1,
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
    
    # Precompute TSS mask (optional dilation_radius widens lane influence)
    print("\nBuilding TSS mask (cached)...")
    tss_mask = build_tss_mask(
        buffered_water.shape[1],
        buffered_water.shape[0],
        root_dir='.',
        dilation_radius=1,
        use_cache=True,
        cache_dir='cache',
        force_recompute=False,
    )

    # Set pixel search radius for A* (in pixels) that equals about 5 nm
    pixel_radius = int(5 * 1852 / ((40075000 / 360) * abs(((max_lat + 10) - (min_lat - 10)) / buffered_water.shape[0]))) 

    print(f"Using pixel search radius: {pixel_radius} pixels")

    # Initialize TSS-aware A* pathfinding using mask for fast cost adjustments
    astar = AStar(
        buffered_water,
        tss_preference=True,
        tss_cost_factor=1,
        tss_search_radius_m=15000,
        tss_mask=tss_mask[0],
        tss_vecs=tss_mask[1],
        pixel_radius=pixel_radius,
        exploration_angles=360,
        heuristic_weight=1,
        max_expansions=None,
    )
    
    # Initialize simplified route calculator
    route_calculator = RouteCalculator(astar)
    
    # Calculate route
    print("\nCalculating route...")
    complete_path, total_distance = route_calculator.optimize_route(pixel_waypoints)

    if complete_path is None:
        print("No complete path found")
        complete_path = []
        total_distance = 0.0

    print(f"\nRoute calculated successfully!")
    print(f"Total distance: {total_distance:.2f} nautical miles")

    # Export routes
    export_path_to_csv(complete_path, "./exports/direct_route.csv")
   

    # Plot results with TSS lanes
    # print("\nGenerating visualization with TSS lanes...")
    plot_route_with_tss(buffered_water, complete_path, tss_geojson_path, pixel_waypoints)

if __name__ == "__main__":
    main()



