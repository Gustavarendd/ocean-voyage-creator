"""Main script for ocean route optimization."""
from shapely.geometry import Point

from config import (
    ROUTE_COORDS, COASTAL_BUFFER_NM, SHIP_OPERATION, TSS_AREA, PREDEFINED_SEGMENTS
)
from core.initialization import load_and_process_images, extract_currents, extract_waves
from core.mask import create_buffered_water_mask
from navigation.astar import AStar
from navigation.route import RouteCalculator
from utils.coordinates import latlon_to_pixel, validate_coordinates, pixel_to_latlon
from visualization.plotting import plot_route, show_water_and_currents
from visualization.export import export_path_to_csv, print_route_analysis

# def is_in_TSS_area(lat, lon):
#     """Check if a point is inside the TSS area."""
#     pixel_waypoint = pixel_to_latlon(lat, lon)
#     point = Point(pixel_waypoint[1], pixel_waypoint[0])  # Note: Shapely uses (lon, lat) order
#     return TSS_AREA.contains(point)

# def enforce_TSS_area_waypoints(pixel_waypoints):
#     """Enforce predefined waypoints if the route enters the TSS area."""
#     print(f"Checking waypoint: {pixel_waypoints}")
#     new_waypoints = []
#     for lat, lon in pixel_waypoints:
#         if is_in_TSS_area(lat, lon):
#             print("Entering TSS area. Using predefined waypoints.")

#             # Find the segment that starts with WP closest to the lat and lon
#             for segment_name, waypoints in PREDEFINED_SEGMENTS.items():
#                 for i, (seg_lat, seg_lon) in enumerate(waypoints):
#                     seg_pixel_waypoints = latlon_to_pixel(seg_lat, seg_lon)
#                     if abs(seg_pixel_waypoints[0] - lat) < 11 and abs(seg_pixel_waypoints[1] - lon) < 110:
#                         new_waypoints.extend([latlon_to_pixel(wp_lat, wp_lon) for wp_lat, wp_lon in waypoints[i:]])
#                         break
#                 else:
#                     continue
#                 break
#         else:
#             new_waypoints.append((lat, lon))
#     return new_waypoints


def main():
    # Load and process images
    currents_np, wave_np, is_water = load_and_process_images(
        "./images/currents_65N_60S_2700x938.png",
        "./images/land_mask_90N_90S_6000x3000.png",
        "./images/wave2.png"
    )
    
    # Extract currents
    U, V = extract_currents(currents_np)

    # Extract wave data
    wave_height, wave_period, wave_direction = extract_waves(wave_np)
    
    # Create water mask
    buffered_water = create_buffered_water_mask(is_water, COASTAL_BUFFER_NM)
    
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

    
    
    # Prepare wave data tuple
    wave_data = (wave_height, wave_period, wave_direction)
    
    # Initialize pathfinding with wave data and max wave height constraint
    max_wave_height = 3  # Example: Avoid areas with wave height > 3m
    astar = AStar(U, V, buffered_water, SHIP_OPERATION['speed_through_water'], wave_data, max_wave_height)
    
    # Initialize route calculator with wave data and max wave height constraint
    route_calculator = RouteCalculator(U, V, astar, wave_data, max_wave_height=max_wave_height)
    
    # Calculate route
    complete_optimized_path, complete_direct_path, stats = route_calculator.optimize_route(pixel_waypoints)

    # complete_direct_path = enforce_TSS_area_waypoints(complete_direct_path)


    if complete_optimized_path is None:
        print("No complete path found")
        return
   


    # Export routes
    export_path_to_csv(complete_optimized_path, "./exports/optimized_route.csv")
    export_path_to_csv(complete_direct_path, "./exports/direct_route.csv")
    
    # Print analysis
    print_route_analysis(stats)
    
    # Plot results
    show_water_and_currents(is_water, U, V)
    plot_route(wave_np, complete_optimized_path, complete_direct_path, pixel_waypoints)
    plot_route(buffered_water, complete_optimized_path, complete_direct_path, pixel_waypoints)

if __name__ == "__main__":
    main()



