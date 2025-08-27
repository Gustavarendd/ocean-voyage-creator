"""Main script for ocean route optimization - simplified for distance-only routing."""

from config import ROUTE_COORDS, COASTAL_BUFFER_NM
from core.initialization import load_and_process_images
from core.mask import create_buffered_water_mask
from navigation.astar import AStar

from navigation.route import RouteCalculator
from utils.coordinates import latlon_to_pixel, validate_coordinates, pixel_to_latlon
from visualization.export import export_path_to_csv
from visualization.plotting import plot_route, plot_route_with_tss

def main():
    # Load and process images
    is_water = load_and_process_images(
        "./images/land_mask_90N_90S_6000x3000.png"
    )
    
    # Create water mask
    buffered_water = create_buffered_water_mask(is_water, COASTAL_BUFFER_NM, False)
   
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

    # Initialize lightweight TSS-aware A* pathfinding
    # tss_geojson_path = "./TSS/separation_lanes_with_direction.geojson"
    astar = AStar(buffered_water)
    
    # Initialize simplified route calculator
    route_calculator = RouteCalculator(astar)
    
    # Calculate route
    complete_path, total_distance = route_calculator.optimize_route(pixel_waypoints)

    if complete_path is None:
        print("No complete path found")
        return

    print(f"\nRoute calculated successfully!")
    print(f"Total distance: {total_distance:.2f} nautical miles")

    # Export routes
    export_path_to_csv(complete_path, "./exports/direct_route.csv")
   

    # Plot results with TSS lanes
    print("\nGenerating visualization with TSS lanes...")
    plot_route_with_tss(buffered_water, complete_path, None, pixel_waypoints)

if __name__ == "__main__":
    main()



