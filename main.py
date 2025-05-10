"""Main script for ocean route optimization."""

import matplotlib.pyplot as plt
from config import (
    ROUTE_COORDS, COASTAL_BUFFER_NM, SHIP_SPEED_KN,
    IMAGE_WIDTH, IMAGE_HEIGHT
)
from core.initialization import load_and_process_images, extract_currents
from core.mask import create_buffered_water_mask
from navigation.astar import AStar
from navigation.route import RouteCalculator
from utils.coordinates import latlon_to_pixel, validate_coordinates
from visualization.plotting import plot_route, show_water_and_currents
from visualization.export import export_path_to_csv, print_route_analysis

def main():
    # Load and process images
    currents_np, is_water = load_and_process_images(
        "currents_65N_60S_2700x938.png",
        "land_mask_90N_90S_6000x3000.png"
    )
    
    # Extract currents
    U, V = extract_currents(currents_np)
    
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
    
    # Initialize pathfinding
    astar = AStar(U, V, buffered_water, SHIP_SPEED_KN)
    route_calculator = RouteCalculator(U, V, astar)
    
    # Calculate route
    complete_path, complete_direct_path, stats = route_calculator.optimize_route(pixel_waypoints)
    
    if complete_path is None:
        print("No complete path found")
        return
    
    # Export routes
    export_path_to_csv(complete_path, "optimized_route.csv")
    export_path_to_csv(complete_direct_path, "direct_route.csv")
    
    # Print analysis
    print_route_analysis(stats)
    
    # Plot results
    show_water_and_currents(is_water, U, V)
    plot_route(buffered_water, complete_path, complete_direct_path, pixel_waypoints)

if __name__ == "__main__":
    main()



