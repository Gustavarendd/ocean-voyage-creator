"""Main script for ocean route optimization - simplified for distance-only routing."""

from config import ROUTE_COORDS, COASTAL_BUFFER_NM
from core.initialization import load_and_process_images, extract_currents, extract_waves
from core.mask import create_buffered_water_mask
from navigation.astar import AStar
from navigation.route import RouteCalculator
from utils.coordinates import latlon_to_pixel, validate_coordinates, pixel_to_latlon
from visualization.export import export_path_to_csv
from visualization.plotting import plot_route

def main():
    # Load and process images
    is_water = load_and_process_images(
        "./images/currents_65N_60S_2700x938.png",
        "./images/land_mask_90N_90S_6000x3000.png"
    )
    
    # Extract currents and waves (for potential future use)
    # U, V = extract_currents(currents_np)
    # wave_height, wave_period, wave_direction = extract_waves(wave_np)
    
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

    # Initialize simplified A* pathfinding (distance-only)
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
    
    # call http://127.0.0.1:8000/tss/correct_geojson with the waypoints for the direct path
    # def export_corrected_geojson():
    #     import requests
    #     url = "http://127.0.0.1:8000/tss/correct_geojson"
    #     waypoints = [(pixel_to_latlon(x, y)) for x, y in complete_path]
    #     # Convert waypoints to [{lat: 45, lon: 3}] format
    #     latlon_waypoints = [{"lat": lat, "lon": lon} for lat, lon in waypoints]

    #     response = requests.post(url, json={
    #                 "waypoints": latlon_waypoints, 
    #                 "max_snap_m": 8000, 
    #                 "include_bridging": True, 
    #                 "sample_spacing_m": 10,
    #                 "multi_clusters": True,
    #                 "debug": True})
    #     if response.status_code == 200:
    #         print("GeoJSON corrected successfully")

  
    # export_corrected_geojson()

    # Plot results
    # show_water_and_currents(is_water, U, V)
    # plot_route(wave_np, complete_optimized_path, complete_direct_path, pixel_waypoints)
    plot_route(buffered_water, complete_path)

if __name__ == "__main__":
    main()



