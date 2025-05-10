"""Export functions for route data."""

import csv
import math
from utils.coordinates import pixel_to_latlon

def export_path_to_csv(path_pixels, output_path="../exports/route.csv"):
    """Export route waypoints to CSV file."""
    with open(output_path, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["lat", "lon"])
        
        for x, y in path_pixels:
            lat, lon = pixel_to_latlon(x, y)
            writer.writerow([lat, lon])
    
    print(f"Route exported to: {output_path}")

def print_route_analysis(stats):
    """Print analysis of route optimization."""
    print("\n=== Complete Route Analysis ===")
    print(f"Direct route total:")
    print(f"  Distance: {stats['direct_dist']:.1f} nautical miles")
    
    if math.isinf(stats['direct_time']):
        print("  Duration: Not possible (route crosses land)")
    else:
        print(f"  Duration (ignoring currents): {stats['true_direct_time']:.1f} hours")
        print(f"  Duration (with currents): {stats['direct_time']:.1f} hours")
    
    print(f"\nOptimized route total:")
    print(f"  Distance: {stats['optimized_dist']:.1f} nautical miles")
    print(f"  Duration: {stats['optimized_time']:.1f} hours")
    
    distance_diff = stats['direct_dist'] - stats['optimized_dist']
    print(f"\nComparison:")
    print(f"  Distance difference: {abs(distance_diff):.1f} nautical miles "
          f"{'longer' if distance_diff < 0 else 'shorter'}")
    
    if not math.isinf(stats['direct_time']):
        time_diff = stats['direct_time'] - stats['optimized_time']
        print(f"  Time savings: {abs(time_diff):.1f} hours "
              f"{'longer' if time_diff < 0 else 'shorter'}")
