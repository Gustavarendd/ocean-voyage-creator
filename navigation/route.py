"""Route calculation and optimization - simplified for distance-only routing."""

from utils.coordinates import pixel_to_latlon
from utils.distance import nm_distance
import math

class RouteCalculator:
    def __init__(self, router):
        self.router = router

    def calculate_route_distance(self, path_segment):
        """Calculate total distance for a route segment."""
        total_distance_nm = 0
        
        for i in range(len(path_segment) - 1):
            x1, y1 = path_segment[i]
            x2, y2 = path_segment[i + 1]
            
            # Convert to lat/lon and calculate distance
            lat1, lon1 = pixel_to_latlon(x1, y1)
            lat2, lon2 = pixel_to_latlon(x2, y2)
            segment_distance_nm = nm_distance(lat1, lon1, lat2, lon2)
            
            total_distance_nm += segment_distance_nm
        
        return total_distance_nm

    def optimize_route(self, waypoints):
        """Find shortest distance path through all waypoints."""
        complete_path = []
        total_distance = 0
        
        for i in range(len(waypoints) - 1):
            start = waypoints[i]
            goal = waypoints[i + 1]
            
            # Calculate shortest path
            path_segment = self.router.find_path(start, goal)
            if not path_segment:
                print(f"No path found between waypoint {i} and {i + 1}")
                return None, None
            
            # Calculate distance for this segment
            segment_distance = self.calculate_route_distance(path_segment)
            total_distance += segment_distance
            
            # Add to complete path (avoid duplicating waypoints)
            if not complete_path:
                complete_path.extend(path_segment)
            else:
                complete_path.extend(path_segment[1:])
        
        return complete_path, total_distance


