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

    def is_point_on_line(self, p1, p2, p3, tolerance=2):
        """Check if point p2 is approximately on the line between p1 and p3.
        
        Args:
            p1, p2, p3: Points as (x, y) tuples
            tolerance: Maximum distance in pixels for considering a point on the line
        
        Returns:
            bool: True if p2 is approximately on the line p1-p3
        """
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        
        # Calculate the distance from point p2 to line p1-p3
        # Using the formula: |Ax + By + C| / sqrt(A² + B²)
        # where line equation is Ax + By + C = 0
        
        # Line from p1 to p3: (y3-y1)x - (x3-x1)y + (x3-x1)y1 - (y3-y1)x1 = 0
        A = y3 - y1
        B = -(x3 - x1)
        C = (x3 - x1) * y1 - (y3 - y1) * x1
        
        # Handle the case where p1 and p3 are the same point
        if A == 0 and B == 0:
            return math.sqrt((x2 - x1)**2 + (y2 - y1)**2) <= tolerance
        
        distance = abs(A * x2 + B * y2 + C) / math.sqrt(A**2 + B**2)
        return distance <= tolerance

    def simplify_straight_lines(self, path):
        """Remove intermediate waypoints that are on straight lines.
        
        Args:
            path: List of (x, y) coordinate tuples
            
        Returns:
            List of (x, y) coordinate tuples with redundant points removed
        """
        if len(path) <= 2:
            return path
        
        simplified_path = [path[0]]  # Always keep the first point
        
        i = 0
        while i < len(path) - 2:
            j = i + 2  # Start checking from two points ahead
            
            # Find the furthest point that still maintains a straight line
            while j < len(path):
                # Check if all points between i and j are on the straight line
                all_on_line = True
                for k in range(i + 1, j):
                    if not self.is_point_on_line(path[i], path[k], path[j]):
                        all_on_line = False
                        break
                
                if not all_on_line:
                    break
                j += 1
            
            # Add the point just before the one that broke the line
            if j > i + 2:  # We found at least one intermediate point to skip
                simplified_path.append(path[j - 1])
                i = j - 1
            else:
                # No intermediate points could be skipped, move to next point
                simplified_path.append(path[i + 1])
                i += 1
        
        # Always keep the last point if it's not already added
        if simplified_path[-1] != path[-1]:
            simplified_path.append(path[-1])
        
        return simplified_path

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
            
            # Add to complete path (avoid duplicating waypoints)
            if not complete_path:
                complete_path.extend(path_segment)
            else:
                complete_path.extend(path_segment[1:])
        
        # Simplify the path by removing intermediate points on straight lines
        print(f"Original path length: {len(complete_path)} points")
        simplified_path = self.simplify_straight_lines(complete_path)
        print(f"Simplified path length: {len(simplified_path)} points")
        print(f"Removed {len(complete_path) - len(simplified_path)} redundant waypoints")
        
        # Calculate total distance using simplified path
        total_distance = self.calculate_route_distance(simplified_path)
        
        return simplified_path, total_distance


