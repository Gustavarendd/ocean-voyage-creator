"""Route calculation and optimization."""

from utils.coordinates import pixel_to_latlon
from utils.distance import nm_distance
from utils.currents import get_average_current
from config import SHIP_OPERATION, MIN_SPEED
import math

class RouteCalculator:
    def __init__(self, U, V, astar, wave_data=None, max_wave_height=None):
        self.U = U
        self.V = V
        self.astar = astar
        self.wave_data = wave_data
        self.max_wave_height = max_wave_height

    def calculate_route_metrics(self, path_segment):
        """Calculate distance and time for a route segment."""
        segment_dist_nm = 0
        segment_time_hours = 0
        
        for i in range(len(path_segment) - 1):
            x1, y1 = path_segment[i]
            x2, y2 = path_segment[i + 1]
            
            # Check wave height constraint
            if self.max_wave_height is not None and self.wave_data is not None:
                wave_height, _, _ = self.wave_data
                if wave_height[y1, x1] > self.max_wave_height or wave_height[y2, x2] > self.max_wave_height:
                    return float('inf'), float('inf')
            
            # Convert to lat/lon and calculate distance
            seg_lat1, seg_lon1 = pixel_to_latlon(x1, y1)
            seg_lat2, seg_lon2 = pixel_to_latlon(x2, y2)
            sub_segment_dist_nm = nm_distance(seg_lat1, seg_lon1, seg_lat2, seg_lon2)
            
            # Calculate time considering currents
            avg_u, avg_v = get_average_current(x1, y1, x2, y2, self.U, self.V)
            
            # Get direction vector, considering dateline crossing
            dx1 = x2 - x1  # Normal distance
            dx2 = x2 - x1 - self.U.shape[1]  # Crossing dateline one way
            dx3 = x2 - x1 + self.U.shape[1]  # Crossing dateline other way
            dy = y2 - y1
            
            # Choose the shortest horizontal distance
            dx = min([dx1, dx2, dx3], key=abs)
            dist = math.hypot(dx, dy)
            if dist > 0:
                dir_x = dx / dist
                dir_y = dy / dist
                
                # Calculate current effect along the shortest path
                current_along_path = avg_u * dir_x + avg_v * dir_y

                if self.wave_data is not None:
                    wave_height, wave_period, wave_direction = self.wave_data
                    ship_heading = math.degrees(math.atan2(-dir_y, dir_x))
                    rel_wave_dir = (wave_direction[y1, x1] - ship_heading) % 360

                    from utils.ship_performance import calculate_net_speed
                    net_speed = calculate_net_speed(
                        SHIP_OPERATION['speed_through_water'],
                        current_along_path,
                        wave_height[y1, x1],
                        wave_period[y1, x1],
                        rel_wave_dir
                    )
                else:
                    net_speed = max(MIN_SPEED, SHIP_OPERATION['speed_through_water'] + current_along_path)
                    
                sub_segment_time = sub_segment_dist_nm / net_speed
                segment_time_hours += sub_segment_time
            
            segment_dist_nm += sub_segment_dist_nm
        
        return segment_dist_nm, segment_time_hours

    def optimize_route(self, waypoints):
        """Find optimized path through all waypoints."""
        complete_optimized_path = []
        complete_direct_path = []
        total_stats = {
            'optimized_dist': 0,
            'optimized_time': 0,
            'direct_dist': 0,
            'direct_time': 0,
            'true_direct_time': 0
        }
        
        for i in range(len(waypoints) - 1):
            start = waypoints[i]
            goal = waypoints[i + 1]
            
            # Calculate optimized path
            path_segment = self.astar.find_path(start, goal)
            if not path_segment:
                return None, None, None
            
            # Calculate direct path
            direct_result = self.astar.find_path(start, goal, direct=True)
            if direct_result:
                direct_path, direct_duration = direct_result
                complete_direct_path.extend(direct_path if not complete_direct_path else direct_path[1:])
                direct_dist, direct_time = self.calculate_route_metrics(direct_path)
                true_direct_time = sum(
                    nm_distance(*pixel_to_latlon(x1, y1), *pixel_to_latlon(x2, y2)) / SHIP_OPERATION['speed_through_water']
                    for (x1, y1), (x2, y2) in zip(direct_path, direct_path[1:])
                )
                
                total_stats['direct_dist'] += direct_dist
                total_stats['direct_time'] += direct_time
                total_stats['true_direct_time'] += true_direct_time
            
            # Calculate optimized route metrics
            opt_dist, opt_time = self.calculate_route_metrics(path_segment)
            total_stats['optimized_dist'] += opt_dist
            total_stats['optimized_time'] += opt_time
            
            complete_optimized_path.extend(path_segment if not complete_optimized_path else path_segment[1:])
        
        return complete_optimized_path, complete_direct_path, total_stats


