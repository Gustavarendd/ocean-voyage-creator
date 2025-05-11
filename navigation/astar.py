"""A* pathfinding implementation for ocean routing."""

import heapq
import math
from config import RADIUS, EXPLORATION_ANGLES
from utils.currents import compute_travel_time

class AStar:
    def __init__(self, U, V, buffered_water, ship_speed, wave_data=None, max_wave_height=None):
        self.U = U
        self.V = V
        self.buffered_water = buffered_water
        self.ship_speed = ship_speed
        self.wave_data = wave_data
        self.max_wave_height = max_wave_height
        self.height, self.width = U.shape

    def find_path(self, start, goal, direct=False):
        """Find path between start and goal points."""
        open_set = []
        heapq.heappush(open_set, (0, start))
        came_from = {}
        g_score = {start: 0}

        while open_set:
            _, current = heapq.heappop(open_set)
            if current == goal:
                path = self._reconstruct_path(came_from, start, goal)
                if direct:
                    duration = self._calculate_path_duration(path)
                    return path, duration
                return path

            for neighbor in self._get_neighbors(current, direct):
                # Initialize current variables
                current_u = 0
                current_v = 0
                
                if direct:
                    # Calculate cost considering both possible paths across the dateline
                    dx1 = neighbor[0] - current[0]  # Normal distance
                    dx2 = neighbor[0] - current[0] - self.width  # Crossing dateline one way
                    dx3 = neighbor[0] - current[0] + self.width  # Crossing dateline other way
                    dy = neighbor[1] - current[1]
                    
                    # Choose the shortest path
                    dx = min([dx1, dx2, dx3], key=abs)
                    cost = math.hypot(dx, dy)
                else:
                    # Calculate current-based cost
                    dx = neighbor[0] - current[0]
                    dy = neighbor[1] - current[1]
                    dist = math.hypot(dx, dy)
                    if dist > 0:
                        dir_x = dx / dist
                        dir_y = dy / dist
                        
                        # Get current at midpoint
                        mid_x = (current[0] + neighbor[0]) // 2
                        mid_y = (current[1] + neighbor[1]) // 2
                        current_u = self.U[mid_y, self._wrap_x(mid_x)]
                        current_v = -self.V[mid_y, self._wrap_x(mid_x)]
                        
                        # Project current onto movement direction
                        current_along_path = current_u * dir_x + current_v * dir_y
                        current_strength = math.hypot(current_u, current_v)
                        
                        # Stronger incentive for favorable currents
                        current_factor = 1.0
                        # if current_along_path > 0:
                        #     # More aggressive reduction for favorable currents
                        #     current_factor = (1.0 - current_strength / self.ship_speed)
                        # elif current_along_path < 0:
                        #     # Increase penalty for adverse currents
                        #     current_factor = 1 + current_strength / self.ship_speed
                        
                        cost = compute_travel_time(
                            current[0], current[1],
                            neighbor[0], neighbor[1],
                            self.ship_speed, self.U, self.V,
                            self.wave_data
                        ) * current_factor

                        
                if cost == float('inf'):
                    continue

                tentative_g = g_score[current] + cost
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    # Modified heuristic to consider currents
                    f = tentative_g + self._heuristic(neighbor, goal, current_u=current_u, current_v=current_v)
                    heapq.heappush(open_set, (f, neighbor))

        return None

    def _get_neighbors(self, point, direct):
        """Get valid neighboring points."""
        neighbors = []
        radius = RADIUS
        num_directions = EXPLORATION_ANGLES
        
        for i in range(num_directions):
            angle = 2 * math.pi * i / num_directions
            dx = int(round(radius * math.cos(angle)))
            dy = int(round(radius * math.sin(angle)))
            
            new_x = point[0] + dx
            new_y = point[1] + dy
            
            wrapped_x = self._wrap_x(new_x)
            
            # Check if point is within bounds and is water
            if (0 <= new_y < self.height and 
                self.buffered_water[new_y, wrapped_x]):

                # Check wave height constraint if specified
                if self.max_wave_height is not None and self.wave_data is not None:
                    wave_height, _, _ = self.wave_data
                    if wave_height[new_y, wrapped_x] > self.max_wave_height:
                        continue  # Skip this neighbor if wave height exceeds limit
                
                neighbors.append((wrapped_x, new_y))
        
        return neighbors

    def _wrap_x(self, x):
        """Wrap x coordinate around the dateline."""
        while x < 0:
            x += self.width
        while x >= self.width:
            x -= self.width
        return x

    def _heuristic(self, a, b, direct=False, current_u=None, current_v=None):
        """Calculate heuristic distance between two points."""
        # Calculate distances in both directions
        dx1 = b[0] - a[0]  # Normal distance
        dx2 = b[0] - a[0] - self.width  # Crossing dateline one way
        dx3 = b[0] - a[0] + self.width  # Crossing dateline other way
        dy = b[1] - a[1]
        
        # Choose the shortest distance
        dx = min([dx1, dx2, dx3], key=abs)
        
        # For direct routes, use pure geometric distance
        if direct:
            return math.hypot(dx, dy)
        
        # Adjust heuristic based on currents
        if current_u is not None and current_v is not None:
            current_strength = math.hypot(current_u, current_v)
            return math.hypot(dx, dy) / self.ship_speed * (1.0 - current_strength / self.ship_speed)
        
        return math.hypot(dx, dy) / self.ship_speed

    def _reconstruct_path(self, came_from, start, goal):
        """Reconstruct path from came_from dictionary."""
        path = []
        current = goal
        while current in came_from:
            path.append(current)
            current = came_from[current]
        path.append(start)
        return path[::-1]

    def _calculate_path_duration(self, path):
        """Calculate the actual duration of a path considering currents."""
        duration = 0
        for i in range(len(path) - 1):
            current = path[i]
            next_point = path[i + 1]
            time = compute_travel_time(
                current[0], current[1],
                next_point[0], next_point[1],
                self.ship_speed, self.U, self.V,
                self.wave_data
            )
            duration += time
        return duration