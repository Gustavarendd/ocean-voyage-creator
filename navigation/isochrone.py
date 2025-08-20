"""Isochrone-based pathfinding implementation for ocean routing."""

import math
import numpy as np
from collections import defaultdict, deque
from config import RADIUS, EXPLORATION_ANGLES
from utils.currents import compute_travel_time

class IsochroneRouter:
    def __init__(self, U, V, buffered_water, ship_speed, wave_data=None, max_wave_height=None):
        self.U = U
        self.V = V
        self.buffered_water = buffered_water
        self.ship_speed = ship_speed
        self.wave_data = wave_data
        self.max_wave_height = max_wave_height
        self.height, self.width = U.shape
        
        # Isochrone parameters
        self.time_step = 0.5  # hours 
        self.max_time = 150.0  # hours (maximum search time)
        self.max_iterations = int(self.max_time / self.time_step)

    def find_path(self, start, goal, direct=False):
        """Find path between start and goal points using isochrone method."""
        if direct:
            # For direct routing, use simple straight line
            return self._calculate_direct_path(start, goal)
        
        print(f"Finding isochrone path from {start} to {goal}")
        
        # Initialize isochrone data structures
        isochrones = {}  # time_step -> set of (x, y) positions
        time_field = np.full((self.height, self.width), np.inf)  # arrival time at each point
        parent_field = {}  # (x, y) -> (parent_x, parent_y)
        
        # Initialize starting point
        start_x, start_y = start
        time_field[start_y, start_x] = 0.0
        isochrones[0] = {start}
        
        # Propagate isochrones
        goal_reached = False
        points_explored = 0
        
        for time_step in range(self.max_iterations):
            current_time = time_step * self.time_step
            
            if time_step not in isochrones:
                continue
                
            current_isochrone = isochrones[time_step]
            if not current_isochrone:
                continue
            
            # Progress update every 20 time steps
            if time_step % 20 == 0:
                print(f"Time step {time_step} ({current_time:.1f}h): {len(current_isochrone)} points")
            
            # Check if we've reached the goal
            if goal in current_isochrone:
                print(f"Goal reached at time step {time_step}")
                goal_reached = True
                break
            
            # Check if we're close to the goal (within a reasonable distance)
            for point in current_isochrone:
                dx1 = goal[0] - point[0]
                dx2 = goal[0] - point[0] - self.width
                dx3 = goal[0] - point[0] + self.width
                dx = min([dx1, dx2, dx3], key=abs)
                dy = goal[1] - point[1]
                distance = math.sqrt(dx*dx + dy*dy)
                
                if distance <= RADIUS * 2:  # More generous distance to goal
                    # Mark goal as reachable from this point
                    parent_field[goal] = point
                    print(f"Goal reachable from {point} (distance: {distance:.1f})")
                    goal_reached = True
                    break
            
            if goal_reached:
                break
            
            # Propagate to next time step
            next_time = current_time + self.time_step
            next_time_step = time_step + 1
            
            if next_time_step not in isochrones:
                isochrones[next_time_step] = set()
            
            # Create a copy of the current isochrone to avoid modification during iteration
            current_points = list(current_isochrone)
            new_neighbors_count = 0
            
            for point in current_points:
                neighbors = self._get_neighbors(point)
                points_explored += 1
                
                for neighbor in neighbors:
                    neighbor_x, neighbor_y = neighbor
                    
                    # Calculate travel time to this neighbor
                    travel_time = compute_travel_time(
                        point[0], point[1],
                        neighbor_x, neighbor_y,
                        self.ship_speed, self.U, self.V,
                        self.wave_data
                    )
                    
                    if travel_time == float('inf'):
                        continue
                    
                    arrival_time = current_time + travel_time
                    
                    # Only update if we found a faster route to this point
                    wrapped_x = self._wrap_x(neighbor_x)
                    if arrival_time < time_field[neighbor_y, wrapped_x]:
                        time_field[neighbor_y, wrapped_x] = arrival_time
                        parent_field[neighbor] = point
                        
                        # Add to appropriate isochrone based on arrival time
                        target_time_step = int(arrival_time / self.time_step)
                        if target_time_step < self.max_iterations:
                            if target_time_step not in isochrones:
                                isochrones[target_time_step] = set()
                            isochrones[target_time_step].add(neighbor)
                            new_neighbors_count += 1
            
            if time_step % 20 == 0 and new_neighbors_count > 0:
                print(f"  Added {new_neighbors_count} new neighbors")
            
            # Early termination if no new neighbors are being found
            if time_step > 10 and all(len(isochrones.get(i, set())) == 0 for i in range(time_step + 1, min(time_step + 10, self.max_iterations))):
                print(f"No expansion for several time steps, terminating early at step {time_step}")
                break
        
        print(f"Explored {points_explored} points")
        
        # Return path if goal was reached
        if goal_reached:
            path = self._reconstruct_path_from_isochrones(parent_field, start, goal)
            print(f"Path found with {len(path)} waypoints")
            return path
        
        # If goal not reached, find closest point to goal and trace back
        print("Goal not reached, finding closest path")
        return self._find_closest_path_to_goal(parent_field, start, goal, time_field)

    def _get_neighbors(self, point):
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

    def _reconstruct_path_from_isochrones(self, parent_field, start, goal):
        """Reconstruct path from parent field."""
        path = []
        current = goal
        
        while current != start and current in parent_field:
            path.append(current)
            current = parent_field[current]
        
        path.append(start)
        return path[::-1]

    def _find_closest_path_to_goal(self, parent_field, start, goal, time_field):
        """Find path to point closest to goal if direct path not found."""
        goal_x, goal_y = goal
        min_distance = float('inf')
        closest_point = None
        
        # Find the closest reachable point to the goal
        for y in range(self.height):
            for x in range(self.width):
                if time_field[y, x] < np.inf:  # Point is reachable
                    # Calculate distance to goal considering dateline wrapping
                    dx1 = goal_x - x
                    dx2 = goal_x - x - self.width
                    dx3 = goal_x - x + self.width
                    dx = min([dx1, dx2, dx3], key=abs)
                    dy = goal_y - y
                    distance = math.hypot(dx, dy)
                    
                    if distance < min_distance:
                        min_distance = distance
                        closest_point = (x, y)
        
        if closest_point is None:
            return None
        
        # Reconstruct path to closest point
        return self._reconstruct_path_from_isochrones(parent_field, start, closest_point)

    def _calculate_direct_path(self, start, goal):
        """Calculate direct path for comparison."""
        path = [start]
        
        # Simple interpolation between start and goal
        start_x, start_y = start
        goal_x, goal_y = goal
        
        # Handle dateline crossing
        dx1 = goal_x - start_x
        dx2 = goal_x - start_x - self.width
        dx3 = goal_x - start_x + self.width
        dx = min([dx1, dx2, dx3], key=abs)
        
        if abs(dx2) < abs(dx1) and abs(dx2) < abs(dx3):
            goal_x = goal_x - self.width
        elif abs(dx3) < abs(dx1) and abs(dx3) < abs(dx2):
            goal_x = goal_x + self.width
        
        dy = goal_y - start_y
        
        # Create interpolated path
        steps = max(abs(goal_x - start_x), abs(dy))
        if steps > 0:
            for i in range(1, steps + 1):
                t = i / steps
                x = int(start_x + t * (goal_x - start_x))
                y = int(start_y + t * dy)
                x = self._wrap_x(x)  # Wrap around dateline
                
                # Only add if it's water
                if (0 <= y < self.height and 
                    self.buffered_water[y, x]):
                    path.append((x, y))
        
        if path[-1] != goal:
            path.append(goal)
        
        # Calculate duration
        duration = self._calculate_path_duration(path)
        return path, duration

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
