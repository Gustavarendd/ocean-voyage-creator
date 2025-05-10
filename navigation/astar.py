"""A* pathfinding implementation for ocean routing."""

import heapq
import math
from config import IMAGE_WIDTH, RADIUS, EXPLORATION_ANGLES
from utils.currents import compute_travel_time

class AStar:
    def __init__(self, U, V, buffered_water, ship_speed):
        self.U = U
        self.V = V
        self.buffered_water = buffered_water
        self.ship_speed = ship_speed
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
                return self._reconstruct_path(came_from, start, goal)

            for neighbor in self._get_neighbors(current, direct):
                cost = compute_travel_time(
                    current[0], current[1],
                    neighbor[0], neighbor[1],
                    self.ship_speed, self.U, self.V, direct
                )
                
                if cost == float('inf'):
                    continue

                tentative_g = g_score[current] + cost
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f = tentative_g + self._heuristic(neighbor, goal)
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
            
            if (0 <= new_y < self.height and 
                self.buffered_water[new_y, wrapped_x]):
                neighbors.append((wrapped_x, new_y))
        
        return neighbors

    def _wrap_x(self, x):
        """Wrap x coordinate around the dateline."""
        if x >= self.width:
            return x - self.width
        elif x < 0:
            return x + self.width
        return x

    def _heuristic(self, a, b):
        """Calculate heuristic distance between two points."""
        dx1 = b[0] - a[0]  # Normal distance
        dx2 = b[0] - a[0] - self.width if b[0] > a[0] else b[0] - a[0] + self.width
        dy = b[1] - a[1]
        
        return math.hypot(dx1 if abs(dx1) < abs(dx2) else dx2, dy) / self.ship_speed

    def _reconstruct_path(self, came_from, start, goal):
        """Reconstruct path from came_from dictionary."""
        path = []
        current = goal
        while current in came_from:
            path.append(current)
            current = came_from[current]
        path.append(start)
        return path[::-1]
