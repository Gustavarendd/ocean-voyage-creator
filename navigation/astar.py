"""A* pathfinding implementation for ocean routing - shortest distance only."""

import heapq
import math
from config import RADIUS, EXPLORATION_ANGLES

class AStar:
    def __init__(self, buffered_water):
        self.buffered_water = buffered_water
        self.height, self.width = buffered_water.shape

    def find_path(self, start, goal):
        """Find shortest distance path between start and goal points."""
        open_set = []
        heapq.heappush(open_set, (0, start))
        came_from = {}
        g_score = {start: 0}

        while open_set:
            _, current = heapq.heappop(open_set)
            if current == goal:
                return self._reconstruct_path(came_from, start, goal)

            for neighbor in self._get_neighbors(current):
                # Calculate cost considering both possible paths across the dateline
                dx1 = neighbor[0] - current[0]  # Normal distance
                dx2 = neighbor[0] - current[0] - self.width  # Crossing dateline one way
                dx3 = neighbor[0] - current[0] + self.width  # Crossing dateline other way
                dy = neighbor[1] - current[1]
                
                # Choose the shortest path
                dx = min([dx1, dx2, dx3], key=abs)
                cost = math.hypot(dx, dy)

                tentative_g = g_score[current] + cost
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f = tentative_g + self._heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f, neighbor))

        return None

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
            
            if (0 <= new_y < self.height and 
                self.buffered_water[new_y, wrapped_x]):
                neighbors.append((wrapped_x, new_y))
        
        return neighbors

    def _wrap_x(self, x):
        """Wrap x coordinate around the dateline."""
        while x < 0:
            x += self.width
        while x >= self.width:
            x -= self.width
        return x

    def _heuristic(self, a, b):
        """Calculate heuristic distance between two points."""
        # Calculate distances in both directions
        dx1 = b[0] - a[0]  # Normal distance
        dx2 = b[0] - a[0] - self.width  # Crossing dateline one way
        dx3 = b[0] - a[0] + self.width  # Crossing dateline other way
        dy = b[1] - a[1]
        
        # Choose the shortest distance
        dx = min([dx1, dx2, dx3], key=abs)
        
        return math.hypot(dx, dy)

    def _reconstruct_path(self, came_from, start, goal):
        """Reconstruct path from came_from dictionary."""
        path = []
        current = goal
        while current in came_from:
            path.append(current)
            current = came_from[current]
        path.append(start)
        return path[::-1]