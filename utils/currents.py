"""Current calculation utilities."""

import math
from config import (
    IMAGE_WIDTH, MIN_SPEED,
    
)

def compute_travel_time(x1, y1, x2, y2, ship_speed, U, V, direct=False):
    """Compute travel time between two points considering ocean currents."""
    # Handle dateline crossing
    dx1 = x2 - x1  # Normal distance
    dx2 = x2 - x1 - IMAGE_WIDTH if x2 > x1 else x2 - x1 + IMAGE_WIDTH  # Across dateline
    dy = y2 - y1
    
    # Use the shorter distance
    dx = dx1 if abs(dx1) <= abs(dx2) else dx2
    
    dist = math.hypot(dx, dy)
    if dist == 0:
        return 0
    
    dir_x = dx / dist
    dir_y = dy / dist
    
    # Handle wrapping for current sampling
    x1_wrapped = x1 % IMAGE_WIDTH
    x2_wrapped = x2 % IMAGE_WIDTH
    
    # Get average current
    avg_u, avg_v = get_average_current(x1_wrapped, y1, x2_wrapped, y2, U, V)
    
    # Project current onto movement direction
    current_along_path = avg_u * dir_x + avg_v * dir_y
    
    net_speed = calculate_net_speed(ship_speed, current_along_path)
    
    if net_speed <= MIN_SPEED:
        return float('inf')
        
    time = dist / net_speed

    
    return time

def get_average_current(x1, y1, x2, y2, U, V):
    """Calculate average current vector between two points."""
    current_u1 = U[y1, x1]
    current_v1 = -V[y1, x1]
    current_u2 = U[y2, x2]
    current_v2 = -V[y2, x2]
    
    return (current_u1 + current_u2) / 2, (current_v1 + current_v2) / 2

def calculate_net_speed(ship_speed, current_along_path):
    """Calculate net speed considering current effects."""
    net_speed = ship_speed + current_along_path
    
    
    
    return net_speed
