from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import math
import heapq
import csv
from scipy import ndimage

# === CONFIG ===
IMAGE_WIDTH = 2700  # Width of the currents image
IMAGE_HEIGHT = 938  # Height of the currents image

# Latitude and longitude bounds for the currents data
LAT_MIN = -60.0  # 60°S
LAT_MAX = 65.0   # 65°N
LON_MIN = -180.0
LON_MAX = 180.0

SHIP_SPEED_KN = 15.0  # ship speed in knots
RADIUS = 1  # Increase to allow larger steps
ROUTE_COORDS = [
    (32, 32), (36, -5) #Port-Said -> Gibraltar

 #  (10, -80), (40, -73.5) #Panama Canal -> New York

#(29.5, -88), (40, -73) #New Orleans -> New York

  # (25.5, -80), (40, -73.5) #Miami -> New York

#(25.5, 122.5),(58, -148) #Taiwan -> Alaska

#(38.7, -70.9), (30.37, -80.134)
]
COASTAL_BUFFER_NM = 12  # Coastal buffer in nautical miles
EXPLORATION_ANGLES = 16  # Reduced to allow more variation between steps


def create_buffered_water_mask(is_water, buffer_nm):
    """Create a water mask with a coastal buffer zone."""
    # Calculate buffer size in pixels
    # Base size at equator (1 degree = 60nm)
    base_pixels_per_nm = (IMAGE_WIDTH / 360) / 60
    
    # We use a larger buffer to account for latitude distortion
    buffer_pixels = int(buffer_nm * base_pixels_per_nm * 2)
    
    # Create a disk-shaped structuring element for the erosion
    structure = ndimage.iterate_structure(
        ndimage.generate_binary_structure(2, 1),
        buffer_pixels
    )
    
    # Create the buffered mask using binary erosion
    buffered_mask = ndimage.binary_erosion(is_water, structure=structure)
    
    return buffered_mask

# === Load Images ===
currents_img = Image.open("currents_65N_60S_2700x938.png")
currents_np = np.array(currents_img)

# Load and process land mask to match currents coverage and resolution
land_mask = Image.open("land_mask_90N_90S_5400x2700.png").convert("L")
land_mask_np = np.array(land_mask)

# Calculate the crop indices for the land mask to match currents coverage
full_height = 2700  # Full height of land mask (90°N to 90°S)
full_lat_range = 180  # Total latitude range in the land mask

# Calculate pixel positions for 65°N and 60°S in the original land mask
north_limit_px = int((90 - LAT_MAX) / 180 * full_height)  # 65°N
south_limit_px = int((90 - LAT_MIN) / 180 * full_height)  # 60°S

# Crop the land mask to match the latitude range of currents data
land_mask_cropped = land_mask_np[north_limit_px:south_limit_px, :]

# Resize the cropped land mask to match currents resolution
land_mask_resized = Image.fromarray(land_mask_cropped).resize(
    (IMAGE_WIDTH, IMAGE_HEIGHT), 
    Image.Resampling.BILINEAR
)
land_mask_np = np.array(land_mask_resized)

# Invert the land/water logic since the mask might be inverted
is_water = land_mask_np < 128  # True for water, False for land

# Create buffered water mask with coastal exclusion zone
buffered_water = create_buffered_water_mask(is_water, COASTAL_BUFFER_NM)

# === Extract Currents ===
def scale_channel(channel, min_val, max_val):
    return min_val + (channel / 255.0) * (max_val - min_val)

R, G, B = currents_np[:, :, 0], currents_np[:, :, 1], currents_np[:, :, 2]
U = scale_channel(R, -1.857, 2.035)
V = scale_channel(G, -1.821, 2.622)


def compute_travel_time(x1, y1, x2, y2, ship_speed, U, V, direct=False):
    # Handle dateline crossing
    dx1 = x2 - x1  # Normal distance
    dx2 = x2 - x1 - IMAGE_WIDTH if x2 > x1 else x2 - x1 + IMAGE_WIDTH  # Across dateline
    dy = y2 - y1
    
    # Use the shorter distance
    if abs(dx1) <= abs(dx2):
        dx = dx1
    else:
        dx = dx2
    
    dist = math.hypot(dx, dy)
    
    if dist == 0:
        return 0
    
    dir_x = dx / dist
    dir_y = dy / dist
    
    # Handle wrapping for current sampling
    x1_wrapped = x1 % IMAGE_WIDTH
    x2_wrapped = x2 % IMAGE_WIDTH
    
    current_u = U[y1, x1_wrapped]
    current_v = -V[y1, x1_wrapped]  # invert for image coords
    
    # Project current onto movement direction
    current_along_path = current_u * dir_x + current_v * dir_y
    if direct:
        # For direct path, we assume no current effect
        net_speed = ship_speed
    else:
        # For normal path, consider the current effect
        net_speed = ship_speed + current_along_path
    
    if net_speed <= 0.01:
        return float('inf')  # ship can't move
    return dist / net_speed


def astar(start, goal, U, V, buffered_water, ship_speed, direct=False):
    height, width = U.shape
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}

    def wrap_x(x):
        """Wrap x coordinate around the dateline"""
        if x >= width:
            return x - width
        elif x < 0:
            return x + width
        return x

    def heuristic(a, b):
        # Calculate both possible distances (across dateline and normal)
        dx1 = b[0] - a[0]  # Normal distance
        dx2 = b[0] - a[0] - width if b[0] > a[0] else b[0] - a[0] + width  # Across dateline
        dy = b[1] - a[1]
        
        # Use the shorter distance
        if abs(dx1) < abs(dx2):
            return math.hypot(dx1, dy) / ship_speed
        return math.hypot(dx2, dy) / ship_speed

    def get_neighbors(point):
        
        neighbors = []
        if direct:
            radius = 5
            num_directions = 64
        else:
            radius = RADIUS 
            num_directions = EXPLORATION_ANGLES 
        
        for i in range(num_directions):
            angle = 2 * math.pi * i / num_directions
            dx = int(round(radius * math.cos(angle)))
            dy = int(round(radius * math.sin(angle)))
            
            # Calculate potential neighbor coordinates
            new_x = point[0] + dx
            new_y = point[1] + dy
            
            # Handle dateline wrapping for x coordinate
            wrapped_x = wrap_x(new_x)
            
            # Check if the wrapped position is valid
            if (0 <= new_y < height and 
                buffered_water[new_y, wrapped_x]):
                neighbors.append((wrapped_x, new_y))
        
        return neighbors

    while open_set:
        _, current = heapq.heappop(open_set)
        if current == goal:
            break

        for neighbor in get_neighbors(current):
            cost = compute_travel_time(current[0], current[1], 
                                    neighbor[0], neighbor[1], 
                                    ship_speed, U, V, direct)
            if cost == float('inf'):
                continue

            tentative_g = g_score[current] + cost
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                g_score[neighbor] = tentative_g
                f = tentative_g + heuristic(neighbor, goal)
                heapq.heappush(open_set, (f, neighbor))
                came_from[neighbor] = current

    # Reconstruct path
    path = []
    cur = goal
    while cur in came_from:
        path.append(cur)
        cur = came_from[cur]
    path.append(start)
    path.reverse()
    return path

# === Coordinate Validation ===
def validate_coordinates(lat, lon, warn=True):
    """Validate coordinates against the available data range."""
    if not (LAT_MIN <= lat <= LAT_MAX):
        if warn:
            print(f"Warning: Latitude {lat}° is outside the current data range ({LAT_MIN}° to {LAT_MAX}°)")
        return False
    if not (LON_MIN <= lon <= LON_MAX):
        if warn:
            print(f"Warning: Longitude {lon}° is outside the current data range ({LON_MIN}° to {LON_MAX}°)")
        return False
    return True

# === Coordinate Mapping ===
def latlon_to_pixel(lat, lon):
    """Convert latitude/longitude to pixel coordinates."""
    if not validate_coordinates(lat, lon):
        raise ValueError(f"Coordinates ({lat}, {lon}) are outside the valid range")
    
    x = int((lon - LON_MIN) / (LON_MAX - LON_MIN) * IMAGE_WIDTH)
    y = int((LAT_MAX - lat) / (LAT_MAX - LAT_MIN) * IMAGE_HEIGHT)
    return x, y

def pixel_to_latlon(x, y):
    """Convert pixel coordinates to latitude/longitude."""
    lon = LON_MIN + x / IMAGE_WIDTH * (LON_MAX - LON_MIN)
    lat = LAT_MAX - y / IMAGE_HEIGHT * (LAT_MAX - LAT_MIN)
    return lat, lon

# === Distance Calculation ===
def nm_distance(lat1, lon1, lat2, lon2):
    """Calculate distance in nautical miles between two lat/lon points"""
    R = 3440.065  # Earth's radius in nautical miles
    return math.acos(
        math.sin(math.radians(lat1)) * math.sin(math.radians(lat2)) +
        math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
        math.cos(math.radians(lon2 - lon1))
    ) * R

# === Travel Time Calculation ===
def calculate_direct_time(lat1, lon1, lat2, lon2, ship_speed):
    """Calculate travel time along direct path considering currents and avoiding land"""
    # Get pixel coordinates
    x1, y1 = latlon_to_pixel(lat1, lon1)
    x2, y2 = latlon_to_pixel(lat2, lon2)
    
    # If either start or end is not in navigable water, return infinite time
    if not buffered_water[y1, x1] or not buffered_water[y2, x2]:
        return float('inf')
    
    # Handle dateline crossing
    dx1 = x2 - x1  # Normal distance
    dx2 = x2 - x1 - IMAGE_WIDTH if x2 > x1 else x2 - x1 + IMAGE_WIDTH  # Across dateline
    dy = y2 - y1
    
    # Use the shorter path
    if abs(dx1) <= abs(dx2):
        dx = dx1
    else:
        dx = dx2
        # Adjust x2 for proper interpolation
        if dx2 > 0:
            x2 = x1 + dx2
        else:
            x2 = x1 + dx2
    
    num_samples = max(abs(dx), abs(dy))
    if num_samples == 0:
        return 0
    
    # Sample points along direct path
    total_time = 0
    prev_x, prev_y = x1, y1
    
    for i in range(1, num_samples + 1):
        # Calculate current point
        t = i / num_samples
        curr_x = int(x1 + dx * t)
        curr_y = int(y1 + dy * t)
        
        # Wrap x coordinates for sampling currents
        curr_x_wrapped = curr_x % IMAGE_WIDTH
        prev_x_wrapped = prev_x % IMAGE_WIDTH
        
        # Check if we crossed land
        if not buffered_water[curr_y, curr_x_wrapped]:
            return float('inf')  # Path crosses land
        
        # Calculate segment distance in nautical miles
        seg_lat1, seg_lon1 = pixel_to_latlon(prev_x_wrapped, prev_y)
        seg_lat2, seg_lon2 = pixel_to_latlon(curr_x_wrapped, curr_y)
        segment_dist_nm = nm_distance(seg_lat1, seg_lon1, seg_lat2, seg_lon2)
        
        # Get average current for this segment
        current_u1, current_v1 = U[prev_y, prev_x_wrapped], -V[prev_y, prev_x_wrapped]
        current_u2, current_v2 = U[curr_y, curr_x_wrapped], -V[curr_y, curr_x_wrapped]
        avg_u = (current_u1 + current_u2) / 2
        avg_v = (current_v1 + current_v2) / 2
        
        # Calculate movement direction
        dx_seg = curr_x - prev_x
        dy_seg = curr_y - prev_y
        dist = math.hypot(dx_seg, dy_seg)
        if dist > 0:
            dir_x = dx_seg / dist
            dir_y = dy_seg / dist
            
            # Project average current onto movement direction
            current_along_path = avg_u * dir_x + avg_v * dir_y
            net_speed = ship_speed + current_along_path
            
            if net_speed <= 0.01:
                net_speed = 0.01  # minimum speed to avoid division by zero
            
            segment_time = segment_dist_nm / net_speed
            total_time += segment_time
        
        prev_x, prev_y = curr_x, curr_y
    
    return total_time

# === Example Visualization ===
def show_water_and_currents():
    plt.figure(figsize=(12, 4))
    plt.imshow(is_water, cmap='gray')
    plt.title("Land Mask (White = Water, Black = Land)")
    plt.axis('off')
    plt.show()

    # Quiver (optional)
    step = 40
    X, Y = np.meshgrid(np.arange(0, U.shape[1], step), np.arange(0, U.shape[0], step))
    U_down = U[::step, ::step]
    V_down = -V[::step, ::step]  # invert Y
    plt.figure(figsize=(12, 4))
    plt.quiver(X, Y, U_down, V_down, scale=20, color='blue')
    plt.title("Ocean Currents Vector Field")
    plt.gca().invert_yaxis()
    plt.show()

def export_path_to_csv(path_pixels, output_path="route.csv"):
    with open(output_path, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["lat", "lon"])
        for x, y in path_pixels:
            lat, lon = pixel_to_latlon(x, y)
            writer.writerow([lat, lon])
    print(f"Route exported to: {output_path}")

# === Run Example ===
if __name__ == "__main__":
    # Define list of waypoints as (lat, lon) pairs
    waypoints = ROUTE_COORDS
    
    # Validate all waypoints
    pixel_waypoints = []
    for i, (lat, lon) in enumerate(waypoints):
        if not validate_coordinates(lat, lon):
            print(f"Error: Waypoint {i} coordinates are outside the available data range")
            exit(1)
        x, y = latlon_to_pixel(lat, lon)
        pixel_waypoints.append((x, y))
        
        print(f"\nWaypoint {i} Debug:")
        print(f"Coords (lat, lon): {lat}, {lon}")
        print(f"Pixels (x, y): {x}, {y}")
        print(f"Is navigable water:", buffered_water[y, x])
    
    # Find optimized path through all waypoints
    complete_path = []
    complete_direct_path = []  # Store the direct path segments
    total_optimized_dist_nm = 0
    total_optimized_time_hours = 0
    total_direct_dist_nm = 0
    total_direct_time_hours = 0
    total_true_direct_time_hours = 0  # Track the true direct time
    
    # Process each pair of consecutive waypoints
    for i in range(len(pixel_waypoints) - 1):
        start = pixel_waypoints[i]
        goal = pixel_waypoints[i + 1]
        
        print(f"\nCalculating route from waypoint {i} to {i+1}...")
        
        path_segment = astar(start, goal, U, V, buffered_water, SHIP_SPEED_KN)
        
        if path_segment:
            # First find the direct route (shortest path ignoring currents)
            direct_path = astar(start, goal, U, V, buffered_water, SHIP_SPEED_KN, True)  # Use high speed to approximate shortest path
            
            if direct_path:
                # Add direct path segment to complete direct path
                complete_direct_path.extend(direct_path if not complete_direct_path else direct_path[1:])
                
                # Calculate direct route metrics by summing the segments
                direct_dist_nm = 0
                direct_time_hours = 0
                true_direct_time_hours = 0  # Time without current effects
                
                for j in range(len(direct_path) - 1):
                    x1, y1 = direct_path[j]
                    x2, y2 = direct_path[j + 1]
                    
                    # Convert segment endpoints to lat/lon
                    seg_lat1, seg_lon1 = pixel_to_latlon(x1, y1)
                    seg_lat2, seg_lon2 = pixel_to_latlon(x2, y2)
                    
                    # Add segment distance
                    seg_dist = nm_distance(seg_lat1, seg_lon1, seg_lat2, seg_lon2)
                    direct_dist_nm += seg_dist
                    
                    # Add segment time at constant speed (truly ignoring currents)
                    true_direct_time_hours += seg_dist / SHIP_SPEED_KN
                    
                    # Calculate time with currents for direct path
                    # Get the average current effect for this segment
                    current_u1, current_v1 = U[y1, x1], -V[y1, x1]
                    current_u2, current_v2 = U[y2, x2], -V[y2, x2]
                    avg_u = (current_u1 + current_u2) / 2
                    avg_v = (current_v1 + current_v2) / 2
                    
                    # Calculate direction vector
                    dx = x2 - x1
                    dy = y2 - y1
                    dist = math.hypot(dx, dy)
                    if dist > 0:
                        dir_x = dx / dist
                        dir_y = dy / dist
                        
                        # Project average current onto movement direction
                        current_along_path = avg_u * dir_x + avg_v * dir_y
                        net_speed = SHIP_SPEED_KN + current_along_path
                        
                        if net_speed <= 0.01:
                            net_speed = 0.01  # minimum speed to avoid division by zero
                        
                        segment_time = seg_dist / net_speed
                        direct_time_hours += segment_time
            else:
                direct_dist_nm = float('inf')
                direct_time_hours = float('inf')
                true_direct_time_hours = float('inf')
            
            total_direct_dist_nm += direct_dist_nm
            total_direct_time_hours += direct_time_hours
            total_true_direct_time_hours += true_direct_time_hours
            
            # Calculate optimized route metrics
            segment_dist_nm = 0
            segment_time_hours = 0
            
            for j in range(len(path_segment) - 1):
                x1, y1 = path_segment[j]
                x2, y2 = path_segment[j + 1]
                
                # Convert segment endpoints to lat/lon
                seg_lat1, seg_lon1 = pixel_to_latlon(x1, y1)
                seg_lat2, seg_lon2 = pixel_to_latlon(x2, y2)
                
                # Calculate segment distance in nautical miles
                sub_segment_dist_nm = nm_distance(seg_lat1, seg_lon1, seg_lat2, seg_lon2)
                segment_dist_nm += sub_segment_dist_nm
                
                # Get the average current effect for this segment
                current_u1, current_v1 = U[y1, x1], -V[y1, x1]
                current_u2, current_v2 = U[y2, x2], -V[y2, x2]
                avg_u = (current_u1 + current_u2) / 2
                avg_v = (current_v1 + current_v2) / 2
                
                # Calculate the direction vector of travel
                dx = x2 - x1
                dy = y2 - y1
                dist = math.hypot(dx, dy)
                if dist > 0:
                    dir_x = dx / dist
                    dir_y = dy / dist
                    
                    # Project average current onto movement direction
                    current_along_path = avg_u * dir_x + avg_v * dir_y
                    net_speed = SHIP_SPEED_KN + current_along_path
                    
                    if net_speed <= 0.01:
                        net_speed = 0.01  # minimum speed to avoid division by zero
                    
                    sub_segment_time = sub_segment_dist_nm / net_speed
                    segment_time_hours += sub_segment_time
            
            total_optimized_dist_nm += segment_dist_nm
            total_optimized_time_hours += segment_time_hours
            complete_path.extend(path_segment if not complete_path else path_segment[1:])
            
            print(f"Segment {i} to {i+1} Analysis:")
            print(f"  Direct distance: {direct_dist_nm:.1f} nm, time: {direct_time_hours:.1f} hours")
            print(f"  True direct time (ignoring currents): {true_direct_time_hours:.1f} hours")
            print(f"  Optimized distance: {segment_dist_nm:.1f} nm, time: {segment_time_hours:.1f} hours")
        else:
            print(f"No path found between waypoints {i} and {i+1}")
            exit(1)
    
    # Calculate total savings
    distance_diff = total_direct_dist_nm - total_optimized_dist_nm
    
    # Print final analysis
    print("\n=== Complete Route Analysis ===")
    print(f"Direct route total:")
    print(f"  Distance: {total_direct_dist_nm:.1f} nautical miles")
    if math.isinf(total_direct_time_hours):
        print("  Duration: Not possible (route crosses land)")
    else:
        print(f"  Duration (ignoring currents): {total_true_direct_time_hours:.1f} hours")
        print(f"  Duration (with currents): {total_direct_time_hours:.1f} hours")
    
    print(f"\nOptimized route total:")
    print(f"  Distance: {total_optimized_dist_nm:.1f} nautical miles")
    print(f"  Duration: {total_optimized_time_hours:.1f} hours")
    
    print(f"\nComparison:")
    print(f"  Distance difference: {abs(distance_diff):.1f} nautical miles {'longer' if distance_diff < 0 else 'shorter'}")
    if math.isinf(total_direct_time_hours):
        print("  Time savings: Not comparable (direct route not possible)")
    else :
        time_diff = total_direct_time_hours - total_optimized_time_hours
        print(f"  Time savings: {abs(time_diff):.1f} hours {'longer' if time_diff < 0 else 'shorter'}")


    # Plot complete path
    plt.imshow(buffered_water, cmap='gray')
    if complete_path and complete_direct_path:
        # Plot optimized route in red
        px, py = zip(*complete_path)
        plt.plot(px, py, 'r-', linewidth=2, label='Optimized Route')
        
        # Plot direct route in blue
        dx, dy = zip(*complete_direct_path)
        plt.plot(dx, dy, 'b--', linewidth=2, label='Direct Route')
        
        # Plot all waypoints
        for i, (x, y) in enumerate(pixel_waypoints):
            if i == 0:
                plt.plot([x], [y], 'go', label='Start')  # start
            elif i == len(pixel_waypoints) - 1:
                plt.plot([x], [y], 'ro', label='End')    # end
            else:
                plt.plot([x], [y], 'yo', label=f'Waypoint {i}')  # intermediate
        
        plt.legend()
        # Export both paths to CSV
        export_path_to_csv(complete_path, "optimized_route.csv")
        export_path_to_csv(complete_direct_path, "direct_route.csv")
        print("\nRoute files exported:")
        print("- Optimized route: optimized_route.csv")
        print("- Direct route: direct_route.csv")
    else:
        print("No complete path found.")
    
    plt.title("Ship Route Through Multiple Waypoints Based on Ocean Currents")
    plt.show()



