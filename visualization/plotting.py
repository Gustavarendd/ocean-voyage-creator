"""Plotting functions for route visualization."""

import matplotlib.pyplot as plt
import numpy as np
import json
from utils.coordinates import latlon_to_pixel
try:
    from core.initialization import (
        ACTIVE_LAT_MIN, ACTIVE_LAT_MAX, ACTIVE_LON_MIN, ACTIVE_LON_MAX
    )
except Exception:
    ACTIVE_LAT_MIN, ACTIVE_LAT_MAX = -90.0, 90.0
    ACTIVE_LON_MIN, ACTIVE_LON_MAX = -180.0, 180.0

def plot_route_with_tss(buffered_water, route_path=None, tss_geojson_path=None, waypoints=None):
    """Plot route with TSS lanes overlay on buffered water."""
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Plot buffered water (water=white, land=black)
    ax.imshow(buffered_water, cmap='gray', alpha=0.7, aspect='equal')
    
    # Plot TSS lanes if provided
    if tss_geojson_path:
        plot_tss_lanes(ax, tss_geojson_path)
    
    # Plot route path
    if route_path:
        route_x = [point[0] for point in route_path]
        route_y = [point[1] for point in route_path]
        ax.plot(route_x, route_y, 'red', linewidth=3, label='Route', alpha=0.8)
        
        # Mark start and end
        ax.plot(route_x[0], route_y[0], 'go', markersize=10, label='Start')
        ax.plot(route_x[-1], route_y[-1], 'ro', markersize=10, label='End')
    
    # Plot waypoints if provided
    if waypoints:
        wp_x = [point[0] for point in waypoints]
        wp_y = [point[1] for point in waypoints]
        ax.plot(wp_x, wp_y, 'yo', markersize=8, label='Waypoints')
        
        # Number the waypoints
        for i, (x, y) in enumerate(waypoints):
            ax.annotate(f'WP{i}', (x, y), xytext=(5, 5), textcoords='offset points',
                       fontsize=10, color='yellow', weight='bold')
    
    ax.set_title("Ocean Route with Traffic Separation Schemes", fontsize=14)
    ax.legend(loc='upper right')
    ax.set_xlabel("X (pixels)")
    ax.set_ylabel("Y (pixels)")
    # Draw active bounds lat/lon annotation for reference
    ax.text(5, 15, f"Lat {ACTIVE_LAT_MIN} .. {ACTIVE_LAT_MAX}\nLon {ACTIVE_LON_MIN} .. {ACTIVE_LON_MAX}",
            fontsize=9, color='yellow', ha='left', va='top', bbox=dict(facecolor='black', alpha=0.3, pad=4))
    
    plt.tight_layout()
    plt.show()

def plot_tss_lanes(ax, tss_geojson_path):
    """Plot TSS lanes from GeoJSON file."""
    try:
        with open(tss_geojson_path, 'r') as f:
            data = json.load(f)
        
        lane_count = 0
        separation_count = 0
        
        for feature in data['features']:
            if (feature['type'] == 'Feature' and 
                'properties' in feature and
                'parsed_other_tags' in feature['properties']):
                
                tags = feature['properties']['parsed_other_tags']
                seamark_type = tags.get('seamark:type', '')
                
                if seamark_type == 'separation_lane':
                    plot_lane_feature(ax, feature, 'blue', 'TSS Lane', lane_count == 0)
                    lane_count += 1
                elif seamark_type in ['separation_line', 'separation_boundary']:
                    plot_lane_feature(ax, feature, 'purple', 'TSS Boundary', separation_count == 0)
                    separation_count += 1
        
        print(f"Plotted {lane_count} TSS lanes and {separation_count} separation boundaries")
        
    except Exception as e:
        print(f"Error plotting TSS lanes: {e}")

def plot_lane_feature(ax, feature, color, label, show_label):
    """Plot a single TSS lane feature."""
    try:
        geometry = feature['geometry']
        
        if geometry['type'] == 'LineString':
            coordinates = geometry['coordinates']
            
            # Convert coordinates to pixels
            pixel_coords = []
            for lon, lat in coordinates:
                if not (ACTIVE_LAT_MIN <= lat <= ACTIVE_LAT_MAX and ACTIVE_LON_MIN <= lon <= ACTIVE_LON_MAX):
                    continue
                try:
                    x, y = latlon_to_pixel(lat, lon, warn=False)
                except Exception:
                    continue
                pixel_coords.append((x, y))
            
            if len(pixel_coords) > 1:
                x_coords = [coord[0] for coord in pixel_coords]
                y_coords = [coord[1] for coord in pixel_coords]
                
                # Plot the line
                if show_label:
                    ax.plot(x_coords, y_coords, color=color, linewidth=1, alpha=0.6, label=label)
                else:
                    ax.plot(x_coords, y_coords, color=color, linewidth=1, alpha=0.6)
                
                # Add arrow to show direction if bearing available
                properties = feature['properties']
                if 'tss_flow_bearing_deg' in properties and len(pixel_coords) > 1:
                    # Add arrow at midpoint
                    mid_idx = len(pixel_coords) // 2
                    if mid_idx < len(pixel_coords) - 1:
                        x1, y1 = pixel_coords[mid_idx]
                        x2, y2 = pixel_coords[mid_idx + 1]
                        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                                  arrowprops=dict(arrowstyle='->', color=color, lw=1))
    
    except Exception as e:
        pass  # Skip problematic features

def plot_route(buffered_water, complete_direct_path):
    """Plot the complete route with waypoints (original function for compatibility)."""
    plt.figure(figsize=(12, 4))
    plt.imshow(buffered_water, cmap='gray')
    
    if complete_direct_path:
        route_x = [point[0] for point in complete_direct_path]
        route_y = [point[1] for point in complete_direct_path]
        plt.plot(route_x, route_y, 'red', linewidth=2, label='Route')
        
        # Mark start and end
        plt.plot(route_x[0], route_y[0], 'go', markersize=8, label='Start')
        plt.plot(route_x[-1], route_y[-1], 'ro', markersize=8, label='End')
        plt.legend()
    
    plt.title("Ship Route Through Multiple Waypoints")
    plt.show()

def plot_waypoints(waypoints):
    """Plot waypoints with different colors for start, end, and intermediate points."""
    for i, (x, y) in enumerate(waypoints):
        if i == 0:
            plt.plot([x], [y], 'go', label='Start')
        elif i == len(waypoints) - 1:
            plt.plot([x], [y], 'ro', label='End')
        else:
            plt.plot([x], [y], 'yo', label=f'Waypoint {i}')

def show_water_and_currents(is_water, U, V):
    """Show water mask and current vector field."""
    plt.figure(figsize=(12, 4))
    plt.imshow(is_water, cmap='gray')
    plt.title("Land Mask (White = Water, Black = Land)")
    plt.axis('off')
    plt.show()

    # Plot current vectors
    step = 40
    X, Y = np.meshgrid(np.arange(0, U.shape[1], step), np.arange(0, U.shape[0], step))
    U_down = U[::step, ::step]
    V_down = -V[::step, ::step]
    
    plt.figure(figsize=(12, 4))
    plt.quiver(X, Y, U_down, V_down, scale=20, color='blue')
    plt.title("Ocean Currents Vector Field")
    plt.gca().invert_yaxis()
    plt.show()
