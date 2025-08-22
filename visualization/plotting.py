"""Plotting functions for route visualization."""

import matplotlib.pyplot as plt
import numpy as np

def plot_route(buffered_water, complete_direct_path):
    """Plot the complete route with waypoints."""
    plt.figure(figsize=(12, 4))
    plt.imshow(buffered_water, cmap='gray')
    
    
    
    plt.title("Ship Route Through Multiple Waypoints Based on Ocean Currents")
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
