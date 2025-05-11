"""Ship performance calculations module."""

import math
import numpy as np
from config import SHIP_CONFIG, SHIP_OPERATION

# Constants for wave calculations
GRAVITY = 9.81  # m/s^2

def calculate_added_resistance(wave_height, wave_period, wave_direction, ship_speed):
    """Calculate added resistance due to waves.
    
    Args:
        wave_height (float): Significant wave height (m)
        wave_period (float): Wave period (s)
        wave_direction (float): Wave direction relative to ship's heading (degrees)
        ship_speed (float): Ship speed through water (knots)
    
    Returns:
        float: Speed loss factor (0-1)
    """
    # Convert ship speed to m/s
    speed_ms = ship_speed * 0.514444

    # Calculate wavelength using deep water approximation
    wavelength = (9.81 * wave_period ** 2) / (2 * math.pi)
    
    # Calculate wave steepness
    wave_steepness = wave_height / wavelength if wavelength > 0 else 0
    
    # Calculate Froude number based on ship length
    froude_number = speed_ms / math.sqrt(9.81 * SHIP_CONFIG["length_pp"])
    
    # Calculate relative wave direction factor (head seas = 1, following seas = 0)
    # Handle both scalar and array inputs
    if isinstance(wave_direction, (int, float)):
        direction_rad = math.radians(wave_direction)
        direction_factor = abs(math.cos(direction_rad))
    else:
        direction_rad = np.radians(wave_direction)
        direction_factor = abs(np.cos(direction_rad))
    
    # Calculate basic added resistance coefficient
    # This is a simplified model - in reality, this would be more complex
    added_resistance_coef = (
        1.0 
        + 2.33 * wave_steepness * direction_factor 
        + 0.425 * froude_number ** 2 * wave_steepness ** 2
    )
    
    # Calculate speed loss factor (0 = no loss, 1 = complete loss)
    # This is a simplified model that increases speed loss with wave height and period
    speed_loss_factor = min(
        0.95,  # Maximum speed loss factor
        (added_resistance_coef - 1.0) * (
            0.5  # Base effect
            + 0.3 * wave_height / SHIP_CONFIG["draft"]  # Wave height effect
            + 0.2 * direction_factor  # Direction effect
        )
    )
    
    # Ensure speed loss factor is between 0 and 1
    return max(0.0, min(0.95, speed_loss_factor))

def calculate_wave_encounter_period(wave_period, wave_direction, ship_speed):
    """Calculate the encounter period of waves.
    
    Args:
        wave_period (float or ndarray): Wave period (s)
        wave_direction (float or ndarray): Wave direction relative to ship's heading (degrees)
        ship_speed (float): Ship speed through water (knots)
    
    Returns:
        float or ndarray: Encounter period (s)
    """
    # Convert ship speed to m/s
    speed_ms = ship_speed * 0.514444
    
    # Convert wave direction to radians
    if isinstance(wave_direction, (int, float)):
        direction_rad = math.radians(wave_direction)
        cos_dir = math.cos(direction_rad)
    else:
        direction_rad = np.radians(wave_direction)
        cos_dir = np.cos(direction_rad)
    
    # Calculate wavelength
    wavelength = (GRAVITY * wave_period ** 2) / (2 * math.pi)
    
    # Calculate encounter period
    denominator = (1 - (2 * math.pi * speed_ms * cos_dir) / (GRAVITY * wave_period))
    
    if isinstance(denominator, (int, float)):
        if abs(denominator) < 0.001:
            return wave_period
    else:
        denominator = np.where(np.abs(denominator) < 0.001, 1, denominator)
    
    encounter_period = wave_period / denominator
    
    if isinstance(encounter_period, (int, float)):
        return max(0.1, encounter_period)
    else:
        return np.maximum(0.1, encounter_period)

def calculate_net_speed(ship_speed, current_along_path, wave_height, wave_period, wave_direction):
    """Calculate net ship speed considering both currents and waves.
    
    Args:
        ship_speed (float): Base ship speed in knots
        current_along_path (float): Current component along ship's path in knots
        wave_height (float): Significant wave height (m)
        wave_period (float): Wave period (s)
        wave_direction (float): Wave direction relative to ship's heading (degrees)
    
    Returns:
        float: Net ship speed in knots
    """
    # Calculate speed loss due to waves
    speed_loss_factor = calculate_added_resistance(
        wave_height, wave_period, wave_direction, ship_speed
    )
    
    # Calculate speed through water accounting for waves
    speed_through_water = ship_speed * (1 - speed_loss_factor)
    
    # Add current effect
    net_speed = speed_through_water + current_along_path
    
    return max(0.1, net_speed)  # Ensure minimum speed
