"""Ship performance calculations module."""

import math
import numpy as np
from config import SHIP_CONFIG, SHIP_OPERATION

# Constants for wave calculations
GRAVITY = 9.81  # m/s^2

import math
import numpy as np

SHIP_CONFIG = {
    "length_pp": 397,
    "beam": 56,
    "draft": 15.5,
    "block_coefficient": 0.68,
}

def calculate_added_resistance(wave_height, wave_period, wave_direction, ship_speed):
    """Calculate added resistance due to waves, adjusted for block coefficient.

    Args:
        wave_height (float): Significant wave height (m)
        wave_period (float): Wave period (s)
        wave_direction (float): Wave direction relative to ship's heading (degrees)
        ship_speed (float): Ship speed through water (knots)

    Returns:
        float: Speed loss factor (0-1)
    """
    # Convert speed to m/s
    speed_ms = ship_speed * 0.514444

    # Calculate wavelength
    wavelength = (9.81 * wave_period ** 2) / (2 * math.pi)
    wave_steepness = wave_height / wavelength if wavelength > 0 else 0

    # Froude number
    froude_number = speed_ms / math.sqrt(9.81 * SHIP_CONFIG["length_pp"])

    # Direction factor
    if isinstance(wave_direction, (int, float)):
        direction_rad = math.radians(wave_direction)
        direction_factor = abs(math.cos(direction_rad))
    else:
        direction_rad = np.radians(wave_direction)
        direction_factor = abs(np.cos(direction_rad))

    # Basic resistance coefficient
    added_resistance_coef = (
        1.0
        + 2.33 * wave_steepness * direction_factor
        + 0.425 * froude_number ** 2 * wave_steepness ** 2
    )

    # Include block coefficient
    block_coef_factor = 1 + 2 * (SHIP_CONFIG["block_coefficient"] - 0.6)
    added_resistance_coef *= block_coef_factor

    # Calculate speed loss
    speed_loss_factor = min(
        0.95,
        (added_resistance_coef - 1.0) * (
            0.5
            + 0.3 * wave_height / SHIP_CONFIG["draft"]
            + 0.2 * direction_factor
        )
    )

    return max(0.0, speed_loss_factor)


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
