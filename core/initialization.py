"""Core initialization and image loading functions."""

from PIL import Image
import numpy as np
from config import *

def load_and_process_images(currents_path, land_mask_path):
    """Load and process the currents and land mask images."""
    # Load currents image
    # currents_img = Image.open(currents_path)
    # currents_np = np.array(currents_img)

    # Load wave image
    # wave_img = Image.open(wave_path)
    # wave_np = np.array(wave_img)

    # wave_np = pad_wave_image(wave_np, 170, 180)  # Pad the wave image if necessary

    # Calculate crop indices for wave image
    # full_height_wave = Image.fromarray(wave_np).height  # Full height of wave image (90°N to 80°S)
    # north_limit_px_wave = int((90 - LAT_MAX) / 180 * full_height_wave)  # 65°N
    # south_limit_px_wave = int((90 - LAT_MIN) / 180 * full_height_wave)  # 60°S

    # Crop and resize wave image
    # wave_np = wave_np[north_limit_px_wave:south_limit_px_wave, :]
    # wave_resized = Image.fromarray(wave_np).resize((IMAGE_WIDTH, IMAGE_HEIGHT), Image.Resampling.NEAREST)
    # wave_np = np.array(wave_resized)
    
    # Load and process land mask
    land_mask = Image.open(land_mask_path).convert("L")
    land_mask_np = np.array(land_mask)
    
    # Calculate crop indices for land mask
    full_height = 3000  # Full height of land mask (90°N to 90°S)
    north_limit_px = int((90 - LAT_MAX) / 180 * full_height)  # 65°N
    south_limit_px = int((90 - LAT_MIN) / 180 * full_height)  # 60°S
    
    # Crop and resize land mask
    land_mask_cropped = land_mask_np[north_limit_px:south_limit_px, :]
    land_mask_resized = Image.fromarray(land_mask_cropped).resize(
        (IMAGE_WIDTH, IMAGE_HEIGHT),
        Image.Resampling.BILINEAR
    )
    land_mask_np = np.array(land_mask_resized)
    
    # Create water mask
    is_water = land_mask_np < 20  # True for water, False for land
    
    return is_water

def extract_currents(currents_np):
    """Extract U and V components from currents image."""
    R, G, _ = currents_np[:, :, 0], currents_np[:, :, 1], currents_np[:, :, 2]
    
    U = scale_channel(R, -1.857, 2.035)
    V = scale_channel(G, -1.821, 2.622)
    
    return U, V

def scale_channel(channel, min_val, max_val):
    """Scale image channel values to current velocities."""
    return min_val + (channel / 255.0) * (max_val - min_val)
