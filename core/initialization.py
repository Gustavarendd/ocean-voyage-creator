"""Core initialization and image loading functions."""

from PIL import Image
import numpy as np
from config import *

def load_and_process_images(currents_path, land_mask_path, wave_path):
    """Load and process the currents and land mask images."""
    # Load currents image
    currents_img = Image.open(currents_path)
    currents_np = np.array(currents_img)

    # Load wave image
    wave_img = Image.open(wave_path)
    wave_np = np.array(wave_img)

    # Calculate crop indices for wave image
    full_height_wave = wave_img.height  # Full height of wave image (90°N to 80°S)
    north_limit_px_wave = int((90 - LAT_MAX) / 180 * full_height_wave)  # 65°N
    south_limit_px_wave = int((90 - LAT_MIN) / 180 * full_height_wave)  # 60°S

    # Crop and resize wave image
    wave_cropped = wave_np[north_limit_px_wave:south_limit_px_wave, :]
    wave_resized = Image.fromarray(wave_cropped).resize(
        (IMAGE_WIDTH, IMAGE_HEIGHT),
        Image.Resampling.BILINEAR
    )
    wave_np = np.array(wave_resized)
    wave_np = pad_wave_image(wave_np, 170, 180)  # Pad the wave image if necessary
    
    # Load and process land mask
    land_mask = Image.open(land_mask_path).convert("L")
    land_mask_np = np.array(land_mask)
    
    # Calculate crop indices for land mask
    full_height_land = land_mask.height  # Full height of land mask (90°N to 90°S)
    north_limit_px_land = int((90 - LAT_MAX) / 180 * full_height_land)  # 65°N
    south_limit_px_land = int((90 - LAT_MIN) / 180 * full_height_land)  # 60°S
    
    # Crop and resize land mask
    land_mask_cropped = land_mask_np[north_limit_px_land:south_limit_px_land, :]
    land_mask_resized = Image.fromarray(land_mask_cropped).resize(
        (IMAGE_WIDTH, IMAGE_HEIGHT),
        Image.Resampling.BILINEAR
    )
    land_mask_np = np.array(land_mask_resized)
    
    # Create water mask
    is_water = land_mask_np < 20  # True for water, False for land
    
    return currents_np, wave_np, is_water

def extract_currents(currents_np):
    """Extract U and V components from currents image."""
    R, G, _ = currents_np[:, :, 0], currents_np[:, :, 1], currents_np[:, :, 2]
    
    U = scale_channel(R, -1.857, 2.035)
    V = scale_channel(G, -1.821, 2.622)
    
    return U, V

def extract_waves(wave_np):
    """Extract Height, period and direction components from wave image."""
    R, G, B = wave_np[:, :, 0], wave_np[:, :, 1], wave_np[:, :, 2]
    
    Direction = scale_channel(B, 0, 360)
    Period = scale_channel(G, 1.5, 14.13)
    Height = scale_channel(R, 0.009999999776482582, 8.420000076293945)
   
    return Height, Period, Direction

def scale_channel(channel, min_val, max_val):
    """Scale image channel values to current velocities."""
    return min_val + (channel / 255.0) * (max_val - min_val)


def pad_wave_image(wave_np, original_lat_coverage=170, target_lat_coverage=180):
    original_height = wave_np.shape[0]
    target_height = int(original_height * (target_lat_coverage / original_lat_coverage))
    pad_pixels = target_height - original_height

    if pad_pixels > 0:
        # Pad at the bottom with zeros (black pixels)
        pad_array = np.zeros((pad_pixels, wave_np.shape[1], wave_np.shape[2]), dtype=wave_np.dtype)
        wave_np_padded = np.vstack((wave_np, pad_array))
        return wave_np_padded
    else:
        return wave_np  # No padding needed