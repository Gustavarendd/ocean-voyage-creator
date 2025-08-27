"""Core initialization and image loading functions."""

from PIL import Image
import numpy as np
from config import *

def load_and_process_images(land_mask_path):
      
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

