"""Water mask creation and processing functions."""

import numpy as np
from scipy import ndimage
from config import IMAGE_WIDTH

def create_buffered_water_mask(is_water, buffer_nm):
    """Create a water mask with a coastal buffer zone while preserving small islands."""
    # Calculate buffer size in pixels
    base_pixels_per_nm = (IMAGE_WIDTH / 360) / 60
    buffer_pixels = int(buffer_nm * base_pixels_per_nm * 2)
    
    # Create structuring elements
    main_structure = ndimage.iterate_structure(
        ndimage.generate_binary_structure(2, 1),
        buffer_pixels // 2
    )
    
    # Create land mask
    land_mask = ~is_water
    labeled_lands, num_features = ndimage.label(land_mask)
    land_sizes = np.bincount(labeled_lands.ravel())
    
    # Define threshold for small islands
    small_island_threshold = (buffer_pixels * 4) ** 2
    
    # Process large and small landmasses separately
    large_lands = np.zeros_like(land_mask)
    small_lands = np.zeros_like(land_mask)
    
    for i in range(1, num_features + 1):
        if land_sizes[i] > small_island_threshold:
            large_lands |= (labeled_lands == i)
        else:
            small_lands |= (labeled_lands == i)
    
    # Apply different buffer sizes
    dilated_large = ndimage.binary_dilation(large_lands, structure=main_structure)
    
    small_structure = ndimage.iterate_structure(
        ndimage.generate_binary_structure(2, 1),
        max(1, buffer_pixels // 4)
    )
    dilated_small = ndimage.binary_dilation(small_lands, structure=small_structure)
    
    # Combine results
    combined_lands = dilated_large | dilated_small
    
    return ~combined_lands  # Return the water mask
