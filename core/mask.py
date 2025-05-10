"""Water mask creation and processing functions."""

import numpy as np
from scipy import ndimage
from config import IMAGE_WIDTH, CRITICAL_REGIONS

def create_buffered_water_mask(is_water, buffer_nm):
    """Create a water mask with a coastal buffer zone while preserving small islands and narrow channels."""
    # Calculate buffer size in pixels
    base_pixels_per_nm = (IMAGE_WIDTH / 360) / 60
    buffer_pixels = int(buffer_nm * base_pixels_per_nm * 2)

    is_water_corrected = preserve_critical_channels(is_water, CRITICAL_REGIONS)
    
    # Create structuring elements
    main_structure = ndimage.iterate_structure(
        ndimage.generate_binary_structure(2, 1),
        buffer_pixels // 2
    )
    
    # Create land mask
    land_mask = ~is_water_corrected
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
    
    # Identify narrow water channels
    water_structure = ndimage.iterate_structure(
        ndimage.generate_binary_structure(2, 1),
        max(1, buffer_pixels // 8)
    )
    narrow_water = ndimage.binary_erosion(is_water_corrected, structure=water_structure)
    preserved_channels = is_water_corrected & narrow_water  # Preserve narrow water channels
    
    # Apply different buffer sizes
    dilated_large = ndimage.binary_dilation(large_lands, structure=main_structure)
    
    small_structure = ndimage.iterate_structure(
        ndimage.generate_binary_structure(2, 1),
        max(1, buffer_pixels // 4)
    )
    dilated_small = ndimage.binary_dilation(small_lands, structure=small_structure)
    
    # Combine results
    combined_lands = dilated_large | dilated_small
    
    # Ensure preserved channels remain open
    final_mask = ~combined_lands | preserved_channels
    
    return final_mask  # Return the water mask


def preserve_critical_channels(is_water, critical_regions):
    """
    Ensure critical water channels remain open in the water mask.
    
    Args:
        is_water (np.ndarray): The original water mask.
        critical_regions (list of tuples): List of regions to preserve, each defined as
                                            (x_min, x_max, y_min, y_max).
    
    Returns:
        np.ndarray: Updated water mask with critical regions preserved.
    """
    for x_min, x_max, y_min, y_max in critical_regions:
        is_water[y_min:y_max, x_min:x_max] = True  # Mark region as water
    return is_water