"""Water mask creation and processing functions."""

import numpy as np
import os
import hashlib
from scipy import ndimage
from config import IMAGE_WIDTH, CRITICAL_REGIONS

def _create_cache_key(is_water, buffer_nm):
    """Create a unique cache key based on input parameters."""
    # Create hash of the water mask and buffer parameters
    water_hash = hashlib.md5(is_water.tobytes()).hexdigest()[:8]
    critical_hash = hashlib.md5(str(CRITICAL_REGIONS).encode()).hexdigest()[:8]
    buffer_str = f"{buffer_nm:.1f}"
    
    return f"{water_hash}_{critical_hash}_{buffer_str}"

def clear_buffered_water_cache():
    """Clear all cached buffered water masks."""
    cache_dir = "cache"
    if not os.path.exists(cache_dir):
        return
    
    import glob
    cache_files = glob.glob(os.path.join(cache_dir, "buffered_water_*.npy"))
    
    for cache_file in cache_files:
        try:
            os.remove(cache_file)
            print(f"Removed cache file: {cache_file}")
        except Exception as e:
            print(f"Error removing cache file {cache_file}: {e}")
    
    if not cache_files:
        print("No cache files found to remove.")

def create_buffered_water_mask(is_water, buffer_nm, force_recompute=False):
    """Create a water mask with a coastal buffer zone while preserving small islands and narrow channels.
    Uses caching to avoid recomputation when the same parameters are used.
    
    Args:
        is_water: Boolean array indicating water (True) vs land (False)
        buffer_nm: Buffer distance in nautical miles
        force_recompute: If True, bypass cache and recompute the mask
    """
    # Create cache directory if it doesn't exist
    cache_dir = "cache"
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    
    # Create a hash of the input parameters to use as cache key
    cache_key = _create_cache_key(is_water, buffer_nm)
    cache_file = os.path.join(cache_dir, f"buffered_water_{cache_key}.npy")
    
    # Check if cached version exists and force_recompute is False
    if not force_recompute and os.path.exists(cache_file):
        print(f"Loading cached buffered water mask from {cache_file}")
        try:
            cached_mask = np.load(cache_file)
            print(f"Successfully loaded cached mask with shape: {cached_mask.shape}")
            return cached_mask
        except Exception as e:
            print(f"Error loading cached mask: {e}. Recomputing...")
    
    if force_recompute:
        print("Force recompute requested - computing buffered water mask...")
    else:
        print("Computing buffered water mask (this may take a moment)...")
    
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
    
    # Save to cache
    try:
        np.save(cache_file, final_mask)
        print(f"Cached buffered water mask saved to {cache_file}")
    except Exception as e:
        print(f"Warning: Could not save cache file: {e}")
    
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