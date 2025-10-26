# src/terrain_analysis.py
import numpy as np
from scipy.ndimage import generic_filter
from scipy import ndimage
import rasterio


#addition slope
def calculate_slope(dem):
    from scipy.ndimage import sobel
    dzdx = sobel(dem, axis=1) / 30  # Assume fixed pixel size = 30m
    dzdy = sobel(dem, axis=0) / 30
    slope = np.sqrt(dzdx**2 + dzdy**2)
    return slope


def mask_invalid_terrain(dem, slope_map, slope_thresh=30, roughness_thresh=10, water_thresh=2):
    from scipy.ndimage import generic_filter

    invalid_mask = np.zeros_like(dem, dtype=bool)
    # Slope constraint
    invalid_mask |= slope_map > np.tan(np.radians(slope_thresh))

    # Roughness constraint
    def std_filter(values):
        return np.std(values)

    roughness = generic_filter(dem, std_filter, size=5)
    invalid_mask |= roughness > roughness_thresh

    # Water body constraint
    invalid_mask |= dem < water_thresh

    return invalid_mask,slope_map, roughness
