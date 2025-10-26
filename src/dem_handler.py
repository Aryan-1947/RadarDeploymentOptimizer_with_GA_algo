# src/dem_handler.py
import rasterio
import os
from rasterio.windows import from_bounds
from config import DATASET_FOLDER

def load_dem(area, dem_folder = os.path.join(os.path.dirname(__file__), '..', 'Dataset')):
    dem_path = os.path.join(dem_folder, area.lower(), f"{area.lower()}_dem.tif")
    print(f"Looking for DEM at: {os.path.abspath(dem_path)}")  # 👈 Add this line

    if not os.path.exists(dem_path):
        raise FileNotFoundError(f"DEM not found for {area}: {dem_path}")

    with rasterio.open(dem_path) as src:
        dem_data = src.read(1)
        transform = src.transform
        profile = src.profile

    return dem_data, transform, profile

# def crop_dem(dem, bounds):
#     xmin, ymin, xmax, ymax = bounds
#     window = from_bounds(xmin, ymin, xmax, ymax, transform=dem.transform)
#     cropped = dem.read(1, window=window)
#     transform = dem.window_transform(window)
#     return cropped, transform


