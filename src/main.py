# Refactored src/main.py

from tqdm import tqdm


import os
import pickle
from datetime import datetime


import numpy as np
import matplotlib.pyplot as plt
from user_input import get_radar_specs, get_crop_coordinates, compute_radar_range
from dem_handler import load_dem
from config import DATASET_FOLDER
from aop import crop_dem, latlon_to_pixel_bounds
from radar_simulation import get_radar_coverage
from terrain_analysis import calculate_slope, mask_invalid_terrain



def get_area_name():
    return input("\U0001F30D Enter the state name (e.g., rajasthan): ").strip().lower()

def get_num_radars():
    return int(input("\U0001F4F1 Enter number of radars to deploy: "))

# new code
def choose_saved_model():
    models_dir = 'saved_models'
    os.makedirs(models_dir, exist_ok=True)
    saved_models = [f for f in os.listdir(models_dir) if f.endswith('.pkl')]

    if not saved_models:
        print("⚠️ No saved models found. Proceeding with training...")
        return None

    print("\n📁 Available Saved Models:")
    for i, fname in enumerate(saved_models):
        print(f"{i + 1}. {fname}")

    choice = int(input("👉 Enter the number of the model to load: ")) - 1
    if 0 <= choice < len(saved_models):
        model_path = os.path.join(models_dir, saved_models[choice])
        with open(model_path, 'rb') as f:
            loaded_positions = pickle.load(f)
        print(f"✅ Loaded model from: {model_path}")
        return loaded_positions
    else:
        print("❌ Invalid choice. Proceeding with training...")
        return None


def main():
    print("\U0001F4CA RADAR DEPLOYMENT OPTIMIZER \U0001F4CA\n")

    # new code
    use_existing = input("🗃️ Do you want to use an existing saved model? (y/n): ").strip().lower()
    loaded_positions = None

    if use_existing == 'y':
        loaded_positions = choose_saved_model()
        if loaded_positions is not None:
            print("📡 Loaded radar positions loaded successfully.")
        else:
            print("🔁 Switching to retraining mode...\n")


    # Step 1: Get state and DEM
    area = get_area_name()
    num_radars = get_num_radars()
    radar_specs = get_radar_specs(num_radars)
    bounds = get_crop_coordinates()

    try:
        dem, transform, _ = load_dem(area)
    except FileNotFoundError:
        print(f"❌ DEM file not found for '{area}'. Ensure it's in Dataset/{area}/")
        return

    try:
        x_min, y_min, x_max, y_max = latlon_to_pixel_bounds(transform, bounds)
        cropped_dem = crop_dem(dem, x_min, y_min, x_max, y_max)

        print("✅ DEM shape (rows, cols):", dem.shape)
        print("✅ DEM transform:", transform)
        print("✅ Pixel size (lon, lat):", transform.a, transform.e)
        print("✅ DEM cropped successfully.")

        import matplotlib.pyplot as plt
        plt.imshow(cropped_dem, cmap='terrain')
        plt.title("Cropped AOP DEM")
        plt.colorbar(label="Elevation (m)")
        plt.show()

    except Exception as e:
        print("❌ Failed to crop DEM:", e)
        return



    #My Slope

    slope_map = calculate_slope(cropped_dem)
    invalid_mask, roughness_map, _ = mask_invalid_terrain(cropped_dem, slope_map, slope_thresh=30)

    # Visualize roughness
    plt.figure(figsize=(8, 5))
    plt.imshow(roughness_map, cmap='viridis')
    plt.title("🪨 Terrain Roughness (Standard Deviation)")
    plt.colorbar(label='Roughness (m)')
    plt.show()

    # 🔍 Diagnostics for Terrain Constraints

    # Print how many pixels were masked
    num_invalid = np.sum(invalid_mask)
    total_pixels = cropped_dem.size
    print(
        f"❌ Invalid pixels due to constraints: {num_invalid} out of {total_pixels} ({(num_invalid / total_pixels) * 100:.2f}%)")

    # Visualize slope > threshold
    plt.figure(figsize=(8, 5))
    plt.imshow(slope_map > np.tan(np.radians(30)), cmap='Reds', alpha=0.7)
    plt.title("⚠️ Slopes > 30°")
    plt.colorbar(label="Masked (1 = True)")
    plt.show()

    #might get removed
    # Visualize the combined invalid mask (all constraints)
    plt.figure(figsize=(8, 5))
    plt.imshow(invalid_mask, cmap='gray')
    plt.title("❌ Combined Invalid Terrain Mask")
    plt.colorbar(label="Invalid (1 = True)")
    plt.show()

    from radar_optimization import RadarOptimizer

    # 🧬 Optimized Radar Deployment using PyGAD
    print("\n🚀 Running PyGAD Optimization for Radar Placement...\n")

    radar_height = radar_specs[0]['Height']  # Assume all radars have same height
    radar_freq = radar_specs[0]['Frequency_Hz']
    Pt = radar_specs[0]['Pt']
    G_dBi = radar_specs[0]['G_dBi']

    r_max = compute_radar_range(Pt, G_dBi, radar_freq, sigma=1.0, S_min=1e-10)
    pixel_size = 30
    radius_pixels = int(r_max / pixel_size)

    if loaded_positions is None:
        optimizer = RadarOptimizer(
            dem=cropped_dem,
            invalid_mask=invalid_mask,
            num_radars=num_radars,
            radius_pixels=radius_pixels,
            radar_height=radar_height
        )
        optimized_positions = optimizer.optimize(generations=5, sol_per_pop=5)

        # Save new model
        os.makedirs('saved_models', exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_filename = f"saved_models/radar_positions_{timestamp}.pkl"
        with open(model_filename, 'wb') as f:
            pickle.dump(optimized_positions, f)
        print(f"\n💾 Optimized radar positions saved to: {model_filename}")
    else:
        optimized_positions = loaded_positions


    """ optimizer = RadarOptimizer(
        dem=cropped_dem,
        invalid_mask=invalid_mask,
        num_radars=num_radars,
        radius_pixels=radius_pixels,
        radar_height=radar_height
    )

    optimized_positions = optimizer.optimize(generations=5, sol_per_pop=5)
    """

    # Plot final coverage
    final_coverage = np.zeros_like(cropped_dem, dtype=bool)
    plt.figure(figsize=(10, 8))
    plt.imshow(cropped_dem, cmap='terrain')

    for i, (y, x) in enumerate(optimized_positions):
        coverage = get_radar_coverage(cropped_dem, (x, y), radius_pixels, radar_height)
        coverage &= ~invalid_mask
        final_coverage |= coverage
        plt.imshow(coverage, cmap='Greens', alpha=0.3)
        plt.scatter(x, y, c='red', marker='x', s=100, label=f'Radar {i + 1}')

    plt.legend()
    plt.title("📡 Optimized Radar Coverage Map (PyGAD)")
    plt.colorbar(label='Elevation (m)')
    plt.show()

    # Final Coverage Stats
    pixel_area_km2 = (pixel_size / 1000) ** 2
    total_pixels = cropped_dem.size
    total_area_km2 = total_pixels * pixel_area_km2
    covered_pixels = np.sum(final_coverage)
    covered_area_km2 = covered_pixels * pixel_area_km2
    coverage_percent = (covered_pixels / total_pixels) * 100

    print(f"\n📏 AOP Total Area     : {total_area_km2:.2f} km²")
    print(f"📡 Optimized Covered  : {covered_area_km2:.2f} km²")
    print(f"✅ Optimized Coverage : {coverage_percent:.2f}%")


# removed it as it is now in optimizer = RadarOptimizer for giving the plotted maps
"""
    # Create directory if it doesn't exist
    os.makedirs('saved_models', exist_ok=True)

    # Generate timestamped filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = f"saved_models/radar_positions_{timestamp}.pkl"

    # Save optimized positions
    with open(model_filename, 'wb') as f:
        pickle.dump(optimized_positions, f)

    print(f"\n💾 Optimized radar positions saved to: {model_filename}")
"""


# before the ga optimization for radar

"""
    #My Radar coverage


    final_coverage = np.zeros_like(cropped_dem, dtype=bool)
    plt.figure(figsize=(10, 8))
    plt.imshow(cropped_dem, cmap='terrain')

    Smin = 1e-10  # Minimum detectable signal
    c = 3e8

    for i, radar_config in enumerate(radar_specs):
        radar_height = radar_config['Height']
        radar_freq = radar_config['Frequency_Hz']
        Pt = radar_config['Pt']
        G_dBi = radar_config['G_dBi']
        G = 10 ** (G_dBi / 10)
        lambda_ = c / radar_freq
        pixel_size = 30

        r_max = compute_radar_range(Pt, G_dBi, radar_freq, sigma=1.0, S_min=1e-10)
        radius_pixels = int(r_max / pixel_size)
        print(f"Radar {i + 1} — Rmax = {r_max / 1000:.2f} km, Radius in pixels = {radius_pixels}")

        origin = (
            cropped_dem.shape[1] // (num_radars + 1) * (i + 1),
            cropped_dem.shape[0] // (num_radars + 1) * (i + 1)
        )

        coverage = get_radar_coverage(cropped_dem, origin, radius_pixels, radar_height)
        coverage &= ~invalid_mask
        final_coverage |= coverage

        plt.imshow(coverage, cmap='Greens', alpha=0.3)
        plt.scatter(origin[0], origin[1], c='red', marker='x', s=100, label=f'Radar {i + 1}')

    plt.legend()
    plt.title("Radar Coverage Map (All Radars)")
    plt.colorbar(label='Elevation (m)')
    plt.show()

    covered_pixels = np.sum(final_coverage)
    total_pixels = cropped_dem.size
    coverage_percent = (covered_pixels / total_pixels) * 100

    pixel_area_km2 = (pixel_size / 1000) ** 2
    total_area_km2 = total_pixels * pixel_area_km2
    covered_area_km2 = covered_pixels * pixel_area_km2

    print(f"\n📏 AOP Total Area     : {total_area_km2:.2f} km²")
    print(f"📡 Covered Area       : {covered_area_km2:.2f} km²")
    print(f"✅ Coverage Percentage: {coverage_percent:.2f}%")
"""


if __name__ == "__main__":
    main()

#rajasthan
#76.55
#27.45
#76.65        80
#27.55        70


#delhi
# 77.21
# 28.46
# 77.25
# 28.49

#sample
#44.25
#-97.50
#44.60
#-97.15

#delhi2
#77.21
#28.48
#77.46
#28.73

# Freq 1.1e9