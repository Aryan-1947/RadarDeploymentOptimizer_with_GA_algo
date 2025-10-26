import math

def get_area_name():
    area = input("Enter the name of the state (e.g., rajasthan): ").strip().lower()
    return area

def get_num_radars():
    while True:
        try:
            n = int(input("Enter number of radars to deploy: "))
            if n > 0:
                return n
            else:
                print("Must be greater than 0.")
        except ValueError:
            print("Invalid input. Please enter a number.")


def compute_radar_range(Pt, G_dBi, f_Hz, sigma=1.0, S_min=1e-10):
    c = 3e8  # speed of light in m/s
    G = 10 ** (G_dBi / 10)  # convert dBi to linear
    wavelength = c / f_Hz

    numerator = Pt * (G ** 2) * (wavelength ** 2) * sigma
    denominator = (4 * math.pi) ** 3 * S_min

    R_max = (numerator / denominator) ** 0.25
    return R_max  # in meters

def get_crop_coordinates():
    print("\n📐 Enter bounding box coordinates for Area of Interest (AOP):")
    xmin = float(input("  - ymin (latitude): "))
    ymin = float(input("  - xmin (longitude): "))
    xmax = float(input("  - ymax (latitude): "))
    ymax = float(input("  - xmax (longitude): "))
    return xmin, ymin, xmax, ymax


def get_radar_specs(n):
    specs = []
    for i in range(n):
        print(f"\n🔧 Enter technical details for Radar {i+1}:")
        Pt = float(input("  - Transmit Power (Watts): "))
        G_dBi = float(input("  - Antenna Gain (dBi): "))
        freq_Hz = float(input("  - Frequency (Hz): "))
        height = float(input("  - Antenna Height above ground (meters): "))


        range_m = compute_radar_range(Pt, G_dBi, freq_Hz)

        print(f"  ✅ Computed Max Range: {range_m/1000:.2f} km")

        specs.append({
            "Pt": Pt,
            "G_dBi": G_dBi,
            "Frequency_Hz": freq_Hz,
            "Height": height,
            "Computed_Range": range_m
        })

    return specs
