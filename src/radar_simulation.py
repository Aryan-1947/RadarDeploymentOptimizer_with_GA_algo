# src/radar_simulation.py

import numpy as np
from shapely.geometry import Point
from math import radians, degrees, atan2, sqrt

#Not even used
# def haversine_distance(x1, y1, x2, y2):
#     # Haversine formula to get distance in meters between lat/lon points
#     R = 6371000  # Earth radius in meters
#     x1, y1, x2, y2 = map(radians, [x1, y1, x2, y2])
#     dx = x2 - x1
#     dy = y2 - y1
#     a = np.sin(dy / 2) ** 2 + np.cos(y1) * np.cos(y2) * np.sin(dx / 2) ** 2
#     c = 2 * np.arcsin(np.sqrt(a))
#     return R * c

def line_of_sight(elev, x0, y0, x1, y1, radar_height):
    num = int(np.hypot(x1 - x0, y1 - y0))
    h0 = elev[y0, x0] + radar_height
    h1 = elev[y1, x1]

    for i in range(1, num):
        xi = int(x0 + i * (x1 - x0) / num)
        yi = int(y0 + i * (y1 - y0) / num)

        if xi < 0 or yi < 0 or xi >= elev.shape[1] or yi >= elev.shape[0]:
            continue

        terrain_height = elev[yi, xi]
        expected_height = h0 + (h1 - h0) * (i / num)

        if terrain_height > expected_height:
            return False
    return True



def get_radar_coverage(elev, origin, radius_pixels, radar_height):
    height, width = elev.shape
    ox, oy = origin
    covered = np.zeros_like(elev, dtype=bool)

    for y in range(max(0, oy - radius_pixels), min(height, oy + radius_pixels)):
        for x in range(max(0, ox - radius_pixels), min(width, ox + radius_pixels)):
            if np.hypot(x - ox, y - oy) <= radius_pixels:
                if line_of_sight(elev, ox, oy, x, y, radar_height=radar_height):
                    covered[y, x] = True
    return covered