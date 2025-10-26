#this file is used after loading dem --> to crop it according to
#the coordinates provided by the user. after which it will be called as aop

from rasterio.transform import rowcol


def latlon_to_pixel_bounds(transform, bounds):
    """
    Convert lat/lon bounding box to raster pixel coordinates.
    """


    min_lat, min_lon, max_lat, max_lon = bounds

    # rowcol expects (lon, lat)
    row_min, col_min = rowcol(transform, min_lon, max_lat)  # top-left
    row_max, col_max = rowcol(transform, max_lon, min_lat)  # bottom-right

    # Sort properly
    x_min, x_max = sorted([col_min, col_max])
    y_min, y_max = sorted([row_min, row_max])

    return x_min, y_min, x_max, y_max


def crop_dem(dem_data, x_min, y_min, x_max, y_max):
    """
    Crop the DEM array to AOP pixel bounds
    """
    return dem_data[y_min:y_max, x_min:x_max]


