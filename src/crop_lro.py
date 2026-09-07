import glob
import math
import rasterio
from rasterio.windows import from_bounds
import numpy as np
import cv2

# LRO reference GeoTIFF
src_path = glob.glob("data/reference/*.TIF")[0]

# EXACT same geographic bounds used for Chandrayaan
min_lon = 339.753
max_lon = 340.087
min_lat = 9.455
max_lat = 9.785

# Lunar radius used by this LRO projection
R = 1737400.0

# LRO GeoTIFF uses Equirectangular projection
# with standard parallel = 10 degrees
standard_parallel = math.radians(10.0)

def lon_to_x(lon_deg):
    return (
        R
        * math.radians(lon_deg)
        * math.cos(standard_parallel)
    )

def lat_to_y(lat_deg):
    return R * math.radians(lat_deg)

# Convert lunar lon/lat into projected metres
left = lon_to_x(min_lon)
right = lon_to_x(max_lon)
bottom = lat_to_y(min_lat)
top = lat_to_y(max_lat)

print("Projected bounds:")
print("Left  :", left)
print("Right :", right)
print("Bottom:", bottom)
print("Top   :", top)

output_tif = "data/test_pairs/pair_001/lro.tif"
output_png = "data/test_pairs/pair_001/lro.png"

with rasterio.open(src_path) as src:

    window = from_bounds(
        left,
        bottom,
        right,
        top,
        transform=src.transform
    )

    window = window.round_offsets().round_lengths()

    print()
    print("Crop window:", window)
    print(
        "Crop size:",
        int(window.width),
        "x",
        int(window.height)
    )

    image = src.read(1, window=window)

    profile = src.profile.copy()
    profile.update(
        height=image.shape[0],
        width=image.shape[1],
        transform=src.window_transform(window),
        compress="lzw"
    )

    with rasterio.open(output_tif, "w", **profile) as dst:
        dst.write(image, 1)

# Ignore nodata pixels for display stretch
valid = image[image > 0]

if valid.size > 0:
    low = np.percentile(valid, 1)
    high = np.percentile(valid, 99)

    image8 = np.clip(
        (image.astype(np.float32) - low)
        / max(high - low, 1)
        * 255,
        0,
        255
    ).astype(np.uint8)
else:
    image8 = image

cv2.imwrite(output_png, image8)

print()
print("Saved:")
print(output_tif)
print(output_png)

print()
print("Intensity:", image.min(), "→", image.max())