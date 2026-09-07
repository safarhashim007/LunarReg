import glob
import rasterio
from rasterio.windows import from_bounds
import numpy as np
import cv2

# Full Chandrayaan-2 TMC-2 image
src_path = glob.glob(
    "data/chandrayaan/tmc_20240623/*.tif"
)[0]

# Copernicus approximate center
center_lat = 9.62
center_lon = 339.92  # same as 20.08° W

# ~10 km × 10 km region
min_lon = 339.753
max_lon = 340.087
min_lat = 9.455
max_lat = 9.785

output_tif = "data/test_pairs/pair_001/chandrayaan.tif"
output_png = "data/test_pairs/pair_001/chandrayaan.png"

with rasterio.open(src_path) as src:
    window = from_bounds(
        min_lon,
        min_lat,
        max_lon,
        max_lat,
        transform=src.transform
    )

    window = window.round_offsets().round_lengths()

    print("Crop window:", window)
    print("Crop size:", int(window.width), "x", int(window.height))

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

# Convert uint16 lunar data to visible 8-bit PNG
valid = image[np.isfinite(image)]

low = np.percentile(valid, 1)
high = np.percentile(valid, 99)

image8 = np.clip(
    (image.astype(np.float32) - low) /
    (high - low) * 255,
    0,
    255
).astype(np.uint8)

cv2.imwrite(output_png, image8)

print()
print("Saved:")
print(output_tif)
print(output_png)
print()
print("Intensity range:", image.min(), "→", image.max())
print("Display stretch:", low, "→", high)