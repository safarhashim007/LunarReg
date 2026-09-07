import os
import glob
import math

import cv2
import numpy as np
import rasterio

from rasterio.windows import from_bounds


# ---------------------------------------------------------
# PATHS
# ---------------------------------------------------------

TMC_PATH = glob.glob(
    "data/chandrayaan/tmc_20240623/*.tif"
)[0]

LRO_PATH = glob.glob(
    "data/reference/*.TIF"
)[0]

OUT = "data/test_pairs/pair_002"

os.makedirs(OUT, exist_ok=True)


# ---------------------------------------------------------
# NEW AOI
#
# Shifted east so that we are INSIDE the actual TMC swath.
# Still inside the Copernicus LRO mosaic.
# ---------------------------------------------------------

MIN_LON = 340.12
MAX_LON = 340.42

MIN_LAT = 9.455
MAX_LAT = 9.785


# ---------------------------------------------------------
# UTILITY: DISPLAY STRETCH
# ---------------------------------------------------------

def stretch(image):
    image = image.astype(np.float32)

    valid = image[image > 0]

    if valid.size == 0:
        return np.zeros(image.shape, dtype=np.uint8)

    low = np.percentile(valid, 1)
    high = np.percentile(valid, 99)

    if high <= low:
        high = low + 1

    result = (
        (image - low)
        / (high - low)
        * 255
    )

    return np.clip(
        result,
        0,
        255
    ).astype(np.uint8)


# =========================================================
# 1. CHANDRAYAAN TMC-2 CROP
# =========================================================

print("===== CHANDRAYAAN =====")

with rasterio.open(TMC_PATH) as src:

    window = from_bounds(
        MIN_LON,
        MIN_LAT,
        MAX_LON,
        MAX_LAT,
        transform=src.transform
    )

    window = window.round_offsets().round_lengths()

    tmc = src.read(
        1,
        window=window
    )

    tmc_transform = src.window_transform(window)

    print("Crop size:", tmc.shape)

    total = tmc.size
    valid = np.count_nonzero(tmc)

    coverage = valid / total * 100

    print(
        "Valid pixels:",
        valid,
        "/",
        total
    )

    print(
        "Valid coverage:",
        round(coverage, 2),
        "%"
    )

    profile = src.profile.copy()

    profile.update(
        height=tmc.shape[0],
        width=tmc.shape[1],
        transform=tmc_transform,
        compress="lzw"
    )

    with rasterio.open(
        f"{OUT}/chandrayaan.tif",
        "w",
        **profile
    ) as dst:
        dst.write(tmc, 1)


tmc8 = stretch(tmc)

cv2.imwrite(
    f"{OUT}/chandrayaan.png",
    tmc8
)


# =========================================================
# 2. LRO CROP
# =========================================================

print()
print("===== LRO =====")

# Lunar radius from LRO CRS
R = 1737400.0

# LRO mosaic uses standard parallel 10°
STANDARD_PARALLEL = math.radians(10.0)


def lon_to_x(lon):
    return (
        R
        * math.radians(lon)
        * math.cos(STANDARD_PARALLEL)
    )


def lat_to_y(lat):
    return (
        R
        * math.radians(lat)
    )


left = lon_to_x(MIN_LON)
right = lon_to_x(MAX_LON)

bottom = lat_to_y(MIN_LAT)
top = lat_to_y(MAX_LAT)


with rasterio.open(LRO_PATH) as src:

    window = from_bounds(
        left,
        bottom,
        right,
        top,
        transform=src.transform
    )

    window = window.round_offsets().round_lengths()

    lro = src.read(
        1,
        window=window
    )

    lro_transform = src.window_transform(window)

    print("Crop size:", lro.shape)

    total = lro.size
    valid = np.count_nonzero(lro)

    coverage = valid / total * 100

    print(
        "Valid coverage:",
        round(coverage, 2),
        "%"
    )

    profile = src.profile.copy()

    profile.update(
        height=lro.shape[0],
        width=lro.shape[1],
        transform=lro_transform,
        compress="lzw"
    )

    with rasterio.open(
        f"{OUT}/lro.tif",
        "w",
        **profile
    ) as dst:
        dst.write(lro, 1)


lro8 = stretch(lro)

cv2.imwrite(
    f"{OUT}/lro.png",
    lro8
)


# =========================================================
# 3. VISUAL SANITY CHECK
# =========================================================

tmc_small = cv2.resize(
    tmc8,
    (lro8.shape[1], lro8.shape[0]),
    interpolation=cv2.INTER_AREA
)

separator = np.full(
    (lro8.shape[0], 10),
    255,
    dtype=np.uint8
)

side = np.hstack(
    [
        tmc_small,
        separator,
        lro8
    ]
)

cv2.imwrite(
    f"{OUT}/side_by_side.png",
    side
)


# =========================================================
# FINISH
# =========================================================

print()
print("===== PAIR 002 CREATED =====")

print("AOI:")
print(
    f"Longitude: {MIN_LON} → {MAX_LON}"
)
print(
    f"Latitude : {MIN_LAT} → {MAX_LAT}"
)

print()

print("Saved:")
print(f"{OUT}/chandrayaan.tif")
print(f"{OUT}/chandrayaan.png")
print(f"{OUT}/lro.tif")
print(f"{OUT}/lro.png")
print(f"{OUT}/side_by_side.png")