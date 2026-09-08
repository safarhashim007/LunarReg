import os
import cv2
import rasterio
import numpy as np

PAIR = "data/test_pairs/pair_001"
OUT = "outputs/pair_001"

os.makedirs(OUT, exist_ok=True)

ch_path = f"{PAIR}/chandrayaan.tif"
lro_path = f"{PAIR}/lro.tif"


def stretch(image, ignore_zero=False):
    image = image.astype(np.float32)

    if ignore_zero:
        valid = image[image > 0]
    else:
        valid = image[np.isfinite(image)]

    if valid.size == 0:
        return np.zeros(image.shape, dtype=np.uint8)

    low = np.percentile(valid, 1)
    high = np.percentile(valid, 99)

    if high <= low:
        high = low + 1

    result = (image - low) / (high - low)
    result = np.clip(result, 0, 1)

    return (result * 255).astype(np.uint8)


# Load both georeferenced crops
with rasterio.open(ch_path) as ch:
    ch_data = ch.read(1)
    print("Chandrayaan size:", ch_data.shape)
    print("Chandrayaan bounds:", ch.bounds)

with rasterio.open(lro_path) as lro:
    lro_data = lro.read(1)
    print("LRO size:", lro_data.shape)
    print("LRO bounds:", lro.bounds)


# Display normalization
ch8 = stretch(ch_data)
lro8 = stretch(lro_data, ignore_zero=True)


# Both crops cover the same geographic AOI.
# Resample Chandrayaan from ~5 m/px to LRO ~20 m/px.
ch_resampled = cv2.resize(
    ch8,
    (lro8.shape[1], lro8.shape[0]),
    interpolation=cv2.INTER_AREA
)


cv2.imwrite(
    f"{OUT}/georef_chandrayaan.png",
    ch_resampled
)

cv2.imwrite(
    f"{OUT}/georef_lro.png",
    lro8
)


# Side by side
separator = np.full(
    (lro8.shape[0], 10),
    255,
    dtype=np.uint8
)

side = np.hstack([
    ch_resampled,
    separator,
    lro8
])

cv2.imwrite(
    f"{OUT}/georef_side_by_side.png",
    side
)


# Overlay
overlay = cv2.addWeighted(
    ch_resampled,
    0.5,
    lro8,
    0.5,
    0
)

cv2.imwrite(
    f"{OUT}/georef_overlay.png",
    overlay
)

print("\nSaved verification images.")