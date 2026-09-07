import os
import cv2
import rasterio
import numpy as np

from rasterio.warp import reproject, Resampling


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


# --------------------------------------------------
# Open reference LRO image
# --------------------------------------------------

with rasterio.open(lro_path) as lro:
    lro_data = lro.read(1)

    print("LRO")
    print("Size:", lro.width, "x", lro.height)
    print("CRS:", lro.crs)
    print("Bounds:", lro.bounds)

    # Destination array uses EXACT LRO grid
    ch_on_lro = np.zeros(
        (lro.height, lro.width),
        dtype=np.float32
    )

    # --------------------------------------------------
    # Reproject Chandrayaan directly onto LRO grid
    # --------------------------------------------------

    with rasterio.open(ch_path) as ch:

        print()
        print("Chandrayaan")
        print("Size:", ch.width, "x", ch.height)
        print("CRS:", ch.crs)
        print("Bounds:", ch.bounds)

        reproject(
            source=rasterio.band(ch, 1),
            destination=ch_on_lro,

            src_transform=ch.transform,
            src_crs=ch.crs,

            dst_transform=lro.transform,
            dst_crs=lro.crs,

            resampling=Resampling.bilinear
        )


# --------------------------------------------------
# Display versions
# --------------------------------------------------

ch8 = stretch(ch_on_lro)
lro8 = stretch(lro_data, ignore_zero=True)

cv2.imwrite(
    f"{OUT}/georef_chandrayaan.png",
    ch8
)

cv2.imwrite(
    f"{OUT}/georef_lro.png",
    lro8
)


# --------------------------------------------------
# 50/50 overlay
# --------------------------------------------------

overlay = cv2.addWeighted(
    ch8,
    0.5,
    lro8,
    0.5,
    0
)

cv2.imwrite(
    f"{OUT}/georef_overlay.png",
    overlay
)


# --------------------------------------------------
# Side by side
# --------------------------------------------------

separator = np.full(
    (lro8.shape[0], 10),
    255,
    dtype=np.uint8
)

side_by_side = np.hstack([
    ch8,
    separator,
    lro8
])

cv2.imwrite(
    f"{OUT}/georef_side_by_side.png",
    side_by_side
)


print()
print("Saved:")
print(f"{OUT}/georef_chandrayaan.png")
print(f"{OUT}/georef_lro.png")
print(f"{OUT}/georef_overlay.png")
print(f"{OUT}/georef_side_by_side.png")