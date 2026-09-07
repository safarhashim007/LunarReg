import os
import math

import cv2
import torch
import rasterio
import numpy as np

from romav2 import RoMaV2


# =========================================================
# CONFIG
# =========================================================

PAIR = "data/test_pairs/pair_002"
OUT = "outputs/pair_002"

CH_PNG = f"{PAIR}/chandrayaan.png"
LRO_PNG = f"{PAIR}/lro.png"

CH_TIF = f"{PAIR}/chandrayaan.tif"
LRO_TIF = f"{PAIR}/lro.tif"

N_MATCHES = 3000

# LRO projection parameters
MOON_RADIUS = 1737400.0
STANDARD_PARALLEL = math.radians(10.0)

os.makedirs(OUT, exist_ok=True)

torch.set_float32_matmul_precision("highest")


# =========================================================
# LOAD IMAGES
# =========================================================

ch_img = cv2.imread(
    CH_PNG,
    cv2.IMREAD_GRAYSCALE
)

lro_img = cv2.imread(
    LRO_PNG,
    cv2.IMREAD_GRAYSCALE
)

if ch_img is None:
    raise RuntimeError(
        f"Could not load {CH_PNG}"
    )

if lro_img is None:
    raise RuntimeError(
        f"Could not load {LRO_PNG}"
    )


H_A, W_A = ch_img.shape
H_B, W_B = lro_img.shape


print("===== EXACT GEOREFERENCE EVALUATION =====")
print()
print("Chandrayaan:", W_A, "x", H_A)
print("LRO        :", W_B, "x", H_B)


# =========================================================
# LOAD ROMA
# =========================================================

print()
print("Loading RoMa v2...")

model = RoMaV2()

model.apply_setting("fast")

model.balanced_sampling = True

print("Model loaded.")
print(
    "Balanced sampling:",
    model.balanced_sampling
)


# =========================================================
# DENSE MATCHING
# =========================================================

print()
print("Running dense matching...")

with torch.inference_mode():

    predictions = model.match(
        CH_PNG,
        LRO_PNG
    )

print("Dense matching complete.")


# =========================================================
# SAMPLE MATCHES
# =========================================================

print()
print(
    "Sampling",
    N_MATCHES,
    "matches..."
)

matches, overlaps, precision_AB, precision_BA = (
    model.sample(
        predictions,
        N_MATCHES
    )
)


# =========================================================
# CONVERT TO PIXEL COORDINATES
# =========================================================

kptsA, kptsB = (
    model.to_pixel_coordinates(
        matches,
        H_A,
        W_A,
        H_B,
        W_B
    )
)


ptsA = (
    kptsA
    .detach()
    .cpu()
    .numpy()
    .astype(np.float64)
)

ptsB = (
    kptsB
    .detach()
    .cpu()
    .numpy()
    .astype(np.float64)
)


print(
    "Sampled correspondences:",
    len(ptsA)
)


# =========================================================
# LOAD GEOTIFF TRANSFORMS
# =========================================================

with rasterio.open(CH_TIF) as ch_ds:

    ch_transform = ch_ds.transform

    print()
    print("Chandrayaan GeoTIFF:")
    print("CRS   :", ch_ds.crs)
    print("Bounds:", ch_ds.bounds)


with rasterio.open(LRO_TIF) as lro_ds:

    lro_transform = lro_ds.transform

    print()
    print("LRO GeoTIFF:")
    print("CRS   :", lro_ds.crs)
    print("Bounds:", lro_ds.bounds)


# =========================================================
# CHANDRAYAAN PIXEL -> LUNAR LONGITUDE/LATITUDE
#
# Use pixel centres.
# =========================================================

cols_A = (
    ptsA[:, 0]
    +
    0.5
)

rows_A = (
    ptsA[:, 1]
    +
    0.5
)


lon = (
    ch_transform.a * cols_A
    +
    ch_transform.b * rows_A
    +
    ch_transform.c
)


lat = (
    ch_transform.d * cols_A
    +
    ch_transform.e * rows_A
    +
    ch_transform.f
)


# =========================================================
# LUNAR LON/LAT -> LRO PROJECTED METRES
#
# LRO uses equirectangular projection.
# Longitude remains in 0–360 convention.
# =========================================================

x_lro = (
    MOON_RADIUS
    *
    np.radians(lon)
    *
    math.cos(
        STANDARD_PARALLEL
    )
)


y_lro = (
    MOON_RADIUS
    *
    np.radians(lat)
)


# =========================================================
# PROJECTED METRES -> EXPECTED LRO PIXEL
# =========================================================

inverse_lro_transform = (
    ~lro_transform
)


expected_x = (
    inverse_lro_transform.a * x_lro
    +
    inverse_lro_transform.b * y_lro
    +
    inverse_lro_transform.c
)


expected_y = (
    inverse_lro_transform.d * x_lro
    +
    inverse_lro_transform.e * y_lro
    +
    inverse_lro_transform.f
)


expected_B = np.column_stack(
    (
        expected_x,
        expected_y
    )
)


# =========================================================
# REMOVE EXPECTED POINTS OUTSIDE LRO IMAGE
# =========================================================

inside = (
    (expected_B[:, 0] >= 0)
    &
    (expected_B[:, 0] < W_B)
    &
    (expected_B[:, 1] >= 0)
    &
    (expected_B[:, 1] < H_B)
)


ptsA_valid = ptsA[inside]
ptsB_valid = ptsB[inside]
expected_B_valid = expected_B[inside]


print()
print(
    "Matches with valid geospatial reference:",
    len(ptsA_valid),
    "/",
    len(ptsA)
)


if len(ptsA_valid) == 0:
    raise RuntimeError(
        "No expected points fall inside LRO image."
    )


# =========================================================
# EXACT GEOREFERENCE ERROR
# =========================================================

errors = np.linalg.norm(
    ptsB_valid
    -
    expected_B_valid,
    axis=1
)


print()
print("===== ROMA GEO ERROR =====")

print(
    "Minimum :",
    round(
        float(
            np.min(errors)
        ),
        3
    ),
    "px"
)

print(
    "Median  :",
    round(
        float(
            np.median(errors)
        ),
        3
    ),
    "px"
)

print(
    "Mean    :",
    round(
        float(
            np.mean(errors)
        ),
        3
    ),
    "px"
)

print(
    "P75     :",
    round(
        float(
            np.percentile(
                errors,
                75
            )
        ),
        3
    ),
    "px"
)

print(
    "P90     :",
    round(
        float(
            np.percentile(
                errors,
                90
            )
        ),
        3
    ),
    "px"
)

print(
    "Maximum :",
    round(
        float(
            np.max(errors)
        ),
        3
    ),
    "px"
)


# =========================================================
# ACCURACY COUNTS
# =========================================================

thresholds = [
    2,
    5,
    10,
    20,
    40
]


print()
print("===== ACCURACY COUNTS =====")


for threshold in thresholds:

    count = int(
        np.sum(
            errors <= threshold
        )
    )

    percentage = (
        count
        /
        len(errors)
        *
        100
    )

    print(
        f"<= {threshold:2d} px : "
        f"{count:4d} "
        f"({percentage:.2f}%)"
    )


# =========================================================
# STRONG MATCH SET
# =========================================================

GOOD_THRESHOLD = 10.0


good_mask = (
    errors <= GOOD_THRESHOLD
)


good_A = (
    ptsA_valid[good_mask]
)

good_B = (
    ptsB_valid[good_mask]
)

good_expected_B = (
    expected_B_valid[good_mask]
)

good_errors = (
    errors[good_mask]
)


print()
print(
    "Strong matches <= 10 px:",
    len(good_A)
)


if len(good_A) > 0:

    print(
        "Strong match RMSE:",
        round(
            float(
                np.sqrt(
                    np.mean(
                        good_errors ** 2
                    )
                )
            ),
            3
        ),
        "px"
    )


# =========================================================
# 8x8 GRID COVERAGE
# =========================================================

GRID = 8

occupied_cells = set()


for x, y in good_A:

    col = int(
        x
        /
        W_A
        *
        GRID
    )

    row = int(
        y
        /
        H_A
        *
        GRID
    )

    col = min(
        max(
            col,
            0
        ),
        GRID - 1
    )

    row = min(
        max(
            row,
            0
        ),
        GRID - 1
    )

    occupied_cells.add(
        (
            row,
            col
        )
    )


coverage = (
    len(occupied_cells)
    /
    (
        GRID
        *
        GRID
    )
)


print(
    "Occupied cells:",
    len(occupied_cells),
    "/",
    GRID * GRID
)

print(
    "8x8 grid coverage:",
    round(
        coverage
        *
        100,
        2
    ),
    "%"
)


# =========================================================
# DISTANCE IN METRES
#
# LRO image ≈ 20 metres per pixel.
# =========================================================

METRES_PER_LRO_PIXEL = 20.0


errors_metres = (
    errors
    *
    METRES_PER_LRO_PIXEL
)


print()
print("===== APPROXIMATE POSITION ERROR =====")

print(
    "Median:",
    round(
        float(
            np.median(
                errors_metres
            )
        ),
        2
    ),
    "m"
)

print(
    "Mean  :",
    round(
        float(
            np.mean(
                errors_metres
            )
        ),
        2
    ),
    "m"
)


# =========================================================
# VISUALIZATION
# =========================================================

ch_small = cv2.resize(
    ch_img,
    (
        W_B,
        H_B
    ),
    interpolation=cv2.INTER_AREA
)


ch_vis = cv2.cvtColor(
    ch_small,
    cv2.COLOR_GRAY2BGR
)

lro_vis = cv2.cvtColor(
    lro_img,
    cv2.COLOR_GRAY2BGR
)


canvas = np.hstack(
    (
        ch_vis,
        lro_vis
    )
)


scale_x = (
    W_B
    /
    W_A
)

scale_y = (
    H_B
    /
    H_A
)


# Indices of good matches
good_indices = np.where(
    good_mask
)[0]


# Limit number of visualized lines
MAX_DRAW = 200


if len(good_indices) > MAX_DRAW:

    selected = np.linspace(
        0,
        len(good_indices) - 1,
        MAX_DRAW
    ).astype(int)

    good_indices = (
        good_indices[selected]
    )


rng = np.random.default_rng(
    42
)


for idx in good_indices:

    pA = ptsA_valid[idx]
    pB = ptsB_valid[idx]

    ax = int(
        round(
            pA[0]
            *
            scale_x
        )
    )

    ay = int(
        round(
            pA[1]
            *
            scale_y
        )
    )

    bx = int(
        round(
            pB[0]
        )
    ) + W_B

    by = int(
        round(
            pB[1]
        )
    )


    colour_values = rng.integers(
        70,
        256,
        size=3
    )

    colour = tuple(
        int(v)
        for v in colour_values
    )


    cv2.circle(
        canvas,
        (
            ax,
            ay
        ),
        2,
        colour,
        -1
    )

    cv2.circle(
        canvas,
        (
            bx,
            by
        ),
        2,
        colour,
        -1
    )

    cv2.line(
        canvas,
        (
            ax,
            ay
        ),
        (
            bx,
            by
        ),
        colour,
        1,
        cv2.LINE_AA
    )


cv2.imwrite(
    f"{OUT}/roma_exact_geo_matches.png",
    canvas
)


# =========================================================
# ERROR HEATMAP
# =========================================================

heatmap = np.zeros(
    (
        H_B,
        W_B
    ),
    dtype=np.float32
)


for point, error in zip(
    ptsB_valid,
    errors
):

    x = int(
        round(
            point[0]
        )
    )

    y = int(
        round(
            point[1]
        )
    )

    if (
        0 <= x < W_B
        and
        0 <= y < H_B
    ):

        heatmap[
            y,
            x
        ] = max(
            heatmap[y, x],
            error
        )


heatmap = cv2.GaussianBlur(
    heatmap,
    (
        0,
        0
    ),
    sigmaX=8
)


if heatmap.max() > 0:

    heatmap_normalized = (
        heatmap
        /
        heatmap.max()
        *
        255
    ).astype(np.uint8)

else:

    heatmap_normalized = np.zeros(
        heatmap.shape,
        dtype=np.uint8
    )


heatmap_colour = cv2.applyColorMap(
    heatmap_normalized,
    cv2.COLORMAP_JET
)


cv2.imwrite(
    f"{OUT}/roma_geo_error_heatmap.png",
    heatmap_colour
)


# =========================================================
# SAVE NUMERICAL DATA
# =========================================================

np.save(
    f"{OUT}/roma_exact_ptsA.npy",
    ptsA_valid
)

np.save(
    f"{OUT}/roma_exact_ptsB.npy",
    ptsB_valid
)

np.save(
    f"{OUT}/roma_expected_ptsB.npy",
    expected_B_valid
)

np.save(
    f"{OUT}/roma_exact_errors.npy",
    errors
)

np.save(
    f"{OUT}/roma_exact_good_mask.npy",
    good_mask
)


# =========================================================
# FINISH
# =========================================================

print()
print("Saved:")

print(
    f"{OUT}/roma_exact_geo_matches.png"
)

print(
    f"{OUT}/roma_geo_error_heatmap.png"
)

print(
    f"{OUT}/roma_exact_errors.npy"
)

print()
print("Done.")