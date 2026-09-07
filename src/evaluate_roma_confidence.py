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

N_MATCHES = 5000

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

if ch_img is None or lro_img is None:
    raise RuntimeError("Could not load image pair")

H_A, W_A = ch_img.shape
H_B, W_B = lro_img.shape


print("===== ROMA CONFIDENCE EVALUATION =====")
print()
print("Chandrayaan:", W_A, "x", H_A)
print("LRO        :", W_B, "x", H_B)


# =========================================================
# ROMA
# =========================================================

print()
print("Loading RoMa v2...")

model = RoMaV2()

model.apply_setting("fast")
model.balanced_sampling = True


print("Running dense matching...")

with torch.inference_mode():

    predictions = model.match(
        CH_PNG,
        LRO_PNG
    )


print("Sampling correspondences...")

matches, overlaps, precision_AB, precision_BA = model.sample(
    predictions,
    N_MATCHES
)


kptsA, kptsB = model.to_pixel_coordinates(
    matches,
    H_A,
    W_A,
    H_B,
    W_B
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

confidence = (
    overlaps
    .detach()
    .cpu()
    .numpy()
    .reshape(-1)
    .astype(np.float64)
)


print("Sampled:", len(ptsA))

print()
print("Confidence:")
print("Min   :", round(float(confidence.min()), 5))
print("Median:", round(float(np.median(confidence)), 5))
print("Mean  :", round(float(confidence.mean()), 5))
print("Max   :", round(float(confidence.max()), 5))


# =========================================================
# EXACT GEOREFERENCE
# ONLY USED FOR EVALUATION
# =========================================================

with rasterio.open(CH_TIF) as ch_ds:
    ch_transform = ch_ds.transform

with rasterio.open(LRO_TIF) as lro_ds:
    lro_transform = lro_ds.transform


# Chandrayaan pixels -> longitude / latitude

cols = ptsA[:, 0] + 0.5
rows = ptsA[:, 1] + 0.5


lon = (
    ch_transform.a * cols
    +
    ch_transform.b * rows
    +
    ch_transform.c
)

lat = (
    ch_transform.d * cols
    +
    ch_transform.e * rows
    +
    ch_transform.f
)


# lon / lat -> LRO projected coordinates

x_lro = (
    MOON_RADIUS
    *
    np.radians(lon)
    *
    math.cos(STANDARD_PARALLEL)
)

y_lro = (
    MOON_RADIUS
    *
    np.radians(lat)
)


# projected coordinates -> LRO pixels

inv = ~lro_transform


expected_x = (
    inv.a * x_lro
    +
    inv.b * y_lro
    +
    inv.c
)

expected_y = (
    inv.d * x_lro
    +
    inv.e * y_lro
    +
    inv.f
)


expected_B = np.column_stack(
    [
        expected_x,
        expected_y
    ]
)


inside = (
    (expected_B[:, 0] >= 0)
    &
    (expected_B[:, 0] < W_B)
    &
    (expected_B[:, 1] >= 0)
    &
    (expected_B[:, 1] < H_B)
)


ptsA = ptsA[inside]
ptsB = ptsB[inside]
expected_B = expected_B[inside]
confidence = confidence[inside]


errors = np.linalg.norm(
    ptsB - expected_B,
    axis=1
)


print()
print(
    "Valid evaluated matches:",
    len(errors)
)


# =========================================================
# CONFIDENCE VS ERROR
# =========================================================

print()
print("===== CONFIDENCE QUANTILES =====")


quantiles = [
    0,
    25,
    50,
    75,
    90,
    95,
    97,
    99
]


for q in quantiles:

    threshold = np.percentile(
        confidence,
        q
    )

    mask = confidence >= threshold

    current_errors = errors[mask]

    if len(current_errors) == 0:
        continue

    within_5 = np.mean(
        current_errors <= 5
    ) * 100

    within_10 = np.mean(
        current_errors <= 10
    ) * 100

    within_20 = np.mean(
        current_errors <= 20
    ) * 100


    print()
    print(
        f"Top {100-q}% "
        f"(confidence >= {threshold:.4f})"
    )

    print(
        "Matches :",
        len(current_errors)
    )

    print(
        "Median error:",
        round(
            float(
                np.median(current_errors)
            ),
            3
        ),
        "px"
    )

    print(
        "<= 5 px :",
        round(
            within_5,
            2
        ),
        "%"
    )

    print(
        "<=10 px :",
        round(
            within_10,
            2
        ),
        "%"
    )

    print(
        "<=20 px :",
        round(
            within_20,
            2
        ),
        "%"
    )


# =========================================================
# TOP-K CONFIDENCE TEST
# =========================================================

print()
print("===== TOP-K CONFIDENCE =====")


order = np.argsort(
    confidence
)[::-1]


top_values = [
    50,
    100,
    200,
    300,
    500,
    1000
]


for k in top_values:

    if k > len(order):
        continue

    indices = order[:k]

    e = errors[indices]

    correct_5 = np.sum(
        e <= 5
    )

    correct_10 = np.sum(
        e <= 10
    )

    correct_20 = np.sum(
        e <= 20
    )


    print()
    print("Top", k)

    print(
        "<= 5 px :",
        correct_5,
        "/",
        k,
        f"({correct_5/k*100:.2f}%)"
    )

    print(
        "<=10 px :",
        correct_10,
        "/",
        k,
        f"({correct_10/k*100:.2f}%)"
    )

    print(
        "<=20 px :",
        correct_20,
        "/",
        k,
        f"({correct_20/k*100:.2f}%)"
    )

    print(
        "Median error:",
        round(
            float(
                np.median(e)
            ),
            3
        ),
        "px"
    )


# =========================================================
# GRID COVERAGE FOR TOP 300
# =========================================================

K = min(
    300,
    len(order)
)

indices = order[:K]

topA = ptsA[indices]
top_errors = errors[indices]

GRID = 8

occupied = set()


for x, y in topA:

    col = min(
        max(
            int(
                x / W_A * GRID
            ),
            0
        ),
        GRID - 1
    )

    row = min(
        max(
            int(
                y / H_A * GRID
            ),
            0
        ),
        GRID - 1
    )

    occupied.add(
        (
            row,
            col
        )
    )


print()
print("===== TOP 300 COVERAGE =====")

print(
    "Grid cells:",
    len(occupied),
    "/",
    GRID * GRID
)

print(
    "Coverage:",
    round(
        len(occupied)
        /
        (GRID * GRID)
        *
        100,
        2
    ),
    "%"
)


# =========================================================
# SAVE CONFIDENCE / ERROR DATA
# =========================================================

np.save(
    f"{OUT}/roma_confidence.npy",
    confidence
)

np.save(
    f"{OUT}/roma_confidence_errors.npy",
    errors
)

np.save(
    f"{OUT}/roma_confidence_ptsA.npy",
    ptsA
)

np.save(
    f"{OUT}/roma_confidence_ptsB.npy",
    ptsB
)


print()
print("Saved confidence evaluation data.")
print("Done.")