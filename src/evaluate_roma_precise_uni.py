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

MOON_RADIUS = 1737400.0
STANDARD_PARALLEL = math.radians(10.0)

GRID = 8

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
    raise RuntimeError(f"Could not load {CH_PNG}")

if lro_img is None:
    raise RuntimeError(f"Could not load {LRO_PNG}")


H_A, W_A = ch_img.shape
H_B, W_B = lro_img.shape


print("===== ROMA PRECISE UNIDIRECTIONAL EVALUATION =====")
print()
print("Image sizes:")
print("Chandrayaan:", W_A, "x", H_A)
print("LRO        :", W_B, "x", H_B)


# =========================================================
# LOAD ROMA
# =========================================================

print()
print("Loading RoMa v2...")

model = RoMaV2()

# Precise spatial resolutions
model.apply_setting("precise")

# Important ablation:
# precise resolution + refinement,
# but disable B -> A matching
model.bidirectional = False

model.balanced_sampling = True


print("Model ready.")
print()
print("Experiment      : PRECISE + UNIDIRECTIONAL")
print("Low resolution  :", model.H_lr, "x", model.W_lr)
print("High resolution :", model.H_hr, "x", model.W_hr)
print("Bidirectional   :", model.bidirectional)
print("Balanced sample :", model.balanced_sampling)


# =========================================================
# DENSE MATCHING
# =========================================================

print()
print("Running PRECISE UNIDIRECTIONAL dense matching...")

with torch.inference_mode():
    predictions = model.match(
        CH_PNG,
        LRO_PNG
    )

print()
print("Dense matching completed ✅")


# =========================================================
# SAMPLE MATCHES
# =========================================================

print()
print(
    "Sampling",
    N_MATCHES,
    "correspondences..."
)

matches, overlaps, precision_AB, precision_BA = model.sample(
    predictions,
    N_MATCHES
)


print(
    "Raw sampled matches:",
    len(matches)
)


# =========================================================
# NORMALIZED -> PIXEL COORDINATES
# =========================================================

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


print(
    "Pixel correspondences:",
    len(ptsA)
)


print()
print("===== CONFIDENCE =====")

print(
    "Minimum:",
    round(float(np.min(confidence)), 5)
)

print(
    "Median :",
    round(float(np.median(confidence)), 5)
)

print(
    "Mean   :",
    round(float(np.mean(confidence)), 5)
)

print(
    "Maximum:",
    round(float(np.max(confidence)), 5)
)


# =========================================================
# LOAD GEOREFERENCE
# ONLY FOR EVALUATION
# =========================================================

with rasterio.open(CH_TIF) as ch_ds:
    ch_transform = ch_ds.transform

with rasterio.open(LRO_TIF) as lro_ds:
    lro_transform = lro_ds.transform


# =========================================================
# CHANDRAYAAN PIXEL -> LON/LAT
# =========================================================

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


# =========================================================
# LON/LAT -> LRO PROJECTED COORDINATES
# =========================================================

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


# =========================================================
# PROJECTED COORDINATES -> EXPECTED LRO PIXELS
# =========================================================

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


# =========================================================
# KEEP VALID EXPECTED POINTS
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


ptsA = ptsA[inside]
ptsB = ptsB[inside]
expected_B = expected_B[inside]
confidence = confidence[inside]


print()
print(
    "Valid evaluated matches:",
    len(ptsA)
)


if len(ptsA) == 0:
    raise RuntimeError(
        "No valid geospatial reference points."
    )


# =========================================================
# EXACT GEOREFERENCE ERROR
# =========================================================

errors = np.linalg.norm(
    ptsB
    -
    expected_B,
    axis=1
)


print()
print("===== PRECISE UNI GEO ERROR =====")

print(
    "Minimum :",
    round(float(np.min(errors)), 3),
    "px"
)

print(
    "Median  :",
    round(float(np.median(errors)), 3),
    "px"
)

print(
    "Mean    :",
    round(float(np.mean(errors)), 3),
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
    round(float(np.max(errors)), 3),
    "px"
)


# =========================================================
# ALL MATCH ACCURACY
# =========================================================

print()
print("===== ALL MATCH ACCURACY =====")


for threshold in [2, 5, 10, 20, 40]:

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
        f"{count:4d} / {len(errors)} "
        f"({percentage:.2f}%)"
    )


# =========================================================
# CONFIDENCE QUANTILES
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


for percentile in quantiles:

    threshold = np.percentile(
        confidence,
        percentile
    )

    current_mask = (
        confidence
        >=
        threshold
    )

    current_errors = errors[
        current_mask
    ]


    if len(current_errors) == 0:
        continue


    within_2 = (
        np.mean(
            current_errors <= 2
        )
        *
        100
    )

    within_5 = (
        np.mean(
            current_errors <= 5
        )
        *
        100
    )

    within_10 = (
        np.mean(
            current_errors <= 10
        )
        *
        100
    )

    within_20 = (
        np.mean(
            current_errors <= 20
        )
        *
        100
    )


    print()

    print(
        f"Top {100 - percentile}% "
        f"(confidence >= {threshold:.4f})"
    )

    print(
        "Matches     :",
        len(current_errors)
    )

    print(
        "Median error:",
        round(
            float(
                np.median(
                    current_errors
                )
            ),
            3
        ),
        "px"
    )

    print(
        "<= 2 px :",
        round(within_2, 2),
        "%"
    )

    print(
        "<= 5 px :",
        round(within_5, 2),
        "%"
    )

    print(
        "<=10 px :",
        round(within_10, 2),
        "%"
    )

    print(
        "<=20 px :",
        round(within_20, 2),
        "%"
    )


# =========================================================
# TOP-K CONFIDENCE
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

    e = errors[
        indices
    ]

    correct_2 = int(
        np.sum(
            e <= 2
        )
    )

    correct_5 = int(
        np.sum(
            e <= 5
        )
    )

    correct_10 = int(
        np.sum(
            e <= 10
        )
    )

    correct_20 = int(
        np.sum(
            e <= 20
        )
    )


    print()
    print("Top", k)

    print(
        "<= 2 px :",
        correct_2,
        "/",
        k,
        f"({correct_2 / k * 100:.2f}%)"
    )

    print(
        "<= 5 px :",
        correct_5,
        "/",
        k,
        f"({correct_5 / k * 100:.2f}%)"
    )

    print(
        "<=10 px :",
        correct_10,
        "/",
        k,
        f"({correct_10 / k * 100:.2f}%)"
    )

    print(
        "<=20 px :",
        correct_20,
        "/",
        k,
        f"({correct_20 / k * 100:.2f}%)"
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
# TOP-300 COVERAGE
# =========================================================

K = min(
    300,
    len(order)
)

top_indices = order[:K]

top_A = ptsA[
    top_indices
]


occupied = set()


for x, y in top_A:

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


coverage = (
    len(occupied)
    /
    (GRID * GRID)
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
        coverage
        *
        100,
        2
    ),
    "%"
)


# =========================================================
# CORRECT MATCH COVERAGE
# Evaluation only
# =========================================================

strong_mask = (
    errors <= 10
)

strong_A = ptsA[
    strong_mask
]


strong_occupied = set()


for x, y in strong_A:

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

    strong_occupied.add(
        (
            row,
            col
        )
    )


strong_coverage = (
    len(strong_occupied)
    /
    (GRID * GRID)
)


print()
print("===== CORRECT MATCH COVERAGE =====")

print(
    "Matches <=10px:",
    int(
        strong_mask.sum()
    )
)

print(
    "Occupied cells:",
    len(strong_occupied),
    "/",
    GRID * GRID
)

print(
    "Coverage:",
    round(
        strong_coverage
        *
        100,
        2
    ),
    "%"
)


# =========================================================
# VISUALIZE TOP CONFIDENCE MATCHES
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
    [
        ch_vis,
        lro_vis
    ]
)


display_scale_x = (
    W_B / W_A
)

display_scale_y = (
    H_B / H_A
)


visual_indices = order[
    :min(
        200,
        len(order)
    )
]


rng = np.random.default_rng(
    42
)


for idx in visual_indices:

    pA = ptsA[idx]
    pB = ptsB[idx]


    x1 = int(
        round(
            pA[0]
            *
            display_scale_x
        )
    )

    y1 = int(
        round(
            pA[1]
            *
            display_scale_y
        )
    )

    x2 = (
        int(
            round(
                pB[0]
            )
        )
        +
        W_B
    )

    y2 = int(
        round(
            pB[1]
        )
    )


    values = rng.integers(
        70,
        256,
        size=3
    )

    colour = tuple(
        int(v)
        for v in values
    )


    cv2.circle(
        canvas,
        (x1, y1),
        2,
        colour,
        -1
    )

    cv2.circle(
        canvas,
        (x2, y2),
        2,
        colour,
        -1
    )

    cv2.line(
        canvas,
        (x1, y1),
        (x2, y2),
        colour,
        1,
        cv2.LINE_AA
    )


cv2.imwrite(
    f"{OUT}/roma_precise_uni_top_matches.png",
    canvas
)


# =========================================================
# VISUALIZE ONLY CORRECT <=10PX MATCHES
# =========================================================

correct_canvas = np.hstack(
    [
        ch_vis.copy(),
        lro_vis.copy()
    ]
)


correct_indices = np.where(
    strong_mask
)[0]


if len(correct_indices) > 200:

    chosen = np.linspace(
        0,
        len(correct_indices) - 1,
        200
    ).astype(int)

    correct_indices = correct_indices[
        chosen
    ]


rng = np.random.default_rng(
    123
)


for idx in correct_indices:

    pA = ptsA[idx]
    pB = ptsB[idx]


    x1 = int(
        round(
            pA[0]
            *
            display_scale_x
        )
    )

    y1 = int(
        round(
            pA[1]
            *
            display_scale_y
        )
    )

    x2 = (
        int(
            round(
                pB[0]
            )
        )
        +
        W_B
    )

    y2 = int(
        round(
            pB[1]
        )
    )


    values = rng.integers(
        70,
        256,
        size=3
    )

    colour = tuple(
        int(v)
        for v in values
    )


    cv2.circle(
        correct_canvas,
        (x1, y1),
        2,
        colour,
        -1
    )

    cv2.circle(
        correct_canvas,
        (x2, y2),
        2,
        colour,
        -1
    )

    cv2.line(
        correct_canvas,
        (x1, y1),
        (x2, y2),
        colour,
        1,
        cv2.LINE_AA
    )


cv2.imwrite(
    f"{OUT}/roma_precise_uni_correct_matches.png",
    correct_canvas
)


# =========================================================
# SAVE DATA
# =========================================================

np.save(
    f"{OUT}/roma_precise_uni_ptsA.npy",
    ptsA
)

np.save(
    f"{OUT}/roma_precise_uni_ptsB.npy",
    ptsB
)

np.save(
    f"{OUT}/roma_precise_uni_expected_B.npy",
    expected_B
)

np.save(
    f"{OUT}/roma_precise_uni_errors.npy",
    errors
)

np.save(
    f"{OUT}/roma_precise_uni_confidence.npy",
    confidence
)


# =========================================================
# FINISH
# =========================================================

print()
print("Saved:")

print(
    f"{OUT}/roma_precise_uni_top_matches.png"
)

print(
    f"{OUT}/roma_precise_uni_correct_matches.png"
)

print(
    f"{OUT}/roma_precise_uni_errors.npy"
)

print(
    f"{OUT}/roma_precise_uni_confidence.npy"
)

print()
print("Done.")