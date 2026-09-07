import os
import math

import cv2
import torch
import torch.nn.functional as F
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
GRID = 8

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
    raise RuntimeError(f"Could not load {CH_PNG}")

if lro_img is None:
    raise RuntimeError(f"Could not load {LRO_PNG}")


H_A, W_A = ch_img.shape
H_B, W_B = lro_img.shape


expected_scale = (
    (W_B / W_A)
    +
    (H_B / H_A)
) / 2.0


print("===== ROMA CYCLE CONSISTENCY EVALUATION =====")
print()

print("Chandrayaan:", W_A, "x", H_A)
print("LRO        :", W_B, "x", H_B)
print("Expected scale:", round(expected_scale, 4))


# =========================================================
# LOAD ROMA
# =========================================================

print()
print("Loading RoMa v2...")

model = RoMaV2()

# Winning configuration from our ablation
model.apply_setting("fast")
model.bidirectional = True
model.balanced_sampling = True


print("Configuration:")
print("  Mode          : FAST")
print("  Low resolution:", model.H_lr, "x", model.W_lr)
print("  High-res pass :", model.H_hr)
print("  Bidirectional :", model.bidirectional)
print("  Balanced      :", model.balanced_sampling)


# =========================================================
# BIDIRECTIONAL DENSE MATCHING
#
# One inference gives us:
#
# warp_AB : Chandrayaan -> LRO
# warp_BA : LRO -> Chandrayaan
# =========================================================

print()
print("Running bidirectional dense matching...")

with torch.inference_mode():

    predictions = model.match(
        CH_PNG,
        LRO_PNG
    )


print("Dense matching complete ✅")


if predictions["warp_BA"] is None:
    raise RuntimeError(
        "warp_BA missing. Bidirectional matching did not run."
    )


# =========================================================
# SAMPLE ONLY FORWARD A -> B MATCHES
#
# We already have the backward dense field.
#
# Temporarily tell sample() to use AB only, otherwise
# RoMa will mix AB and BA correspondences together.
# =========================================================

print()
print(
    "Sampling",
    N_MATCHES,
    "forward correspondences..."
)

model.bidirectional = False

matches, forward_conf, _, _ = model.sample(
    predictions,
    N_MATCHES
)

# Restore setting
model.bidirectional = True


print(
    "Forward sampled matches:",
    len(matches)
)


# =========================================================
# PIXEL COORDINATES
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


forward_conf_np = (
    forward_conf
    .detach()
    .cpu()
    .numpy()
    .reshape(-1)
    .astype(np.float64)
)


# =========================================================
# CYCLE:
#
# A --RoMa--> B --RoMa--> A'
#
# For a consistent match:
#
# A' should be close to original A.
# =========================================================

A_norm = matches[:, 0:2]
B_norm = matches[:, 2:4]


warp_BA = predictions["warp_BA"]

# BHWC -> BCHW for grid_sample
warp_BA_chw = warp_BA.permute(
    0,
    3,
    1,
    2
)


# Query backward warp exactly at each predicted B location.
query_grid = B_norm.reshape(
    1,
    -1,
    1,
    2
)


with torch.inference_mode():

    A_back_norm = F.grid_sample(
        warp_BA_chw,
        query_grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False
    )


# Shape:
# 1 x 2 x N x 1
#
# -> N x 2
A_back_norm = (
    A_back_norm[
        0,
        :,
        :,
        0
    ]
    .T
)


# =========================================================
# SAMPLE BACKWARD CONFIDENCE AT B
# =========================================================

overlap_BA = predictions["overlap_BA"]

overlap_BA_chw = overlap_BA.permute(
    0,
    3,
    1,
    2
)


with torch.inference_mode():

    reverse_conf = F.grid_sample(
        overlap_BA_chw,
        query_grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False
    )


reverse_conf = (
    reverse_conf[
        0,
        0,
        :,
        0
    ]
)


reverse_conf_np = (
    reverse_conf
    .detach()
    .cpu()
    .numpy()
    .astype(np.float64)
)


# =========================================================
# NORMALIZED A' -> ORIGINAL CHANDRAYAAN PIXELS
# =========================================================

A_back_x = (
    (A_back_norm[:, 0] + 1.0)
    /
    2.0
    *
    W_A
)

A_back_y = (
    (A_back_norm[:, 1] + 1.0)
    /
    2.0
    *
    H_A
)


A_back_px = torch.stack(
    [
        A_back_x,
        A_back_y
    ],
    dim=1
)


A_back_px = (
    A_back_px
    .detach()
    .cpu()
    .numpy()
    .astype(np.float64)
)


# =========================================================
# CYCLE ERROR
#
# Measured first in Chandrayaan pixels.
# =========================================================

cycle_error_A = np.linalg.norm(
    A_back_px
    -
    ptsA,
    axis=1
)


# Convert to approximate LRO-pixel equivalent.
#
# Example:
# 40 Chandrayaan px * 0.25 ≈ 10 LRO px
cycle_error_equiv = (
    cycle_error_A
    *
    expected_scale
)


# =========================================================
# COMBINED CONFIDENCE
#
# Require both directions to be confident.
# =========================================================

combined_conf = np.sqrt(
    np.clip(
        forward_conf_np,
        0,
        1
    )
    *
    np.clip(
        reverse_conf_np,
        0,
        1
    )
)


# =========================================================
# CYCLE SCORE
#
# High confidence + low cycle error = high score.
#
# This score does NOT use GeoTIFF ground truth.
# =========================================================

cycle_score = (
    combined_conf
    /
    (
        1.0
        +
        cycle_error_equiv
    )
)


# =========================================================
# FILTER INVALID CYCLES
# =========================================================

valid_cycle = (
    np.isfinite(
        cycle_error_equiv
    )
    &
    np.isfinite(
        combined_conf
    )
    &
    (reverse_conf_np > 0)
)


ptsA = ptsA[valid_cycle]
ptsB = ptsB[valid_cycle]

forward_conf_np = forward_conf_np[valid_cycle]
reverse_conf_np = reverse_conf_np[valid_cycle]

combined_conf = combined_conf[valid_cycle]

cycle_error_A = cycle_error_A[valid_cycle]
cycle_error_equiv = cycle_error_equiv[valid_cycle]

cycle_score = cycle_score[valid_cycle]


print()
print(
    "Valid cycle matches:",
    len(ptsA)
)


# =========================================================
# BASIC CYCLE STATISTICS
# =========================================================

print()
print("===== CYCLE ERROR =====")

print(
    "A-pixel minimum :",
    round(
        float(
            np.min(
                cycle_error_A
            )
        ),
        3
    )
)

print(
    "A-pixel median  :",
    round(
        float(
            np.median(
                cycle_error_A
            )
        ),
        3
    )
)

print(
    "A-pixel mean    :",
    round(
        float(
            np.mean(
                cycle_error_A
            )
        ),
        3
    )
)


print()
print("LRO-equivalent cycle error:")

print(
    "Minimum :",
    round(
        float(
            np.min(
                cycle_error_equiv
            )
        ),
        3
    ),
    "px"
)

print(
    "Median  :",
    round(
        float(
            np.median(
                cycle_error_equiv
            )
        ),
        3
    ),
    "px"
)

print(
    "Mean    :",
    round(
        float(
            np.mean(
                cycle_error_equiv
            )
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
                cycle_error_equiv,
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
                cycle_error_equiv,
                90
            )
        ),
        3
    ),
    "px"
)


# =========================================================
# EXACT GEOREFERENCE GROUND TRUTH
#
# IMPORTANT:
#
# Everything above is image-only.
#
# GeoTIFF is used ONLY from here onward to evaluate whether
# cycle consistency separates true from false matches.
# =========================================================

with rasterio.open(CH_TIF) as ch_ds:
    ch_transform = ch_ds.transform

with rasterio.open(LRO_TIF) as lro_ds:
    lro_transform = lro_ds.transform


# RoMa pixel coordinates are already pixel-centre style
# floating coordinates.
cols = ptsA[:, 0]
rows = ptsA[:, 1]


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

forward_conf_np = forward_conf_np[inside]
reverse_conf_np = reverse_conf_np[inside]

combined_conf = combined_conf[inside]

cycle_error_A = cycle_error_A[inside]
cycle_error_equiv = cycle_error_equiv[inside]

cycle_score = cycle_score[inside]


geo_error = np.linalg.norm(
    ptsB
    -
    expected_B,
    axis=1
)


print()
print(
    "Georef evaluated matches:",
    len(
        geo_error
    )
)


# =========================================================
# BASELINE
# =========================================================

print()
print("===== BASELINE GEO ACCURACY =====")


for threshold in [
    2,
    5,
    10,
    20,
    40
]:

    count = int(
        np.sum(
            geo_error
            <=
            threshold
        )
    )

    print(
        f"<= {threshold:2d} px : "
        f"{count:4d} / {len(geo_error)} "
        f"({count / len(geo_error) * 100:.2f}%)"
    )


print(
    "Median geo error:",
    round(
        float(
            np.median(
                geo_error
            )
        ),
        3
    ),
    "px"
)


# =========================================================
# DOES CYCLE CONSISTENCY WORK?
# =========================================================

print()
print("===== CYCLE THRESHOLD TEST =====")


cycle_thresholds = [
    1,
    2,
    5,
    10,
    20,
    40
]


for threshold in cycle_thresholds:

    mask = (
        cycle_error_equiv
        <=
        threshold
    )

    count = int(
        mask.sum()
    )


    if count == 0:
        continue


    current = geo_error[
        mask
    ]


    correct_5 = (
        np.mean(
            current <= 5
        )
        *
        100
    )

    correct_10 = (
        np.mean(
            current <= 10
        )
        *
        100
    )

    correct_20 = (
        np.mean(
            current <= 20
        )
        *
        100
    )


    print()

    print(
        f"Cycle <= {threshold} "
        f"LRO-equivalent px"
    )

    print(
        "Matches:",
        count
    )

    print(
        "Median geo error:",
        round(
            float(
                np.median(
                    current
                )
            ),
            3
        ),
        "px"
    )

    print(
        "<= 5 px :",
        round(
            correct_5,
            2
        ),
        "%"
    )

    print(
        "<=10 px :",
        round(
            correct_10,
            2
        ),
        "%"
    )

    print(
        "<=20 px :",
        round(
            correct_20,
            2
        ),
        "%"
    )


# =========================================================
# CONFIDENCE + CYCLE
#
# Test combinations that can actually be used in V6.
# =========================================================

print()
print("===== CONFIDENCE + CYCLE =====")


confidence_thresholds = [
    0.30,
    0.40,
    0.50,
    0.60
]


cycle_limits = [
    5,
    10,
    20
]


for conf_threshold in confidence_thresholds:

    for cycle_limit in cycle_limits:

        mask = (
            (combined_conf >= conf_threshold)
            &
            (
                cycle_error_equiv
                <=
                cycle_limit
            )
        )


        count = int(
            mask.sum()
        )


        if count < 5:
            continue


        current = geo_error[
            mask
        ]


        correct10 = (
            np.mean(
                current <= 10
            )
            *
            100
        )


        correct20 = (
            np.mean(
                current <= 20
            )
            *
            100
        )


        print(
            f"conf >= {conf_threshold:.2f}, "
            f"cycle <= {cycle_limit:2d}px : "
            f"{count:4d} matches | "
            f"<=10px {correct10:6.2f}% | "
            f"<=20px {correct20:6.2f}% | "
            f"median {np.median(current):7.3f}px"
        )


# =========================================================
# TOP-K BY CYCLE SCORE
# =========================================================

print()
print("===== TOP-K CYCLE SCORE =====")


order = np.argsort(
    cycle_score
)[::-1]


for k in [
    50,
    100,
    150,
    200,
    300,
    500,
    1000
]:

    if k > len(order):
        continue


    idx = order[:k]

    current = geo_error[
        idx
    ]


    c5 = int(
        np.sum(
            current <= 5
        )
    )

    c10 = int(
        np.sum(
            current <= 10
        )
    )

    c20 = int(
        np.sum(
            current <= 20
        )
    )


    print()

    print(
        "Top",
        k
    )

    print(
        "<= 5 px :",
        c5,
        "/",
        k,
        f"({c5 / k * 100:.2f}%)"
    )

    print(
        "<=10 px :",
        c10,
        "/",
        k,
        f"({c10 / k * 100:.2f}%)"
    )

    print(
        "<=20 px :",
        c20,
        "/",
        k,
        f"({c20 / k * 100:.2f}%)"
    )

    print(
        "Median geo error:",
        round(
            float(
                np.median(
                    current
                )
            ),
            3
        ),
        "px"
    )

    print(
        "Median cycle:",
        round(
            float(
                np.median(
                    cycle_error_equiv[
                        idx
                    ]
                )
            ),
            3
        ),
        "px"
    )


# =========================================================
# TOP-300 CYCLE SCORE SPATIAL COVERAGE
# =========================================================

K = min(
    300,
    len(order)
)


top_indices = order[
    :K
]


top_A = ptsA[
    top_indices
]


occupied = set()


for x, y in top_A:

    col = min(
        max(
            int(
                x
                /
                W_A
                *
                GRID
            ),
            0
        ),
        GRID - 1
    )

    row = min(
        max(
            int(
                y
                /
                H_A
                *
                GRID
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
    len(
        occupied
    )
    /
    (GRID * GRID)
)


print()
print("===== TOP 300 CYCLE COVERAGE =====")

print(
    "Grid cells:",
    len(
        occupied
    ),
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
# VISUALIZE TOP CYCLE-SCORE MATCHES
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
    W_B
    /
    W_A
)

display_scale_y = (
    H_B
    /
    H_A
)


draw_indices = order[
    :min(
        200,
        len(order)
    )
]


rng = np.random.default_rng(
    42
)


for idx in draw_indices:

    pA = ptsA[
        idx
    ]

    pB = ptsB[
        idx
    ]


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
        (
            x1,
            y1
        ),
        2,
        colour,
        -1
    )

    cv2.circle(
        canvas,
        (
            x2,
            y2
        ),
        2,
        colour,
        -1
    )

    cv2.line(
        canvas,
        (
            x1,
            y1
        ),
        (
            x2,
            y2
        ),
        colour,
        1,
        cv2.LINE_AA
    )


cv2.imwrite(
    f"{OUT}/roma_cycle_top_matches.png",
    canvas
)


# =========================================================
# SAVE NUMERICAL DATA
# =========================================================

np.savez(
    f"{OUT}/roma_cycle_evaluation.npz",

    ptsA=ptsA,
    ptsB=ptsB,

    forward_confidence=
    forward_conf_np,

    reverse_confidence=
    reverse_conf_np,

    combined_confidence=
    combined_conf,

    cycle_error_A=
    cycle_error_A,

    cycle_error_equiv=
    cycle_error_equiv,

    cycle_score=
    cycle_score,

    geo_error=
    geo_error
)


# =========================================================
# FINISH
# =========================================================

print()
print("Saved:")

print(
    f"{OUT}/roma_cycle_top_matches.png"
)

print(
    f"{OUT}/roma_cycle_evaluation.npz"
)

print()
print("Done.")