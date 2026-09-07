import os
import math

import cv2
import torch
import numpy as np

from romav2 import RoMaV2


# =========================================================
# CONFIG
# =========================================================

PAIR = "data/test_pairs/pair_002"
OUT = "outputs/pair_002"

CH_PATH = f"{PAIR}/chandrayaan.png"
LRO_PATH = f"{PAIR}/lro.png"

# RoMa
SAMPLE_COUNT = 5000

# Seed stage
SEED_TOP_K = 150
SEED_RANSAC_THRESHOLD = 12.0

# Expansion stage
EXPANSION_THRESHOLD = 18.0

# Spatial balancing
GRID = 8
TOP_PER_CELL = 8

# Final RANSAC
FINAL_RANSAC_THRESHOLD = 5.0

os.makedirs(
    OUT,
    exist_ok=True
)

torch.set_float32_matmul_precision(
    "highest"
)


# =========================================================
# LOAD IMAGES
# =========================================================

ch = cv2.imread(
    CH_PATH,
    cv2.IMREAD_GRAYSCALE
)

lro = cv2.imread(
    LRO_PATH,
    cv2.IMREAD_GRAYSCALE
)


if ch is None:
    raise RuntimeError(
        f"Could not load {CH_PATH}"
    )

if lro is None:
    raise RuntimeError(
        f"Could not load {LRO_PATH}"
    )


H_A, W_A = ch.shape
H_B, W_B = lro.shape


print(
    "===== LUNARREG ROMA V5 ====="
)

print()

print(
    "Chandrayaan:",
    W_A,
    "x",
    H_A
)

print(
    "LRO        :",
    W_B,
    "x",
    H_B
)


# =========================================================
# EXPECTED SCALE
#
# Both experimental crops cover approximately the same AOI.
# This is used as a sanity metric only.
#
# GeoTIFF coordinates are NOT used for correspondence
# selection or registration.
# =========================================================

expected_scale_x = (
    W_B / W_A
)

expected_scale_y = (
    H_B / H_A
)

expected_scale = (
    expected_scale_x
    +
    expected_scale_y
) / 2.0


print()

print(
    "Expected approximate scale:",
    round(
        expected_scale,
        4
    )
)


# =========================================================
# LOAD ROMA
# =========================================================

print()
print(
    "Loading RoMa v2..."
)


model = RoMaV2()


# ---------------------------------------------------------
# Winning configuration from our ablation study
# ---------------------------------------------------------

model.apply_setting(
    "fast"
)

model.bidirectional = True

model.balanced_sampling = True


print(
    "Configuration:"
)

print(
    "  Mode          : FAST"
)

print(
    "  Low resolution:",
    model.H_lr,
    "x",
    model.W_lr
)

print(
    "  High-res pass :",
    model.H_hr
)

print(
    "  Bidirectional :",
    model.bidirectional
)

print(
    "  Balanced      :",
    model.balanced_sampling
)


# =========================================================
# DENSE MATCHING
# =========================================================

print()
print(
    "Running RoMa dense matching..."
)


with torch.inference_mode():

    predictions = model.match(
        CH_PATH,
        LRO_PATH
    )


print(
    "Dense matching complete ✅"
)


# =========================================================
# SAMPLE CORRESPONDENCES
# =========================================================

print()

print(
    "Sampling",
    SAMPLE_COUNT,
    "correspondences..."
)


matches, overlaps, precision_AB, precision_BA = (
    model.sample(
        predictions,
        SAMPLE_COUNT
    )
)


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
    .astype(np.float32)
)

ptsB = (
    kptsB
    .detach()
    .cpu()
    .numpy()
    .astype(np.float32)
)


confidence = (
    overlaps
    .detach()
    .cpu()
    .numpy()
    .reshape(-1)
    .astype(np.float32)
)


print(
    "Raw sampled matches:",
    len(ptsA)
)


# =========================================================
# REMOVE INVALID PIXEL COORDINATES
# =========================================================

valid = (

    (ptsA[:, 0] >= 0)
    &
    (ptsA[:, 0] < W_A)
    &
    (ptsA[:, 1] >= 0)
    &
    (ptsA[:, 1] < H_A)

    &

    (ptsB[:, 0] >= 0)
    &
    (ptsB[:, 0] < W_B)
    &
    (ptsB[:, 1] >= 0)
    &
    (ptsB[:, 1] < H_B)
)


ptsA = ptsA[
    valid
]

ptsB = ptsB[
    valid
]

confidence = confidence[
    valid
]


print(
    "Valid pixel matches:",
    len(ptsA)
)


# =========================================================
# SORT BY ROMA CONFIDENCE
# =========================================================

order = np.argsort(
    confidence
)[::-1]


# =========================================================
# STAGE 1
# HIGH-CONFIDENCE SEED
#
# Our experiment showed the top ~100-150 RoMa matches are
# dramatically more reliable than the full sample.
#
# Do NOT force grid coverage here.
#
# First find a geometrically coherent seed transform.
# =========================================================

seed_count = min(
    SEED_TOP_K,
    len(order)
)


seed_indices = order[
    :seed_count
]


seed_A = ptsA[
    seed_indices
]

seed_B = ptsB[
    seed_indices
]

seed_conf = confidence[
    seed_indices
]


print()
print(
    "===== STAGE 1: SEED ====="
)

print(
    "Seed candidates:",
    len(seed_A)
)

print(
    "Seed confidence:"
)

print(
    "  Minimum:",
    round(
        float(
            seed_conf.min()
        ),
        4
    )
)

print(
    "  Median :",
    round(
        float(
            np.median(
                seed_conf
            )
        ),
        4
    )
)

print(
    "  Maximum:",
    round(
        float(
            seed_conf.max()
        ),
        4
    )
)


# =========================================================
# SEED RANSAC
# =========================================================

M_seed, seed_mask = (
    cv2.estimateAffinePartial2D(

        seed_A,
        seed_B,

        method=cv2.RANSAC,

        ransacReprojThreshold=
        SEED_RANSAC_THRESHOLD,

        maxIters=100000,

        confidence=0.9999,

        refineIters=30
    )
)


if (
    M_seed is None
    or
    seed_mask is None
):

    raise RuntimeError(
        "Seed RANSAC failed."
    )


seed_mask = (
    seed_mask
    .ravel()
    .astype(bool)
)


seed_inliers = int(
    seed_mask.sum()
)


seed_ratio = (
    seed_inliers
    /
    len(seed_A)
)


print()

print(
    "Seed RANSAC inliers:",
    seed_inliers,
    "/",
    len(seed_A)
)

print(
    "Seed inlier ratio:",
    round(
        seed_ratio,
        4
    )
)


# =========================================================
# SEED TRANSFORM PARAMETERS
# =========================================================

seed_a = float(
    M_seed[0, 0]
)

seed_b = float(
    M_seed[0, 1]
)


seed_scale = math.sqrt(

    seed_a * seed_a
    +
    seed_b * seed_b
)


seed_rotation = math.degrees(

    math.atan2(

        M_seed[1, 0],
        M_seed[0, 0]
    )
)


print(
    "Seed scale:",
    round(
        seed_scale,
        4
    )
)


print(
    "Seed rotation:",
    round(
        seed_rotation,
        3
    ),
    "deg"
)


# =========================================================
# STAGE 2
# GEOMETRIC EXPANSION
#
# Apply the seed transform to ALL RoMa matches.
#
# Matches near the seed geometric consensus are accepted.
#
# This allows lower-confidence but geometrically valid
# matches to re-enter the solution.
# =========================================================

ones = np.ones(

    (
        len(ptsA),
        1
    ),

    dtype=np.float32
)


ptsA_h = np.hstack(
    [
        ptsA,
        ones
    ]
)


predicted_all = (

    M_seed
    @
    ptsA_h.T

).T


seed_residuals = np.linalg.norm(

    predicted_all
    -
    ptsB,

    axis=1
)


expansion_mask = (

    seed_residuals
    <=
    EXPANSION_THRESHOLD
)


expanded_A = ptsA[
    expansion_mask
]

expanded_B = ptsB[
    expansion_mask
]

expanded_conf = confidence[
    expansion_mask
]

expanded_residuals = seed_residuals[
    expansion_mask
]


print()
print(
    "===== STAGE 2: GEOMETRIC EXPANSION ====="
)

print(
    "Matches consistent with seed:",
    len(expanded_A),
    "/",
    len(ptsA)
)


if len(expanded_A) < 10:

    raise RuntimeError(
        "Too few geometrically consistent matches."
    )


print(
    "Residual median:",
    round(
        float(
            np.median(
                expanded_residuals
            )
        ),
        3
    ),
    "px"
)


print(
    "Residual mean:",
    round(
        float(
            np.mean(
                expanded_residuals
            )
        ),
        3
    ),
    "px"
)


# =========================================================
# COMBINED QUALITY SCORE
#
# High confidence = good
# Low geometric residual = good
# =========================================================

quality_score = (

    expanded_conf
    /
    (
        1.0
        +
        expanded_residuals
    )
)


# =========================================================
# STAGE 3
# SPATIAL BALANCING
#
# NOW we use the 8x8 grid.
#
# Important difference from V4:
#
# V4:
#   forced matches from every area BEFORE knowing geometry.
#
# V5:
#   first establishes correct geometry,
#   THEN spatially distributes the geometrically valid set.
# =========================================================

selected_indices = []

expanded_cells = set()


for row in range(
    GRID
):

    for col in range(
        GRID
    ):

        x_min = (
            col
            /
            GRID
            *
            W_A
        )

        x_max = (
            (col + 1)
            /
            GRID
            *
            W_A
        )

        y_min = (
            row
            /
            GRID
            *
            H_A
        )

        y_max = (
            (row + 1)
            /
            GRID
            *
            H_A
        )


        cell_mask = (

            (expanded_A[:, 0] >= x_min)
            &
            (expanded_A[:, 0] < x_max)
            &
            (expanded_A[:, 1] >= y_min)
            &
            (expanded_A[:, 1] < y_max)
        )


        cell_indices = np.where(
            cell_mask
        )[0]


        if len(
            cell_indices
        ) == 0:

            continue


        expanded_cells.add(
            (
                row,
                col
            )
        )


        # Highest quality score first
        sorted_cell = (

            cell_indices[
                np.argsort(
                    quality_score[
                        cell_indices
                    ]
                )[::-1]
            ]
        )


        keep = sorted_cell[
            :TOP_PER_CELL
        ]


        selected_indices.extend(
            keep.tolist()
        )


selected_indices = np.array(

    selected_indices,

    dtype=np.int32
)


if len(
    selected_indices
) < 3:

    raise RuntimeError(
        "Spatial balancing produced too few matches."
    )


balanced_A = expanded_A[
    selected_indices
]

balanced_B = expanded_B[
    selected_indices
]

balanced_conf = expanded_conf[
    selected_indices
]

balanced_residual = expanded_residuals[
    selected_indices
]


expanded_coverage = (

    len(
        expanded_cells
    )

    /
    (
        GRID
        *
        GRID
    )
)


print()
print(
    "===== STAGE 3: SPATIAL BALANCING ====="
)

print(
    "Geometrically valid cells:",
    len(
        expanded_cells
    ),
    "/",
    GRID * GRID
)

print(
    "Expanded coverage:",
    round(
        expanded_coverage
        *
        100,
        2
    ),
    "%"
)

print(
    "Balanced matches:",
    len(
        balanced_A
    )
)


# =========================================================
# STAGE 4
# FINAL RANSAC
# =========================================================

print()
print(
    "===== STAGE 4: FINAL RANSAC ====="
)


M_final, final_mask = (
    cv2.estimateAffinePartial2D(

        balanced_A,
        balanced_B,

        method=cv2.RANSAC,

        ransacReprojThreshold=
        FINAL_RANSAC_THRESHOLD,

        maxIters=100000,

        confidence=0.9999,

        refineIters=50
    )
)


if (
    M_final is None
    or
    final_mask is None
):

    raise RuntimeError(
        "Final RANSAC failed."
    )


final_mask = (

    final_mask
    .ravel()
    .astype(bool)
)


inlier_A = balanced_A[
    final_mask
]

inlier_B = balanced_B[
    final_mask
]

inlier_conf = balanced_conf[
    final_mask
]


inliers = len(
    inlier_A
)


inlier_ratio = (

    inliers
    /
    len(
        balanced_A
    )
)


# =========================================================
# FINAL TRANSFORM PARAMETERS
# =========================================================

a = float(
    M_final[0, 0]
)

b = float(
    M_final[0, 1]
)


scale = math.sqrt(

    a * a
    +
    b * b
)


rotation = math.degrees(

    math.atan2(

        M_final[1, 0],
        M_final[0, 0]
    )
)


tx = float(
    M_final[0, 2]
)

ty = float(
    M_final[1, 2]
)


scale_error = (

    abs(
        scale
        -
        expected_scale
    )

    /
    expected_scale

    *
    100
)


# =========================================================
# FINAL RMSE
# =========================================================

ones = np.ones(

    (
        len(
            balanced_A
        ),
        1
    ),

    dtype=np.float32
)


balanced_h = np.hstack(
    [
        balanced_A,
        ones
    ]
)


predicted_final = (

    M_final
    @
    balanced_h.T

).T


final_errors = (

    predicted_final[
        final_mask
    ]

    -
    balanced_B[
        final_mask
    ]
)


rmse = np.sqrt(

    np.mean(

        np.sum(

            final_errors ** 2,

            axis=1
        )
    )
)


# =========================================================
# FINAL SPATIAL COVERAGE
# =========================================================

final_cells = set()


for x, y in inlier_A:

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


    final_cells.add(
        (
            row,
            col
        )
    )


final_coverage = (

    len(
        final_cells
    )

    /
    (
        GRID
        *
        GRID
    )
)


# =========================================================
# RESULTS
# =========================================================

print()
print(
    "===== ROMA V5 RESULTS ====="
)


print(
    "Total sampled      :",
    len(
        ptsA
    )
)


print(
    "Seed candidates    :",
    len(
        seed_A
    )
)


print(
    "Seed inliers       :",
    seed_inliers
)


print(
    "Expanded matches   :",
    len(
        expanded_A
    )
)


print(
    "Balanced matches   :",
    len(
        balanced_A
    )
)


print(
    "Final inliers      :",
    inliers
)


print(
    "Final inlier ratio :",
    round(
        inlier_ratio,
        4
    )
)


print(
    "RMSE               :",
    round(
        float(
            rmse
        ),
        4
    ),
    "px"
)


print(
    "Scale              :",
    round(
        scale,
        4
    )
)


print(
    "Expected scale     :",
    round(
        expected_scale,
        4
    )
)


print(
    "Scale error        :",
    round(
        scale_error,
        2
    ),
    "%"
)


print(
    "Rotation           :",
    round(
        rotation,
        3
    ),
    "deg"
)


print(
    "Translation        :",
    round(
        tx,
        2
    ),
    round(
        ty,
        2
    )
)


print(
    "Expanded coverage  :",
    round(
        expanded_coverage
        *
        100,
        2
    ),
    "%"
)


print(
    "Final coverage     :",
    round(
        final_coverage
        *
        100,
        2
    ),
    "%"
)


print(
    "Median confidence  :",
    round(
        float(
            np.median(
                inlier_conf
            )
        ),
        4
    )
)


# =========================================================
# QUALITY GATE
# =========================================================

passed = (

    inliers >= 30

    and

    inlier_ratio >= 0.40

    and

    rmse <= 5.0

    and

    final_coverage >= 0.25

    and

    0.18 <= scale <= 0.32

    and

    abs(
        rotation
    ) <= 5.0
)


print()


if passed:

    print(
        "REGISTRATION: PASS ✅"
    )

else:

    print(
        "REGISTRATION: FAIL ❌"
    )


print(
    "==========================="
)


# =========================================================
# SAVE MATCH ARRAYS
# =========================================================

np.save(
    f"{OUT}/roma_v5_inlier_A.npy",
    inlier_A
)


np.save(
    f"{OUT}/roma_v5_inlier_B.npy",
    inlier_B
)


np.save(
    f"{OUT}/roma_v5_confidence.npy",
    inlier_conf
)


np.save(
    f"{OUT}/roma_v5_transform.npy",
    M_final
)


# =========================================================
# MATCH VISUALIZATION
# =========================================================

ch_small = cv2.resize(

    ch,

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

    lro,

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


draw_A = (
    inlier_A.copy()
)


draw_A[:, 0] *= (
    display_scale_x
)

draw_A[:, 1] *= (
    display_scale_y
)


indices = np.arange(
    len(
        inlier_A
    )
)


MAX_DRAW = 200


if len(
    indices
) > MAX_DRAW:

    chosen = np.linspace(

        0,

        len(
            indices
        ) - 1,

        MAX_DRAW
    ).astype(
        int
    )

    indices = indices[
        chosen
    ]


rng = np.random.default_rng(
    42
)


for idx in indices:

    pA = draw_A[
        idx
    ]

    pB = inlier_B[
        idx
    ]


    x1 = int(
        round(
            pA[0]
        )
    )

    y1 = int(
        round(
            pA[1]
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

    f"{OUT}/roma_v5_matches.png",

    canvas
)


# =========================================================
# REGISTER CHANDRAYAAN -> LRO
# =========================================================

registered = cv2.warpAffine(

    ch,

    M_final,

    (
        W_B,
        H_B
    ),

    flags=cv2.INTER_LINEAR
)


cv2.imwrite(

    f"{OUT}/roma_v5_registered.png",

    registered
)


# =========================================================
# OVERLAY
# =========================================================

overlay = cv2.addWeighted(

    registered,

    0.5,

    lro,

    0.5,

    0
)


cv2.imwrite(

    f"{OUT}/roma_v5_overlay.png",

    overlay
)


# =========================================================
# DIFFERENCE
# =========================================================

difference = cv2.absdiff(

    registered,

    lro
)


cv2.imwrite(

    f"{OUT}/roma_v5_difference.png",

    difference
)


# =========================================================
# GRID VISUALIZATION
# =========================================================

grid_img = ch_vis.copy()


cell_w = (
    W_B
    /
    GRID
)

cell_h = (
    H_B
    /
    GRID
)


for i in range(
    1,
    GRID
):

    x = int(
        i
        *
        cell_w
    )

    cv2.line(

        grid_img,

        (
            x,
            0
        ),

        (
            x,
            H_B
        ),

        (
            255,
            255,
            255
        ),

        1
    )


for i in range(
    1,
    GRID
):

    y = int(
        i
        *
        cell_h
    )

    cv2.line(

        grid_img,

        (
            0,
            y
        ),

        (
            W_B,
            y
        ),

        (
            255,
            255,
            255
        ),

        1
    )


for row, col in final_cells:

    cx = int(

        (
            col
            +
            0.5
        )

        *
        cell_w
    )


    cy = int(

        (
            row
            +
            0.5
        )

        *
        cell_h
    )


    cv2.circle(

        grid_img,

        (
            cx,
            cy
        ),

        6,

        (
            255,
            255,
            255
        ),

        -1
    )


cv2.imwrite(

    f"{OUT}/roma_v5_grid.png",

    grid_img
)


# =========================================================
# FINISH
# =========================================================

print()
print(
    "Saved:"
)

print(
    f"{OUT}/roma_v5_matches.png"
)

print(
    f"{OUT}/roma_v5_registered.png"
)

print(
    f"{OUT}/roma_v5_overlay.png"
)

print(
    f"{OUT}/roma_v5_difference.png"
)

print(
    f"{OUT}/roma_v5_grid.png"
)

print()
print(
    "Done."
)