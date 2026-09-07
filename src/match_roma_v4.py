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

SAMPLE_COUNT = 5000

# Spatial balancing
GRID = 8
TOP_PER_CELL = 8

# RANSAC
RANSAC_THRESHOLD = 5.0

os.makedirs(OUT, exist_ok=True)

torch.set_float32_matmul_precision("highest")


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


print("===== LUNARREG ROMA V4 =====")

print()
print("Image sizes:")
print("Chandrayaan:", W_A, "x", H_A)
print("LRO        :", W_B, "x", H_B)


# =========================================================
# EXPECTED SCALE
#
# Both crops cover approximately the same geographic area.
# This is used only as a sanity check after registration.
# =========================================================

expected_scale_x = W_B / W_A
expected_scale_y = H_B / H_A

expected_scale = (
    expected_scale_x
    +
    expected_scale_y
) / 2.0


print()
print(
    "Expected scale:",
    round(
        expected_scale,
        4
    )
)


# =========================================================
# LOAD ROMA V2
# =========================================================

print()
print("Loading RoMa v2...")

model = RoMaV2()

model.apply_setting("fast")

# RoMa's own diverse sampler stays enabled.
model.balanced_sampling = True

print("Model ready.")


# =========================================================
# DENSE MATCHING
# =========================================================

print()
print("Running dense matching...")

with torch.inference_mode():

    predictions = model.match(
        CH_PATH,
        LRO_PATH
    )


print("Dense matching complete.")


# =========================================================
# SAMPLE CORRESPONDENCES
# =========================================================

print()
print(
    "Sampling",
    SAMPLE_COUNT,
    "matches..."
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
    "Sampled:",
    len(ptsA)
)

print(
    "Confidence range:",
    round(
        float(confidence.min()),
        4
    ),
    "→",
    round(
        float(confidence.max()),
        4
    )
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


ptsA = ptsA[valid]
ptsB = ptsB[valid]
confidence = confidence[valid]


print(
    "Valid pixel matches:",
    len(ptsA)
)


# =========================================================
# SPATIALLY BALANCED CONFIDENCE SELECTION
#
# Divide Chandrayaan into an 8x8 grid.
#
# Within EACH cell:
#     sort RoMa correspondences by confidence
#     keep the strongest N
#
# This prevents all matches clustering in one easy region.
# =========================================================

selected_indices = []

cell_counts = np.zeros(
    (
        GRID,
        GRID
    ),
    dtype=np.int32
)


for row in range(GRID):

    for col in range(GRID):

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
            (ptsA[:, 0] >= x_min)
            &
            (ptsA[:, 0] < x_max)
            &
            (ptsA[:, 1] >= y_min)
            &
            (ptsA[:, 1] < y_max)
        )


        indices = np.where(
            cell_mask
        )[0]


        if len(indices) == 0:
            continue


        # Highest confidence first
        order = indices[
            np.argsort(
                confidence[indices]
            )[::-1]
        ]


        keep = order[
            :TOP_PER_CELL
        ]


        selected_indices.extend(
            keep.tolist()
        )


        cell_counts[
            row,
            col
        ] = len(keep)


selected_indices = np.array(
    selected_indices,
    dtype=np.int32
)


if len(selected_indices) < 3:

    raise RuntimeError(
        "Too few spatially balanced matches."
    )


selA = ptsA[
    selected_indices
]

selB = ptsB[
    selected_indices
]

sel_conf = confidence[
    selected_indices
]


selected_cells = int(
    np.sum(
        cell_counts > 0
    )
)


selected_coverage = (
    selected_cells
    /
    (GRID * GRID)
)


print()
print("===== SPATIAL SELECTION =====")

print(
    "Selected matches:",
    len(selA)
)

print(
    "Occupied cells:",
    selected_cells,
    "/",
    GRID * GRID
)

print(
    "Selection coverage:",
    round(
        selected_coverage
        *
        100,
        2
    ),
    "%"
)

print(
    "Selected confidence:"
)

print(
    "  Min   :",
    round(
        float(sel_conf.min()),
        4
    )
)

print(
    "  Median:",
    round(
        float(np.median(sel_conf)),
        4
    )
)

print(
    "  Mean  :",
    round(
        float(sel_conf.mean()),
        4
    )
)

print(
    "  Max   :",
    round(
        float(sel_conf.max()),
        4
    )
)


# =========================================================
# FIRST RANSAC
#
# Partial affine:
#     scale
#     rotation
#     translation
#
# No arbitrary perspective deformation.
# =========================================================

print()
print("Running first RANSAC...")

M1, mask1 = cv2.estimateAffinePartial2D(
    selA,
    selB,
    method=cv2.RANSAC,
    ransacReprojThreshold=RANSAC_THRESHOLD,
    maxIters=50000,
    confidence=0.999,
    refineIters=20
)


if M1 is None or mask1 is None:

    raise RuntimeError(
        "First RANSAC failed."
    )


mask1 = (
    mask1
    .ravel()
    .astype(bool)
)


first_inliers = int(
    mask1.sum()
)


print(
    "First RANSAC inliers:",
    first_inliers,
    "/",
    len(selA)
)


if first_inliers < 3:

    raise RuntimeError(
        "Not enough RANSAC inliers."
    )


# =========================================================
# SECOND-STAGE RESIDUAL FILTER
#
# Use the initial transform to calculate residuals.
# Keep points reasonably close to the consensus.
# =========================================================

ones = np.ones(
    (
        len(selA),
        1
    ),
    dtype=np.float32
)


selA_h = np.hstack(
    [
        selA,
        ones
    ]
)


predicted1 = (
    M1
    @
    selA_h.T
).T


residuals = np.linalg.norm(
    predicted1
    -
    selB,
    axis=1
)


SECOND_STAGE_THRESHOLD = 8.0


second_mask = (
    residuals
    <=
    SECOND_STAGE_THRESHOLD
)


stage2_A = selA[
    second_mask
]

stage2_B = selB[
    second_mask
]

stage2_conf = sel_conf[
    second_mask
]


print()
print(
    "Matches after residual filter:",
    len(stage2_A)
)


if len(stage2_A) < 3:

    raise RuntimeError(
        "Too few matches after residual filtering."
    )


# =========================================================
# FINAL RANSAC
# =========================================================

print(
    "Running final RANSAC..."
)


M, mask = cv2.estimateAffinePartial2D(
    stage2_A,
    stage2_B,
    method=cv2.RANSAC,
    ransacReprojThreshold=4.0,
    maxIters=50000,
    confidence=0.999,
    refineIters=30
)


if M is None or mask is None:

    raise RuntimeError(
        "Final RANSAC failed."
    )


mask = (
    mask
    .ravel()
    .astype(bool)
)


inlier_A = stage2_A[
    mask
]

inlier_B = stage2_B[
    mask
]

inlier_conf = stage2_conf[
    mask
]


inliers = len(
    inlier_A
)


inlier_ratio = (
    inliers
    /
    len(stage2_A)
)


# =========================================================
# TRANSFORM PARAMETERS
# =========================================================

a = float(
    M[0, 0]
)

b = float(
    M[0, 1]
)


scale = math.sqrt(
    a * a
    +
    b * b
)


rotation = math.degrees(
    math.atan2(
        M[1, 0],
        M[0, 0]
    )
)


tx = float(
    M[0, 2]
)

ty = float(
    M[1, 2]
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
# RMSE
# =========================================================

ones = np.ones(
    (
        len(stage2_A),
        1
    ),
    dtype=np.float32
)


stage2_A_h = np.hstack(
    [
        stage2_A,
        ones
    ]
)


predicted = (
    M
    @
    stage2_A_h.T
).T


errors = (
    predicted[mask]
    -
    stage2_B[mask]
)


rmse = np.sqrt(
    np.mean(
        np.sum(
            errors ** 2,
            axis=1
        )
    )
)


# =========================================================
# FINAL INLIER GRID COVERAGE
# =========================================================

occupied = set()


for x, y in inlier_A:

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


inlier_coverage = (
    len(occupied)
    /
    (GRID * GRID)
)


# =========================================================
# RESULTS
# =========================================================

print()
print("===== ROMA V4 RESULTS =====")

print(
    "Sampled          :",
    SAMPLE_COUNT
)

print(
    "Balanced selected:",
    len(selA)
)

print(
    "Stage-2 matches  :",
    len(stage2_A)
)

print(
    "Final inliers    :",
    inliers
)

print(
    "Inlier ratio     :",
    round(
        inlier_ratio,
        4
    )
)

print(
    "RMSE             :",
    round(
        float(rmse),
        4
    ),
    "px"
)

print(
    "Scale            :",
    round(
        scale,
        4
    )
)

print(
    "Expected scale   :",
    round(
        expected_scale,
        4
    )
)

print(
    "Scale error      :",
    round(
        scale_error,
        2
    ),
    "%"
)

print(
    "Rotation         :",
    round(
        rotation,
        3
    ),
    "deg"
)

print(
    "Translation      :",
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
    "Selection coverage:",
    round(
        selected_coverage
        *
        100,
        2
    ),
    "%"
)

print(
    "Inlier coverage  :",
    round(
        inlier_coverage
        *
        100,
        2
    ),
    "%"
)

print(
    "Median inlier confidence:",
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
    inlier_ratio >= 0.25
    and
    rmse <= 5.0
    and
    inlier_coverage >= 0.30
    and
    abs(
        scale
        -
        expected_scale
    ) <= 0.05
    and
    abs(rotation) <= 5.0
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


print("===========================")


# =========================================================
# VISUALIZE FINAL INLIERS
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
    len(inlier_A)
)


MAX_DRAW = 200


if len(indices) > MAX_DRAW:

    selected = np.linspace(
        0,
        len(indices) - 1,
        MAX_DRAW
    ).astype(int)

    indices = indices[
        selected
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
    f"{OUT}/roma_v4_matches.png",
    canvas
)


# =========================================================
# REGISTER IMAGE
# =========================================================

registered = cv2.warpAffine(
    ch,
    M,
    (
        W_B,
        H_B
    ),
    flags=cv2.INTER_LINEAR
)


cv2.imwrite(
    f"{OUT}/roma_v4_registered.png",
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
    f"{OUT}/roma_v4_overlay.png",
    overlay
)


# =========================================================
# DIFFERENCE IMAGE
# =========================================================

difference = cv2.absdiff(
    registered,
    lro
)


cv2.imwrite(
    f"{OUT}/roma_v4_difference.png",
    difference
)


# =========================================================
# GRID COVERAGE IMAGE
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


for row, col in occupied:

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
    f"{OUT}/roma_v4_grid.png",
    grid_img
)


# =========================================================
# SAVE NUMERICAL DATA
# =========================================================

np.save(
    f"{OUT}/roma_v4_inlier_A.npy",
    inlier_A
)

np.save(
    f"{OUT}/roma_v4_inlier_B.npy",
    inlier_B
)

np.save(
    f"{OUT}/roma_v4_confidence.npy",
    inlier_conf
)

np.save(
    f"{OUT}/roma_v4_transform.npy",
    M
)


# =========================================================
# FINISH
# =========================================================

print()
print("Saved:")

print(
    f"{OUT}/roma_v4_matches.png"
)

print(
    f"{OUT}/roma_v4_registered.png"
)

print(
    f"{OUT}/roma_v4_overlay.png"
)

print(
    f"{OUT}/roma_v4_difference.png"
)

print(
    f"{OUT}/roma_v4_grid.png"
)

print()
print("Done.")