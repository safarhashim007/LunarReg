import os
import math

import cv2
import torch
import torch.nn.functional as F
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

# Cycle-score seed
SEED_TOP_K = 75

# A slightly generous seed threshold
SEED_RANSAC_THRESHOLD = 10.0

# Expansion from seed model
EXPANSION_THRESHOLD = 15.0

# Spatial distribution
GRID = 8
TOP_PER_CELL = 8

# Final fit
FINAL_RANSAC_THRESHOLD = 4.0

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

if ch is None or lro is None:
    raise RuntimeError("Could not load image pair")


H_A, W_A = ch.shape
H_B, W_B = lro.shape


expected_scale = (
    (W_B / W_A)
    +
    (H_B / H_A)
) / 2.0


print("===== LUNARREG ROMA V6 =====")
print()

print("Chandrayaan:", W_A, "x", H_A)
print("LRO        :", W_B, "x", H_B)

print(
    "Expected approximate scale:",
    round(expected_scale, 4)
)


# =========================================================
# ROMA
# =========================================================

print()
print("Loading RoMa v2...")

model = RoMaV2()

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
# DENSE BIDIRECTIONAL MATCHING
# =========================================================

print()
print("Running bidirectional RoMa...")

with torch.inference_mode():

    predictions = model.match(
        CH_PATH,
        LRO_PATH
    )


if predictions["warp_BA"] is None:
    raise RuntimeError(
        "Backward warp not generated"
    )


print("Dense matching complete ✅")


# =========================================================
# SAMPLE FORWARD MATCHES ONLY
# =========================================================

print()
print(
    "Sampling",
    SAMPLE_COUNT,
    "forward matches..."
)


# Prevent sample() mixing BA samples into our AB set
model.bidirectional = False

matches, forward_conf, _, _ = model.sample(
    predictions,
    SAMPLE_COUNT
)

model.bidirectional = True


kptsA, kptsB = model.to_pixel_coordinates(
    matches,
    H_A,
    W_A,
    H_B,
    W_B
)


ptsA = (
    kptsA.detach()
    .cpu()
    .numpy()
    .astype(np.float32)
)

ptsB = (
    kptsB.detach()
    .cpu()
    .numpy()
    .astype(np.float32)
)


forward_conf = (
    forward_conf.detach()
    .cpu()
    .numpy()
    .reshape(-1)
    .astype(np.float32)
)


print(
    "Forward matches:",
    len(ptsA)
)


# =========================================================
# BACKWARD CYCLE QUERY
# =========================================================

A_norm = matches[:, :2]
B_norm = matches[:, 2:4]


warp_BA = predictions["warp_BA"]

warp_BA_chw = warp_BA.permute(
    0,
    3,
    1,
    2
)


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


A_back_norm = (
    A_back_norm[
        0,
        :,
        :,
        0
    ].T
)


# =========================================================
# BACKWARD CONFIDENCE
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


reverse_conf = (
    reverse_conf.detach()
    .cpu()
    .numpy()
    .astype(np.float32)
)


# =========================================================
# A' NORMALIZED -> CHANDRAYAAN PIXELS
# =========================================================

back_x = (
    (A_back_norm[:, 0] + 1)
    / 2
    * W_A
)

back_y = (
    (A_back_norm[:, 1] + 1)
    / 2
    * H_A
)


back_A = torch.stack(
    [back_x, back_y],
    dim=1
)


back_A = (
    back_A.detach()
    .cpu()
    .numpy()
    .astype(np.float32)
)


# =========================================================
# CYCLE ERROR
# =========================================================

cycle_error_A = np.linalg.norm(
    back_A - ptsA,
    axis=1
)


# Express error in approximately LRO pixels
cycle_error = (
    cycle_error_A
    *
    expected_scale
)


combined_conf = np.sqrt(
    np.clip(
        forward_conf,
        0,
        1
    )
    *
    np.clip(
        reverse_conf,
        0,
        1
    )
)


# =========================================================
# CYCLE QUALITY SCORE
# =========================================================

cycle_score = (
    combined_conf
    /
    (
        1.0
        +
        cycle_error
    )
)


# =========================================================
# BASIC VALIDITY FILTER
# =========================================================

valid = (
    np.isfinite(cycle_error)
    &
    np.isfinite(cycle_score)
    &
    (reverse_conf > 0)
    &
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

combined_conf = combined_conf[valid]
cycle_error = cycle_error[valid]
cycle_score = cycle_score[valid]


print()
print(
    "Valid cycle matches:",
    len(ptsA)
)


# =========================================================
# STAGE 1 — CYCLE-SCORE SEED
# =========================================================

order = np.argsort(
    cycle_score
)[::-1]


seed_count = min(
    SEED_TOP_K,
    len(order)
)


seed_idx = order[
    :seed_count
]

seed_A = ptsA[
    seed_idx
]

seed_B = ptsB[
    seed_idx
]


print()
print("===== STAGE 1: CYCLE SEED =====")

print(
    "Seed matches:",
    len(seed_A)
)

print(
    "Median cycle error:",
    round(
        float(
            np.median(
                cycle_error[
                    seed_idx
                ]
            )
        ),
        3
    ),
    "px"
)

print(
    "Median combined confidence:",
    round(
        float(
            np.median(
                combined_conf[
                    seed_idx
                ]
            )
        ),
        4
    )
)


# =========================================================
# SEED RANSAC
# =========================================================

M_seed, seed_mask = cv2.estimateAffinePartial2D(
    seed_A,
    seed_B,

    method=cv2.RANSAC,

    ransacReprojThreshold=
        SEED_RANSAC_THRESHOLD,

    maxIters=100000,

    confidence=0.9999,

    refineIters=50
)


if M_seed is None or seed_mask is None:
    raise RuntimeError(
        "Seed RANSAC failed"
    )


seed_mask = (
    seed_mask
    .ravel()
    .astype(bool)
)


seed_inliers = int(
    seed_mask.sum()
)


a = float(
    M_seed[0, 0]
)

b = float(
    M_seed[0, 1]
)


seed_scale = math.sqrt(
    a * a
    +
    b * b
)


seed_rotation = math.degrees(
    math.atan2(
        M_seed[1, 0],
        M_seed[0, 0]
    )
)


print()

print(
    "Seed inliers:",
    seed_inliers,
    "/",
    len(seed_A)
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
# STAGE 2 — GEOMETRIC EXPANSION
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


pred_all = (
    M_seed
    @
    ptsA_h.T
).T


residual = np.linalg.norm(
    pred_all
    -
    ptsB,
    axis=1
)


expand_mask = (
    residual
    <=
    EXPANSION_THRESHOLD
)


expanded_A = ptsA[
    expand_mask
]

expanded_B = ptsB[
    expand_mask
]

expanded_cycle = cycle_error[
    expand_mask
]

expanded_conf = combined_conf[
    expand_mask
]

expanded_residual = residual[
    expand_mask
]


print()
print(
    "===== STAGE 2: EXPANSION ====="
)

print(
    "Expanded matches:",
    len(expanded_A)
)

print(
    "Median residual:",
    round(
        float(
            np.median(
                expanded_residual
            )
        ),
        3
    ),
    "px"
)


if len(expanded_A) < 10:
    raise RuntimeError(
        "Too few expanded matches"
    )


# =========================================================
# QUALITY SCORE FOR SPATIAL SELECTION
# =========================================================

quality = (
    expanded_conf
    /
    (
        1
        +
        expanded_cycle
        +
        expanded_residual
    )
)


# =========================================================
# STAGE 3 — SPATIAL BALANCING
# =========================================================

selected = []

valid_cells = set()


for row in range(GRID):

    for col in range(GRID):

        xmin = (
            col
            /
            GRID
            *
            W_A
        )

        xmax = (
            (col + 1)
            /
            GRID
            *
            W_A
        )

        ymin = (
            row
            /
            GRID
            *
            H_A
        )

        ymax = (
            (row + 1)
            /
            GRID
            *
            H_A
        )


        cell_mask = (
            (expanded_A[:, 0] >= xmin)
            &
            (expanded_A[:, 0] < xmax)
            &
            (expanded_A[:, 1] >= ymin)
            &
            (expanded_A[:, 1] < ymax)
        )


        idx = np.where(
            cell_mask
        )[0]


        if len(idx) == 0:
            continue


        valid_cells.add(
            (row, col)
        )


        idx = idx[
            np.argsort(
                quality[idx]
            )[::-1]
        ]


        selected.extend(
            idx[
                :TOP_PER_CELL
            ].tolist()
        )


selected = np.asarray(
    selected,
    dtype=np.int32
)


balanced_A = expanded_A[
    selected
]

balanced_B = expanded_B[
    selected
]


print()
print(
    "===== STAGE 3: SPATIAL BALANCING ====="
)

print(
    "Valid cells:",
    len(valid_cells),
    "/",
    GRID * GRID
)

print(
    "Coverage:",
    round(
        len(valid_cells)
        /
        (GRID * GRID)
        *
        100,
        2
    ),
    "%"
)

print(
    "Balanced matches:",
    len(balanced_A)
)


# =========================================================
# STAGE 4 — FINAL RANSAC
# =========================================================

M_final, final_mask = cv2.estimateAffinePartial2D(
    balanced_A,
    balanced_B,

    method=cv2.RANSAC,

    ransacReprojThreshold=
        FINAL_RANSAC_THRESHOLD,

    maxIters=100000,

    confidence=0.9999,

    refineIters=50
)


if M_final is None or final_mask is None:
    raise RuntimeError(
        "Final RANSAC failed"
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


inliers = len(
    inlier_A
)


ratio = (
    inliers
    /
    len(balanced_A)
)


# =========================================================
# FINAL PARAMETERS
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


# =========================================================
# RMSE
# =========================================================

ones = np.ones(
    (
        len(balanced_A),
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


pred = (
    M_final
    @
    balanced_h.T
).T


error_vectors = (
    pred[
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
            error_vectors ** 2,
            axis=1
        )
    )
)


# =========================================================
# FINAL COVERAGE
# =========================================================

final_cells = set()


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

    final_cells.add(
        (row, col)
    )


final_coverage = (
    len(final_cells)
    /
    (GRID * GRID)
)


# =========================================================
# RESULTS
# =========================================================

print()
print("===== ROMA V6 RESULTS =====")

print(
    "Sampled        :",
    SAMPLE_COUNT
)

print(
    "Seed matches   :",
    len(seed_A)
)

print(
    "Seed inliers   :",
    seed_inliers
)

print(
    "Expanded       :",
    len(expanded_A)
)

print(
    "Balanced       :",
    len(balanced_A)
)

print(
    "Final inliers  :",
    inliers
)

print(
    "Inlier ratio   :",
    round(
        ratio,
        4
    )
)

print(
    "RMSE           :",
    round(
        float(rmse),
        4
    ),
    "px"
)

print(
    "Scale          :",
    round(
        scale,
        4
    )
)

print(
    "Expected scale :",
    round(
        expected_scale,
        4
    )
)

print(
    "Rotation       :",
    round(
        rotation,
        3
    ),
    "deg"
)

print(
    "Translation    :",
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
    "Final coverage :",
    round(
        final_coverage
        *
        100,
        2
    ),
    "%"
)


# =========================================================
# QUALITY GATE
# =========================================================

passed = (
    inliers >= 25
    and
    ratio >= 0.35
    and
    rmse <= 5.0
    and
    final_coverage >= 0.25
    and
    0.18 <= scale <= 0.32
    and
    abs(rotation) <= 5.0
)


print()

print(
    "REGISTRATION:",
    "PASS ✅"
    if passed
    else
    "FAIL ❌"
)


print(
    "=========================="
)


# =========================================================
# VISUALIZATION
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


sx = W_B / W_A
sy = H_B / H_A


draw_A = inlier_A.copy()

draw_A[:, 0] *= sx
draw_A[:, 1] *= sy


rng = np.random.default_rng(
    42
)


for i in range(
    min(
        len(inlier_A),
        200
    )
):

    ax = int(
        round(
            draw_A[i, 0]
        )
    )

    ay = int(
        round(
            draw_A[i, 1]
        )
    )

    bx = (
        int(
            round(
                inlier_B[i, 0]
            )
        )
        +
        W_B
    )

    by = int(
        round(
            inlier_B[i, 1]
        )
    )


    colour = tuple(
        int(v)
        for v in rng.integers(
            70,
            256,
            size=3
        )
    )


    cv2.circle(
        canvas,
        (ax, ay),
        2,
        colour,
        -1
    )

    cv2.circle(
        canvas,
        (bx, by),
        2,
        colour,
        -1
    )

    cv2.line(
        canvas,
        (ax, ay),
        (bx, by),
        colour,
        1,
        cv2.LINE_AA
    )


cv2.imwrite(
    f"{OUT}/roma_v6_matches.png",
    canvas
)


# =========================================================
# REGISTER IMAGE
# =========================================================

registered = cv2.warpAffine(
    ch,
    M_final,
    (
        W_B,
        H_B
    )
)


cv2.imwrite(
    f"{OUT}/roma_v6_registered.png",
    registered
)


overlay = cv2.addWeighted(
    registered,
    0.5,
    lro,
    0.5,
    0
)


cv2.imwrite(
    f"{OUT}/roma_v6_overlay.png",
    overlay
)


difference = cv2.absdiff(
    registered,
    lro
)


cv2.imwrite(
    f"{OUT}/roma_v6_difference.png",
    difference
)


# =========================================================
# SAVE NUMERICAL RESULTS
# =========================================================

np.save(
    f"{OUT}/roma_v6_transform.npy",
    M_final
)

np.save(
    f"{OUT}/roma_v6_inlier_A.npy",
    inlier_A
)

np.save(
    f"{OUT}/roma_v6_inlier_B.npy",
    inlier_B
)


print()
print("Saved:")

print(
    f"{OUT}/roma_v6_matches.png"
)

print(
    f"{OUT}/roma_v6_registered.png"
)

print(
    f"{OUT}/roma_v6_overlay.png"
)

print(
    f"{OUT}/roma_v6_difference.png"
)

print()
print("Done.")