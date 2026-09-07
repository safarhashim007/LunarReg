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

A_PATH = f"{PAIR}/chandrayaan.png"
B_PATH = f"{PAIR}/lro.png"

SAMPLE_COUNT = 2000

# Geographic consistency threshold in LRO pixels.
# 40 px at ~20 m/px ≈ 800 m.
GEO_THRESHOLD = 40.0

GRID_SIZE = 8

os.makedirs(OUT, exist_ok=True)

torch.set_float32_matmul_precision("highest")


# =========================================================
# LOAD IMAGES
# =========================================================

imgA = cv2.imread(
    A_PATH,
    cv2.IMREAD_GRAYSCALE
)

imgB = cv2.imread(
    B_PATH,
    cv2.IMREAD_GRAYSCALE
)

if imgA is None:
    raise RuntimeError(
        f"Could not load Chandrayaan image: {A_PATH}"
    )

if imgB is None:
    raise RuntimeError(
        f"Could not load LRO image: {B_PATH}"
    )


H_A, W_A = imgA.shape
H_B, W_B = imgB.shape


print("===== LUNARREG ROMA V3 =====")
print()
print("Image sizes:")
print("Chandrayaan:", W_A, "x", H_A)
print("LRO        :", W_B, "x", H_B)


# =========================================================
# LOAD ROMA V2
# =========================================================

print()
print("Loading RoMa v2...")

model = RoMaV2()

# Use fast setting first.
model.apply_setting("fast")

# Helps produce more spatially diverse samples.
model.balanced_sampling = True

print("Model ready.")
print("Balanced sampling:", model.balanced_sampling)


# =========================================================
# DENSE MATCHING
# =========================================================

print()
print("Running dense matching...")

with torch.inference_mode():
    preds = model.match(
        A_PATH,
        B_PATH
    )

print("Dense matching complete.")


# =========================================================
# SAMPLE MATCHES
# =========================================================

print()
print(
    "Sampling",
    SAMPLE_COUNT,
    "correspondences..."
)

matches, overlaps, precision_AB, precision_BA = model.sample(
    preds,
    SAMPLE_COUNT
)

print(
    "Sampled matches:",
    len(matches)
)


# =========================================================
# CONVERT TO PIXEL COORDINATES
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
    .astype(np.float32)
)

ptsB = (
    kptsB
    .detach()
    .cpu()
    .numpy()
    .astype(np.float32)
)


print(
    "Pixel correspondences:",
    len(ptsA)
)


# =========================================================
# COARSE GEOGRAPHIC CONSISTENCY FILTER
#
# Both crops represent the same geographic AOI.
# Therefore a point's normalized position in A should be
# roughly equal to its normalized position in B.
# =========================================================

expected_B = np.zeros_like(
    ptsB,
    dtype=np.float32
)

expected_B[:, 0] = (
    ptsA[:, 0] / W_A
) * W_B

expected_B[:, 1] = (
    ptsA[:, 1] / H_A
) * H_B


geo_error = np.linalg.norm(
    ptsB - expected_B,
    axis=1
)


print()
print("===== GEOGRAPHIC PRIOR =====")

print(
    "Minimum error:",
    round(float(np.min(geo_error)), 3),
    "px"
)

print(
    "Median error :",
    round(float(np.median(geo_error)), 3),
    "px"
)

print(
    "Mean error   :",
    round(float(np.mean(geo_error)), 3),
    "px"
)

print(
    "Maximum error:",
    round(float(np.max(geo_error)), 3),
    "px"
)

print(
    "Threshold    :",
    GEO_THRESHOLD,
    "px"
)


geo_mask = (
    geo_error < GEO_THRESHOLD
)

ptsA_filtered = ptsA[geo_mask]
ptsB_filtered = ptsB[geo_mask]

filtered_geo_error = geo_error[geo_mask]


print()
print(
    "Matches after geographic filter:",
    len(ptsA_filtered),
    "/",
    len(ptsA)
)


if len(ptsA_filtered) < 10:
    print()
    print(
        "REGISTRATION FAILED:"
        " too few geographically plausible matches."
    )
    raise SystemExit


# =========================================================
# RANSAC PARTIAL AFFINE
#
# Scale + rotation + translation.
# Safer than a fully free homography for these ortho crops.
# =========================================================

print()
print("Running RANSAC...")

M, mask = cv2.estimateAffinePartial2D(
    ptsA_filtered,
    ptsB_filtered,
    method=cv2.RANSAC,
    ransacReprojThreshold=5.0,
    maxIters=20000,
    confidence=0.999,
    refineIters=20
)


if M is None or mask is None:
    print(
        "REGISTRATION FAILED:"
        " affine transform estimation failed."
    )
    raise SystemExit


mask = (
    mask
    .ravel()
    .astype(bool)
)

inliers = int(
    mask.sum()
)

inlier_ratio = (
    inliers
    / len(ptsA_filtered)
)


# =========================================================
# TRANSFORM PARAMETERS
# =========================================================

a = float(M[0, 0])
b = float(M[0, 1])

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


# =========================================================
# REPROJECTION RMSE
# =========================================================

ones = np.ones(
    (
        ptsA_filtered.shape[0],
        1
    ),
    dtype=np.float32
)

ptsA_h = np.hstack(
    [
        ptsA_filtered,
        ones
    ]
)

predicted = (
    M @ ptsA_h.T
).T

errors = (
    predicted[mask]
    -
    ptsB_filtered[mask]
)

squared_errors = np.sum(
    errors ** 2,
    axis=1
)

rmse = np.sqrt(
    np.mean(
        squared_errors
    )
)


# =========================================================
# INLIER GEOGRAPHIC ERROR
# =========================================================

inlier_geo_error = (
    filtered_geo_error[mask]
)

median_geo_error = float(
    np.median(
        inlier_geo_error
    )
)

mean_geo_error = float(
    np.mean(
        inlier_geo_error
    )
)


# =========================================================
# SPATIAL GRID COVERAGE
#
# Requirement asks for uniformly distributed match points.
# Use an 8x8 grid and measure how many cells contain
# verified inlier matches.
# =========================================================

occupied_cells = set()

for x, y in ptsA_filtered[mask]:

    col = int(
        x / W_A * GRID_SIZE
    )

    row = int(
        y / H_A * GRID_SIZE
    )

    col = min(
        max(col, 0),
        GRID_SIZE - 1
    )

    row = min(
        max(row, 0),
        GRID_SIZE - 1
    )

    occupied_cells.add(
        (row, col)
    )


grid_coverage = (
    len(occupied_cells)
    /
    (
        GRID_SIZE
        *
        GRID_SIZE
    )
)


# =========================================================
# EXPECTED SCALE
#
# Because both crops cover approximately the same AOI:
#
# expected horizontal scale ≈ W_B / W_A
# expected vertical scale   ≈ H_B / H_A
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


scale_error_percent = (
    abs(
        scale
        -
        expected_scale
    )
    /
    expected_scale
    *
    100.0
)


# =========================================================
# RESULTS
# =========================================================

print()
print("===== ROMA V3 RESULTS =====")

print(
    "Original sampled :",
    len(ptsA)
)

print(
    "Geo filtered     :",
    len(ptsA_filtered)
)

print(
    "Inliers          :",
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
        scale_error_percent,
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
    "Grid coverage    :",
    round(
        grid_coverage
        *
        100,
        2
    ),
    "%"
)

print(
    "Median geo error :",
    round(
        median_geo_error,
        3
    ),
    "px"
)

print(
    "Mean geo error   :",
    round(
        mean_geo_error,
        3
    ),
    "px"
)


# =========================================================
# QUALITY GATE
# =========================================================

passed = (
    inliers >= 30
    and
    inlier_ratio >= 0.20
    and
    rmse <= 5.0
    and
    grid_coverage >= 0.25
    and
    0.20 <= scale <= 0.30
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

print("============================")


# =========================================================
# SAVE NUMERICAL MATCH DATA
# =========================================================

np.save(
    f"{OUT}/roma_v3_ptsA.npy",
    ptsA_filtered
)

np.save(
    f"{OUT}/roma_v3_ptsB.npy",
    ptsB_filtered
)

np.save(
    f"{OUT}/roma_v3_inlier_mask.npy",
    mask
)

np.save(
    f"{OUT}/roma_v3_transform.npy",
    M
)


# =========================================================
# VISUALIZATION
# =========================================================

A_vis = cv2.cvtColor(
    imgA,
    cv2.COLOR_GRAY2BGR
)

B_vis = cv2.cvtColor(
    imgB,
    cv2.COLOR_GRAY2BGR
)


# Resize Chandrayaan to same display dimensions as LRO.
A_small = cv2.resize(
    A_vis,
    (
        W_B,
        H_B
    ),
    interpolation=cv2.INTER_AREA
)


scale_vis_x = (
    W_B / W_A
)

scale_vis_y = (
    H_B / H_A
)


ptsA_vis = (
    ptsA_filtered.copy()
)

ptsA_vis[:, 0] *= (
    scale_vis_x
)

ptsA_vis[:, 1] *= (
    scale_vis_y
)


# ---------------------------------------------------------
# Inlier match canvas
# ---------------------------------------------------------

canvas = np.hstack(
    [
        A_small,
        B_vis
    ]
)


inlier_indices = np.where(
    mask
)[0]


# Limit display to avoid unreadable clutter.
MAX_DRAW = 200

if len(inlier_indices) > MAX_DRAW:
    step = max(
        1,
        len(inlier_indices)
        //
        MAX_DRAW
    )

    inlier_indices = (
        inlier_indices[::step][
            :MAX_DRAW
        ]
    )


rng = np.random.default_rng(
    42
)


for idx in inlier_indices:

    pA = ptsA_vis[idx]
    pB = ptsB_filtered[idx]

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


    color_array = rng.integers(
        70,
        256,
        size=3
    )

    color = tuple(
        int(v)
        for v in color_array
    )


    cv2.circle(
        canvas,
        (
            x1,
            y1
        ),
        2,
        color,
        -1
    )

    cv2.circle(
        canvas,
        (
            x2,
            y2
        ),
        2,
        color,
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
        color,
        1,
        cv2.LINE_AA
    )


cv2.imwrite(
    f"{OUT}/roma_v3_matches.png",
    canvas
)


# =========================================================
# REGISTER CHANDRAYAAN -> LRO
# =========================================================

registered = cv2.warpAffine(
    imgA,
    M,
    (
        W_B,
        H_B
    ),
    flags=cv2.INTER_LINEAR
)


cv2.imwrite(
    f"{OUT}/roma_v3_registered.png",
    registered
)


# =========================================================
# OVERLAY
# =========================================================

overlay = cv2.addWeighted(
    registered,
    0.5,
    imgB,
    0.5,
    0
)


cv2.imwrite(
    f"{OUT}/roma_v3_overlay.png",
    overlay
)


# =========================================================
# DIFFERENCE IMAGE
# =========================================================

difference = cv2.absdiff(
    registered,
    imgB
)

cv2.imwrite(
    f"{OUT}/roma_v3_difference.png",
    difference
)


# =========================================================
# GRID VISUALIZATION
# =========================================================

grid_vis = A_small.copy()

cell_w = (
    W_B
    /
    GRID_SIZE
)

cell_h = (
    H_B
    /
    GRID_SIZE
)


for i in range(
    1,
    GRID_SIZE
):

    x = int(
        round(
            i
            *
            cell_w
        )
    )

    cv2.line(
        grid_vis,
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
    GRID_SIZE
):

    y = int(
        round(
            i
            *
            cell_h
        )
    )

    cv2.line(
        grid_vis,
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


for row, col in occupied_cells:

    center_x = int(
        (
            col
            +
            0.5
        )
        *
        cell_w
    )

    center_y = int(
        (
            row
            +
            0.5
        )
        *
        cell_h
    )

    cv2.circle(
        grid_vis,
        (
            center_x,
            center_y
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
    f"{OUT}/roma_v3_grid_coverage.png",
    grid_vis
)


# =========================================================
# FINISH
# =========================================================

print()
print("Saved:")

print(
    f"{OUT}/roma_v3_matches.png"
)

print(
    f"{OUT}/roma_v3_registered.png"
)

print(
    f"{OUT}/roma_v3_overlay.png"
)

print(
    f"{OUT}/roma_v3_difference.png"
)

print(
    f"{OUT}/roma_v3_grid_coverage.png"
)

print()
print("Done.")