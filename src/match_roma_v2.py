import os
import cv2
import torch
import numpy as np
import math

from romav2 import RoMaV2


PAIR = "data/test_pairs/pair_002"
OUT = "outputs/pair_002"

os.makedirs(OUT, exist_ok=True)

A_PATH = f"{PAIR}/chandrayaan.png"
B_PATH = f"{PAIR}/lro.png"

torch.set_float32_matmul_precision("highest")


# ---------------------------------------------------------
# READ ORIGINAL IMAGE DIMENSIONS
# ---------------------------------------------------------

imgA = cv2.imread(A_PATH, cv2.IMREAD_GRAYSCALE)
imgB = cv2.imread(B_PATH, cv2.IMREAD_GRAYSCALE)

if imgA is None or imgB is None:
    raise RuntimeError("Could not load image pair")

H_A, W_A = imgA.shape
H_B, W_B = imgB.shape

print("Image sizes:")
print("Chandrayaan:", W_A, "x", H_A)
print("LRO        :", W_B, "x", H_B)


# ---------------------------------------------------------
# LOAD ROMA
# ---------------------------------------------------------

print()
print("Loading RoMa v2...")

model = RoMaV2()

model.apply_setting("fast")

# Diverse sampling helps our uniform-distribution objective
model.balanced_sampling = True

print("Model ready.")
print("Balanced sampling:", model.balanced_sampling)


# ---------------------------------------------------------
# DENSE MATCH
# ---------------------------------------------------------

print()
print("Running dense matching...")

with torch.inference_mode():
    preds = model.match(
        A_PATH,
        B_PATH
    )

print("Dense matching complete.")


# ---------------------------------------------------------
# SAMPLE CORRESPONDENCES
# ---------------------------------------------------------

N = 2000

print()
print("Sampling", N, "correspondences...")

matches, overlaps, precision_AB, precision_BA = model.sample(
    preds,
    N
)

print("Sampled matches:", len(matches))


# ---------------------------------------------------------
# NORMALIZED -> PIXEL COORDINATES
# ---------------------------------------------------------

kptsA, kptsB = model.to_pixel_coordinates(
    matches,
    H_A,
    W_A,
    H_B,
    W_B
)

ptsA = kptsA.detach().cpu().numpy().astype(np.float32)
ptsB = kptsB.detach().cpu().numpy().astype(np.float32)

print("Pixel correspondences:", len(ptsA))


# ---------------------------------------------------------
# GEOMETRIC VERIFICATION
#
# Since both images are geographic crops of same AOI,
# use partial affine first rather than arbitrary homography.
# ---------------------------------------------------------

M, mask = cv2.estimateAffinePartial2D(
    ptsA,
    ptsB,
    method=cv2.RANSAC,
    ransacReprojThreshold=5.0,
    maxIters=20000,
    confidence=0.999,
    refineIters=20
)

if M is None or mask is None:
    print("REGISTRATION FAILED: transform estimation failed")
    raise SystemExit

mask = mask.ravel().astype(bool)

inliers = int(mask.sum())
inlier_ratio = inliers / len(ptsA)


# ---------------------------------------------------------
# RMSE
# ---------------------------------------------------------

ones = np.ones(
    (ptsA.shape[0], 1),
    dtype=np.float32
)

ptsA_h = np.hstack([
    ptsA,
    ones
])

pred = (
    M @ ptsA_h.T
).T

err = pred[mask] - ptsB[mask]

rmse = np.sqrt(
    np.mean(
        np.sum(
            err ** 2,
            axis=1
        )
    )
)


# ---------------------------------------------------------
# TRANSFORM PARAMETERS
# ---------------------------------------------------------

a = M[0, 0]
b = M[0, 1]

scale = math.sqrt(
    a * a + b * b
)

rotation = math.degrees(
    math.atan2(
        M[1, 0],
        M[0, 0]
    )
)

tx = M[0, 2]
ty = M[1, 2]


# ---------------------------------------------------------
# SPATIAL COVERAGE
#
# 8×8 grid, closer to the PS requirement for distributed
# correspondences across image.
# ---------------------------------------------------------

GRID = 8

occupied = set()

for x, y in ptsA[mask]:

    col = min(
        int(x / W_A * GRID),
        GRID - 1
    )

    row = min(
        int(y / H_A * GRID),
        GRID - 1
    )

    occupied.add(
        (row, col)
    )

grid_coverage = (
    len(occupied)
    / (GRID * GRID)
)


# ---------------------------------------------------------
# RESULTS
# ---------------------------------------------------------

print()
print("===== ROMA V2 RESULTS =====")

print("Sampled       :", len(ptsA))
print("Inliers       :", inliers)
print(
    "Inlier ratio  :",
    round(inlier_ratio, 4)
)

print(
    "RMSE          :",
    round(float(rmse), 4),
    "px"
)

print(
    "Scale         :",
    round(scale, 4)
)

print(
    "Rotation      :",
    round(rotation, 3),
    "deg"
)

print(
    "Translation   :",
    round(tx, 2),
    round(ty, 2)
)

print(
    "Grid coverage :",
    round(
        grid_coverage * 100,
        2
    ),
    "%"
)


# ---------------------------------------------------------
# QUALITY GATE
# ---------------------------------------------------------

passed = (
    inliers >= 30
    and inlier_ratio >= 0.10
    and rmse <= 5.0
    and grid_coverage >= 0.30
)

print()
print(
    "REGISTRATION:",
    "PASS ✅" if passed else "FAIL ❌"
)

print("===========================")


# ---------------------------------------------------------
# VISUALIZE INLIERS
# ---------------------------------------------------------

A_vis = cv2.cvtColor(
    imgA,
    cv2.COLOR_GRAY2BGR
)

B_vis = cv2.cvtColor(
    imgB,
    cv2.COLOR_GRAY2BGR
)

# Resize A only for visualization alongside B
A_small = cv2.resize(
    A_vis,
    (W_B, H_B),
    interpolation=cv2.INTER_AREA
)

# Scale A keypoints to resized visualization
sx = W_B / W_A
sy = H_B / H_A

ptsA_vis = ptsA.copy()

ptsA_vis[:, 0] *= sx
ptsA_vis[:, 1] *= sy


# Combine canvases
canvas = np.hstack([
    A_small,
    B_vis
])

# Draw max 150 inliers so visualization stays readable
indices = np.where(mask)[0]

if len(indices) > 150:
    indices = indices[:150]

rng = np.random.default_rng(42)

for idx in indices:

    p1 = ptsA_vis[idx]
    p2 = ptsB[idx]

    x1 = int(p1[0])
    y1 = int(p1[1])

    x2 = int(p2[0]) + W_B
    y2 = int(p2[1])

    color = tuple(
        int(v)
        for v in rng.integers(
            50,
            256,
            size=3
        )
    )

    cv2.circle(
        canvas,
        (x1, y1),
        2,
        color,
        -1
    )

    cv2.circle(
        canvas,
        (x2, y2),
        2,
        color,
        -1
    )

    cv2.line(
        canvas,
        (x1, y1),
        (x2, y2),
        color,
        1
    )


cv2.imwrite(
    f"{OUT}/roma_v2_matches.png",
    canvas
)


# ---------------------------------------------------------
# REGISTER IMAGE
# ---------------------------------------------------------

registered = cv2.warpAffine(
    imgA,
    M,
    (
        W_B,
        H_B
    )
)

cv2.imwrite(
    f"{OUT}/roma_v2_registered.png",
    registered
)


overlay = cv2.addWeighted(
    registered,
    0.5,
    imgB,
    0.5,
    0
)

cv2.imwrite(
    f"{OUT}/roma_v2_overlay.png",
    overlay
)


print()
print("Saved:")
print(f"{OUT}/roma_v2_matches.png")
print(f"{OUT}/roma_v2_registered.png")
print(f"{OUT}/roma_v2_overlay.png")