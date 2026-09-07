import os
import cv2
import numpy as np
import math

PAIR = "data/test_pairs/pair_002"
OUT = "outputs/pair_002"

os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------
# LOAD
# ---------------------------------------------------------

ch = cv2.imread(
    f"{PAIR}/chandrayaan.png",
    cv2.IMREAD_GRAYSCALE
)

lro = cv2.imread(
    f"{PAIR}/lro.png",
    cv2.IMREAD_GRAYSCALE
)

if ch is None or lro is None:
    raise RuntimeError("Could not load image pair")

print("Original sizes:")
print("Chandrayaan:", ch.shape)
print("LRO        :", lro.shape)

# ---------------------------------------------------------
# PUT BOTH ON SAME PIXEL GRID
# ---------------------------------------------------------

ch = cv2.resize(
    ch,
    (lro.shape[1], lro.shape[0]),
    interpolation=cv2.INTER_AREA
)

print()
print("Matching sizes:")
print("Chandrayaan:", ch.shape)
print("LRO        :", lro.shape)

# ---------------------------------------------------------
# CONTRAST NORMALIZATION
# ---------------------------------------------------------

clahe = cv2.createCLAHE(
    clipLimit=2.5,
    tileGridSize=(8, 8)
)

ch_clahe = clahe.apply(ch)
lro_clahe = clahe.apply(lro)

# ---------------------------------------------------------
# GRADIENT REPRESENTATION
#
# We care more about terrain shape than raw brightness.
# ---------------------------------------------------------

def gradient_image(img):
    gx = cv2.Sobel(
        img,
        cv2.CV_32F,
        1,
        0,
        ksize=3
    )

    gy = cv2.Sobel(
        img,
        cv2.CV_32F,
        0,
        1,
        ksize=3
    )

    mag = cv2.magnitude(gx, gy)

    mag = cv2.normalize(
        mag,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )

    return mag.astype(np.uint8)


ch_proc = gradient_image(ch_clahe)
lro_proc = gradient_image(lro_clahe)

cv2.imwrite(
    f"{OUT}/chandrayaan_gradient.png",
    ch_proc
)

cv2.imwrite(
    f"{OUT}/lro_gradient.png",
    lro_proc
)

# ---------------------------------------------------------
# SIFT
# ---------------------------------------------------------

sift = cv2.SIFT_create(
    nfeatures=10000,
    contrastThreshold=0.005,
    edgeThreshold=20
)

kp1, des1 = sift.detectAndCompute(ch_proc, None)
kp2, des2 = sift.detectAndCompute(lro_proc, None)

print()
print("SIFT keypoints:")
print("Chandrayaan:", len(kp1))
print("LRO        :", len(kp2))

if des1 is None or des2 is None:
    raise RuntimeError("No SIFT descriptors generated")

# ---------------------------------------------------------
# BIDIRECTIONAL MATCHING
# ---------------------------------------------------------

bf = cv2.BFMatcher(cv2.NORM_L2)

forward = bf.knnMatch(des1, des2, k=2)
backward = bf.knnMatch(des2, des1, k=2)

ratio = 0.85

forward_good = {}

for pair in forward:
    if len(pair) != 2:
        continue

    m, n = pair

    if m.distance < ratio * n.distance:
        forward_good[m.queryIdx] = m


backward_good = {}

for pair in backward:
    if len(pair) != 2:
        continue

    m, n = pair

    if m.distance < ratio * n.distance:
        backward_good[m.queryIdx] = m


# Keep only mutual matches
good = []

for q_idx, m in forward_good.items():

    reverse = backward_good.get(m.trainIdx)

    if reverse is None:
        continue

    if reverse.trainIdx == q_idx:
        good.append(m)


print()
print("Mutual good matches:", len(good))

if len(good) < 3:
    print("FAILED: Not enough correspondences.")
    raise SystemExit

# ---------------------------------------------------------
# POINTS
# ---------------------------------------------------------

src_pts = np.float32(
    [kp1[m.queryIdx].pt for m in good]
)

dst_pts = np.float32(
    [kp2[m.trainIdx].pt for m in good]
)

# ---------------------------------------------------------
# RANSAC SIMILARITY / PARTIAL AFFINE
#
# Scale + rotation + translation.
# Much safer than arbitrary homography here.
# ---------------------------------------------------------

M, mask = cv2.estimateAffinePartial2D(
    src_pts,
    dst_pts,
    method=cv2.RANSAC,
    ransacReprojThreshold=4.0,
    maxIters=10000,
    confidence=0.999,
    refineIters=20
)

if M is None or mask is None:
    print("FAILED: Transform estimation failed.")
    raise SystemExit

mask = mask.ravel().astype(bool)

inliers = int(mask.sum())
inlier_ratio = inliers / len(good)

# ---------------------------------------------------------
# TRANSFORM PARAMETERS
# ---------------------------------------------------------

a = M[0, 0]
b = M[0, 1]

scale = math.sqrt(a * a + b * b)

rotation = math.degrees(
    math.atan2(
        M[1, 0],
        M[0, 0]
    )
)

tx = M[0, 2]
ty = M[1, 2]

# ---------------------------------------------------------
# RMSE
# ---------------------------------------------------------

ones = np.ones(
    (src_pts.shape[0], 1),
    dtype=np.float32
)

src_h = np.hstack(
    [src_pts, ones]
)

predicted = (
    M @ src_h.T
).T

errors = predicted[mask] - dst_pts[mask]

rmse = np.sqrt(
    np.mean(
        np.sum(
            errors ** 2,
            axis=1
        )
    )
)

# ---------------------------------------------------------
# SPATIAL COVERAGE
#
# Divide image into 4x4 cells and see how many cells
# contain at least one verified correspondence.
# ---------------------------------------------------------

grid = 4
occupied = set()

for point in src_pts[mask]:

    x, y = point

    col = min(
        int(x / ch.shape[1] * grid),
        grid - 1
    )

    row = min(
        int(y / ch.shape[0] * grid),
        grid - 1
    )

    occupied.add((row, col))

coverage = len(occupied) / (grid * grid)

# ---------------------------------------------------------
# RESULTS
# ---------------------------------------------------------

print()
print("===== SIFT V2 RESULTS =====")

print("Good matches :", len(good))
print("Inliers      :", inliers)
print(
    "Inlier ratio :",
    round(inlier_ratio, 4)
)

print(
    "RMSE         :",
    round(float(rmse), 4),
    "px"
)

print(
    "Scale        :",
    round(scale, 4)
)

print(
    "Rotation     :",
    round(rotation, 3),
    "deg"
)

print(
    "Translation  :",
    round(tx, 2),
    round(ty, 2)
)

print(
    "Grid coverage:",
    round(coverage * 100, 2),
    "%"
)

# ---------------------------------------------------------
# QUALITY GATE
# ---------------------------------------------------------

passed = (
    inliers >= 15
    and inlier_ratio >= 0.25
    and rmse <= 5.0
    and coverage >= 0.25
    and 0.8 <= scale <= 1.2
)

print()
print(
    "REGISTRATION:",
    "PASS ✅" if passed else "FAIL ❌"
)

print("===========================")

# ---------------------------------------------------------
# MATCH VISUALIZATION
# ---------------------------------------------------------

draw_mask = mask.astype(int).tolist()

match_vis = cv2.drawMatches(
    ch_proc,
    kp1,
    lro_proc,
    kp2,
    good,
    None,
    matchesMask=draw_mask,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)

cv2.imwrite(
    f"{OUT}/sift_v2_matches.png",
    match_vis
)

# ---------------------------------------------------------
# REGISTRATION
# ---------------------------------------------------------

registered = cv2.warpAffine(
    ch,
    M,
    (
        lro.shape[1],
        lro.shape[0]
    )
)

cv2.imwrite(
    f"{OUT}/sift_v2_registered.png",
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
    f"{OUT}/sift_v2_overlay.png",
    overlay
)

print()
print("Saved:")
print(f"{OUT}/sift_v2_matches.png")
print(f"{OUT}/sift_v2_registered.png")
print(f"{OUT}/sift_v2_overlay.png")