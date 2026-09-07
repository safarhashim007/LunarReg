import os
import cv2
import numpy as np

PAIR = "data/test_pairs/pair_001"
OUT = "outputs/pair_001"

os.makedirs(OUT, exist_ok=True)

ch_path = f"{PAIR}/chandrayaan.png"
lro_path = f"{PAIR}/lro.png"

# --------------------------------------------------
# 1. Load images
# --------------------------------------------------

ch = cv2.imread(ch_path, cv2.IMREAD_GRAYSCALE)
lro = cv2.imread(lro_path, cv2.IMREAD_GRAYSCALE)

if ch is None:
    raise RuntimeError("Could not load Chandrayaan image")

if lro is None:
    raise RuntimeError("Could not load LRO image")

print("Original sizes:")
print("Chandrayaan:", ch.shape)
print("LRO        :", lro.shape)

# --------------------------------------------------
# 2. Resolution sanity test
#
# Chandrayaan ≈ 5 m/pixel
# LRO ≈ 20 m/pixel
#
# Resize Chandrayaan to LRO dimensions so we first
# verify that the geographic pair really matches.
# --------------------------------------------------

ch_scaled = cv2.resize(
    ch,
    (lro.shape[1], lro.shape[0]),
    interpolation=cv2.INTER_AREA
)

print()
print("Matching sizes:")
print("Chandrayaan:", ch_scaled.shape)
print("LRO        :", lro.shape)

# --------------------------------------------------
# 3. Local contrast normalization
# --------------------------------------------------

clahe = cv2.createCLAHE(
    clipLimit=2.0,
    tileGridSize=(8, 8)
)

ch_proc = clahe.apply(ch_scaled)
lro_proc = clahe.apply(lro)

cv2.imwrite(f"{OUT}/chandrayaan_processed.png", ch_proc)
cv2.imwrite(f"{OUT}/lro_processed.png", lro_proc)

# --------------------------------------------------
# 4. SIFT features
# --------------------------------------------------

sift = cv2.SIFT_create(
    nfeatures=10000,
    contrastThreshold=0.01
)

kp1, des1 = sift.detectAndCompute(ch_proc, None)
kp2, des2 = sift.detectAndCompute(lro_proc, None)

print()
print("SIFT keypoints:")
print("Chandrayaan:", len(kp1))
print("LRO        :", len(kp2))

if des1 is None or des2 is None:
    raise RuntimeError("SIFT could not generate descriptors")

# --------------------------------------------------
# 5. Descriptor matching
# --------------------------------------------------

matcher = cv2.BFMatcher(cv2.NORM_L2)

knn = matcher.knnMatch(
    des1,
    des2,
    k=2
)

# Lowe ratio test
good = []

for pair in knn:
    if len(pair) != 2:
        continue

    m, n = pair

    if m.distance < 0.78 * n.distance:
        good.append(m)

print()
print("Good matches:", len(good))

# --------------------------------------------------
# 6. Geometric verification using RANSAC
# --------------------------------------------------

if len(good) < 4:
    print("Not enough matches to estimate homography.")
    raise SystemExit

src_pts = np.float32(
    [kp1[m.queryIdx].pt for m in good]
).reshape(-1, 1, 2)

dst_pts = np.float32(
    [kp2[m.trainIdx].pt for m in good]
).reshape(-1, 1, 2)

H, mask = cv2.findHomography(
    src_pts,
    dst_pts,
    cv2.RANSAC,
    3.0,
    maxIters=10000,
    confidence=0.999
)

if H is None:
    print("Homography estimation failed.")
    raise SystemExit

mask = mask.ravel().astype(bool)

inlier_count = int(mask.sum())
inlier_ratio = inlier_count / len(good)

# --------------------------------------------------
# 7. RMSE of inlier correspondences
# --------------------------------------------------

predicted = cv2.perspectiveTransform(
    src_pts,
    H
)

errors = predicted[mask] - dst_pts[mask]

rmse = np.sqrt(
    np.mean(
        np.sum(errors ** 2, axis=2)
    )
)

print()
print("===== RESULTS =====")
print("Good matches :", len(good))
print("Inliers      :", inlier_count)
print("Inlier ratio :", round(inlier_ratio, 4))
print("RMSE         :", round(float(rmse), 4), "px")
print("===================")

# --------------------------------------------------
# 8. Draw verified matches
# --------------------------------------------------

match_mask = mask.astype(int).tolist()

visualization = cv2.drawMatches(
    ch_proc,
    kp1,
    lro_proc,
    kp2,
    good,
    None,
    matchesMask=match_mask,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)

cv2.imwrite(
    f"{OUT}/sift_matches.png",
    visualization
)

# --------------------------------------------------
# 9. Register Chandrayaan onto LRO
# --------------------------------------------------

registered = cv2.warpPerspective(
    ch_scaled,
    H,
    (lro.shape[1], lro.shape[0])
)

cv2.imwrite(
    f"{OUT}/registered.png",
    registered
)

# --------------------------------------------------
# 10. Overlay
# --------------------------------------------------

overlay = cv2.addWeighted(
    registered,
    0.5,
    lro,
    0.5,
    0
)

cv2.imwrite(
    f"{OUT}/overlay.png",
    overlay
)

print()
print("Saved:")
print(f"{OUT}/sift_matches.png")
print(f"{OUT}/registered.png")
print(f"{OUT}/overlay.png")