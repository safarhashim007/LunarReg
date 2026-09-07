import os
import math

import cv2
import numpy as np


# =========================================================
# CONFIG
# =========================================================

DATA = "outputs/pair_002/roma_cycle_evaluation.npz"

W_A = 1801
H_A = 1981

W_B = 448
H_B = 500


# =========================================================
# LOAD SAVED CYCLE EXPERIMENT
# =========================================================

if not os.path.exists(DATA):
    raise RuntimeError(
        f"Could not find {DATA}"
    )


data = np.load(DATA)


ptsA = data["ptsA"].astype(np.float32)
ptsB = data["ptsB"].astype(np.float32)

geo_error = data["geo_error"].astype(np.float32)

cycle_error = data["cycle_error_equiv"].astype(np.float32)
cycle_score = data["cycle_score"].astype(np.float32)

combined_conf = data["combined_confidence"].astype(np.float32)


print("===== TRUE GEOMETRY DIAGNOSTIC =====")

print()
print("Loaded matches:", len(ptsA))


# =========================================================
# EXPECTED APPROXIMATE SCALE FROM IMAGE DIMENSIONS
# =========================================================

expected_sx = W_B / W_A
expected_sy = H_B / H_A

expected_mean = (
    expected_sx
    +
    expected_sy
) / 2


print()
print("Image-size geometry:")

print(
    "Expected scale X:",
    round(expected_sx, 6)
)

print(
    "Expected scale Y:",
    round(expected_sy, 6)
)

print(
    "Expected mean   :",
    round(expected_mean, 6)
)


# =========================================================
# HELPER
# =========================================================

def rmse_affine(M, A, B):

    ones = np.ones(
        (len(A), 1),
        dtype=np.float32
    )

    A_h = np.hstack(
        [A, ones]
    )

    predicted = (
        M @ A_h.T
    ).T

    error = (
        predicted
        -
        B
    )

    return float(
        np.sqrt(
            np.mean(
                np.sum(
                    error ** 2,
                    axis=1
                )
            )
        )
    )


# =========================================================
# TEST TRUE MATCH SETS
#
# Geo-reference is used ONLY to choose the evaluation set.
# It will not be used in the final matcher.
# =========================================================

for threshold in [5, 10, 20]:

    true_mask = (
        geo_error
        <=
        threshold
    )

    A = ptsA[true_mask]
    B = ptsB[true_mask]

    print()
    print()
    print(
        "======================================"
    )

    print(
        f"GROUND-TRUTH MATCHES <= {threshold}px"
    )

    print(
        "======================================"
    )

    print(
        "Matches:",
        len(A)
    )

    if len(A) < 6:
        print(
            "Not enough points."
        )
        continue


    # =====================================================
    # PARTIAL AFFINE
    #
    # Uniform scale + rotation + translation
    # =====================================================

    M_partial, mask_partial = (
        cv2.estimateAffinePartial2D(
            A,
            B,

            method=cv2.RANSAC,

            ransacReprojThreshold=5.0,

            maxIters=100000,

            confidence=0.9999,

            refineIters=50
        )
    )


    if M_partial is not None:

        mask_partial = (
            mask_partial
            .ravel()
            .astype(bool)
        )

        PA = A[mask_partial]
        PB = B[mask_partial]

        a = float(
            M_partial[0, 0]
        )

        b = float(
            M_partial[0, 1]
        )

        scale = math.sqrt(
            a * a
            +
            b * b
        )

        rotation = math.degrees(
            math.atan2(
                M_partial[1, 0],
                M_partial[0, 0]
            )
        )

        tx = float(
            M_partial[0, 2]
        )

        ty = float(
            M_partial[1, 2]
        )

        rmse = rmse_affine(
            M_partial,
            PA,
            PB
        )


        print()
        print(
            "--- PARTIAL AFFINE ---"
        )

        print(
            "Inliers:",
            len(PA),
            "/",
            len(A)
        )

        print(
            "Scale:",
            round(scale, 6)
        )

        print(
            "Rotation:",
            round(rotation, 4),
            "deg"
        )

        print(
            "Translation:",
            round(tx, 3),
            round(ty, 3)
        )

        print(
            "RMSE:",
            round(rmse, 4),
            "px"
        )


    # =====================================================
    # FULL AFFINE
    #
    # Supports:
    # x/y scale difference
    # rotation
    # shear
    # translation
    # =====================================================

    M_full, mask_full = (
        cv2.estimateAffine2D(
            A,
            B,

            method=cv2.RANSAC,

            ransacReprojThreshold=5.0,

            maxIters=100000,

            confidence=0.9999,

            refineIters=50
        )
    )


    if M_full is not None:

        mask_full = (
            mask_full
            .ravel()
            .astype(bool)
        )

        FA = A[mask_full]
        FB = B[mask_full]


        linear = M_full[:, :2]


        # =================================================
        # SVD decomposes the affine matrix into effective
        # scales and rotation.
        # =================================================

        U, singular_values, Vt = (
            np.linalg.svd(
                linear
            )
        )

        R = U @ Vt


        if np.linalg.det(R) < 0:

            U[:, -1] *= -1

            R = U @ Vt


        rotation_full = math.degrees(
            math.atan2(
                R[1, 0],
                R[0, 0]
            )
        )


        scale_1 = float(
            singular_values[0]
        )

        scale_2 = float(
            singular_values[1]
        )


        anisotropy = (
            max(scale_1, scale_2)
            /
            min(scale_1, scale_2)
        )


        rmse_full = rmse_affine(
            M_full,
            FA,
            FB
        )


        print()
        print(
            "--- FULL AFFINE ---"
        )

        print(
            "Inliers:",
            len(FA),
            "/",
            len(A)
        )

        print(
            "Matrix:"
        )

        print(
            np.array2string(
                M_full,
                precision=6,
                suppress_small=True
            )
        )

        print(
            "Effective scale 1:",
            round(scale_1, 6)
        )

        print(
            "Effective scale 2:",
            round(scale_2, 6)
        )

        print(
            "Anisotropy ratio:",
            round(anisotropy, 6)
        )

        print(
            "Rotation:",
            round(
                rotation_full,
                4
            ),
            "deg"
        )

        print(
            "Translation:",
            round(
                float(
                    M_full[0, 2]
                ),
                3
            ),
            round(
                float(
                    M_full[1, 2]
                ),
                3
            )
        )

        print(
            "RMSE:",
            round(
                rmse_full,
                4
            ),
            "px"
        )


# =========================================================
# FALSE MODE DIAGNOSTIC
#
# Find the high-cycle-score matches that are actually bad.
# This helps confirm the -13 degree competing mode.
# =========================================================

top_order = np.argsort(
    cycle_score
)[::-1]


top100 = top_order[:100]

true_top100 = (
    geo_error[top100]
    <=
    10
)

false_top100 = (
    geo_error[top100]
    >
    20
)


print()
print()
print(
    "======================================"
)
print(
    "TOP-100 CYCLE SCORE ANALYSIS"
)
print(
    "======================================"
)

print(
    "Correct <=10px:",
    int(
        true_top100.sum()
    )
)

print(
    "Clearly false >20px:",
    int(
        false_top100.sum()
    )
)


false_indices = top100[
    false_top100
]


if len(false_indices) >= 6:

    false_A = ptsA[
        false_indices
    ]

    false_B = ptsB[
        false_indices
    ]


    M_false, false_mask = (
        cv2.estimateAffinePartial2D(
            false_A,
            false_B,

            method=cv2.RANSAC,

            ransacReprojThreshold=10.0,

            maxIters=100000,

            confidence=0.9999,

            refineIters=50
        )
    )


    if M_false is not None:

        fa = float(
            M_false[0, 0]
        )

        fb = float(
            M_false[0, 1]
        )

        false_scale = math.sqrt(
            fa * fa
            +
            fb * fb
        )

        false_rotation = math.degrees(
            math.atan2(
                M_false[1, 0],
                M_false[0, 0]
            )
        )


        print()
        print(
            "False-mode transform:"
        )

        print(
            "Scale:",
            round(
                false_scale,
                6
            )
        )

        print(
            "Rotation:",
            round(
                false_rotation,
                4
            ),
            "deg"
        )

        print(
            "Translation:",
            round(
                float(
                    M_false[0, 2]
                ),
                3
            ),
            round(
                float(
                    M_false[1, 2]
                ),
                3
            )
        )


print()
print(
    "Diagnostic complete."
)