import os
import math
import csv

import cv2
import torch
import torch.nn.functional as F
import rasterio
import numpy as np

from romav2 import RoMaV2


# =========================================================
# LUNARREG V7
#
# Multi-Hypothesis Geometric Model Selection
#
# Pipeline:
#
# RoMa Fast + Bidirectional
#       ↓
# Forward/backward cycle consistency
#       ↓
# Top cycle-quality candidates
#       ↓
# Thousands of transform hypotheses
#       ↓
# Score each using:
#   - weighted geometric consensus
#   - cycle consistency
#   - spatial coverage
#   - residual quality
#   - expected scale prior
#   - expected orientation prior
#       ↓
# Refine best hypotheses
#       ↓
# Spatially balanced final fit
#       ↓
# Registration
#
# IMPORTANT:
# GeoTIFF coordinates are used ONLY at the END
# for independent evaluation.
# =========================================================


# =========================================================
# CONFIG
# =========================================================

PAIR = "data/test_pairs/pair_002"
OUT = "outputs/pair_002"

CH_PATH = f"{PAIR}/chandrayaan.png"
LRO_PATH = f"{PAIR}/lro.png"

CH_TIF = f"{PAIR}/chandrayaan.tif"
LRO_TIF = f"{PAIR}/lro.tif"


# RoMa
SAMPLE_COUNT = 5000


# Candidate pool
CANDIDATE_TOP_K = 300


# Hypothesis generation
PAIR_HYPOTHESES = 4000
SUBSET_HYPOTHESES = 400

SUBSET_SIZE = 20


# Geometric scoring
HYPOTHESIS_INLIER_THRESHOLD = 8.0

MIN_HYPOTHESIS_INLIERS = 12


# Hypothesis refinement
REFINE_TOP_N = 30

REFINE_RANSAC_THRESHOLD = 6.0


# Final registration
FINAL_RANSAC_THRESHOLD = 5.0

GRID = 8

TOP_PER_CELL = 10


# ---------------------------------------------------------
# Physical / metadata priors
#
# These are NOT GeoTIFF control points.
#
# Our two orthorectified crops cover approximately the
# same north-up lunar area.
#
# Therefore:
#
# expected scale ≈ image-size ratio
# expected rotation ≈ 0 degrees
#
# These are SOFT priors, not hard answers.
# ---------------------------------------------------------

EXPECTED_ROTATION_DEG = 0.0

ROTATION_PRIOR_SIGMA_DEG = 5.0

SCALE_PRIOR_SIGMA_FRACTION = 0.06


# Broad hard sanity limits
MIN_SCALE = 0.15
MAX_SCALE = 0.40

MAX_ABS_ROTATION = 35.0


# Minimal-pair stability
MIN_PAIR_DISTANCE_A = 100.0
MIN_PAIR_DISTANCE_B = 15.0


# Moon projection for FINAL evaluation only
MOON_RADIUS = 1737400.0

STANDARD_PARALLEL = math.radians(10.0)


RANDOM_SEED = 42


os.makedirs(
    OUT,
    exist_ok=True
)


torch.set_float32_matmul_precision(
    "highest"
)


rng = np.random.default_rng(
    RANDOM_SEED
)


# =========================================================
# HELPERS
# =========================================================

def wrap_angle_deg(angle):

    return (
        (angle + 180.0)
        % 360.0
        -
        180.0
    )


def transform_parameters(M):

    a = float(
        M[0, 0]
    )

    c = float(
        M[1, 0]
    )

    scale = math.sqrt(
        a * a
        +
        c * c
    )

    rotation = math.degrees(
        math.atan2(
            c,
            a
        )
    )

    tx = float(
        M[0, 2]
    )

    ty = float(
        M[1, 2]
    )

    return (
        scale,
        rotation,
        tx,
        ty
    )


def apply_transform(
    M,
    points
):

    ones = np.ones(
        (
            len(points),
            1
        ),
        dtype=np.float32
    )

    points_h = np.hstack(
        [
            points,
            ones
        ]
    )

    return (
        M
        @
        points_h.T
    ).T


def residuals_for_transform(
    M,
    A,
    B
):

    predicted = apply_transform(
        M,
        A
    )

    return np.linalg.norm(
        predicted
        -
        B,
        axis=1
    )


def coverage_from_points(
    points,
    width,
    height,
    grid
):

    occupied = set()

    for x, y in points:

        col = min(
            max(
                int(
                    x
                    /
                    width
                    *
                    grid
                ),
                0
            ),
            grid - 1
        )

        row = min(
            max(
                int(
                    y
                    /
                    height
                    *
                    grid
                ),
                0
            ),
            grid - 1
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
        (
            grid
            *
            grid
        )
    )

    return (
        coverage,
        occupied
    )


def similarity_from_two_points(
    A1,
    A2,
    B1,
    B2
):

    vec_A = (
        A2
        -
        A1
    )

    vec_B = (
        B2
        -
        B1
    )


    dist_A = float(
        np.linalg.norm(
            vec_A
        )
    )

    dist_B = float(
        np.linalg.norm(
            vec_B
        )
    )


    if (
        dist_A
        <
        MIN_PAIR_DISTANCE_A
    ):
        return None


    if (
        dist_B
        <
        MIN_PAIR_DISTANCE_B
    ):
        return None


    scale = (
        dist_B
        /
        dist_A
    )


    if (
        scale < MIN_SCALE
        or
        scale > MAX_SCALE
    ):
        return None


    angle_A = math.atan2(
        float(
            vec_A[1]
        ),
        float(
            vec_A[0]
        )
    )

    angle_B = math.atan2(
        float(
            vec_B[1]
        ),
        float(
            vec_B[0]
        )
    )


    theta = (
        angle_B
        -
        angle_A
    )


    rotation = math.degrees(
        theta
    )

    rotation = wrap_angle_deg(
        rotation
    )


    if (
        abs(
            rotation
        )
        >
        MAX_ABS_ROTATION
    ):
        return None


    cos_t = math.cos(
        theta
    )

    sin_t = math.sin(
        theta
    )


    R = np.array(
        [
            [
                scale * cos_t,
                -scale * sin_t
            ],
            [
                scale * sin_t,
                scale * cos_t
            ]
        ],
        dtype=np.float64
    )


    translation = (
        B1
        -
        R
        @
        A1
    )


    M = np.zeros(
        (
            2,
            3
        ),
        dtype=np.float64
    )

    M[:, :2] = R

    M[:, 2] = translation


    return M


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


expected_scale_x = (
    W_B
    /
    W_A
)

expected_scale_y = (
    H_B
    /
    H_A
)

expected_scale = (
    expected_scale_x
    +
    expected_scale_y
) / 2.0


scale_prior_sigma = (
    expected_scale
    *
    SCALE_PRIOR_SIGMA_FRACTION
)


print(
    "===== LUNARREG ROMA V7 ====="
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

print()

print(
    "Expected scale X:",
    round(
        expected_scale_x,
        6
    )
)

print(
    "Expected scale Y:",
    round(
        expected_scale_y,
        6
    )
)

print(
    "Expected mean scale:",
    round(
        expected_scale,
        6
    )
)

print(
    "Expected rotation:",
    EXPECTED_ROTATION_DEG,
    "deg"
)


# =========================================================
# LOAD ROMA
# =========================================================

print()
print(
    "Loading RoMa v2..."
)


model = RoMaV2()

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
# BIDIRECTIONAL DENSE MATCHING
# =========================================================

print()
print(
    "Running bidirectional RoMa..."
)


with torch.inference_mode():

    predictions = model.match(
        CH_PATH,
        LRO_PATH
    )


if predictions["warp_BA"] is None:

    raise RuntimeError(
        "Backward RoMa warp was not produced."
    )


print(
    "Dense matching complete ✅"
)


# =========================================================
# SAMPLE FORWARD MATCHES ONLY
# =========================================================

print()

print(
    "Sampling",
    SAMPLE_COUNT,
    "forward correspondences..."
)


model.bidirectional = False


matches, forward_conf, _, _ = (
    model.sample(
        predictions,
        SAMPLE_COUNT
    )
)


model.bidirectional = True


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


forward_conf = (
    forward_conf
    .detach()
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

B_norm = matches[
    :,
    2:4
]


warp_BA = predictions[
    "warp_BA"
]


warp_BA_chw = (
    warp_BA.permute(
        0,
        3,
        1,
        2
    )
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

overlap_BA = predictions[
    "overlap_BA"
]


overlap_BA_chw = (
    overlap_BA.permute(
        0,
        3,
        1,
        2
    )
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
    reverse_conf
    .detach()
    .cpu()
    .numpy()
    .astype(np.float32)
)


# =========================================================
# BACKWARD POSITION -> CHANDRAYAAN PIXELS
# =========================================================

back_x = (
    (
        A_back_norm[:, 0]
        +
        1.0
    )
    /
    2.0
    *
    W_A
)

back_y = (
    (
        A_back_norm[:, 1]
        +
        1.0
    )
    /
    2.0
    *
    H_A
)


back_A = torch.stack(
    [
        back_x,
        back_y
    ],
    dim=1
)


back_A = (
    back_A
    .detach()
    .cpu()
    .numpy()
    .astype(np.float32)
)


# =========================================================
# CYCLE ERROR
# =========================================================

cycle_error_A = np.linalg.norm(
    back_A
    -
    ptsA,
    axis=1
)


# Convert to approximate LRO-pixel units
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
# VALIDITY FILTER
# =========================================================

valid = (
    np.isfinite(
        cycle_error
    )
    &
    np.isfinite(
        cycle_score
    )
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


ptsA = ptsA[
    valid
]

ptsB = ptsB[
    valid
]

combined_conf = combined_conf[
    valid
]

cycle_error = cycle_error[
    valid
]

cycle_score = cycle_score[
    valid
]


print()

print(
    "Valid matches:",
    len(ptsA)
)


# =========================================================
# BASE WEIGHT
#
# High combined confidence
# +
# low forward/backward cycle error
# =========================================================

base_weight = (
    combined_conf
    *
    np.exp(
        -cycle_error
        /
        6.0
    )
)


weight_p95 = max(
    float(
        np.percentile(
            base_weight,
            95
        )
    ),
    1e-8
)


total_weight = max(
    float(
        np.sum(
            base_weight
        )
    ),
    1e-8
)


# =========================================================
# HYPOTHESIS SCORING
# =========================================================

def score_hypothesis(
    M,
    label
):

    if M is None:
        return None


    scale, rotation, tx, ty = (
        transform_parameters(
            M
        )
    )


    if (
        not np.isfinite(
            scale
        )
    ):
        return None


    if (
        scale < MIN_SCALE
        or
        scale > MAX_SCALE
    ):
        return None


    rotation_delta = wrap_angle_deg(
        rotation
        -
        EXPECTED_ROTATION_DEG
    )


    if (
        abs(
            rotation_delta
        )
        >
        MAX_ABS_ROTATION
    ):
        return None


    residual = residuals_for_transform(
        M,
        ptsA,
        ptsB
    )


    inlier_mask = (
        residual
        <=
        HYPOTHESIS_INLIER_THRESHOLD
    )


    inlier_count = int(
        inlier_mask.sum()
    )


    if (
        inlier_count
        <
        MIN_HYPOTHESIS_INLIERS
    ):
        return None


    inlier_A = ptsA[
        inlier_mask
    ]


    coverage, cells = (
        coverage_from_points(
            inlier_A,
            W_A,
            H_A,
            GRID
        )
    )


    weighted_support = (
        float(
            np.sum(
                base_weight[
                    inlier_mask
                ]
            )
        )
        /
        total_weight
    )


    median_residual = float(
        np.median(
            residual[
                inlier_mask
            ]
        )
    )


    residual_score = math.exp(
        -median_residual
        /
        3.0
    )


    inlier_quality = (
        float(
            np.mean(
                base_weight[
                    inlier_mask
                ]
            )
        )
        /
        weight_p95
    )


    inlier_quality = float(
        np.clip(
            inlier_quality,
            0.0,
            1.0
        )
    )


    # -----------------------------------------------------
    # Scale prior
    # -----------------------------------------------------

    scale_z = (
        scale
        -
        expected_scale
    ) / scale_prior_sigma


    scale_prior = math.exp(
        -0.5
        *
        scale_z
        *
        scale_z
    )


    # -----------------------------------------------------
    # Orientation prior
    # -----------------------------------------------------

    rot_z = (
        rotation_delta
        /
        ROTATION_PRIOR_SIGMA_DEG
    )


    rotation_prior = math.exp(
        -0.5
        *
        rot_z
        *
        rot_z
    )


    # -----------------------------------------------------
    # Data evidence
    # -----------------------------------------------------

    data_score = (
        0.40
        *
        weighted_support
        +
        0.25
        *
        coverage
        +
        0.20
        *
        residual_score
        +
        0.15
        *
        inlier_quality
    )


    # -----------------------------------------------------
    # Physical plausibility
    #
    # Soft multiplier.
    #
    # Even a low-prior hypothesis still retains 30% of its
    # data score, so the prior does not completely dictate
    # the solution.
    # -----------------------------------------------------

    prior_score = math.sqrt(
        scale_prior
        *
        rotation_prior
    )


    final_score = (
        data_score
        *
        (
            0.30
            +
            0.70
            *
            prior_score
        )
    )


    return {
        "label": label,

        "M": M.astype(
            np.float64
        ),

        "score": float(
            final_score
        ),

        "data_score": float(
            data_score
        ),

        "prior_score": float(
            prior_score
        ),

        "weighted_support": float(
            weighted_support
        ),

        "coverage": float(
            coverage
        ),

        "inliers": inlier_count,

        "median_residual": float(
            median_residual
        ),

        "scale": float(
            scale
        ),

        "rotation": float(
            rotation
        ),

        "tx": float(
            tx
        ),

        "ty": float(
            ty
        ),

        "scale_prior": float(
            scale_prior
        ),

        "rotation_prior": float(
            rotation_prior
        ),

        "mask": inlier_mask
    }


# =========================================================
# CANDIDATE POOL
# =========================================================

order = np.argsort(
    cycle_score
)[::-1]


candidate_count = min(
    CANDIDATE_TOP_K,
    len(order)
)


candidate_indices = order[
    :candidate_count
]


candidate_A = ptsA[
    candidate_indices
]

candidate_B = ptsB[
    candidate_indices
]


print()
print(
    "===== CANDIDATE POOL ====="
)

print(
    "Candidates:",
    len(
        candidate_A
    )
)

print(
    "Median cycle error:",
    round(
        float(
            np.median(
                cycle_error[
                    candidate_indices
                ]
            )
        ),
        3
    ),
    "px"
)

print(
    "Median confidence:",
    round(
        float(
            np.median(
                combined_conf[
                    candidate_indices
                ]
            )
        ),
        4
    )
)


# =========================================================
# GENERATE HYPOTHESES
# =========================================================

print()
print(
    "===== GENERATING HYPOTHESES ====="
)


hypotheses = []


# ---------------------------------------------------------
# A. Deterministic RANSAC hypotheses from various top-Ks
# ---------------------------------------------------------

for k in [
    30,
    50,
    75,
    100,
    150,
    200,
    300
]:

    if (
        k
        >
        len(
            candidate_A
        )
    ):
        continue


    M, mask = cv2.estimateAffinePartial2D(
        candidate_A[:k],
        candidate_B[:k],

        method=cv2.RANSAC,

        ransacReprojThreshold=8.0,

        maxIters=100000,

        confidence=0.9999,

        refineIters=30
    )


    result = score_hypothesis(
        M,
        f"top{k}_ransac"
    )


    if (
        result
        is not None
    ):
        hypotheses.append(
            result
        )


# ---------------------------------------------------------
# B. Random minimal-pair similarity hypotheses
# ---------------------------------------------------------

pair_success = 0


for n in range(
    PAIR_HYPOTHESES
):

    i, j = rng.choice(
        len(
            candidate_A
        ),
        size=2,
        replace=False
    )


    M = similarity_from_two_points(
        candidate_A[i],
        candidate_A[j],
        candidate_B[i],
        candidate_B[j]
    )


    if M is None:
        continue


    result = score_hypothesis(
        M,
        "pair"
    )


    if result is None:
        continue


    hypotheses.append(
        result
    )

    pair_success += 1


# ---------------------------------------------------------
# C. Random-subset RANSAC hypotheses
# ---------------------------------------------------------

subset_success = 0


for n in range(
    SUBSET_HYPOTHESES
):

    count = min(
        SUBSET_SIZE,
        len(
            candidate_A
        )
    )


    subset_idx = rng.choice(
        len(
            candidate_A
        ),
        size=count,
        replace=False
    )


    subset_A = candidate_A[
        subset_idx
    ]

    subset_B = candidate_B[
        subset_idx
    ]


    M, mask = cv2.estimateAffinePartial2D(
        subset_A,
        subset_B,

        method=cv2.RANSAC,

        ransacReprojThreshold=8.0,

        maxIters=5000,

        confidence=0.999,

        refineIters=20
    )


    result = score_hypothesis(
        M,
        "subset_ransac"
    )


    if result is None:
        continue


    hypotheses.append(
        result
    )

    subset_success += 1


print(
    "Valid pair hypotheses:",
    pair_success
)

print(
    "Valid subset hypotheses:",
    subset_success
)

print(
    "Total scored hypotheses:",
    len(
        hypotheses
    )
)


if (
    len(
        hypotheses
    )
    ==
    0
):

    raise RuntimeError(
        "No valid transform hypotheses generated."
    )


# =========================================================
# SORT INITIAL HYPOTHESES
# =========================================================

hypotheses.sort(
    key=lambda x: x["score"],
    reverse=True
)


print()
print(
    "===== TOP INITIAL HYPOTHESES ====="
)


for rank, h in enumerate(
    hypotheses[:10],
    start=1
):

    print()

    print(
        f"#{rank}"
    )

    print(
        "Source          :",
        h["label"]
    )

    print(
        "Score           :",
        round(
            h["score"],
            5
        )
    )

    print(
        "Inliers         :",
        h["inliers"]
    )

    print(
        "Coverage        :",
        round(
            h["coverage"]
            *
            100,
            2
        ),
        "%"
    )

    print(
        "Scale           :",
        round(
            h["scale"],
            5
        )
    )

    print(
        "Rotation        :",
        round(
            h["rotation"],
            3
        ),
        "deg"
    )

    print(
        "Median residual :",
        round(
            h["median_residual"],
            3
        ),
        "px"
    )

    print(
        "Scale prior     :",
        round(
            h["scale_prior"],
            4
        )
    )

    print(
        "Rotation prior  :",
        round(
            h["rotation_prior"],
            4
        )
    )


# =========================================================
# SAVE INITIAL HYPOTHESIS TABLE
# =========================================================

csv_path = (
    f"{OUT}/roma_v7_hypotheses.csv"
)


with open(
    csv_path,
    "w",
    newline=""
) as csvfile:

    writer = csv.writer(
        csvfile
    )

    writer.writerow(
        [
            "rank",
            "source",
            "score",
            "data_score",
            "prior_score",
            "inliers",
            "coverage",
            "weighted_support",
            "median_residual",
            "scale",
            "rotation",
            "tx",
            "ty",
            "scale_prior",
            "rotation_prior"
        ]
    )


    for rank, h in enumerate(
        hypotheses,
        start=1
    ):

        writer.writerow(
            [
                rank,
                h["label"],
                h["score"],
                h["data_score"],
                h["prior_score"],
                h["inliers"],
                h["coverage"],
                h["weighted_support"],
                h["median_residual"],
                h["scale"],
                h["rotation"],
                h["tx"],
                h["ty"],
                h["scale_prior"],
                h["rotation_prior"]
            ]
        )


# =========================================================
# REFINE TOP HYPOTHESES
# =========================================================

print()
print(
    "===== REFINING TOP HYPOTHESES ====="
)


refined = []


for rank, h in enumerate(
    hypotheses[
        :min(
            REFINE_TOP_N,
            len(
                hypotheses
            )
        )
    ],
    start=1
):

    residual = residuals_for_transform(
        h["M"],
        ptsA,
        ptsB
    )


    mask = (
        residual
        <=
        HYPOTHESIS_INLIER_THRESHOLD
    )


    A = ptsA[
        mask
    ]

    B = ptsB[
        mask
    ]


    if (
        len(A)
        <
        6
    ):
        continue


    M_refined, ransac_mask = (
        cv2.estimateAffinePartial2D(
            A,
            B,

            method=cv2.RANSAC,

            ransacReprojThreshold=
                REFINE_RANSAC_THRESHOLD,

            maxIters=100000,

            confidence=0.9999,

            refineIters=50
        )
    )


    result = score_hypothesis(
        M_refined,
        f"refined_{rank}"
    )


    if (
        result
        is not None
    ):

        refined.append(
            result
        )


# Keep original hypotheses too
all_final_candidates = (
    hypotheses[:REFINE_TOP_N]
    +
    refined
)


all_final_candidates.sort(
    key=lambda x: x["score"],
    reverse=True
)


best = all_final_candidates[
    0
]


print()

print(
    "Refined hypotheses:",
    len(
        refined
    )
)


print()
print(
    "===== WINNING HYPOTHESIS ====="
)

print(
    "Source:",
    best["label"]
)

print(
    "Score:",
    round(
        best["score"],
        6
    )
)

print(
    "Data score:",
    round(
        best["data_score"],
        6
    )
)

print(
    "Prior score:",
    round(
        best["prior_score"],
        6
    )
)

print(
    "Inliers:",
    best["inliers"]
)

print(
    "Coverage:",
    round(
        best["coverage"]
        *
        100,
        2
    ),
    "%"
)

print(
    "Scale:",
    round(
        best["scale"],
        6
    )
)

print(
    "Rotation:",
    round(
        best["rotation"],
        4
    ),
    "deg"
)

print(
    "Translation:",
    round(
        best["tx"],
        3
    ),
    round(
        best["ty"],
        3
    )
)

print(
    "Median residual:",
    round(
        best["median_residual"],
        4
    ),
    "px"
)


# =========================================================
# FINAL CONSENSUS FROM WINNING MODEL
# =========================================================

best_residual = residuals_for_transform(
    best["M"],
    ptsA,
    ptsB
)


consensus_mask = (
    best_residual
    <=
    HYPOTHESIS_INLIER_THRESHOLD
)


consensus_A = ptsA[
    consensus_mask
]

consensus_B = ptsB[
    consensus_mask
]

consensus_conf = combined_conf[
    consensus_mask
]

consensus_cycle = cycle_error[
    consensus_mask
]

consensus_residual = best_residual[
    consensus_mask
]


print()
print(
    "===== WINNING CONSENSUS ====="
)

print(
    "Consensus matches:",
    len(
        consensus_A
    )
)


# =========================================================
# SPATIAL BALANCING
# =========================================================

quality = (
    consensus_conf
    *
    np.exp(
        -consensus_cycle
        /
        6.0
    )
    /
    (
        1.0
        +
        consensus_residual
    )
)


selected = []

available_cells = set()


for row in range(
    GRID
):

    for col in range(
        GRID
    ):

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
            (consensus_A[:, 0] >= xmin)
            &
            (consensus_A[:, 0] < xmax)
            &
            (consensus_A[:, 1] >= ymin)
            &
            (consensus_A[:, 1] < ymax)
        )


        idx = np.where(
            cell_mask
        )[0]


        if (
            len(idx)
            ==
            0
        ):
            continue


        available_cells.add(
            (
                row,
                col
            )
        )


        idx = idx[
            np.argsort(
                quality[
                    idx
                ]
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


if (
    len(selected)
    <
    3
):

    raise RuntimeError(
        "Too few spatially balanced correspondences."
    )


balanced_A = consensus_A[
    selected
]

balanced_B = consensus_B[
    selected
]


print()

print(
    "Available consensus cells:",
    len(
        available_cells
    ),
    "/",
    GRID * GRID
)

print(
    "Consensus coverage:",
    round(
        len(
            available_cells
        )
        /
        (
            GRID
            *
            GRID
        )
        *
        100,
        2
    ),
    "%"
)

print(
    "Balanced correspondences:",
    len(
        balanced_A
    )
)


# =========================================================
# FINAL RANSAC
# =========================================================

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
        "Final V7 RANSAC failed."
    )


# =========================================================
# FINAL RESIDUAL AGAINST ALL MATCHES
# =========================================================

final_all_residual = residuals_for_transform(
    M_final,
    ptsA,
    ptsB
)


final_all_mask = (
    final_all_residual
    <=
    FINAL_RANSAC_THRESHOLD
)


final_A = ptsA[
    final_all_mask
]

final_B = ptsB[
    final_all_mask
]

final_conf = combined_conf[
    final_all_mask
]

final_cycle = cycle_error[
    final_all_mask
]

final_residual = final_all_residual[
    final_all_mask
]


final_scale, final_rotation, final_tx, final_ty = (
    transform_parameters(
        M_final
    )
)


final_rmse = float(
    np.sqrt(
        np.mean(
            final_residual
            **
            2
        )
    )
)


final_coverage, final_cells = (
    coverage_from_points(
        final_A,
        W_A,
        H_A,
        GRID
    )
)


scale_error_percent = (
    abs(
        final_scale
        -
        expected_scale
    )
    /
    expected_scale
    *
    100.0
)


print()
print(
    "===== ROMA V7 FINAL RESULTS ====="
)

print(
    "Final inliers       :",
    len(
        final_A
    )
)

print(
    "Final inlier ratio  :",
    round(
        len(
            final_A
        )
        /
        len(
            ptsA
        ),
        4
    )
)

print(
    "RMSE                :",
    round(
        final_rmse,
        4
    ),
    "px"
)

print(
    "Scale               :",
    round(
        final_scale,
        6
    )
)

print(
    "Expected scale      :",
    round(
        expected_scale,
        6
    )
)

print(
    "Scale error         :",
    round(
        scale_error_percent,
        2
    ),
    "%"
)

print(
    "Rotation            :",
    round(
        final_rotation,
        4
    ),
    "deg"
)

print(
    "Translation         :",
    round(
        final_tx,
        3
    ),
    round(
        final_ty,
        3
    )
)

print(
    "Final coverage      :",
    round(
        final_coverage
        *
        100,
        2
    ),
    "%"
)

print(
    "Median confidence   :",
    round(
        float(
            np.median(
                final_conf
            )
        ),
        4
    )
)

print(
    "Median cycle error  :",
    round(
        float(
            np.median(
                final_cycle
            )
        ),
        4
    ),
    "px"
)


# =========================================================
# QUALITY GATE
# =========================================================

passed = (
    len(
        final_A
    )
    >=
    30
    and
    final_rmse
    <=
    5.0
    and
    final_coverage
    >=
    0.25
    and
    scale_error_percent
    <=
    8.0
    and
    abs(
        wrap_angle_deg(
            final_rotation
            -
            EXPECTED_ROTATION_DEG
        )
    )
    <=
    5.0
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


# =========================================================
# GEOREFERENCE EVALUATION
#
# IMPORTANT:
#
# Everything above is already finished.
#
# GeoTIFF is used ONLY now.
#
# It does NOT influence the winning hypothesis.
# =========================================================

print()
print(
    "===== INDEPENDENT GEOREFERENCE EVALUATION ====="
)


try:

    with rasterio.open(
        CH_TIF
    ) as ch_ds:

        ch_transform = (
            ch_ds.transform
        )


    with rasterio.open(
        LRO_TIF
    ) as lro_ds:

        lro_transform = (
            lro_ds.transform
        )


    cols = ptsA[
        :,
        0
    ]

    rows = ptsA[
        :,
        1
    ]


    lon = (
        ch_transform.a
        *
        cols
        +
        ch_transform.b
        *
        rows
        +
        ch_transform.c
    )


    lat = (
        ch_transform.d
        *
        cols
        +
        ch_transform.e
        *
        rows
        +
        ch_transform.f
    )


    x_lro = (
        MOON_RADIUS
        *
        np.radians(
            lon
        )
        *
        math.cos(
            STANDARD_PARALLEL
        )
    )


    y_lro = (
        MOON_RADIUS
        *
        np.radians(
            lat
        )
    )


    inv = ~lro_transform


    expected_x = (
        inv.a
        *
        x_lro
        +
        inv.b
        *
        y_lro
        +
        inv.c
    )


    expected_y = (
        inv.d
        *
        x_lro
        +
        inv.e
        *
        y_lro
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


    # -----------------------------------------------------
    # Evaluate actual RoMa correspondences selected by V7
    # -----------------------------------------------------

    selected_eval_mask = (
        inside
        &
        final_all_mask
    )


    geo_match_error = np.linalg.norm(
        ptsB[
            selected_eval_mask
        ]
        -
        expected_B[
            selected_eval_mask
        ],
        axis=1
    )


    print()

    print(
        "Selected V7 matches evaluated:",
        len(
            geo_match_error
        )
    )


    if (
        len(
            geo_match_error
        )
        >
        0
    ):

        print(
            "Match median geo error:",
            round(
                float(
                    np.median(
                        geo_match_error
                    )
                ),
                3
            ),
            "px"
        )


        for threshold in [
            2,
            5,
            10,
            20
        ]:

            count = int(
                np.sum(
                    geo_match_error
                    <=
                    threshold
                )
            )

            percentage = (
                count
                /
                len(
                    geo_match_error
                )
                *
                100
            )


            print(
                f"<= {threshold:2d} px : "
                f"{count:4d} / "
                f"{len(geo_match_error)} "
                f"({percentage:.2f}%)"
            )


    # -----------------------------------------------------
    # Evaluate the V7 transform itself
    #
    # Compare where V7 predicts every Chandrayaan point
    # should land versus the independent GeoTIFF location.
    # -----------------------------------------------------

    transformed_A = apply_transform(
        M_final,
        ptsA
    )


    transform_geo_error = np.linalg.norm(
        transformed_A[
            inside
        ]
        -
        expected_B[
            inside
        ],
        axis=1
    )


    print()

    print(
        "Transform absolute evaluation:"
    )

    print(
        "Median:",
        round(
            float(
                np.median(
                    transform_geo_error
                )
            ),
            3
        ),
        "px"
    )

    print(
        "Mean  :",
        round(
            float(
                np.mean(
                    transform_geo_error
                )
            ),
            3
        ),
        "px"
    )

    print(
        "P90   :",
        round(
            float(
                np.percentile(
                    transform_geo_error,
                    90
                )
            ),
            3
        ),
        "px"
    )


except Exception as exc:

    print(
        "Georeference evaluation skipped:"
    )

    print(
        exc
    )


# =========================================================
# SAVE ARRAYS
# =========================================================

np.save(
    f"{OUT}/roma_v7_transform.npy",
    M_final
)

np.save(
    f"{OUT}/roma_v7_inlier_A.npy",
    final_A
)

np.save(
    f"{OUT}/roma_v7_inlier_B.npy",
    final_B
)

np.save(
    f"{OUT}/roma_v7_residual.npy",
    final_residual
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


sx = (
    W_B
    /
    W_A
)

sy = (
    H_B
    /
    H_A
)


draw_A = final_A.copy()

draw_A[
    :,
    0
] *= sx

draw_A[
    :,
    1
] *= sy


indices = np.arange(
    len(
        final_A
    )
)


if (
    len(
        indices
    )
    >
    250
):

    indices = rng.choice(
        indices,
        size=250,
        replace=False
    )


draw_rng = np.random.default_rng(
    123
)


for i in indices:

    ax = int(
        round(
            draw_A[
                i,
                0
            ]
        )
    )

    ay = int(
        round(
            draw_A[
                i,
                1
            ]
        )
    )

    bx = (
        int(
            round(
                final_B[
                    i,
                    0
                ]
            )
        )
        +
        W_B
    )

    by = int(
        round(
            final_B[
                i,
                1
            ]
        )
    )


    colour = tuple(
        int(v)
        for v in draw_rng.integers(
            70,
            256,
            size=3
        )
    )


    cv2.circle(
        canvas,
        (
            ax,
            ay
        ),
        2,
        colour,
        -1
    )

    cv2.circle(
        canvas,
        (
            bx,
            by
        ),
        2,
        colour,
        -1
    )

    cv2.line(
        canvas,
        (
            ax,
            ay
        ),
        (
            bx,
            by
        ),
        colour,
        1,
        cv2.LINE_AA
    )


cv2.imwrite(
    f"{OUT}/roma_v7_matches.png",
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
    ),
    flags=cv2.INTER_LINEAR
)


cv2.imwrite(
    f"{OUT}/roma_v7_registered.png",
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
    f"{OUT}/roma_v7_overlay.png",
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
    f"{OUT}/roma_v7_difference.png",
    difference
)


# =========================================================
# FINAL GRID VISUALIZATION
# =========================================================

grid_image = ch_vis.copy()


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
        round(
            i
            *
            cell_w
        )
    )

    cv2.line(
        grid_image,
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
        round(
            i
            *
            cell_h
        )
    )

    cv2.line(
        grid_image,
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
        grid_image,
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
    f"{OUT}/roma_v7_grid.png",
    grid_image
)


# =========================================================
# FINISH
# =========================================================

print()
print(
    "Saved:"
)

print(
    f"{OUT}/roma_v7_matches.png"
)

print(
    f"{OUT}/roma_v7_registered.png"
)

print(
    f"{OUT}/roma_v7_overlay.png"
)

print(
    f"{OUT}/roma_v7_difference.png"
)

print(
    f"{OUT}/roma_v7_grid.png"
)

print(
    f"{OUT}/roma_v7_hypotheses.csv"
)

print()
print(
    "Done."
)