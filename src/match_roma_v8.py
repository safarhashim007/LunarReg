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
# LUNARREG V8
#
# Multi-hypothesis selection
# +
# LOCKED robust weighted refinement
#
# NO unrestricted final RANSAC.
#
# Pipeline:
#
# RoMa Fast + Bidirectional
#       ↓
# Forward/backward cycle consistency
#       ↓
# Top cycle candidates
#       ↓
# Thousands of similarity hypotheses
#       ↓
# Score using:
#   - geometric support
#   - confidence
#   - cycle consistency
#   - spatial coverage
#   - scale prior
#   - orientation prior
#       ↓
# Choose best geometric mode
#       ↓
# LOCK MODE
#       ↓
# iterative weighted refinement
#   20px → 16px → 12px → 10px → 8px
#       ↓
# spatial cell weighting
#       ↓
# final similarity transform
#
# GeoTIFF is ONLY used after registration
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


# Initial hypothesis scoring
HYPOTHESIS_INLIER_THRESHOLD = 8.0
MIN_HYPOTHESIS_INLIERS = 12


# ---------------------------------------------------------
# LOCKED refinement thresholds
# ---------------------------------------------------------

REFINEMENT_THRESHOLDS = [
    20.0,
    16.0,
    12.0,
    10.0,
    8.0
]


# Spatial grid
GRID = 8


# ---------------------------------------------------------
# Physical metadata priors
#
# Pair consists of approximately north-up orthorectified
# images covering the same AOI.
#
# These are soft priors.
# ---------------------------------------------------------

EXPECTED_ROTATION_DEG = 0.0

ROTATION_PRIOR_SIGMA_DEG = 5.0

SCALE_PRIOR_SIGMA_FRACTION = 0.06


# Broad sanity bounds
MIN_SCALE = 0.15
MAX_SCALE = 0.40

MAX_ABS_ROTATION = 35.0


# Minimal-pair stability
MIN_PAIR_DISTANCE_A = 100.0
MIN_PAIR_DISTANCE_B = 15.0


# ---------------------------------------------------------
# Mode locking
#
# Once V8 finds the winning hypothesis, weighted
# refinement is not allowed to jump to a completely
# different geometric mode.
# ---------------------------------------------------------

MAX_SCALE_CHANGE_FROM_INITIAL = 0.06

MAX_ROTATION_CHANGE_FROM_INITIAL = 4.0

MAX_TRANSLATION_STEP = 20.0


# Blend new weighted solution with previous transform
REFINEMENT_ALPHA = 0.70


# Final reported residual thresholds
STRICT_THRESHOLD = 5.0
RELAXED_THRESHOLD = 8.0


# Moon projection
# Evaluation only
MOON_RADIUS = 1737400.0

STANDARD_PARALLEL = math.radians(
    10.0
)


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
# HELPER FUNCTIONS
# =========================================================

def wrap_angle_deg(angle):

    return (
        (angle + 180.0)
        % 360.0
        -
        180.0
    )


# ---------------------------------------------------------
# Extract similarity transform parameters
# ---------------------------------------------------------

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


# ---------------------------------------------------------
# Build similarity transform matrix
# ---------------------------------------------------------

def make_similarity(
    scale,
    rotation_deg,
    tx,
    ty
):

    theta = math.radians(
        rotation_deg
    )

    c = math.cos(
        theta
    )

    s = math.sin(
        theta
    )

    return np.array(
        [
            [
                scale * c,
                -scale * s,
                tx
            ],
            [
                scale * s,
                scale * c,
                ty
            ]
        ],
        dtype=np.float64
    )


# ---------------------------------------------------------
# Apply affine/similarity transform
# ---------------------------------------------------------

def apply_transform(
    M,
    points
):

    ones = np.ones(
        (
            len(points),
            1
        ),
        dtype=np.float64
    )

    points_h = np.hstack(
        [
            points.astype(
                np.float64
            ),
            ones
        ]
    )

    return (
        M
        @
        points_h.T
    ).T


# ---------------------------------------------------------
# Residuals
# ---------------------------------------------------------

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


# ---------------------------------------------------------
# Grid coverage
# ---------------------------------------------------------

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
        len(
            occupied
        )
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


# ---------------------------------------------------------
# Two-point similarity hypothesis
# ---------------------------------------------------------

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


    rotation = wrap_angle_deg(
        math.degrees(
            theta
        )
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
# WEIGHTED SIMILARITY FIT
#
# Weighted Umeyama-style solution.
#
# Finds:
#
# B ≈ scale * R * A + translation
#
# without RANSAC.
# =========================================================

def weighted_similarity_fit(
    A,
    B,
    weights
):

    A = A.astype(
        np.float64
    )

    B = B.astype(
        np.float64
    )

    weights = weights.astype(
        np.float64
    )


    valid = (
        np.isfinite(
            weights
        )
        &
        (weights > 0)
    )


    A = A[
        valid
    ]

    B = B[
        valid
    ]

    weights = weights[
        valid
    ]


    if len(A) < 3:
        return None


    weight_sum = float(
        np.sum(
            weights
        )
    )


    if weight_sum <= 0:
        return None


    weights = (
        weights
        /
        weight_sum
    )


    mu_A = np.sum(
        A
        *
        weights[:, None],
        axis=0
    )


    mu_B = np.sum(
        B
        *
        weights[:, None],
        axis=0
    )


    X = (
        A
        -
        mu_A
    )

    Y = (
        B
        -
        mu_B
    )


    # Covariance mapping X -> Y
    covariance = (
        Y.T
        @
        (
            X
            *
            weights[:, None]
        )
    )


    U, singular_values, Vt = (
        np.linalg.svd(
            covariance
        )
    )


    correction = np.eye(
        2
    )


    if (
        np.linalg.det(
            U @ Vt
        )
        <
        0
    ):

        correction[
            1,
            1
        ] = -1.0


    R = (
        U
        @
        correction
        @
        Vt
    )


    variance_A = float(
        np.sum(
            weights
            *
            np.sum(
                X ** 2,
                axis=1
            )
        )
    )


    if variance_A <= 1e-12:
        return None


    scale = float(
        np.sum(
            singular_values
            *
            np.diag(
                correction
            )
        )
        /
        variance_A
    )


    translation = (
        mu_B
        -
        scale
        *
        R
        @
        mu_A
    )


    M = np.zeros(
        (
            2,
            3
        ),
        dtype=np.float64
    )


    M[
        :,
        :2
    ] = (
        scale
        *
        R
    )


    M[
        :,
        2
    ] = translation


    return M


# =========================================================
# CELL-BALANCED WEIGHTS
#
# Each occupied spatial grid cell receives approximately
# equal total influence.
# =========================================================

def spatially_balance_weights(
    points,
    weights,
    width,
    height,
    grid
):

    balanced = np.zeros_like(
        weights,
        dtype=np.float64
    )


    for row in range(
        grid
    ):

        for col in range(
            grid
        ):

            xmin = (
                col
                /
                grid
                *
                width
            )

            xmax = (
                (col + 1)
                /
                grid
                *
                width
            )

            ymin = (
                row
                /
                grid
                *
                height
            )

            ymax = (
                (row + 1)
                /
                grid
                *
                height
            )


            mask = (
                (points[:, 0] >= xmin)
                &
                (points[:, 0] < xmax)
                &
                (points[:, 1] >= ymin)
                &
                (points[:, 1] < ymax)
            )


            idx = np.where(
                mask
            )[0]


            if len(idx) == 0:
                continue


            cell_weights = weights[
                idx
            ]


            cell_sum = float(
                np.sum(
                    cell_weights
                )
            )


            if cell_sum <= 0:
                continue


            # Cell total becomes ~1.
            balanced[
                idx
            ] = (
                cell_weights
                /
                cell_sum
            )


    return balanced


# =========================================================
# BLEND TWO SIMILARITY TRANSFORMS
# =========================================================

def blend_similarity(
    old_M,
    new_M,
    alpha
):

    old_scale, old_rot, old_tx, old_ty = (
        transform_parameters(
            old_M
        )
    )

    new_scale, new_rot, new_tx, new_ty = (
        transform_parameters(
            new_M
        )
    )


    rotation_delta = wrap_angle_deg(
        new_rot
        -
        old_rot
    )


    blended_rotation = (
        old_rot
        +
        alpha
        *
        rotation_delta
    )


    blended_scale = (
        old_scale
        *
        (1.0 - alpha)
        +
        new_scale
        *
        alpha
    )


    blended_tx = (
        old_tx
        *
        (1.0 - alpha)
        +
        new_tx
        *
        alpha
    )


    blended_ty = (
        old_ty
        *
        (1.0 - alpha)
        +
        new_ty
        *
        alpha
    )


    return make_similarity(
        blended_scale,
        blended_rotation,
        blended_tx,
        blended_ty
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
    "===== LUNARREG ROMA V8 ====="
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
# ROMA
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
# BIDIRECTIONAL MATCHING
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


if predictions[
    "warp_BA"
] is None:

    raise RuntimeError(
        "Backward RoMa warp missing."
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
    len(
        ptsA
    )
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
# BACKWARD POINT -> A PIXELS
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
# CYCLE METRICS
# =========================================================

cycle_error_A = np.linalg.norm(
    back_A
    -
    ptsA,
    axis=1
)


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
# VALIDITY
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
    len(
        ptsA
    )
)


# =========================================================
# BASE IMAGE-ONLY QUALITY WEIGHT
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
    # Rotation prior
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
    # Data score
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
        )
    }


# =========================================================
# CANDIDATE POOL
# =========================================================

order = np.argsort(
    cycle_score
)[::-1]


candidate_count = min(
    CANDIDATE_TOP_K,
    len(
        order
    )
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
# Deterministic RANSAC hypotheses
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
        candidate_A[
            :k
        ],
        candidate_B[
            :k
        ],

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


    if result is not None:

        hypotheses.append(
            result
        )


# ---------------------------------------------------------
# Random pair hypotheses
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
# Random subset RANSAC
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


    idx = rng.choice(
        len(
            candidate_A
        ),
        size=count,
        replace=False
    )


    M, mask = cv2.estimateAffinePartial2D(
        candidate_A[
            idx
        ],
        candidate_B[
            idx
        ],

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


if len(
    hypotheses
) == 0:

    raise RuntimeError(
        "No valid V8 hypotheses generated."
    )


# =========================================================
# SORT HYPOTHESES
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
            6
        )
    )

    print(
        "Rotation        :",
        round(
            h["rotation"],
            4
        ),
        "deg"
    )

    print(
        "Translation     :",
        round(
            h["tx"],
            3
        ),
        round(
            h["ty"],
            3
        )
    )

    print(
        "Median residual :",
        round(
            h["median_residual"],
            3
        ),
        "px"
    )


# =========================================================
# WINNER
# =========================================================

best = hypotheses[
    0
]


M_initial = best[
    "M"
].copy()


initial_scale, initial_rotation, initial_tx, initial_ty = (
    transform_parameters(
        M_initial
    )
)


print()
print(
    "===== INITIAL WINNING MODE ====="
)

print(
    "Score:",
    round(
        best["score"],
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
        initial_scale,
        6
    )
)

print(
    "Rotation:",
    round(
        initial_rotation,
        4
    ),
    "deg"
)

print(
    "Translation:",
    round(
        initial_tx,
        3
    ),
    round(
        initial_ty,
        3
    )
)


# =========================================================
# SAVE HYPOTHESES
# =========================================================

csv_path = (
    f"{OUT}/roma_v8_hypotheses.csv"
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
            "ty"
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
                h["ty"]
            ]
        )


# =========================================================
# LOCKED ROBUST REFINEMENT
# =========================================================

print()
print(
    "===== LOCKED ROBUST REFINEMENT ====="
)


M_current = (
    M_initial.copy()
)


refinement_log = []


for iteration, threshold in enumerate(
    REFINEMENT_THRESHOLDS,
    start=1
):

    residual = residuals_for_transform(
        M_current,
        ptsA,
        ptsB
    )


    consensus = (
        residual
        <=
        threshold
    )


    A = ptsA[
        consensus
    ]

    B = ptsB[
        consensus
    ]

    current_conf = combined_conf[
        consensus
    ]

    current_cycle = cycle_error[
        consensus
    ]

    current_residual = residual[
        consensus
    ]


    if len(A) < 10:

        print(
            "Too few points at threshold",
            threshold
        )

        break


    # -----------------------------------------------------
    # Tukey-like robust residual weighting
    #
    # Points close to transform receive much more weight.
    # -----------------------------------------------------

    u = (
        current_residual
        /
        threshold
    )


    robust_weight = (
        1.0
        -
        u ** 2
    )


    robust_weight = np.clip(
        robust_weight,
        0.0,
        1.0
    )


    robust_weight = (
        robust_weight
        **
        2
    )


    # -----------------------------------------------------
    # Image-only confidence/cycle weighting
    # -----------------------------------------------------

    image_weight = (
        current_conf
        *
        np.exp(
            -current_cycle
            /
            6.0
        )
    )


    weights = (
        image_weight
        *
        robust_weight
    )


    # -----------------------------------------------------
    # Equalize spatial influence
    # -----------------------------------------------------

    weights = spatially_balance_weights(
        A,
        weights,
        W_A,
        H_A,
        GRID
    )


    if (
        np.sum(
            weights
        )
        <=
        0
    ):

        print(
            "Invalid weights."
        )

        break


    M_candidate = weighted_similarity_fit(
        A,
        B,
        weights
    )


    if M_candidate is None:

        print(
            "Weighted fit failed."
        )

        break


    candidate_scale, candidate_rotation, candidate_tx, candidate_ty = (
        transform_parameters(
            M_candidate
        )
    )


    # =====================================================
    # MODE LOCK
    # =====================================================

    scale_change = (
        abs(
            candidate_scale
            -
            initial_scale
        )
        /
        initial_scale
    )


    rotation_change = abs(
        wrap_angle_deg(
            candidate_rotation
            -
            initial_rotation
        )
    )


    current_scale, current_rotation, current_tx, current_ty = (
        transform_parameters(
            M_current
        )
    )


    translation_step = math.sqrt(
        (
            candidate_tx
            -
            current_tx
        ) ** 2
        +
        (
            candidate_ty
            -
            current_ty
        ) ** 2
    )


    accepted = True


    if (
        scale_change
        >
        MAX_SCALE_CHANGE_FROM_INITIAL
    ):

        accepted = False


    if (
        rotation_change
        >
        MAX_ROTATION_CHANGE_FROM_INITIAL
    ):

        accepted = False


    if (
        translation_step
        >
        MAX_TRANSLATION_STEP
    ):

        accepted = False


    if accepted:

        M_new = blend_similarity(
            M_current,
            M_candidate,
            REFINEMENT_ALPHA
        )

    else:

        M_new = (
            M_current.copy()
        )


    new_scale, new_rotation, new_tx, new_ty = (
        transform_parameters(
            M_new
        )
    )


    new_residual = residuals_for_transform(
        M_new,
        ptsA,
        ptsB
    )


    new_mask = (
        new_residual
        <=
        threshold
    )


    coverage, cells = coverage_from_points(
        ptsA[
            new_mask
        ],
        W_A,
        H_A,
        GRID
    )


    median_res = float(
        np.median(
            new_residual[
                new_mask
            ]
        )
    )


    print()

    print(
        f"Iteration {iteration}"
    )

    print(
        "Threshold      :",
        threshold,
        "px"
    )

    print(
        "Consensus      :",
        int(
            new_mask.sum()
        )
    )

    print(
        "Coverage       :",
        round(
            coverage
            *
            100,
            2
        ),
        "%"
    )

    print(
        "Accepted update:",
        accepted
    )

    print(
        "Scale          :",
        round(
            new_scale,
            6
        )
    )

    print(
        "Rotation       :",
        round(
            new_rotation,
            4
        ),
        "deg"
    )

    print(
        "Translation    :",
        round(
            new_tx,
            3
        ),
        round(
            new_ty,
            3
        )
    )

    print(
        "Median residual:",
        round(
            median_res,
            3
        ),
        "px"
    )


    refinement_log.append(
        [
            iteration,
            threshold,
            int(
                new_mask.sum()
            ),
            coverage,
            accepted,
            new_scale,
            new_rotation,
            new_tx,
            new_ty,
            median_res
        ]
    )


    M_current = (
        M_new
    )


# =========================================================
# FINAL V8 TRANSFORM
# =========================================================

M_final = (
    M_current.copy()
)


final_scale, final_rotation, final_tx, final_ty = (
    transform_parameters(
        M_final
    )
)


all_residual = residuals_for_transform(
    M_final,
    ptsA,
    ptsB
)


# =========================================================
# STRICT INLIERS <=5PX
# =========================================================

strict_mask = (
    all_residual
    <=
    STRICT_THRESHOLD
)


strict_A = ptsA[
    strict_mask
]

strict_B = ptsB[
    strict_mask
]

strict_residual = all_residual[
    strict_mask
]


strict_coverage, strict_cells = (
    coverage_from_points(
        strict_A,
        W_A,
        H_A,
        GRID
    )
)


if len(
    strict_residual
) > 0:

    strict_rmse = float(
        np.sqrt(
            np.mean(
                strict_residual
                **
                2
            )
        )
    )

else:

    strict_rmse = float(
        "inf"
    )


# =========================================================
# RELAXED INLIERS <=8PX
# =========================================================

relaxed_mask = (
    all_residual
    <=
    RELAXED_THRESHOLD
)


relaxed_A = ptsA[
    relaxed_mask
]

relaxed_B = ptsB[
    relaxed_mask
]

relaxed_residual = all_residual[
    relaxed_mask
]


relaxed_coverage, relaxed_cells = (
    coverage_from_points(
        relaxed_A,
        W_A,
        H_A,
        GRID
    )
)


if len(
    relaxed_residual
) > 0:

    relaxed_rmse = float(
        np.sqrt(
            np.mean(
                relaxed_residual
                **
                2
            )
        )
    )

else:

    relaxed_rmse = float(
        "inf"
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


# =========================================================
# FINAL RESULTS
# =========================================================

print()
print(
    "===== ROMA V8 FINAL RESULTS ====="
)

print()

print(
    "Initial scale       :",
    round(
        initial_scale,
        6
    )
)

print(
    "Final scale         :",
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

print()

print(
    "Initial rotation    :",
    round(
        initial_rotation,
        4
    ),
    "deg"
)

print(
    "Final rotation      :",
    round(
        final_rotation,
        4
    ),
    "deg"
)

print()

print(
    "Initial translation :",
    round(
        initial_tx,
        3
    ),
    round(
        initial_ty,
        3
    )
)

print(
    "Final translation   :",
    round(
        final_tx,
        3
    ),
    round(
        final_ty,
        3
    )
)


print()
print(
    "--- STRICT <=5PX CONSENSUS ---"
)

print(
    "Inliers             :",
    len(
        strict_A
    )
)

print(
    "Inlier ratio        :",
    round(
        len(
            strict_A
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
        strict_rmse,
        4
    ),
    "px"
)

print(
    "Coverage            :",
    round(
        strict_coverage
        *
        100,
        2
    ),
    "%"
)


print()
print(
    "--- RELAXED <=8PX CONSENSUS ---"
)

print(
    "Inliers             :",
    len(
        relaxed_A
    )
)

print(
    "Inlier ratio        :",
    round(
        len(
            relaxed_A
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
        relaxed_rmse,
        4
    ),
    "px"
)

print(
    "Coverage            :",
    round(
        relaxed_coverage
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
    len(
        strict_A
    )
    >=
    30
    and
    strict_rmse
    <=
    5.0
    and
    relaxed_coverage
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
# INDEPENDENT GEOREFERENCE EVALUATION
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


    inv = (
        ~lro_transform
    )


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
    # Evaluate selected correspondences
    # -----------------------------------------------------

    selected_eval = (
        inside
        &
        relaxed_mask
    )


    selected_geo_error = np.linalg.norm(
        ptsB[
            selected_eval
        ]
        -
        expected_B[
            selected_eval
        ],
        axis=1
    )


    print()

    print(
        "Selected V8 matches:",
        len(
            selected_geo_error
        )
    )


    if len(
        selected_geo_error
    ) > 0:

        print(
            "Median match geo error:",
            round(
                float(
                    np.median(
                        selected_geo_error
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
                    selected_geo_error
                    <=
                    threshold
                )
            )


            percentage = (
                count
                /
                len(
                    selected_geo_error
                )
                *
                100
            )


            print(
                f"<= {threshold:2d} px : "
                f"{count:4d} / "
                f"{len(selected_geo_error)} "
                f"({percentage:.2f}%)"
            )


    # -----------------------------------------------------
    # Evaluate transform itself
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
        "Minimum:",
        round(
            float(
                np.min(
                    transform_geo_error
                )
            ),
            3
        ),
        "px"
    )

    print(
        "Median :",
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
        "Mean   :",
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
        "P90    :",
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
# SAVE REFINEMENT LOG
# =========================================================

with open(
    f"{OUT}/roma_v8_refinement.csv",
    "w",
    newline=""
) as csvfile:

    writer = csv.writer(
        csvfile
    )

    writer.writerow(
        [
            "iteration",
            "threshold",
            "consensus",
            "coverage",
            "accepted",
            "scale",
            "rotation",
            "tx",
            "ty",
            "median_residual"
        ]
    )


    writer.writerows(
        refinement_log
    )


# =========================================================
# SAVE ARRAYS
# =========================================================

np.save(
    f"{OUT}/roma_v8_transform.npy",
    M_final
)

np.save(
    f"{OUT}/roma_v8_strict_A.npy",
    strict_A
)

np.save(
    f"{OUT}/roma_v8_strict_B.npy",
    strict_B
)

np.save(
    f"{OUT}/roma_v8_relaxed_A.npy",
    relaxed_A
)

np.save(
    f"{OUT}/roma_v8_relaxed_B.npy",
    relaxed_B
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


draw_A = relaxed_A.copy()

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
        relaxed_A
    )
)


if len(
    indices
) > 250:

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
                relaxed_B[
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
            relaxed_B[
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
    f"{OUT}/roma_v8_matches.png",
    canvas
)


# =========================================================
# REGISTER CHANDRAYAAN
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
    f"{OUT}/roma_v8_registered.png",
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
    f"{OUT}/roma_v8_overlay.png",
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
    f"{OUT}/roma_v8_difference.png",
    difference
)


# =========================================================
# GRID VISUALIZATION
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


for row, col in relaxed_cells:

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
    f"{OUT}/roma_v8_grid.png",
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
    f"{OUT}/roma_v8_matches.png"
)

print(
    f"{OUT}/roma_v8_registered.png"
)

print(
    f"{OUT}/roma_v8_overlay.png"
)

print(
    f"{OUT}/roma_v8_difference.png"
)

print(
    f"{OUT}/roma_v8_grid.png"
)

print(
    f"{OUT}/roma_v8_hypotheses.csv"
)

print(
    f"{OUT}/roma_v8_refinement.csv"
)

print()
print(
    "Done."
)