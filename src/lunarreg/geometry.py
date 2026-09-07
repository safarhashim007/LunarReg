"""Geometry extracted from V9; no IO or model initialization."""
import math
import numpy as np
from .config import *
def wrap_angle_deg(angle):
    return (angle + 180.0) % 360.0 - 180.0

def transform_parameters(M):
    a = float(M[0, 0])
    c = float(M[1, 0])
    scale = math.sqrt(a * a + c * c)
    rotation = math.degrees(math.atan2(c, a))
    tx = float(M[0, 2])
    ty = float(M[1, 2])
    return (scale, rotation, tx, ty)

def make_similarity(scale, rotation_deg, tx, ty):
    theta = math.radians(rotation_deg)
    c = math.cos(theta)
    s = math.sin(theta)
    return np.array([[scale * c, -scale * s, tx], [scale * s, scale * c, ty]], dtype=np.float64)

def apply_transform(M, points):
    ones = np.ones((len(points), 1), dtype=np.float64)
    points_h = np.hstack([points.astype(np.float64), ones])
    return (M @ points_h.T).T

def residuals_for_transform(M, A, B):
    predicted = apply_transform(M, A)
    return np.linalg.norm(predicted - B, axis=1)

def coverage_from_points(points, width, height, grid):
    occupied = set()
    for x, y in points:
        col = min(max(int(x / width * grid), 0), grid - 1)
        row = min(max(int(y / height * grid), 0), grid - 1)
        occupied.add((row, col))
    coverage = len(occupied) / (grid * grid)
    return (coverage, occupied)

def similarity_from_two_points(A1, A2, B1, B2):
    vec_A = A2 - A1
    vec_B = B2 - B1
    dist_A = float(np.linalg.norm(vec_A))
    dist_B = float(np.linalg.norm(vec_B))
    if dist_A < MIN_PAIR_DISTANCE_A:
        return None
    if dist_B < MIN_PAIR_DISTANCE_B:
        return None
    scale = dist_B / dist_A
    if not np.isfinite(scale) or scale <= 0:
        return None
    angle_A = math.atan2(float(vec_A[1]), float(vec_A[0]))
    angle_B = math.atan2(float(vec_B[1]), float(vec_B[0]))
    theta = angle_B - angle_A
    rotation = wrap_angle_deg(math.degrees(theta))
    if abs(rotation) > MAX_ABS_ROTATION:
        return None
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    R = np.array([[scale * cos_t, -scale * sin_t], [scale * sin_t, scale * cos_t]], dtype=np.float64)
    translation = B1 - R @ A1
    M = np.zeros((2, 3), dtype=np.float64)
    M[:, :2] = R
    M[:, 2] = translation
    return M

def weighted_similarity_fit(A, B, weights):
    A = A.astype(np.float64)
    B = B.astype(np.float64)
    weights = weights.astype(np.float64)
    valid = np.isfinite(weights) & (weights > 0)
    A = A[valid]
    B = B[valid]
    weights = weights[valid]
    if len(A) < 3:
        return None
    weight_sum = float(np.sum(weights))
    if weight_sum <= 0:
        return None
    weights = weights / weight_sum
    mu_A = np.sum(A * weights[:, None], axis=0)
    mu_B = np.sum(B * weights[:, None], axis=0)
    X = A - mu_A
    Y = B - mu_B
    covariance = Y.T @ (X * weights[:, None])
    U, singular_values, Vt = np.linalg.svd(covariance)
    correction = np.eye(2)
    if np.linalg.det(U @ Vt) < 0:
        correction[1, 1] = -1.0
    R = U @ correction @ Vt
    variance_A = float(np.sum(weights * np.sum(X ** 2, axis=1)))
    if variance_A <= 1e-12:
        return None
    scale = float(np.sum(singular_values * np.diag(correction)) / variance_A)
    translation = mu_B - scale * R @ mu_A
    M = np.zeros((2, 3), dtype=np.float64)
    M[:, :2] = scale * R
    M[:, 2] = translation
    return M

def spatially_balance_weights(points, weights, width, height, grid):
    balanced = np.zeros_like(weights, dtype=np.float64)
    for row in range(grid):
        for col in range(grid):
            xmin = col / grid * width
            xmax = (col + 1) / grid * width
            ymin = row / grid * height
            ymax = (row + 1) / grid * height
            mask = (points[:, 0] >= xmin) & (points[:, 0] < xmax) & (points[:, 1] >= ymin) & (points[:, 1] < ymax)
            idx = np.where(mask)[0]
            if len(idx) == 0:
                continue
            cell_weights = weights[idx]
            cell_sum = float(np.sum(cell_weights))
            if cell_sum <= 0:
                continue
            balanced[idx] = cell_weights / cell_sum
    return balanced

def blend_similarity(old_M, new_M, alpha):
    old_scale, old_rot, old_tx, old_ty = transform_parameters(old_M)
    new_scale, new_rot, new_tx, new_ty = transform_parameters(new_M)
    rotation_delta = wrap_angle_deg(new_rot - old_rot)
    blended_rotation = old_rot + alpha * rotation_delta
    blended_scale = old_scale * (1.0 - alpha) + new_scale * alpha
    blended_tx = old_tx * (1.0 - alpha) + new_tx * alpha
    blended_ty = old_ty * (1.0 - alpha) + new_ty * alpha
    return make_similarity(blended_scale, blended_rotation, blended_tx, blended_ty)

def weighted_affine_fit(A, B, weights):
    """Weighted least-squares fit for B ~= L A + t; no RANSAC."""
    valid = np.isfinite(weights) & (weights > 0)
    A, B, weights = (A[valid], B[valid], weights[valid])
    if len(A) < 3 or float(np.sum(weights)) <= 0:
        return None
    X = np.column_stack([A.astype(np.float64), np.ones(len(A))])
    if np.linalg.matrix_rank(X) < 3 or not np.isfinite(B).all():
        return None
    sqrt_w = np.sqrt(weights.astype(np.float64))[:, None]
    try:
        params, _, _, _ = np.linalg.lstsq(X * sqrt_w, B.astype(np.float64) * sqrt_w, rcond=None)
    except np.linalg.LinAlgError:
        return None
    return params.T.astype(np.float64)

def affine_components(M):
    """Return scale axes, anisotropy, shear, and polar rotation."""
    L = M[:, :2].astype(np.float64)
    U, singular, Vt = np.linalg.svd(L)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    rotation = math.degrees(math.atan2(R[1, 0], R[0, 0]))
    scale_mean = float(np.mean(singular))
    anisotropy = float(singular[0] / max(singular[1], 1e-12) - 1.0)
    symmetric = R.T @ L / max(scale_mean, 1e-12)
    shear = float(0.5 * (symmetric[0, 1] + symmetric[1, 0]))
    return (float(singular[0]), float(singular[1]), anisotropy, shear, rotation)

def constrain_affine(candidate, coarse_M):
    """Reject candidates outside the micro-refinement envelope."""
    if not np.isfinite(candidate).all() or np.linalg.det(candidate[:, :2]) <= 0:
        return None
    sx, sy, anisotropy, shear, rotation = affine_components(candidate)
    _, _, _, _, coarse_rotation = affine_components(coarse_M)
    translation_change = float(np.linalg.norm(candidate[:, 2] - coarse_M[:, 2]))
    if not np.isfinite([sx, sy, anisotropy, shear, rotation, translation_change]).all():
        return None
    if anisotropy > MAX_AFFINE_ANISOTROPY:
        return None
    if abs(shear) > MAX_AFFINE_SHEAR:
        return None
    if abs(wrap_angle_deg(rotation - coarse_rotation)) > MAX_AFFINE_ROTATION_CHANGE:
        return None
    if translation_change > MAX_AFFINE_TRANSLATION_CHANGE:
        return None
    return candidate

def blend_affine(old_M, new_M, alpha):
    """Damped affine update; constraints are checked again after blending."""
    return old_M * (1.0 - alpha) + new_M * alpha
