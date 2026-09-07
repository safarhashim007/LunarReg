"""V9 multi-hypothesis selection and locked global refinement; image-only."""
import csv
import math
import cv2
import numpy as np
from .config import *
from .geometry import *
def solve(ptsA, ptsB, combined_conf, cycle_error, source_shape, reference_shape, OUT, seed=42, expected_scale=None, rotation=0.0):
    H_A, W_A = source_shape
    H_B, W_B = reference_shape
    if expected_scale is not None and (not np.isfinite(expected_scale) or expected_scale <= 0):
        raise ValueError("Expected scale must be finite and positive")
    if not np.isfinite(rotation): raise ValueError("Expected rotation must be finite")
    if not all(np.isfinite(x).all() for x in (ptsA,ptsB,combined_conf,cycle_error)):
        raise ValueError("Nonfinite correspondences")
    expected_scale = expected_scale or (W_B / W_A + H_B / H_A) / 2
    scale_prior_sigma = expected_scale * SCALE_PRIOR_SIGMA_FRACTION
    MIN_SCALE = expected_scale * 0.6
    MAX_SCALE = expected_scale * 1.6
    EXPECTED_ROTATION_DEG = rotation
    rng = np.random.default_rng(seed)
    cv2.setRNGSeed(seed)
    cycle_score = combined_conf / (1 + cycle_error)
    if len(ptsA) < 30:
        raise ValueError('Fewer than 30 valid correspondences')
    base_weight = combined_conf * np.exp(-cycle_error / 6.0)
    weight_p95 = max(float(np.percentile(base_weight, 95)), 1e-08)
    total_weight = max(float(np.sum(base_weight)), 1e-08)

    def score_hypothesis(M, label):
        if M is None:
            return None
        scale, rotation, tx, ty = transform_parameters(M)
        if not np.isfinite(scale):
            return None
        if scale < MIN_SCALE or scale > MAX_SCALE:
            return None
        rotation_delta = wrap_angle_deg(rotation - EXPECTED_ROTATION_DEG)
        if abs(rotation_delta) > MAX_ABS_ROTATION:
            return None
        residual = residuals_for_transform(M, ptsA, ptsB)
        inlier_mask = residual <= HYPOTHESIS_INLIER_THRESHOLD
        inlier_count = int(inlier_mask.sum())
        if inlier_count < MIN_HYPOTHESIS_INLIERS:
            return None
        inlier_A = ptsA[inlier_mask]
        coverage, cells = coverage_from_points(inlier_A, W_A, H_A, GRID)
        weighted_support = float(np.sum(base_weight[inlier_mask])) / total_weight
        median_residual = float(np.median(residual[inlier_mask]))
        residual_score = math.exp(-median_residual / 3.0)
        inlier_quality = float(np.mean(base_weight[inlier_mask])) / weight_p95
        inlier_quality = float(np.clip(inlier_quality, 0.0, 1.0))
        scale_z = (scale - expected_scale) / scale_prior_sigma
        scale_prior = math.exp(-0.5 * scale_z * scale_z)
        rot_z = rotation_delta / ROTATION_PRIOR_SIGMA_DEG
        rotation_prior = math.exp(-0.5 * rot_z * rot_z)
        data_score = 0.4 * weighted_support + 0.25 * coverage + 0.2 * residual_score + 0.15 * inlier_quality
        prior_score = math.sqrt(scale_prior * rotation_prior)
        final_score = data_score * (0.3 + 0.7 * prior_score)
        return {'label': label, 'M': M.astype(np.float64), 'score': float(final_score), 'data_score': float(data_score), 'prior_score': float(prior_score), 'weighted_support': float(weighted_support), 'coverage': float(coverage), 'inliers': inlier_count, 'median_residual': float(median_residual), 'scale': float(scale), 'rotation': float(rotation), 'tx': float(tx), 'ty': float(ty), 'scale_prior': float(scale_prior), 'rotation_prior': float(rotation_prior)}
    order = np.argsort(cycle_score)[::-1]
    candidate_count = min(CANDIDATE_TOP_K, len(order))
    candidate_indices = order[:candidate_count]
    candidate_A = ptsA[candidate_indices]
    candidate_B = ptsB[candidate_indices]
    print()
    print('===== CANDIDATE POOL =====')
    print('Candidates:', len(candidate_A))
    print('Median cycle error:', round(float(np.median(cycle_error[candidate_indices])), 3), 'px')
    print('Median confidence:', round(float(np.median(combined_conf[candidate_indices])), 4))
    print()
    print('===== GENERATING HYPOTHESES =====')
    hypotheses = []
    for k in [30, 50, 75, 100, 150, 200, 300]:
        if k > len(candidate_A):
            continue
        M, mask = cv2.estimateAffinePartial2D(candidate_A[:k], candidate_B[:k], method=cv2.RANSAC, ransacReprojThreshold=8.0, maxIters=100000, confidence=0.9999, refineIters=30)
        result = score_hypothesis(M, f'top{k}_ransac')
        if result is not None:
            hypotheses.append(result)
    pair_success = 0
    for n in range(PAIR_HYPOTHESES):
        i, j = rng.choice(len(candidate_A), size=2, replace=False)
        M = similarity_from_two_points(candidate_A[i], candidate_A[j], candidate_B[i], candidate_B[j])
        if M is None:
            continue
        result = score_hypothesis(M, 'pair')
        if result is None:
            continue
        hypotheses.append(result)
        pair_success += 1
    subset_success = 0
    for n in range(SUBSET_HYPOTHESES):
        count = min(SUBSET_SIZE, len(candidate_A))
        idx = rng.choice(len(candidate_A), size=count, replace=False)
        M, mask = cv2.estimateAffinePartial2D(candidate_A[idx], candidate_B[idx], method=cv2.RANSAC, ransacReprojThreshold=8.0, maxIters=5000, confidence=0.999, refineIters=20)
        result = score_hypothesis(M, 'subset_ransac')
        if result is None:
            continue
        hypotheses.append(result)
        subset_success += 1
    print('Valid pair hypotheses:', pair_success)
    print('Valid subset hypotheses:', subset_success)
    print('Total scored hypotheses:', len(hypotheses))
    if len(hypotheses) == 0:
        raise RuntimeError('No valid V8 hypotheses generated.')
    hypotheses.sort(key=lambda x: x['score'], reverse=True)
    print()
    print('===== TOP INITIAL HYPOTHESES =====')
    for rank, h in enumerate(hypotheses[:10], start=1):
        print()
        print(f'#{rank}')
        print('Source          :', h['label'])
        print('Score           :', round(h['score'], 5))
        print('Inliers         :', h['inliers'])
        print('Coverage        :', round(h['coverage'] * 100, 2), '%')
        print('Scale           :', round(h['scale'], 6))
        print('Rotation        :', round(h['rotation'], 4), 'deg')
        print('Translation     :', round(h['tx'], 3), round(h['ty'], 3))
        print('Median residual :', round(h['median_residual'], 3), 'px')
    best = hypotheses[0]
    M_initial = best['M'].copy()
    initial_scale, initial_rotation, initial_tx, initial_ty = transform_parameters(M_initial)
    print()
    print('===== INITIAL WINNING MODE =====')
    print('Score:', round(best['score'], 6))
    print('Inliers:', best['inliers'])
    print('Coverage:', round(best['coverage'] * 100, 2), '%')
    print('Scale:', round(initial_scale, 6))
    print('Rotation:', round(initial_rotation, 4), 'deg')
    print('Translation:', round(initial_tx, 3), round(initial_ty, 3))
    csv_path = f'{OUT}/hypotheses.csv'
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['rank', 'source', 'score', 'data_score', 'prior_score', 'inliers', 'coverage', 'weighted_support', 'median_residual', 'scale', 'rotation', 'tx', 'ty'])
        for rank, h in enumerate(hypotheses, start=1):
            writer.writerow([rank, h['label'], h['score'], h['data_score'], h['prior_score'], h['inliers'], h['coverage'], h['weighted_support'], h['median_residual'], h['scale'], h['rotation'], h['tx'], h['ty']])
    print()
    print('===== LOCKED ROBUST REFINEMENT =====')
    M_current = M_initial.copy()
    refinement_log = []
    for iteration, threshold in enumerate(REFINEMENT_THRESHOLDS, start=1):
        residual = residuals_for_transform(M_current, ptsA, ptsB)
        consensus = residual <= threshold
        A = ptsA[consensus]
        B = ptsB[consensus]
        current_conf = combined_conf[consensus]
        current_cycle = cycle_error[consensus]
        current_residual = residual[consensus]
        if len(A) < 10:
            print('Too few points at threshold', threshold)
            break
        u = current_residual / threshold
        robust_weight = 1.0 - u ** 2
        robust_weight = np.clip(robust_weight, 0.0, 1.0)
        robust_weight = robust_weight ** 2
        image_weight = current_conf * np.exp(-current_cycle / 6.0)
        weights = image_weight * robust_weight
        weights = spatially_balance_weights(A, weights, W_A, H_A, GRID)
        if np.sum(weights) <= 0:
            print('Invalid weights.')
            break
        M_candidate = weighted_similarity_fit(A, B, weights)
        if M_candidate is None:
            print('Weighted fit failed.')
            break
        candidate_scale, candidate_rotation, candidate_tx, candidate_ty = transform_parameters(M_candidate)
        scale_change = abs(candidate_scale - initial_scale) / initial_scale
        rotation_change = abs(wrap_angle_deg(candidate_rotation - initial_rotation))
        current_scale, current_rotation, current_tx, current_ty = transform_parameters(M_current)
        translation_step = math.sqrt((candidate_tx - current_tx) ** 2 + (candidate_ty - current_ty) ** 2)
        accepted = True
        if scale_change > MAX_SCALE_CHANGE_FROM_INITIAL:
            accepted = False
        if rotation_change > MAX_ROTATION_CHANGE_FROM_INITIAL:
            accepted = False
        if translation_step > MAX_TRANSLATION_STEP:
            accepted = False
        if accepted:
            M_new = blend_similarity(M_current, M_candidate, REFINEMENT_ALPHA)
        else:
            M_new = M_current.copy()
        new_scale, new_rotation, new_tx, new_ty = transform_parameters(M_new)
        new_residual = residuals_for_transform(M_new, ptsA, ptsB)
        new_mask = new_residual <= threshold
        coverage, cells = coverage_from_points(ptsA[new_mask], W_A, H_A, GRID)
        median_res = float(np.median(new_residual[new_mask]))
        print()
        print(f'Iteration {iteration}')
        print('Threshold      :', threshold, 'px')
        print('Consensus      :', int(new_mask.sum()))
        print('Coverage       :', round(coverage * 100, 2), '%')
        print('Accepted update:', accepted)
        print('Scale          :', round(new_scale, 6))
        print('Rotation       :', round(new_rotation, 4), 'deg')
        print('Translation    :', round(new_tx, 3), round(new_ty, 3))
        print('Median residual:', round(median_res, 3), 'px')
        refinement_log.append([iteration, threshold, int(new_mask.sum()), coverage, accepted, new_scale, new_rotation, new_tx, new_ty, median_res])
        M_current = M_new
    M_final = M_current.copy()
    M_v8_coarse = M_final.copy()
    print()
    print('===== V9 CONSTRAINED AFFINE MICRO-REFINEMENT =====')
    v8_residual = residuals_for_transform(M_v8_coarse, ptsA, ptsB)
    v8_consistent = v8_residual <= RELAXED_THRESHOLD
    affine_pool_A = ptsA[v8_consistent]
    affine_pool_B = ptsB[v8_consistent]
    affine_pool_conf = combined_conf[v8_consistent]
    affine_pool_cycle = cycle_error[v8_consistent]
    print('Frozen V8-consistent pool:', len(affine_pool_A))
    M_affine = M_v8_coarse.copy()
    affine_refinement_log = []
    for iteration, threshold in enumerate(AFFINE_REFINEMENT_THRESHOLDS, start=1):
        pool_residual = residuals_for_transform(M_affine, affine_pool_A, affine_pool_B)
        consensus = pool_residual <= threshold
        A = affine_pool_A[consensus]
        B = affine_pool_B[consensus]
        residual = pool_residual[consensus]
        conf = affine_pool_conf[consensus]
        cycle = affine_pool_cycle[consensus]
        accepted = False
        if len(A) >= AFFINE_MIN_CONSENSUS:
            u = np.clip(residual / threshold, 0.0, 1.0)
            robust = (1.0 - u * u) ** 2
            image_weight = conf * np.exp(-cycle / 6.0)
            weights = spatially_balance_weights(A, image_weight * robust, W_A, H_A, GRID)
            candidate = weighted_affine_fit(A, B, weights)
            if candidate is not None:
                candidate = constrain_affine(candidate, M_v8_coarse)
            if candidate is not None:
                blended = blend_affine(M_affine, candidate, AFFINE_ALPHA)
                blended = constrain_affine(blended, M_v8_coarse)
                if blended is not None:
                    M_affine = blended
                    accepted = True
        new_residual = residuals_for_transform(M_affine, affine_pool_A, affine_pool_B)
        new_mask = new_residual <= threshold
        coverage, _ = coverage_from_points(affine_pool_A[new_mask], W_A, H_A, GRID)
        sx_aff, sy_aff, anis_aff, shear_aff, rot_aff = affine_components(M_affine)
        median_res = float(np.median(new_residual[new_mask])) if np.any(new_mask) else float('inf')
        translation_delta = float(np.linalg.norm(M_affine[:, 2] - M_v8_coarse[:, 2]))
        print()
        print(f'Affine iteration {iteration}')
        print('Threshold      :', threshold, 'px')
        print('Consensus      :', int(new_mask.sum()))
        print('Coverage       :', round(coverage * 100, 2), '%')
        print('Accepted update:', accepted)
        print('Axes           :', round(sx_aff, 6), round(sy_aff, 6))
        print('Anisotropy     :', round(anis_aff, 5))
        print('Shear          :', round(shear_aff, 5))
        print('Rotation       :', round(rot_aff, 4), 'deg')
        print('Translation Δ  :', round(translation_delta, 3), 'px')
        print('Median residual:', round(median_res, 3), 'px')
        affine_refinement_log.append([iteration, threshold, int(new_mask.sum()), coverage, accepted, sx_aff, sy_aff, anis_aff, shear_aff, rot_aff, M_affine[0, 2], M_affine[1, 2], translation_delta, median_res])
    M_final = M_affine.copy()
    final_scale, final_rotation, final_tx, final_ty = transform_parameters(M_final)
    all_residual = residuals_for_transform(M_final, ptsA, ptsB)
    strict_mask = all_residual <= STRICT_THRESHOLD
    strict_A = ptsA[strict_mask]
    strict_B = ptsB[strict_mask]
    strict_residual = all_residual[strict_mask]
    strict_coverage, strict_cells = coverage_from_points(strict_A, W_A, H_A, GRID)
    if len(strict_residual) > 0:
        strict_rmse = float(np.sqrt(np.mean(strict_residual ** 2)))
    else:
        strict_rmse = float('inf')
    relaxed_mask = all_residual <= RELAXED_THRESHOLD
    relaxed_A = ptsA[relaxed_mask]
    relaxed_B = ptsB[relaxed_mask]
    relaxed_residual = all_residual[relaxed_mask]
    relaxed_coverage, relaxed_cells = coverage_from_points(relaxed_A, W_A, H_A, GRID)
    if len(relaxed_residual) > 0:
        relaxed_rmse = float(np.sqrt(np.mean(relaxed_residual ** 2)))
    else:
        relaxed_rmse = float('inf')
    scale_error_percent = abs(final_scale - expected_scale) / expected_scale * 100.0
    print()
    print('===== ROMA V9 FINAL RESULTS =====')
    print()
    print('Initial scale       :', round(initial_scale, 6))
    print('Final scale         :', round(final_scale, 6))
    print('Expected scale      :', round(expected_scale, 6))
    print('Scale error         :', round(scale_error_percent, 2), '%')
    print()
    print('Initial rotation    :', round(initial_rotation, 4), 'deg')
    print('Final rotation      :', round(final_rotation, 4), 'deg')
    print()
    print('Initial translation :', round(initial_tx, 3), round(initial_ty, 3))
    print('Final translation   :', round(final_tx, 3), round(final_ty, 3))
    axis_x, axis_y, final_anisotropy, final_shear, final_polar_rotation = affine_components(M_final)
    print()
    print('V8 coarse matrix    :', np.array2string(M_v8_coarse, precision=7))
    print('V9 affine matrix    :', np.array2string(M_final, precision=7))
    print('Affine axes         :', round(axis_x, 6), round(axis_y, 6))
    print('Affine anisotropy   :', round(final_anisotropy, 5))
    print('Affine shear        :', round(final_shear, 5))
    print('Polar rotation      :', round(final_polar_rotation, 4), 'deg')
    print()
    print('--- STRICT <=5PX CONSENSUS ---')
    print('Inliers             :', len(strict_A))
    print('Inlier ratio        :', round(len(strict_A) / len(ptsA), 4))
    print('RMSE                :', round(strict_rmse, 4), 'px')
    print('Coverage            :', round(strict_coverage * 100, 2), '%')
    print()
    print('--- RELAXED <=8PX CONSENSUS ---')
    print('Inliers             :', len(relaxed_A))
    print('Inlier ratio        :', round(len(relaxed_A) / len(ptsA), 4))
    print('RMSE                :', round(relaxed_rmse, 4), 'px')
    print('Coverage            :', round(relaxed_coverage * 100, 2), '%')
    passed = len(strict_A) >= 30 and strict_rmse <= 5.0 and (relaxed_coverage >= 0.25) and (scale_error_percent <= 8.0) and (abs(wrap_angle_deg(final_rotation - EXPECTED_ROTATION_DEG)) <= 5.0)
    print()
    if passed:
        print('REGISTRATION: PASS ✅')
    else:
        print('REGISTRATION: FAIL ❌')
    return dict(transform=M_final, coarse_transform=M_v8_coarse, strict_A=strict_A, strict_B=strict_B, relaxed_A=relaxed_A, relaxed_B=relaxed_B, strict_rmse=float(strict_rmse), strict_coverage=float(strict_coverage), relaxed_coverage=float(relaxed_coverage), passed=bool(passed), similarity_log=refinement_log, affine_log=affine_refinement_log)
