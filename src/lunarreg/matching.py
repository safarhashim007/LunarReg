"""Bidirectional RoMa sampling and reverse-cycle queries from V9."""
import numpy as np
from pathlib import Path
import tempfile


def blind_rotation_angles(step_degrees):
    """Return a complete, non-prior orientation sweep over [0, 360)."""
    if not np.isfinite(step_degrees) or not 0 < step_degrees <= 180:
        raise ValueError('Blind rotation step must be greater than 0 and at most 180 degrees')
    return tuple(float(angle) for angle in np.arange(0.0, 360.0, step_degrees))


def map_rotated_points_to_source(points, inverse_rotation, source_shape):
    """Map points from a fixed-canvas rotated source back to original pixels."""
    import cv2
    points = np.asarray(points, dtype=np.float32)
    original = cv2.transform(points.reshape(1, -1, 2), inverse_rotation)[0]
    height, width = source_shape
    valid = ((original[:, 0] >= 0) & (original[:, 0] < width)
             & (original[:, 1] >= 0) & (original[:, 1] < height))
    return original.astype(np.float32), valid


def match(source, reference, source_shape, reference_shape, model, sample_count=5000, *, blind=False):
    # Keep rotation-sweep utilities and CLI-adjacent imports lightweight.
    import torch
    import torch.nn.functional as F
    CH_PATH = str(source)
    LRO_PATH = str(reference)
    H_A, W_A = source_shape
    H_B, W_B = reference_shape
    SAMPLE_COUNT = sample_count
    expected_scale = None if blind else (W_B / W_A + H_B / H_A) / 2
    print()
    print('Running bidirectional RoMa...')
    with torch.inference_mode():
        predictions = model.match(CH_PATH, LRO_PATH)
    if predictions['warp_BA'] is None:
        raise RuntimeError('Backward RoMa warp missing.')
    print('Dense matching complete ✅')
    print()
    print('Sampling', SAMPLE_COUNT, 'forward correspondences...')
    model.bidirectional = False
    matches, forward_conf, _, _ = model.sample(predictions, SAMPLE_COUNT)
    model.bidirectional = True
    kptsA, kptsB = model.to_pixel_coordinates(matches, H_A, W_A, H_B, W_B)
    ptsA = kptsA.detach().cpu().numpy().astype(np.float32)
    ptsB = kptsB.detach().cpu().numpy().astype(np.float32)
    forward_conf = forward_conf.detach().cpu().numpy().reshape(-1).astype(np.float32)
    print('Forward matches:', len(ptsA))
    B_norm = matches[:, 2:4]
    warp_BA = predictions['warp_BA']
    warp_BA_chw = warp_BA.permute(0, 3, 1, 2)
    query_grid = B_norm.reshape(1, -1, 1, 2)
    with torch.inference_mode():
        A_back_norm = F.grid_sample(warp_BA_chw, query_grid, mode='bilinear', padding_mode='zeros', align_corners=False)
    A_back_norm = A_back_norm[0, :, :, 0].T
    overlap_BA = predictions['overlap_BA']
    overlap_BA_chw = overlap_BA.permute(0, 3, 1, 2)
    with torch.inference_mode():
        reverse_conf = F.grid_sample(overlap_BA_chw, query_grid, mode='bilinear', padding_mode='zeros', align_corners=False)
    reverse_conf = reverse_conf[0, 0, :, 0]
    reverse_conf = reverse_conf.detach().cpu().numpy().astype(np.float32)
    back_x = (A_back_norm[:, 0] + 1.0) / 2.0 * W_A
    back_y = (A_back_norm[:, 1] + 1.0) / 2.0 * H_A
    back_A = torch.stack([back_x, back_y], dim=1)
    back_A = back_A.detach().cpu().numpy().astype(np.float32)
    if blind:
        # RoMa to_pixel uses edge-origin coordinates; OpenCV rotations and
        # the held-out CSV use integer pixel centers. Keep V9 unchanged.
        ptsA -= .5
        ptsB -= .5
        back_A -= .5
    from .cycle import cycle_metrics
    cycle_error, combined_conf, cycle_score = cycle_metrics(ptsA, back_A, forward_conf, reverse_conf, expected_scale)
    valid = np.isfinite(cycle_error) & np.isfinite(cycle_score) & (reverse_conf > 0) & (ptsA[:, 0] >= 0) & (ptsA[:, 0] < W_A) & (ptsA[:, 1] >= 0) & (ptsA[:, 1] < H_A) & (ptsB[:, 0] >= 0) & (ptsB[:, 0] < W_B) & (ptsB[:, 1] >= 0) & (ptsB[:, 1] < H_B)
    ptsA = ptsA[valid]
    ptsB = ptsB[valid]
    combined_conf = combined_conf[valid]
    cycle_error = cycle_error[valid]
    cycle_score = cycle_score[valid]
    print()
    print('Valid matches:', len(ptsA))
    return (ptsA, ptsB, combined_conf, cycle_error)


def match_blind_rotation_sweep(source, reference, source_image, reference_shape, model, *, step_degrees=30.0,
                               sample_count=5000, temporary_directory=None):
    """Collect image-only RoMa candidates across the full source orientation range.

    Each rotated temporary source retains the original canvas dimensions. Returned
    source points are mapped back before pooling, so downstream fitting sees only
    the original image coordinate system.
    """
    import cv2

    source_image = np.asarray(source_image)
    height, width = source_image.shape[:2]
    angles = blind_rotation_angles(step_degrees)
    pooled = [[], [], [], []]
    attempts = []
    temp_root = Path(temporary_directory) if temporary_directory else None
    with tempfile.TemporaryDirectory(prefix='blind_rotation_sources_', dir=temp_root) as directory:
        for index, angle in enumerate(angles):
            if angle == 0.0:
                candidate_path = source
                inverse_rotation = None
            else:
                rotation = cv2.getRotationMatrix2D(((width - 1) / 2, (height - 1) / 2), angle, 1.0)
                inverse_rotation = cv2.invertAffineTransform(rotation)
                rotated = cv2.warpAffine(source_image, rotation, (width, height), flags=cv2.INTER_LINEAR,
                                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                candidate_path = Path(directory) / f'source_{index:03d}_{angle:g}deg.png'
                if not cv2.imwrite(str(candidate_path), rotated):
                    raise RuntimeError(f'Could not write temporary rotated source for {angle:g} degrees')
            A, B, confidence, cycle = match(candidate_path, reference, (height, width), reference_shape,
                                             model, sample_count=sample_count, blind=True)
            if inverse_rotation is not None and len(A):
                A, valid = map_rotated_points_to_source(A, inverse_rotation, (height, width))
                A, B, confidence, cycle = A[valid], B[valid], confidence[valid], cycle[valid]
            pooled[0].append(A)
            pooled[1].append(B)
            pooled[2].append(confidence)
            pooled[3].append(cycle)
            attempts.append({'rotation_degrees': angle, 'valid_matches': int(len(A))})
            if temp_root is not None:
                # Durable image-only evidence after every completed orientation.
                from .io import json_write
                np.savez_compressed(temp_root / f'rotation_{index:03d}_matches.npz',
                                    source=A, reference=B, confidence=confidence, cycle_error=cycle)
                json_write(temp_root / 'rotation_progress.json',
                           dict(attempts=attempts, complete=len(attempts)==len(angles)))
            print(f'Rotation {angle:g}: {len(A)} valid matches', flush=True)
    return (*(np.concatenate(values, axis=0) for values in pooled), attempts)
