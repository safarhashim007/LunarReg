"""Bidirectional RoMa sampling and reverse-cycle queries from V9."""
import torch
import torch.nn.functional as F
import numpy as np
def match(source, reference, source_shape, reference_shape, model, sample_count=5000):
    CH_PATH = str(source)
    LRO_PATH = str(reference)
    H_A, W_A = source_shape
    H_B, W_B = reference_shape
    SAMPLE_COUNT = sample_count
    expected_scale = (W_B / W_A + H_B / H_A) / 2
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
