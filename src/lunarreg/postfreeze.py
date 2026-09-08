"""Separate held-out evaluator. Never imported by matching or fitting."""
import argparse
import json
from pathlib import Path
import numpy as np
from .io import sha256, json_write
from .geometry import apply_transform


def error_metrics(errors):
    e = np.asarray(errors, dtype=float)
    if not len(e) or not np.isfinite(e).all():
        raise ValueError('Errors must be nonempty and finite')
    return dict(count=len(e), median=float(np.median(e)), mean=float(e.mean()),
                rmse=float(np.sqrt(np.mean(e**2))), p90=float(np.percentile(e,90)),
                **{f'pct_lt_{t:g}px':float(100*np.mean(e<t)) for t in (.5,1,2,5,10,20)})


def frozen_transform(run):
    run = Path(run)
    receipt = json.loads((run/'transform_frozen.json').read_text())
    if receipt['sha256'] != sha256(run/'transform.npy'):
        raise ValueError('Frozen transform hash mismatch')
    m = np.load(run/'transform.npy', allow_pickle=False)
    if m.shape != (2,3) or not np.isfinite(m).all() or np.linalg.det(m[:,:2])<=0 or np.linalg.cond(m[:,:2])>=100:
        raise ValueError('Invalid frozen affine transform')
    return m, receipt


def evaluate(run, controls, output, expected_count=14520):
    # Integrity and freeze checks MUST precede opening controls.
    m, receipt = frozen_transform(run)
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    data = np.genfromtxt(controls, delimiter=',', names=True)
    a = np.column_stack([data['source_x'],data['source_y']])
    b = np.column_stack([data['lro_x'],data['lro_y']])
    if len(a)!=expected_count or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Unexpected control count or nonfinite controls')
    delta = apply_transform(m,a)-b
    report = dict(transform_sha256=receipt['sha256'], controls_sha256=sha256(controls),
                  reference_pixels=error_metrics(np.linalg.norm(delta,axis=1)),
                  before_reference_pixels=error_metrics(np.linalg.norm(a-b,axis=1)),
                  convention='zero-based pixel centers; no points discarded',
                  interpretation='Agreement with ISRO system geometry and LRO mapping; not surveyed absolute ground truth')
    # Geometry is now unsealed. Estimate a local differential from neighboring
    # control coordinates, never from the fitted registration scale.
    if len(a)<9 or len(np.unique(a,axis=0)) != len(a):
        raise ValueError('Need at least nine distinct source controls')
    neighbors=[]
    for start in range(0,len(a),128):
        distance=np.sum((a[start:start+128,None,:]-a[None,:,:])**2,axis=2)
        neighbors.append(np.argpartition(distance,8,axis=1)[:,:9])
    ix=np.concatenate(neighbors)
    da = a[ix]-a[:,None,:]
    db = b[ix]-b[:,None,:]
    j = np.linalg.pinv(da) @ db  # row-vector Jacobian, source -> reference
    local_error = np.sqrt(np.mean(np.sum((da@j-db)**2,axis=2),axis=1))
    condition = np.linalg.cond(j)
    support_radius = np.max(np.linalg.norm(db,axis=2),axis=1)
    valid = (local_error<=.1)&(condition<100)&np.isfinite(condition)&(np.linalg.norm(delta,axis=1)<=support_radius)
    report['source_pixel_conversion'] = dict(valid=bool(valid.all()),
        valid_controls=int(valid.sum()), total_controls=len(a),
        max_local_linearization_rmse=float(local_error.max()),
        method='Inverse local 9-neighbor geometry Jacobian, post-freeze only. Units are source PNG pixels, not native detector pixels.',
        limitation='First-order pixel-equivalent displacement, not proof of recovered source detail; invalid if any local fit RMSE exceeds 0.1 reference pixels or condition number reaches 100, or displacement exceeds local reference support radius.')
    if valid.all():
        equivalent = np.einsum('ni,nij->nj',delta,np.linalg.inv(j))
        report['source_pixel_equivalent'] = error_metrics(np.linalg.norm(equivalent,axis=1))
    else:
        report['source_pixel_equivalent'] = None
    report['real_subpixel_criterion'] = 'P90 < 1 pixel in both reference and valid source-equivalent units'
    report['real_subpixel_achieved'] = bool(report['reference_pixels']['p90']<1 and report['source_pixel_equivalent'] is not None and report['source_pixel_equivalent']['p90']<1)
    # Detect mutation during evaluation too.
    frozen_transform(run)
    json_write(output,report)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',required=True); p.add_argument('--controls',required=True)
    p.add_argument('--output',required=True)
    a=p.parse_args(); evaluate(a.run,a.controls,a.output)

if __name__=='__main__': main()
