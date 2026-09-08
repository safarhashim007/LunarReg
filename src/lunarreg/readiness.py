"""Conservative post-run certification; missing evidence fails closed."""
import argparse
import contextlib
import json
from pathlib import Path
import tempfile
import numpy as np
from .io import json_write, sha256, read_image
from .postfreeze import frozen_transform
from .geometry import apply_transform
from .evaluation import coverage


def certify(run, evaluation, regression, output):
    run=Path(run); checks={name:dict(status='FAIL',reason='Required evidence unavailable') for name in ('transform_integrity','image_only_configuration','input_identity','image_quality_gate','full_rotation_sweep','inlier_support','spatial_coverage','deterministic_fit_replay','heldout_evaluation','real_subpixel_accuracy','output_validation')}
    def check(name,passed,reason):
        checks[name]=dict(status='PASS' if passed else 'FAIL',reason=reason)
    try:
        m,receipt=frozen_transform(run)
        check('transform_integrity',True,'Finite orientation-preserving affine; frozen hash verified')
        cfg=json.loads((run/'config.json').read_text())
        report=json.loads((run/'metrics.json').read_text())
        check('image_only_configuration',cfg.get('registration_mode')=='blind' and not any(cfg.get(k) for k in ('source_geotiff','reference_geotiff','expected_scale','expected_rotation','frozen_baseline')),'Blind image-only configuration without priors')
        p=report['provenance']
        check('input_identity',sha256(cfg['source'])==p['source_sha256'] and sha256(cfg['reference'])==p['reference_sha256'],'Input image hashes match run provenance')
        check('image_quality_gate',report['quality_gate'],'Original image-only solver quality gate')
        sweep=json.loads((run/'rotation_sweep.json').read_text())['attempts']
        check('full_rotation_sweep',[a['rotation_degrees'] for a in sweep]==list(range(0,360,30)),'Required twelve orientations completed')
        with np.load(run/'matches.npz',allow_pickle=False) as z:
            a,b,c,e=[z[k] for k in ('source','reference','confidence','cycle_error')]
        residual=np.linalg.norm(apply_transform(m,a)-b,axis=1)
        inliers=residual<=5
        source=read_image(cfg['source']); reference=read_image(cfg['reference'])
        sc=coverage(a[inliers],source.shape); rc=coverage(b[inliers],reference.shape)
        check('inlier_support',int(inliers.sum())>=30 and float(inliers.mean())>=.01,f'{int(inliers.sum())}/{len(a)} within 5 reference pixels; minimum 30 and ratio 0.01')
        check('spatial_coverage',sc>=25 and rc>=25,f'Source {sc:.3f}%, reference {rc:.3f}%; minimum 25% each')
        from .global_refinement import solve
        repeats=[]
        with tempfile.TemporaryDirectory() as tmp:
            for i in range(2):
                dest=Path(tmp)/str(i);dest.mkdir()
                with (dest/'fit.log').open('w') as log,contextlib.redirect_stdout(log):
                    r=solve(a,b,c,e,source.shape,reference.shape,str(dest),cfg['seed'],blind=True)
                repeats.append(r['transform'])
        check('deterministic_fit_replay',all(np.array_equal(m,r) for r in repeats),'Two same-seed fits from saved matches must exactly equal frozen transform; neural inference repeatability is not certified')
        ev=json.loads(Path(evaluation).read_text())
        check('heldout_evaluation',ev['transform_sha256']==receipt['sha256'] and ev['reference_pixels']['count']==14520,'All 14520 controls evaluated against this frozen transform')
        check('real_subpixel_accuracy',ev['real_subpixel_achieved'],'Required P90 <1 pixel in reference and valid source-equivalent units')
        import cv2
        images_valid=all((im is not None and im.shape==reference.shape) for im in (cv2.imread(str(run/f),0) for f in ('registered.png','overlay.png','difference.png')))
        check('output_validation',images_valid and all((run/f).is_file() and (run/f).stat().st_size>0 for f in ('transform.npy','matches.npz','rotation_sweep.json','metrics.json')),'Required machine-readable outputs present and parsed')
    except Exception as exc:
        check('run_completion',False,f'{type(exc).__name__}: {exc}')
    tmc_path=Path(regression).with_name('tmc_regression.json')
    tmc=json.loads(tmc_path.read_text()) if tmc_path.exists() else {}
    check('tmc_v9_regression',tmc.get('passed',False),'Actual pair_002 frozen V9 replay must be bitwise identical and historical hashes unchanged')
    log=Path(regression).read_text() if Path(regression).exists() else ''
    check('regression_tests','\nOK\n' in log and 'FAILED' not in log,'Full unittest suite including legacy defaults, blind isolation and evaluator integrity')
    isolation = json.loads((run/'geometry_isolation.json').read_text()) if (run/'geometry_isolation.json').exists() else {}
    check('no_geometry_leakage', isolation.get('passed',False) and 'test_pipeline_freezes_before_evaluation_and_never_reads_controls' in log and '\nOK\n' in log,'Runtime file-open guard and PNG input restriction plus guarded regression; native library system calls are outside Python audit scope')
    result=dict(production_ready=all(c['status']=='PASS' for c in checks.values()),checks=checks,
                policy='All listed checks required. Thresholds declared before unsealing evaluation; no tuning from controls.',
                neural_repeatability='not_run; deterministic replay covers frozen sampled correspondences only')
    result['verdict']='PASS' if result['production_ready'] else 'FAIL'
    if Path(output).exists(): raise FileExistsError(output)
    json_write(output,result)
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for flag in ('run','evaluation','regression','output'): p.add_argument('--'+flag,required=True)
    a=p.parse_args();certify(a.run,a.evaluation,a.regression,a.output)
