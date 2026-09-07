"""Known pixel-center affine truth is used only after each estimator finishes."""
import contextlib
import logging
from pathlib import Path
import cv2
import numpy as np
from .io import read_image,json_write,csv_write
from .runtime import seed_all,load_model,fine_extractor
from .matching import match
from .global_refinement import solve
from .geometry import apply_transform
from .local_refinement import descriptors,refine,supported_field
from .evaluation import metrics,coverage


def controlled_cases(seed=42,size=384):
    rng=np.random.default_rng(seed)
    specs=[('translation',0,1,(2.35,-1.65),0,'none'),('rotation',2.2,1,(0,0),0,'none'),('scale',0,1.025,(-2,1),0,'none'),('affine',-1.5,.99,(1.3,-2.4),.008,'none'),('gamma',1.2,1.01,(2.35,-1.65),0,'gamma'),('illumination',-1,1,(1.75,2.25),.006,'illumination')]
    for name,angle,scale,shift,shear,photo in specs:
        M=cv2.getRotationMatrix2D(((size-1)/2,(size-1)/2),angle,scale)
        M[:,2]+=shift;M[0,1]+=shear
        yield name,M,photo,rng


def synthesize(source,M,photo,rng):
    h,w=source.shape
    target=cv2.warpAffine(source,M,(w,h),flags=cv2.INTER_CUBIC)
    valid=cv2.warpAffine(np.ones(source.shape,np.float32),M,(w,h))
    x=target.astype(float)/255
    if photo=='gamma': x=.1+.8*x**.65
    elif photo=='illumination':
        xx=np.linspace(-1,1,w)[None,:];yy=np.linspace(-1,1,h)[:,None]
        x=x*(.8+.25*xx)+.1+.05*yy+rng.normal(0,.007,x.shape)
    return np.uint8(np.clip(x*255,0,255)),valid


def run_benchmark(args):
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    logging.basicConfig(level=logging.INFO,force=True)
    seed_all(args.seed)
    source=cv2.resize(read_image(args.source),(384,384),interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(out/'source.png'),source)
    model=load_model('fast',args.device);extractor=fine_extractor(args.device)
    source_features=descriptors(source,extractor)
    rows=[]; reports={}
    points=np.stack(np.meshgrid(np.linspace(55,328,12),np.linspace(55,328,12)),axis=-1).reshape(-1,2)
    for name,truth,photo,rng in controlled_cases(args.seed):
        logging.info('Controlled case: %s',name)
        case=out/name;case.mkdir(); target,valid=synthesize(source,truth,photo,rng)
        cv2.imwrite(str(case/'reference.png'),target)
        report={'ground_truth_transform':truth.tolist(),'photometric_change':photo}
        # Truth is not passed to matching or global solver.
        try:
            seed_all(args.seed)
            with open(case/'stages.log','w') as log,contextlib.redirect_stdout(log):
                A,B,conf,cycle=match(out/'source.png',case/'reference.png',source.shape,target.shape,model)
                result=solve(A,B,conf,cycle,source.shape,target.shape,str(case),seed=args.seed)
            M=result['transform'];np.save(case/'coarse_transform.npy',M)
            coarse=cv2.warpAffine(source,M,target.shape[::-1])
            warp_valid=cv2.warpAffine(np.ones(source.shape,np.float32),M,target.shape[::-1])
            predicted=apply_transform(M,points)
            localrows=refine(descriptors(coarse,extractor),descriptors(target,extractor),warp_valid,predicted)
            csv_write(case/'local_controls.csv',localrows)
            corrected=predicted+supported_field(predicted,localrows)
            expected=apply_transform(truth,points)
            for method,pred in [('coarse_v9_style',predicted),('v11_supported_field',corrected)]:
                metric=metrics(np.linalg.norm(pred-expected,axis=1))
                row=dict(case=name,method=method,**metric,coverage_pct=coverage(points,source.shape),failure_rate_pct=0. if result['passed'] else 100.)
                rows.append(row)
            report['quality_gate']=result['passed']
            report['accepted_local_controls']=sum(r['reason']=='accepted' for r in localrows)
            np.savez_compressed(case/'evaluation.npz',source_points=points,expected=expected,coarse=predicted,refined=corrected)
        except Exception as exc:
            logging.exception('Global benchmark case failed')
            report['failure']=str(exc)
        # Isolated local capture test: known small residuals, without a learned coarse stage.
        # Identity prediction is independent of the true matrix; truth remains evaluation-only.
        localrows=refine(source_features,descriptors(target,extractor),np.ones(source.shape,np.float32),points)
        accepted=np.array([r['reason']=='accepted' for r in localrows])
        delta=np.array([[r['dx'],r['dy']] for r in localrows])
        expected=apply_transform(truth,points)
        pred=points+delta*accepted[:,None]
        for label,mask in [('local_isolated_all',np.ones(len(points),bool)),('local_isolated_accepted',accepted)]:
            row=dict(case=name,method=label,**metrics(np.linalg.norm(pred[mask]-expected[mask],axis=1)),coverage_pct=coverage(points[mask],source.shape),failure_rate_pct=float(100*(1-accepted.mean())))
            rows.append(row)
        csv_write(case/'isolated_local_controls.csv',localrows)
        reports[name]=report
        json_write(case/'report.json',report)
        csv_write(out/'metrics.csv',rows)
    from .validation import negative_local_checks
    json_write(out/'negative_local.json',negative_local_checks(source,extractor,args.seed))
    json_write(out/'metrics.json',dict(seed=args.seed,units='reference pixels; exact synthetic pixel-center ground truth',cases=reports,metrics=rows,limitations='Single lunar source, six controlled warps; no surveyed real cross-sensor landmark truth. Isolated accepted-only accuracy excludes rejected controls; all-point and rejection rates are reported.'))
    return 0 if all('failure' not in r for r in reports.values()) else 2
