"""Production orchestration: matching/fitting completes before optional metadata evaluation."""
import contextlib
import logging
from pathlib import Path
from collections import Counter
import cv2
import numpy as np
from .io import read_image,sha256,json_write,csv_write
from .geometry import apply_transform,transform_parameters,wrap_angle_deg
from .evaluation import metrics,coverage,georeference
from .runtime import seed_all,load_model,fine_extractor
from .visualization import save_registration


def run(args):
    out=Path(args.output)
    out.mkdir(parents=True,exist_ok=False)
    logging.basicConfig(level=logging.INFO,handlers=[logging.FileHandler(out/'run.log'),logging.StreamHandler()],force=True)
    try:
        source=read_image(args.source); reference=read_image(args.reference)
        seed_all(args.seed)
        provenance=dict(source_sha256=sha256(args.source),reference_sha256=sha256(args.reference),seed=args.seed,mode=args.mode,device_requested=args.device)
        json_write(out/'config.json',vars(args))
        if args.frozen_baseline:
            root=Path(args.frozen_baseline)
            # Frozen replay is explicit and bound to its original input images.
            original=root.parents[1]/'data/test_pairs'/root.name
            if sha256(original/'chandrayaan.png')!=provenance['source_sha256'] or sha256(original/'lro.png')!=provenance['reference_sha256']:
                raise ValueError('Frozen baseline input identity mismatch')
            M=np.load(root/'roma_v9_transform.npy',allow_pickle=False)
            A=np.load(root/'roma_v9_relaxed_A.npy',allow_pickle=False)
            strictA=np.load(root/'roma_v9_strict_A.npy');strictB=np.load(root/'roma_v9_strict_B.npy')
            result=dict(transform=M,relaxed_A=A,strict_rmse=float(np.sqrt(np.mean(np.sum((apply_transform(M,strictA)-strictB)**2,axis=1)))),strict_coverage=coverage(strictA,source.shape)/100,relaxed_coverage=coverage(A,source.shape)/100,passed=len(strictA)>=30)
            scale,rotation,_,_=transform_parameters(M)
            expected=(reference.shape[1]/source.shape[1]+reference.shape[0]/source.shape[0])/2
            result['passed']=bool(len(strictA)>=30 and result['strict_rmse']<=5 and result['relaxed_coverage']>=.25 and abs(scale/expected-1)<=.08 and abs(wrap_angle_deg(rotation))<=5)
            provenance['baseline_transform_sha256']=sha256(root/'roma_v9_transform.npy')
            provenance['baseline_array_sha256']={p.name:sha256(p) for p in root.glob('roma_v9_*.npy')}
            provenance['mode']='frozen_v9_replay'
            np.savez_compressed(out/'controls.npz',source=A)
        else:
            from .matching import match
            from .global_refinement import solve
            logging.info('Running fresh bidirectional matching')
            try:
                model=load_model(args.mode,args.device)
                with open(out/'stages.log','w') as log,contextlib.redirect_stdout(log):
                    A,B,conf,cycle=match(args.source,args.reference,source.shape,reference.shape,model)
            except RuntimeError:
                if args.device=='cpu': raise
                logging.exception('Accelerator failed; retrying on CPU')
                model=load_model(args.mode,'cpu')
                seed_all(args.seed)
                with open(out/'stages.log','a') as log,contextlib.redirect_stdout(log):
                    A,B,conf,cycle=match(args.source,args.reference,source.shape,reference.shape,model)
            del model
            np.savez_compressed(out/'matches.npz',source=A,reference=B,confidence=conf,cycle_error=cycle)
            with open(out/'stages.log','a') as log,contextlib.redirect_stdout(log):
                result=solve(A,B,conf,cycle,source.shape,reference.shape,str(out),args.seed,args.expected_scale,args.expected_rotation)
            for key in ('similarity_log','affine_log'): json_write(out/f'{key}.json',result[key])
            A=result['relaxed_A'];M=result['transform']
        if M.shape!=(2,3) or not np.isfinite(M).all() or np.linalg.det(M[:,:2])<=0: raise ValueError('Invalid transform')
        np.save(out/'transform.npy',M)
        report=dict(provenance=provenance,quality_gate=bool(result['passed']),internal_fit_rmse=result['strict_rmse'],strict_coverage_pct=result['strict_coverage']*100,relaxed_coverage_pct=result['relaxed_coverage']*100,local_default=False)
        # Default always emits the validated global transform. Optional V11 is a separate candidate.
        save_registration(out,source,reference,M)
        h,w=reference.shape; yy,xx=np.mgrid[:h,:w];grid=np.column_stack((xx.ravel(),yy.ravel()))
        field=np.zeros((h,w,2),np.float32)
        if args.experimental_local and not args.no_local_refine:
            from .local_refinement import descriptors,refine,supported_field
            logging.info('Evaluating experimental V11 learned-descriptor refinement')
            coarse=cv2.warpAffine(source,M,(w,h)); valid=cv2.warpAffine(np.ones(source.shape,np.float32),M,(w,h))
            extractor=fine_extractor(args.device)
            rows=refine(descriptors(coarse,extractor),descriptors(reference,extractor),valid,apply_transform(M,A))
            csv_write(out/'v11_controls.csv',rows)
            field=supported_field(grid,rows).reshape(h,w,2).astype(np.float32)
            np.save(out/'v11_field.npy',field)
            save_registration(out,source,reference,M,field,'v11_')
            report['v11']=dict(controls=len(rows),reasons=dict(Counter(r['reason'] for r in rows)),accepted_coverage_pct=coverage(np.array([[r['x'],r['y']] for r in rows if r['reason']=='accepted']).reshape(-1,2),reference.shape),changed_pixel_pct=float(100*np.mean(np.linalg.norm(field,axis=2)>1e-6)),status='experimental; not selected for production')
        # Everything above is finalized without opening either GeoTIFF.
        if bool(args.source_geotiff)!=bool(args.reference_geotiff): raise ValueError('Provide both evaluation GeoTIFFs')
        if args.source_geotiff:
            pools={'controls':A,'uniform_grid':np.stack(np.meshgrid(np.linspace(0,source.shape[1]-1,40),np.linspace(0,source.shape[0]-1,40)),axis=-1).reshape(-1,2)}
            if args.frozen_baseline and (Path(args.frozen_baseline)/'roma_cycle_evaluation.npz').is_file():
                with np.load(Path(args.frozen_baseline)/'roma_cycle_evaluation.npz',allow_pickle=False) as cache:
                    pools['cached_sample']=cache['ptsA'].astype(np.float64)
            report['metadata_agreement']={} 
            for name,points in pools.items():
                pred=apply_transform(M,points)
                report['metadata_agreement'][name]={}
                for center in (False,True):
                    truth=georeference(args.source_geotiff,args.reference_geotiff,points,source.shape,reference.shape,center)
                    inside=(truth[:,0]>=0)&(truth[:,0]<w)&(truth[:,1]>=0)&(truth[:,1]<h)
                    delta=cv2.remap(field,pred[:,0].astype(np.float32).reshape(-1,1),pred[:,1].astype(np.float32).reshape(-1,1),cv2.INTER_LINEAR).reshape(-1,2)
                    report['metadata_agreement'][name]['pixel_center' if center else 'legacy_corner']={'v9':metrics(np.linalg.norm(pred[inside]-truth[inside],axis=1)),'v11':metrics(np.linalg.norm((pred+delta)[inside]-truth[inside],axis=1))}
        json_write(out/'metrics.json',report)
        flat=[dict(pool=p,convention=c,method=m,**v) for p,pv in report.get('metadata_agreement',{}).items() for c,cv in pv.items() for m,v in cv.items()]
        csv_write(out/'metrics.csv',flat)
        logging.info('Completed; quality gate=%s',report['quality_gate'])
        return 0 if report['quality_gate'] else 2
    except Exception as exc:
        logging.exception('Run failed')
        json_write(out/'failure.json',dict(reason=str(exc),type=type(exc).__name__))
        return 1
