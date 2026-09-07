"""V9 frozen affine + image-only, bounded gradient phase local corrections.

No RoMa rerun, global fit, or georeference information in the local estimator.
The saved affine is ALWAYS the V9 matrix; local displacement is a separate field
on the V9-warped LRO grid. Rejected/unsupported points retain V9 predictions.
GeoTIFF evaluation uses V9's corner-coordinate convention for comparability;
pixel-center metrics are also supplied. Benchmark selection is reported as such,
not an independent claim of generalization to unseen imagery.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import math
import shutil
from collections import Counter
import cv2
import numpy as np
import rasterio


def apply(M, p):
    return np.asarray(p) @ M[:, :2].T + M[:, 2]


def gradient(im):
    im = cv2.GaussianBlur(im.astype(np.float32), (0, 0), 1.0)
    gx = cv2.Scharr(im, cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(im, cv2.CV_32F, 0, 1)
    g = np.log1p(cv2.magnitude(gx, gy))
    mu = cv2.GaussianBlur(g, (0, 0), 5)
    sd = np.sqrt(np.maximum(cv2.GaussianBlur(g*g, (0, 0), 5)-mu*mu, .01))
    return np.ascontiguousarray(np.clip((g-mu)/sd, -3, 3), dtype=np.float32)


def patch(im, p, n):
    return cv2.getRectSubPix(im, (n, n), tuple(map(float, p)))


def ncc(a, b):
    a, b = a-a.mean(), b-b.mean()
    return float(np.sum(a*b)/max(float(np.linalg.norm(a)*np.linalg.norm(b)), 1e-8))


def phase(a, b):
    # Copies: OpenCV can multiply inputs by its window in-place.
    shift, response = cv2.phaseCorrelate(a.copy(), b.copy(), cv2.createHanningWindow(a.shape[::-1], cv2.CV_32F))
    return np.asarray(shift), float(response)


def bounded_phase(a,b):
 win=cv2.createHanningWindow(a.shape[::-1],cv2.CV_32F)
 f=np.fft.fft2((a-a.mean())*win);g=np.fft.fft2((b-b.mean())*win); c=g*f.conj();c/=np.maximum(np.abs(c),1e-8)
 corr=np.fft.fftshift(np.fft.ifft2(c).real);cy,cx=np.array(corr.shape)//2
 region=corr[cy-15:cy+16,cx-15:cx+16];iy,ix=np.unravel_index(region.argmax(),region.shape); yy,xx=cy+iy-15,cx+ix-15
 bg=region.copy();bg[max(0,iy-2):iy+3,max(0,ix-2):ix+3]=np.nan
 psr=(region[iy,ix]-np.nanmean(bg))/max(np.nanstd(bg),1e-8)
 d=np.array([xx-cx,yy-cy],float)
 for j,(v1,v2,v3) in enumerate([(corr[yy,xx-1],corr[yy,xx],corr[yy,xx+1]),(corr[yy-1,xx],corr[yy,xx],corr[yy+1,xx])]):
  d[j]+=np.clip(.5*(v1-v3)/(v1-2*v2+v3+1e-12),-.5,.5)
 return d,float(np.clip((psr-3)/15,0,1))


def magnitude_gradient(im):
    im=cv2.GaussianBlur(im.astype(np.float32),(0,0),1.5)
    return np.sqrt(cv2.magnitude(cv2.Scharr(im,cv2.CV_32F,1,0),cv2.Scharr(im,cv2.CV_32F,0,1))+1)


def local(gA, gB, valid, p, estimator=phase):
    row = dict(x=float(p[0]), y=float(p[1]), dx=0., dy=0., response=0., cycle=999., agreement=999., ncc_before=0., ncc_after=0., reason='border')
    h, w = gA.shape
    if min(p[0], p[1], w-1-p[0], h-1-p[1]) < 56:
        return row
    if patch(valid, p, 83).min() < .999:
        row['reason']='warp_border'; return row
    estimates, responses = [], []
    for size in (48, 64, 80):
        a,b=patch(gA,p,size),patch(gB,p,size)
        d,r=estimator(a,b)
        estimates.append(d); responses.append(r)
    d = np.median(estimates, axis=0)
    row.update(dx=float(d[0]),dy=float(d[1]),response=float(min(responses)),agreement=float(np.max(np.linalg.norm(np.asarray(estimates)-d,axis=1))))
    if not np.isfinite(d).all() or not np.isfinite(responses).all():
        row['reason']='nonfinite'; return row
    if np.linalg.norm(d)>15:
        row['reason']='search_bound'; return row
    if min(responses)<.12:
        row['reason']='phase_response'; return row
    if row['agreement']>3:
        row['reason']='scale_disagreement'; return row
    # Recenter target; reverse registration must return to the source center.
    a=patch(gA,p,65); b=patch(gB,p+d,65)
    reverse, rr=estimator(b,a)
    row['cycle']=float(np.linalg.norm(reverse))
    row['ncc_before']=ncc(a,patch(gB,p,65)); row['ncc_after']=ncc(a,b)
    if rr<.12 or row['cycle']>2:
        row['reason']='recentered_cycle'; return row
    if row['ncc_after']<.15 or row['ncc_after']-row['ncc_before']<.02:
        row['reason']='ncc_validation'; return row
    row['reason']='accepted'
    return row


def field_at(points, controls, shifts, weights, radius=60):
    result=np.zeros((len(points),2),np.float64)
    if len(controls)<3: return result
    for start in range(0,len(points),20000):
        q=points[start:start+20000]
        all_dist=np.linalg.norm(q[:,None,:]-controls[None,:,:],axis=2)
        idx=np.argsort(all_dist,axis=1)[:,:min(8,len(controls))]
        dist=np.take_along_axis(all_dist,idx,axis=1)
        ww=np.maximum(1-dist/radius,0)**2*weights[idx]
        supported=(dist<radius).sum(axis=1)>=3
        # Compact support: gradually return to zero near the support boundary.
        taper=np.clip(1-dist[:,0]/radius,0,1)
        result[start:start+len(q)] = ((ww[:,:,None]*shifts[idx]).sum(axis=1)/np.maximum(ww.sum(axis=1)[:,None],1e-12))*supported[:,None]*taper[:,None]
    return result


def expected_geo(pair, A, center=False):
    with rasterio.open(pair/'chandrayaan.tif') as a, rasterio.open(pair/'lro.tif') as b:
        # Check assumptions of the V9 evaluator rather than silently using them.
        assert a.crs.is_geographic and b.crs.is_projected
        assert 'Equirectangular' in b.crs.to_wkt() and '1737400' in a.crs.to_wkt()
        assert (a.height,a.width)==cv2.imread(str(pair/'chandrayaan.png'),0).shape
        assert (b.height,b.width)==cv2.imread(str(pair/'lro.png'),0).shape
        T,U=a.transform,~b.transform
        p=A+(.5 if center else 0)
        lon=T.a*p[:,0]+T.b*p[:,1]+T.c
        lat=T.d*p[:,0]+T.e*p[:,1]+T.f
        x=1737400*np.radians(lon)*math.cos(math.radians(10)); y=1737400*np.radians(lat)
        return np.column_stack([U.a*x+U.b*y+U.c,U.d*x+U.e*y+U.f])-(.5 if center else 0)


def metrics(e):
    if not len(e): return {'count':0}
    return dict(count=len(e),median=float(np.median(e)),mean=float(np.mean(e)),p90=float(np.percentile(e,90)), **{f'within_{t}px_pct':float(100*np.mean(e<=t)) for t in (1,2,5,10,20)})


def coverage(p, shape):
    if not len(p): return 0.
    h,w=shape
    return len(np.unique(np.clip((p/[w,h]*8).astype(int),0,7),axis=0))/64*100


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]); args=parser.parse_args()
    root=args.root; pair=root/'data/test_pairs/pair_002'; out=root/'outputs/pair_002'; out.mkdir(parents=True,exist_ok=True)
    hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.glob('roma_v9_*')}
    M=np.load(out/'roma_v9_transform.npy'); A=np.load(out/'roma_v9_relaxed_A.npy').astype(float)
    ch=cv2.imread(str(pair/'chandrayaan.png'),0); lro=cv2.imread(str(pair/'lro.png'),0)
    if ch is None or lro is None: raise RuntimeError('Missing input image')
    h,w=lro.shape; P=apply(M,A)
    coarse=cv2.warpAffine(ch,M,(w,h)); valid=cv2.warpAffine(np.ones(ch.shape,np.float32),M,(w,h))
    ga,gb=gradient(coarse),gradient(lro)
    rows=[local(ga,gb,valid,p) for p in P]
    accepted=np.array([r['reason']=='accepted' for r in rows])
    D=np.array([[r['dx'],r['dy']] for r in rows]); quality=np.array([max(r['response'],0)*max(r['ncc_after'],0)*np.exp(-r['cycle']/2) for r in rows])
    # Eliminate locally inconsistent corrections without any global model fit.
    if accepted.sum()>=4:
        inds=np.flatnonzero(accepted)
        for i in inds:
            neighbours=inds[np.linalg.norm(P[inds]-P[i],axis=1)<70]
            neighbours=neighbours[neighbours!=i]
            if len(neighbours)<2 or np.linalg.norm(D[i]-np.median(D[neighbours],axis=0))>3:
                accepted[i]=False; rows[i]['reason']='neighbour_consistency'
    else:
        for i in np.flatnonzero(accepted): rows[i]['reason']='insufficient_support'
        accepted[:]=False
    # Spatial balancing prevents clustered RoMa points dominating a local field.
    cells=np.clip((P/[w,h]*8).astype(int),0,7); counts=Counter(map(tuple,cells[accepted]))
    for i in np.flatnonzero(accepted): quality[i]/=counts[tuple(cells[i])]
    # Safer independent variant: smooth gradient magnitude, phase peak search
    # restricted to +/-15 before estimation; then damp and cap the correction.
    safe_ga,safe_gb=magnitude_gradient(coarse),magnitude_gradient(lro)
    safe_rows=[local(safe_ga,safe_gb,valid,p,estimator=bounded_phase) for p in P]
    conservative=np.array([r['reason']=='accepted' and r['agreement']<=1.5 and r['cycle']<=1 for r in safe_rows])
    safe_D=np.array([[r['dx'],r['dy']] for r in safe_rows])
    safe_quality=np.array([max(r['response'],0)*max(r['ncc_after'],0)*np.exp(-r['cycle']/2) for r in safe_rows])
    safe_inds=np.flatnonzero(conservative)
    for i in safe_inds:
        neighbours=safe_inds[(np.linalg.norm(P[safe_inds]-P[i],axis=1)<70)&(safe_inds!=i)]
        if len(neighbours)<2 or np.linalg.norm(safe_D[i]-np.median(safe_D[neighbours],axis=0))>2:
            conservative[i]=False;safe_rows[i]['reason']='neighbour_consistency'
    safe_counts=Counter(map(tuple,cells[conservative]))
    for i in np.flatnonzero(conservative):safe_quality[i]/=safe_counts[tuple(cells[i])]
    variants={'phase':(accepted,1.), 'conservative':(conservative,.35)}
    y,x=np.mgrid[:h,:w]; grid=np.column_stack([x.ravel(),y.ravel()]).astype(float)
    fields={}
    for name,(mask,gain) in variants.items():
        shift=(safe_D if name=='conservative' else D)[mask].copy()
        if name=='conservative': shift*=np.minimum(1,6/np.maximum(np.linalg.norm(shift,axis=1),1e-9))[:,None]
        fields[name]=field_at(grid,P[mask],shift,(safe_quality if name=='conservative' else quality)[mask])*gain
        np.save(out/f'roma_v10_{name}_field.npy',fields[name].reshape(h,w,2).astype(np.float32))
    # Registration complete. Only now load georeferencing and evaluate all variants.
    cache=np.load(out/'roma_cycle_evaluation.npz')
    pools={'v9_controls':A, 'cached_sample':cache['ptsA'].astype(float)}
    # Cached RoMa scores are kept with their exact original points, never
    # misattributed to V9's separately sampled controls.
    quality_mask=(cache['combined_confidence']>=.2)&(cache['cycle_error_equiv']<=6)
    pools['cached_high_quality']=cache['ptsA'][quality_mask].astype(float)
    np.savez_compressed(out/'roma_v10_cached_quality.npz',source=cache['ptsA'],confidence=cache['combined_confidence'],cycle_error_lro_px=cache['cycle_error_equiv'],high_quality_mask=quality_mask)
    # Independent, evenly distributed lattice reduces dependence on the old sampler.
    gy,gx=np.mgrid[0:ch.shape[0]:80,0:ch.shape[1]:80]; pools['uniform_grid']=np.column_stack([gx.ravel(),gy.ravel()]).astype(float)
    report={'historical_v9':{'median':8.419,'mean':8.821,'p90':13.682},'notes':['V9 full sample and per-control RoMa quality were not saved; historical sample cannot be reproduced exactly.','Local phase response, recentered reverse cycle and cross-scale consistency replace unavailable per-control scores; no nearest-match attribution.','All variants fixed before GeoTIFF evaluation. Promotion on this pair is benchmark selection, not held-out validation.','Rejected/unsupported regions keep V9; field is not a new affine transform.'], 'rejection_counts':dict(Counter(r['reason'] for r in rows)), 'safer_rejection_counts':dict(Counter(r['reason'] for r in safe_rows)), 'controls':len(A), 'accepted':int(accepted.sum()),'conservative_accepted':int(conservative.sum()),'source_coverage_pct':coverage(A,ch.shape),'accepted_source_coverage_pct':coverage(A[accepted],ch.shape),'conservative_source_coverage_pct':coverage(A[conservative],ch.shape),'evaluation':{}}
    for pool,points in pools.items():
        pred=apply(M,points); expected=expected_geo(pair,points)
        inside=(expected[:,0]>=0)&(expected[:,0]<w)&(expected[:,1]>=0)&(expected[:,1]<h)
        result={'v9':metrics(np.linalg.norm(pred[inside]-expected[inside],axis=1))}
        for name,f in fields.items():
            displacement=np.column_stack([cv2.remap(f[:,j].reshape(h,w).astype(np.float32),pred[:,0].astype(np.float32).reshape(-1,1),pred[:,1].astype(np.float32).reshape(-1,1),cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT).ravel() for j in range(2)])
            refined=pred+displacement
            error=np.linalg.norm(refined-expected,axis=1); changed=np.linalg.norm(displacement,axis=1)>1e-6
            result[name]=metrics(error[inside]); result[name]['changed_count']=int((changed&inside).sum())
            result[name]['changed_before']=metrics(np.linalg.norm(pred[inside&changed]-expected[inside&changed],axis=1)); result[name]['changed_after']=metrics(error[inside&changed])
            centered=expected_geo(pair,points,True)
            result[name]['pixel_center_before']=metrics(np.linalg.norm(pred[inside]-centered[inside],axis=1)); result[name]['pixel_center_after']=metrics(np.linalg.norm(refined[inside]-centered[inside],axis=1))
            np.savez_compressed(out/f'roma_v10_{pool}_{name}.npz',source=points,before=pred,after=refined,expected_evaluation_only=expected,inside=inside,changed=changed)
        report['evaluation'][pool]=result
    selected='v9'
    for name in variants:
        # Require all aggregate errors to improve on both broad evaluation pools.
        if all(all(report['evaluation'][pool][name][k]<report['evaluation'][pool]['v9'][k] for k in ('median','mean','p90')) for pool in ('cached_sample','uniform_grid')):
            selected=name; break
    report['selected']=selected; report['improved']=selected!='v9'
    report['configuration']={'patch_sizes':[48,64,80],'max_correction_lro_px':15,'minimum_phase_response':.12,'max_cross_scale_disagreement_px':3,'max_recentered_cycle_px':2,'minimum_gradient_ncc':.15,'minimum_ncc_gain':.02,'field_support_radius_px':60,'minimum_support_neighbours':3,'safe_gain':.35,'safe_max_raw_shift_px':6,'bounded_phase_confidence':'clip((peak_to_sidelobe_zscore - 3)/15, 0, 1)'}
    final=np.zeros((h*w,2)) if selected=='v9' else fields[selected]
    np.save(out/'roma_v10_transform.npy',M); np.save(out/'roma_v10_displacement.npy',final.reshape(h,w,2).astype(np.float32))
    # Invert q=p+d(p) by fixed point iteration; do not simply negate a spatial field.
    for name,f in {**fields,'final':final}.items():
        field=f.reshape(h,w,2).astype(np.float32); source=grid.copy().astype(np.float32)
        for _ in range(12):
            d=cv2.remap(field,source[:,0].reshape(h,w),source[:,1].reshape(h,w),cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT).reshape(-1,2)
            source=grid.astype(np.float32)-d
        registered=cv2.remap(coarse,source[:,0].reshape(h,w),source[:,1].reshape(h,w),cv2.INTER_LINEAR)
        cv2.imwrite(str(out/f'roma_v10_{name}_registered.png'),registered)
        cv2.imwrite(str(out/f'roma_v10_{name}_overlay.png'),cv2.addWeighted(registered,.5,lro,.5,0))
        cv2.imwrite(str(out/f'roma_v10_{name}_difference.png'),cv2.absdiff(registered,lro))
    vis=cv2.cvtColor(lro,cv2.COLOR_GRAY2BGR)
    for i,p in enumerate(P):
        if accepted[i]: cv2.arrowedLine(vis,tuple(np.rint(p).astype(int)),tuple(np.rint(p+D[i]).astype(int)),(0,255,0),1,tipLength=.3)
        else: cv2.circle(vis,tuple(np.rint(p).astype(int)),1,(0,0,180),-1)
    for k in range(1,8):
        cv2.line(vis,(k*w//8,0),(k*w//8,h-1),(100,100,100),1); cv2.line(vis,(0,k*h//8),(w-1,k*h//8),(100,100,100),1)
    cv2.imwrite(str(out/'roma_v10_local_corrections.png'),vis)
    for name,rr,mask in [('phase',rows,accepted),('conservative',safe_rows,conservative)]:
        for i,r in enumerate(rr):r['field_accepted']=bool(mask[i]);r['source_x']=float(A[i,0]);r['source_y']=float(A[i,1])
        with open(out/f'roma_v10_{name}_local_refinement.csv','w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rr[0])); writer.writeheader(); writer.writerows(rr)
    shutil.copyfile(out/'roma_v10_phase_local_refinement.csv',out/'roma_v10_local_refinement.csv')
    with open(out/'roma_v10_summary.csv','w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['pool','variant','count','median','mean','p90']+[f'within_{t}px_pct' for t in (1,2,5,10,20)])
        for pool,results in report['evaluation'].items():
            for name,values in results.items():writer.writerow([pool,name]+[values.get(k) for k in ('count','median','mean','p90','within_1px_pct','within_2px_pct','within_5px_pct','within_10px_pct','within_20px_pct')])
    assert hashes=={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.glob('roma_v9_*')},'V9 outputs changed!'
    report['v9_sha256']=hashes
    (out/'roma_v10_metrics.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    print(json.dumps({k:v for k,v in report.items() if k not in ('v9_sha256','evaluation')},indent=2))
    for pool,r in report['evaluation'].items():
        print(pool,json.dumps({k:{m:v for m,v in s.items() if m in ('count','median','mean','p90','changed_count')} for k,s in r.items()}))
    print('Completed. Selected:',selected)


if __name__=='__main__': main()
