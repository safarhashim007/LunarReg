"""Experimental V11: trained RoMa fine descriptors, bounded mutual patch search.
No geographic inputs. Rejects ambiguous, border, cross-scale and cycle failures.
"""
import cv2
import numpy as np


def descriptors(image, extractor):
    import torch
    d=next(extractor.parameters()).device
    x=torch.from_numpy(np.repeat(image[None,None],3,axis=1).copy()).to(d).float()/255
    with torch.inference_mode():
        features=extractor(x)[2][0].float().cpu().numpy()
    features=cv2.resize(features,image.shape[::-1],interpolation=cv2.INTER_LINEAR)
    return np.ascontiguousarray(features/np.maximum(np.linalg.norm(features,axis=2,keepdims=True),1e-6),dtype=np.float32)


def cut(image,p,size):
    # remap handles arbitrary feature channels, unlike getRectSubPix.
    axis=np.arange(size,dtype=np.float32)-(size-1)/2
    x,y=np.meshgrid(axis+p[0],axis+p[1])
    return cv2.remap(image,x.astype(np.float32),y.astype(np.float32),cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)


def search(a,b,p,q,size=13,radius=18):
    template=cut(a,p,size)
    target=cut(b,q,size+2*radius)
    numerator=cv2.matchTemplate(target,template,cv2.TM_CCORR)
    energy=np.sum(target*target,axis=2) if target.ndim==3 else target*target
    integral=cv2.integral(energy)
    norm=integral[size:,size:]-integral[:-size,size:]-integral[size:,:-size]+integral[:-size,:-size]
    scores=(numerator/np.maximum(np.sqrt(norm*np.sum(template*template)),1e-9)).astype(np.float32)
    iy,ix=np.unravel_index(np.argmax(scores),scores.shape)
    peak=float(scores[iy,ix]); background=scores.copy()
    background[max(0,iy-2):iy+3,max(0,ix-2):ix+3]=-1
    margin=peak-float(background.max())
    delta=np.array([ix-radius,iy-radius],float)
    if 0<ix<2*radius and 0<iy<2*radius:
        for j,(lo,mid,hi) in enumerate(((scores[iy,ix-1],peak,scores[iy,ix+1]),(scores[iy-1,ix],peak,scores[iy+1,ix]))):
            curvature=lo-2*mid+hi
            if curvature < -1e-8: delta[j]+=np.clip(.5*(lo-hi)/curvature,-.5,.5)
    return delta,peak,margin


def refine(a,b,valid,points,radius=18):
    rows=[]
    h,w=valid.shape
    for p in points:
        row=dict(x=float(p[0]),y=float(p[1]),dx=0.,dy=0.,score=0.,margin=0.,cycle=0.,agreement=0.,reason='border')
        border=radius+11
        if min(p[0],p[1],w-1-p[0],h-1-p[1])<border:
            rows.append(row); continue
        if cut(valid,p,2*border+1).min()<.999:
            row['reason']='warp_border'; rows.append(row); continue
        estimates=[search(a,b,p,p,n,radius) for n in (13,21)]
        delta=np.mean([e[0] for e in estimates],axis=0)
        row.update(dx=float(delta[0]),dy=float(delta[1]),score=min(e[1] for e in estimates),margin=min(e[2] for e in estimates),agreement=float(np.linalg.norm(estimates[0][0]-estimates[1][0])))
        reverse,score,margin=search(b,a,p+delta,p,21,radius)
        row['cycle']=float(np.linalg.norm(reverse))
        if not np.isfinite(delta).all(): row['reason']='nonfinite'
        elif np.max(np.abs(delta))>=radius-1: row['reason']='search_boundary'
        elif row['score']<.85 or score<.85: row['reason']='weak_similarity'
        elif row['margin']<.015 or margin<.015: row['reason']='ambiguous_peak'
        elif row['agreement']>1.0: row['reason']='scale_disagreement'
        elif row['cycle']>1.0: row['reason']='mutual_cycle'
        else: row['reason']='accepted'
        rows.append(row)
    return rows


def supported_field(points,rows,radius=60):
    accepted=[r for r in rows if r['reason']=='accepted']
    output=np.zeros((len(points),2),float)
    if len(accepted)<3: return output
    controls=np.array([[r['x'],r['y']] for r in accepted]); shifts=np.array([[r['dx'],r['dy']] for r in accepted])
    # Require independent neighbouring controls and consistent displacements.
    distance=np.linalg.norm(controls[:,None]-controls[None,:],axis=2)
    keep=np.array([np.sum((distance[i]>1)&(distance[i]<radius))>=2 and np.linalg.norm(shifts[i]-np.median(shifts[(distance[i]>1)&(distance[i]<radius)],axis=0))<=2 for i in range(len(controls))])
    controls,shifts=controls[keep],shifts[keep]
    if len(controls)<3: return output
    for start in range(0,len(points),2048):
        dist=np.linalg.norm(points[start:start+2048,None]-controls[None],axis=2)
        weights=np.maximum(1-dist/radius,0)**2
        supported=(dist<radius).sum(axis=1)>=3
        output[start:start+len(dist)]=(weights@shifts)/np.maximum(weights.sum(axis=1,keepdims=True),1e-9)*supported[:,None]*np.clip(1-dist.min(axis=1)/radius,0,1)[:,None]
    return output
