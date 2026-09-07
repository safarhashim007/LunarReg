"""Repeatable learned-descriptor rejection checks, independent of GeoTIFF metadata."""
import numpy as np
from .local_refinement import descriptors,refine


def negative_local_checks(source,extractor,seed=42):
    rng=np.random.default_rng(seed)
    h,w=source.shape
    points=np.stack(np.meshgrid(np.linspace(55,w-56,8),np.linspace(55,h-56,8)),axis=-1).reshape(-1,2)
    a=descriptors(source,extractor)
    results={}
    for name,target in [('unrelated_noise',rng.integers(0,256,source.shape,dtype=np.uint8)),('textureless',np.full(source.shape,127,np.uint8))]:
        rows=refine(a,descriptors(target,extractor),np.ones(source.shape,np.float32),points)
        accepted=sum(r['reason']=='accepted' for r in rows)
        results[name]=dict(controls=len(rows),false_accept_count=accepted,false_accept_pct=100*accepted/len(rows))
    return results
