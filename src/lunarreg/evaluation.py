"""Explicit post-hoc metrics. No registration module imports georeference code."""
import numpy as np


def metrics(errors):
    e=np.asarray(errors,float)
    if e.ndim!=1 or not np.isfinite(e).all(): raise ValueError('Errors must be finite 1D values')
    if not len(e): return dict(count=0,median=None,mean=None,p90=None,rmse=None)
    return dict(count=len(e),median=float(np.median(e)),mean=float(e.mean()),p90=float(np.percentile(e,90)),rmse=float(np.sqrt(np.mean(e*e))),**{f'within_{t:g}px_pct':float(100*np.mean(e<=t)) for t in (.5,1,2,5,10,20)})


def coverage(points,shape):
    if not len(points): return 0.
    h,w=shape
    p=np.asarray(points)
    p=p[(p[:,0]>=0)&(p[:,1]>=0)&(p[:,0]<w)&(p[:,1]<h)]
    return len(np.unique((p/[w,h]*8).astype(int),axis=0))/64*100


def georeference(source_tif,reference_tif,points,source_shape,reference_shape,center=True):
    import rasterio
    from rasterio.warp import transform
    with rasterio.open(source_tif) as a,rasterio.open(reference_tif) as b:
        if (a.height,a.width)!=tuple(source_shape) or (b.height,b.width)!=tuple(reference_shape):
            raise ValueError('GeoTIFF dimensions must match registration images')
        if a.crs is None or b.crs is None: raise ValueError('GeoTIFF CRS missing')
        p=np.asarray(points,dtype=np.float64)+(.5 if center else 0)
        x=a.transform.a*p[:,0]+a.transform.b*p[:,1]+a.transform.c
        y=a.transform.d*p[:,0]+a.transform.e*p[:,1]+a.transform.f
        params=b.crs.to_dict()
        source_params=a.crs.to_dict()
        # GDAL wraps lunar longitudes 340 degrees to -20, but this raster stores
        # unwrapped easting. Preserve the explicitly stored longitude branch
        # for spherical equirectangular CRSs; do not infer it from matches.
        if a.crs.is_geographic and params.get('proj')=='eqc' and 'R' in params and float(source_params.get('R',source_params.get('a',0)))==float(params['R']):
            radius=float(params['R'])
            x=radius*np.cos(np.radians(float(params.get('lat_ts',0))))*np.radians(x-float(params.get('lon_0',0)))+float(params.get('x_0',0))
            y=radius*np.radians(y-float(params.get('lat_0',0)))+float(params.get('y_0',0))
        else:
            x,y=transform(a.crs,b.crs,x.tolist(),y.tolist())
        inv=~b.transform
        return np.column_stack((inv.a*np.array(x)+inv.b*np.array(y)+inv.c,inv.d*np.array(x)+inv.e*np.array(y)+inv.f))-(.5 if center else 0)
