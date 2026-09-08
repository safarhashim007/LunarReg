"""Create an image-only OHRC pair; never imports or runs registration.

Geometry locates the reference crop and supplies evaluation-only controls.
Source TIFF uses GCPs, NOT an affine approximation or an orthorectification.
"""
import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import cv2
import numpy as np
import rasterio
from rasterio.control import GroundControlPoint
from rasterio.windows import Window, from_bounds
from rasterio.warp import transform

ROOT = Path(__file__).resolve().parents[1]
BASE = 'https://pds.lroc.im-ldi.com/data/LRO-L-LROC-5-RDR-V1.0/LROLRC_2001/EXTRAS/BROWSE/NAC_POLE/NAC_POLE_SOUTH_CM_MAX/'
DEFAULT_REFERENCE = BASE + 'NAC_POLE_SOUTH_CM_MAX_P892S2250.TIF'
CRS = '+proj=stere +lat_0=-90 +lat_ts=-90 +lon_0=0 +R=1737400 +units=m +no_defs'
GEO = '+proj=longlat +R=1737400 +no_defs'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path, algorithm='sha256'):
    h = hashlib.new(algorithm)
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


def load_source(raw):
    images = list(raw.glob('data/calibrated/*/*.img'))
    require(len(images) == 1, 'Expected exactly one OHRC image')
    img = images[0]
    label = ET.parse(img.with_suffix('.xml')).getroot()
    get = lambda tag: label.find('.//{*}' + tag).text.strip()
    axes = {a.find('{*}axis_name').text: int(a.find('{*}elements').text)
            for a in label.findall('.//{*}Axis_Array')}
    require(get('data_type') == 'UnsignedByte', 'Only labeled UnsignedByte supported')
    require(get('axis_index_order') == 'Last Index Fastest', 'Unexpected axis order')
    require(int(get('offset')) == 0, 'Nonzero image offset unsupported')
    shape = (axes['Line'], axes['Sample'])
    require(img.stat().st_size == int(get('file_size')) == int(np.prod(shape)), 'Image size mismatch')
    require(digest(img, 'md5') == get('md5_checksum'), 'ISRO image checksum mismatch')
    csvs = list(raw.glob('geometry/calibrated/*/*.csv'))
    require(len(csvs) == 1, 'Expected exactly one geometry CSV')
    csv = csvs[0]
    cl = ET.parse(csv.with_suffix('.xml')).getroot()
    require(digest(csv, 'md5') == cl.find('.//{*}md5_checksum').text.strip(), 'Geometry checksum mismatch')
    require(csv.read_text().splitlines()[0].strip() == 'Longitude,Latitude,Pixel,Scan', 'Wrong geometry columns')
    g = np.loadtxt(csv, delimiter=',', skiprows=1)
    require(np.isfinite(g).all(), 'Nonfinite geometry')
    require(((g[:, 0] >= 0) & (g[:, 0] < 360) & (g[:, 1] >= -90) & (g[:, 1] <= 90)).all(), 'Expected east longitude 0..360 and valid latitude')
    px, rows = np.unique(g[:, 2]), np.unique(g[:, 3])
    require(len(px)*len(rows) == len(g), 'Incomplete geometry grid')
    require(np.array_equal(g[:,2:], g[:,2:].astype(int)), 'Noninteger pixel indices')
    require((px[0],px[-1],rows[0],rows[-1]) == (0,shape[1]-1,0,shape[0]-1), 'Geometry must span zero-based native raster')
    require(np.array_equal(g[:,2],np.tile(px,len(rows))) and np.array_equal(g[:,3],np.repeat(rows,len(px))), 'Geometry row order mismatch')
    # Check all four labeled corners: prevents accidental flip, transpose or wrong product.
    for name, index in [('upper_left',0),('upper_right',len(px)-1),('lower_left',-len(px)),('lower_right',-1)]:
        lon = float(get(name+'_longitude')); lat = float(get(name+'_latitude'))
        require(abs((g[index,0]-lon+180)%360-180)<2e-5 and abs(g[index,1]-lat)<2e-5, 'Corner orientation mismatch: '+name)
    return img, csv, shape, g


def normalize(a, valid):
    require(valid.any(), 'No valid pixels')
    lo, hi = np.percentile(a[valid], [1,99])
    require(hi > lo, 'No usable image contrast')
    out = np.clip((a.astype(float)-lo)*255/(hi-lo),0,255).astype('uint8')
    out[~valid]=0
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw',type=Path,default=ROOT/'data/validation/ohrc_001/raw/extracted')
    p.add_argument('--reference',default=DEFAULT_REFERENCE)
    p.add_argument('--output',type=Path,default=ROOT/'data/test_pairs/pair_ohrc_001')
    p.add_argument('--row-start',type=int,default=0)
    p.add_argument('--size',type=int,default=12000)
    p.add_argument('--downsample',type=int,default=4)
    p.add_argument('--min-valid',type=float,default=.95)
    a=p.parse_args()
    require(not a.output.exists(), 'Output exists; choose a new --output (never overwrite)')
    require(a.downsample>0 and a.size>0 and a.size%a.downsample==0,'Size must be a positive multiple of downsample')
    img,csv,shape,g=load_source(a.raw)
    r=a.row_start; n=a.size; d=a.downsample
    require(r>=0 and r+n<=shape[0] and n<=shape[1], 'Crop outside native image')
    # Crop choice is fixed before reading image content. No matching/ranking.
    selected=g[(g[:,3]>=r)&(g[:,3]<r+n)&(g[:,2]<n)]
    require(len(selected)>20,'Too few geometry controls')
    x,y=transform(GEO,CRS,((selected[:,0]+180)%360-180).tolist(),selected[:,1].tolist())
    x=np.array(x); y=np.array(y)
    raw=np.memmap(img,dtype='uint8',mode='r',shape=shape)
    source=cv2.resize(np.asarray(raw[r:r+n,:n]),(n//d,n//d),interpolation=cv2.INTER_AREA)
    # No special constants are declared in the ISRO label: zero is a valid DN.
    sv=np.ones(source.shape,dtype=bool)
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR',CPL_VSIL_CURL_ALLOWED_EXTENSIONS='.TIF,.tif',GDAL_HTTP_TIMEOUT='45',GDAL_HTTP_MAX_RETRY='1'):
        with rasterio.open(a.reference) as ref:
            require(ref.crs is not None,'Reference has no CRS')
            require(abs(ref.crs.to_dict().get('R',ref.crs.to_dict().get('a',0))-1737400)<1,'Reference must use lunar radius 1737400 m')
            rx,ry=transform(CRS,ref.crs,x.tolist(),y.tolist())
            # 100 m padding exceeds one geometry grid interval at native resolution.
            w=from_bounds(min(rx)-100,min(ry)-100,max(rx)+100,max(ry)+100,ref.transform)
            w=Window(int(np.floor(w.col_off)),int(np.floor(w.row_off)),int(np.ceil(w.width))+2,int(np.ceil(w.height))+2)
            require(w.col_off>=0 and w.row_off>=0 and w.col_off+w.width<=ref.width and w.row_off+w.height<=ref.height,'Reference tile does not contain crop; supply matching --reference')
            lro=ref.read(1,window=w); lv=ref.read_masks(1,window=w)>0
            lt=ref.window_transform(w); lc=ref.crs; nd=ref.nodata
            reference_info={'shape':[ref.height,ref.width],'crs':ref.crs.to_wkt(),'window':[w.col_off,w.row_off,w.width,w.height]}
    require(lv.mean()>=a.min_valid,'LRO valid fraction below threshold')
    # GDAL GCP coordinates use pixel corners; CSV integer indices denote centers.
    sx=(selected[:,2]+.5)/d-.5; sy=(selected[:,3]-r+.5)/d-.5
    inv=~lt
    lx,ly=inv*(np.asarray(rx),np.asarray(ry)); lx-=.5; ly-=.5
    a.output.mkdir(parents=True,exist_ok=False)
    gcps=[GroundControlPoint(row=float(sy[i]+.5),col=float(sx[i]+.5),x=float(x[i]),y=float(y[i])) for i in range(0,len(x),max(1,len(x)//400))]
    with rasterio.open(a.output/'chandrayaan.tif','w',driver='GTiff',height=source.shape[0],width=source.shape[1],count=1,dtype='uint8',compress='deflate',gcps=gcps,crs=CRS) as f: f.write(source,1)
    with rasterio.open(a.output/'lro.tif','w',driver='GTiff',height=lro.shape[0],width=lro.shape[1],count=1,dtype=lro.dtype,transform=lt,crs=lc,nodata=nd,compress='deflate') as f:
        f.write(lro,1); f.write_mask(lv.astype('uint8')*255)
    sp=normalize(source,sv); lp=normalize(lro,lv)
    for name,arr in [('chandrayaan',sp),('lro',lp)]: require(cv2.imwrite(str(a.output/(name+'.png')),arr),'PNG write failed')
    previews=[cv2.resize(z,(round(z.shape[1]*500/z.shape[0]),500)) for z in (sp,lp)]
    require(cv2.imwrite(str(a.output/'side_by_side.png'),np.hstack(previews)),'Preview write failed')
    np.savetxt(a.output/'geometry_controls_EVALUATION_ONLY.csv',np.column_stack([sx,sy,lx,ly,selected]),delimiter=',',header='source_x,source_y,lro_x,lro_y,longitude_east_360,latitude,native_pixel,native_scan',comments='')
    meta={'pair':a.output.name,'held_out':True,'registration_run':False,'source_chandrayaan':str(img),'source_geometry':str(csv),'source_reference':a.reference,'source_sha256':digest(img),'geometry_sha256':digest(csv),'native_shape':shape,'native_crop':[0,r,n,n],'downsample':d,'chandrayaan_shape':list(source.shape),'lro_shape':list(lro.shape),'chandrayaan_valid_fraction':float(sv.mean()),'lro_valid_fraction':float(lv.mean()),'aoi_geometry_samples':{'lon_min':float(selected[:,0].min()),'lon_max':float(selected[:,0].max()),'lat_min':float(selected[:,1].min()),'lat_max':float(selected[:,1].max())},'reference':reference_info,'geometry_controls':len(selected),'source_representation':'Native scan order, area-downsampled; GCP geolocation only, no affine transform or geometry warp.','pixel_convention':'Zero-based centers in control CSV; GCPs use GDAL corner coordinates. East-positive longitude normalized only for projection.','validity':'OHRC label declares no nodata; all DNs including zero count valid. LRO uses dataset mask. Valid does not mean illuminated.','evaluation_policy':'Use only PNG pixels to select RoMa correspondences and fit LunarReg. Load geometry controls only after image-only transform is frozen. Controls measure agreement with ISRO system geometry and LRO mapping, not independently surveyed absolute truth. No controls were used to fit registration.','selection_policy':'Fixed native crop; reference selected by geographic coverage. No image matching or registration-quality selection.'}
    meta['output_sha256']={f.name:digest(f) for f in a.output.iterdir() if f.is_file()}
    (a.output/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    for name,expected in [('chandrayaan',source),('lro',lro)]:
        with rasterio.open(a.output/(name+'.tif')) as f: require(np.array_equal(f.read(1),expected),'TIFF roundtrip failed')
        require(cv2.imread(str(a.output/(name+'.png')),0).shape==expected.shape,'PNG shape mismatch')
    print(json.dumps(meta,indent=2))

if __name__=='__main__':
    main()
