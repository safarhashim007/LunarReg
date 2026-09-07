import cv2
import numpy as np


def save_registration(out,source,reference,matrix,field=None,prefix=''):
    h,w=reference.shape
    registered=cv2.warpAffine(source,matrix,(w,h))
    if field is not None and np.any(field):
        # Invert y=x+d(x) iteratively. Direct subtraction is not an exact inverse.
        yy,xx=np.mgrid[:h,:w].astype(np.float32); x=xx.copy();y=yy.copy()
        for _ in range(12):
            d=cv2.remap(field.astype(np.float32),x,y,cv2.INTER_LINEAR)
            x=xx-d[:,:,0];y=yy-d[:,:,1]
        registered=cv2.remap(registered,x,y,cv2.INTER_LINEAR)
    for name,im in [('registered',registered),('overlay',cv2.addWeighted(registered,.5,reference,.5,0)),('difference',cv2.absdiff(registered,reference))]:
        if not cv2.imwrite(str(out/f'{prefix}{name}.png'),im): raise IOError('Image output failed')
    return registered
