"""Python file-open guard for blind fitting; image decoders receive PNGs only."""
import os
import sys

class GeometrySeal:
    def __init__(self):
        self.active=True
        self.denied=[]
        sys.addaudithook(self.audit)

    def audit(self,event,args):
        if not self.active or event!='open' or not args: return
        try: name=os.fsdecode(args[0]).lower()
        except TypeError: return
        if name.endswith(('.tif','.tiff')) or 'geometry_controls_evaluation_only' in name:
            self.denied.append(name)
            raise PermissionError('Geometry is sealed until the image-only transform is frozen')

    def close(self):
        self.active=False
