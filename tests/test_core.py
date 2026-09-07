import contextlib
import io
import tempfile
import unittest
from pathlib import Path
import cv2
import numpy as np
from lunarreg.geometry import apply_transform,weighted_similarity_fit,weighted_affine_fit,constrain_affine,make_similarity
from lunarreg.cycle import cycle_metrics
from lunarreg.evaluation import metrics,coverage
from lunarreg.local_refinement import refine,supported_field
from lunarreg.io import read_image
from lunarreg.global_refinement import solve

class GeometryTests(unittest.TestCase):
    def setUp(self):
        self.rng=np.random.default_rng(42)
        self.A=self.rng.uniform(0,400,(400,2))
        self.M=np.array([[1.01,-.02,2.35],[.02,1.01,-1.65]])
    def test_apply(self):
        np.testing.assert_allclose(apply_transform(self.M,np.array([[0,0],[1,0]])),[[2.35,-1.65],[3.36,-1.63]])
    def test_weighted_similarity(self):
        np.testing.assert_allclose(weighted_similarity_fit(self.A,apply_transform(self.M,self.A),np.ones(400)),self.M,atol=1e-8)
    def test_affine(self):
        M=self.M.copy();M[0,1]+=.006
        np.testing.assert_allclose(weighted_affine_fit(self.A,apply_transform(M,self.A),np.ones(400)),M,atol=1e-8)
    def test_degenerate(self):
        self.assertIsNone(weighted_affine_fit(np.ones((5,2)),np.ones((5,2)),np.ones(5)))
        self.assertIsNone(weighted_similarity_fit(self.A,self.A,np.zeros(400)))
    def test_constraints(self):
        self.assertIsNotNone(constrain_affine(self.M,self.M))
        for M in [self.M*np.nan,np.array([[-1,0,0],[0,1,0]]),np.array([[2,0,0],[0,1,0]]),self.M+np.array([[0,0,20],[0,0,0]])]:
            self.assertIsNone(constrain_affine(M,self.M))
    def test_cycle(self):
        e,c,s=cycle_metrics(np.array([[0,0]]),np.array([[3,4]]),np.array([.81]),np.array([.49]),.25)
        np.testing.assert_allclose(e,[1.25]);np.testing.assert_allclose(c,[.63]);np.testing.assert_allclose(s,[.28])
    def test_metrics(self):
        m=metrics([0,.5,1,2,5]);self.assertEqual(m['within_1px_pct'],60.)
        self.assertIsNone(metrics([])['rmse'])
        self.assertEqual(coverage(np.empty((0,2)),(100,100)),0)
    def test_known_global_with_outliers(self):
        B=apply_transform(self.M,self.A)+self.rng.normal(0,.05,self.A.shape)
        B[:60]=self.rng.uniform(0,400,(60,2))
        with tempfile.TemporaryDirectory() as out,contextlib.redirect_stdout(io.StringIO()):
            result=solve(self.A,B,np.ones(400),np.zeros(400),(400,400),(400,400),out)
        self.assertTrue(result['passed'])
        self.assertLess(np.median(np.linalg.norm(apply_transform(result['transform'],self.A)-apply_transform(self.M,self.A),axis=1)),.15)
    def test_insufficient(self):
        with tempfile.TemporaryDirectory() as out,self.assertRaises(ValueError):
            solve(self.A[:3],self.A[:3],np.ones(3),np.zeros(3),(400,400),(400,400),out)
    def test_input(self):
        with self.assertRaises(FileNotFoundError):read_image('/does/not/exist.png')
        with tempfile.TemporaryDirectory() as out:
            p=Path(out)/'blank.png';cv2.imwrite(str(p),np.zeros((80,80),np.uint8))
            with self.assertRaises(ValueError):read_image(p)

class LocalTests(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(7)
        a=cv2.GaussianBlur(rng.normal(size=(160,160,4)).astype(np.float32),(0,0),1)
        self.a=a;self.valid=np.ones((160,160),np.float32);self.p=np.array([[70.,70.],[90.,70.],[70.,90.],[90.,90.]])
    def test_positive(self):
        b=cv2.warpAffine(self.a,np.array([[1,0,2.35],[0,1,-1.65]],float),(160,160))
        rows=refine(self.a,b,self.valid,self.p)
        self.assertTrue(all(r['reason']=='accepted' for r in rows),rows)
        self.assertLess(max(np.linalg.norm([r['dx']-2.35,r['dy']+1.65]) for r in rows),.3)
    def test_negative(self):
        b=np.random.default_rng(9).normal(size=self.a.shape).astype(np.float32)
        self.assertTrue(all(r['reason']!='accepted' for r in refine(self.a,b,self.valid,self.p)))
    def test_ambiguous(self):
        a=np.ones_like(self.a)
        self.assertTrue(all(r['reason']!='accepted' for r in refine(a,a,self.valid,self.p)))
    def test_border(self):
        self.assertEqual(refine(self.a,self.a,self.valid,np.array([[1,1]]))[0]['reason'],'border')
    def test_field_no_support(self):
        np.testing.assert_array_equal(supported_field(self.p,[]),np.zeros((4,2)))



class IntegrationGuards(unittest.TestCase):
    def test_many_feature_channels(self):
        from lunarreg.local_refinement import search
        a=np.random.default_rng(3).normal(size=(70,70,128)).astype(np.float32)
        delta,score,margin=search(a,a,np.array([35.,35.]),np.array([35.,35.]))
        self.assertGreater(score,.999)
        self.assertLess(np.linalg.norm(delta),.1)
    def test_geographic_precision_and_branch(self):
        import rasterio
        from rasterio.transform import Affine
        from lunarreg.evaluation import georeference
        with tempfile.TemporaryDirectory() as out:
            a=Path(out)/'a.tif'; b=Path(out)/'b.tif'
            for p,crs,t in [(a,'+proj=longlat +R=1737400 +no_defs',Affine(.001,0,340,0,-.001,10)),(b,'+proj=eqc +lat_ts=10 +R=1737400 +units=m +no_defs',Affine(20,0,10150000,0,-20,303000))]:
                with rasterio.open(p,'w',driver='GTiff',width=100,height=100,count=1,dtype='uint8',crs=crs,transform=t) as ds:ds.write(np.zeros((1,100,100),np.uint8))
            point=np.array([[30,40]],np.float32)
            expected=np.array([[(1737400*np.cos(np.radians(10))*np.radians(340.03)-10150000)/20,(303000-1737400*np.radians(9.96))/20]])
            np.testing.assert_allclose(georeference(a,b,point,(100,100),(100,100),False),expected,atol=1e-8)
    def test_output_preservation(self):
        from argparse import Namespace
        from lunarreg.pipeline import run
        with tempfile.TemporaryDirectory() as out:
            p=Path(out)/'existing';p.mkdir();(p/'sentinel').write_text('preserve')
            with self.assertRaises(FileExistsError):run(Namespace(output=str(p)))
            self.assertEqual((p/'sentinel').read_text(),'preserve')

if __name__=='__main__':unittest.main()
