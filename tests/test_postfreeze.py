import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from lunarreg.postfreeze import evaluate, error_metrics
from lunarreg.io import sha256

class PostfreezeTests(unittest.TestCase):
    def test_strict_thresholds(self):
        m=error_metrics([0,.5,1,2,5])
        self.assertEqual(m['pct_lt_0.5px'],20)
        self.assertEqual(m['pct_lt_1px'],40)
    def test_missing_freeze_never_opens_controls(self):
        with tempfile.TemporaryDirectory() as d, patch('numpy.genfromtxt') as read:
            with self.assertRaises(FileNotFoundError): evaluate(d,'secret.csv',Path(d)/'evaluation.json')
            read.assert_not_called()
    def test_tamper_never_opens_controls(self):
        with tempfile.TemporaryDirectory() as d, patch('numpy.genfromtxt') as read:
            p=Path(d); np.save(p/'transform.npy',np.eye(2,3))
            (p/'transform_frozen.json').write_text(json.dumps({'sha256':'wrong'}))
            with self.assertRaises(ValueError): evaluate(p,'secret.csv',p/'evaluation.json')
            read.assert_not_called()
    def test_anisotropic_conversion(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d); m=np.array([[2.,0,.2],[0,3.,.3]])
            np.save(p/'transform.npy',m)
            (p/'transform_frozen.json').write_text(json.dumps({'sha256':sha256(p/'transform.npy')}))
            a=np.array([(x,y) for x in range(5) for y in range(5)],float)
            np.savetxt(p/'controls.csv',np.column_stack([a,a*[2,3]]),delimiter=',',header='source_x,source_y,lro_x,lro_y',comments='')
            r=evaluate(p,p/'controls.csv',p/'evaluation.json',25)
            self.assertAlmostEqual(r['reference_pixels']['median'],np.hypot(.2,.3))
            self.assertAlmostEqual(r['source_pixel_equivalent']['median'],np.hypot(.1,.1))
            with self.assertRaises(FileExistsError): evaluate(p,p/'controls.csv',p/'evaluation.json',25)

class SealTests(unittest.TestCase):
    def test_geometry_sealed_until_close(self):
        from lunarreg.isolation import GeometrySeal
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'geometry_controls_EVALUATION_ONLY.csv'
            p.write_text('secret')
            seal=GeometrySeal()
            try:
                with self.assertRaises(PermissionError): p.read_text()
                with self.assertRaises(PermissionError): open(Path(d)/'source.tif','rb')
                self.assertEqual(len(seal.denied),2)
            finally: seal.close()
            self.assertEqual(p.read_text(),'secret')

class DeviceTests(unittest.TestCase):
    def test_cpu_device_reaches_matcher_and_refiner(self):
        import sys
        import types
        from lunarreg.runtime import configure_roma_device
        modules={name:types.ModuleType(name) for name in ('romav2','romav2.matcher','romav2.refiner','romav2.features','unrelated')}
        for m in modules.values(): m.device='mps'
        with patch.dict(sys.modules,modules):
            configure_roma_device('cpu')
        for name in ('romav2.matcher','romav2.refiner','romav2.features'):
            self.assertEqual(modules[name].device,'cpu')
        self.assertEqual(modules['unrelated'].device,'mps')
