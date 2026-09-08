"""Image-only synthetic verification; never reads held-out geometry controls."""
import contextlib
import csv
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from lunarreg.cycle import cycle_metrics
from lunarreg.geometry import apply_transform, make_similarity, similarity_from_two_points, transform_parameters, wrap_angle_deg
from lunarreg.global_refinement import solve
from lunarreg.matching import blind_rotation_angles, map_rotated_points_to_source


class BlindTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(17)
        self.a = self.rng.uniform(0, 400, (240, 2))

    def fit(self, a, b, shape=(400, 400), **kwargs):
        with tempfile.TemporaryDirectory() as out, contextlib.redirect_stdout(io.StringIO()):
            result = solve(a, b, np.ones(len(a)), np.zeros(len(a)), (400, 400), shape, out, **kwargs)
            with (Path(out)/'hypotheses.csv').open() as stream:
                rows = list(csv.DictReader(stream))
        return result, rows

    def test_source_pixel_cycle_and_legacy_units(self):
        args = (np.array([[0., 0.]]), np.array([[3., 4.]]), np.ones(1), np.ones(1))
        self.assertEqual(cycle_metrics(*args)[0][0], 5)
        self.assertEqual(cycle_metrics(*args, .25)[0][0], 1.25)

    def test_rotation_sweep_covers_full_range_and_maps_back_to_source(self):
        self.assertEqual(blind_rotation_angles(90), (0.0, 90.0, 180.0, 270.0))
        with self.assertRaises(ValueError):
            blind_rotation_angles(0)
        import cv2
        rotation = cv2.getRotationMatrix2D((49.5, 49.5), 90, 1.0)
        point = np.array([[30., 40.]], np.float32)
        rotated = cv2.transform(point.reshape(1, -1, 2), rotation)[0]
        restored, valid = map_rotated_points_to_source(rotated, cv2.invertAffineTransform(rotation), (100, 100))
        np.testing.assert_allclose(restored, point, atol=1e-4)
        self.assertTrue(valid[0])

    def test_two_point_full_rotation(self):
        a = np.array([[0., 0.], [200., 0.]])
        for angle in (-179.9, -120, 90, 179.9):
            m = make_similarity(.2, angle, 200, 200)
            b = apply_transform(m, a)
            self.assertIsNone(similarity_from_two_points(*a, *b))
            np.testing.assert_allclose(similarity_from_two_points(*a, *b, blind=True), m, atol=1e-10)

    def test_arbitrary_rotation_scale_and_shape_independence(self):
        for angle, scale in [(125, .2), (-145, 2.7), (179.9, .7), (-179.9, .7)]:
            with self.subTest(angle=angle):
                m = make_similarity(scale, angle, 1500, 1500)
                b = apply_transform(m, self.a) + self.rng.normal(0, .04, self.a.shape)
                b[:35] = self.rng.uniform(0, 2500, (35, 2))
                result, rows = self.fit(self.a, b, blind=True)
                other, other_rows = self.fit(self.a, b, shape=(4439, 4684), blind=True)
                self.assertTrue(result['passed'])
                np.testing.assert_array_equal(result['transform'], other['transform'])
                self.assertEqual(rows, other_rows)
                self.assertTrue(all(r['score'] == r['data_score'] for r in rows))
                self.assertLess(np.median(np.linalg.norm(apply_transform(result['transform'], self.a)-apply_transform(m,self.a), axis=1)), .1)
                initial_rotation = float(rows[0]['rotation'])
                self.assertLess(abs(wrap_angle_deg(transform_parameters(result['coarse_transform'])[1]-initial_rotation)), 4)

    def test_data_support_beats_north_up_decoy(self):
        a = self.rng.uniform(0, 400, (300, 2))
        m = make_similarity(.35, 135, 300, 300)
        b = apply_transform(m, a)
        b[:80] = a[:80]  # weaker, perfectly north-up and shape-compatible mode
        result, rows = self.fit(a, b, blind=True)
        self.assertTrue(result['passed'])
        self.assertLess(np.max(np.linalg.norm(apply_transform(result['transform'], a)-apply_transform(m, a), axis=1)), .05)

    def test_blind_rejects_priors(self):
        for kwargs in ({'expected_scale': 1}, {'rotation': 0}):
            with self.assertRaises(ValueError):
                self.fit(self.a, self.a, blind=True, **kwargs)

    def test_low_coverage_fails_blind_gate(self):
        a = self.rng.uniform(0, 15, (100, 2))
        result, _ = self.fit(a, apply_transform(make_similarity(2, 100, 100, 100), a), blind=True)
        self.assertFalse(result['passed'])

    def test_legacy_default_matches_explicit_legacy(self):
        b = apply_transform(make_similarity(1.01, 1, 2, 3), self.a)
        default, rows = self.fit(self.a, b)
        legacy, legacy_rows = self.fit(self.a, b, blind=False, rotation=0)
        np.testing.assert_array_equal(default['transform'], legacy['transform'])
        self.assertEqual(rows, legacy_rows)

    def test_pipeline_freezes_before_evaluation_and_never_reads_controls(self):
        from argparse import Namespace
        import builtins
        import json
        from lunarreg import pipeline
        from lunarreg.io import sha256
        m = make_similarity(.3, 130, 200, 200)
        b = apply_transform(m, self.a)
        real_open = builtins.open
        real_io_open = io.open
        def guard(opener):
            def checked(path, *args, **kwargs):
                if 'geometry_controls_EVALUATION_ONLY.csv' in str(path):
                    raise AssertionError('Evaluation controls opened')
                return opener(path, *args, **kwargs)
            return checked
        with tempfile.TemporaryDirectory() as root:
            out = Path(root)/'run'
            source = Path(root)/'source.png'
            source.write_bytes(b'image stub')
            args = Namespace(output=str(out), source=str(source), reference=str(source),
                             seed=42, mode='fast', device='cpu', registration_mode='blind',
                             expected_scale=None, expected_rotation=None, frozen_baseline=None,
                             blind_rotation_step=30.0,
                             experimental_local=False, no_local_refine=True,
                             source_geotiff='evaluation-source', reference_geotiff='evaluation-reference')
            def evaluate(*args):
                frozen = json.loads((out/'transform_frozen.json').read_text())
                self.assertEqual(frozen['sha256'], sha256(out/'transform.npy'))
                return apply_transform(np.load(out/'transform.npy'), args[2])
            with patch('builtins.open', guard(real_open)), patch('io.open', guard(real_io_open)), \
                 patch.object(pipeline, 'read_image', return_value=np.ones((400,400),np.uint8)), \
                 patch.object(pipeline, 'seed_all'), patch.object(pipeline, 'load_model'), \
                 patch.object(pipeline, 'save_registration'), \
                 patch.object(pipeline, 'georeference', side_effect=evaluate) as evaluation, \
                 patch('lunarreg.matching.match_blind_rotation_sweep', return_value=(self.a,b,np.ones(240),np.zeros(240),[{'rotation_degrees': 0.0, 'valid_matches': 240}])) as matching:
                self.assertEqual(pipeline.run(args), 0)
                self.assertEqual(matching.call_args.kwargs['step_degrees'], 30.0)
                self.assertTrue(evaluation.called)
            report = json.loads((out/'metrics.json').read_text())
            self.assertEqual(report['provenance']['cycle_error_units'], 'source_pixels')

    def test_blind_pipeline_skips_only_torch_wide_determinism(self):
        from argparse import Namespace
        from lunarreg import pipeline
        with tempfile.TemporaryDirectory() as root:
            out = Path(root) / 'run'
            args = Namespace(output=str(out), source='source.png', reference='reference.png',
                             seed=42, mode='fast', device='cpu', registration_mode='blind',
                             expected_scale=None, expected_rotation=None, frozen_baseline=None,
                             blind_rotation_step=30.0,
                             experimental_local=False, no_local_refine=True,
                             source_geotiff=None, reference_geotiff=None)
            with patch.object(pipeline, 'read_image', return_value=np.ones((400, 400), np.uint8)), \
                 patch.object(pipeline, 'sha256', return_value='hash'), \
                 patch.object(pipeline, 'seed_all') as seed, \
                 patch.object(pipeline, 'load_model'), \
                 patch.object(pipeline, 'save_registration'), \
                 patch('lunarreg.matching.match_blind_rotation_sweep', return_value=(self.a, self.a, np.ones(240), np.zeros(240), [])):
                self.assertEqual(pipeline.run(args), 0)
            self.assertEqual(seed.call_args.kwargs, {'deterministic': False})

    def test_cli_help_and_invalid_blind_flags_without_heavy_imports(self):
        script = '''
import importlib.abc, runpy, sys
class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in ('torch','romav2','cv2','numpy','rasterio'):
            raise AssertionError('Heavy import: '+fullname)
sys.meta_path.insert(0, Guard())
sys.argv = ['lunarreg'] + sys.argv[1:]
runpy.run_module('lunarreg', run_name='__main__')
'''
        cases = [(['--help'], 0), (['register','--help'], 0)]
        base = ['register','--source','a','--reference','b','--output','c','--registration-mode','blind']
        cases += [(base + flag, 2) for flag in (['--expected-scale','1'],['--expected-rotation','0'],['--frozen-baseline','x'])]
        for args, code in cases:
            p = subprocess.run([sys.executable,'-c',script,*args], capture_output=True, text=True, timeout=10)
            self.assertEqual(p.returncode, code, p.stderr)
            self.assertNotIn('Heavy import:',p.stderr)


if __name__ == '__main__':
    unittest.main()
