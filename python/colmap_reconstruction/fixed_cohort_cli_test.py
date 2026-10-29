# SPDX-License-Identifier: BSD-3-Clause
"""Real installed-executable CPU checks; set COLMAP_TEST_EXECUTABLE to run.

No pycolmap, source imports, CUDA, dataset-specific headers or private tooling.
"""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


@unittest.skipUnless(os.environ.get('COLMAP_TEST_EXECUTABLE'),
                     'set COLMAP_TEST_EXECUTABLE to the built colmap binary')
class FixedCohortCommandTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.workspace = self.root / 'workspace'
        for directory in ('images/nested', 'sparse', 'stereo'):
            (self.workspace / directory).mkdir(parents=True)
        (self.workspace / 'sparse/cameras.txt').write_text(
            '1 PINHOLE 20 10 20 20 10 5\n')
        self.names = ['a.pgm', 'nested/b.pgm', 'c.pgm']
        images = []
        for i, name in enumerate(self.names):
            (self.workspace / 'images' / name).write_bytes(
                b'P5\n20 10\n255\n' + bytes(range(200)))
            images.append(f'{i+1} 1 0 0 0 {i} 0 0 1 {name}\n\n')
        (self.workspace / 'sparse/images.txt').write_text(''.join(images))
        (self.workspace / 'sparse/points3D.txt').write_text('')
        self.config = self.workspace / 'stereo/patch-match.cfg'
        self.config.write_text(
            'c.pgm\nnested/b.pgm,a.pgm,nested/b.pgm\n'
            'a.pgm\nc.pgm,nested/b.pgm\n'
            'nested/b.pgm\na.pgm,c.pgm\n')
        self.output = self.root / 'fresh'
        self.args = [
            '--workspace_path', str(self.workspace),
            '--output_path', str(self.output), '--prepare_only', '1',
            '--deadline_seconds', '30',
            '--host_capacity_bytes', str(1 << 30),
            '--device_capacity_bytes', str(1 << 30),
            '--loader_scratch_host_bytes', str(1 << 20),
            '--process_host_bytes', str(1 << 20),
            '--process_device_bytes', str(1 << 20),
            '--unknown_host_bytes', str(1 << 20),
            '--unknown_device_bytes', str(1 << 20),
            '--unknown_provenance', 'test-only uncalibrated allowance',
            '--reserve_host_bytes', str(1 << 20),
            '--reserve_device_bytes', str(1 << 20),
            '--PatchMatchStereo.depth_min', '1',
            '--PatchMatchStereo.depth_max', '10',
            '--PatchMatchStereo.sigma_spatial', '5',
            '--PatchMatchStereo.num_threads', '1',
            '--PatchMatchStereo.gpu_index', '0',
        ]

    def invoke(self, args=None):
        return subprocess.run(
            [os.environ['COLMAP_TEST_EXECUTABLE'], 'experimental_fixed_cohort']
            + (self.args if args is None else args),
            capture_output=True, text=True, timeout=40)

    def refuse(self, expected):
        result = self.invoke()
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn(expected, result.stdout + result.stderr)
        self.assertFalse(self.output.exists())

    def test_help_and_parser(self):
        result = self.invoke(['--help'])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('prepare_only', result.stdout + result.stderr)
        self.assertNotEqual(self.invoke([]).returncode, 0)
        self.assertNotEqual(self.invoke(['--not_an_option', '1']).returncode, 0)

    def test_prepare_actual_workspace_no_outputs_both_policies_and_owners(self):
        for owners in (1, 2):
            for policy in ('photo_and_geo', 'final_geo_only'):
                result = self.invoke(self.args + ['--active_owners', str(owners),
                                                  '--output_policy', policy])
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn('NO GPU execution', result.stdout)
                self.assertFalse(self.output.exists())
        # Existing outputs are never inspected, touched or removed in prepare_only.
        self.output.mkdir()
        sentinel = self.output / 'sentinel'
        sentinel.write_text('preserve')
        self.assertEqual(self.invoke().returncode, 0)
        self.assertEqual(sentinel.read_text(), 'preserve')

    def test_geo_producer_coverage_before_gpu(self):
        self.config.write_text('a.pgm\nc.pgm,nested/b.pgm\n')
        self.refuse('missing source producer')

    def test_photo_additional_images_allowed(self):
        self.config.write_text('a.pgm\nc.pgm,nested/b.pgm\nnested/b.pgm\na.pgm,c.pgm\n')
        geo = self.root / 'geo.cfg'
        geo.write_text('a.pgm\nnested/b.pgm\nnested/b.pgm\na.pgm\n')
        result = self.invoke(self.args + ['--geo_config_path', str(geo)])
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_auto_refused(self):
        self.config.write_text('a.pgm\n__auto__, 2\n')
        self.refuse('only explicit ordered sources')

    def test_unsafe_reference_refused(self):
        self.config.write_text('../a.pgm\nc.pgm\n')
        self.refuse('unsafe image name')

    def test_cap_refusal(self):
        self.args[self.args.index('--host_capacity_bytes') + 1] = '1'
        self.refuse('exceeds capacity')

    def test_loader_scratch_refusal(self):
        self.args[self.args.index('--loader_scratch_host_bytes') + 1] = '1'
        self.refuse('loader scratch below known logical inventory')

    def test_depth_and_sigma_not_invented(self):
        for option in ('depth_min', 'sigma_spatial'):
            with self.subTest(option=option):
                args = self.args.copy()
                args[args.index('--PatchMatchStereo.' + option) + 1] = '-1'
                result = self.invoke(args)
                self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertFalse(self.output.exists())


if __name__ == '__main__':
    unittest.main()
