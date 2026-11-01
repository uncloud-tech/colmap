# SPDX-License-Identifier: BSD-3-Clause
"""Real installed-executable CPU checks; set COLMAP_TEST_EXECUTABLE to run.

No pycolmap, source imports, CUDA, dataset-specific headers or private tooling.
"""
import json
import os
import re
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

    def test_native_workspace_plan_resolves_distinct_depths_without_global_envelope(self):
        # Per-camera translations change the SAME sparse points' depth ranges.
        images = []
        for i, name in enumerate(self.names):
            images.append(f'{i+1} 1 0 0 0 {i} 0 {i} 1 {name}\n10 5 1 11 5 2\n')
        (self.workspace / 'sparse/images.txt').write_text(''.join(images))
        (self.workspace / 'sparse/points3D.txt').write_text(
            '1 0 0 5 128 128 128 0 1 0 2 0 3 0\n'
            '2 0 0 7 128 128 128 0 1 1 2 1 3 1\n')
        # optimized_dense_plan requires native automatic-source configuration.
        # Explicit duplicate-source fixtures remain covered by the other cases.
        self.config.write_text(''.join(name + '\n__auto__, 8\n' for name in self.names))
        args = list(self.args)
        for key in ('depth_min', 'depth_max', 'sigma_spatial'):
            args[args.index('--PatchMatchStereo.' + key) + 1] = '-1'
        args += ['--PatchMatchStereo.owned_distinct_source_cache', '1']
        device = os.environ.get('COLMAP_TEST_CUDA_DEVICE', '0')
        args[args.index('--PatchMatchStereo.gpu_index') + 1] = device
        result = self.invoke(args + ['--resolve_workspace_plan', '1'])
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertFalse(self.output.exists())

        # The native CPU plan is an independent typed oracle, not a fabricated
        # execution receipt. Both consumers resolve the same automatic-source
        # configuration; compare their effective vectors without sorting.
        plan_path = self.root / 'plan.json'
        planned = subprocess.run([
            os.environ['COLMAP_TEST_EXECUTABLE'], 'optimized_dense_plan',
            '--workspace_path', str(self.workspace), '--output_path', str(plan_path),
            '--source_cap', '8', '--num_samples', '15', '--gpu_index', device,
            '--owned_tex_lookahead', 'false'],
            capture_output=True, text=True, timeout=40)
        self.assertEqual(planned.returncode, 0, planned.stdout + planned.stderr)
        plan = json.loads(plan_path.read_text())
        names = {image['index']: image['name'] for image in plan['images']}
        expected_depths = {'a.pgm': (3.75, 8.75),
                           'nested/b.pgm': (4.5, 10.0),
                           'c.pgm': (5.25, 11.25)}
        for phase in ('photo', 'geo'):
            jobs = plan['phases'][phase]['jobs']
            self.assertEqual(len(jobs), 3)
            self.assertEqual({names[job['reference']] for job in jobs}, set(self.names))
            for job in jobs:
                name = names[job['reference']]
                self.assertEqual((job['options']['depth_min'], job['options']['depth_max']),
                                 expected_depths[name])
                self.assertEqual(job['options']['sigma_spatial'], 5)
                self.assertEqual(len(job['sources']), 2)
                self.assertEqual({names[index] for index in job['sources']},
                                 set(self.names) - {name})

        if os.environ.get('COLMAP_TEST_OBSERVED_CUDA'):
            # One tiny actual adapter execution, not a new GPU matrix. Only
            # this branch proves that execution retained the typed plan facts.
            report_path = self.root / 'execution.json'
            execution_args = args + ['--resolve_workspace_plan', '1',
                                     '--report_path', str(report_path)]
            execution_args[execution_args.index('--prepare_only') + 1] = '0'
            execution_args[execution_args.index('--unknown_device_bytes') + 1] = str(4 << 20)
            execution_args[execution_args.index('--unknown_provenance') + 1] = (
                'tiny 20x10 fixture: explicit 4 MiB unknown device allowance, uncalibrated')
            executed = self.invoke(execution_args)
            self.assertEqual(executed.returncode, 0, executed.stdout + executed.stderr)
            report = json.loads(report_path.read_text())
            self.assertEqual(report['status'], 'completed')
            self.assertEqual(report['effective_backend'], 'experimental_fixed_cohort')
            for phase in ('photo', 'geo'):
                # Compare lists, never sets/sorted sources: native effective
                # order and duplicate resolution must survive the adapter.
                self.assertEqual(report['phases'][phase], [
                    {key: job[key] for key in ('reference', 'sources', 'options')}
                    for job in plan['phases'][phase]['jobs']])
            # Keep the subsequent prepare-only refusal's no-output assertion.
            self.output = self.root / 'refused-output'
            args[args.index('--output_path') + 1] = str(self.output)
        # No implicit PHOTO producer expansion to repair incomplete configs.
        self.config.write_text('a.pgm\nnested/b.pgm,c.pgm\n')
        refused = self.invoke(args + ['--resolve_workspace_plan', '1'])
        self.assertNotEqual(refused.returncode, 0)
        self.assertFalse(self.output.exists())

    def test_same_executable_build_identity(self):
        import json
        result = subprocess.run([os.environ['COLMAP_TEST_EXECUTABLE'], 'reconstruction_info'],
                                capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        identity = json.loads(result.stdout)
        self.assertRegex(identity['source_commit'], r'^[0-9a-f]{40}$')
        self.assertRegex(identity['allocation_fingerprint'], r'^[0-9a-f]{64}$')
        if os.environ.get('COLMAP_EXPECT_ALLOCATION_FINGERPRINT'):
            self.assertEqual(identity['allocation_fingerprint'],
                             os.environ['COLMAP_EXPECT_ALLOCATION_FINGERPRINT'])
        for arguments in (['--not_an_option'], ['--deadline_seconds', '0'],
                          ['--gpu_index', '-1', '--deadline_seconds', '1'],
                          ['--gpu_index', '0', '--deadline_seconds', '0']):
            refused = subprocess.run([os.environ['COLMAP_TEST_EXECUTABLE'], 'reconstruction_info',
                                      *arguments], capture_output=True, text=True, timeout=10)
            self.assertNotEqual(refused.returncode, 0)

    def test_same_executable_visible_inventory(self):
        import json
        identity = subprocess.run([os.environ['COLMAP_TEST_EXECUTABLE'], 'reconstruction_info'],
                                  capture_output=True, text=True, timeout=10)
        self.assertEqual(identity.returncode, 0, identity.stderr)
        cuda = json.loads(identity.stdout)['cuda_compiled']
        if cuda and not os.environ.get('COLMAP_TEST_OBSERVED_CUDA'):
            self.skipTest('CUDA runtime inventory requires opt-in COLMAP_TEST_OBSERVED_CUDA')
        result = subprocess.run([os.environ['COLMAP_TEST_EXECUTABLE'], 'reconstruction_info',
                                 '--deadline_seconds', '5'], capture_output=True, text=True, timeout=10)
        if not cuda:
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('requires native CUDA', result.stderr)
            return
        self.assertEqual(result.returncode, 0, result.stderr)
        inventory = json.loads(result.stdout)['device_inventory']
        self.assertEqual([d['ordinal'] for d in inventory], list(range(len(inventory))))
        for device in inventory:
            self.assertIsInstance(device['usable'], bool)
            if device['usable']:
                self.assertTrue(device['device_uuid'])
                self.assertTrue(device['name'])
                self.assertGreater(device['compute_capability_major'], 0)
                self.assertGreaterEqual(int(device['total_device_bytes']), int(device['available_device_bytes']))
                exact = subprocess.run([os.environ['COLMAP_TEST_EXECUTABLE'], 'reconstruction_info',
                                        '--gpu_index', str(device['ordinal']), '--deadline_seconds', '5'],
                                       capture_output=True, text=True, timeout=10)
                self.assertEqual(exact.returncode, 0, exact.stderr)
                observation = json.loads(exact.stdout)
                self.assertEqual(observation['device_uuid'], device['device_uuid'])
                self.assertEqual(observation['device_ordinal'], device['ordinal'])
            else:
                self.assertTrue(device['reason'])
        masked = subprocess.run([os.environ['COLMAP_TEST_EXECUTABLE'], 'reconstruction_info',
                                 '--deadline_seconds', '5'], capture_output=True, text=True, timeout=10,
                                env=dict(os.environ, CUDA_VISIBLE_DEVICES=''))
        self.assertEqual(masked.returncode, 0, masked.stderr)
        self.assertEqual(json.loads(masked.stdout)['device_inventory'], [])
        self.assertFalse(self.output.exists())

    def test_help_and_parser(self):
        result = self.invoke(['--help'])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('prepare_only', result.stdout + result.stderr)
        self.assertIn('photo_source_storage', result.stdout + result.stderr)
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

    def test_auto_storage_requires_auto_owners_and_reports_effective_binding(self):
        refused = self.invoke(self.args + ['--photo_source_storage', 'auto'])
        self.assertNotEqual(refused.returncode, 0)
        self.assertIn('auto storage requires auto_active_owners', refused.stderr)
        for phase in (False, True):
            args = self.args + ['--photo_source_storage', 'auto', '--auto_active_owners', '1']
            if phase:
                args += ['--auto_phase_active_owners', '1']
            result = self.invoke(args)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn('requested_photo_source_storage=auto; effective_photo_source_storage=private', result.stdout)
            self.assertIn('equal counts prefer private', result.stdout)
            self.assertIn('PHOTO active_owners=3; GEO active_owners=3', result.stdout)
            self.assertFalse(self.output.exists())
        (self.workspace / 'images/a.pgm').unlink()
        args[args.index('--host_capacity_bytes') + 1] = '0'
        refused = self.invoke(args)
        self.assertIn('auto PHOTO source storage: private:', refused.stderr)
        self.assertIn('; shared:', refused.stderr)
        self.assertNotIn('Cannot read', refused.stderr)

    def test_auto_storage_native_caps_and_fail_closed_before_decode(self):
        args = self.phase_args(heavy_photo=True)
        for phase in (False, True):
            for cap_storage, cap_count, expected_storage, expected_count in (
                    ('shared', 1, 'shared', 1),
                    ('shared', 3, 'shared', 3),
                    ('private', 3, 'private', 3)):
                prepared = self.invoke(args + ['--photo_source_storage', cap_storage,
                                               '--active_owners', str(cap_count)])
                self.assertEqual(prepared.returncode, 0, prepared.stdout + prepared.stderr)
                match = re.search(r'admitted host=(\d+) device=(\d+)', prepared.stdout)
                bounded = args + ['--auto_active_owners', '1', '--photo_source_storage', 'auto']
                bounded[bounded.index('--device_capacity_bytes') + 1] = str(int(match.group(2)) + (1 << 20))
                if phase:
                    bounded += ['--auto_phase_active_owners', '1']
                result = self.invoke(bounded)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn(f'effective_photo_source_storage={expected_storage}', result.stdout)
                geo_count = 3 if phase else expected_count
                self.assertIn(f'PHOTO active_owners={expected_count}; GEO active_owners={geo_count}', result.stdout)
        # Unsupported options must not be classified as an ordinary capacity
        # refusal, nor masked by missing pixels or an alternate storage plan.
        (self.workspace / 'images/a.pgm').unlink()
        invalid = args + ['--auto_active_owners', '1', '--photo_source_storage', 'auto']
        invalid[invalid.index('--device_capacity_bytes') + 1] = '0'
        invalid[invalid.index('--PatchMatchStereo.sigma_spatial') + 1] = '-1'
        result = self.invoke(invalid)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('invalid phase options', result.stderr)
        self.assertNotIn('no singleton fits', result.stderr)
        invalid[invalid.index('--PatchMatchStereo.sigma_spatial') + 1] = '5'
        (self.workspace / 'sparse/cameras.txt').write_text(
            '1 SIMPLE_RADIAL 20 10 20 10 5 0.1\n')
        result = self.invoke(invalid)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('requires undistorted cameras', result.stderr)
        self.assertNotIn('no singleton fits', result.stderr)
        self.assertFalse(self.output.exists())

    def test_automatic_count_and_ceiling_before_payload_decode(self):
        (self.workspace / 'images/d.pgm').write_bytes(
            b'P5\n20 10\n255\n' + bytes(range(200)))
        with (self.workspace / 'sparse/images.txt').open('a') as images:
            images.write('4 1 0 0 0 4 0 0 1 d.pgm\n\n')
        with self.config.open('a') as config:
            config.write('d.pgm\na.pgm,c.pgm,a.pgm\n')
        for storage in ('private', 'shared'):
            for ceiling, chosen, reason in ((0, 4, 'job count'), (3, 3, 'user ceiling'),
                                             (2, 2, 'user ceiling'), (1, 1, 'user ceiling'),
                                             (99, 4, 'job count')):
                result = self.invoke(self.args + [
                    '--auto_active_owners', '1', '--active_owners_ceiling', str(ceiling),
                    '--photo_source_storage', storage])
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn(f'active_owners={chosen}; selection=automatic; limiting_reason={reason}', result.stdout)
                self.assertFalse(self.output.exists())
        # Missing pixels prove singleton refusal happens at metadata inventory,
        # not after decode, GPU setup, allocation failure or a solver trial.
        (self.workspace / 'images/a.pgm').unlink()
        args = self.args + ['--auto_active_owners', '1']
        args[args.index('--host_capacity_bytes') + 1] = '1'
        result = self.invoke(args)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('no singleton fits: host capacity', result.stdout + result.stderr)
        self.assertFalse(self.output.exists())

    def test_automatic_capacity_boundaries_use_same_manual_envelope(self):
        args = self.args.copy()
        args[args.index('--loader_scratch_host_bytes') + 1] = '4096'
        for count in (1, 2, 3):
            manual = self.invoke(args + ['--active_owners', str(count)])
            self.assertEqual(manual.returncode, 0, manual.stdout + manual.stderr)
            match = re.search(r'admitted host=(\d+) device=(\d+)', manual.stdout)
            self.assertIsNotNone(match, manual.stdout)
            for resource, admitted in zip(('host', 'device'), map(int, match.groups())):
                bounded = args + ['--auto_active_owners', '1']
                bounded[bounded.index(f'--{resource}_capacity_bytes') + 1] = str(admitted + (1 << 20))
                result = self.invoke(bounded)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                reason = f'{resource} capacity' if count < 3 else 'job count'
                self.assertIn(f'active_owners={count}; selection=automatic; limiting_reason={reason}', result.stdout)
                if count == 1:
                    bounded[bounded.index(f'--{resource}_capacity_bytes') + 1] = str(admitted + (1 << 20) - 1)
                    refused = self.invoke(bounded)
                    self.assertNotEqual(refused.returncode, 0)
                    self.assertIn(f'no singleton fits: {resource} capacity', refused.stdout + refused.stderr)
        self.assertFalse(self.output.exists())

    def phase_args(self, heavy_photo=False):
        # Strongly asymmetric, ordered duplicate source lists, same producer set.
        photo = self.workspace / 'stereo/photo.cfg'
        geo = self.workspace / 'stereo/geo.cfg'
        for path, repeats in ((photo, 20 if heavy_photo else 1),
                              (geo, 1 if heavy_photo else 20)):
            path.write_text(''.join(name + '\n' + ','.join(
                [self.names[(i + 1) % 3]] * repeats) + '\n'
                for i, name in enumerate(self.names)))
        return self.args + ['--photo_config_path', str(photo),
                            '--geo_config_path', str(geo)]

    def test_phase_selection_prepare_and_refusal_before_decode(self):
        invalid = self.invoke(self.args + ['--auto_phase_active_owners', '1'])
        self.assertNotEqual(invalid.returncode, 0)
        self.assertIn('auto_phase_active_owners requires auto_active_owners',
                      invalid.stdout + invalid.stderr)
        for heavy_photo in (False, True):
            args = self.phase_args(heavy_photo)
            manual = self.invoke(args)
            self.assertEqual(manual.returncode, 0, manual.stdout + manual.stderr)
            match = re.search(r'admitted host=(\d+) device=(\d+)', manual.stdout)
            args[args.index('--device_capacity_bytes') + 1] = str(int(match.group(2)) + (1 << 20))
            result = self.invoke(args + ['--auto_active_owners', '1',
                                         '--auto_phase_active_owners', '1'])
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            heavy, light = ('PHOTO', 'GEO') if heavy_photo else ('GEO', 'PHOTO')
            self.assertIn(f'{heavy} active_owners=1; limiting_reason=device capacity', result.stdout)
            self.assertIn(f'{light} active_owners=3; limiting_reason=job count', result.stdout)
        (self.workspace / 'images/a.pgm').unlink()
        args[args.index('--device_capacity_bytes') + 1] = '0'
        refused = self.invoke(args + ['--auto_active_owners', '1', '--auto_phase_active_owners', '1'])
        self.assertNotEqual(refused.returncode, 0)
        self.assertIn('PHOTO no singleton fits: device capacity', refused.stdout + refused.stderr)
        self.assertFalse(self.output.exists())

    def test_manual_higher_count_and_invalid_count(self):
        result = self.invoke(self.args + ['--active_owners', '3'])
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('active_owners=3; selection=manual', result.stdout)
        for count in ('0', '-1', '4', '18446744073709551616'):
            self.assertNotEqual(self.invoke(self.args + ['--active_owners', count]).returncode, 0)
        for extra in (['--active_owners_ceiling', '2'],
                      ['--auto_active_owners', '1', '--active_owners_ceiling', '-1'],
                      ['--auto_active_owners', '1', '--active_owners', '2']):
            self.assertNotEqual(self.invoke(self.args + extra).returncode, 0)
        self.assertFalse(self.output.exists())

    def test_photo_source_storage_default_private_and_shared_inventory(self):
        for storage in (None, 'private', 'shared'):
            with self.subTest(storage=storage):
                args = self.args if storage is None else self.args + [
                    '--photo_source_storage', storage]
                result = self.invoke(args)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                mode = storage or 'private'
                # Three distinct 20x10 uint8 layers, despite repeated ordinals.
                union = 600 if mode == 'shared' else 0
                self.assertIn(f'photo_source_storage={mode}; logical PHOTO union '
                              f'host={union} device={union} bytes', result.stdout)
                self.assertIn('NO GPU execution', result.stdout)
                self.assertFalse(self.output.exists())

    def test_invalid_photo_source_storage_refuses_before_workspace_or_output(self):
        # A missing workspace ensures option validation precedes input loading.
        self.args[self.args.index('--workspace_path') + 1] = str(self.root / 'missing')
        self.output.mkdir()
        sentinel = self.output / 'sentinel'
        sentinel.write_text('preserve')
        for value in ('AUTO', 'SHARED', ''):
            with self.subTest(value=value):
                result = self.invoke(self.args + ['--photo_source_storage', value])
                self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn('invalid photo_source_storage: expected private, shared or auto',
                              result.stdout + result.stderr)
                self.assertEqual(list(self.output.iterdir()), [sentinel])
                self.assertEqual(sentinel.read_text(), 'preserve')
                self.assertNotIn('admitted host=', result.stdout)

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

    def test_byte_options_are_lossless_unsigned_integers(self):
        index = self.args.index('--host_capacity_bytes') + 1
        for value in ('-1', '18446744073709551616', '1.5', '123x'):
            with self.subTest(value=value):
                args = self.args.copy()
                args[index] = value
                result = self.invoke(args)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('invalid unsigned byte quantity',
                              result.stdout + result.stderr)
                self.assertFalse(self.output.exists())
        args = self.args.copy()
        args[index] = str(1 << 34)
        result = self.invoke(args)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertFalse(self.output.exists())

    def test_observation_requires_automatic_selection(self):
        result = self.invoke(self.args + ['--observe_capacities', '1'])
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('observe_capacities requires auto_active_owners',
                      result.stdout + result.stderr)
        self.assertFalse(self.output.exists())

    def test_manual_capacities_remain_required(self):
        for name in ('host', 'device'):
            args = self.args.copy()
            index = args.index(f'--{name}_capacity_bytes')
            del args[index:index + 2]
            result = self.invoke(args)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('manual mode requires host_capacity_bytes and device_capacity_bytes',
                          result.stdout + result.stderr)
            self.assertFalse(self.output.exists())

    def test_observation_invalid_device_and_caps_refuse_before_query(self):
        for device in ('-1', '0,1', '01', '2147483648'):
            args = self.args + ['--observe_capacities', '1', '--auto_active_owners', '1']
            args[args.index('--PatchMatchStereo.gpu_index') + 1] = device
            result = self.invoke(args)
            self.assertNotEqual(result.returncode, 0)
            self.assertNotIn('CUDA capacity query failed', result.stdout + result.stderr)
            self.assertNotIn('observed_device=', result.stdout)
            self.assertFalse(self.output.exists())
        for value in ('', '-1', '18446744073709551616', 'unspecified'):
            args = self.args + ['--observe_capacities', '1', '--auto_active_owners', '1']
            args[args.index('--host_capacity_bytes') + 1] = value
            result = self.invoke(args)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('invalid unsigned byte quantity', result.stdout + result.stderr)
            self.assertFalse(self.output.exists())

    @unittest.skipUnless(os.environ.get('COLMAP_TEST_NO_CUDA'),
                         'explicit CPU-only build check required')
    def test_observation_cpu_build_is_unsupported_not_shortage(self):
        result = self.invoke(self.args + ['--observe_capacities', '1', '--auto_active_owners', '1'])
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('capacity observation requires native CUDA', result.stdout + result.stderr)
        self.assertNotIn('no singleton fits', result.stdout + result.stderr)
        self.assertFalse(self.output.exists())

    @unittest.skipUnless(os.environ.get('COLMAP_TEST_OBSERVED_CUDA'),
                         'explicit real CUDA observation opt-in required')
    def test_observed_prepare_caps_device_and_query_failure(self):
        args = self.args + ['--observe_capacities', '1', '--auto_active_owners', '1']
        device = os.environ.get('COLMAP_TEST_CUDA_DEVICE', '0')
        args[args.index('--PatchMatchStereo.gpu_index') + 1] = device
        result = self.invoke(args)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn(f'observed_device={device};', result.stdout)
        self.assertIn('device_uuid=', result.stdout)
        match = re.search(r'observed_free_host_bytes=(\d+); observed_free_device_bytes=(\d+); observed_total_device_bytes=(\d+); effective_host_capacity_bytes=(\d+); effective_device_capacity_bytes=(\d+)', result.stdout)
        self.assertIsNotNone(match, result.stdout)
        host, free, total, effective_host, effective_device = map(int, match.groups())
        self.assertLessEqual(free, total)
        self.assertEqual(effective_host, min(host, 1 << 30))
        self.assertEqual(effective_device, min(free, 1 << 30))
        self.assertIn('NO solver execution', result.stdout)
        self.assertFalse(self.output.exists())
        # Absent caps use observations, zero caps refuse; never credit declared
        # process/context bytes or silently fall back to manual mode.
        uncapped = args.copy()
        for resource in ('host', 'device'):
            index = uncapped.index(f'--{resource}_capacity_bytes')
            del uncapped[index:index + 2]
        result = self.invoke(uncapped)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        for resource in ('host', 'device'):
            bounded = args.copy()
            bounded[bounded.index(f'--{resource}_capacity_bytes') + 1] = '0'
            refused = self.invoke(bounded)
            self.assertNotEqual(refused.returncode, 0)
            self.assertIn(f'no singleton fits: {resource} capacity', refused.stdout + refused.stderr)
        args[args.index('--PatchMatchStereo.gpu_index') + 1] = '2147483647'
        failed = self.invoke(args)
        self.assertNotEqual(failed.returncode, 0)
        self.assertIn('CUDA capacity query failed', failed.stdout + failed.stderr)
        self.assertNotIn('no singleton fits', failed.stdout + failed.stderr)
        self.assertFalse(self.output.exists())

    @unittest.skipUnless(os.environ.get('COLMAP_TEST_OBSERVED_CUDA'),
                         'explicit real CUDA execution opt-in required')
    def test_observed_execution_preserves_manual_maps_and_budget_counts(self):
        args = self.args.copy()
        args[args.index('--prepare_only') + 1] = '0'
        # Execution-only allowance for this tiny fixture, not a calibrated
        # production bound. Owner buffers showed 512-byte pitches; snapshot
        # and staging pitches were not logged. Assuming matching pitches,
        # three depth+normal snapshots would add 51,840 bytes and the largest
        # GEO staging 12,960 bytes (3 sources including duplicates). This
        # hypothesized 64,800-byte delta exceeds the original ~64 KiB margin. Declaring 4 MiB unknown device
        # overhead (process remains 1 MiB) supplies >158,000 bytes in the SAME
        # 3% margin. CPU boundary tests retain their independent 1 MiB values.
        args[args.index('--unknown_device_bytes') + 1] = str(4 << 20)
        args[args.index('--unknown_provenance') + 1] = (
            'tiny 20x10 GPU fixture: explicit 4 MiB unknown device allowance; '
            'snapshot/staging pitch hypothesis, not physical-fit proof')
        args[args.index('--PatchMatchStereo.gpu_index') + 1] = os.environ.get('COLMAP_TEST_CUDA_DEVICE', '0')
        args += ['--output_policy', 'photo_and_geo', '--PatchMatchStereo.num_iterations', '1']
        control = self.invoke(args)
        self.assertEqual(control.returncode, 0, control.stdout + control.stderr)
        expected = {path.relative_to(self.output): path.read_bytes()
                    for path in self.output.rglob('*.bin')}
        self.assertEqual(len(expected), 12)  # 3 references x 2 phases x 2 kinds.
        for storage in ('private', 'shared'):
            for count in (1, 2, 3):
                for graphs in (0, 1):
                    with self.subTest(storage=storage, count=count, graphs=graphs):
                        # Derive each count's cap from the exact execution
                        # envelope, including its fixture-only allowance.
                        prepare_args = args.copy()
                        prepare_args[prepare_args.index('--prepare_only') + 1] = '1'
                        prepared = self.invoke(prepare_args + ['--active_owners', str(count),
                                                              '--photo_source_storage', storage])
                        self.assertEqual(prepared.returncode, 0, prepared.stdout + prepared.stderr)
                        match = re.search(r'admitted host=(\d+) device=(\d+)', prepared.stdout)
                        self.assertIsNotNone(match, prepared.stdout)
                        # One canonical margin already in admitted; only the
                        # existing unallocated reserve is added, no second margin.
                        device_cap = int(match.group(2)) + (1 << 20)
                        observed = args + ['--observe_capacities', '1', '--auto_active_owners', '1',
                                           '--active_owners_ceiling', '3', '--photo_source_storage', storage,
                                           '--PatchMatchStereo.owned_cuda_graphs', str(graphs)]
                        observed[observed.index('--device_capacity_bytes') + 1] = str(device_cap)
                        output = self.root / f'observed-{storage}-{count}-{graphs}'
                        observed[observed.index('--output_path') + 1] = str(output)
                        result = self.invoke(observed)
                        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                        self.assertIn(f'active_owners={count}; selection=automatic', result.stdout)
                        actual = {path.relative_to(output): path.read_bytes()
                                  for path in output.rglob('*.bin')}
                        self.assertEqual(actual, expected)

    @unittest.skipUnless(os.environ.get('COLMAP_TEST_OBSERVED_CUDA'),
                         'explicit real CUDA execution opt-in required')
    def test_phase_execution_unequal_counts_full_path_identity(self):
        for heavy_photo in (False, True):
            args = self.phase_args(heavy_photo)
            # Same honest execution-only unknown allowance as the PR21 fixture.
            args[args.index('--unknown_device_bytes') + 1] = str(4 << 20)
            args[args.index('--unknown_provenance') + 1] = 'tiny 20x10 fixture: explicit 4 MiB unknown device allowance, uncalibrated'
            args[args.index('--PatchMatchStereo.gpu_index') + 1] = os.environ.get('COLMAP_TEST_CUDA_DEVICE', '0')
            args += ['--output_policy', 'photo_and_geo', '--PatchMatchStereo.num_iterations', '1']
            for storage in ('private', 'shared'):
                prepared = self.invoke(args + ['--photo_source_storage', storage])
                self.assertEqual(prepared.returncode, 0, prepared.stdout + prepared.stderr)
                match = re.search(r'admitted host=(\d+) device=(\d+)', prepared.stdout)
                cap = int(match.group(2)) + (1 << 20)
                args_run = args + ['--photo_source_storage', storage]
                args_run[args_run.index('--prepare_only') + 1] = '0'
                control_path = self.root / f'phase-control-{heavy_photo}-{storage}'
                args_run[args_run.index('--output_path') + 1] = str(control_path)
                control = self.invoke(args_run)
                self.assertEqual(control.returncode, 0, control.stdout + control.stderr)
                expected = {p.relative_to(control_path): p.read_bytes() for p in control_path.rglob('*.bin')}
                self.assertEqual(len(expected), 12)
                for observed in (False, True):
                    for graphs in (0, 1):
                        output = self.root / f'phase-{heavy_photo}-{storage}-{observed}-{graphs}'
                        run = args_run + ['--auto_active_owners', '1', '--auto_phase_active_owners', '1',
                                          '--PatchMatchStereo.owned_cuda_graphs', str(graphs)]
                        if observed:
                            run += ['--observe_capacities', '1']
                        run[run.index('--device_capacity_bytes') + 1] = str(cap)
                        run[run.index('--output_path') + 1] = str(output)
                        result = self.invoke(run)
                        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                        heavy, light = ('PHOTO', 'GEO') if heavy_photo else ('GEO', 'PHOTO')
                        self.assertIn(f'{heavy} active_owners=1;', result.stdout)
                        self.assertIn(f'{light} active_owners=3;', result.stdout)
                        self.assertEqual({p.relative_to(output): p.read_bytes() for p in output.rglob('*.bin')}, expected)

    @unittest.skipUnless(os.environ.get('COLMAP_TEST_OBSERVED_CUDA'),
                         'explicit real CUDA execution opt-in required')
    def test_auto_storage_execution_matches_explicit_effective_mode(self):
        args = self.phase_args(heavy_photo=True)
        original_inputs = {p.relative_to(self.workspace): p.read_bytes()
                           for p in self.workspace.rglob('*') if p.is_file()}
        args[args.index('--unknown_device_bytes') + 1] = str(4 << 20)
        args[args.index('--unknown_provenance') + 1] = (
            'tiny 20x10 fixture: explicit 4 MiB unknown device allowance, uncalibrated')
        args[args.index('--PatchMatchStereo.gpu_index') + 1] = os.environ.get('COLMAP_TEST_CUDA_DEVICE', '0')
        args += ['--output_policy', 'photo_and_geo', '--PatchMatchStereo.num_iterations', '1']
        for storage in ('private', 'shared'):
            for count in (1, 3):
                prepared = self.invoke(args + ['--photo_source_storage', storage,
                                               '--active_owners', str(count)])
                self.assertEqual(prepared.returncode, 0, prepared.stdout + prepared.stderr)
                match = re.search(r'admitted host=(\d+) device=(\d+)', prepared.stdout)
                cap = int(match.group(2)) + (1 << 20)
                control_path = self.root / f'auto-control-{storage}-{count}'
                control_args = args + ['--photo_source_storage', storage, '--active_owners', str(count)]
                control_args[control_args.index('--prepare_only') + 1] = '0'
                control_args[control_args.index('--output_path') + 1] = str(control_path)
                control = self.invoke(control_args)
                self.assertEqual(control.returncode, 0, control.stdout + control.stderr)
                expected = {p.relative_to(control_path): p.read_bytes() for p in control_path.rglob('*.bin')}
                self.assertEqual(len(expected), 12)
                output = self.root / f'auto-{storage}-{count}'
                run = args + ['--auto_active_owners', '1', '--photo_source_storage', 'auto',
                              '--active_owners_ceiling', str(count), '--observe_capacities', '1']
                run[run.index('--prepare_only') + 1] = '0'
                run[run.index('--device_capacity_bytes') + 1] = str(cap)
                run[run.index('--output_path') + 1] = str(output)
                result = self.invoke(run)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn(f'effective_photo_source_storage={storage}', result.stdout)
                self.assertIn(f'PHOTO active_owners={count}; GEO active_owners={count}', result.stdout)
                union_bytes = 600 if storage == 'shared' else 0
                self.assertIn(f'photo_source_storage={storage}; logical PHOTO union host={union_bytes} device={union_bytes}', result.stdout)
                self.assertIn(f'executed PHOTO owner_limit={count}; GEO owner_limit={count}; actual PHOTO union host={union_bytes} device={union_bytes}', result.stdout)
                self.assertEqual({p.relative_to(output): p.read_bytes() for p in output.rglob('*.bin')}, expected)
        self.assertEqual({p.relative_to(self.workspace): p.read_bytes()
                          for p in self.workspace.rglob('*') if p.is_file()}, original_inputs)

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
