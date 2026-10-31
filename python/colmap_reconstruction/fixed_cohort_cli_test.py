# SPDX-License-Identifier: BSD-3-Clause
"""Real installed-executable CPU checks; set COLMAP_TEST_EXECUTABLE to run.

No pycolmap, source imports, CUDA, dataset-specific headers or private tooling.
"""
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
        for value in ('auto', 'SHARED', ''):
            with self.subTest(value=value):
                result = self.invoke(self.args + ['--photo_source_storage', value])
                self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn('invalid photo_source_storage: expected private or shared',
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
