# SPDX-License-Identifier: BSD-3-Clause
"""Installed frontend public seams; synthetic native commands are not GPU proof."""
import tempfile
from pathlib import Path
import sys
import unittest


class FrontendTest(unittest.TestCase):
    def test_undistortion_budget_is_scoped_on_success_and_native_failure(self):
        import json
        from unittest.mock import patch
        from colmap_reconstruction.cli import Reconstruction, parse_args
        from colmap_reconstruction.process import NativeFailure

        for cap in (None, 3000):
            for fails in (False, True):
                with self.subTest(cap=cap, fails=fails), tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    argv = ['--photos', str(root), '--workspace', str(root)]
                    if cap is not None:
                        argv += ['--host-budget-bytes', str(cap)]
                    for name in ('loader-scratch-host', 'process-host', 'unknown-host',
                                 'process-device', 'unknown-device', 'reserve-host', 'reserve-device'):
                        argv += ['--' + name + '-bytes', '0']
                    reconstruction = Reconstruction(parse_args(argv), sys.executable)
                    reconstruction.assets.mkdir()
                    reconstruction.identity = {'allocation_fingerprint': 'test-only'}
                    observed_free = 1000
                    failure = NativeFailure('undistortion failed')
                    calls = []

                    def native(command, *arguments, capture=False):
                        reconstruction.process.sequence += 1
                        calls.append(command)
                        if command == 'reconstruction_info':
                            return json.dumps(dict(allocation_fingerprint='test-only',
                                available_host_bytes=observed_free, available_device_bytes=1000))
                        if command == 'mapper':
                            (root / 'sparse/0').mkdir()
                        if command == 'model_analyzer':
                            return json.dumps(dict(num_registered_images=2, num_points3D=1))
                        if command in ('feature_extractor', 'exhaustive_matcher', 'mapper'):
                            self.assertEqual(reconstruction.process.host_budget_bytes, 1000)
                        elif command == 'image_undistorter':
                            self.assertEqual(reconstruction.process.host_budget_bytes, 1000)
                            self.assertEqual(arguments[arguments.index('--host_budget_bytes') + 1], 1000)
                            if fails:
                                raise failure
                        else:
                            self.assertEqual(reconstruction.process.host_budget_bytes, cap)

                    with patch.object(reconstruction, 'native', side_effect=native):
                        if fails:
                            with self.assertRaises(NativeFailure) as raised:
                                reconstruction.prepare()
                            self.assertIs(raised.exception, failure)
                        else:
                            reconstruction.prepare()
                        self.assertIn('image_undistorter', calls)
                        self.assertEqual(reconstruction.process.host_budget_bytes, cap)
                        # A later observation/launch must not inherit the earlier
                        # 1000-byte FREE snapshot as its RSS-monitor ceiling.
                        observed_free = 2000
                        self.assertEqual(reconstruction.resources()['host'], 2000)
                        reconstruction.native('optimized_dense_plan')

    def test_extraction_budget_is_scoped_and_keeps_3200(self):
        from unittest.mock import patch
        from colmap_reconstruction.cli import Reconstruction, parse_args
        from colmap_reconstruction.process import NativeFailure
        for previous in (None, 3000):
            with self.subTest(previous=previous), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                reconstruction = Reconstruction(parse_args([
                    '--photos', str(root), '--workspace', str(root),
                    '--loader-scratch-host-bytes', '0', '--process-host-bytes', '0',
                    '--unknown-host-bytes', '0', '--reserve-host-bytes', '0']), sys.executable)
                reconstruction.process.host_budget_bytes = previous
                failure = NativeFailure('extraction failed')
                def native(command, *arguments, **kwargs):
                    self.assertEqual(command, 'feature_extractor')
                    self.assertEqual(reconstruction.process.host_budget_bytes, 2000)
                    self.assertEqual(arguments[arguments.index('--FeatureExtraction.host_budget_bytes') + 1], 2000)
                    self.assertEqual(arguments[arguments.index('--FeatureExtraction.max_image_size') + 1], '3200')
                    self.assertNotIn('--FeatureExtraction.use_gpu', arguments)
                    raise failure
                with patch.object(reconstruction, 'resources', return_value={'host': 2000}), \
                     patch.object(reconstruction, 'native', side_effect=native):
                    with self.assertRaises(NativeFailure) as raised:
                        reconstruction.prepare()
                    self.assertIs(raised.exception, failure)
                self.assertEqual(reconstruction.process.host_budget_bytes, previous)

    def test_dense_launch_uses_canonical_adaptive_backend_and_explicit_allowances(self):
        from colmap_reconstruction.cli import Reconstruction, parse_args
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            argv = ['--photos', str(root), '--workspace', str(root)]
            for name in ('loader-scratch-host', 'process-host', 'process-device',
                         'unknown-host', 'unknown-device', 'reserve-host', 'reserve-device'):
                argv += ['--' + name + '-bytes', '123']
            args = parse_args(argv + ['--experimental-local', '--unknown-provenance', 'test-only'])
            reconstruction = Reconstruction(args, sys.executable)
            reconstruction.graph = {'phases': {'geo': {'requested_options': {}}}}
            command = reconstruction.dense_arguments()
            for key in ('resolve_workspace_plan', 'auto_active_owners',
                        'auto_phase_active_owners', 'observe_capacities'):
                self.assertEqual(command[command.index('--' + key) + 1], '1')
            self.assertEqual(command[command.index('--photo_source_storage') + 1], 'auto')
            self.assertEqual(command[command.index('--unknown_host_bytes') + 1], '123')
            self.assertNotIn('--PatchMatchStereo.depth_min', command)
            self.assertNotIn('--PatchMatchStereo.depth_max', command)
            self.assertEqual(args.max_image_size, -1)

    def test_incomplete_or_nonfinite_maps_are_not_success(self):
        import struct
        from colmap_reconstruction.cli import validate_dense_map
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'depth.bin'
            for payload in (b'2&1&1&' + struct.pack('<f', 1),
                            b'2&1&1&' + struct.pack('<ff', 1, float('nan')),
                            b'1&1&1&' + struct.pack('<f', 1)):
                path.write_bytes(payload)
                with self.assertRaises(ValueError):
                    validate_dense_map(path, 2, 1, 1)
            path.write_bytes(b'2&1&1&' + struct.pack('<ff', 1, 2))
            validate_dense_map(path, 2, 1, 1)


if __name__ == '__main__':
    unittest.main()
