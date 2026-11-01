# SPDX-License-Identifier: BSD-3-Clause
"""Installed frontend public seams; synthetic native commands are not GPU proof."""
import tempfile
from pathlib import Path
import sys
import unittest


def photo_summary(width=4004, height=3001, orientation=None):
    return dict(schema_version=1, image_path='/photos', photo_count=1,
        photos=[dict(path='photo.jpg', width=width, height=height, orientation=orientation)],
        problems=[], non_image_files=[], header_only=True, orientation_applied=False,
        build=dict(revision='a' * 8, cuda_compiled=True))


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
                    reconstruction.identity = {'allocation_fingerprint': 'test-only', 'source_commit': 'a' * 40}
                    observed_free = 1000
                    failure = NativeFailure('undistortion failed')
                    calls = []

                    def native(command, *arguments, capture=False):
                        reconstruction.process.sequence += 1
                        calls.append(command)
                        if command == 'photo_directory_summary':
                            return json.dumps(photo_summary())
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

    def test_extraction_budget_is_scoped_and_preserves_over_3200(self):
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
                reconstruction.assets.mkdir()
                reconstruction.identity = {'source_commit': 'a' * 40}
                failure = NativeFailure('extraction failed')
                def native(command, *arguments, **kwargs):
                    if command == 'photo_directory_summary':
                        import json
                        return json.dumps(photo_summary(width=4004, height=3001))
                    self.assertEqual(command, 'feature_extractor')
                    self.assertEqual(reconstruction.process.host_budget_bytes, 2000)
                    self.assertEqual(arguments[arguments.index('--FeatureExtraction.host_budget_bytes') + 1], 2000)
                    self.assertEqual(arguments[arguments.index('--FeatureExtraction.max_image_size') + 1], '4004')
                    self.assertEqual(arguments[arguments.index('--FeatureExtraction.use_gpu') + 1], 'true')
                    raise failure
                with patch.object(reconstruction, 'resources', return_value={'host': 2000}), \
                     patch.object(reconstruction, 'native', side_effect=native):
                    with self.assertRaises(NativeFailure) as raised:
                        reconstruction.prepare()
                    self.assertIs(raised.exception, failure)
                self.assertEqual(reconstruction.process.host_budget_bytes, previous)

    def test_invalid_inspection_refuses_before_extraction(self):
        import copy
        import json
        from unittest.mock import patch
        from colmap_reconstruction.cli import Reconstruction, parse_args
        summaries = ['not JSON', '[]', '{}']
        for key, value in (('photos', []), ('photo_count', 2), ('problems', [{'path': 'bad.jpg'}]),
                           ('header_only', False), ('orientation_applied', True),
                           ('schema_version', True), ('build', {'revision': 'b' * 8, 'cuda_compiled': True})):
            summary = photo_summary()
            summary[key] = value
            summaries.append(json.dumps(summary))
        for key in ('width', 'height'):
            for value in (-1, 0, True, 3.5, '4004', 2**31):
                summary = copy.deepcopy(photo_summary())
                summary['photos'][0][key] = value
                summaries.append(json.dumps(summary))
        for payload in summaries:
            with self.subTest(payload=payload), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                reconstruction = Reconstruction(parse_args(['--photos', str(root),
                    '--workspace', str(root)]), sys.executable)
                reconstruction.assets.mkdir()
                reconstruction.identity = {'source_commit': 'a' * 40}
                with patch.object(reconstruction, 'native', return_value=payload) as native:
                    with self.assertRaises(ValueError):
                        reconstruction.prepare()
                self.assertEqual([call.args[0] for call in native.call_args_list], ['photo_directory_summary'])
                self.assertFalse((root / 'database.db').exists())

    def test_cuda_sift_refuses_only_effective_width_truncation(self):
        import json
        from unittest.mock import patch
        from colmap_reconstruction.cli import Reconstruction, parse_args
        from colmap_reconstruction.process import NativeFailure
        # EXIF 6/8 swap working width/height; null, 1, 3 and unsupported
        # mirrored orientations do not. A 3001-pixel height alone is valid.
        for width, height, orientation, refuses in (
                (4004, 3001, None, False), (4004, 3001, 1, False),
                (4004, 3001, 3, False), (4004, 3001, 5, False),
                (4004, 3001, 6, True), (4004, 3001, 8, True),
                (4003, 3004, None, True), (4003, 3004, 6, False),
                (6204, 4132, None, False), (6201, 4130, None, True)):
            with self.subTest(width=width, height=height, orientation=orientation), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                reconstruction = Reconstruction(parse_args(['--photos', str(root),
                    '--workspace', str(root)]), sys.executable)
                reconstruction.assets.mkdir()
                reconstruction.identity = {'source_commit': 'a' * 40}
                for name in ('loader_scratch_host', 'process_host', 'unknown_host', 'reserve_host'):
                    setattr(reconstruction.args, name + '_bytes', 0)
                commands = []
                def native(command, *arguments, **kwargs):
                    commands.append(command)
                    if command == 'photo_directory_summary':
                        summary = photo_summary(width, height, orientation)
                        summary['photos'][0]['path'] = 'DSC_0393.JPG'
                        return json.dumps(summary)
                    self.assertEqual(command, 'feature_extractor')
                    raise NativeFailure('test boundary')
                with patch.object(reconstruction, 'native', side_effect=native), \
                     patch.object(reconstruction, 'resources', return_value={'host': 2**30}):
                    if refuses:
                        with self.assertRaisesRegex(ValueError, 'DSC_0393.JPG: effective working width.*multiple of four.*no automatic resize/fallback'):
                            reconstruction.prepare()
                        self.assertEqual(commands, ['photo_directory_summary'])
                    else:
                        with self.assertRaises(NativeFailure):
                            reconstruction.prepare()
                        self.assertEqual(commands, ['photo_directory_summary', 'feature_extractor'])

    def test_resolution_report_distinguishes_geometry_from_explicit_resize(self):
        import json
        from unittest.mock import patch
        from colmap_reconstruction.cli import Reconstruction, parse_args
        for cap in (-1, 1600):
            with self.subTest(cap=cap), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                reconstruction = Reconstruction(parse_args(['--photos', str(root), '--workspace', str(root),
                    '--max-image-size', str(cap)]), sys.executable)
                reconstruction.assets.mkdir()
                reconstruction.identity = {'source_commit': 'a' * 40}
                with patch.object(reconstruction, 'native', return_value=json.dumps(photo_summary())):
                    self.assertEqual(reconstruction.inspect_photos(), 4004)
                report = json.loads((reconstruction.assets / 'resolution-intent.json').read_text())
                self.assertEqual(report['original']['photos'][0]['width'], 4004)
                self.assertEqual(report['extraction']['max_image_size'], 4004)
                self.assertEqual(report['extraction']['working_dimensions'],
                    [dict(path='photo.jpg', width=4004, height=3001)])
                self.assertEqual(report['undistortion']['max_image_size'], cap)
                self.assertEqual(report['undistortion']['geometric_options'],
                    dict(blank_pixels=0.0, min_scale=0.2, max_scale=2.0))
                self.assertIsNone(report['undistortion']['resolved_dimensions'])
                self.assertEqual(report['stereo'], dict(max_image_size=-1, map_precision='float32'))
                self.assertEqual(report['fusion']['max_image_size'], -1)
                self.assertFalse(report['automatic_resource_resize'])

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
