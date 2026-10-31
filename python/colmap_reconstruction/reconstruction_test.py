# SPDX-License-Identifier: BSD-3-Clause
"""Installed frontend public seams; synthetic native commands are not GPU proof."""
import tempfile
from pathlib import Path
import sys
import unittest


class FrontendTest(unittest.TestCase):
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
