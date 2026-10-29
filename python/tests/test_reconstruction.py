# SPDX-License-Identifier: BSD-3-Clause
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class ReconstructionTest(unittest.TestCase):
    def test_supported_build_binding_records_commit_not_a_commit_whitelist(self) -> None:
        from colmap_reconstruction import producer
        identity = {'source_commit': 'a' * 40, 'lineage_sha': producer.LINEAGE,
                    'fingerprint': producer.FINGERPRINT}
        self.assertEqual(producer.compiled_binding(identity), identity)
        identity['source_commit'] = 'b' * 40
        self.assertEqual(producer.compiled_binding(identity), identity)
        identity['fingerprint'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'fingerprint'):
            producer.compiled_binding(identity)

    def test_photo_wire_uses_same_build_identity_and_native_options(self) -> None:
        from colmap_reconstruction import producer, wire
        identity = dict(source_commit='b' * 40, lineage_sha=producer.LINEAGE,
                        fingerprint=producer.FINGERPRINT)
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            config = workspace / 'pair.cfg'
            config.write_text('ref\nsource\n')
            # Four inventory-sensitive native lines plus exact canonical paths.
            options = (b'COLMAP_OPTIMIZED_PHOTO_OPTIONS_V1\n'
                       b'opt_bool geom_consistency 0\nopt_bool filter 0\n'
                       b'opt_int max_image_size -1\n'
                       b'opt_double cache_size 3ff0000000000000\n')
            for key, path in [('workspace', workspace), ('config', config)]:
                raw = os.fsencode(path)
                options += b'opt_string ' + key.encode() + b' ' + str(len(raw)).encode() + b' ' + raw.hex().encode() + b'\n'
            job = dict(provenance='native-snapshot', reference_image_idx=0,
                       source_image_idxs=[2, 1], width=3, height=2,
                       source_dimensions=[[3, 2], [2, 2]], cache_size_gib=1,
                       options_bytes=options, workspace=str(workspace), config=str(config),
                       files={str(config): 'a' * 64})
            artifact = wire.emit_native_v3(jobs=[job], ceiling=1, rolling_active_jobs=1,
                capacity_bytes=dict(host=8 * 2**30, device=8 * 2**30),
                build_identity=identity)
            payload = wire.export_fire(artifact)
            self.assertIn(b'source_commit=' + b'b' * 40, payload)
            self.assertIn(b'source_image_idxs=2 1', payload)
            with self.assertRaisesRegex(ValueError, 'capacity|admit'):
                wire.emit_native_v3(jobs=[job], ceiling=1, rolling_active_jobs=1,
                    capacity_bytes=dict(host=1, device=1), build_identity=identity)

    def test_cmake_install_launcher_runs_outside_source_without_python_roots(self) -> None:
        root = Path(__file__).resolve().parents[2]
        # Some CI hosts mount /tmp noexec; use an executable sibling directory.
        with tempfile.TemporaryDirectory(dir=root.parent) as directory:
            temp = Path(directory)
            # Install rules are generated without configuring/building pycolmap.
            configured = subprocess.run(['cmake', '-S', str(root / 'python/tests/cmake_fixture'),
                '-B', str(temp / 'build'), '-DPRODUCT_ROOT=' + str(root),
                '-DCMAKE_INSTALL_PREFIX=' + str(temp / 'prefix')], capture_output=True, text=True)
            self.assertEqual(configured.returncode, 0, configured.stderr)
            installed = subprocess.run(['cmake', '--install', str(temp / 'build')], capture_output=True, text=True)
            self.assertEqual(installed.returncode, 0, installed.stderr)
            env = {k: v for k, v in os.environ.items() if k not in ('PYTHONPATH', 'PYTHONHOME')}
            result = subprocess.run([str(temp / 'prefix/bin/colmap-reconstruct'), '--help'],
                cwd=temp, env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('--workspace', result.stdout)

    def test_owned_process_deadline_drains_child_and_preserves_exit_failure(self) -> None:
        from colmap_reconstruction.process import NativeProcess, NativeFailure
        with tempfile.TemporaryDirectory() as directory:
            process = NativeProcess(Path(directory), timeout_seconds=0.1)
            with self.assertRaisesRegex(NativeFailure, 'deadline'):
                process.run([sys.executable, '-c', 'import time; time.sleep(60)'])
            self.assertIsNone(process.child)
            second = Path(directory) / 'second'
            second.mkdir()
            process = NativeProcess(second, timeout_seconds=5)
            with self.assertRaisesRegex(NativeFailure, 'exit 7'):
                process.run([sys.executable, '-c', 'raise SystemExit(7)'])

    def test_resource_admission_preserves_absolute_headroom_without_second_margin(self) -> None:
        from colmap_reconstruction.resources import admit_resources
        admit_resources(host_available=1100, device_available=2200,
                        host_budget=1000, device_budget=2000,
                        host_headroom=100, device_headroom=200)
        with self.assertRaisesRegex(ValueError, 'host'):
            admit_resources(host_available=1099, device_available=2200,
                            host_budget=1000, device_budget=2000,
                            host_headroom=100, device_headroom=200)
        with self.assertRaisesRegex(ValueError, 'device'):
            admit_resources(host_available=1100, device_available=2199,
                            host_budget=1000, device_budget=2000,
                            host_headroom=100, device_headroom=200)

    def test_cli_refuses_hard_caps_and_existing_workspace_before_native_launch(self) -> None:
        from colmap_reconstruction.cli import main
        with tempfile.TemporaryDirectory() as directory:
            photos = Path(directory) / 'photos'
            photos.mkdir()
            args = ['--experimental-local', '--photos', str(photos),
                    '--workspace', str(Path(directory) / 'new'),
                    '--host-budget-bytes', '1000', '--device-budget-bytes', '1000']
            self.assertEqual(main(args + ['--require-hard-memory-cap']), 2)
            self.assertFalse((Path(directory) / 'new').exists())
            self.assertEqual(main(args + ['--workspace', str(photos)]), 2)

    def test_complete_cli_wires_fresh_sfm_all_references_geo_and_fusion(self) -> None:
        from colmap_reconstruction.resources import host_available_bytes
        if host_available_bytes() < 2500 * 2**20:
            self.skipTest('synthetic native ledger needs 2.5 GiB available for real admission checks')
        import shutil
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=root.parent.parent) as directory:
            temp = Path(directory)
            configured = subprocess.run(['cmake', '-S', str(root / 'tests/cmake_fixture'),
                '-B', str(temp / 'build'), '-DPRODUCT_ROOT=' + str(root.parent),
                '-DCMAKE_INSTALL_PREFIX=' + str(temp / 'prefix')], capture_output=True, text=True)
            self.assertEqual(configured.returncode, 0, configured.stderr)
            installed = subprocess.run(['cmake', '--install', str(temp / 'build')], capture_output=True, text=True)
            self.assertEqual(installed.returncode, 0, installed.stderr)
            native = temp / 'prefix/bin/colmap'
            shutil.copy(root / 'tests/fake_native.py', native)
            native.chmod(0o755)
            (temp / 'nvidia-smi').symlink_to(native)
            photos = temp / 'photos'
            photos.mkdir()
            env = dict(os.environ, PATH=str(temp) + os.pathsep + os.environ['PATH'])
            env.pop('PYTHONPATH', None)
            env.pop('PYTHONHOME', None)
            argv = [str(temp / 'prefix/bin/colmap-reconstruct'),
                '--experimental-local', '--photos', str(photos),
                '--workspace', str(temp / 'output'), '--host-budget-bytes', str(2300 * 2**20),
                '--device-budget-bytes', str(8 * 2**30), '--host-headroom-bytes', str(64 * 2**20), '--no-lookahead', '--cache-size-gib', '0.01']
            completed = subprocess.run(argv, env=env, cwd=temp, capture_output=True, text=True)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            result = json.loads((temp / 'output/result.json').read_text())
            self.assertEqual(result['photo_references'], 2)
            self.assertEqual(result['geo_references'], 2)
            self.assertEqual(result['vertices'], 1)
            self.assertFalse(result['performance_qualified'])
            self.assertFalse(result['hard_memory_cap'])
            bad_argv = [str(value).replace(str(temp / 'output'), str(temp / 'bad-output')) for value in argv]
            refused = subprocess.run(bad_argv, env=dict(env, FAKE_BAD_FINGERPRINT='1'),
                cwd=temp, capture_output=True, text=True)
            self.assertNotEqual(refused.returncode, 0)
            self.assertIn('fingerprint mismatch', refused.stderr)
            self.assertFalse((temp / 'bad-output/sparse').exists())

    def test_output_validation_rejects_truncation_and_nonfinite_maps(self) -> None:
        import struct
        from colmap_reconstruction.cli import validate_photo_map, validate_ply
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'map.bin'
            expected = dict(width=2, height=1, channels=1)
            path.write_bytes(b'2&1&1&' + struct.pack('<ff', 1, 2))
            validate_photo_map(path, expected)
            path.write_bytes(b'2&1&1&' + struct.pack('<ff', 1, float('nan')))
            with self.assertRaisesRegex(ValueError, 'nonfinite'):
                validate_photo_map(path, expected)
            ply = Path(directory) / 'points.ply'
            ply.write_bytes(b'ply\nformat binary_little_endian 1.0\nelement vertex 2\nproperty float x\nproperty float y\nproperty float z\nend_header\n' + struct.pack('<fff', 1, 2, 3))
            with self.assertRaisesRegex(ValueError, 'truncated'):
                validate_ply(ply)

    def test_canonical_photo_charge_has_one_margin_per_disjoint_term(self) -> None:
        from colmap_reconstruction import model
        self.assertEqual(model.photo_phase_private_charge(3, 2, 2, cache_size_gib=1,
            input_pixels=16, max_source_width=3, max_source_height=2),
            dict(host=182, device=2388))
        with self.assertRaisesRegex(ValueError, 'cache'):
            model.photo_phase_private_charge(3, 2, 2, cache_size_gib=1e-9,
                input_pixels=16, max_source_width=3, max_source_height=2)

    def test_owned_process_observes_cancel_and_host_overbudget(self) -> None:
        from colmap_reconstruction.process import NativeProcess, NativeFailure
        import threading
        with tempfile.TemporaryDirectory() as directory:
            process = NativeProcess(Path(directory), timeout_seconds=5)
            timer = threading.Timer(0.1, lambda: setattr(process, 'cancelled', True))
            timer.start()
            try:
                with self.assertRaisesRegex(NativeFailure, 'cancelled'):
                    process.run([sys.executable, '-c', 'import time; time.sleep(60)'])
            finally:
                timer.join()
            self.assertIsNone(process.child)
        with tempfile.TemporaryDirectory() as directory:
            process = NativeProcess(Path(directory), timeout_seconds=5, host_budget_bytes=1024)
            with self.assertRaisesRegex(NativeFailure, 'RSS exceeds budget'):
                process.run([sys.executable, '-c', 'import time; time.sleep(60)'])
            self.assertIsNone(process.child)

    def test_packaged_geo_support_matches_actual_native_build_closure(self) -> None:
        import importlib.util
        from colmap_reconstruction import resident
        root = Path(__file__).resolve().parents[2]
        spec = importlib.util.spec_from_file_location('resident_generator',
            root / 'scripts/resident/generate_resident_allocation_fingerprint.py')
        generator = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(generator)
        self.assertEqual(generator.compute_fingerprint(root), resident.SUPPORTED_ALLOCATION_FINGERPRINT)

    def test_package_import_is_independent_of_extension_and_external_roots(self) -> None:
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(
                [sys.executable, '-I', '-c',
                 'import sys; sys.path.insert(0, sys.argv[1]); '
                 'from colmap_reconstruction import model, producer, wire; '
                 'assert model.margin_bytes(101) == 105; '
                 'assert model.photo_phase_boundary_pixels(3, 2) == 6; '
                 'assert not any(k == "hardware" or k == "tools" or k == "pycolmap" for k in sys.modules)',
                 str(root)], cwd=directory, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == '__main__':
    unittest.main()
