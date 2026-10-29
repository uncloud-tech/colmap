# SPDX-License-Identifier: BSD-3-Clause
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


def fake_cuda_environment(temp, env):
    """Build only a synthetic Driver API seam, never install/use a CUDA driver."""
    import shutil
    compiler = shutil.which('cc')
    if not compiler:
        raise unittest.SkipTest('C compiler required for synthetic CUDA identity seam')
    source = Path(__file__).with_name('fake_cuda.c')
    subprocess.run([compiler, '-shared', '-fPIC', str(source), '-o', str(temp / 'libcuda.so.1')], check=True)
    return dict(env, LD_LIBRARY_PATH=str(temp) + os.pathsep + env.get('LD_LIBRARY_PATH', ''))


class ReconstructionTest(unittest.TestCase):
    def test_resources_follow_cuda_visible_ordinal_not_management_index(self) -> None:
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=root.parent.parent) as directory:
            temp = Path(directory)
            env = fake_cuda_environment(temp, dict(os.environ, CUDA_VISIBLE_DEVICES='1,0',
                                                   CUDA_DEVICE_ORDER='PCI_BUS_ID'))
            script = '''import sys
sys.path.insert(0, sys.argv[1])
from types import SimpleNamespace
from colmap_reconstruction.resources import observe_resources
class Process:
    environment = dict(__import__('os').environ)
    def run(self, argv, capture=False):
        assert argv[2] == 'GPU-00000000-0000-0000-0000-000000000002', argv
        return b'32768\\n'
a = SimpleNamespace(gpu_index=0, host_budget_bytes=1, device_budget_bytes=1,
    host_headroom_bytes=0, device_headroom_bytes=0)
r = observe_resources(Process(), a)
assert r['cuda_device_uuid'] == 'GPU-00000000-0000-0000-0000-000000000002'
assert r['cuda_visible_devices'] == '1,0'
assert Process.environment['CUDA_VISIBLE_DEVICES'] == '1,0'
'''
            result = subprocess.run([sys.executable, '-I', '-B', '-c', script, str(root)],
                                    env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_photo_concurrency_is_explicit_one_or_two_default_one(self) -> None:
        from colmap_reconstruction.cli import parse_args
        base = ['--photos', '/photos', '--workspace', '/new',
                '--host-budget-bytes', '1000', '--device-budget-bytes', '1000']
        self.assertEqual(parse_args(base).photo_concurrent_solvers, 1)
        self.assertEqual(parse_args(base + ['--photo-concurrent-solvers', '2']).photo_concurrent_solvers, 2)
        for value in ('0', '3', 'auto'):
            with self.subTest(value=value), self.assertRaises(SystemExit):
                parse_args(base + ['--photo-concurrent-solvers', value])

    def test_stereo_timeout_uses_real_native_integer_parser_without_extension(self) -> None:
        import time
        from colmap_reconstruction.cli import Reconstruction, parse_args
        from colmap_reconstruction.process import NativeFailure
        import shutil
        compiler = shutil.which('g++')
        if not compiler:
            self.skipTest('g++ required for real native parser regression')
        root = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory(dir=root.parent) as directory:
            temp = Path(directory)
            probe = temp / 'probe'
            subprocess.run([compiler, '-std=c++17', '-pthread', '-I' + str(root / 'src'),
                str(root / 'python/tests/reconstruction_protocol_probe.cc'), '-o', str(probe)], check=True)
            args = parse_args(['--photos', str(temp), '--workspace', str(temp),
                               '--host-budget-bytes', '1000', '--device-budget-bytes', '1000'])
            reconstruction = Reconstruction(args, sys.executable)
            reconstruction.process.deadline = time.monotonic() + 1.9
            for phase in ('photo', 'geo'):
                flags = reconstruction.strict_flags(phase, temp / 'forecast', 0)
                seconds = flags[flags.index('--owned_run_timeout_seconds') + 1]
                self.assertEqual(seconds, '1')
                parsed = subprocess.run([str(probe), 'timeout', seconds], capture_output=True)
                self.assertEqual(parsed.returncode, 0, parsed.stderr)
            reconstruction.process.deadline = time.monotonic() + 0.5
            with self.assertRaisesRegex(NativeFailure, 'deadline'):
                reconstruction.strict_flags('photo', temp / 'forecast', 0)

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

    def prepare_sparse_models(self, models, selection):
        """Exercise preparation with real synthetic native subprocesses, no GPU."""
        from unittest.mock import patch
        from colmap_reconstruction.cli import Reconstruction, parse_args
        import shutil
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[2].parent) as directory:
            temp = Path(directory)
            native = temp / 'colmap'
            shutil.copy(Path(__file__).with_name('fake_native.py'), native)
            native.chmod(0o755)
            workspace = temp / 'output'
            workspace.mkdir()
            args = ['--photos', str(temp), '--workspace', str(workspace),
                    '--host-budget-bytes', str(2**30), '--device-budget-bytes', str(2**30)]
            if selection is not None:
                args += ['--sparse-model', selection]
            calls = temp / 'calls.jsonl'
            with patch.dict(os.environ, FAKE_SPARSE_MODELS=json.dumps(models), FAKE_CALLS=str(calls)):
                reconstruction = Reconstruction(parse_args(args), str(native))
                reconstruction.assets.mkdir()
                with patch.object(reconstruction, 'resources'):
                    reconstruction.prepare()
            return (json.loads((reconstruction.assets / 'sparse-model.json').read_text()),
                    [json.loads(line) for line in calls.read_text().splitlines()])

    def test_sparse_largest_selects_ws1_by_registered_images(self) -> None:
        record, calls = self.prepare_sparse_models({
            '0': dict(num_registered_images=5, num_points3D=229),
            '1': dict(num_registered_images=9, num_points3D=4380)}, 'largest')
        self.assertEqual(record['name'], '1')
        self.assertEqual(record['policy'], 'largest')
        self.assertEqual(record['counts']['num_registered_images'], 9)
        self.assertEqual([Path(a['--path']).name for c, a in calls if c == 'model_analyzer'], ['0', '1'])
        self.assertEqual([Path(a['--input_path']).name for c, a in calls if c == 'image_undistorter'], ['1'])

    def test_sparse_largest_rejects_invalid_native_counts(self) -> None:
        good = dict(num_registered_images=9, num_points3D=3498)
        invalid = [None, [], {}, dict(num_registered_images=9),
                   dict(num_registered_images=True, num_points3D=4),
                   dict(num_registered_images=9.0, num_points3D=4),
                   dict(num_registered_images=9, num_points3D=True),
                   dict(num_registered_images=9, num_points3D=1.5),
                   dict(num_registered_images=9, num_points3D='4'),
                   dict(num_registered_images=0, num_points3D=0),
                   dict(num_registered_images=1, num_points3D=4),
                   dict(num_registered_images=9, num_points3D=0)]
        for counts in invalid:
            with self.subTest(counts=counts), self.assertRaisesRegex(ValueError, 'invalid|insufficient'):
                self.prepare_sparse_models({'0': good, '1': counts}, 'largest')

    def test_sparse_largest_handles_ws2_renumbering_and_lexicographic_ties(self) -> None:
        ws2 = {'0': dict(num_registered_images=9, num_points3D=3498)}
        record, calls = self.prepare_sparse_models(ws2, 'largest')
        self.assertEqual((record['name'], record['policy']), ('0', 'largest'))
        # Point count and numeric directory ordering must not break image-count ties.
        record, calls = self.prepare_sparse_models({
            '2': dict(num_registered_images=9, num_points3D=9999),
            '10': dict(num_registered_images=9, num_points3D=4)}, 'largest')
        self.assertEqual(record['name'], '10')
        self.assertEqual(sum(c == 'model_analyzer' for c, a in calls), 2)

    def test_sparse_default_and_exact_name_selection_stay_explicit(self) -> None:
        models = {'0': dict(num_registered_images=5, num_points3D=229),
                  '1': dict(num_registered_images=9, num_points3D=4380)}
        with self.assertRaisesRegex(ValueError, 'select --sparse-model'):
            self.prepare_sparse_models(models, None)
        for selection in ('0', '1'):
            record, calls = self.prepare_sparse_models(models, selection)
            self.assertEqual((record['name'], record['policy']), (selection, 'explicit'))
            self.assertEqual([Path(a['--path']).name for c, a in calls if c == 'model_analyzer'], [selection])
        ws2 = {'0': dict(num_registered_images=9, num_points3D=3498)}
        with self.assertRaisesRegex(ValueError, 'select --sparse-model'):
            self.prepare_sparse_models(ws2, '1')
        record, calls = self.prepare_sparse_models(ws2, None)
        self.assertEqual((record['name'], record['policy']), ('0', 'single'))

    def test_sparse_selection_refuses_no_models_and_native_failure(self) -> None:
        from colmap_reconstruction.process import NativeFailure
        for selection in (None, '0', 'largest'):
            with self.subTest(selection=selection), self.assertRaisesRegex(ValueError, 'select --sparse-model'):
                self.prepare_sparse_models({}, selection)
        with self.assertRaises(NativeFailure):
            self.prepare_sparse_models({'0': 'native-failure'}, 'largest')
        with self.assertRaises(json.JSONDecodeError):
            self.prepare_sparse_models({'0': 'malformed-json'}, 'largest')
        for selection in (None, '0'):
            with self.subTest(selection=selection), self.assertRaisesRegex(ValueError, 'insufficient'):
                self.prepare_sparse_models({'0': dict(num_registered_images=0, num_points3D=0)}, selection)

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
            env = fake_cuda_environment(temp, env)
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

    def test_photo_n2_launches_all_four_jobs_once_before_geo(self) -> None:
        from colmap_reconstruction import wire
        from colmap_reconstruction.resources import host_available_bytes
        if os.geteuid() == 0:
            self.skipTest('ordinary local CLI deliberately refuses root')
        if host_available_bytes() < 2500 * 2**20:
            self.skipTest('synthetic ledger requires 2.5 GiB available')
        import shutil
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=root.parent.parent) as directory:
            temp = Path(directory)
            native = temp / 'colmap'
            shutil.copy(root / 'tests/fake_native.py', native)
            native.chmod(0o755)
            (temp / 'nvidia-smi').symlink_to(native)
            (temp / 'photos').mkdir()
            env = dict(os.environ, PATH=str(temp) + os.pathsep + os.environ['PATH'],
                       FAKE_REGISTERED_IMAGES='4', FAKE_CALLS=str(temp / 'calls'),
                       CUDA_VISIBLE_DEVICES='1,0', CUDA_DEVICE_ORDER='PCI_BUS_ID')
            env = fake_cuda_environment(temp, env)
            # No install/source-external imports; launch the stdlib entry under -I.
            argv = [sys.executable, '-I', '-B', '-c',
                    'import sys; sys.path.insert(0, sys.argv.pop(1)); '
                    'from colmap_reconstruction.cli import main; raise SystemExit(main())',
                    str(root), '--colmap', str(native), '--experimental-local',
                    '--photos', str(temp / 'photos'), '--workspace', str(temp / 'output'),
                    '--host-budget-bytes', str(2300 * 2**20), '--device-budget-bytes', str(8 * 2**30),
                    '--host-headroom-bytes', str(64 * 2**20), '--cache-size-gib', '0.01',
                    '--no-lookahead', '--photo-concurrent-solvers', '2']
            completed = subprocess.run(argv, env=env, cwd=temp, capture_output=True, text=True)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            planning = temp / 'output/planning'
            self.assertEqual(len(list(planning.glob('photo-*.forecast'))), 1)
            payload = (planning / 'photo-0.forecast').read_bytes()
            launch = wire.parse_v3_launch(payload, requested_active_jobs=2,
                options_bytes=[(planning / ('photo-%d.options' % i)).read_bytes() for i in range(4)])
            self.assertEqual(launch['ceiling'], 2)
            self.assertEqual([j['forecast']['reference_image_idx'] for j in launch['jobs']], [0, 1, 2, 3])
            self.assertEqual([j['forecast']['source_image_idxs'] for j in launch['jobs']], [[1], [2], [3], [0]])
            self.assertEqual(len(launch['bindings']), 4)
            # Canonical 3x2/single-source fixture: one process/cache charge,
            # two already-margined private owners, no shared-gray credits.
            exact_peak = dict(host=2222968047, device=2211911306)
            wire.parse_v3_launch(payload, requested_active_jobs=2, capacity_bytes=exact_peak)
            for side in ('host', 'device'):
                insufficient = dict(exact_peak)
                insufficient[side] -= 1
                with self.assertRaisesRegex(ValueError, 'capacity-exceeded: ' + side):
                    wire.parse_v3_launch(payload, requested_active_jobs=2, capacity_bytes=insufficient)
            if shutil.which('g++'):
                probe = temp / 'probe'
                subprocess.run(['g++', '-std=c++17', '-pthread', '-I' + str(root.parent / 'src'),
                    str(root / 'tests/reconstruction_protocol_probe.cc'), '-o', str(probe)], check=True)
                native_parsed = subprocess.run([str(probe), 'wire', str(planning / 'photo-0.forecast'),
                    '2', str(exact_peak['host']), str(exact_peak['device'])], capture_output=True, text=True)
                self.assertEqual(native_parsed.returncode, 0, native_parsed.stderr)
                self.assertEqual(native_parsed.stdout, '2 4 4')
                native_refused = subprocess.run([str(probe), 'wire', str(planning / 'photo-0.forecast'),
                    '2', str(exact_peak['host'] - 1), str(exact_peak['device'])], capture_output=True, text=True)
                self.assertNotEqual(native_refused.returncode, 0)
                self.assertIn('top-N private charges exceed caps', native_refused.stderr)
            for i in range(4):
                self.assertEqual((planning / ('photo-%d.cfg' % i)).read_text(),
                                 'image%d.jpg\n__auto__, 20\n' % i)
            calls = [json.loads(line) for line in (temp / 'calls').read_text().splitlines()]
            photo = [(i, a) for i, (c, a) in enumerate(calls) if '--owned_photo_forecast' in a]
            self.assertEqual(len(photo), 1)
            self.assertEqual(photo[0][1]['--owned_run_max_concurrent_solvers'], '2')
            self.assertEqual(photo[0][1]['--PatchMatchStereo.gpu_index'], '0')
            for observation in planning.glob('resources-*.json'):
                recorded = json.loads(observation.read_text())
                self.assertEqual(recorded['cuda_ordinal'], 0)
                self.assertEqual(recorded['cuda_device_uuid'], 'GPU-00000000-0000-0000-0000-000000000002')
                self.assertEqual(recorded['cuda_visible_devices'], '1,0')
            geo = [(i, a) for i, (c, a) in enumerate(calls) if '--owned_run_forecast' in a]
            self.assertEqual(len(geo), 1)
            self.assertGreater(geo[0][0], photo[0][0])
            self.assertEqual(geo[0][1]['--owned_run_max_concurrent_solvers'], '1')
            self.assertEqual(json.loads((temp / 'output/result.json').read_text())['photo_references'], 4)
            # Even a partially writing failed PHOTO cannot cross the GEO barrier.
            failed_argv = [v.replace(str(temp / 'output'), str(temp / 'failed')) for v in argv]
            failed = subprocess.run(failed_argv, env=dict(env, FAKE_PHOTO_FAILURE='1',
                FAKE_CALLS=str(temp / 'failed-calls')), cwd=temp, capture_output=True, text=True)
            self.assertNotEqual(failed.returncode, 0)
            failed_calls = [json.loads(line) for line in (temp / 'failed-calls').read_text().splitlines()]
            self.assertFalse(any('--owned_run_forecast' in a or '--owned_metadata_export' in a
                                 or c == 'stereo_fusion' for c, a in failed_calls))
            self.assertTrue((temp / 'failed/failure.json').is_file())
            # Default N1 remains four singleton native launches, never truncates.
            n1_argv = [v.replace(str(temp / 'output'), str(temp / 'n1')) for v in argv[:-2]]
            n1 = subprocess.run(n1_argv, env=dict(env, FAKE_CALLS=str(temp / 'n1-calls')),
                                cwd=temp, capture_output=True, text=True)
            self.assertEqual(n1.returncode, 0, n1.stderr)
            n1_calls = [json.loads(line) for line in (temp / 'n1-calls').read_text().splitlines()]
            n1_photo = [a for c, a in n1_calls if '--owned_photo_forecast' in a]
            self.assertEqual(len(n1_photo), 4)
            for a in n1_photo:
                self.assertEqual(a['--owned_run_max_concurrent_solvers'], '1')
                n1_launch = wire.parse_v3_launch(Path(a['--owned_photo_forecast']).read_bytes())
                self.assertEqual(len(n1_launch['jobs']), 1)
            self.assertEqual(json.loads((temp / 'n1/result.json').read_text())['photo_references'], 4)

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

    def test_python_validation_observes_cancellation_and_expiry_between_chunks(self) -> None:
        import struct
        import time
        from colmap_reconstruction.cli import digest, validate_photo_map, validate_ply
        from colmap_reconstruction.process import NativeProcess, NativeFailure
        with tempfile.TemporaryDirectory() as directory:
            temp = Path(directory)
            path = temp / 'payload'
            cases = [
                (lambda check: digest(path, check=check), b'x' * (2 * 1024**2), 2),
                (lambda check: validate_photo_map(path, dict(width=600000, height=1, channels=1), check=check),
                 b'600000&1&1&' + struct.pack('<f', 1) * 600000, 3),
                (lambda check: validate_ply(path, check=check),
                 b'ply\nformat ascii 1.0\nelement vertex 10000\nproperty float x\nproperty float y\nproperty float z\nend_header\n' + b'0 0 0\n' * 10000, 9),
                (lambda check: validate_ply(path, check=check),
                 b'ply\nformat binary_little_endian 1.0\nelement vertex 10000\nproperty float x\nproperty float y\nproperty float z\nend_header\n' + struct.pack('<fff', 0, 0, 0) * 10000, 9),
            ]
            for operation, content, cancel_at in cases:
                path.write_bytes(content)
                for reason in ('cancelled', 'deadline'):
                    with self.subTest(operation=operation, reason=reason):
                        process = NativeProcess(temp, timeout_seconds=10)
                        calls = []
                        def check():
                            calls.append(1)
                            if len(calls) == cancel_at:
                                if reason == 'cancelled':
                                    process.cancelled = True
                                else:
                                    process.deadline = time.monotonic() - 1
                            process.remaining()
                        with self.assertRaisesRegex(NativeFailure, reason):
                            operation(check)
                        self.assertEqual(len(calls), cancel_at)

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
