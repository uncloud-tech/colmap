# SPDX-License-Identifier: BSD-3-Clause
# Selectively ported PR16 701da8fd1; see colmap_reconstruction/NOTICE.
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


def allowances():
    argv = ['--unknown-provenance', 'test-only explicit uncalibrated allowances']
    for name in ('loader-scratch-host', 'process-host', 'process-device',
                 'unknown-host', 'unknown-device', 'reserve-host', 'reserve-device'):
        argv += ['--' + name + '-bytes', str(1024 * 1024)]
    return argv


class ReconstructionTest(unittest.TestCase):
    def test_cmake_install_launcher_runs_outside_source_without_python_roots(self) -> None:
        root = Path(__file__).resolve().parents[2]
        # Some CI hosts mount /tmp noexec; use an executable sibling directory.
        with tempfile.TemporaryDirectory(dir=root.parent) as directory:
            temp = Path(directory)
            (temp / 'photos').mkdir()
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
            if os.geteuid() == 0:
                refused = subprocess.run([str(temp / 'prefix/bin/colmap-reconstruct'),
                    '--experimental-local', '--yes', '--photos', str(temp / 'photos'), '--workspace', str(temp / 'root-refused'),
                    *allowances()], cwd=temp, env=env, capture_output=True)
                self.assertNotEqual(refused.returncode, 0)
                self.assertFalse((temp / 'root-refused').exists())
                return  # Ordinary-user full-chain seam is exercised on non-root hosts.
            import shutil
            native = temp / 'prefix/bin/colmap'
            shutil.copy(Path(__file__).with_name('fake_native.py'), native)
            native.chmod(0o755)
            for scenario in ('good', 'defaults', 'depth', 'source-order', 'fingerprint', 'fusion', 'resources',
                             'fusion-input', 'fusion-output', 'fusion-unsupported',
                             'undistort-input', 'undistort-worker', 'undistort-unsupported'):
                calls = temp / (scenario + '.jsonl')
                workspace = temp / scenario
                scenario_env = dict(env, FAKE_CALLS=str(calls), PYTHONPATH='/absent/lab:/absent/photogram')
                if scenario not in ('good', 'defaults'):
                    scenario_env['FAKE_BAD_' + {'depth': 'DEPTH', 'source-order': 'ORDER', 'fingerprint': 'FP', 'fusion': 'FUSION', 'resources': 'FREE',
                        'fusion-input': 'FUSION_INPUT', 'fusion-output': 'FUSION_OUTPUT',
                        'fusion-unsupported': 'FUSION_UNSUPPORTED',
                        'undistort-input': 'UNDISTORT_INPUT', 'undistort-worker': 'UNDISTORT_WORKER',
                        'undistort-unsupported': 'UNDISTORT_UNSUPPORTED'}[scenario]] = '1'
                result = subprocess.run([str(temp / 'prefix/bin/colmap-reconstruct'),
                    '--experimental-local', '--yes', '--photos', str(temp / 'photos'), '--workspace', str(workspace),
                    *(allowances() if scenario != 'defaults' else [])],
                    cwd=temp, env=scenario_env, capture_output=True, text=True)
                if scenario in ('good', 'defaults'):
                    self.assertEqual(result.returncode, 0, result.stderr)
                    report = json.loads((workspace / 'result.json').read_text())
                    self.assertEqual(report['backend'], 'experimental_fixed_cohort')
                    self.assertEqual(report['device_selection']['ordinal'], 0)
                    self.assertEqual(report['device_selection']['device_uuid'], 'GPU-synthetic')
                    self.assertEqual(report['device_inventory'][0]['name'], 'Synthetic CUDA')
                    commands = [command for command, arguments in map(json.loads, calls.read_text().splitlines())
                                if command != 'reconstruction_info' or '--gpu_index' not in arguments]
                    if scenario == 'defaults':
                        policy = report['resource_policy']
                        self.assertEqual(policy['version'], 'experimental-stereo-v1')
                        expected = dict(loader_scratch_host_bytes=536870912,
                            process_host_bytes=536870912, process_device_bytes=536870912,
                            unknown_host_bytes=1073741824, unknown_device_bytes=1073741824,
                            reserve_host_bytes=536870912, reserve_device_bytes=536870912)
                        dense = [a for c, a in map(json.loads, calls.read_text().splitlines())
                                 if c == 'experimental_fixed_cohort'][0]
                        for name, value in expected.items():
                            self.assertEqual(policy['terms'][name]['bytes'], value)
                            self.assertEqual(policy['terms'][name]['origin'], 'default')
                            self.assertEqual(policy['terms'][name]['classification'], 'conservative-policy')
                            self.assertEqual(dense['--' + name], str(value))
                        fusion = [a for c, a in map(json.loads, calls.read_text().splitlines())
                                  if c == 'stereo_fusion'][0]
                        self.assertEqual(fusion['--StereoFusion.host_budget_bytes'], str(16 * 1024**3))
                        self.assertEqual(fusion['--StereoFusion.policy_host_bytes'], str(2 * 1024**3))
                        self.assertEqual(fusion['--StereoFusion.reserve_host_bytes'], str(512 * 1024**2))
                        undistort = [a for c, a in map(json.loads, calls.read_text().splitlines())
                                     if c == 'image_undistorter'][0]
                        self.assertEqual(undistort['--host_budget_bytes'], str(16 * 1024**3))
                        self.assertEqual(undistort['--policy_host_bytes'], str(2 * 1024**3))
                        self.assertEqual(undistort['--reserve_host_bytes'], str(512 * 1024**2))
                        self.assertIn('undistortion', policy)
                        self.assertIsNone(policy['whole_pipeline_physical_fit'])
                        self.assertEqual(json.loads((workspace / 'planning/resource-policy.json').read_text()), policy)
                    self.assertEqual(commands, ['reconstruction_info', 'photo_directory_summary', 'feature_extractor',
                        'exhaustive_matcher', 'mapper', 'model_analyzer', 'image_undistorter',
                        'optimized_dense_plan', 'experimental_fixed_cohort', 'stereo_fusion'])
                else:
                    self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
                    self.assertFalse((workspace / 'result.json').exists())
                if not workspace.exists():
                    self.assertEqual(scenario, 'resources')
                    continue  # Preview refusal must not create a workspace/failure record.
                before = calls.read_bytes()
                again = subprocess.run([str(temp / 'prefix/bin/colmap-reconstruct'),
                    '--experimental-local', '--yes', '--photos', str(temp / 'photos'), '--workspace', str(workspace),
                    *allowances()], cwd=temp, env=scenario_env, capture_output=True)
                self.assertNotEqual(again.returncode, 0)
                self.assertEqual(calls.read_bytes(), before)

    @unittest.skipUnless(sys.platform == 'linux' and os.geteuid() != 0,
                         'Experimental local execution requires non-root Linux')
    def test_resolution_no_fit_never_retries_at_reduced_size(self):
        import shutil
        from colmap_reconstruction.cli import main
        for stage in ('EXTRACTION_NOFIT', 'NOFIT', 'DENSE_RESIZE'):
            with self.subTest(stage=stage), tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[2].parent) as directory:
                root = Path(directory)
                photos = root / 'photos'
                photos.mkdir()
                original = photos / 'original.jpg'
                original.write_bytes(b'original fixture untouched')
                native = root / 'colmap'
                shutil.copy(Path(__file__).with_name('fake_native.py'), native)
                native.chmod(0o755)
                calls = root / 'calls.jsonl'
                from unittest.mock import patch
                with patch.dict(os.environ, {'FAKE_' + stage: '1', 'FAKE_CALLS': str(calls)}):
                    status = main(['--experimental-local', '--yes', '--photos', str(photos),
                        '--workspace', str(root / 'output'), '--colmap', str(native), *allowances()])
                self.assertEqual(status, 2)
                self.assertEqual(original.read_bytes(), b'original fixture untouched')
                self.assertFalse((root / 'output/result.json').exists())
                records = list(map(json.loads, calls.read_text().splitlines()))
                extraction = [a for c, a in records if c == 'feature_extractor']
                self.assertEqual(len(extraction), 1)
                self.assertEqual(extraction[0]['--FeatureExtraction.max_image_size'], '4004')
                self.assertEqual(extraction[0]['--FeatureExtraction.use_gpu'], 'true')
                self.assertNotIn('stereo_fusion', [c for c, a in records])
                self.assertNotIn('patch_match_stereo', [c for c, a in records])
                if stage == 'DENSE_RESIZE':
                    self.assertNotIn('experimental_fixed_cohort', [c for c, a in records])
                    self.assertIn('unrequested resolution cap',
                        json.loads((root / 'output/failure.json').read_text())['reason'])
                if stage == 'NOFIT':
                    undistort = [a for c, a in records if c == 'image_undistorter']
                    self.assertEqual(len(undistort), 1)
                    self.assertEqual(undistort[0]['--max_image_size'], '-1')
                    self.assertEqual(len([c for c, a in records if c == 'experimental_fixed_cohort']), 1)
                    effective = json.loads((root / 'output/planning/resolution-effective.json').read_text())
                    self.assertEqual(effective['images'], [
                        dict(name='image0.pgm', width=3, height=2),
                        dict(name='image1.pgm', width=3, height=2)])
                    self.assertEqual(effective['stereo_max_image_size'], -1)
                    self.assertEqual(effective['fusion_max_image_size'], -1)
                    self.assertEqual(effective['map_precision'], 'float32')

    @unittest.skipUnless(sys.platform == 'linux' and os.geteuid() != 0,
                         'Experimental local execution requires non-root Linux')
    def test_default_policy_partial_override_and_lower_only_caps(self):
        import shutil
        root = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory(dir=root.parent) as directory:
            temp = Path(directory)
            (temp / 'photos').mkdir()
            native = temp / 'colmap'
            shutil.copy(Path(__file__).with_name('fake_native.py'), native)
            native.chmod(0o755)
            workspace = temp / 'output'
            calls = temp / 'calls.jsonl'
            env = dict(os.environ, PYTHONPATH=str(root / 'python'), FAKE_CALLS=str(calls))
            result = subprocess.run([sys.executable, '-m', 'colmap_reconstruction.cli',
                '--colmap', str(native), '--experimental-local', '--yes', '--photos', str(temp / 'photos'),
                '--workspace', str(workspace), '--process-host-bytes', '123',
                '--loader-scratch-host-bytes', '0',
                '--host-budget-bytes', str(12 * 1024**3),
                '--device-budget-bytes', str(100 * 1024**3)],
                cwd=temp, env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            policy = json.loads((workspace / 'result.json').read_text())['resource_policy']
            self.assertEqual(policy['terms']['process_host_bytes']['origin'], 'user')
            self.assertEqual(policy['terms']['process_host_bytes']['classification'], 'expert-override')
            self.assertEqual(policy['terms']['loader_scratch_host_bytes']['classification'], 'expert-override')
            self.assertEqual(policy['terms']['process_device_bytes']['classification'], 'conservative-policy')
            self.assertEqual(policy['terms']['process_host_bytes']['bytes'], 123)
            self.assertEqual(policy['terms']['process_device_bytes']['origin'], 'default')
            self.assertEqual(policy['terms']['loader_scratch_host_bytes']['origin'], 'user')
            self.assertEqual(policy['terms']['loader_scratch_host_bytes']['bytes'], 0)
            observations = list((workspace / 'planning').glob('resources-*.json'))
            self.assertTrue(observations)
            for path in observations:
                observation = json.loads(path.read_text())
                # Literal FREE values already exclude live usage. A large cap
                # cannot raise them; no process/unknown subtraction or margin.
                self.assertEqual(observation['effective_capacity_bytes'],
                                 dict(host=12 * 1024**3, device=8 * 1024**3))
                self.assertEqual(observation['capacity_classification'], 'measured-free-limited-by-user-cap')
            dense = [a for c, a in map(json.loads, calls.read_text().splitlines())
                     if c == 'experimental_fixed_cohort'][0]
            self.assertEqual(dense['--process_host_bytes'], '123')
            self.assertEqual(dense['--unknown_device_bytes'], str(1024**3))
            self.assertEqual(dense['--host_capacity_bytes'], str(12 * 1024**3))
            self.assertEqual(dense['--device_capacity_bytes'], str(100 * 1024**3))

    @unittest.skipUnless(sys.platform == 'linux' and os.geteuid() != 0,
                         'Experimental local execution requires non-root Linux')
    def test_wrong_shaped_native_json_refuses_without_output_or_traceback(self) -> None:
        import shutil
        root = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory(dir=root.parent) as directory:
            temp = Path(directory)
            (temp / 'photos').mkdir()
            native = temp / 'colmap'
            shutil.copy(Path(__file__).with_name('fake_native.py'), native)
            native.chmod(0o755)
            for index, identity in enumerate(([], None,
                    dict(schema_version=1, cuda_compiled=True, allocation_fingerprint=[]),
                    dict(schema_version=1, cuda_compiled=True, allocation_fingerprint=None))):
                with self.subTest(identity=identity):
                    workspace = temp / ('output-%d' % index)
                    env = dict(os.environ, PYTHONPATH=str(root / 'python'),
                               FAKE_IDENTITY_JSON=json.dumps(identity))
                    result = subprocess.run([sys.executable, '-m', 'colmap_reconstruction.cli',
                        '--colmap', str(native), '--experimental-local', '--yes', '--photos', str(temp / 'photos'),
                        '--workspace', str(workspace), *allowances()],
                        cwd=temp, env=env, capture_output=True, text=True)
                    self.assertEqual(result.returncode, 2, result.stderr)
                    self.assertFalse(workspace.exists())
                    self.assertIn('Reconstruction refused/failed:', result.stderr)
                    self.assertNotIn('Traceback', result.stderr)
                    self.assertFalse((workspace / 'result.json').exists())

    @unittest.skipUnless(sys.platform == 'linux' and os.geteuid() != 0,
                         'Experimental local execution requires non-root Linux')
    def test_automatic_policy_refusals_do_not_launch_heavy_stages(self):
        import shutil
        root = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory(dir=root.parent) as directory:
            temp = Path(directory)
            (temp / 'photos').mkdir()
            native = temp / 'colmap'
            shutil.copy(Path(__file__).with_name('fake_native.py'), native)
            native.chmod(0o755)
            stale = json.dumps(dict(schema_version=1, cuda_compiled=True,
                                   source_commit='a' * 40, allocation_fingerprint='c' * 64))
            cases = [
                (['--unknown-host-bytes', '123'], {}, 'require --unknown-provenance'),
                (['--unknown-provenance', '   '], {}, 'must not be blank'),
                (['--process-host-bytes', '-1'], {}, 'must be nonnegative'),
                (['--reserve-device-bytes', 'not-an-int'], {}, 'invalid'),
                (['--process-device-bytes', str(2**64)], {}, 'uint64'),
                (['--host-budget-bytes', str(2**64)], {}, 'uint64'),
                ([], {'FAKE_IDENTITY_JSON': stale}, 'no supported allocation binding'),
                (['--process-host-bytes', '123'], {'FAKE_IDENTITY_JSON': stale}, 'no supported allocation binding'),
                (['--host-budget-bytes', str(512 * 1024**2)], {}, 'host reserve unavailable'),
                ([], {'FAKE_BAD_FREE': '1'}, 'device reserve unavailable'),
            ]
            for index, (overrides, extra_env, reason) in enumerate(cases):
                with self.subTest(reason=reason, overrides=overrides):
                    workspace = temp / ('refused-%d' % index)
                    calls = temp / ('calls-%d.jsonl' % index)
                    env = dict(os.environ, PYTHONPATH=str(root / 'python'), FAKE_CALLS=str(calls), **extra_env)
                    result = subprocess.run([sys.executable, '-m', 'colmap_reconstruction.cli',
                        '--colmap', str(native), '--experimental-local', '--yes', '--photos', str(temp / 'photos'),
                        '--workspace', str(workspace), *overrides],
                        cwd=temp, env=env, capture_output=True, text=True)
                    self.assertEqual(result.returncode, 2, result.stderr)
                    self.assertIn(reason, result.stderr)
                    self.assertNotIn('Traceback', result.stderr)
                    self.assertFalse((workspace / 'result.json').exists())
                    if calls.exists():
                        self.assertTrue(all(c == 'reconstruction_info'
                            for c, a in map(json.loads, calls.read_text().splitlines())))
                    if workspace.exists():
                        self.assertEqual(json.loads((workspace / 'failure.json').read_text())['status'], 'failed')
                        if 'reserve unavailable' in reason:
                            observations = list((workspace / 'planning').glob('resources-*.json'))
                            self.assertEqual(len(observations), 1)
                            observation = json.loads(observations[0].read_text())
                            self.assertIn('effective_capacity_bytes', observation)

    @unittest.skipUnless(sys.platform == 'linux' and os.geteuid() != 0,
                         'Experimental local execution requires non-root Linux')
    def test_expert_binding_compatibility_partial_unknown_and_native_no_fit(self):
        import shutil
        root = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory(dir=root.parent) as directory:
            temp = Path(directory)
            (temp / 'photos').mkdir()
            native = temp / 'colmap'
            shutil.copy(Path(__file__).with_name('fake_native.py'), native)
            native.chmod(0o755)
            cases = [
                ('expert', allowances(), {'FAKE_ALLOCATION_FP': 'c' * 64}),
                ('partial', ['--unknown-host-bytes', '123', '--unknown-provenance', 'my basis'], {}),
                ('no-fit', [], {'FAKE_NOFIT': '1'}),
            ]
            for name, overrides, extra_env in cases:
                with self.subTest(name=name):
                    workspace = temp / name
                    calls = temp / (name + '.jsonl')
                    env = dict(os.environ, PYTHONPATH=str(root / 'python'), FAKE_CALLS=str(calls), **extra_env)
                    result = subprocess.run([sys.executable, '-m', 'colmap_reconstruction.cli',
                        '--colmap', str(native), '--experimental-local', '--yes', '--photos', str(temp / 'photos'),
                        '--workspace', str(workspace), *overrides],
                        cwd=temp, env=env, capture_output=True, text=True)
                    if name == 'no-fit':
                        self.assertEqual(result.returncode, 2, result.stderr)
                        self.assertFalse((workspace / 'result.json').exists())
                        self.assertFalse((workspace / 'points.ply').exists())
                        self.assertFalse((workspace / 'cohort').exists())
                        self.assertNotIn('stereo_fusion', [c for c, a in map(json.loads, calls.read_text().splitlines())])
                        self.assertIn('no feasible singleton',
                                      ''.join(p.read_text() for p in workspace.glob('*.stderr')))
                    else:
                        self.assertEqual(result.returncode, 0, result.stderr)
                        policy = json.loads((workspace / 'result.json').read_text())['resource_policy']
                        if name == 'expert':
                            self.assertEqual(policy['selection'], 'expert')
                            self.assertEqual(policy['allocation_fingerprint'], 'c' * 64)
                            self.assertTrue(all(t['origin'] == 'user' for t in policy['terms'].values()))
                            self.assertTrue(all(t['classification'] == 'expert-override'
                                                for t in policy['terms'].values()))
                        else:
                            self.assertEqual(policy['terms']['unknown_host_bytes']['provenance'], 'my basis')
                            self.assertEqual(policy['terms']['unknown_host_bytes']['bytes'], 123)
                            self.assertEqual(policy['terms']['unknown_device_bytes']['origin'], 'default')
                            self.assertEqual(policy['terms']['unknown_device_bytes']['bytes'], 1024**3)

    def test_automatic_policy_binding_matches_current_canonical_native_closure(self):
        import runpy
        from colmap_reconstruction.resource_policy import ALLOCATION_FINGERPRINT
        root = Path(__file__).resolve().parents[2]
        generator = runpy.run_path(str(root / 'scripts/resident/generate_resident_allocation_fingerprint.py'))
        self.assertEqual(generator['compute_fingerprint'](root), ALLOCATION_FINGERPRINT)

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

    @unittest.skipUnless(sys.platform == 'linux' and hasattr(os, 'pidfd_open'),
                         'Linux pidfds required for PID-reuse-safe descendant checks')
    def test_owned_process_drains_term_ignoring_descendant_after_leader_exit(self) -> None:
        from colmap_reconstruction.process import NativeProcess
        import select
        import signal
        import socket
        import threading

        # The descendant acknowledges readiness only after ignoring TERM. Its
        # leader cannot exit until we hold a pidfd, so cleanup never targets a
        # recycled PID. A readable pidfd means exited, including a legitimate
        # orphan zombie on Linux hosts whose PID 1 does not promptly reap.
        script = '''
import os, signal, socket, sys
read_fd, write_fd = os.pipe()
if os.fork():
    os.close(write_fd)
    assert os.read(read_fd, 1) == b'!'
    os._exit(0)
os.close(read_fd)
signal.signal(signal.SIGTERM, signal.SIG_IGN)
signal.alarm(20)
with socket.socket(socket.AF_UNIX) as ready:
    ready.settimeout(5)
    ready.connect(sys.argv[1])
    ready.sendall(('%d %d\\n' % (os.getpid(), os.getppid())).encode())
    assert ready.recv(1) == b'!'
os.write(write_fd, b'!')
os.close(write_fd)
while True:
    signal.pause()
'''
        with tempfile.TemporaryDirectory() as directory:
            process = NativeProcess(Path(directory), timeout_seconds=5)
            failures = []

            def run():
                try:
                    process.run([sys.executable, '-c', script, str(ready_path)])
                except BaseException as error:
                    failures.append(error)

            ready_path = Path(directory) / 'ready.sock'
            pidfd = None
            worker = threading.Thread(target=run)
            with socket.socket(socket.AF_UNIX) as listener:
                listener.settimeout(5)
                listener.bind(str(ready_path))
                listener.listen(1)
                worker.start()
                try:
                    connection, _ = listener.accept()
                    with connection:
                        connection.settimeout(5)
                        with connection.makefile('rb') as message:
                            descendant, leader = map(int, message.readline(128).split())
                        pidfd = os.pidfd_open(descendant)
                        self.assertEqual(os.getpgid(descendant), leader)
                        self.assertEqual(os.getpgid(leader), leader)
                        self.assertEqual(select.select([pidfd], [], [], 0)[0], [])
                        connection.sendall(b'!')
                    worker.join(timeout=8)
                    self.assertFalse(worker.is_alive(), 'native run did not finish')
                    self.assertEqual(failures, [])
                    self.assertIsNone(process.child)
                    self.assertEqual(select.select([pidfd], [], [], 3)[0], [pidfd],
                                     'TERM-ignoring descendant survived leader exit')
                finally:
                    if pidfd is not None:
                        try:
                            signal.pidfd_send_signal(pidfd, signal.SIGKILL)
                            self.assertEqual(select.select([pidfd], [], [], 3)[0], [pidfd])
                        except ProcessLookupError:
                            pass
                        finally:
                            os.close(pidfd)
                    worker.join(timeout=8)
                    self.assertFalse(worker.is_alive(), 'native worker leaked')

    def test_cli_refuses_hard_caps_and_existing_workspace_before_native_launch(self) -> None:
        from colmap_reconstruction.cli import main
        with tempfile.TemporaryDirectory() as directory:
            photos = Path(directory) / 'photos'
            photos.mkdir()
            args = ['--experimental-local', '--yes', '--photos', str(photos),
                    '--workspace', str(Path(directory) / 'new'),
                    '--host-budget-bytes', '1000', '--device-budget-bytes', '1000'] + allowances()
            self.assertEqual(main(args + ['--require-hard-memory-cap']), 2)
            self.assertFalse((Path(directory) / 'new').exists())
            self.assertEqual(main(args + ['--workspace', str(photos)]), 2)
            self.assertEqual(main(args + ['--colmap', sys.executable,
                '--unknown-host-bytes', '0', '--unknown-device-bytes', '0']), 2)
            self.assertFalse((Path(directory) / 'new').exists())

    def prepare_sparse_models(self, models, selection):
        """Exercise preparation with real synthetic native subprocesses, no GPU."""
        from unittest.mock import patch
        from colmap_reconstruction.cli import Reconstruction, parse_args
        import shutil
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[2].parent) as directory:
            temp = Path(directory)
            (temp / 'photos').mkdir()
            native = temp / 'colmap'
            shutil.copy(Path(__file__).with_name('fake_native.py'), native)
            native.chmod(0o755)
            workspace = temp / 'output'
            workspace.mkdir()
            args = ['--photos', str(temp / 'photos'), '--workspace', str(workspace),
                    '--host-budget-bytes', str(2**30), '--device-budget-bytes', str(2**30)] + allowances()
            if selection is not None:
                args += ['--sparse-model', selection]
            calls = temp / 'calls.jsonl'
            with patch.dict(os.environ, FAKE_SPARSE_MODELS=json.dumps(models), FAKE_CALLS=str(calls)):
                reconstruction = Reconstruction(parse_args(args), str(native))
                reconstruction.assets.mkdir()
                reconstruction.identity = {'source_commit': 'a' * 40}
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
