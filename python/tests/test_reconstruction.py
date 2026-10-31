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
                    '--experimental-local', '--photos', str(temp), '--workspace', str(temp / 'root-refused'),
                    *allowances()], cwd=temp, env=env, capture_output=True)
                self.assertNotEqual(refused.returncode, 0)
                self.assertFalse((temp / 'root-refused').exists())
                return  # Ordinary-user full-chain seam is exercised on non-root hosts.
            import shutil
            native = temp / 'prefix/bin/colmap'
            shutil.copy(Path(__file__).with_name('fake_native.py'), native)
            native.chmod(0o755)
            calls = temp / 'calls.jsonl'
            for scenario in ('good', 'depth', 'source-order', 'fingerprint', 'fusion', 'resources'):
                workspace = temp / scenario
                scenario_env = dict(env, FAKE_CALLS=str(calls), PYTHONPATH='/absent/lab:/absent/photogram')
                if scenario != 'good':
                    scenario_env['FAKE_BAD_' + {'depth': 'DEPTH', 'source-order': 'ORDER', 'fingerprint': 'FP', 'fusion': 'FUSION', 'resources': 'FREE'}[scenario]] = '1'
                result = subprocess.run([str(temp / 'prefix/bin/colmap-reconstruct'),
                    '--experimental-local', '--photos', str(temp), '--workspace', str(workspace),
                    *allowances()], cwd=temp, env=scenario_env, capture_output=True, text=True)
                if scenario == 'good':
                    self.assertEqual(result.returncode, 0, result.stderr)
                    report = json.loads((workspace / 'result.json').read_text())
                    self.assertEqual(report['backend'], 'experimental_fixed_cohort')
                    commands = [command for command, arguments in map(json.loads, calls.read_text().splitlines())
                                if command != 'reconstruction_info' or not arguments]
                    self.assertEqual(commands, ['reconstruction_info', 'feature_extractor',
                        'exhaustive_matcher', 'mapper', 'model_analyzer', 'image_undistorter',
                        'optimized_dense_plan', 'experimental_fixed_cohort', 'stereo_fusion'])
                else:
                    self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
                    self.assertFalse((workspace / 'result.json').exists())
                before = calls.read_bytes()
                again = subprocess.run([str(temp / 'prefix/bin/colmap-reconstruct'),
                    '--experimental-local', '--photos', str(temp), '--workspace', str(workspace),
                    *allowances()], cwd=temp, env=scenario_env, capture_output=True)
                self.assertNotEqual(again.returncode, 0)
                self.assertEqual(calls.read_bytes(), before)

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
            args = ['--experimental-local', '--photos', str(photos),
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
            native = temp / 'colmap'
            shutil.copy(Path(__file__).with_name('fake_native.py'), native)
            native.chmod(0o755)
            workspace = temp / 'output'
            workspace.mkdir()
            args = ['--photos', str(temp), '--workspace', str(workspace),
                    '--host-budget-bytes', str(2**30), '--device-budget-bytes', str(2**30)] + allowances()
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
