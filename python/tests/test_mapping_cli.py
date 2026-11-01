# SPDX-License-Identifier: BSD-3-Clause
"""Actual mapper MAIN failure/signal seams; no extraction/matching or GPU."""
import os
from pathlib import Path
import selectors
import signal
import sqlite3
import struct
import subprocess
import tempfile
import time
import unittest


@unittest.skipUnless(os.environ.get('COLMAP_TEST_EXECUTABLE'),
                     'set COLMAP_TEST_EXECUTABLE to built/installed colmap')
class MappingCliTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.executable = os.environ['COLMAP_TEST_EXECUTABLE']

    def fixture(self, count=2):
        path = self.root / 'database.db'
        result = subprocess.run([self.executable, 'database_creator', '--database_path', str(path)],
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout)
        with sqlite3.connect(path) as db:
            db.execute('INSERT INTO cameras(camera_id,model,width,height,params,prior_focal_length) '
                       'VALUES(1,0,100,100,?,1)', (struct.pack('<3d', 50, 50, 50),))
            db.executemany('INSERT INTO images(image_id,name,camera_id) VALUES(?,?,1)',
                           ((i, f'{i}.jpg') for i in range(1, count + 1)))
            db.executemany('INSERT INTO keypoints(image_id,rows,cols,data) VALUES(?,1,6,?)',
                           ((i, struct.pack('<6f', 50, 50, 1, 0, 0, 1)) for i in range(1, count + 1)))
        return path

    def command(self, path, budget=64 * 1024**2):
        output = self.root / 'sparse'
        output.mkdir(exist_ok=True)
        return [self.executable, 'mapper', '--database_path', str(path),
                '--image_path', str(self.root), '--output_path', str(output),
                '--Mapper.host_budget_bytes', str(budget)]

    def run_command(self, command):
        import resource

        def bound_address_space():
            cap = 8 * 1024**3
            resource.setrlimit(resource.RLIMIT_AS, (cap, cap))

        return subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              timeout=60, preexec_fn=bound_address_space)

    def test_strict_cli_and_unsupported_modes_return_one(self):
        for name in ('host_budget_bytes', 'policy_host_bytes', 'reserve_host_bytes'):
            for value in ('-1', '+1', '01', '18446744073709551616', '1e3', '1.5', '1x', ' 1', '1 ', ''):
                with self.subTest(name=name, value=value):
                    missing = self.root / 'must-not-exist.db'
                    command = self.command(missing)
                    if name == 'host_budget_bytes':
                        command[-1] = value
                    else:
                        command += ['--Mapper.' + name, value]
                    result = self.run_command(command)
                    self.assertEqual(result.returncode, 1, result.stdout)
                    self.assertIn(b'requires a canonical byte allowance', result.stdout)
                    self.assertFalse(missing.exists())
        path = self.fixture()
        for extra in (['--Mapper.ba_use_gpu', 'true'], ['--Mapper.ba_local_backend', 'CASPAR'],
                      ['--input_path', str(self.root)]):
            result = self.run_command(self.command(path) + extra)
            self.assertEqual(result.returncode, 1, result.stdout)
            self.assertEqual(list((self.root / 'sparse').iterdir()), [])

    def test_loader_refusal_and_corrupt_shapes_never_publish(self):
        path = self.fixture()
        result = self.run_command(self.command(path, 1))
        self.assertEqual(result.returncode, 1, result.stdout)
        self.assertIn(b'mapping logical host', result.stdout)
        for rows, cols, data in ((4294967297, 2, b'\0' * 8), (1.5, 2, b'\0' * 8),
                                 (1, 3, b'\0' * 12), (2, 2, b'\0' * 8),
                                 (1, 2, '12345678')):
            with self.subTest(rows=rows, cols=cols):
                with sqlite3.connect(path) as db:
                    db.execute('DELETE FROM two_view_geometries')
                    db.execute('INSERT INTO two_view_geometries(pair_id,rows,cols,data,config) VALUES(2147483649,?,?,?,2)',
                               (rows, cols, data))
                result = self.run_command(self.command(path))
                self.assertEqual(result.returncode, 1, result.stdout)
                self.assertIn(b'owned payloads retired, no success flush', result.stdout)
                self.assertNotIn(b'bad_alloc', result.stdout)
                self.assertEqual(list((self.root / 'sparse').iterdir()), [])
                with sqlite3.connect(path) as db:
                    self.assertEqual(db.execute('SELECT rows,cols,data FROM two_view_geometries').fetchall(),
                                     [(rows, cols, data)])
        # Tiny serialized camera header declares an impossible parameter count;
        # rejection must precede resize, not rely on bad_alloc/host OOM.
        camera = struct.pack('<IiQQBQ', 1, 0, 100, 100, 1, 2**63)
        with sqlite3.connect(path) as db:
            db.execute('DELETE FROM two_view_geometries')
            db.execute('INSERT INTO two_view_geometries(pair_id,rows,cols,data,config,camera1) '
                       'VALUES(2147483649,1,2,?,2,?)', (b'\0' * 8, camera))
        result = self.run_command(self.command(path))
        self.assertEqual(result.returncode, 1, result.stdout)
        self.assertIn(b'owned payloads retired, no success flush', result.stdout)
        self.assertNotIn(b'bad_alloc', result.stdout)
        self.assertEqual(list((self.root / 'sparse').iterdir()), [])

    def test_main_sole_signal_owner_returns_130_during_loader(self):
        path = self.fixture(count=10000)
        process = subprocess.Popen(self.command(path), stdin=subprocess.DEVNULL,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=0)
        captured = bytearray()
        try:
            with selectors.DefaultSelector() as selector:
                selector.register(process.stdout, selectors.EVENT_READ)
                deadline = time.monotonic() + 60
                while b'Mapping initial DB/model and graph logical admission:' not in captured:
                    remaining = deadline - time.monotonic()
                    self.assertGreater(remaining, 0, bytes(captured))
                    self.assertTrue(selector.select(remaining), bytes(captured))
                    block = os.read(process.stdout.fileno(), 65536)
                    self.assertTrue(block, bytes(captured))
                    captured.extend(block)
                process.send_signal(signal.SIGINT)
                tail, _ = process.communicate(timeout=60)
                captured.extend(tail)
            self.assertEqual(process.returncode, 130, bytes(captured))
            self.assertIn(b'Graceful shutdown completed', captured)
            self.assertEqual(list((self.root / 'sparse').iterdir()), [])
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
            process.stdout.close()
