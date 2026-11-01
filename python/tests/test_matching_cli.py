# SPDX-License-Identifier: BSD-3-Clause
"""Real CPU matching CLI/DB/main seams; no extraction or GPU fallback."""
import math
import os
from pathlib import Path
import random
import re
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
class MatchingCliTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.executable = os.environ['COLMAP_TEST_EXECUTABLE']

    def fixture(self, name, images=4, rows=48):
        path = self.root / name
        created = subprocess.run([self.executable, 'database_creator', '--database_path', str(path)],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=30)
        self.assertEqual(created.returncode, 0, created.stdout)
        rng = random.Random(42)
        points = [(rng.uniform(-1, 1), rng.uniform(-1, 1), rng.uniform(3, 8)) for _ in range(rows)]
        descriptors = bytes(rng.randrange(256) for _ in range(rows * 128))
        with sqlite3.connect(path) as db:
            db.execute('INSERT INTO cameras(camera_id,model,width,height,params,prior_focal_length) '
                       'VALUES(1,0,1000,1000,?,1)', (struct.pack('<3d', 500, 500, 500),))
            for image in range(1, images + 1):
                db.execute('INSERT INTO images(image_id,name,camera_id) VALUES(?,?,1)', (image, f'{image}.jpg'))
                keypoints = b''.join(struct.pack('<6f', 500 + 500 * (x - image * .1) / z,
                    500 + 500 * y / z, 1, 0, 0, 1) for x, y, z in points)
                db.execute('INSERT INTO keypoints(image_id,rows,cols,data) VALUES(?,?,6,?)',
                           (image, rows, keypoints))
                db.execute('INSERT INTO descriptors(image_id,type,rows,cols,data) VALUES(?,0,?,128,?)',
                           (image, rows, descriptors))
        return path

    def command(self, database, budget):
        return [self.executable, 'exhaustive_matcher', '--database_path', str(database),
                '--FeatureMatching.use_gpu', 'false', '--FeatureMatching.num_threads', '1',
                '--FeatureMatching.host_budget_bytes', str(budget)]

    def run_command(self, command):
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=90)
        return result.returncode, result.stdout.decode(errors='replace')

    def snapshot(self, path):
        with sqlite3.connect(path) as db:
            # ALL fields, inlier ordering and optional matrices/poses/cameras are
            # retained, not just counts. Historical geometry is NOT an oracle
            # without repeatability/effective-input binding; raw matching is.
            return (db.execute('SELECT * FROM matches ORDER BY pair_id').fetchall(),
                    db.execute('SELECT * FROM two_view_geometries ORDER BY pair_id').fetchall())

    def test_real_pipeline_raw_matches_backpressure_and_verifier_outputs(self):
        control, generous, tight = (self.fixture(name) for name in ('control.db', 'generous.db', 'tight.db'))
        for path, budget in ((control, 0), (generous, 64 * 1024**2)):
            rc, output = self.run_command(self.command(path, budget))
            self.assertEqual(rc, 0, output)
        persistent = int(re.search(r'persistent_known_bytes=(\d+)', output).group(1))
        pair = max(map(int, re.findall(r'known_payload_bytes=(\d+)', output)))
        capacity = (103 * (persistent + pair) + 99) // 100
        rc, output = self.run_command(self.command(tight, capacity))
        self.assertEqual(rc, 0, output)
        self.assertIn('Feature matching admission backpressure:', output)
        self.assertIn('live_host_bytes=0 live_device_bytes=0 success=1', output)
        expected = self.snapshot(control)[0]
        self.assertEqual(len(expected), 6)
        ordered = b''.join(struct.pack('<2I', i, i) for i in range(48))
        for _, rows, cols, blob in expected:
            self.assertEqual((rows, cols, blob), (48, 2, ordered))
        for path in (generous, tight):
            raw, geometry = self.snapshot(path)
            self.assertEqual(raw, expected)
            self.assertEqual(len(geometry), 6)
            for row in geometry:
                self.assertGreater(row[1], 0)
                self.assertEqual(row[2], 2)
                inliers = list(struct.iter_unpack('<2I', row[3]))
                self.assertEqual(len(inliers), row[1])
                self.assertEqual(inliers, [(i, i) for i, _ in inliers])
                self.assertEqual([i for i, _ in inliers], sorted(i for i, _ in inliers))
                # Every available numeric matrix/pose is finite; preserve NULLs.
                for blob in row[5:10]:
                    if blob is not None:
                        self.assertTrue(all(math.isfinite(value[0]) for value in struct.iter_unpack('<d', blob)))

    def test_malformed_full_width_stored_match_metadata_refuses_and_drains(self):
        # Even a regression must not be allowed to request the original 32 GiB
        # matrix. Runtime executes this CPU-only seam with a bounded address
        # space; the expected error must be metadata refusal, not bad_alloc.
        import resource

        def bounded_address_space():
            cap = 8 * 1024**3
            resource.setrlimit(resource.RLIMIT_AS, (cap, cap))

        cases = ((4294967297, 2, b'\0' * 8, 'unsupported database matrix dimensions'),
                 (9223372036854775807, 2, b'', 'unsupported database matrix dimensions'),
                 (1, 3, b'\0' * 12, 'shape.cols'),
                 (-1, 2, b'', 'unsupported database matrix dimensions'),
                 (1, -2, b'', 'unsupported database matrix dimensions'),
                 (2, 2, b'\0' * 8, 'shape/BLOB length mismatch'),
                 (1.5, 2, b'\0' * 8, 'sqlite3_column_type'),
                 (1, 2, '12345678', 'sqlite3_column_int'))
        for index, (rows, cols, blob, error) in enumerate(cases):
            with self.subTest(rows=rows, cols=cols, blob_type=type(blob).__name__):
                path = self.fixture(f'malformed-matches-{index}.db')
                with sqlite3.connect(path) as db:
                    db.execute('INSERT INTO matches(pair_id,rows,cols,data) VALUES(?,?,?,?)',
                               (2147483647 + 2, rows, cols, blob))
                result = subprocess.run(self.command(path, 64 * 1024**2),
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=90,
                    preexec_fn=bounded_address_space)
                output = result.stdout.decode(errors='replace')
                self.assertEqual(result.returncode, 1, output)
                self.assertIn(error, output)
                self.assertIn('workers drained: live_host_bytes=0 live_device_bytes=0 success=0', output)
                with sqlite3.connect(path) as db:
                    self.assertEqual(db.execute('SELECT rows,cols,data FROM matches').fetchall(),
                                     [(rows, cols, blob)])
                    self.assertEqual(db.execute('SELECT count(*) FROM two_view_geometries').fetchone()[0], 0)

    def test_singleton_unsupported_and_strict_cli_status(self):
        for label, extra, phrase in (
                ('singleton', [], 'singleton refused'),
                ('guided', ['--FeatureMatching.guided_matching', 'true'], 'non-guided/non-rig'),
                ('rig', ['--FeatureMatching.rig_verification', 'true'], 'non-guided/non-rig'),
                ('device-on-cpu', ['--FeatureMatching.device_budget_bytes', '1000000'], 'CPU backend')):
            path = self.fixture(label + '.db')
            rc, output = self.run_command(self.command(path, 1 if label == 'singleton' else 64 * 1024**2) + extra)
            self.assertEqual(rc, 1, output)
            self.assertIn(phrase, output)
            self.assertEqual(self.snapshot(path), ([], []))
        for name in ('host_budget_bytes', 'policy_host_bytes', 'reserve_host_bytes',
                     'device_budget_bytes', 'policy_device_bytes', 'reserve_device_bytes'):
            for value in ('-1', '+1', '01', '18446744073709551616', '1e3', '1.5', '1x', ' 1', '1 ', ''):
                with self.subTest(name=name, value=value):
                    command = self.command(self.root / 'must-not-exist.db', 0)
                    if name == 'host_budget_bytes':
                        command[-1] = value
                    else:
                        command += ['--FeatureMatching.' + name, value]
                    rc, output = self.run_command(command)
                    self.assertEqual(rc, 1, output)
                    self.assertIn('requires a canonical byte allowance', output)
                    self.assertFalse((self.root / 'must-not-exist.db').exists())

    def test_real_main_sigint_drains(self):
        path = self.fixture('cancel.db', images=35, rows=200)
        process = subprocess.Popen(self.command(path, 64 * 1024**2),
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=0)
        captured = bytearray()
        try:
            with selectors.DefaultSelector() as selector:
                selector.register(process.stdout, selectors.EVENT_READ)
                deadline = time.monotonic() + 60
                while b'Feature matching admitted pair:' not in captured:
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
            self.assertIn(b'workers drained: live_host_bytes=0 live_device_bytes=0', captured)
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
            process.stdout.close()


if __name__ == '__main__':
    unittest.main()
