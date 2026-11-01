# SPDX-License-Identifier: BSD-3-Clause
"""Real CPU CLI/main signal ownership, not GPU execution qualification."""
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
import zlib


def png(width, height, rgb=False):
    def chunk(kind, data):
        return (struct.pack('>I', len(data)) + kind + data
                + struct.pack('>I', zlib.crc32(kind + data)))
    rows = []
    for y in range(height):
        row = bytes(255 if width // 3 <= x < 2 * width // 3
                    and height // 3 <= y < 2 * height // 3 else 0
                    for x in range(width))
        if rgb:
            row = bytes(value for value in row for _ in range(3))
        rows.append(b'\0' + row)
    return (b'\x89PNG\r\n\x1a\n'
            + chunk(b'IHDR', struct.pack('>IIBBBBB', width, height, 8,
                                        2 if rgb else 0, 0, 0, 0))
            + chunk(b'IDAT', zlib.compress(b''.join(rows))) + chunk(b'IEND', b''))


@unittest.skipUnless(os.environ.get('COLMAP_TEST_EXECUTABLE'),
                     'set COLMAP_TEST_EXECUTABLE to built/installed colmap')
class ExtractionCliTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.photos = self.root / 'photos'
        self.photos.mkdir()
        self.executable = os.environ['COLMAP_TEST_EXECUTABLE']

    def command(self, database, budget):
        return [self.executable, 'feature_extractor', '--database_path', str(database),
                '--image_path', str(self.photos), '--FeatureExtraction.use_gpu', 'false',
                '--FeatureExtraction.num_threads', '2',
                '--FeatureExtraction.max_image_size', '3200',
                '--FeatureExtraction.host_budget_bytes', str(budget)]

    def blobs(self, database):
        with sqlite3.connect(database) as db:
            return db.execute('SELECT images.name, keypoints.rows, keypoints.cols, '
                'keypoints.data, descriptors.rows, descriptors.cols, descriptors.data '
                'FROM images JOIN keypoints USING(image_id) '
                'JOIN descriptors USING(image_id) ORDER BY images.name').fetchall()

    def test_real_main_success_matches_zero_budget_and_singleton_refuses(self):
        for name, w, h, rgb in (('a.png', 128, 128, False),
                                ('b.png', 93, 67, True), ('c.png', 1, 1, False)):
            (self.photos / name).write_bytes(png(w, h, rgb))
        control = self.root / 'control.db'
        candidate = self.root / 'candidate.db'
        for database, budget in ((control, 0), (candidate, 32 * 1024**2)):
            result = subprocess.run(self.command(database, budget),
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=60)
            self.assertEqual(result.returncode, 0, result.stdout.decode(errors='replace'))
        expected = self.blobs(control)
        self.assertEqual(len(expected), 3)
        self.assertGreater(sum(row[1] for row in expected), 0)
        self.assertEqual(self.blobs(candidate), expected)
        refused = self.root / 'refused.db'
        result = subprocess.run(self.command(refused, 1),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=60)
        self.assertEqual(result.returncode, 1, result.stdout.decode(errors='replace'))
        self.assertIn(b'before decode', result.stdout)
        self.assertIn(b'workers drained', result.stdout)
        self.assertEqual(self.blobs(refused), [])

    def test_real_main_budget_constructor_refuses_without_abort(self):
        (self.photos / 'a.png').write_bytes(png(128, 128))
        command = self.command(self.root / 'constructor.db', 1)
        command += ['--FeatureExtraction.policy_host_bytes', '1']
        result = subprocess.run(command, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, timeout=60)
        self.assertEqual(result.returncode, 1, result.stdout.decode(errors='replace'))
        self.assertIn(b'singleton refused before decode', result.stdout)

    def test_real_main_sigint_drains_without_success(self):
        # Send SIGINT on the native predecode admission event, not a sleep.
        image = png(2048, 1536)
        for index in range(16):
            (self.photos / f'{index:02}.png').write_bytes(image)
        database = self.root / 'cancelled.db'
        process = subprocess.Popen(self.command(database, 256 * 1024**2),
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, bufsize=0)
        captured = bytearray()
        try:
            with selectors.DefaultSelector() as selector:
                selector.register(process.stdout, selectors.EVENT_READ)
                deadline = time.monotonic() + 60
                while b'Feature extraction admitted image before decode:' not in captured:
                    remaining = deadline - time.monotonic()
                    self.assertGreater(remaining, 0, bytes(captured))
                    self.assertTrue(selector.select(remaining), bytes(captured))
                    data = os.read(process.stdout.fileno(), 65536)
                    self.assertTrue(data, bytes(captured))
                    captured.extend(data)
            process.send_signal(signal.SIGINT)
            tail, _ = process.communicate(timeout=60)
            captured.extend(tail)
            self.assertEqual(process.returncode, 128 + signal.SIGINT,
                             captured.decode(errors='replace'))
            self.assertIn(b'workers drained', captured)
            self.assertIn(b'Graceful shutdown completed', captured)
            self.assertLess(len(self.blobs(database)), 16)
        finally:
            if process.poll() is None:
                process.kill()
                process.communicate()
            process.stdout.close()


if __name__ == '__main__':
    unittest.main()
