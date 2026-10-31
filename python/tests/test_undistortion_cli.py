# SPDX-License-Identifier: BSD-3-Clause
"""Real executable regression: main owns signals, the controller drains workers."""
import os
from pathlib import Path
import selectors
import signal
import struct
import subprocess
import tempfile
import time
import unittest
import zlib


@unittest.skipUnless(os.environ.get('COLMAP_TEST_EXECUTABLE'),
                     'set COLMAP_TEST_EXECUTABLE to the built colmap binary')
class UndistortionCliTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.executable = os.environ['COLMAP_TEST_EXECUTABLE']

    def fixture(self, width=16, height=12, count=2):
        photos = self.root / 'photos'
        model = self.root / 'model'
        photos.mkdir()
        model.mkdir()
        # Stdlib-only RGB PNG and legacy text model; no pycolmap dependency.
        def chunk(kind, payload):
            return (struct.pack('>I', len(payload)) + kind + payload
                    + struct.pack('>I', zlib.crc32(kind + payload)))

        png = (b'\x89PNG\r\n\x1a\n'
               + chunk(b'IHDR', struct.pack('>IIBBBBB', width, height, 8, 2, 0, 0, 0))
               + chunk(b'IDAT', zlib.compress(
                   (b'\0' + b'\x14\x28\x3c' * width) * height))
               + chunk(b'IEND', b''))
        (model / 'cameras.txt').write_text(
            f'1 SIMPLE_RADIAL {width} {height} {width} {width/2} {height/2} 0.05\n')
        rows = []
        for image_id in range(1, count + 1):
            name = f'{image_id}.png'
            (photos / name).write_bytes(png)
            rows.append(f'{image_id} 1 0 0 0 {image_id} 0 0 1 {name}\n\n')
        (model / 'images.txt').write_text(''.join(rows))
        (model / 'points3D.txt').write_text('')
        return [self.executable, 'image_undistorter', '--image_path', str(photos),
                '--input_path', str(model), '--num_threads', '1',
                '--max_image_size', '9']

    def assert_incomplete(self, output):
        for relative in ('stereo/patch-match.cfg', 'stereo/fusion.cfg',
                         'sparse/cameras.bin', 'sparse/cameras.txt'):
            self.assertFalse((output / relative).exists(), relative)

    def test_real_main_budgeted_success_and_singleton_refusal(self):
        command = self.fixture()
        output = self.root / 'success'
        result = subprocess.run(command + ['--output_path', str(output),
            '--host_budget_bytes', str(256 * 1024**2)],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout.decode(errors='replace'))
        for name in ('1.png', '2.png'):
            self.assertTrue((output / 'images' / name).is_file())
        self.assertTrue((output / 'stereo/fusion.cfg').is_file())
        self.assertTrue((output / 'sparse/cameras.bin').is_file())

        refused = self.root / 'refused'
        result = subprocess.run(command + ['--output_path', str(refused),
            '--host_budget_bytes', '1'], stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, timeout=60)
        self.assertEqual(result.returncode, 1, result.stdout.decode(errors='replace'))
        self.assertIn(b'before decode', result.stdout)
        # The CLI creates its output root before calling the controller.
        self.assertTrue(refused.is_dir())
        self.assertEqual(list(refused.iterdir()), [])
        self.assert_incomplete(refused)

    def test_real_main_sigint_drains_admitted_workers(self):
        # One sufficiently large warp per batch leaves work pending when the
        # first admission log arrives. No sleep/timing guess triggers SIGINT.
        command = self.fixture(width=2048, height=1536, count=8)
        output = self.root / 'cancelled'
        process = subprocess.Popen(command + ['--output_path', str(output),
            '--host_budget_bytes', str(256 * 1024**2)], stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=0)
        captured = bytearray()
        try:
            with selectors.DefaultSelector() as selector:
                selector.register(process.stdout, selectors.EVENT_READ)
                deadline = time.monotonic() + 60
                while b'Undistortion admitted workers:' not in captured:
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
            self.assert_incomplete(output)
            # Snapshot only after the process has closed: all owned tasks have
            # finished, and no later image or completion publication is possible.
            self.assertLess(len(list((output / 'images').glob('*.png'))), 8)
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=10)
            process.stdout.close()


if __name__ == '__main__':
    unittest.main()
