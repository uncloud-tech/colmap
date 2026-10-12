# SPDX-License-Identifier: BSD-3-Clause
"""CPU compile checks for the owned-only sweep-width configuration."""
import pathlib
import subprocess
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[2]
SOURCE = ROOT / "src/colmap/mvs/patch_match_cuda_owned.cu"


class OwnedSweepTiles(unittest.TestCase):
    def test_width_guards_and_shared_indices(self):
        text = SOURCE.read_text()
        begin = text.index("#ifndef COLMAP_OWNED_SWEEP_THREADS")
        end = text.index("constexpr int kOwnedElementwiseThreads = 32;")
        config = text[begin:end] + "constexpr int kOwnedElementwiseThreads = 32;\n"
        for width in (None, 8, 16, 32, 0, 4, 64):
            with self.subTest(width=width), tempfile.TemporaryDirectory() as tmp:
                expected = 32 if width is None else width
                source = pathlib.Path(tmp) / "test.cc"
                source.write_text(config + f"\nstatic_assert(THREADS_PER_BLOCK == {expected});\n"
                                  "static_assert(kOwnedElementwiseThreads == 32);\nint main() {}\n")
                command = ["c++", "-std=c++17", "-fsyntax-only", str(source)]
                if width is not None:
                    command.append(f"-DCOLMAP_OWNED_SWEEP_THREADS={width}")
                result = subprocess.run(command, capture_output=True, text=True)
                self.assertEqual(result.returncode == 0, expected in (8, 16, 32), result.stderr)
        # Window radius 5: every per-thread neighborhood stays in its own
        # shared tile, including boundary threads. Logical global coordinates
        # are independent of block partitioning.
        for width in (8, 16, 32):
            for col in range(100):
                block, thread = divmod(col, width)
                for offset in range(-5, 6):
                    local = width + thread + offset
                    self.assertTrue(0 <= local < 3 * width)
                    self.assertEqual(block * width - width + local, col + offset)
        self.assertIn("static_assert(kWindowRadius <= THREADS_PER_BLOCK", text)
        launch = text[text.index("  elem_wise_block_size_.x ="):text.index("  elem_wise_block_size_.x =") + 380]
        self.assertNotIn("THREADS_PER_BLOCK", launch)


if __name__ == "__main__":
    unittest.main()
