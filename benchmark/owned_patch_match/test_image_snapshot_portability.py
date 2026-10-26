# SPDX-License-Identifier: BSD-3-Clause
"""Compile the image snapshot timestamp expression with a wide file-clock rep.

This models libc++'s __int128 file duration on a CPU host, not a native macOS run.
"""

import pathlib
import subprocess
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[2]


class ImageSnapshotPortability(unittest.TestCase):
    def test_wide_file_clock_preserves_nanoseconds(self):
        source = (ROOT / "src/colmap/exe/image_test.cc").read_text()
        start = source.index("    content +=")
        end = source.index("    result.emplace", start)
        expression = source[start:end]
        cpp = r"""
#include <cassert>
#include <chrono>
#include <string>
struct Entry {
  using Duration = std::chrono::duration<__int128_t, std::nano>;
  using Time = std::chrono::time_point<std::chrono::system_clock, Duration>;
  Time value;
  Time last_write_time() const { return value; }
};
int main() {
  // Contemporary Unix and libstdc++ file-clock epochs, including 1ns changes.
  for (long long ticks : {1761566400123456789LL, 1761566400123456790LL,
                         -4676097599876543211LL, -4676097599876543210LL}) {
    Entry entry{Entry::Time{Entry::Duration{ticks}}};
    std::string content;
""" + expression + r"""
    assert(content == "\nmtime=" + std::to_string(ticks));
  }
}
"""
        with tempfile.TemporaryDirectory(prefix="image-snapshot-") as tmp:
            root = pathlib.Path(tmp)
            probe = root / "probe.cc"
            probe.write_text(cpp)
            binary = root / "probe"
            compiled = subprocess.run(
                ["g++", "-std=c++17", "-Wall", "-Wextra", "-Werror",
                 str(probe), "-o", str(binary)],
                capture_output=True, text=True, timeout=30,
            )
            self.assertEqual(compiled.returncode, 0, compiled.stderr)
            result = subprocess.run(
                [str(binary)], capture_output=True, text=True, timeout=10,
            )
            self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
