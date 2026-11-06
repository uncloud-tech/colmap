# SPDX-License-Identifier: BSD-3-Clause
"""CPU payload-admission tests. These do not qualify CUDA extraction."""
import pathlib
import subprocess
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[2]


class StreamedExtractionTest(unittest.TestCase):
    def test_checked_live_payload(self):
        code = r'''
#include "src/colmap/feature/sift_streaming.h"
#include <cassert>
#include <cstdint>
#include <stdexcept>
using namespace colmap;
template<class F> void Refuses(F f) {
  bool caught = false;
  try { f(); } catch (const std::exception&) { caught = true; }
  assert(caught);
}
int main() {
  // Bitmap + retained caller capacity + staging + metadata + result payload.
  assert(CheckSiftStreamingPayload(100, 200, 300, 400, 10, 152,
                                   2520, 0, 0) == 2520);
  Refuses([] { CheckSiftStreamingPayload(100, 200, 300, 400, 10, 152,
                                         2519, 0, 0); });
  assert(CheckSiftStreamingPayload(100, 200, 300, 400, 10, 152,
                                   2820, 100, 200) == 2520);
  Refuses([] { CheckSiftStreamingPayload(100, 200, 300, 400, 10, 152,
                                         2819, 100, 200); });
  Refuses([] { CheckSiftStreamingPayload(0, 0, 0, 0, UINT64_MAX, 2,
                                         UINT64_MAX, 0, 0); });
  Refuses([] { CheckSiftStreamingPayload(UINT64_MAX, 1, 0, 0, 0, 0,
                                         UINT64_MAX, 0, 0); });
  Refuses([] { CheckSiftStreamingPayload(0, 0, 0, 0, 0, 0, 0, 0, 0); });
  Refuses([] { CheckSiftStreamingPayload(0, 0, 0, 0, 0, 0, 100, 101, 0); });
  Refuses([] { CheckSiftStreamingPayload(0, 0, 0, 0, 0, 0,
                                         UINT64_MAX, UINT64_MAX, 1); });
}
'''
        with tempfile.TemporaryDirectory(dir=ROOT.parent) as tmp:
            source = pathlib.Path(tmp) / "test.cc"
            source.write_text(code)
            binary = pathlib.Path(tmp) / "test"
            subprocess.run(
                ["c++", "-std=c++17", "-Wall", "-Wextra", "-Werror", "-I", str(ROOT),
                 str(source), "-o", str(binary)], check=True
            )
            subprocess.run([str(binary)], check=True)


if __name__ == "__main__":
    unittest.main()
