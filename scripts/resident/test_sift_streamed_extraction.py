# SPDX-License-Identifier: BSD-3-Clause
"""CPU payload-admission tests. These do not qualify CUDA extraction."""

import pathlib
import subprocess
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[2]


class StreamedExtractionTest(unittest.TestCase):
    def test_windows_cuda_keeps_resident_tests_without_posix_calls(self):
        # Preprocess actual guards without requiring CUDA/GTest headers. This
        # checks source selection, not native Windows compilation/qualification.
        cases = {
            "sift_test.cc": [
                "Nominal",
                "InputDomainAndImmutability",
                "ExactDomainCudaUpsampling",
                "ArbitraryWidthsAndReuse",
            ],
            "sift_resampling_test.cu": [
                "NonfiniteCountsPreserveBits",
                "AlignedAllocationEdgesTailsAndReuse",
            ],
        }
        for name, portable_tests in cases.items():
            with self.subTest(source=name):
                source = (ROOT / "src/colmap/feature" / name).read_text()
                without_includes = "\n".join(
                    line
                    for line in source.splitlines()
                    if not line.lstrip().startswith("#include")
                )

                def preprocess(*defines):
                    return subprocess.check_output(
                        [
                            "c++",
                            "-E",
                            "-P",
                            "-x",
                            "c++",
                            "-DCOLMAP_CUDA_ENABLED",
                            *defines,
                            "-",
                        ],
                        input=without_includes,
                        text=True,
                    )

                windows = preprocess("-D_WIN32", "-U__linux__")
                posix = preprocess("-U_WIN32")
                for call in ("mkdtemp(", "rmdir("):
                    self.assertFalse(
                        call in windows, f"Windows retained {call}"
                    )
                self.assertIn("mkdtemp(", posix)
                for test in portable_tests:
                    self.assertIn(test, windows)
                if "std::filesystem::" in posix:
                    # macOS/POSIX also needs this header, not Linux alone.
                    headers = subprocess.check_output(
                        [
                            "c++",
                            "-E",
                            "-P",
                            "-x",
                            "c++",
                            "-U_WIN32",
                            "-U__linux__",
                            "-DCOLMAP_CUDA_ENABLED",
                            "-",
                        ],
                        input="\n".join(
                            "FILESYSTEM_HEADER"
                            if line == "#include <filesystem>"
                            else line
                            for line in source.splitlines()
                            if not line.lstrip().startswith("#include")
                            or line == "#include <filesystem>"
                        ),
                        text=True,
                    )
                    self.assertIn("FILESYSTEM_HEADER", headers)

    def test_streaming_help_describes_current_bounded_domain(self):
        source = (ROOT / "src/colmap/controllers/option_manager.cc").read_text()
        help_text = source.split('"SiftExtraction.streaming",', 1)[1].split(
            "AddDefaultByteOption", 1
        )[0]
        self.assertIn(
            "first_octave -1 or 0, including aligned widths", help_text
        )
        self.assertIn("Nonzero single-root", help_text)
        self.assertIn("Windows", help_text)
        self.assertIn("fail closed", help_text)
        self.assertNotIn("non-4-aligned width only", help_text)
        self.assertNotIn("default -1 unsupported", help_text)

    def test_checked_live_payload(self):
        code = r"""
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
"""
        with tempfile.TemporaryDirectory(dir=ROOT.parent) as tmp:
            source = pathlib.Path(tmp) / "test.cc"
            source.write_text(code)
            binary = pathlib.Path(tmp) / "test"
            subprocess.run(
                [
                    "c++",
                    "-std=c++17",
                    "-Wall",
                    "-Wextra",
                    "-Werror",
                    "-I",
                    str(ROOT),
                    str(source),
                    "-o",
                    str(binary),
                ],
                check=True,
            )
            subprocess.run([str(binary)], check=True)


if __name__ == "__main__":
    unittest.main()
