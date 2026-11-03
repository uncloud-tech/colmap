# SPDX-License-Identifier: BSD-3-Clause
"Configure the real MVS rules with stub targets; no CUDA build or GPU needed."

import pathlib
import subprocess
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[2]


class BlackwellBuildSelectionTest(unittest.TestCase):
    def configure(self, architectures, opt_in=None):
        with tempfile.TemporaryDirectory(
            prefix=".blackwell-test-", dir=ROOT
        ) as directory:
            source = pathlib.Path(directory)
            (source / "empty.cc").write_text("// Configure-only target.\n")
            # Read the real root option declaration, without configuring its
            # heavyweight dependencies. Missing options must not default to ON.
            options = "\n".join(
                line
                for line in (ROOT / "CMakeLists.txt").read_text().splitlines()
                if line.startswith("option(MVS_NATIVE_BLACKWELL_ENABLED ")
            )
            (source / "CMakeLists.txt").write_text(
                f"""cmake_minimum_required(VERSION 3.24)
project(ArchitectureSelection LANGUAGES CXX)
{options}
set(CUDA_ENABLED ON)
set(TESTS_ENABLED OFF)
function(COLMAP_ADD_LIBRARY)
    cmake_parse_arguments(ARG "" "NAME" "" ${{ARGN}})
    add_library(${{ARG_NAME}} STATIC "{source.as_posix()}/empty.cc")
endfunction()
function(COLMAP_ADD_TEST)
endfunction()
set(PROJECT_SOURCE_DIR "{ROOT.as_posix()}")
add_subdirectory("{ROOT.as_posix()}/src/colmap/mvs" mvs)
get_target_property(archs colmap_mvs_cuda CUDA_ARCHITECTURES)
file(WRITE "${{CMAKE_BINARY_DIR}}/architectures" "${{archs}}")
"""
            )
            build = source / "build"
            command = [
                "cmake",
                "-S",
                str(source),
                "-B",
                str(build),
                "-DCMAKE_CUDA_ARCHITECTURES=" + architectures,
            ]
            if opt_in is not None:
                command.append("-DMVS_NATIVE_BLACKWELL_ENABLED=" + opt_in)
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(
                result.returncode, 0, result.stdout + result.stderr
            )
            return (build / "architectures").read_text(), result.stderr

    def test_native_sm120_is_explicit_experimental_opt_in(self):
        selected, warnings = self.configure("120")
        self.assertEqual(selected, "90-virtual")
        self.assertIn("compiler bug", warnings)
        selected, warnings = self.configure("120", "ON")
        self.assertEqual(selected, "120")
        self.assertIn("experimental", warnings)
        self.assertIn("correctness", warnings)

    def test_pre_blackwell_targets_are_unchanged(self):
        for opt_in in (None, "OFF", "ON"):
            with self.subTest(opt_in=opt_in):
                selected, _ = self.configure("86;89", opt_in)
                self.assertEqual(selected, "86;89")

    def test_mixed_targets_preserve_default_workaround(self):
        selected, _ = self.configure("86;89;120-real", "OFF")
        self.assertEqual(selected, "86;89;90-virtual")
        selected, _ = self.configure("86;89;120-real", "ON")
        self.assertEqual(selected, "86;89;120-real")


if __name__ == "__main__":
    unittest.main()
