# SPDX-License-Identifier: BSD-3-Clause
"""Scoped install plumbing test, not a full COLMAP SDK build.

Only the heavy library/test targets are stubbed. The MVS CMake rules and both
fingerprint generators are real, and a downstream compiler uses installed,
relocated headers without access to the build include directory.
"""

import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def run(*args):
    result = subprocess.run(args, text=True, capture_output=True)
    if result.returncode:
        raise AssertionError(f"{args!r}\n{result.stdout}\n{result.stderr}")
    return result.stdout


class GeneratedHeaderInstallTest(unittest.TestCase):
    @unittest.skipUnless(
        shutil.which("cmake") and shutil.which("c++"),
        "CMake and a C++ compiler required",
    )
    def test_cpu_install_and_relocated_consumer(self):
        # Executables stay in the workspace: /tmp may be mounted noexec.
        with tempfile.TemporaryDirectory(
            prefix="sdk-install-", dir=ROOT
        ) as tmp:
            work = Path(tmp)
            source = work / "fixture"
            source.mkdir()
            (source / "empty.cc").write_text("// Build plumbing only.\n")
            (source / "CMakeLists.txt").write_text(
                f'''cmake_minimum_required(VERSION 3.20)
project(InstallFixture LANGUAGES CXX)
include(GNUInstallDirs)
set(CUDA_ENABLED OFF)
set(TESTS_ENABLED OFF)
function(COLMAP_ADD_LIBRARY)
    cmake_parse_arguments(ARG "" "NAME" "" ${{ARGN}})
    add_library(${{ARG_NAME}} STATIC
        "${{CMAKE_CURRENT_FUNCTION_LIST_DIR}}/empty.cc")
endfunction()
function(COLMAP_ADD_TEST)
endfunction()
# Use actual source and generator rules, without configuring heavy dependencies.
set(PROJECT_SOURCE_DIR "{ROOT.as_posix()}")
add_subdirectory("{ROOT.as_posix()}/src/colmap/mvs" mvs)
# Match the unified SDK include/export convention in the root CMakeLists.txt.
add_library(colmap INTERFACE)
target_link_libraries(colmap INTERFACE colmap_mvs)
target_include_directories(colmap INTERFACE
    "$<INSTALL_INTERFACE:${{CMAKE_INSTALL_INCLUDEDIR}}>")
install(TARGETS colmap colmap_mvs EXPORT fixture-targets)
install(EXPORT fixture-targets NAMESPACE colmap:: DESTINATION lib/cmake)
'''
            )
            for includedir in ("include", "sdk/include"):
                with self.subTest(includedir=includedir):
                    build = work / includedir.replace("/", "-")
                    prefix = build / "prefix"
                    run(
                        "cmake",
                        "-S",
                        str(source),
                        "-B",
                        str(build),
                        f"-DCMAKE_INSTALL_PREFIX={prefix}",
                        f"-DCMAKE_INSTALL_INCLUDEDIR={includedir}",
                    )
                    # Fresh install target must first generate both headers.
                    run(
                        "cmake",
                        "--build",
                        str(build),
                        "--target",
                        "install",
                        "--parallel",
                        "1",
                    )
                    relocated = build / "relocated"
                    prefix.rename(relocated)
                    for name in ("photo_phase", "resident_allocation"):
                        expected = build / f"{name}.h"
                        command = [
                            "python3",
                            str(
                                ROOT
                                / "scripts/resident"
                                / f"generate_{name}_fingerprint.py"
                            ),
                            "--repo-root",
                            str(ROOT),
                            "--output",
                            str(expected),
                        ]
                        if name == "resident_allocation":
                            command += [
                                "--depfile",
                                str(expected.with_suffix(".d")),
                            ]
                        run(*command)
                        installed = (
                            relocated
                            / includedir
                            / "colmap/mvs"
                            / "generated"
                            / f"{name}_fingerprint.h"
                        )
                        self.assertEqual(
                            installed.read_bytes(), expected.read_bytes()
                        )
                    consumer = build / "consumer"
                    consumer.mkdir()
                    (consumer / "main.cc").write_text(
                        """\
#include "colmap/mvs/generated/photo_phase_fingerprint.h"
#include "colmap/mvs/generated/resident_allocation_fingerprint.h"
int main() {
  using namespace colmap::mvs;
  static_assert(sizeof(kPhotoPhaseV1Fingerprint) == 65);
  static_assert(sizeof(kPhotoPhaseV1SourceCommit) == 41);
  static_assert(sizeof(kResidentAllocationFingerprint) == 65);
}
"""
                    )
                    (consumer / "CMakeLists.txt").write_text(
                        "cmake_minimum_required(VERSION 3.20)\n"
                        "project(Consumer LANGUAGES CXX)\n"
                        f'include("{relocated.as_posix()}/lib/cmake/fixture-targets.cmake")\n'
                        "add_executable(consumer main.cc)\n"
                        "target_compile_features(consumer PRIVATE cxx_std_17)\n"
                        "target_link_libraries(consumer PRIVATE\n"
                        "  colmap::colmap)\n"
                    )
                    run(
                        "cmake",
                        "-S",
                        str(consumer),
                        "-B",
                        str(consumer / "build"),
                    )
                    run(
                        "cmake",
                        "--build",
                        str(consumer / "build"),
                        "--parallel",
                        "1",
                    )


if __name__ == "__main__":
    unittest.main()
