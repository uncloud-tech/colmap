# SPDX-License-Identifier: BSD-3-Clause
"""Configure real MVS rules with stub targets, without CUDA or a GPU."""

import pathlib
import subprocess
import unittest
from tempfile import TemporaryDirectory

ROOT = pathlib.Path(__file__).resolve().parents[2]


class BlackwellBuildSelectionTest(unittest.TestCase):
    def test_workspace_execution_fixture_uses_host_compiler(self):
        mvs = ROOT / "src/colmap/mvs"
        cmake = (mvs / "CMakeLists.txt").read_text()
        registration = cmake.split(
            "    COLMAP_ADD_TEST(\n"
            "        NAME owned_fixed_cohort_pipeline_test",
            1,
        )[1].split("\n    )", 1)[0]
        registration = (
            "COLMAP_ADD_TEST(NAME owned_fixed_cohort_pipeline_test"
            + registration
            + "\n)"
        )
        sources = registration.split("SRCS", 1)[1].split("LINK_LIBS", 1)[0]
        sources = sources.split()
        workspace = next(
            name for name in sources if "workspace_execution" in name
        )
        contents = (mvs / workspace).read_text()
        self.assertNotRegex(
            contents,
            r"\b(?:__global__|__device__|__shared__|__constant__)\b|<<<",
        )
        self.assertEqual(contents.count("TEST(FixedCohortWorkspaceGpu,"), 2)
        with TemporaryDirectory(prefix=".host-fixture-", dir=ROOT) as tmp:
            source = pathlib.Path(tmp)
            for name in sources:
                (source / name).write_text((mvs / name).read_text())
            # Exercise the real test macro and registration. Other sources
            # remain CUDA in production; no SDK/compiler is faked here.
            device_sources = " ".join(
                name for name in sources if name != workspace
            )
            (source / "CMakeLists.txt").write_text(f'''
cmake_minimum_required(VERSION 3.24)
project(WorkspaceFixtureLanguage LANGUAGES CXX)
enable_testing()
include("{ROOT.as_posix()}/cmake/CMakeHelper.cmake")
set(TESTS_ENABLED ON)
set(FOLDER_NAME mvs)
add_library(colmap_mvs_cuda INTERFACE)
add_library(colmap_gtest_main INTERFACE)
set_source_files_properties({device_sources} PROPERTIES HEADER_FILE_ONLY ON)
{registration}
set_target_properties(colmap_mvs_owned_fixed_cohort_pipeline_test
    PROPERTIES LINKER_LANGUAGE CXX)
get_source_file_property(language {workspace} LANGUAGE)
get_target_property(links colmap_mvs_owned_fixed_cohort_pipeline_test
    LINK_LIBRARIES)
file(WRITE "${{CMAKE_BINARY_DIR}}/language" "${{language}}")
file(WRITE "${{CMAKE_BINARY_DIR}}/links" "${{links}}")
''')
            build = source / "build"
            result = subprocess.run(
                ["cmake", "-S", str(source), "-B", str(build)],
                capture_output=True,
                text=True,
            )
            self.assertEqual(
                result.returncode, 0, result.stdout + result.stderr
            )
            self.assertEqual((build / "language").read_text(), "CXX")
            self.assertEqual(
                (build / "links").read_text(),
                "colmap_mvs_cuda;colmap_gtest_main",
            )
            self.assertIn(
                "mvs/owned_fixed_cohort_pipeline_test",
                (build / "CTestTestfile.cmake").read_text(),
            )
        generator = mvs / "testdata/generate_fixed_cohort_workspace_fixture.py"
        self.assertIn(f"../{workspace}", generator.read_text())

    def configure(self, architectures, opt_in=None):
        with TemporaryDirectory(prefix=".blackwell-test-", dir=ROOT) as tmp:
            source = pathlib.Path(tmp)
            (source / "empty.cc").write_text("// Configure-only target.\n")
            # Read the real root option declaration, without configuring its
            # heavyweight dependencies. Exercise the public option default.
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

    def test_native_sm120_is_default_with_explicit_compatibility_fallback(self):
        for option in (None, "ON"):
            with self.subTest(option=option):
                selected, warnings = self.configure("120", option)
                self.assertEqual(selected, "120")
                self.assertIn("correctness", warnings)
        selected, warnings = self.configure("120", "OFF")
        self.assertEqual(selected, "90-virtual")
        self.assertIn("compiler bug", warnings)

    def test_architecture_selection_does_not_equate_sm100_with_sm120(self):
        for architecture in (
            "100-real",
            "120-real",
            "100;120",
            "86;89;120-real",
        ):
            with self.subTest(architecture=architecture):
                selected, warnings = self.configure(architecture)
                self.assertEqual(selected, architecture)
                self.assertIn("SM100", warnings)
                self.assertIn("unqualified", warnings)
                self.assertIn("SM120", warnings)
                self.assertIn("bounded", warnings)
                self.assertIn("MVS_NATIVE_BLACKWELL_ENABLED=OFF", warnings)
        for architecture in ("100-real", "120-real", "100;120"):
            with self.subTest(compatibility=architecture):
                selected, _ = self.configure(architecture, "OFF")
                self.assertEqual(selected, "90-virtual")

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
