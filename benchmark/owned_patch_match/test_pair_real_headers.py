# SPDX-License-Identifier: BSD-3-Clause
"""Real COLMAP-header regression for PatchMatchController's Workspace owner."""

import pathlib
import subprocess
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[2]


class PairRealHeaders(unittest.TestCase):
    def test_workspace_completeness_is_provided_at_driver_call_site(self):
        driver = (ROOT / "src/colmap/exe/owned_pair.cc").read_text()
        self.assertIn('#include "colmap/mvs/workspace.h"', driver)

        # This uses the actual public headers. A call site needs to create a
        # PatchMatchController, not just name its type: the implicit destructor
        # then destroys unique_ptr<Workspace> and requires Workspace complete.
        body = """#include "colmap/mvs/patch_match.h"\n"""
        instantiate = """
void InstantiateController() {
  colmap::mvs::PatchMatchOptions options;
  colmap::mvs::PatchMatchController controller(
      options, {}, "COLMAP", "option-all");
}
"""
        include_dirs = ["-I", str(ROOT / "src")]
        if pathlib.Path("/usr/include/eigen3").is_dir():
            include_dirs += ["-I", "/usr/include/eigen3"]

        with tempfile.TemporaryDirectory(
            prefix="owned-pair-real-headers-"
        ) as tmp:
            source = pathlib.Path(tmp) / "controller.cc"
            source.write_text(body + instantiate)
            command = [
                "g++",
                "-std=c++17",
                "-fsyntax-only",
                *include_dirs,
                str(source),
            ]

            # Prove this is the exact failure addressed, rather than a synthetic
            # incomplete-type fixture or a fake PatchMatchController.
            missing = subprocess.run(
                command, capture_output=True, text=True, timeout=30
            )
            self.assertNotEqual(missing.returncode, 0, missing.stdout)
            self.assertIn("invalid application of", missing.stderr)
            self.assertIn("incomplete type", missing.stderr)
            self.assertIn("colmap::mvs::Workspace", missing.stderr)

            # Match the production call site's header dependency: completeness
            # is supplied locally without changing PatchMatchController's API.
            source.write_text(
                '#include "colmap/mvs/workspace.h"\n' + body + instantiate
            )
            complete = subprocess.run(
                command, capture_output=True, text=True, timeout=30
            )
            self.assertEqual(complete.returncode, 0, complete.stderr)


if __name__ == "__main__":
    unittest.main()
