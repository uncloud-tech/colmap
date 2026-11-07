# SPDX-License-Identifier: BSD-3-Clause
"""Exercise Windows checkout line endings with real Git blobs and generator."""

import importlib.util
import os
import re
import shlex
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GENERATOR = ROOT / "scripts/resident/generate_photo_phase_fingerprint.py"
spec = importlib.util.spec_from_file_location("photo_generator", GENERATOR)
generator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(generator)


class WindowsCheckoutTest(unittest.TestCase):
    def test_resident_cli_tests_keep_portable_windows_coverage(self):
        cmake = (ROOT / "src/colmap/exe/CMakeLists.txt").read_text()
        permission_case = "ExportRefusesUnwritableParentFailClosed"
        for name in ("resident_cli_contract", "resident_cli_cohort"):
            with self.subTest(source=name):
                source = (ROOT / f"src/colmap/exe/{name}_test.cc").read_text()
                # Exercise actual guards without dependency headers. This is
                # source-selection proof, not native Windows compilation.
                stripped = "\n".join(
                    "UNISTD_HEADER" if line == "#include <unistd.h>" else line
                    for line in source.splitlines()
                    if not line.lstrip().startswith("#include")
                    or line == "#include <unistd.h>"
                )

                def preprocess(input_source, *defines):
                    return subprocess.check_output(
                        ["c++", "-E", "-P", "-x", "c++", *defines, "-"],
                        input=input_source,
                        text=True,
                    )

                windows = preprocess(
                    stripped, "-D_WIN32", "-U__linux__", "-U__unix__"
                )
                linux = preprocess(stripped, "-U_WIN32")
                for token in (
                    "UNISTD_HEADER",
                    "::geteuid(",
                    "::dup(",
                    "::dup2(",
                    "::fileno(",
                    "::open(",
                    "::close(",
                ):
                    self.assertFalse(token in windows, token)
                tests = set(re.findall(r"TEST_F\(\s*\w+,\s*(\w+)", source))
                windows_tests = set(
                    re.findall(r"TEST_F\(\s*\w+,\s*(\w+)", windows)
                )
                self.assertEqual(windows_tests, tests - {permission_case})
                self.assertEqual(
                    set(re.findall(r"TEST_F\(\s*\w+,\s*(\w+)", linux)), tests
                )
                if name == "resident_cli_contract":
                    self.assertIn(permission_case, linux)
                    self.assertIn("::geteuid(", linux)
                else:
                    self.assertEqual(len(tests), 7)
                    self.assertIn("CaptureStderr()", windows)
                    self.assertIn("GetCapturedStderr()", windows)
                # Both targets remain registered under their existing MVS/
                # CUDA conditions, without a Windows target exclusion.
                registration = cmake.index(f"NAME {name}_test")
                block = cmake[cmake.rfind("if(", 0, registration) :]
                block = block.split("endif()", 1)[0]
                self.assertNotIn("WIN32", block)
                self.assertIn(f"SRCS {name}_test.cc", block)

    def test_workflow_checkout_preserves_committed_fingerprint_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # Confine --global to this fixture, not the host Git config.
            env = dict(
                os.environ,
                GIT_CONFIG_GLOBAL=str(root / "global.config"),
                GIT_CONFIG_NOSYSTEM="1",
            )

            def git(*args):
                return subprocess.check_output(
                    ["git", "-C", str(root), *args],
                    env=env,
                    stderr=subprocess.DEVNULL,
                )

            git("init", "-q")
            objects = subprocess.check_output(
                ["git", "-C", str(ROOT), "rev-parse", "--git-path", "objects"],
                text=True,
            ).strip()
            # Read-only object reuse supplies actual ancestry and committed
            # closure, without copying or changing the source repository.
            (root / ".git/objects/info/alternates").write_text(
                str((ROOT / objects).resolve()) + "\n"
            )
            revision = subprocess.check_output(
                ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                text=True,
            ).strip()
            git("update-ref", "HEAD", revision)
            git("config", "--global", "core.autocrlf", "true")
            git("checkout", "HEAD", "--", *generator.CLOSURE)
            first = generator.CLOSURE[0]
            blob = git("show", f"HEAD:{first}")
            working = (root / first).read_bytes()
            self.assertNotIn(b"\r\n", blob)
            self.assertIn(b"\r\n", working)
            self.assertEqual(working.replace(b"\r\n", b"\n"), blob)
            with self.assertRaisesRegex(ValueError, first):
                generator.generate(root, root / "rejected.h")
            self.assertFalse((root / "rejected.h").exists())

            workflow = (
                ROOT / ".github/workflows/build-windows.yml"
            ).read_text()
            before_checkout = workflow.split("- uses: actions/checkout@", 1)[0]
            commands = [
                line.strip()
                for line in before_checkout.splitlines()
                if line.strip().startswith("git config ")
            ]
            self.assertEqual(
                len(commands),
                1,
                "Configure LF checkout before actions/checkout",
            )
            subprocess.run(
                shlex.split(commands[0]), cwd=root, env=env, check=True
            )
            self.assertEqual(git("config", "core.autocrlf").strip(), b"false")
            # Like a fresh checkout, not an in-place timestamp optimization.
            for path in generator.CLOSURE:
                (root / path).unlink()
            git("checkout", "HEAD", "--", *generator.CLOSURE)
            for path in generator.CLOSURE:
                self.assertEqual(
                    (root / path).read_bytes(), git("show", f"HEAD:{path}")
                )
            output = root / "accepted.h"
            generator.generate(root, output)
            self.assertIn(revision, output.read_text())
            self.assertIn(generator.LINEAGE, output.read_text())
            self.assertIn("fetch-depth: 0", workflow)


if __name__ == "__main__":
    unittest.main()
