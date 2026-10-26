# SPDX-License-Identifier: BSD-3-Clause
"""Exercise Windows checkout line endings with real Git blobs and generator."""

import importlib.util
import os
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
