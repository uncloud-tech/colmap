# SPDX-License-Identifier: BSD-3-Clause
"""Public CI event/security contract; no third-party test dependencies."""

import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


class LeanMvsCiTest(unittest.TestCase):
    def test_upstream_excludes_only_experimental_pr_base(self):
        for name in ("docker", "mac", "pycolmap", "ubuntu", "windows"):
            with self.subTest(workflow=name):
                text = (
                    ROOT / f".github/workflows/build-{name}.yml"
                ).read_text()
                pr = text.split("  pull_request:\n", 1)[1].split(
                    "  release:", 1
                )[0]
                self.assertIn("branches-ignore: [dev/streaming-workflow]", pr)
                self.assertIn(
                    "types: [ assigned, opened, synchronize, reopened ]", pr
                )
                self.assertIn("      - main\n      - release/*", text)
                self.assertIn("types: [ published, edited ]", text)

    def test_blacksmith_trust_and_checkout_boundary(self):
        text = (ROOT / ".github/workflows/lean-mvs.yml").read_text()
        self.assertIn("branches: [dev/streaming-workflow]", text)
        self.assertEqual(
            text.count(
                "    if: github.event_name == 'workflow_dispatch' || "
                "github.event.pull_request.head.repo.full_name "
                "== github.repository\n"
            ),
            2,
        )
        self.assertEqual(
            text.count(
                "ref: ${{ github.event.pull_request.head.sha || github.sha }}"
            ),
            2,
        )
        self.assertEqual(text.count("persist-credentials: false"), 2)
        self.assertEqual(text.count("fetch-depth: 0"), 2)
        self.assertIn("permissions:\n  contents: read", text)
        self.assertNotIn("pull_request_target", text)
        self.assertNotIn("--privileged", text)
        self.assertLess(
            text.index("apt-get install -y git ca-certificates"),
            text.rindex("uses: actions/checkout@v7"),
        )
        self.assertIn(
            "container:\n      image: "
            "nvidia/cuda:12.8.1-devel-ubuntu24.04@sha256:"
            "4b9ed5fa8361736996499f64ecebf25d4ec37ff56e4d11323ccde10aa36e0c43",
            text,
        )

    def test_container_trust_is_limited_to_checked_out_workspace(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            directory = Path(directory)
            workspace = directory / "workspace"
            other = directory / "other"
            env = {
                **os.environ,
                "GIT_CONFIG_GLOBAL": str(directory / "gitconfig"),
                "GIT_CONFIG_NOSYSTEM": "1",
            }
            for repository in (workspace, other):
                subprocess.run(
                    ["git", "init", "-q", str(repository)],
                    env=env,
                    check=True,
                )
            env["GIT_TEST_ASSUME_DIFFERENT_OWNER"] = "1"
            env["GITHUB_WORKSPACE"] = str(workspace)

            def inspect(repository):
                return subprocess.run(
                    [
                        "git",
                        "-C",
                        str(repository),
                        "rev-parse",
                        "--show-toplevel",
                    ],
                    env=env,
                    text=True,
                    capture_output=True,
                )

            self.assertNotEqual(inspect(workspace).returncode, 0)
            command = (
                'git config --global --add safe.directory "$GITHUB_WORKSPACE"'
            )
            workflow = (ROOT / ".github/workflows/lean-mvs.yml").read_text()
            self.assertIn(command, workflow)
            subprocess.run(["bash", "-c", command], env=env, check=True)
            self.assertEqual(inspect(workspace).returncode, 0)
            self.assertNotEqual(inspect(other).returncode, 0)
            trusted = subprocess.check_output(
                ["git", "config", "--global", "--get-all", "safe.directory"],
                env=env,
                text=True,
            )
            self.assertEqual(trusted.splitlines(), [str(workspace)])

    def test_format_refuses_invalid_base(self):
        result = subprocess.run(
            [
                "bash",
                str(ROOT / "scripts/ci/check_mvs_format.sh"),
                "not-a-commit",
            ],
            text=True,
            capture_output=True,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Invalid format base", result.stderr)

    def test_format_propagates_diff_failure(self):
        real_git = shutil.which("git")
        self.assertIsNotNone(real_git)
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            directory = Path(directory)
            validation_log = directory / "validated"
            commands = {
                "git": (
                    "#!/bin/sh\n"
                    'if [ "$1" = diff ]; then\n'
                    '  echo "Synthetic diff failure" >&2\n'
                    "  exit 42\n"
                    "fi\n"
                    'if [ "$1" = rev-parse ]; then\n'
                    '  echo base-valid >> "$VALIDATION_LOG"\n'
                    "fi\n"
                    'exec "$REAL_GIT" "$@"\n'
                ),
                "ruff": "#!/bin/sh\necho 'ruff 0.15.20'\n",
                "clang-format": (
                    "#!/bin/sh\necho 'clang-format version 22.1.5'\n"
                ),
            }
            for name, content in commands.items():
                command = directory / name
                command.write_text(content)
                command.chmod(0o755)
            result = subprocess.run(
                ["bash", str(ROOT / "scripts/ci/check_mvs_format.sh"), "HEAD"],
                text=True,
                capture_output=True,
                env={
                    **os.environ,
                    "PATH": str(directory) + os.pathsep + os.environ["PATH"],
                    "REAL_GIT": real_git,
                    "VALIDATION_LOG": str(validation_log),
                },
            )
            self.assertIn("base-valid", validation_log.read_text())
        self.assertEqual(result.returncode, 42)
        self.assertIn("Synthetic diff failure", result.stderr)

    def test_cpu_runner_refuses_missing_contract_before_compiling(self):
        # Exercise the helper's real public CLI against an incomplete checkout.
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            result = subprocess.run(
                ["bash", str(ROOT / "scripts/ci/run_mvs_cpu.sh"), directory],
                text=True,
                capture_output=True,
            )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Missing registered CPU contract", result.stderr)


if __name__ == "__main__":
    unittest.main()
