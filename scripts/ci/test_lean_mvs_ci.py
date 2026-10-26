# SPDX-License-Identifier: BSD-3-Clause
"""Public CI event/security contract; no third-party test dependencies."""

import subprocess
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

    def test_cpu_runner_refuses_missing_contract_before_compiling(self):
        # Exercise the helper's real public CLI against an incomplete checkout.
        import tempfile

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
