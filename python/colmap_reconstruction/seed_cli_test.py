# SPDX-License-Identifier: BSD-3-Clause
"""Frontend dispatch tests, not native or numerical repeatability proof."""

import io
import json
import os
import shutil
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


@unittest.skipUnless(
    sys.platform == "linux" and os.geteuid() != 0,
    "Experimental local execution requires non-root Linux",
)
class SeedDispatchTest(unittest.TestCase):
    def run_frontend(self, root, seed, *, bad_dense_seed=None):
        product = Path(__file__).resolve().parents[2]
        photos = root / "photos"
        photos.mkdir()
        native = root / "colmap"
        shutil.copy(product / "python/tests/fake_native.py", native)
        native.chmod(0o755)
        calls = root / "calls.jsonl"
        env = dict(
            os.environ,
            PYTHONPATH=str(product / "python"),
            FAKE_CALLS=str(calls),
        )
        if bad_dense_seed:
            name = (
                "FAKE_BAD_SEED"
                if bad_dense_seed == "requested"
                else "FAKE_BAD_JOB_SEED"
            )
            env[name] = "1"
        argv = [
            "--experimental-local",
            "--yes",
            "--colmap",
            str(native),
            "--photos",
            str(photos),
            "--workspace",
            str(root / "output"),
        ]
        if seed is not None:
            argv += ["--random-seed", str(seed)]
        from colmap_reconstruction.cli import Reconstruction, main

        def prepare_and_plan(reconstruction):
            reconstruction.prepare()
            reconstruction.plan()

        stdout, stderr = io.StringIO(), io.StringIO()
        # Isolate unrelated native extraction admission, as FrontendTest does.
        # Exercise real prepare/plan dispatch, not the GPU cohort seam.
        with (
            patch.dict(os.environ, env),
            patch(
                "colmap_reconstruction.cli.Reconstruction.select_extraction",
                return_value={"mode": "resident"},
            ),
            patch.object(Reconstruction, "run", prepare_and_plan),
            redirect_stdout(stdout),
            redirect_stderr(stderr),
        ):
            status = main(argv)
        result = SimpleNamespace(
            returncode=status,
            stdout=stdout.getvalue(),
            stderr=stderr.getvalue(),
        )
        self.assertTrue(calls.is_file(), result.stdout + result.stderr)
        records = [json.loads(line) for line in calls.read_text().splitlines()]
        return result, dict(records)

    def test_stochastic_seed_reaches_matching_mapping_and_dense_planning(self):
        for requested, expected in ((None, 0), (0, 0), (37, 37), (-1, -1)):
            with (
                self.subTest(seed=requested),
                tempfile.TemporaryDirectory(
                    dir=Path(__file__).resolve().parents[2].parent
                ) as d,
            ):
                root = Path(d)
                result, calls = self.run_frontend(root, requested)
                self.assertEqual(result.returncode, 0, result.stderr)
                for command, option in (
                    ("exhaustive_matcher", "--TwoViewGeometry.random_seed"),
                    ("mapper", "--Mapper.random_seed"),
                    ("optimized_dense_plan", "--random_seed"),
                ):
                    with self.subTest(command=command):
                        self.assertEqual(
                            calls[command].get(option), str(expected)
                        )
                approved = json.loads(
                    (root / "output/planning/approved-plan.json").read_text()
                )
                for stage in ("matching", "mapping", "stereo"):
                    actual = approved["effective_dispatch"][stage][
                        "random_seed"
                    ]
                    self.assertEqual(actual, expected)

    def test_native_dense_seed_mismatch_refuses_before_execution(self):
        for mismatch in ("requested", "job"):
            with (
                self.subTest(mismatch=mismatch),
                tempfile.TemporaryDirectory(
                    dir=Path(__file__).resolve().parents[2].parent
                ) as d,
            ):
                root = Path(d)
                result, calls = self.run_frontend(
                    root, 37, bad_dense_seed=mismatch
                )
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertIn("random seed", result.stderr)
                self.assertNotIn("experimental_fixed_cohort", calls)
                self.assertNotIn("stereo_fusion", calls)
                self.assertFalse((root / "output/result.json").exists())


if __name__ == "__main__":
    unittest.main()
