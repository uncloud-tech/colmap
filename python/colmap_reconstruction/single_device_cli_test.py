# SPDX-License-Identifier: BSD-3-Clause
"""Actual installed native planning/binding; no GPU or mock binary required."""

import json
import os
import unittest

from . import fixed_cohort_cli_test as fixture


@unittest.skipUnless(
    os.environ.get("COLMAP_TEST_EXECUTABLE"),
    "set COLMAP_TEST_EXECUTABLE to the built colmap binary",
)
class SingleDeviceCommandTest(unittest.TestCase):
    setUp = fixture.FixedCohortCommandTest.setUp
    invoke = fixture.FixedCohortCommandTest.invoke

    def automatic_args(self, report):
        return self.args + [
            "--automatic_stereo_storage",
            "1",
            "--auto_active_owners",
            "1",
            "--auto_phase_active_owners",
            "1",
            "--photo_source_storage",
            "auto",
            "--stereo_devices",
            "0:" + "a" * 32 + ":" + str(1 << 30),
            "--stereo_disk_capacity_bytes",
            str(1 << 30),
            "--report_path",
            str(report),
        ]

    def test_single_device_metadata_prepare_binds_without_driver_or_pixels(
        self,
    ):
        report = self.root / "automatic-plan.json"
        result = self.invoke(self.automatic_args(report))
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        plan = json.loads(report.read_bytes())
        self.assertEqual(plan["execution_kind"], "resident_snapshot")
        self.assertRegex(plan["allocation_fingerprint"], r"^[0-9a-f]{64}$")
        self.assertEqual(
            plan["snapshot"]["effective_photo_source_storage"], "private"
        )
        self.assertEqual(
            [p["phase"] for p in plan["snapshot"]["phases"]], ["photo", "geo"]
        )
        self.assertIn(
            "owned_cuda_graphs", plan["scientific_inputs"][0]["photo_options"]
        )
        self.assertIn(
            "owned_async_pool", plan["scientific_inputs"][0]["geo_options"]
        )
        self.assertFalse(self.output.exists())
        # Existing multi-GPU API still refuses an ordinary one-device request.
        ordinary = self.automatic_args(self.root / "ordinary.json")
        ordinary[ordinary.index("--automatic_stereo_storage") + 1] = "0"
        ordinary += ["--observe_capacities", "1"]
        refused = self.invoke(ordinary)
        self.assertNotEqual(refused.returncode, 0)
        self.assertIn("at least two", refused.stdout + refused.stderr)

    def test_single_device_changed_approval_refuses_before_driver_and_decode(
        self,
    ):
        original = self.root / "original-plan.json"
        prepared = self.invoke(self.automatic_args(original))
        self.assertEqual(
            prepared.returncode, 0, prepared.stdout + prepared.stderr
        )
        plan = json.loads(original.read_bytes())
        for change in ("mode", "policy", "device", "capacity", "fingerprint"):
            mutated = json.loads(original.read_bytes())
            if change == "mode":
                mutated["execution_kind"] = "single_device_host_exchange"
            elif change == "policy":
                mutated["scientific_inputs"][0]["geo_options"][
                    "owned_async_pool"
                ] = True
            elif change == "device":
                mutated["phases"][0]["groups"][0]["uuid"] = "b" * 32
            elif change == "capacity":
                mutated["host_capacity_bytes"] = "1"
            else:
                mutated["allocation_fingerprint"] = "0" * 64
            approved = self.root / ("changed-" + change + ".json")
            approved.write_text(json.dumps(mutated))
            report = self.root / ("execution-" + change + ".json")
            args = self.automatic_args(report)
            args[args.index("--prepare_only") + 1] = "0"
            result = self.invoke(
                args + ["--approved_multi_gpu_plan", str(approved)]
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(
                "changed since approval", result.stdout + result.stderr
            )
            self.assertFalse(self.output.exists())
            self.assertFalse(report.exists())
        self.assertEqual(plan["execution_kind"], "resident_snapshot")
        # A native-byte-identical approval passes binding and reaches the real
        # UUID/FREE gate (the fixture UUID is deliberately not a real device).
        args = self.automatic_args(self.root / "unchanged-execution.json")
        args[args.index("--prepare_only") + 1] = "0"
        unchanged = self.invoke(
            args + ["--approved_multi_gpu_plan", str(original)]
        )
        self.assertNotEqual(unchanged.returncode, 0)
        self.assertNotIn(
            "changed since approval", unchanged.stdout + unchanged.stderr
        )
        # Changing actual execution options, not JSON formatting, also binds.
        for key in ("owned_async_pool", "owned_cuda_graphs"):
            changed = self.invoke(
                args
                + [
                    "--approved_multi_gpu_plan",
                    str(original),
                    "--PatchMatchStereo." + key,
                    "1",
                ]
            )
            self.assertNotEqual(changed.returncode, 0)
            self.assertIn(
                "changed since approval", changed.stdout + changed.stderr
            )
            self.assertFalse(self.output.exists())

    def test_single_device_no_fit_is_native_metadata_refusal_before_decode(
        self,
    ):
        (self.workspace / "images/a.pgm").unlink()
        report = self.root / "no-fit.json"
        args = self.automatic_args(report)
        args[args.index("--host_capacity_bytes") + 1] = "1"
        result = self.invoke(args)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(
            "minimum HOST-exchange envelope does not fit",
            result.stdout + result.stderr,
        )
        self.assertNotIn("Cannot read", result.stdout + result.stderr)
        self.assertFalse(report.exists())
        self.assertFalse(self.output.exists())
