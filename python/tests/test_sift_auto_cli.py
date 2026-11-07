# SPDX-License-Identifier: BSD-3-Clause
"Actual installed native inventory command; CPU only, not CUDA qualification."

import json
import os
import subprocess
import unittest


@unittest.skipUnless(
    os.environ.get("COLMAP_TEST_EXECUTABLE"),
    "set COLMAP_TEST_EXECUTABLE to built/installed colmap",
)
class SiftAutoCliTest(unittest.TestCase):
    def plan(self, host, device, disk, *extra):
        return subprocess.run(
            [
                os.environ["COLMAP_TEST_EXECUTABLE"],
                "sift_extraction_plan",
                "--dimensions",
                "264x129,256x129,264x129,257x129,256x129,264x130,256x129",
                "--host_capacity_bytes",
                str(host),
                "--device_capacity_bytes",
                str(device),
                "--disk_capacity_bytes",
                str(disk),
                *extra,
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )

    def test_actual_inventory_thresholds_and_history_capacity(self):
        initial = self.plan(2**40, 2**40, 2**40)
        self.assertEqual(initial.returncode, 0, initial.stderr)
        record = json.loads(initial.stdout)
        self.assertEqual(record["mode"], "resident")
        self.assertEqual(record["input_history_bytes"], 137280)
        self.assertEqual(record["retained_pyramid_width"], 528)
        self.assertEqual(record["retained_pyramid_height"], 260)
        self.assertFalse(record["mode_transitions"])
        self.assertEqual(record["num_threads"], 1)
        exact = self.plan(
            record["resident_host_required_bytes"],
            record["resident_device_required_bytes"],
            0,
        )
        self.assertEqual(json.loads(exact.stdout)["mode"], "resident")
        streamed = self.plan(
            2**40,
            record["stream_device_required_bytes"],
            record["scratch_static_bytes"],
        )
        self.assertEqual(json.loads(streamed.stdout)["mode"], "streamed")
        disk = self.plan(
            2**40,
            record["stream_device_required_bytes"],
            record["scratch_static_bytes"] - 1,
        )
        self.assertEqual(json.loads(disk.stdout)["mode"], "refused")
        neither = self.plan(1, 1, 1)
        self.assertEqual(json.loads(neither.stdout)["mode"], "refused")

    def test_actual_command_rejects_unsupported_options_and_byte_syntax(self):
        unsupported = self.plan(
            2**30, 2**30, 2**30, "--SiftExtraction.first_octave", "0"
        )
        self.assertNotEqual(unsupported.returncode, 0)
        invalid = self.plan(2**30, 2**30, 2**30, "--policy_host_bytes", "-1")
        self.assertNotEqual(invalid.returncode, 0)
        self.assertIn("canonical byte allowance", invalid.stderr)

    def test_all_byte_fields_use_the_canonical_native_adapter(self):
        for name in (
            "host_capacity_bytes",
            "device_capacity_bytes",
            "disk_capacity_bytes",
            "policy_host_bytes",
            "policy_device_bytes",
            "reserve_host_bytes",
            "reserve_device_bytes",
        ):
            for value in (
                "-1",
                "+1",
                "01",
                "18446744073709551616",
                "1.5",
                "1e3",
                "1x",
                " 1",
                "1 ",
                "",
            ):
                with self.subTest(option=name, value=value):
                    result = subprocess.run(
                        [
                            os.environ["COLMAP_TEST_EXECUTABLE"],
                            "sift_extraction_plan",
                            "--dimensions",
                            "256x128",
                            "--" + name,
                            value,
                        ],
                        capture_output=True,
                        text=True,
                        timeout=30,
                    )
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn("canonical byte allowance", result.stderr)
