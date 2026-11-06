# SPDX-License-Identifier: BSD-3-Clause
"""Real generated allocation identity versus frontend compatibility bindings.

No native executable is mocked to echo the policy pin. Generating its compiled
header exercises the same source closure as the native CMake build, not CUDA.
"""

import json
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from colmap_reconstruction.cli import Reconstruction, parse_args
from colmap_reconstruction.fixed_cohort import (
    _FIXED_COHORT_FINGERPRINT,
    ByteCharge,
    ImageShape,
    Job,
    PhaseOptions,
    forecast_fixed_cohort,
)
from colmap_reconstruction.resource_policy import (
    ALLOCATION_FINGERPRINT,
    DEFAULTS,
    select_policy,
)

ROOT = Path(__file__).resolve().parents[2]


class GeneratedPolicyBindingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with tempfile.TemporaryDirectory() as directory:
            header = Path(directory) / "allocation.h"
            subprocess.run(
                [
                    sys.executable,
                    str(
                        ROOT / "scripts/resident/"
                        "generate_resident_allocation_fingerprint.py"
                    ),
                    "--repo-root",
                    str(ROOT),
                    "--output",
                    str(header),
                    "--depfile",
                    str(Path(directory) / "allocation.d"),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            match = re.search(
                r'kResidentAllocationFingerprint\[\]\s*=\s*"([0-9a-f]{64})"',
                header.read_text(),
            )
            if match is None:
                raise AssertionError("canonical generated identity missing")
            cls.fingerprint = match.group(1)

    def test_both_frontends_match_actual_generated_native_identity(self):
        self.assertEqual(ALLOCATION_FINGERPRINT, self.fingerprint)
        self.assertEqual(_FIXED_COHORT_FINGERPRINT, self.fingerprint)

    def args(self):
        return parse_args(["--photos", "/photos", "--workspace", "/output"])

    def test_generated_identity_selects_unchanged_default_allowances(self):
        args = self.args()
        policy = select_policy(
            args, {"allocation_fingerprint": self.fingerprint}
        )
        self.assertEqual(policy["allocation_fingerprint"], self.fingerprint)
        self.assertEqual(policy["selection"], "defaults-and-overrides")
        # Independent contract values: the storage reduction earns no policy
        # allowance, reserve or capacity discount.
        expected = {
            "loader_scratch_host_bytes": 512 * 1024**2,
            "process_host_bytes": 512 * 1024**2,
            "process_device_bytes": 512 * 1024**2,
            "unknown_host_bytes": 1024**3,
            "unknown_device_bytes": 1024**3,
            "reserve_host_bytes": 512 * 1024**2,
            "reserve_device_bytes": 512 * 1024**2,
        }
        self.assertEqual(DEFAULTS, expected)
        for name, value in expected.items():
            self.assertEqual(getattr(args, name), value)
            self.assertEqual(policy["terms"][name]["bytes"], value)
            self.assertEqual(policy["terms"][name]["origin"], "default")

    def test_unknown_identity_refuses_defaults_but_not_complete_expert(self):
        unknown = "0" * 64
        previous = (
            "a2b88f947d71bc1adbab0f54ee71bfafb382fa3e9e0ea852ec7352b89b312b54"
        )
        gaussian_only = (
            "e5a6a377b7f0042e7ab1421321738c9e37883125489563f142a445614c711dc2"
        )
        pre_installed_streaming = (
            "291996962f62c34e7c087a7c0cfd5317f7d1eb2646b70acde15b8bc4af2bbe63"
        )
        for fingerprint in (
            unknown,
            previous,
            gaussian_only,
            pre_installed_streaming,
            "12067da160255267bc0cbb033f410b41e5e8443d2147bc817ba22a11ab96bdc3",
            "18ba1102892dc048ae8f3ec5c43d5eeb81c986986acecd1821500e2c014adf83",
        ):
            with (
                self.subTest(fingerprint=fingerprint),
                self.assertRaisesRegex(
                    ValueError, "no supported allocation binding"
                ),
            ):
                select_policy(
                    self.args(), {"allocation_fingerprint": fingerprint}
                )
        args = self.args()
        for name, value in DEFAULTS.items():
            setattr(args, name, value)
        args.unknown_provenance = "explicit test-only expert declaration"
        policy = select_policy(args, {"allocation_fingerprint": unknown})
        self.assertEqual(policy["selection"], "expert")
        self.assertEqual(policy["allocation_fingerprint"], unknown)

    def test_generated_native_identity_preserves_odd_width_domain_support(self):
        for fingerprint in (self.fingerprint, "0" * 64):
            with self.subTest(fingerprint=fingerprint):
                with tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    reconstruction = Reconstruction(
                        parse_args(
                            ["--photos", str(root), "--workspace", str(root)]
                        ),
                        sys.executable,
                    )
                    reconstruction.assets.mkdir()
                    reconstruction.identity = {
                        "source_commit": "a" * 40,
                        "allocation_fingerprint": fingerprint,
                    }
                    summary = dict(
                        schema_version=1,
                        image_path=str(root),
                        photo_count=1,
                        photos=[
                            dict(
                                path="photo.jpg",
                                width=6201,
                                height=4130,
                                orientation=None,
                            )
                        ],
                        problems=[],
                        non_image_files=[],
                        header_only=True,
                        orientation_applied=False,
                        build=dict(revision="a" * 8, cuda_compiled=True),
                    )
                    # Only image-header discovery is synthetic. The build
                    # identity above came from the canonical source generator.
                    with patch.object(
                        reconstruction,
                        "native",
                        return_value=json.dumps(summary),
                    ):
                        if fingerprint == self.fingerprint:
                            self.assertEqual(
                                reconstruction.inspect_photos(), 6201
                            )
                        else:
                            with self.assertRaisesRegex(
                                ValueError,
                                "matching native odd-width CUDA SIFT",
                            ):
                                reconstruction.inspect_photos()

    def test_fixed_cohort_source_binding_uses_same_generated_closure(self):
        photo = PhaseOptions(False, False, "0", 1, 5, 5)
        geo = PhaseOptions(True, True, "0", 1, 5, 5)
        budget = forecast_fixed_cohort(
            (ImageShape(5, 4, 1, 20), ImageShape(3, 3, 1, 9)),
            (Job(0, (1,), (1,), photo, geo), Job(1, (0,), (0,), photo, geo)),
            max_active_owners=1,
            source_root=ROOT,
            process=ByteCharge(100, 200),
            unknown=ByteCharge(300, 400),
            unknown_provenance="explicit unqualified test allowance",
            headroom_reserve=ByteCharge(10, 20),
        )
        self.assertEqual(budget.source.fingerprint, self.fingerprint)
        self.assertFalse(budget.physical_fit_certified)
        self.assertFalse(budget.calibration_certified)


if __name__ == "__main__":
    unittest.main()
