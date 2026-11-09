# SPDX-License-Identifier: BSD-3-Clause
"""Native-output publication contract; synthetic maps are not GPU proof."""

import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from .cli import Reconstruction, parse_args


class PhotoRetentionTest(unittest.TestCase):
    def test_photo_maps_validate_before_geo_publication(self):
        for defect in (
            None,
            "missing",
            "shape",
            "partial",
            "nonfinite",
            "policy",
        ):
            with (
                self.subTest(defect=defect),
                tempfile.TemporaryDirectory() as d,
            ):
                root = Path(d)
                args = parse_args(
                    [
                        "--photos",
                        str(root),
                        "--workspace",
                        str(root),
                        "--output-policy",
                        "photo_and_geo",
                    ]
                )
                reconstruction = Reconstruction(args, sys.executable)
                reconstruction.assets.mkdir()
                stereo = reconstruction.dense / "stereo"
                stereo.mkdir(parents=True)
                (stereo / "fusion.cfg").write_text("image.pgm\nlate.pgm\n")
                reconstruction.identity = {"allocation_fingerprint": "test"}
                reconstruction.input_files = {}
                reconstruction.graph = {
                    "phases": {
                        phase: {
                            "requested_options": {"gpu_index": "0"},
                            "jobs": [
                                {
                                    "reference": reference,
                                    "sources": [1 - reference],
                                    "options": {},
                                }
                                for reference in (0, 1)
                            ],
                        }
                        for phase in ("photo", "geo")
                    }
                }
                reconstruction.images = {
                    0: {"name": "image.pgm", "width": 1, "height": 1},
                    1: {"name": "late.pgm", "width": 1, "height": 1},
                }

                def native(
                    command,
                    *arguments,
                    defect=defect,
                    root=root,
                    reconstruction=reconstruction,
                ):
                    self.assertEqual(command, "experimental_fixed_cohort")
                    self.assertEqual(
                        arguments[arguments.index("--output_policy") + 1],
                        "photo_and_geo",
                    )
                    report = {
                        "status": "completed",
                        "effective_backend": "experimental_fixed_cohort",
                        "allocation_fingerprint": "test",
                        "output_policy": (
                            "final_geo_only"
                            if defect == "policy"
                            else "photo_and_geo"
                        ),
                        "phases": {
                            phase: reconstruction.graph["phases"][phase]["jobs"]
                            for phase in ("photo", "geo")
                        },
                    }
                    (reconstruction.assets / "cohort.json").write_text(
                        json.dumps(report)
                    )
                    for kind, channels in (
                        ("depth_maps", 1),
                        ("normal_maps", 3),
                    ):
                        folder = root / "cohort/stereo" / kind
                        folder.mkdir(parents=True)
                        for suffix in ("photometric", "geometric"):
                            values = [1.0] * channels
                            header = f"1&1&{channels}&".encode()
                            # First reference is always valid. A later PHOTO
                            # failure must still prevent ALL GEO publication.
                            (folder / f"image.pgm.{suffix}.bin").write_bytes(
                                header
                                + struct.pack("<" + "f" * channels, *values)
                            )
                            path = folder / f"late.pgm.{suffix}.bin"
                            if (
                                suffix == "photometric"
                                and kind == "normal_maps"
                            ):
                                if defect == "missing":
                                    continue
                                if defect == "shape":
                                    header = f"2&1&{channels}&".encode()
                                if defect == "nonfinite":
                                    values[0] = float("nan")
                            payload = header + struct.pack(
                                "<" + "f" * channels, *values
                            )
                            if suffix == "photometric" and defect == "partial":
                                payload = payload[:-1]
                            path.write_bytes(payload)

                with (
                    patch.object(reconstruction, "native", side_effect=native),
                    patch.object(
                        reconstruction,
                        "resources",
                        return_value={"host": 1 << 30, "device": 1 << 30},
                    ),
                ):
                    if defect is None:
                        reconstruction.dense_stage()
                    else:
                        with self.assertRaises((ValueError, OSError)):
                            reconstruction.dense_stage()
                published = reconstruction.dense / "stereo/depth_maps"
                if defect is None:
                    for name in ("image.pgm", "late.pgm"):
                        self.assertTrue(
                            (published / (name + ".geometric.bin")).is_file()
                        )
                    self.assertTrue(
                        (
                            root
                            / "cohort/stereo/depth_maps"
                            / "image.pgm.photometric.bin"
                        ).is_file()
                    )
                else:
                    self.assertFalse(published.exists())


if __name__ == "__main__":
    unittest.main()
