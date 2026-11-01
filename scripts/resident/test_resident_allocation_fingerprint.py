# SPDX-License-Identifier: BSD-3-Clause
"""Canonical build binding must change with phase allocation behavior."""

import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "resident_generator",
    Path(__file__).with_name("generate_resident_allocation_fingerprint.py"),
)
assert spec is not None and spec.loader is not None
generator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(generator)

FUSION_ALLOCATION_SOURCES = (
    "src/colmap/mvs/fusion.h",
    "src/colmap/mvs/fusion.cc",
    "src/colmap/mvs/fusion_resource.h",
    "src/colmap/mvs/fusion_mapped_mat.h",
    "src/colmap/mvs/fusion_mapped_mat.cc",
    "src/colmap/mvs/fusion_dependencies.h",
    "src/colmap/util/ply.h",
    "src/colmap/util/eigen_alignment.h",
    "src/colmap/util/threading.h",
    "src/colmap/util/threading.cc",
    "src/colmap/util/oiio_utils.h",
    "src/colmap/util/oiio_utils.cc",
)


UNDISTORTION_ALLOCATION_SOURCES = (
    "src/colmap/controllers/undistorters.h",
    "src/colmap/controllers/undistorters.cc",
    "src/colmap/controllers/undistortion_resource.h",
    "src/colmap/image/undistortion.h",
    "src/colmap/image/undistortion.cc",
    "src/colmap/exe/image.cc",
    "src/colmap/exe/colmap.cc",
    "src/colmap/util/cancellation.h",
    "src/colmap/util/cancellation.cc",
)


class ResidentAllocationFingerprintTest(unittest.TestCase):
    def test_phase_mutations_change_compiled_binding_and_rebuild_dependencies(self):
        repo = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            # Copy real source bytes, including phase files even when a
            # regression omits them from the generator's declared closure.
            sources = dict.fromkeys(
                (*generator.CLOSURE, *FUSION_ALLOCATION_SOURCES,
                 *UNDISTORTION_ALLOCATION_SOURCES)
            )
            for name in sources:
                target = root / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes((repo / name).read_bytes())
            output = root / "generated.h"
            depfile = root / "generated.d"

            def generate():
                subprocess.run(
                    [
                        sys.executable,
                        str(repo / "scripts/resident" /
                            "generate_resident_allocation_fingerprint.py"),
                        "--repo-root", str(root), "--output", str(output),
                        "--depfile", str(depfile),
                    ],
                    check=True,
                )
                return output.read_text()

            baseline = generate()
            for name in (*FUSION_ALLOCATION_SOURCES,
                         *UNDISTORTION_ALLOCATION_SOURCES):
                with self.subTest(source=name):
                    source = root / name
                    original = source.read_bytes()
                    try:
                        source.write_bytes(
                            original + b"\n// allocation mutation test\n"
                        )
                        self.assertNotEqual(baseline, generate())
                        self.assertIn(str(source.resolve()), depfile.read_text())
                    finally:
                        source.write_bytes(original)
            self.assertEqual(baseline, generate())


if __name__ == "__main__":
    unittest.main()
