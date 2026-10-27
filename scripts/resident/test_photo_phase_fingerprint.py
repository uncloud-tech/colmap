# SPDX-License-Identifier: BSD-3-Clause
"""Build-boundary tests; fixtures are TEST-EVIDENCE, not attested numbers."""

import hashlib
import importlib.util
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

spec = importlib.util.spec_from_file_location(
    "photo_generator",
    Path(__file__).with_name("generate_photo_phase_fingerprint.py"),
)
generator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(generator)


class PhotoFingerprintTest(unittest.TestCase):
    def test_graph_allocation_and_host_boundary_sources_are_attested(self):
        required = {
            "src/colmap/mvs/owned_input_bank.h",
            "src/colmap/mvs/owned_input_bank.cc",
            "src/colmap/mvs/owned_input_bank.cu",
            "src/colmap/mvs/owned_cuda_execution.h",
            "src/colmap/mvs/cuda_texture.h",
            "src/colmap/mvs/owned_graph.h",
            "src/colmap/mvs/owned_graph.cu",
            "src/colmap/mvs/owned_graph_topology.h",
            "src/colmap/mvs/owned_graph_coordinator.h",
            "src/colmap/mvs/owned_async_pool.h",
            "src/colmap/mvs/gpu_mat.h",
            "src/colmap/mvs/owned_selection_boundary.h",
            "src/colmap/mvs/patch_match.h",
            "src/colmap/mvs/patch_match.cc",
            "src/colmap/mvs/patch_match_cuda_owned.h",
            "src/colmap/mvs/patch_match_options.h",
        }
        self.assertTrue(required <= set(generator.CLOSURE))
        resident_spec = importlib.util.spec_from_file_location(
            "resident_generator",
            Path(__file__).with_name(
                "generate_resident_allocation_fingerprint.py"
            ),
        )
        resident = importlib.util.module_from_spec(resident_spec)
        resident_spec.loader.exec_module(resident)
        self.assertTrue(required <= set(resident.CLOSURE))

    def test_committed_closure_recomputed_and_dirty_missing_refuse(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            def git(*args):
                return subprocess.check_output(
                    ["git", "-C", tmp, *args], stderr=subprocess.DEVNULL
                )

            git("init", "-q")
            # The hot cache header exists even if omitted from the closure.
            cache_header = "src/colmap/mvs/owned_source_cache.h"
            for path in (*generator.CLOSURE, cache_header):
                target = root / path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(b"TEST-EVIDENCE\n")
            git("add", "src")
            git(
                "-c",
                "user.name=Test",
                "-c",
                "user.email=test@example.invalid",
                "commit",
                "-qm",
                "fixture",
            )
            # TEST-EVIDENCE ancestry base for this isolated synthetic repo.
            lineage = git("rev-parse", "HEAD").decode().strip()
            patcher = mock.patch.object(
                generator, "LINEAGE", lineage, create=True
            )
            patcher.start()
            self.addCleanup(patcher.stop)
            output = root / "generated.h"
            generator.generate(root, output)
            # Independent encoding of this deliberately uniform fixture.
            encoded = b"".join(
                path.encode() + b"\0TEST-EVIDENCE\n\n"
                for path in generator.CLOSURE
            )
            self.assertIn(
                hashlib.sha256(encoded).hexdigest(), output.read_text()
            )
            self.assertEqual(len(generator.CLOSURE), 28)
            self.assertIn(
                f'kPhotoPhaseV1SourceCommit[] = "{lineage}"', output.read_text()
            )
            # Non-closure commits still require a new per-build attestation.
            first = output.read_text()
            git(
                "-c",
                "user.name=Test",
                "-c",
                "user.email=test@example.invalid",
                "commit",
                "--allow-empty",
                "-qm",
                "same closure, new revision",
            )
            generator.generate(root, output)
            revision = git("rev-parse", "HEAD^{commit}").decode().strip()
            self.assertIn(
                f'kPhotoPhaseV1SourceCommit[] = "{revision}"',
                output.read_text(),
            )
            self.assertIn(
                hashlib.sha256(encoded).hexdigest(), output.read_text()
            )
            self.assertNotEqual(first, output.read_text())
            self.assertIn(f'"{lineage}";', output.read_text())
            original = output.read_text()
            source = root / cache_header
            source.write_bytes(b"changed\n")
            with self.assertRaisesRegex(ValueError, cache_header):
                generator.generate(root, output)
            self.assertEqual(original, output.read_text())
            git("add", "src")
            git(
                "-c",
                "user.name=Test",
                "-c",
                "user.email=test@example.invalid",
                "commit",
                "-qm",
                "new closure",
            )
            generator.generate(root, output)
            self.assertNotEqual(original, output.read_text())
            self.assertNotIn(
                hashlib.sha256(encoded).hexdigest(), output.read_text()
            )
            source.unlink()
            with self.assertRaises(FileNotFoundError):
                generator.generate(root, output)
            source.write_bytes(b"changed\n")
            # A valid commit with the same tree, but no shared ancestry, must
            # not be labeled as this build's required lineage.
            unrelated = (
                git(
                    "-c",
                    "user.name=Test",
                    "-c",
                    "user.email=test@example.invalid",
                    "commit-tree",
                    git("rev-parse", "HEAD^{tree}").decode().strip(),
                    "-m",
                    "unrelated TEST-EVIDENCE",
                )
                .decode()
                .strip()
            )
            before = output.read_text()
            with mock.patch.object(generator, "LINEAGE", unrelated):
                with self.assertRaisesRegex(ValueError, "lineage.*ancestor"):
                    generator.generate(root, output)
            self.assertEqual(before, output.read_text())


if __name__ == "__main__":
    unittest.main()
