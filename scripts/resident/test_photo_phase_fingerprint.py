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
    def test_graph_api_signatures_preserve_default_edges(self):
        # Compile actual call sites against versioned API shapes, not an SDK.
        root = Path(__file__).resolve().parents[2]
        source = (root / "src/colmap/mvs/owned_graph.cu").read_text()
        node_call = source.split("      cudaGraphNode_t node;", 1)[1].split(
            "      body =", 1
        )[0]
        edge_call = source.split(
            "    for (const auto& edge : topology.dependencies) {", 1
        )[1].split("\n  } catch", 1)[0]
        code = (
            r"""
#include <cassert>
#include <stdexcept>
#include <utility>
#include <vector>
using cudaGraph_t = int;
using cudaGraphNode_t = int;
struct cudaGraphNodeParams {};
struct cudaGraphEdgeData {};
static cudaGraphNodeParams* expected_params;
static int node_calls = 0, edge_calls = 0;
static int fail_node = 0, fail_edge = 0;
void Check(int error) {
  if (error) throw std::runtime_error("injected CUDA error");
}
int cudaGraphAddNode(cudaGraphNode_t* node, cudaGraph_t graph,
                     const cudaGraphNode_t* dependencies,
#if CUDART_VERSION >= 13000
                     const cudaGraphEdgeData* data,
#endif
                     size_t count, cudaGraphNodeParams* params) {
  ++node_calls;
  assert(graph == 7 && !dependencies && count == 0);
  assert(params == expected_params);
#if CUDART_VERSION >= 13000
  assert(!data); // Default/zeroed edge data, not a changed dependency type.
#endif
  *node = 17;
  return fail_node;
}
int cudaGraphAddDependencies(cudaGraph_t graph,
                             const cudaGraphNode_t* from,
                             const cudaGraphNode_t* to,
#if CUDART_VERSION >= 13000
                             const cudaGraphEdgeData* data,
#endif
                             size_t count) {
  assert(graph == 7 && count == 1);
#if CUDART_VERSION >= 13000
  assert(!data);
#endif
  assert(*from == (edge_calls == 0 ? 12 : 10));
  assert(*to == (edge_calls == 0 ? 10 : 11));
  return ++edge_calls == fail_edge ? 1 : 0;
}
void AddConditional() {
  cudaGraph_t graph_ = 7;
  cudaGraphNodeParams condition{};
  expected_params = &condition;
  cudaGraphNode_t node;
"""
            + node_call
            + r"""
  assert(node == 17);
}
void AddEdges() {
  cudaGraph_t graph = 7;
  std::vector<cudaGraphNode_t> children{10, 11, 12};
  struct {
    std::vector<std::pair<size_t, size_t>> dependencies{{2, 0}, {0, 1}};
  } topology;
  for (const auto& edge : topology.dependencies) {
"""
            + edge_call
            + r"""
}
int main() {
  AddConditional(); assert(node_calls == 1);
  AddEdges(); assert(edge_calls == 2);
  fail_node = 1;
  bool refused = false;
  try { AddConditional(); } catch (const std::runtime_error&) {
    refused = true;
  }
  assert(refused && node_calls == 2);
  for (fail_edge = 1; fail_edge <= 2; ++fail_edge) {
    edge_calls = 0; refused = false;
    try { AddEdges(); } catch (const std::runtime_error&) {
      refused = true;
    }
    assert(refused && edge_calls == fail_edge);
  }
}
"""
        )
        with tempfile.TemporaryDirectory(dir=root.parent) as directory:
            test = Path(directory) / "graph.cc"
            binary = Path(directory) / "graph"
            test.write_text(code)
            for version in (12080, 12090, 13000, 13020):
                with self.subTest(cudart_version=version):
                    subprocess.run(
                        [
                            "c++",
                            "-std=c++17",
                            "-Wall",
                            "-Wextra",
                            "-Werror",
                            f"-DCUDART_VERSION={version}",
                            str(test),
                            "-o",
                            str(binary),
                        ],
                        check=True,
                    )
                    subprocess.run([str(binary)], check=True)

    def test_graph_allocation_and_host_boundary_sources_are_attested(self):
        required = {
            "src/colmap/mvs/owned_input_bank.h",
            "src/colmap/mvs/owned_photo_output.h",
            "src/colmap/mvs/owned_photo_output_state.h",
            "src/colmap/mvs/owned_geo_photo_input.h",
            "src/colmap/mvs/owned_geo_photo_input.cu",
            "src/colmap/mvs/owned_stereo_admission.h",
            "src/colmap/mvs/owned_stereo_admission.cc",
            "src/colmap/exe/fixed_cohort.h",
            "src/colmap/exe/fixed_cohort.cc",
            "src/colmap/mvs/owned_fixed_cohort_budget.h",
            "src/colmap/mvs/owned_fixed_cohort_inventory.h",
            "src/colmap/mvs/owned_fixed_cohort_pipeline.h",
            "src/colmap/mvs/owned_fixed_cohort_pipeline.cu",
            "src/colmap/mvs/multi_gpu_phase_plan.h",
            "src/colmap/mvs/multi_gpu_phase_plan.cc",
            "src/colmap/mvs/multi_gpu_workers.h",
            "src/colmap/mvs/multi_gpu_fixed_cohort.h",
            "src/colmap/mvs/multi_gpu_fixed_cohort.cc",
            "src/colmap/mvs/multi_gpu_fixed_cohort.cu",
            "src/colmap/exe/multi_gpu_plan_json.h",
            "src/colmap/mvs/owned_fixed_cohort_validation.h",
            "src/colmap/mvs/owned_fixed_cohort_validation.cc",
            "src/colmap/mvs/owned_fixed_cohort_workspace.h",
            "src/colmap/mvs/owned_fixed_cohort_workspace.cc",
            "src/colmap/mvs/owned_pair_run.h",
            "src/colmap/mvs/native_run_deadline.h",
            "src/colmap/mvs/owned_rotation_memory.h",
            "src/colmap/mvs/owned_resident_minima.h",
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
            "src/colmap/mvs/gpu_mat_prng.h",
            "src/colmap/mvs/owned_selection_boundary.h",
            "src/colmap/mvs/patch_match.h",
            "src/colmap/mvs/patch_match.cc",
            "src/colmap/mvs/patch_match_cuda_owned.h",
            "src/colmap/mvs/patch_match_options.h",
            "src/colmap/mvs/optimized_photo_options.h",
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
            self.assertEqual(len(generator.CLOSURE), 59)
            self.assertTrue(
                {
                    "src/colmap/mvs/owned_strip_storage.h",
                    "src/colmap/mvs/owned_strip_storage.cc",
                    "src/colmap/mvs/owned_strip_view.h",
                }
                <= set(generator.CLOSURE)
            )
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
            for new_header in (
                "owned_photo_output.h",
                "owned_photo_output_state.h",
                "owned_geo_photo_input.h",
                "owned_geo_photo_input.cu",
                "owned_fixed_cohort_budget.h",
                "owned_fixed_cohort_inventory.h",
                "owned_fixed_cohort_pipeline.h",
                "owned_fixed_cohort_pipeline.cu",
                "owned_fixed_cohort_validation.h",
                "owned_fixed_cohort_validation.cc",
                "owned_fixed_cohort_workspace.h",
                "owned_fixed_cohort_workspace.cc",
            ):
                added = root / "src/colmap/mvs" / new_header
                added.write_bytes(b"dirty snapshot allocation\n")
                with self.assertRaisesRegex(ValueError, new_header):
                    generator.generate(root, output)
                self.assertEqual(original, output.read_text())
                added.write_bytes(b"TEST-EVIDENCE\n")
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
