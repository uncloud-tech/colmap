#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""Recompute PHOTO's ordered committed-source closure for the compiled adapter.

No declared digest is accepted as input. Working bytes must equal committed
bytes, so a dirty allocation closure cannot compile under a stale attestation.
Run on every build (including HEAD changes); write only when content changes.
Encoding matches photogram.hardware.memory_model._fingerprint_digest exactly.
"""

import argparse
import hashlib
import subprocess
from pathlib import Path

# Required ancestry base, never substituted with the current build revision.
LINEAGE = "cf7891059c402c333b84b752de3acb8b084abdad"

CLOSURE = (
    "src/colmap/mvs/optimized_photo_forecast.h",
    "src/colmap/mvs/optimized_photo_options.h",
    "src/colmap/mvs/optimized_photo_paths.h",
    "src/colmap/mvs/optimized_photo_runner.h",
    "src/colmap/mvs/owned_rotation_storage.h",
    "src/colmap/mvs/owned_selection_scratch.h",
    "src/colmap/mvs/owned_source_cache.h",
    "src/colmap/mvs/owned_prng_boundary.h",
    "src/colmap/mvs/owned_prng.h",
    "src/colmap/mvs/patch_match_cuda_owned.cu",
    "src/colmap/mvs/owned_pair_coordinator.h",
    "src/colmap/mvs/owned_pair_validation.h",
    # Opt-in graph/pool allocation and the common host preparation boundary.
    "src/colmap/mvs/owned_input_bank.h",
    "src/colmap/mvs/owned_photo_output.h",
    "src/colmap/mvs/owned_geo_photo_input.h",
    "src/colmap/mvs/owned_geo_photo_input.cu",
    "src/colmap/mvs/owned_fixed_cohort_budget.h",
    "src/colmap/mvs/owned_fixed_cohort_pipeline.h",
    "src/colmap/mvs/owned_fixed_cohort_pipeline.cu",
    "src/colmap/mvs/owned_fixed_cohort_validation.h",
    "src/colmap/mvs/owned_fixed_cohort_validation.cc",
    "src/colmap/mvs/owned_fixed_cohort_workspace.h",
    "src/colmap/mvs/owned_fixed_cohort_workspace.cc",
    "src/colmap/mvs/owned_pair_run.h",
    "src/colmap/mvs/native_run_deadline.h",
    "src/colmap/mvs/owned_rotation_memory.h",
    "src/colmap/mvs/owned_resident_minima.h",
    "src/colmap/mvs/owned_photo_output_state.h",
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
)


def generate(root, output):
    def git(*args):
        return subprocess.check_output(["git", "-C", str(root), *args])

    revision = git("rev-parse", "HEAD^{commit}").decode().strip()
    ancestry = subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "merge-base",
            "--is-ancestor",
            LINEAGE,
            revision,
        ],
        capture_output=True,
    )
    if ancestry.returncode != 0:
        raise ValueError(
            f"PHOTO lineage {LINEAGE} is not a verified ancestor "
            f"of source_commit {revision}"
        )
    digest = hashlib.sha256()
    for path in CLOSURE:
        data = git("show", f"{revision}:{path}")
        if (root / path).read_bytes() != data:
            raise ValueError(
                f"PHOTO compiled closure differs from committed source: {path}"
            )
        digest.update(path.encode() + b"\0" + data + b"\n")
    content = (
        "// SPDX-License-Identifier: BSD-3-Clause\n"
        "// Generated from committed PHOTO closure; do not edit.\n"
        "#pragma once\nnamespace colmap::mvs {\n"
        "// Required ancestry base; fingerprint covers SourceCommit below.\n"
        "inline constexpr char kPhotoPhaseV1Lineage[] =\n"
        f'    "{LINEAGE}";\n'
        'inline constexpr char kPhotoPhaseV1Fingerprint[] = "'
        f'{digest.hexdigest()}";\n'
        f'inline constexpr char kPhotoPhaseV1SourceCommit[] = "{revision}";\n'
        "}  // namespace colmap::mvs\n"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    if not output.exists() or output.read_text() != content:
        output.write_text(content)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    generate(args.repo_root, args.output)
