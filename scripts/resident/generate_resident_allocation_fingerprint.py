#!/usr/bin/env python3
"""Generate the compiled resident allocation-closure fingerprint header.

Build-time generator for the slice-B resident runner. It computes, at build
time, the SHA-256 over the FROZEN ORDERED allocation closure declared below
and writes the generated header consumed by the product CLI's resident
forecast gate (`kResidentAllocationFingerprint`).

Algorithm (frozen; matches the agreed field contract):
    digest = SHA-256()
    for entry in CLOSURE (in listed order):
        digest.update(entry_path.encode('utf-8'))
        digest.update(b'\\x00')
        digest.update(file bytes of entry)
        digest.update(b'\\x00')
    fingerprint = lowercase hex digest

Frozen properties:
  * CLOSURE is an explicit stable ordered list of repository-relative paths
    (forward slashes) -- never a glob, never absolute paths, never derived
    from the filesystem;
  * the generated output header is EXCLUDED from the closure, so the digest
    cannot recurse into itself;
  * no self-referential SHA, no source-commit embedding, no registry, and
    no compatibility decision belong here: the versioned external registry
    that performs compatibility selection lives only in the canonical
    Python (photogram) tooling; the native side compares the wire
    fingerprints against THIS compiled digest only;
  * the generator fails closed on any missing/unreadable closure entry;
  * build systems must ensure regeneration whenever any closure entry
    changes: this script emits a DEPFILE with every entry (exact under the
    Ninja generator; for non-depfile generators pass every entry as an
    explicit DEPENDS in the invoking build rule);
  * the header is regenerated (written only when content changes) and is
    never committed, never hand-edited; no digest may be invented anywhere
    else.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path, PurePosixPath

# FROZEN ORDERED ALLOCATION CLOSURE (colmap source repository-relative,
# forward slashes). Order is part of the digest contract. Contents:
# the runner / wire / gates / controller / options / bank / paired-
# coordinator / rolling owner loop / bounded normal queue / private-minima
# helper, plus the actual CUDA / warp / rescale / storage / map-chain
# dependencies and the standard CLI surface the runner dispatches through.
# FOCUSED DEPENDENCY AUDIT (Sol 3cdc fix 6; 2026-10-15): all transitive
# allocation-relevant colmap/VLFeat sources of the resident path were
# traced from the closure roots (runner/wire/gates/controller/bank/
# coordinator/queue) through their direct includes. Added this round with
# rationale, in dependency order:
#   sensor/bitmap.{h,cc}: grayscale decode/storage path (bank inputs);
#   mvs/mat_read_internal.h + util endian/file/string reading chain is
#     reached via mat.cc; mat_read_internal.h is the payload decoder;
#   mvs/workspace.{h,cc}, mvs/model.{h,cc}: resolved invocation
#     workspace/model construction consumed by the controller;
#   thirdparty/VLFeat/imopv.{h,c} + generic.{h,c}: the smoothing/
#     convolution allocation implementation and allocator wrappers that
#     image/warp.cc routes through;
#   mvs/owned_pair_validation.h: checked accounting helpers consumed
#     directly by the runner/minima;
#   mvs/resident_contract.{h,cc}: the contract codec governing how bound
#     input identities/records are captured, revalidated and interpreted;
#   mvs/resident_cohorts.h: the typed FIFO cohort validated-plan boundary
#     (structural/admission arithmetic validation plus the explicit
#     planner-MAX vs rolling-SUM records any per-cohort allocation decision
#     consumes), built on the planner/bank/coordinator inputs already
#     listed above;
#   mvs/resident_wire_lexical.h: the strict forecast-wire lexical
#     primitives shared by the full-bank and cohort wire parsers (moved
#     from resident_forecast.cc; used code paths behave identically, with
#     one repaired previously-DEAD helper, the exact 64-lowercase-hex
#     digest-text rule the cohort scope requires);
#   mvs/resident_cohort_wire.{h,cc}: the distinct resident-cohort-forecast
#     v1 wire parser + per-cohort admission-unit gates
#     (binding/parameters/hard-FitsCaps): they run BEFORE any payload and
#     are part of the per-cohort allocation decision path;
#   mvs/resident_cohort_wire_minima.cc: the per-cohort NATIVE charge
#     floors + singleton feasibility gate (allowance-inclusive bank
#     charge before the single canonical margin, exact normal allowances,
#     private floors) around never-recaptured original identities;
#   mvs/resident_cohort_runner.h: the explicit cohort EXECUTOR (gate
#     sequence strictly before any payload + strictly sequential
#     per-cohort loop over the concrete banks/pipeline/coordinator with
#     original identities and the wire ledger composition).
# Fusion allocation/admission audit (2026-10-31): the same binding now
# certifies mapped CPU fusion as well as stereo. Include the fusion roots,
# checked input/output ledger, mapped storage and dependency admission.
# Their payload dependencies (bitmap, mat, workspace/model and fixed-cohort
# validation) are already above. Also bind PlyPoint layout, aligned vector
# storage, task scheduling/draining (buffer coexistence), and OIIO thread
# initialization. Decoder internals and incidental controller/traversal/
# metadata overhead remain explicit policy allowances, not certified peaks.
# Deliberately EXCLUDED (documented): the stock CUDA backend
# (patch_match_cuda.{h,cc}) never constructed on the owned path,
# consistency_graph (explicitly refused by option admission), test/
# benchmark sources, and external dependency internals (e.g., OpenImageIO
# itself), which no source digest can certify.
CLOSURE = (
    "src/colmap/mvs/patch_match_options.h",
    "src/colmap/mvs/patch_match_options.cc",
    "src/colmap/mvs/owned_tile.h",
    "src/colmap/mvs/owned_source_cache.h",
    "src/colmap/mvs/owned_texture_row_batch.h",
    "src/colmap/mvs/owned_cuda_execution.h",
    "src/colmap/mvs/owned_rotation_storage.h",
    "src/colmap/mvs/owned_selection_scratch.h",
    "src/colmap/mvs/gpu_mat.h",
    "src/colmap/mvs/gpu_mat_prng.h",
    "src/colmap/mvs/gpu_mat_prng.cu",
    "src/colmap/mvs/owned_prng_seed.h",
    "src/colmap/mvs/owned_prng.h",
    "src/colmap/mvs/owned_prng_boundary.h",
    "src/colmap/mvs/gpu_mat_ref_image.h",
    "src/colmap/mvs/gpu_mat_ref_image.cu",
    "src/colmap/sensor/bitmap.h",
    "src/colmap/sensor/bitmap.cc",
    "src/colmap/mvs/mat.h",
    "src/colmap/mvs/mat.cc",
    "src/colmap/mvs/mat_read_internal.h",
    "src/colmap/mvs/depth_map.h",
    "src/colmap/mvs/depth_map.cc",
    "src/colmap/mvs/normal_map.h",
    "src/colmap/mvs/normal_map.cc",
    "src/colmap/mvs/image.h",
    "src/colmap/mvs/image.cc",
    "src/thirdparty/VLFeat/generic.h",
    "src/thirdparty/VLFeat/generic.c",
    "src/thirdparty/VLFeat/imopv.h",
    "src/thirdparty/VLFeat/imopv.c",
    "src/colmap/image/warp.h",
    "src/colmap/image/warp.cc",
    "src/colmap/mvs/owned_input_bank.h",
    "src/colmap/mvs/owned_photo_output.h",
    "src/colmap/mvs/owned_geo_photo_input.h",
    "src/colmap/mvs/owned_geo_photo_input.cu",
    "src/colmap/mvs/owned_fixed_cohort_budget.h",
    "src/colmap/mvs/owned_fixed_cohort_inventory.h",
    "src/colmap/mvs/owned_fixed_cohort_pipeline.h",
    "src/colmap/mvs/owned_fixed_cohort_pipeline.cu",
    "src/colmap/mvs/owned_fixed_cohort_validation.h",
    "src/colmap/mvs/owned_fixed_cohort_validation.cc",
    "src/colmap/mvs/owned_fixed_cohort_workspace.h",
    "src/colmap/mvs/owned_fixed_cohort_workspace.cc",
    "src/colmap/mvs/owned_photo_output_state.h",
    "src/colmap/mvs/owned_input_bank.cc",
    "src/colmap/mvs/owned_input_bank.cu",
    "src/colmap/mvs/cuda_texture.h",
    "src/colmap/mvs/workspace.h",
    "src/colmap/mvs/workspace.cc",
    "src/colmap/mvs/model.h",
    "src/colmap/mvs/model.cc",
    "src/colmap/mvs/owned_pair_validation.h",
    "src/colmap/mvs/owned_pair_coordinator.h",
    "src/colmap/mvs/native_run_deadline.h",
    "src/colmap/mvs/owned_pair_run.h",
    "src/colmap/mvs/owned_normal_lookahead.h",
    "src/colmap/mvs/patch_match.h",
    "src/colmap/mvs/patch_match.cc",
    "src/colmap/mvs/patch_match_cuda_owned.h",
    "src/colmap/mvs/patch_match_cuda_owned.cu",
    "src/colmap/mvs/resident_contract.h",
    "src/colmap/mvs/resident_contract.cc",
    "src/colmap/mvs/owned_rotation_memory.h",
    "src/colmap/mvs/owned_resident_minima.h",
    "src/colmap/mvs/resident_forecast.h",
    "src/colmap/mvs/resident_forecast.cc",
    "src/colmap/mvs/resident_forecast_minima.cc",
    "src/colmap/mvs/resident_cohorts.h",
    "src/colmap/mvs/resident_wire_lexical.h",
    "src/colmap/mvs/resident_cohort_wire.h",
    "src/colmap/mvs/resident_cohort_wire.cc",
    "src/colmap/mvs/resident_cohort_wire_minima.cc",
    "src/colmap/mvs/resident_cohort_runner.h",
    "src/colmap/mvs/resident_runner.h",
    "src/colmap/mvs/optimized_dispatch.h",
    "src/colmap/mvs/optimized_receipt.h",
    "src/colmap/mvs/optimized_photo_forecast.h",
    "src/colmap/mvs/optimized_photo_options.h",
    "src/colmap/mvs/optimized_photo_paths.h",
    "src/colmap/mvs/optimized_photo_reservation.h",
    "src/colmap/mvs/optimized_photo_runner.h",
    "src/colmap/mvs/owned_pair_image.h",
    "src/colmap/exe/mvs.cc",
    # Fixed-cohort observation/cap selection is an allocation admission root.
    "src/colmap/exe/fixed_cohort.h",
    "src/colmap/exe/fixed_cohort.cc",
    "src/colmap/exe/optimized_dense_plan.h",
    "src/colmap/exe/optimized_dense_plan.cc",
    "src/colmap/exe/optimized_dense_plan_io.h",
    "src/colmap/exe/optimized_dense_plan_options.h",
    "src/colmap/mvs/owned_graph.h",
    "src/colmap/mvs/owned_graph.cu",
    "src/colmap/mvs/owned_graph_topology.h",
    "src/colmap/mvs/owned_graph_coordinator.h",
    "src/colmap/mvs/owned_async_pool.h",
    "src/colmap/mvs/owned_selection_boundary.h",
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
    # Undistortion admission, actual camera geometry/branch allocation,
    # header revalidation and owned worker drain. Bitmap/warp/threading/OIIO
    # initialization and native host observation are already bound above.
    "src/colmap/controllers/undistorters.h",
    "src/colmap/controllers/undistorters.cc",
    "src/colmap/controllers/undistortion_resource.h",
    "src/colmap/image/undistortion.h",
    "src/colmap/image/undistortion.cc",
    "src/colmap/scene/camera.h",
    "src/colmap/scene/camera.cc",
    "src/colmap/scene/reconstruction.h",
    "src/colmap/scene/reconstruction.cc",
    "src/colmap/scene/reconstruction_io.h",
    "src/colmap/scene/reconstruction_io.cc",
    "src/colmap/scene/reconstruction_io_binary.h",
    "src/colmap/scene/reconstruction_io_binary.cc",
    "src/colmap/scene/reconstruction_io_text.h",
    "src/colmap/scene/reconstruction_io_text.cc",
    "src/colmap/scene/reconstruction_io_utils.h",
    "src/colmap/scene/reconstruction_io_utils.cc",
    "src/colmap/scene/image.h",
    "src/colmap/scene/image.cc",
    "src/colmap/scene/frame.h",
    "src/colmap/scene/frame.cc",
    "src/colmap/sensor/rig.h",
    "src/colmap/sensor/rig.cc",
    "src/colmap/util/base_controller.h",
    "src/colmap/util/base_controller.cc",
    "src/colmap/util/hash_containers.h",
    "src/colmap/util/types.h",
    "src/colmap/sensor/models.h",
    "src/colmap/sensor/models.cc",
    "src/colmap/sensor/models/base.h",
    "src/colmap/sensor/models/runtime.h",
    "src/colmap/sensor/models/pinhole.h",
    "src/colmap/sensor/models/radial.h",
    "src/colmap/sensor/models/opencv.h",
    "src/colmap/sensor/models/fov.h",
    "src/colmap/sensor/models/division.h",
    "src/colmap/sensor/models/eucm.h",
    "src/colmap/sensor/models/fisheye.h",
    "src/colmap/sensor/models/fisheye_radial.h",
    "src/colmap/sensor/models/thin_prism.h",
    "src/colmap/sensor/models/spherical.h",
    "src/colmap/sensor/models/jacobian.h",
    "src/colmap/math/math.h",
    "src/colmap/util/file.h",
    "src/colmap/util/file.cc",
    "src/colmap/util/cancellation.h",
    "src/colmap/util/cancellation.cc",
    "src/colmap/exe/image.cc",
    # Main owns the signal state used to cancel and drain admitted workers.
    "src/colmap/exe/colmap.cc",
    # Feature extraction: header admission, queue payload ownership, checked
    # SIFT scratch/output growth, writer serialization and backend allocation
    # implementations. Library workspace is policy UNKNOWN, not a certified
    # bound; binding its source is NOT a physical memory model.
    "src/colmap/controllers/feature_extraction.h",
    "src/colmap/controllers/feature_extraction.cc",
    "src/colmap/controllers/extraction_resource.h",
    "src/colmap/controllers/image_reader.h",
    "src/colmap/controllers/image_reader.cc",
    "src/colmap/controllers/option_manager.h",
    "src/colmap/controllers/option_manager.cc",
    "src/colmap/feature/extractor.h",
    "src/colmap/feature/extractor.cc",
    "src/colmap/feature/sift.h",
    "src/colmap/feature/sift.cc",
    "src/colmap/feature/types.h",
    "src/colmap/feature/types.cc",
    "src/colmap/feature/utils.h",
    "src/colmap/feature/utils.cc",
    "src/colmap/scene/database.h",
    "src/colmap/scene/database.cc",
    "src/colmap/scene/database_sqlite.cc",
    "src/colmap/exe/feature.cc",
    "src/colmap/util/cuda.h",
    "src/colmap/util/cuda.cc",
    "src/colmap/util/opengl_utils.h",
    "src/colmap/util/opengl_utils.cc",
    "src/thirdparty/VLFeat/sift.h",
    "src/thirdparty/VLFeat/sift.c",
    "src/thirdparty/VLFeat/covdet.h",
    "src/thirdparty/VLFeat/covdet.c",
    "src/thirdparty/VLFeat/scalespace.h",
    "src/thirdparty/VLFeat/scalespace.c",
    "src/thirdparty/VLFeat/mathop.h",
    "src/thirdparty/VLFeat/mathop.c",
    "src/thirdparty/VLFeat/host.h",
    "src/thirdparty/VLFeat/host.c",
    "src/thirdparty/VLFeat/random.h",
    "src/thirdparty/VLFeat/random.c",
    "src/thirdparty/VLFeat/stringop.h",
    "src/thirdparty/VLFeat/stringop.c",
    "src/thirdparty/SiftGPU/FrameBufferObject.h",
    "src/thirdparty/SiftGPU/FrameBufferObject.cpp",
    "src/thirdparty/SiftGPU/GlobalUtil.h",
    "src/thirdparty/SiftGPU/GlobalUtil.cpp",
    "src/thirdparty/SiftGPU/GLTexImage.h",
    "src/thirdparty/SiftGPU/GLTexImage.cpp",
    "src/thirdparty/SiftGPU/ProgramGLSL.h",
    "src/thirdparty/SiftGPU/ProgramGLSL.cpp",
    "src/thirdparty/SiftGPU/ProgramGPU.h",
    "src/thirdparty/SiftGPU/PyramidGL.h",
    "src/thirdparty/SiftGPU/PyramidGL.cpp",
    "src/thirdparty/SiftGPU/ShaderMan.h",
    "src/thirdparty/SiftGPU/ShaderMan.cpp",
    "src/thirdparty/SiftGPU/SiftGPU.h",
    "src/thirdparty/SiftGPU/SiftGPU.cpp",
    "src/thirdparty/SiftGPU/SiftPyramid.h",
    "src/thirdparty/SiftGPU/SiftPyramid.cpp",
    "src/thirdparty/SiftGPU/CuTexImage.h",
    "src/thirdparty/SiftGPU/CuTexImage.cpp",
    "src/thirdparty/SiftGPU/ProgramCU.cu",
    "src/thirdparty/SiftGPU/ProgramCU.h",
    "src/thirdparty/SiftGPU/PyramidCU.h",
    "src/thirdparty/SiftGPU/PyramidCU.cpp",
    "scripts/resident/generate_resident_allocation_fingerprint.py",
)

HEADER_COMMENT = """// GENERATED FILE - DO NOT EDIT.
// Regenerated at build time by scripts/resident/generate_resident_
// allocation_fingerprint.py from the frozen ordered allocation closure.
// The generated header is EXCLUDED from the closure (no self-reference).
// Fingerprint = SHA-256 over, in frozen listed order, for every entry:
// utf8(entry path) + NUL + file bytes + NUL. The versioned compatibility
// registry lives only in the canonical Python tooling; the native side
// compares the wire fingerprints to kResidentAllocationFingerprint only.
"""


def check_entry(name: str) -> None:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or not str(path) == name:
        raise SystemExit(
            "closure entries must be repo-relative forward-slash paths: "
            f"{name!r}"
        )
    if not name.endswith((".h", ".c", ".cc", ".cpp", ".cu", ".py")):
        raise SystemExit(
            f"closure entries must be source or generator files: {name!r}"
        )


def compute_fingerprint(repo_root: Path) -> str:
    digest = hashlib.sha256()
    for name in CLOSURE:
        check_entry(name)
        entry = repo_root / name
        if not entry.is_file():
            raise SystemExit(
                "closure entry is missing or not a regular file: "
                f"{name!s} (repo root {repo_root!s}); the closure is frozen "
                "so the build fails closed rather than hashing a stale tree"
            )
        content = entry.read_bytes()
        digest.update(name.encode("utf-8"))
        digest.update(b"\x00")
        digest.update(content)
        digest.update(b"\x00")
    return digest.hexdigest()


def render_header(fingerprint: str) -> str:
    lines = [HEADER_COMMENT.rstrip("\n")]
    lines.append("#pragma once")
    lines.append("")
    lines.append("namespace colmap {")
    lines.append("namespace mvs {")
    lines.append("")
    lines.append(
        "// Compiled allocation-closure fingerprint of the exact source inputs"
    )
    lines.append(
        "// this binary was built with (see the header comment above)."
    )
    lines.append(
        "// Consumers MUST compare the wire fingerprints against this value and"
    )
    lines.append(
        "// MUST NOT accept any substitute digest, placeholder or registry."
    )
    lines.append("inline constexpr char kResidentAllocationFingerprint[] =")
    lines.append(f'    "{fingerprint!s}";')
    lines.append("")
    lines.append("}  // namespace mvs")
    lines.append("}  // namespace colmap")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate the compiled resident allocation fingerprint "
        "header from the frozen ordered closure."
    )
    parser.add_argument(
        "--repo-root", required=True, help="COLMAP source repository root"
    )
    parser.add_argument(
        "--output", required=True, help="generated header output path"
    )
    parser.add_argument(
        "--depfile",
        required=True,
        help="depfile path listing every closure entry",
    )
    args = parser.parse_args()

    repo_root = Path(args.repo_root)
    if not repo_root.is_dir():
        raise SystemExit(f"repo root is not a directory: {args.repo_root!s}")

    fingerprint = compute_fingerprint(repo_root)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    header = render_header(fingerprint)
    # Write only when content changes so unchanged builds do not churn.
    current = output.read_text() if output.exists() else None
    if current != header:
        temporary = output.with_name(output.name + ".tmp")
        temporary.write_text(header)
        temporary.replace(output)

    depfile = Path(args.depfile)
    depfile.parent.mkdir(parents=True, exist_ok=True)
    # Absolute paths keep the depfile valid from any build working dir.
    targets = " ".join(
        str((repo_root / PurePosixPath(name)).resolve()) for name in CLOSURE
    )
    depfile.write_text(f"{output.resolve()!s}: {targets!s}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
