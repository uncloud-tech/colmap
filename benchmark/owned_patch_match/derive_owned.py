# SPDX-License-Identifier: BSD-3-Clause
"""One-time stock-derived sibling creation. Never edits the stock hot TU."""

import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MVS = ROOT / "src/colmap/mvs"
BASE = "c6ab4f897c94f10b15bdecc442dcfce4688f1b76"


def base(name):
    return subprocess.check_output(
        ["git", "-C", str(ROOT), "show", f"{BASE}:src/colmap/mvs/{name}"],
        text=True,
    )


def once(text, old, new):
    assert text.count(old) == 1, (old, text.count(old))
    return text.replace(old, new)


header = base("patch_match_cuda.h")
header = once(
    header,
    "#pragma once",
    (
        "#pragma once\n"
        "\n"
        "#if !defined(COLMAP_HIP_ENABLED)\n"
        '#include "colmap/mvs/owned_cuda_execution.h"\n'
        '#include "colmap/mvs/patch_match_calibration.h"'
    ),
)
header = once(header, "namespace mvs {", "namespace mvs {\nnamespace owned {")
header = once(header, "  void Run();", "  ~PatchMatchCuda();\n\n  void Run();")
header = once(
    header,
    "  void ComputeCudaConfig();",
    (
        "  ReferenceCalibration CurrentCalibration() const {\n"
        "    return SnapshotReferenceCalibration(ref_K_host_[rotation"
        "_in_half_pi_],\n"
        "                                         ref_inv_K_host_[rot"
        "ation_in_half_pi_]);\n"
        "  }\n"
        "\n"
        "  void ComputeCudaConfig();"
    ),
)
header = once(
    header,
    "  const PatchMatch::Problem problem_;",
    (
        "  const PatchMatch::Problem problem_;\n"
        "\n"
        "  // Created after device selection; drained before resource"
        "s, destroyed last.\n"
        "  std::unique_ptr<OwnedCudaStream> stream_;"
    ),
)
header = once(
    header, "}  // namespace mvs", "}  // namespace owned\n}  // namespace mvs"
)
header += "\n#endif  // !defined(COLMAP_HIP_ENABLED)\n"

text = base("patch_match_cuda.cu")
text = once(
    text,
    '#include "colmap/mvs/patch_match_cuda.h"',
    (
        '#include "colmap/mvs/patch_match_cuda_owned.h"\n'
        '#include "colmap/mvs/patch_match_execution.h"'
    ),
)
text = once(text, "namespace mvs {", "namespace mvs {\nnamespace owned {")
text = once(
    text,
    (
        "// Calibration of reference image as {fx, cx, fy, cy}.\n"
        "__constant__ float ref_K[4];\n"
        "// Calibration of reference image as {1/fx, -cx/fx, 1/fy, -c"
        "y/fy}.\n"
        "__constant__ float ref_inv_K[4];"
    ),
    "// Calibration is supplied as a private by-value kernel launch snapshot.",
)
text = text.replace("ref_inv_K[", "calibration.inv_K[").replace(
    "ref_K[", "calibration.K["
)
# Add the same immutable calibration through each device helper
# and recursive call.
for name in [
    "GenerateRandomNormal",
    "PerturbNormal",
    "ComputePointAtDepth",
    "PropagateDepth",
    "ComposeHomography",
    "ComputeGeomConsistencyCost",
]:
    text = text.replace(name + "(", name + "(calibration, ")
    pattern = r"(__device__ inline (?:void|float) " + name + r"\()calibration, "
    text, n = re.subn(
        pattern, r"\1const ReferenceCalibration& calibration, ", text
    )
    assert n == 1, (name, n)
for name in ["InitNormalMap", "ComputeInitialCost", "SweepFromTopToBottom"]:
    text = once(
        text,
        "__global__ void " + name + "(",
        "__global__ void " + name + "(const ReferenceCalibration calibration, ",
    )
text = once(
    text,
    "  __device__ PhotoConsistencyCostComputer(\n",
    (
        "  const ReferenceCalibration calibration;\n"
        "\n"
        "  __device__ PhotoConsistencyCostComputer(\n"
        "      const ReferenceCalibration calibration,\n"
    ),
)
text = once(
    text,
    "      : local_ref_image(ref_image_texture),",
    (
        "      : calibration(calibration),\n"
        "        local_ref_image(ref_image_texture),"
    ),
)
assert text.count("pcc_computer(ref_image_texture,") == 2
text = text.replace(
    "pcc_computer(ref_image_texture,",
    "pcc_computer(calibration, ref_image_texture,",
)
# Keep all non-stream launch geometry unchanged.
text = text.replace(
    "<<<sweep_grid_size_, sweep_block_size_>>>",
    "<<<sweep_grid_size_, sweep_block_size_, 0, stream_->Get()>>>",
)
text = text.replace(
    "<<<elem_wise_grid_size_, elem_wise_block_size_>>>",
    "<<<elem_wise_grid_size_, elem_wise_block_size_, 0, stream_->Get()>>>",
)
text = once(
    text,
    "          cost_map_->View(),\n          depth_map_->View(),",
    (
        "          CurrentCalibration(), cost_map_->View(),\n"
        "          depth_map_->View(),"
    ),
)
text = once(
    text,
    "          global_workspace_->View(),",
    "          CurrentCalibration(), global_workspace_->View(),",
)
text = once(
    text,
    "        normal_map_->View(), rand_state_map_->View());",
    (
        "        CurrentCalibration(), normal_map_->View(), rand_stat"
        "e_map_->View());"
    ),
)
# Replace only the owned host dispatch; no unsupported configuration fallback.
start = text.index("void PatchMatchCuda::Run() {")
end = text.index("DepthMap PatchMatchCuda::GetDepthMap()", start)
text = (
    text[:start]
    + """PatchMatchCuda::~PatchMatchCuda() {
  if (stream_) {
    stream_->DrainBeforeDestruction();
  }
}

void PatchMatchCuda::Run() {
  stream_->Activate();
  RunWithWindowSizeAndStep<11, 1>();
}

"""
    + text[end:]
)
text = once(
    text,
    (
        "  SetBestCudaDevice(std::stoi(options_.gpu_index));\n"
        "  InitRefImage();\n"
        "  InitSourceImages();\n"
        "  InitTransforms();\n"
        "  InitWorkspaceMemory();"
    ),
    """  ValidatePatchMatchExecution(PatchMatchExecution::kOwnedCuda,
                              options_.window_radius, options_.window_step,
                              options_.geom_consistency, true);
  SetBestCudaDevice(std::stoi(options_.gpu_index));
  stream_ = std::make_unique<OwnedCudaStream>();
  try {
    InitRefImage();
    InitSourceImages();
    InitTransforms();
    InitWorkspaceMemory();
    stream_->Synchronize();
  } catch (...) {
    // This class's destructor does not run on constructor failure. Fence here
    // before member texture/array/buffer destructors can run.
    stream_->DrainBeforeDestruction();
    throw;
  }""",
)
text = text.replace("CUDA_SYNC_AND_CHECK();", "stream_->Synchronize();")
text = re.sub(
    r"\bCudaTimer (\w+);", r"OwnedCudaTimer \1(stream_->Get());", text
)
text = text.replace("->CopyToMat()", "->CopyToMat(stream_->Get())")
for name in [
    "GetDepthMap",
    "GetNormalMap",
    "GetSelProbMap",
    "GetConsistentImageIdxs",
]:
    pattern = r"(PatchMatchCuda::" + name + r"\(\) const \{)"
    text, n = re.subn(pattern, r"\1\n  stream_->Activate();", text)
    assert n == 1
text, n = re.subn(
    r"->Rotate\(([^;]+)\);", r"->Rotate(\1, stream_->Get());", text
)
assert n == 9, n
text, n = re.subn(
    r"->FillWithScalar\(([^;]+)\);",
    r"->FillWithScalar(\1, stream_->Get());",
    text,
)
assert n == 2, n
text = once(text, "*rand_state_map_);", "*rand_state_map_, stream_->Get());")
text = text.replace(
    "std::make_unique<GpuMatPRNG>(ref_width_, ref_height_)",
    "std::make_unique<GpuMatPRNG>(ref_width_, ref_height_, stream_->Get())",
)
text = once(
    text,
    "new GpuMatPRNG(width, height)",
    "new GpuMatPRNG(width, height, stream_->Get())",
)
text, n = re.subn(
    r"->CopyToDevice\(([^;]+)\);", r"->CopyToDevice(\1, stream_->Get());", text
)
assert n == 2, n
text, n = re.subn(
    r"->Filter\(([^;]+)\);", r"->Filter(\1, stream_->Get());", text
)
assert n == 1, n
for name, count in [("FromGpuMat", 1), ("FromHostArray", 3)]:
    text, n = re.subn(
        r"::" + name + r"\(([^;]+)\);",
        r"::" + name + r"(\1, stream_->Get());",
        text,
    )
    assert n == count, (name, n)
start = text.index("  // Bind 0 degrees version to constant global memory.")
end = text.index(
    (
        "  //////////////////////////////////////////////////////////"
        "////////////////////"
    ),
    start,
)
text = text[:start] + text[end:]
start = text.index("  // Rotate calibration.")
end = text.index("  // Recompute Cuda configuration", start)
text = (
    text[:start]
    + "  // The next launch snapshots the new orientation; no module write.\n\n"
    + text[end:]
)
text = once(
    text, "}  // namespace mvs", "}  // namespace owned\n}  // namespace mvs"
)
assert "__constant__" not in text and "cudaMemcpyToSymbol" not in text
text = "\n".join(line.rstrip() for line in text.splitlines()) + "\n"
for name, data in [
    ("patch_match_cuda_owned.h", header),
    ("patch_match_cuda_owned.cu", text),
]:
    path = MVS / name
    assert not path.exists(), f"refusing to overwrite {path}"
    path.write_text(data)
print(
    "Created owned sibling from canonical base; original stock so"
    "urce not edited."
)
