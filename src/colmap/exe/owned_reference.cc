// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/exe/mvs.h"
#include "colmap/util/logging.h"

// Experimental, CUDA-only, one-reference executable. Uses the existing stereo
// parser and input preparation; execution is an explicit per-controller value.
int main(int argc, char** argv) {
  colmap::InitializeGlog(argv);
  return colmap::RunPatchMatchStereoWithExecution(
      argc, argv, colmap::mvs::PatchMatchExecution::kOwnedCuda);
}
