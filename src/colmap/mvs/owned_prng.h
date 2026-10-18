// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "colmap/mvs/gpu_mat_prng.h"

#include <memory>

#if !defined(COLMAP_HIP_ENABLED)
namespace colmap {
namespace mvs {
namespace owned {

// Allocate the same XORWOW map and finish initialization on the supplied stream
// before exposing it. Negative seeds retain legacy initialization; nonnegative
// seeds use current reference-map coordinates only on the owned backend.
std::unique_ptr<GpuMatPRNG> MakeOwnedPrng(int width,
                                       int height,
                                       cudaStream_t stream,
                                       int random_seed);
void ReinitializeOwnedPrng(GpuMatPRNG& states,
                          cudaStream_t stream,
                          int random_seed);

}  // namespace owned
}  // namespace mvs
}  // namespace colmap
#endif
