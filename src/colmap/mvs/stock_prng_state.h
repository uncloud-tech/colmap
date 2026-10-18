// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <cstddef>

namespace colmap {
namespace mvs {

// Logical state layout only, not physical GPU allocation or fit evidence.
// This header deliberately does not include CUDA or cuRAND headers.
struct StockPrngStateLayout {
  std::size_t bytes;
  std::size_t alignment;
};

#if defined(COLMAP_CUDA_ENABLED) && defined(COLMAP_MVS_ENABLED) && \
    !defined(COLMAP_HIP_ENABLED)
// Compiled in the same CUDA translation unit as GpuMatPRNG. Returns only
// sizeof/alignof constants; does not allocate, launch, or call the CUDA
// runtime.
StockPrngStateLayout GetStockPrngStateLayout();
#endif

}  // namespace mvs
}  // namespace colmap
