// SPDX-License-Identifier: BSD-3-Clause
#pragma once

namespace colmap::mvs::owned {

// Keep one window row's texture results live before consuming any of them.
// Fetch/consume are inlined callables; there is no runtime callback dispatch.
// This expresses independent requests, not a guarantee of NVCC issue order or
// register allocation. Native SASS must confirm batching and no array spills.
#if defined(__CUDACC__) || defined(__HIPCC__)
#define COLMAP_OWNED_TEXTURE_ROW_INLINE __host__ __device__ __forceinline__
#else
#define COLMAP_OWNED_TEXTURE_ROW_INLINE inline
#endif

template <int kWindowSize, int kWindowStep, class Fetch, class Consume>
COLMAP_OWNED_TEXTURE_ROW_INLINE void OwnedTextureRowBatch(Fetch fetch,
                                                          Consume consume) {
  static_assert(kWindowSize > 0, "window size must be positive");
  static_assert(kWindowStep > 0, "window step must be positive");
  if constexpr (kWindowSize > 0 && kWindowStep > 0) {
    constexpr int kRadius = kWindowSize / 2;
    constexpr int kSamples = 2 * kRadius / kWindowStep + 1;
    float colors[kSamples];
#if defined(__CUDACC__) || defined(__HIPCC__)
#pragma unroll
#endif
    for (int sample = 0; sample < kSamples; ++sample) {
      colors[sample] = fetch(-kRadius + sample * kWindowStep);
    }
#if defined(__CUDACC__) || defined(__HIPCC__)
#pragma unroll
#endif
    for (int sample = 0; sample < kSamples; ++sample) {
      consume(-kRadius + sample * kWindowStep, colors[sample]);
    }
  }
}

#undef COLMAP_OWNED_TEXTURE_ROW_INLINE

}  // namespace colmap::mvs::owned
