// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <type_traits>

#if defined(__CUDACC__)
#define COLMAP_OWNED_SOURCE_CACHE_HD __host__ __device__
#else
#define COLMAP_OWNED_SOURCE_CACHE_HD
#endif

namespace colmap::mvs::owned {

// Shared by the host launch decision and the device's lane-private shared
// storage. The sample bound is independent of the number of source images.
template <int kTileThreads>
struct OwnedSourceCache {
  static constexpr bool kSupported = kTileThreads == 16 || kTileThreads == 32;
  static constexpr int kMaxSamples = 15;
  using Values = float[7][kMaxSamples][kTileThreads];
  using Sources = int[kMaxSamples][kTileThreads];

  // Only initialized slots in this lane/row are eligible. The four PHOTO
  // planes exclude geometry, and winner 0 already lives in cost_map.
  COLMAP_OWNED_SOURCE_CACHE_HD static bool FindWinningPhotoCost(
      const Sources& sources,
      const Values& values,
      int size,
      int lane,
      int winner,
      int source,
      float& cost) {
    if (winner == 0) return false;
    for (int slot = 0; slot < size; ++slot) {
      if (sources[slot][lane] == source) {
        cost = values[winner - 1][slot][lane];
        return true;
      }
    }
    return false;
  }

  static constexpr bool Enabled(bool requested, int samples) {
    return kSupported && requested && samples <= kMaxSamples;
  }
};

// Launch receives compile-time booleans, not runtime kernel parameters.
template <int kTileThreads, class Launch>
void DispatchOwnedSourceCache(bool cache,
                              bool lookahead,
                              int samples,
                              Launch launch) {
  using Cache = OwnedSourceCache<kTileThreads>;
  if constexpr (Cache::kSupported) {
    if (Cache::Enabled(cache, samples)) {
      if (lookahead) {
        launch(std::true_type{}, std::true_type{});
      } else {
        launch(std::true_type{}, std::false_type{});
      }
      return;
    }
  }
  launch(std::false_type{}, std::false_type{});
}

}  // namespace colmap::mvs::owned

#undef COLMAP_OWNED_SOURCE_CACHE_HD
