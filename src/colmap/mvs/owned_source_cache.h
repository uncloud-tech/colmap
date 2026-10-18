// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <type_traits>

namespace colmap::mvs::owned {

// Shared by the host launch decision and the device's lane-private shared
// storage. The sample bound is independent of the number of source images.
template <int kTileThreads>
struct OwnedSourceCache {
  static constexpr bool kSupported = kTileThreads == 16 || kTileThreads == 32;
  static constexpr int kMaxSamples = 15;
  using Values = float[7][kMaxSamples][kTileThreads];
  using Sources = int[kMaxSamples][kTileThreads];

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
