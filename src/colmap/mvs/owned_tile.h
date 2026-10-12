// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <stdexcept>
#include <string>
#include <type_traits>

namespace colmap::mvs::owned {

inline int ValidateOwnedTileThreads(int threads) {
  if (threads != 8 && threads != 16 && threads != 32) {
    throw std::invalid_argument("owned_tile_threads must be 8, 16, or 32");
  }
  return threads;
}

inline int ParseOwnedTileThreads(const std::string& text) {
  if (text == "8") return 8;
  if (text == "16") return 16;
  if (text == "32") return 32;
  throw std::invalid_argument("owned_tile_threads must be 8, 16, or 32");
}

// The same bounded dispatch is exercised by CPU tests and the CUDA owner.
// No global state or device-dependent selection.
template <class Launch>
void DispatchOwnedTile(int threads, Launch launch) {
  switch (ValidateOwnedTileThreads(threads)) {
    case 8: launch(std::integral_constant<int, 8>{}); break;
    case 16: launch(std::integral_constant<int, 16>{}); break;
    case 32: launch(std::integral_constant<int, 32>{}); break;
  }
}

}  // namespace colmap::mvs::owned
