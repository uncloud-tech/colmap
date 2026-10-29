// SPDX-License-Identifier: BSD-3-Clause
#include "colmap/mvs/owned_photo_output_state.h"

#include <cassert>
#include <stdexcept>

using colmap::mvs::owned::PhotoOutputState;
int main() {
  PhotoOutputState state;
  auto refuses = [&](bool geometry, size_t width, size_t height, int rotation) {
    try {
      state.RequireSnapshot(geometry, width, height, rotation);
    } catch (const std::logic_error&) {
      return true;
    }
    return false;
  };
  assert(refuses(false, 5, 4, 0));
  state.BeginRun();
  state.FinishRun();
  assert(!refuses(false, 5, 4, 0));
  assert(refuses(true, 5, 4, 0));
  assert(refuses(false, 0, 4, 0));
  assert(refuses(false, 5, 0, 0));
  assert(refuses(false, 5, 4, 1));
  // A failed/cancelled rerun never reaches FinishRun: no stale publication.
  state.BeginRun();
  assert(refuses(false, 5, 4, 0));
}
