// SPDX-License-Identifier: BSD-3-Clause
#include "colmap/mvs/owned_source_cache.h"

#include <cassert>
#include <cstring>
#include <initializer_list>

using namespace colmap::mvs::owned;

// Dependency-free CPU regression at the production winner lookup seam.
template <int kTile>
void CheckWinnerLookup() {
  using Cache = OwnedSourceCache<kTile>;
  typename Cache::Sources sources{};
  typename Cache::Values values{};
  for (int lane = 0; lane < kTile; ++lane) {
    for (int slot = 0; slot < Cache::kMaxSamples; ++slot) {
      sources[slot][lane] = 100 * lane + slot;
      for (int candidate = 0; candidate < 4; ++candidate) {
        values[candidate][slot][lane] =
            candidate == 0 ? -0.0f
                           : float(1000 * candidate + 100 * lane + slot);
      }
    }
    for (int size : {0, 1, 7, 15}) {
      for (int winner = 0; winner < 5; ++winner) {
        for (int slot = 0; slot <= Cache::kMaxSamples; ++slot) {
          float cost = -123.0f;
          const bool hit = Cache::FindWinningPhotoCost(
              sources, values, size, lane, winner, 100 * lane + slot, cost);
          assert(hit == (winner > 0 && slot < size));
          if (hit) {
            assert(std::memcmp(&cost,
                               &values[winner - 1][slot][lane],
                               sizeof(float)) == 0);
          } else {
            assert(cost == -123.0f);
          }
        }
      }
    }
  }
}

int main() {
  CheckWinnerLookup<16>();
  CheckWinnerLookup<32>();
}
