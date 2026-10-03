// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/mvs/sweep_tile.h"
#include "colmap/mvs/benchmark_seed.h"
#include "colmap/mvs/patch_match_options.h"

#include <algorithm>
#include <climits>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef COLMAP_SWEEP_STANDALONE
#include <gtest/gtest.h>
#endif

namespace {
namespace st = colmap::mvs::sweep_tile;
#define REQUIRE(x)                                              \
  do {                                                          \
    if (!(x)) throw std::runtime_error(std::string("Failed: ") + #x); \
  } while (false)

// Coordinate-coded infinite image: checks identity, not just initialization.
long long Pixel(int row, int col) {
  return 1000000000LL + 100000LL * row + col;
}
long long checks = 0;

template <int kTile, bool kPadded>
bool Model(int radius, int step, int width, bool bad_pitch = false) {
  using Layout = st::RefLayout<kTile>;
  constexpr int pitch = kPadded ? st::kSharedColumns : Layout::kNumColumns;
  const auto index = [](int row, int col) {
    return kPadded ? st::RefIndex(row, col) : Layout::Index(row, col);
  };
  const int consumer_pitch =
      bad_pitch ? (kPadded ? 3 * kTile : st::kSharedColumns) : pitch;
  const int rows = 2 * radius + 1;
  for (int block = 0; block < (width + kTile - 1) / kTile; ++block) {
    std::vector<long long> data(rows * pitch, LLONG_MIN);
    for (int image_row = 0; image_row < 4; ++image_row) {
      // Copy ownership/order models LocalRefImage::Read; all lanes load.
      if (image_row == 0) {
        for (int r = 0; r < rows; ++r) {
          for (int t = 0; t < kTile; ++t) {
            for (int j = 0; j < 3; ++j) {
              data[index(r, t + j * kTile)] =
                  Pixel(r - radius, block * kTile - kTile + t + j * kTile);
            }
          }
        }
      } else {
        for (int r = 1; r < rows; ++r) {
          for (int t = 0; t < kTile; ++t) {
            for (int j = 0; j < 3; ++j) {
              data[index(r - 1, t + j * kTile)] =
                  data[index(r, t + j * kTile)];
            }
          }
        }
        for (int t = 0; t < kTile; ++t) {
          for (int j = 0; j < 3; ++j) {
            data[index(rows - 1, t + j * kTile)] =
                Pixel(image_row + radius,
                      block * kTile - kTile + t + j * kTile);
          }
        }
      }
      for (int r = 0; r < rows; ++r) {
        for (int c = 3 * kTile; c < pitch; ++c) {
          REQUIRE(data[index(r, c)] == LLONG_MIN);
        }
      }
      for (int t = 0; t < kTile; ++t) {
        if (block * kTile + t >= width) continue;
        int base = kPadded ? st::PatchColumn(kTile, radius, t)
                           : Layout::PatchColumn(radius, t);
        const int center = radius * consumer_pitch + base + radius;
        if (center < 0 || center >= static_cast<int>(data.size()) ||
            data[center] != Pixel(image_row, block * kTile + t)) {
          return false;
        }
        ++checks;
        for (int dr = -radius; dr <= radius; dr += step) {
          int idx = base;
          for (int dc = -radius; dc <= radius; dc += step) {
            if (idx < 0 || idx >= static_cast<int>(data.size()) ||
                data[idx] != Pixel(image_row + dr, block * kTile + t + dc)) {
              return false;
            }
            idx += step;
            ++checks;
          }
          base += step * consumer_pitch;
        }
      }
    }
  }
  return true;
}

template <int kTile>
void CheckLayouts() {
  static_assert(st::RefLayout<kTile>::kNumColumns == 3 * kTile);
  for (int radius = 1; radius <= kTile; ++radius) {
    for (int step : {1, 2}) {
      for (int width : {1, 7, 8, 9, 15, 16, 17, 31, 32, 33, 65}) {
        REQUIRE((Model<kTile, false>(radius, step, width)));
        REQUIRE((Model<kTile, true>(radius, step, width)));
      }
    }
  }
  if constexpr (kTile < 32) {
    // Reject both mismatches: packed consumer/padded producer AND inverse.
    REQUIRE((!Model<kTile, true>(5, 1, 33, true)));
    REQUIRE((!Model<kTile, false>(5, 1, 33, true)));
  }
}

void RunBenchmarkSeedChecks() {
  namespace bs = colmap::mvs::benchmark_seed;
  // Deterministic: identical inputs give identical seeds.
  REQUIRE(bs::PixelSeed(7, 11, 13) == bs::PixelSeed(7, 11, 13));
  // Distinct pixels and distinct seeds do not collide in a dense sample.
  std::vector<uint64_t> seen;
  for (int64_t row = 0; row < 16; ++row) {
    for (int64_t col = 0; col < 16; ++col) {
      seen.push_back(bs::PixelSeed(7, row, col));
      seen.push_back(bs::PixelSeed(8, row, col));
    }
  }
  std::sort(seen.begin(), seen.end());
  REQUIRE(std::adjacent_find(seen.begin(), seen.end()) == seen.end());
  // Coordinate-coded, not launch-coded: neighbouring pixels differ in both axes.
  REQUIRE(bs::PixelSeed(7, 100, 200) != bs::PixelSeed(7, 100, 201));
  REQUIRE(bs::PixelSeed(7, 100, 200) != bs::PixelSeed(7, 101, 200));
  REQUIRE(bs::PixelSeed(7, 100, 200) != bs::PixelSeed(9, 100, 200));
  // Large coordinate range stays well mixed (no truncation to 32 bits).
  REQUIRE(bs::PixelSeed(7, 1 << 20, 1 << 20) != bs::PixelSeed(7, 0, 0));
  // The option default must preserve historical (production) seeding.
  colmap::mvs::PatchMatchOptions options;
  REQUIRE(options.random_seed == -1);
  std::cout << "PASS benchmark seed: deterministic, coordinate-indexed, "
               "option default preserves legacy seeding\n";
}

void RunTests() {
  REQUIRE(st::Select(3930, 2210, 170, 5, 0) == 8);
  REQUIRE(st::Select(3930, 2210, 82, 5, 0) == 16);
  REQUIRE(st::Select(3930, 2210, 36, 5, 0) == 32);
  for (int sms : {0, 36, 82, 170}) {
    REQUIRE(st::Select(2210, 3930, sms, 5, 0) ==
            st::Select(3930, 2210, sms, 5, 0));
    for (int tile : {8, 16, 32}) {
      REQUIRE(st::Select(3930, 2210, sms, 5, tile) == tile);
    }
    for (int tile : {16, 32}) {
      if (sms > 0) {
        int boundary = (2 * sms - 1) * tile + 1;
        REQUIRE(st::Select(boundary, 1, sms, 1, 0) == tile);
        REQUIRE(st::Select(boundary - 1, 1, sms, 1, 0) == tile / 2);
      }
    }
  }
  REQUIRE(st::Select(1, 1, 170, 5, 0) == 8);
  REQUIRE(st::Select(3930, 2210, 170, 9, 0) == 16);
  REQUIRE(st::Select(3930, 2210, 170, 17, 0) == 32);
  REQUIRE(st::Select(3930, 2210, 170, 9, 8) == 0);
  REQUIRE(st::Select(3930, 2210, 170, 17, 16) == 0);
  REQUIRE(st::Select(1, 1, 0, 1, 0) == 32);
  REQUIRE(st::Select(1, 1, -1, 1, 0) == 32);
  REQUIRE(st::Select(0, 1, 170, 5, 0) == 0);
  REQUIRE(st::Select(1, 1, 170, 33, 0) == 0);
  REQUIRE(st::Select(1, 1, 170, 0, 0) == 0);
  REQUIRE(st::Select(1, 1, 170, 5, 12) == 0);
  REQUIRE(st::Select(INT_MAX, 1, INT_MAX, 5, 0) == 8);
  CheckLayouts<8>();
  CheckLayouts<16>();
  CheckLayouts<32>();
  RunBenchmarkSeedChecks();
  std::cout << "PASS selector, specialized + padded layout identity; " << checks
            << " pixel checks; both pitch-mismatch mutants rejected\n";
}
#undef REQUIRE
}  // namespace

#ifdef COLMAP_SWEEP_STANDALONE
int main() { RunTests(); }
#else
TEST(SweepTile, SelectionAndSharedLayout) { RunTests(); }
#endif
