// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/mvs/owned_prng_seed.h"
#include "colmap/mvs/patch_match_options.h"

#include <algorithm>
#include <vector>

#include <gtest/gtest.h>

namespace colmap {
namespace mvs {
namespace {

TEST(OwnedPrngSeed, KnownSeedsAndDistinctness) {
  EXPECT_EQ(owned::prng_seed::PixelSeed(0, 0, 0), 0xa706dd2f4d197e6full);
  EXPECT_EQ(owned::prng_seed::PixelSeed(7, 11, 13), 0xe15895f55529dce7ull);
  EXPECT_EQ(owned::prng_seed::PixelSeed(2147483647, 1048576, 1048576),
            0x53428268c4118a07ull);
  std::vector<uint64_t> seeds;
  for (int seed : {7, 8}) {
    for (int row = 0; row < 16; ++row) {
      for (int col = 0; col < 16; ++col) {
        seeds.push_back(owned::prng_seed::PixelSeed(seed, row, col));
      }
    }
  }
  std::sort(seeds.begin(), seeds.end());
  EXPECT_EQ(std::adjacent_find(seeds.begin(), seeds.end()), seeds.end());
}

TEST(OwnedPrngSeed, OptionDomain) {
  PatchMatchOptions options;
  EXPECT_EQ(options.random_seed, -1);
  EXPECT_TRUE(options.Check());
  options.random_seed = 0;
  EXPECT_TRUE(options.Check());
  options.random_seed = 2147483647;
  EXPECT_TRUE(options.Check());
  options.random_seed = -2;
  EXPECT_FALSE(options.Check());
}

}  // namespace
}  // namespace mvs
}  // namespace colmap
