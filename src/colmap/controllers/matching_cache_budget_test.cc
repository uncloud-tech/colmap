// SPDX-License-Identifier: BSD-3-Clause
#include "colmap/controllers/matching_cache_budget.h"

#include <cassert>
#include <iostream>

int main() {
  colmap::MatchingCacheInventory inventory{{9, 1, 2}, {1, 8, 3}, {2, 1, 7}};
  // Three independent LRUs may retain three different largest images.
  assert(inventory.Retained(1) == 24);
  assert(inventory.Retained(2) == 31);
  assert(inventory.Retained(100) == 34);
  // 24 retained + 10 pinned + 5 pair = 39, once-margined to 41 + 2 reserve.
  assert(inventory.Select(3, 10, 5, 43, 0, 2) == 1);
  assert(inventory.Select(3, 10, 5, 53, 0, 2) == 3);
  bool refused = false;
  try {
    inventory.Select(3, 10, 5, 42, 0, 2);
  } catch (const std::runtime_error&) {
    refused = true;
  }
  assert(refused);
  assert(inventory.Select(3, 10, 5, 49, 0, 2) == 1);
  assert(inventory.Select(3, 10, 5, 50, 0, 2) == 2);
  assert(colmap::MatchingCacheInventory{}.Select(100, 0, 0, 0, 0, 0) == 100);
  bool overflow = false;
  try {
    colmap::MatchingCacheInventory{{UINT64_MAX, 1}, {}, {}}.Retained(2);
  } catch (const std::overflow_error&) {
    overflow = true;
  }
  assert(overflow);
  // Removing retained owners never removes workers' pin/load requirements.
  refused = false;
  try {
    inventory.Select(100, 1000, 1, 1000, 0, 0);
  } catch (const std::runtime_error&) {
    refused = true;
  }
  assert(refused);
  std::cout << "matching cache inventory PASS\n";
}
