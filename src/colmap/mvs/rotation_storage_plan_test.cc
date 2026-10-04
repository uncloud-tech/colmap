// SPDX-License-Identifier: BSD-3-Clause
#include "colmap/mvs/rotation_storage_plan.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef COLMAP_ROTATION_STORAGE_STANDALONE
#include <gtest/gtest.h>
#endif

namespace {
using namespace colmap::mvs::rotation_storage;
#define REQUIRE(x) do { if (!(x)) throw std::runtime_error( \
  std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": " #x); } while (false)

RotationBacking Owned(size_t bytes, int* releases = nullptr) {
  return {std::shared_ptr<void>(new uint8_t[bytes], [releases](void* p) {
            delete[] static_cast<uint8_t*>(p);
            if (releases) ++*releases;
          }), bytes};
}

void CheckLayoutsAndOwnership() {
  const auto p = MakePlan(3, 2, 61, 4);
  REQUIRE(p);
  REQUIRE(p->source_pitch == 12 && p->rotated_pitch == 8);
  REQUIRE(p->plane_bytes == 24 && p->backing_bytes == 1464);
  for (auto dims : {std::pair<size_t, size_t>{1, 9}, {9, 1}, {7, 7}, {13, 7}}) {
    for (size_t depth : {1, 61, 64}) {
      const auto a = MakePlan(dims.first, dims.second, depth, 4);
      const auto b = MakePlan(dims.second, dims.first, depth, 4);
      REQUIRE(a && b && a->plane_bytes == b->plane_bytes);
      REQUIRE(a->backing_bytes == b->backing_bytes);
    }
  }
  REQUIRE(!MakePlan(0, 2, 1, 4));
  REQUIRE(!MakePlan(2, 0, 1, 4));
  REQUIRE(!MakePlan(2, 2, 0, 4));
  REQUIRE(!MakePlan(2, 2, 1, 0));
  const size_t imax = std::numeric_limits<int>::max();
  REQUIRE(!MakePlan(imax + 1, 1, 1, 1));
  REQUIRE(!MakePlan(1, imax + 1, 1, 1));
  REQUIRE(!MakePlan(1, 1, imax + 1, 1));
  REQUIRE(!MakePlan(imax, 1, 1, 4));
  REQUIRE(!MakePlan(1, imax, 1, 4));
  REQUIRE(!MakePlan(imax / 4, imax / 4, imax, 4));
  REQUIRE(!MakePlan(2, 1, 1, std::numeric_limits<size_t>::max()));
  REQUIRE(!MakeView({}, 3, 2, 61, 4));

  int frees = 0;
  {
    auto backing = Owned(p->backing_bytes, &frees);
    auto source = MakeView(backing, 3, 2, 61, 4);
    REQUIRE(source);
    backing = {};
    auto dest = ReorientView(*source, 2, 3);
    REQUIRE(dest && dest->backing.allocation.get() == source->backing.allocation.get());
    REQUIRE(!ReorientView(*source, 1, 6));
    REQUIRE(frees == 0);
    auto workspace = MakeView(Owned(24), 2, 3, 1, 4);
    REQUIRE(workspace && ValidateReuse(*source, *dest, *workspace));
    auto bad = *dest;
    ++bad.layout.source_pitch;
    REQUIRE(!ValidateReuse(*source, bad, *workspace));
    REQUIRE(!ReorientView(bad, 3, 2));
    bad = *dest;
    --bad.backing.capacity_bytes;
    REQUIRE(!ValidateReuse(*source, bad, *workspace));
    REQUIRE(!MakeView(bad.backing, 2, 3, 61, 4));
    // An equal pointer with an unrelated control block is not shared ownership.
    bad = *dest;
    bad.backing.allocation = std::shared_ptr<void>(bad.backing.allocation.get(), [](void*) {});
    REQUIRE(!ValidateReuse(*source, bad, *workspace));
    auto wrong_shape = ReorientView(*workspace, 3, 2);
    REQUIRE(wrong_shape && !ValidateReuse(*source, *dest, *wrong_shape));
    auto overlap = *workspace;
    overlap.backing.allocation = std::shared_ptr<void>(
        static_cast<uint8_t*>(source->backing.allocation.get()) + 4, [](void*) {});
    REQUIRE(!ValidateReuse(*source, *dest, overlap));
    overlap.backing.allocation = std::shared_ptr<void>(
        source->backing.allocation, static_cast<uint8_t*>(source->backing.allocation.get()) + 4);
    REQUIRE(!ValidateReuse(*source, *dest, overlap));
    auto overflow = *workspace;
    overflow.backing.allocation = std::shared_ptr<void>(
        reinterpret_cast<void*>(std::numeric_limits<uintptr_t>::max() - 3), [](void*) {});
    REQUIRE(!ValidateReuse(*source, *dest, overflow));
    // Release all aliases, including intentionally invalid copies, then final owner.
    overlap = {}; bad = {}; source.reset();
    REQUIRE(frees == 0);
    dest.reset();
    REQUIRE(frees == 1);
  }
  REQUIRE(frees == 1);
}

std::vector<uint32_t> Oracle(const std::vector<uint32_t>& input,
                             size_t width, size_t height, size_t depth) {
  std::vector<uint32_t> result(input.size());
  for (size_t s = 0; s < depth; ++s)
    for (size_t y = 0; y < height; ++y)
      for (size_t x = 0; x < width; ++x)
        result[s * width * height + (width - 1 - x) * height + y] =
            input[s * width * height + y * width + x];
  return result;
}

void CheckBitsAndSchedule() {
  REQUIRE(Oracle({0, 1, 2, 3, 4, 5}, 3, 2, 1) ==
          std::vector<uint32_t>({2, 5, 1, 4, 0, 3}));
  const uint32_t special[] = {0, 0x80000000u, 0x7fc12345u, 0x7f800001u,
                              0x7f800000u, 0xff800000u, 0xffffffffu};
  for (auto dims : {std::pair<size_t, size_t>{3, 2}, {1, 9}, {9, 1}, {7, 7}, {13, 7}}) {
    for (size_t depth : {1, 61, 64}) {
      size_t width = dims.first, height = dims.second;
      const size_t count = width * height * depth;
      std::vector<uint32_t> original(count);
      for (size_t i = 0; i < count; ++i)
        original[i] = i < 7 ? special[i] : static_cast<uint32_t>(i * 2654435761u);
      std::vector<uint32_t> guarded(count + 2, 0xdeadbeefu);
      std::copy(original.begin(), original.end(), guarded.begin() + 1);
      std::vector<uint32_t> scratch(width * height + 2, 0xa5a5a5a5u);
      for (int turn = 0; turn < 4; ++turn) {
        const auto plan = MakePlan(width, height, depth, 4);
        REQUIRE(plan);
        const std::vector<uint32_t> before(guarded.begin() + 1, guarded.end() - 1);
        const auto expected = Oracle(before, width, height, depth);
        size_t next_layer = 0;
        bool want_rotate = true;
        ForEachLayer(*plan,
          [&](size_t layer, size_t offset) {
            REQUIRE(want_rotate && layer == next_layer && offset == layer * plan->plane_bytes);
            want_rotate = false;
            const auto* input = reinterpret_cast<const uint8_t*>(guarded.data() + 1) + offset;
            // CPU stand-in for the unchanged rotation kernel: byte copies only.
            for (size_t out_y = 0; out_y < width; ++out_y)
              for (size_t out_x = 0; out_x < height; ++out_x)
                std::memcpy(&scratch[1 + out_y * height + out_x],
                            input + (out_x * width + width - 1 - out_y) * 4, 4);
          },
          [&](size_t layer, size_t offset) {
            REQUIRE(!want_rotate && layer == next_layer);
            std::memcpy(reinterpret_cast<uint8_t*>(guarded.data() + 1) + offset,
                        scratch.data() + 1, plan->plane_bytes);
            want_rotate = true; ++next_layer;
          });
        REQUIRE(want_rotate && next_layer == depth);
        REQUIRE(std::equal(expected.begin(), expected.end(), guarded.begin() + 1));
        REQUIRE(guarded.front() == 0xdeadbeefu && guarded.back() == 0xdeadbeefu);
        REQUIRE(scratch.front() == 0xa5a5a5a5u && scratch.back() == 0xa5a5a5a5u);
        std::swap(width, height);
      }
      REQUIRE(std::equal(original.begin(), original.end(), guarded.begin() + 1));
    }
  }
  int calls = 0;
  try {
    ForEachLayer(*MakePlan(3, 2, 61, 4),
                 [&](size_t, size_t) { ++calls; },
                 [&](size_t, size_t) { ++calls; throw std::runtime_error("copy failure"); });
    REQUIRE(false);
  } catch (const std::runtime_error& e) {
    REQUIRE(std::string(e.what()) == "copy failure");
  }
  REQUIRE(calls == 2);
}
void RunTests() { CheckLayoutsAndOwnership(); CheckBitsAndSchedule(); }
}  // namespace

#ifdef COLMAP_ROTATION_STORAGE_STANDALONE
int main() {
  try { RunTests(); std::puts("rotation storage tests: PASS"); }
  catch (const std::exception& e) { std::fprintf(stderr, "%s\n", e.what()); return 1; }
}
#else
TEST(RotationStoragePlan, LayoutOwnershipAndBits) { EXPECT_NO_THROW(RunTests()); }
#endif
