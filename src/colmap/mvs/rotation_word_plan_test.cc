// SPDX-License-Identifier: BSD-3-Clause
#include "colmap/mvs/rotation_storage_plan.h"

#include <algorithm>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>
#ifndef COLMAP_ROTATION_STORAGE_STANDALONE
#include <gtest/gtest.h>
#endif

namespace {
void Check(bool ok) {
  if (!ok) throw std::runtime_error("rotation word test failed");
}
// Opaque stand-ins exercise full state object representations, including padding.
// Actual curandState is CUDA-only; its ABI must additionally be checked on CUDA.
void Test(size_t width, size_t height, size_t bytes, size_t depth) {
  using namespace colmap::mvs::rotation_storage;
  constexpr size_t guard = 64;
  const size_t count = width * height;
  std::vector<unsigned char> data(2 * guard + count * bytes * depth, 0xa5);
  for (size_t i = guard; i < data.size() - guard; ++i)
    data[i] = static_cast<unsigned char>((i * 131 + i / 7) & 255);
  if (bytes == 4) {
    const uint32_t bits[] = {0x00000000, 0x80000000, 0x7f800000,
                             0xff800000, 0x7fc01234, 0x7fa05678};
    for (size_t i = 0; i < count * depth; ++i)
      std::memcpy(data.data() + guard + i * bytes, &bits[i % 6], 4);
  }
  const auto original = data;
  std::vector<unsigned char> scratch(2 * guard + count * 4, 0xcc);
  for (int turn = 0; turn < 4; ++turn) {
    const auto before = data;
    auto expected = data;
    for (size_t s = 0; s < depth; ++s)
      for (size_t y = 0; y < height; ++y)
        for (size_t x = 0; x < width; ++x)
          std::memcpy(expected.data() + guard +
                          (s * count + (width - 1 - x) * height + y) * bytes,
                      before.data() + guard + (s * count + y * width + x) * bytes,
                      bytes);
    const auto plan = MakePlan(width, height, depth, bytes);
    Check(plan.has_value());
    if (bytes % 4 == 0) {
      size_t calls = 0;
      ForEachWord(*plan,
          [&](size_t layer, size_t word) {
            Check(calls++ == 2 * (layer * (bytes / 4) + word));
            std::fill(scratch.begin() + guard, scratch.end() - guard, 0x99);
            std::vector<bool> written(count, false);
            for (size_t y = 0; y < height; ++y)
              for (size_t x = 0; x < width; ++x) {
                const size_t dst = RotatedIndex(y * width + x, width, height);
                Check(!written.at(dst));
                written[dst] = true;
                std::memcpy(scratch.data() + guard + dst * 4,
                            data.data() + guard +
                                (layer * count + y * width + x) * bytes + word * 4,
                            4);
              }
            Check(std::all_of(written.begin(), written.end(), [](bool b) { return b; }));
          },
          [&](size_t layer, size_t word) {
            Check(calls++ == 2 * (layer * (bytes / 4) + word) + 1);
            for (size_t i = 0; i < count; ++i)
              std::memcpy(data.data() + guard + (layer * count + i) * bytes + word * 4,
                          scratch.data() + guard + i * 4, 4);
          });
      Check(calls == 2 * depth * bytes / 4);
    } else {
      ForEachLayer(*plan,
          [&](size_t, size_t offset) {
            for (size_t y = 0; y < height; ++y)
              for (size_t x = 0; x < width; ++x)
                scratch[guard + RotatedIndex(y * width + x, width, height)] =
                    data[guard + offset + y * width + x];
          },
          [&](size_t, size_t offset) {
            std::memcpy(data.data() + guard + offset, scratch.data() + guard, count);
          });
    }
    Check(data == expected);
    Check(std::all_of(scratch.begin(), scratch.begin() + guard,
                      [](unsigned char b) { return b == 0xcc; }));
    Check(std::all_of(scratch.end() - guard, scratch.end(),
                      [](unsigned char b) { return b == 0xcc; }));
    std::swap(width, height);
  }
  Check(data == original);
}
}  // namespace
void RunTests() {
  using namespace colmap::mvs::rotation_storage;
  RotationBacking backing{std::shared_ptr<void>(new unsigned char[24], [](void* p) {
                            delete[] static_cast<unsigned char*>(p);
                          }), 24};
  const auto storage = MakeView(backing, 3, 2, 1, 4);
  Check(storage.has_value());
  const auto bytes = MakeWorkspaceView(*storage, 2, 3, 1);
  Check(bytes.has_value() && bytes->backing.capacity_bytes == 6);
  Check(SameOwner(storage->backing, bytes->backing));
  Check(MakeWorkspaceView(*storage, 2, 3, 4).has_value());
  Check(!MakeWorkspaceView(*storage, 2, 3, 8).has_value());
  Check(!MakeWorkspaceView({}, 2, 3, 1).has_value());
  Check(!MakeWorkspaceView(*storage, 0, 3, 1).has_value());

  for (const auto& shape : std::vector<std::pair<size_t, size_t>>{
           {1, 1}, {1, 17}, {17, 1}, {3, 2}, {2, 3}, {13, 29}, {32, 32}, {65, 31}}) {
    Test(shape.first, shape.second, 1, 64);  // byte masks
    Test(shape.first, shape.second, 1, 8);   // packed mask bytes
    Test(shape.first, shape.second, 4, 2);   // packed mask words
    Test(shape.first, shape.second, 4, 1);  // float bits
    Test(shape.first, shape.second, 4, 3);  // normals
    Test(shape.first, shape.second, 48, 1); // XORWOW-sized opaque state
    Test(shape.first, shape.second, 16, 1); // compact state representation
    Test(shape.first, shape.second, 24, 1); // alternative compact state
  }
  std::cout << "rotation word tests passed (64 shape/type cases, four rotations each)\n";
}

#ifdef COLMAP_ROTATION_STORAGE_STANDALONE
int main() { RunTests(); }
#else
TEST(RotationWordPlan, BitPreservingSchedule) { RunTests(); }
#endif
