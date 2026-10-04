// SPDX-License-Identifier: BSD-3-Clause

// Standalone host test: no CUDA runtime or project dependencies required.
#include "colmap/mvs/packed_consistency_mask.h"

#include <cstdlib>
#include <iostream>
#include <limits>
#include <utility>
#include <string>
#include <type_traits>
#include <vector>

namespace {
using colmap::mvs::PackedConsistencyMaskDepth;
using colmap::mvs::PackedConsistencyMaskView;
using colmap::mvs::UnpackConsistentImageIdxs;

void Check(bool condition) {
  if (!condition) {
    std::cerr << "Packed consistency mask check failed\n";
    std::exit(EXIT_FAILURE);
  }
}

// Independent reference for the existing GetConsistentImageIdxs layout.
std::vector<int> Reference(const std::vector<uint8_t>& mask,
                           size_t width,
                           size_t height,
                           const std::vector<int>& sources) {
  std::vector<int> result;
  for (size_t row = 0; row < height; ++row) {
    for (size_t col = 0; col < width; ++col) {
      std::vector<int> pixel;
      for (size_t s = 0; s < sources.size(); ++s) {
        if (mask[(s * height + row) * width + col]) {
          pixel.push_back(sources[s]);
        }
      }
      if (!pixel.empty()) {
        result.push_back(static_cast<int>(col));
        result.push_back(static_cast<int>(row));
        result.push_back(static_cast<int>(pixel.size()));
        result.insert(result.end(), pixel.begin(), pixel.end());
      }
    }
  }
  return result;
}

// ConsistencyGraph::Write format: ASCII dimensions then little-endian int32.
std::string Export(size_t width, size_t height, const std::vector<int>& list) {
  std::string bytes = std::to_string(width) + "&" +
                      std::to_string(height) + "&1&";
  for (const int value : list) {
    for (int byte = 0; byte < 4; ++byte) {
      bytes.push_back(static_cast<char>(static_cast<uint32_t>(value) >>
                                        (8 * byte)));
    }
  }
  return bytes;
}

// Check the same sparse lookup semantics used by ConsistencyGraph, independently
// of the list builder and of source-plane addressing in the packed view.
void CheckGraph(const std::vector<int>& list,
                const std::vector<uint8_t>& reference,
                size_t width,
                size_t height,
                const std::vector<int>& sources) {
  std::vector<std::vector<int>> graph(width * height);
  for (size_t i = 0; i < list.size();) {
    Check(i + 2 < list.size());
    const size_t col = list[i++];
    const size_t row = list[i++];
    const size_t count = list[i++];
    Check(col < width && row < height && count <= list.size() - i);
    auto& pixel = graph[row * width + col];
    Check(pixel.empty());
    pixel.assign(list.begin() + i, list.begin() + i + count);
    i += count;
  }
  for (size_t row = 0; row < height; ++row) {
    for (size_t col = 0; col < width; ++col) {
      std::vector<int> expected;
      for (size_t source = 0; source < sources.size(); ++source) {
        if (reference[(source * height + row) * width + col]) {
          expected.push_back(sources[source]);
        }
      }
      Check(graph[row * width + col] == expected);
    }
  }
}

void Test(size_t width, size_t height, size_t sources, int pattern) {
  size_t pitch = width + 13;
  const size_t planes = PackedConsistencyMaskDepth(sources);
  std::vector<uint8_t> data(pitch * height * planes + 16, 0);
  struct HostByteMat {
    uint8_t* ptr;
    size_t pitch;
    size_t width;
    size_t height;
    uint8_t* GetPtr() { return ptr; }
    size_t GetPitch() const { return pitch; }
    size_t GetWidth() const { return width; }
    size_t GetHeight() const { return height; }
  } storage{data.data(), pitch, width, height};
  auto view = colmap::mvs::MakePackedConsistencyMaskView(storage, sources);
  std::vector<uint8_t> reference(width * height * sources, 0);
  std::vector<int> ids(sources);
  for (size_t s = 0; s < sources; ++s) {
    ids[s] = static_cast<int>(1000 - 3 * s);
  }
  // Write in reverse source order; repeatedly set and clear individual bits.
  for (size_t row = 0; row < height; ++row) {
    for (size_t col = 0; col < width; ++col) {
      for (size_t s = sources; s-- > 0;) {
        const bool value = pattern == 1 ||
            (pattern == 2 && (s + col + row) % 2 == 0) ||
            (pattern == 3 && (s * 17 + col * 7 + row * 11) % 13 < 5) ||
            (pattern == 4 && (s == 0 || s == sources - 1));
        view.Set(row, col, s, !value);
        view.Set(row, col, s, value);
        reference[(s * height + row) * width + col] = value;
      }
      // Filter rejection clears every source of a pixel.
      if (pattern == 5 || (pattern == 3 && (row + col) % 5 == 0)) {
        for (size_t s = 0; s < sources; ++s) {
          view.Set(row, col, s, 1);
        }
        for (size_t s = 0; s < sources; ++s) {
          view.Set(row, col, s, 0);
          reference[(s * height + row) * width + col] = 0;
        }
      }
      // Out-of-range bits must never be read or written.
      view.Set(row, col, sources, 1);
      Check(!view.Get(row, col, sources));
      for (size_t s = 0; s < sources; ++s) {
        Check(view.Get(row, col, s) ==
              reference[(s * height + row) * width + col]);
      }
    }
  }
  for (size_t p = 0; p < planes; ++p) {
    for (size_t r = 0; r < height; ++r) {
      for (size_t c = width; c < pitch; ++c) {
        Check(data[(p * height + r) * pitch + c] == 0);
      }
      if (p + 1 == planes && sources % 8 != 0) {
        for (size_t c = 0; c < width; ++c) {
          Check((data[(p * height + r) * pitch + c] >> (sources % 8)) == 0);
        }
      }
    }
  }
  for (size_t i = pitch * height * planes; i < data.size(); ++i) {
    Check(data[i] == 0);
  }
  // Even dirty unused bits must not leak into consistent-image lists.
  if (sources % 8 != 0) {
    for (size_t row = 0; row < height; ++row) {
      for (size_t col = 0; col < width; ++col) {
        data[((planes - 1) * height + row) * pitch + col] |=
            static_cast<uint8_t>(0xff << (sources % 8));
      }
    }
  }
  const auto expected = Reference(reference, width, height, ids);
  Check(UnpackConsistentImageIdxs(view, ids) == expected);
  CheckGraph(UnpackConsistentImageIdxs(view, ids), reference, width, height, ids);
  Check(Export(width, height, UnpackConsistentImageIdxs(view, ids)) ==
        Export(width, height, expected));

  // Model the unchanged GpuMat::Rotate/CudaRotate per-plane mapping.
  // Compare lists after each rotation, including non-square dimensions.
  for (int rotation = 0; rotation < 4; ++rotation) {
    const size_t rotated_pitch = height + 9;
    std::vector<uint8_t> rotated(rotated_pitch * width * planes, 0);
    std::vector<uint8_t> rotated_reference(reference.size(), 0);
    for (size_t r = 0; r < height; ++r) {
      for (size_t c = 0; c < width; ++c) {
        for (size_t p = 0; p < planes; ++p) {
          rotated[(p * width + width - 1 - c) * rotated_pitch + r] =
              data[(p * height + r) * pitch + c];
        }
        for (size_t s = 0; s < sources; ++s) {
          rotated_reference[(s * width + width - 1 - c) * height + r] =
              reference[(s * height + r) * width + c];
        }
      }
    }
    PackedConsistencyMaskView rotated_view{
        rotated.data(), rotated_pitch, height, width, sources};
    const auto rotated_expected = Reference(rotated_reference, height, width, ids);
    Check(UnpackConsistentImageIdxs(rotated_view, ids) == rotated_expected);
    CheckGraph(UnpackConsistentImageIdxs(rotated_view, ids),
               rotated_reference, height, width, ids);
    Check(Export(height, width, UnpackConsistentImageIdxs(rotated_view, ids)) ==
          Export(height, width, rotated_expected));
    // Inverse the mapping to verify byte-exact recovery of every plane.
    for (size_t p = 0; p < planes; ++p) {
      for (size_t r = 0; r < height; ++r) {
        for (size_t c = 0; c < width; ++c) {
          Check(rotated[(p * width + width - 1 - c) * rotated_pitch + r] ==
                data[(p * height + r) * pitch + c]);
        }
      }
    }
    data = std::move(rotated);
    reference = std::move(rotated_reference);
    std::swap(width, height);
    pitch = rotated_pitch;
  }
  Check(UnpackConsistentImageIdxs(
            {data.data(), pitch, width, height, sources}, ids) == expected);
}
}  // namespace

int main() {
  static_assert(std::is_trivially_copyable<PackedConsistencyMaskView>::value,
                "Kernel arguments must be trivially copyable");
  for (const size_t sources : {1, 7, 8, 9, 63, 64, 65, 129, 257}) {
    for (int pattern = 0; pattern < 6; ++pattern) {
      Test(11, 7, sources, pattern);
      Test(1, 13, sources, pattern);
      Test(17, 1, sources, pattern);
    }
  }
  Check(PackedConsistencyMaskDepth(0) == 0);
  const size_t max_sources = std::numeric_limits<size_t>::max();
  Check(PackedConsistencyMaskDepth(max_sources) == max_sources / 8 + 1);
  Check(UnpackConsistentImageIdxs({nullptr, 0, 0, 0, 0}, {}).empty());
  std::cout << "PASS: packed round-trip, padding, filtering, rotation, lists, "
               "and graph export equivalence (162 cases)\n";
}
