// SPDX-License-Identifier: BSD-3-Clause
#include "colmap/mvs/compact_prng.h"
#include "colmap/mvs/benchmark_seed.h"

#include <cstdlib>
#include <iostream>
#include <vector>

using namespace colmap::mvs;
void Check(bool value) {
  if (!value) { std::cerr << "compact PRNG check failed\n"; std::exit(1); }
}
struct FullView {
  uint64_t* data;
  size_t width;
  uint64_t Get(size_t row, size_t col) const { return data[row * width + col]; }
  void Set(size_t row, size_t col, uint64_t value) const {
    data[row * width + col] = value;
  }
};
void Test(size_t width, size_t height) {
  PrngPerimeter edge{width, height};
  std::vector<uint64_t> full(width * height), compact(edge.Size());
  std::vector<bool> seen(width * height, false);
  for (size_t r = 0; r < height; ++r)
    for (size_t c = 0; c < width; ++c)
      full[r * width + c] = benchmark_seed::PixelSeed(0, r, c);
  for (size_t i = 0; i < edge.Size(); ++i) {
    const auto p = edge.Coordinate(i);
    Check(p.row < height && p.col < width);
    Check(p.row == 0 || p.col == 0 || p.row == height - 1 || p.col == width - 1);
    Check(!seen[p.row * width + p.col]);
    seen[p.row * width + p.col] = true;
    Check(edge.Index(p.row, p.col) == i);
    compact[i] = full[p.row * width + p.col];
  }
  for (size_t r = 0; r < height; ++r)
    for (size_t c = 0; c < width; ++c)
      Check(seen[r * width + c] ==
            (r == 0 || c == 0 || r == height - 1 || c == width - 1));
  size_t w = width, h = height;
  // Repeated rounds exercise updated corner states shared by adjacent edges.
  for (int turn = 0; turn < 20; ++turn) {
    const int orientation = turn % 4;
    PrngRowView<uint64_t, FullView> view{{nullptr, 0}, compact.data(), edge, orientation};
    for (size_t c = 0; c < w; ++c) {
      const auto p = edge.RowZero(orientation, c);
      if (orientation == 0) Check(p.row == 0 && p.col == c);
      if (orientation == 1) Check(p.row == c && p.col == width - 1);
      if (orientation == 2) Check(p.row == height - 1 && p.col == width - 1 - c);
      if (orientation == 3) Check(p.row == height - 1 - c && p.col == 0);
      Check(view.Get(0, c) == full[c]);
      const auto next = benchmark_seed::Mix(full[c]);
      full[c] = next;
      view.Set(0, c, next);
      Check(compact[edge.Index(p.row, p.col)] == next);
    }
    std::vector<uint64_t> rotated(w * h);
    // Independent forward rotation, identical geometry to cuda_rotate.h.
    for (size_t r = 0; r < h; ++r)
      for (size_t c = 0; c < w; ++c)
        rotated[(w - 1 - c) * h + r] = full[r * w + c];
    full.swap(rotated);
    std::swap(w, h);
    if (orientation == 3) {
      for (size_t i = 0; i < edge.Size(); ++i) {
        const auto p = edge.Coordinate(i);
        Check(compact[i] == full[p.row * width + p.col]);
      }
    }
  }
  // Opt-out view delegates all coordinates unchanged to the full map.
  PrngRowView<uint64_t, FullView> fallback{{full.data(), width}, nullptr, edge, 0};
  for (size_t r = 0; r < height; ++r)
    for (size_t c = 0; c < width; ++c) {
      Check(fallback.Get(r, c) == full[r * width + c]);
      fallback.Set(r, c, 42);
      Check(full[r * width + c] == 42);
    }
}
int main() {
  for (bool geom : {false, true})
    for (int seed : {-1, 0, 1})
      for (const char* flag : {static_cast<const char*>(nullptr), "", "0", "1", "true", "10"})
        Check(UseCompactPrng(geom, seed, flag) ==
              (geom && seed == 0 && flag && flag[0] == '1' && flag[1] == '\0'));
  for (size_t w = 1; w <= 35; ++w)
    for (size_t h = 1; h <= 35; ++h) Test(w, h);
  Test(3930, 2210);
  std::cout << "PASS: gate, perimeter bijection, four orientations, 5 round trips, full-map equality, opt-out delegation\n";
}
