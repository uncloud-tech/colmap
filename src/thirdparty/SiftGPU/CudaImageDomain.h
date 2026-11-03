// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#if defined(__CUDACC__) || defined(__HIPCC__)
#define SIFTGPU_DOMAIN_HD __host__ __device__
#else
#define SIFTGPU_DOMAIN_HD
#endif

namespace siftgpu {
// Preserve historical rounded octave domains for supported input widths. For
// previously truncated inputs, capacity may be rounded but the domain is exact.
SIFTGPU_DOMAIN_HD inline int LevelWidth(int width, bool exact_domain) {
  return exact_domain ? (width > 0 ? width : 1) : ((width + 3) / 4) * 4;
}
SIFTGPU_DOMAIN_HD inline bool IsFeatureColumn(int col, int width) {
  return col > 0 && col < width - 1;
}
SIFTGPU_DOMAIN_HD inline int HistogramWidth(int width) {
  return (width + 3) / 4;
}
SIFTGPU_DOMAIN_HD inline int UpsampleRow(int row, int height, bool exact_domain) {
  return exact_domain && row >= height ? height - 1 : row;
}
SIFTGPU_DOMAIN_HD inline int UpsampleNeighbor(
    int col, int row, int width, int height, bool exact_domain) {
  const int next = exact_domain && col + 1 >= width ? width - 1 : col + 1;
  return UpsampleRow(row, height, exact_domain) * width + next;
}
}  // namespace siftgpu

#undef SIFTGPU_DOMAIN_HD
