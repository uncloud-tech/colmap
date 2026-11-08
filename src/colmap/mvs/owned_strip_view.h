// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include "colmap/mvs/gpu_mat.h"

namespace colmap::mvs::owned {
// Geometry stays global; only source-scaled mutable storage is column local.
// The physical view retains its allocation's slice stride even for rotations.
template <class T>
struct StripVolumeView {
  GpuMatView<T> physical;
  size_t origin, width, height;
  __host__ __device__ StripVolumeView(GpuMatView<T> view)
      : physical(view), origin(0), width(view.width), height(view.height) {}
  __host__ __device__
  StripVolumeView(GpuMatView<T> view, size_t first, size_t w, size_t h)
      : physical(view), origin(first), width(w), height(h) {}
  __host__ __device__ size_t GetWidth() const { return width; }
  __host__ __device__ size_t GetHeight() const { return height; }
  __host__ __device__ size_t GetDepth() const { return physical.depth; }
  __device__ T Get(size_t r, size_t c, size_t s = 0) const {
    return physical.Get(r, c - origin, s);
  }
  __device__ void Set(size_t r, size_t c, size_t s, T value) {
    physical.Set(r, c - origin, s, value);
  }
  __device__ void Set(size_t r, size_t c, T value) {
    physical.Set(r, c - origin, value);
  }
};
}  // namespace colmap::mvs::owned
