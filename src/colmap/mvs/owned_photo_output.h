// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#if !defined(COLMAP_HIP_ENABLED)
#include "colmap/mvs/gpu_mat.h"
#include "colmap/mvs/image.h"
#include "colmap/mvs/owned_cuda_execution.h"
#include "colmap/mvs/patch_match_options.h"

#include <array>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

namespace colmap::mvs::owned {
class PatchMatchCuda;

// Frozen at producer construction, not reconstructed from a recyclable index
// at publication. This binds camera/pose and logical path, NOT pixel contents.
struct PhotoOutputCamera {
  explicit PhotoOutputCamera(const Image& image)
      : path(image.GetPath()),
        width(image.GetWidth()),
        height(image.GetHeight()) {
    std::memcpy(K.data(), image.GetK(), sizeof(K));
    std::memcpy(R.data(), image.GetR(), sizeof(R));
    std::memcpy(T.data(), image.GetT(), sizeof(T));
    std::memcpy(P.data(), image.GetP(), sizeof(P));
    std::memcpy(inv_P.data(), image.GetInvP(), sizeof(inv_P));
  }
  bool Matches(const Image& image) const {
    return path == image.GetPath() && width == image.GetWidth() &&
           height == image.GetHeight() &&
           std::memcmp(K.data(), image.GetK(), sizeof(K)) == 0 &&
           std::memcmp(R.data(), image.GetR(), sizeof(R)) == 0 &&
           std::memcmp(T.data(), image.GetT(), sizeof(T)) == 0 &&
           std::memcmp(P.data(), image.GetP(), sizeof(P)) == 0 &&
           std::memcmp(inv_P.data(), image.GetInvP(), sizeof(inv_P)) == 0;
  }
  std::filesystem::path path;
  size_t width, height;
  std::array<float, 9> K, R;
  std::array<float, 3> T;
  std::array<float, 12> P, inv_P;
};

// Read-only planar device view. Coordinates are original output orientation.
struct PhotoOutputView {
  const float* ptr;
  size_t pitch;
  size_t width;
  size_t height;
  size_t depth;

  __device__ float Get(size_t row, size_t col, size_t slice = 0) const {
    const auto* bytes = reinterpret_cast<const char*>(ptr);
    return reinterpret_cast<const float*>(bytes +
                                          pitch * (slice * height + row))[col];
  }
};

// COPY-backed, synchronized PHOTO output; not a GEO binding or zero-copy path.
// One object identifies one successful producer version, never a worker slot.
// Caller must serialize producer Run/SnapshotPhotoOutput; no concurrent
// mutation.
class PhotoOutputSnapshot
    : public std::enable_shared_from_this<PhotoOutputSnapshot> {
 public:
  ~PhotoOutputSnapshot() noexcept;
  PhotoOutputSnapshot(const PhotoOutputSnapshot&) = delete;
  PhotoOutputSnapshot& operator=(const PhotoOutputSnapshot&) = delete;

  int ReferenceImageIdx() const { return reference_image_idx_; }
  const PhotoOutputCamera& Camera() const { return camera_; }
  const std::vector<int>& SourceImageIdxs() const { return source_image_idxs_; }
  const PatchMatchOptions& Options() const { return options_; }
  int DeviceIndex() const { return device_; }
  size_t Width() const { return depth_->GetWidth(); }
  size_t Height() const { return depth_->GetHeight(); }
  size_t DepthBytes() const { return depth_->GetPitch() * Height(); }
  size_t NormalBytes() const { return normal_->GetPitch() * Height() * 3; }
  // Additional pitched storage at copy peak, while producer storage is live.
  // No admission credits: producer/workspace and all other live copies are
  // extra.
  size_t RetainedBytes() const { return DepthBytes() + NormalBytes(); }
  size_t CopyAdditionalPeakBytes() const { return RetainedBytes(); }
  // Source current maps plus retained copies at publication; excludes all
  // other producer rotation/scratch/texture/pool allocations, NOT process peak.
  size_t CopyLiveMapBytes() const { return source_bytes_ + RetainedBytes(); }

  // Synchronous lease on an EXISTING owner stream. Keep all view uses, graph
  // launches AND graph teardown inside callback. No view/graph may escape it.
  // Completion is fenced on both callback success and exception. The strong
  // local owner permits callback to reset the caller's final snapshot handle.
  template <typename Consumer>
  void WithDeviceMaps(OwnedCudaStream& stream, Consumer consume) const {
    auto keep_alive = shared_from_this();
    int current = -1;
    CUDA_SAFE_CALL(cudaGetDevice(&current));
    if (current != device_ || stream.DeviceIndex() != device_) {
      throw std::invalid_argument("PHOTO output consumer device mismatch");
    }
    ScopedStreamCompletion completion(stream.Get());
    try {
      consume(View(*depth_), View(*normal_));
    } catch (...) {
      stream.Activate();
      throw;
    }
    stream.Activate();
    completion.Wait();
  }

 private:
  friend class PatchMatchCuda;
  PhotoOutputSnapshot(int reference_image_idx,
                      PhotoOutputCamera camera,
                      std::vector<int> source_image_idxs,
                      PatchMatchOptions options,
                      int device,
                      const GpuMat<float>& depth,
                      const GpuMat<float>& normal,
                      cudaStream_t stream);
  static PhotoOutputView View(const GpuMat<float>& map) {
    return {map.GetPtr(),
            map.GetPitch(),
            map.GetWidth(),
            map.GetHeight(),
            map.GetDepth()};
  }

  const int reference_image_idx_;
  const PhotoOutputCamera camera_;
  const std::vector<int> source_image_idxs_;
  const PatchMatchOptions options_;
  const int device_;
  const size_t source_bytes_;
  std::unique_ptr<GpuMat<float>> depth_;
  std::unique_ptr<GpuMat<float>> normal_;
};
}  // namespace colmap::mvs::owned
#endif
