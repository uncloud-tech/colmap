// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#if !defined(COLMAP_HIP_ENABLED)
#include "colmap/mvs/owned_cuda_execution.h"
#include "colmap/mvs/owned_rotation_storage.h"
#include "colmap/mvs/patch_match_calibration.h"

#include "colmap/mvs/cuda_texture.h"
#include "colmap/mvs/depth_map.h"
#include "colmap/mvs/gpu_mat.h"
#include "colmap/mvs/gpu_mat_prng.h"
#include "colmap/mvs/gpu_mat_ref_image.h"
#include "colmap/mvs/image.h"
#include "colmap/mvs/normal_map.h"
#include "colmap/mvs/patch_match.h"
#include "colmap/util/cuda_to_hip.h"

#include <functional>
#include <iostream>
#include <memory>
#include <vector>

namespace colmap {
namespace mvs {
namespace owned {

class PatchMatchCuda {
 public:
  PatchMatchCuda(const PatchMatchOptions& options,
                 const PatchMatch::Problem& problem);

  ~PatchMatchCuda();

  void Run();
  void DrainForCleanup() noexcept {
    if (stream_) {
      stream_->DrainBeforeDestruction();
    }
  }
  void SetSweepObserver(std::function<void(int, int)> observer) {
    sweep_observer_ = std::move(observer);
  }

  DepthMap GetDepthMap() const;
  NormalMap GetNormalMap() const;
  Mat<float> GetSelProbMap() const;
  std::vector<int> GetConsistentImageIdxs() const;

 private:
  template <int kWindowSize, int kWindowStep>
  void RunWithWindowSizeAndStep();

  ReferenceCalibration CurrentCalibration() const {
    return SnapshotReferenceCalibration(ref_K_host_[rotation_in_half_pi_],
                                         ref_inv_K_host_[rotation_in_half_pi_]);
  }

  void ComputeCudaConfig();

  void BindRefImageTexture();

  void InitRefImage();
  void InitSourceImages();
  void InitTransforms();
  void InitWorkspaceMemory();
  void InitRotationStorage();
  void EmitRotationAllocations();

  // Rotate reference image by 90 degrees in counter-clockwise direction.
  void Rotate();

  const PatchMatchOptions options_;
  const PatchMatch::Problem problem_;

  // Created after device selection; drained before resources, destroyed last.
  std::unique_ptr<OwnedCudaStream> stream_;

  // Dimensions for sweeping from top to bottom, i.e. one thread per column.
  dim3 sweep_block_size_;
  dim3 sweep_grid_size_;
  // Dimensions for element-wise operations, i.e. one thread per pixel.
  dim3 elem_wise_block_size_;
  dim3 elem_wise_grid_size_;

  // Original (not rotated) dimension of reference image.
  size_t ref_width_;
  size_t ref_height_;

  // Rotation of reference image in pi/2. This is equivalent to the number of
  // calls to `rotate` mod 4.
  int rotation_in_half_pi_;

  // Reference and source image input data.
  // Two layered arrays, W*H bytes each (logical resident bytes: 2*W*H).
  // The active pointer borrows the parity owner; both survive until drain.
  std::unique_ptr<CudaArrayLayeredTexture<uint8_t>> ref_image_textures_[2];
  CudaArrayLayeredTexture<uint8_t>* ref_image_texture_ = nullptr;
  std::unique_ptr<CudaArrayLayeredTexture<uint8_t>> src_images_texture_;
  std::unique_ptr<CudaArrayLayeredTexture<float>> src_depth_maps_texture_;

  // Relative poses from rotated versions of reference image to source images
  // corresponding to _rotationInHalfPi:
  //
  //    [S(1), S(2), S(3), ..., S(n)]
  //
  // where n is the number of source images and:
  //
  //    S(i) = [K_i(0, 0), K_i(0, 2), K_i(1, 1), K_i(1, 2), R_i(:), T_i(:)
  //            C_i(:), P(:), P^-1(:)]
  //
  // where i denotes the index of the source image and K is its calibration.
  // R, T, C, P, P^-1 denote the relative rotation, translation, camera
  // center, projection, and inverse projection from there reference to the
  // i-th source image.
  std::unique_ptr<CudaArrayLayeredTexture<float>> poses_texture_[4];

  // Calibration matrix for rotated versions of reference image
  // as {K[0, 0], K[0, 2], K[1, 1], K[1, 2]} corresponding to _rotationInHalfPi.
  float ref_K_host_[4][4];
  float ref_inv_K_host_[4][4];

  // Data for reference image.
  OwnedRotationLayout rotation_layout_;
  OwnedRotationPair<GpuMatRefImage> ref_image_;
  OwnedRotationPair<GpuMat<float>> depth_map_;
  OwnedRotationPair<GpuMat<float>> normal_map_;
  OwnedRotationPair<GpuMat<float>> sel_prob_map_;
  OwnedRotationPair<GpuMat<float>> prev_sel_prob_map_;
  OwnedRotationPair<GpuMat<float>> cost_map_;
  OwnedRotationPair<GpuMatPRNG> rand_state_map_;
  OwnedRotationPair<GpuMat<uint8_t>> consistency_mask_;

  // Shared memory is too small to hold local state for each thread,
  // so this is workspace memory in global memory.
  std::unique_ptr<GpuMat<float>> global_workspace_;
  // Host-only telemetry; never passed to a kernel or active in the ONE path.
  std::function<void(int, int)> sweep_observer_;
};

}  // namespace owned
}  // namespace mvs
}  // namespace colmap

#endif  // !defined(COLMAP_HIP_ENABLED)
