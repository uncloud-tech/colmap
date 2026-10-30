// SPDX-License-Identifier: BSD-3-Clause
#include "colmap/mvs/owned_geo_photo_input.h"

#if !defined(COLMAP_HIP_ENABLED)
#include "colmap/mvs/optimized_photo_options.h"
#include "colmap/mvs/owned_pair_validation.h"

#include <algorithm>
#include <utility>

namespace colmap::mvs::owned {
namespace {
void Require(bool condition) {
  if (!condition)
    throw std::invalid_argument("Invalid fresh PHOTO -> GEO binding");
}

void ValidateSnapshot(const GeoPhotoInput::Snapshot& snapshot,
                      int index,
                      const PatchMatch::Problem& problem,
                      int device) {
  Require(snapshot && index >= 0 &&
          static_cast<size_t>(index) < problem.images->size());
  const auto& image = problem.images->at(index);
  Require(snapshot->ReferenceImageIdx() == index &&
          snapshot->DeviceIndex() == device &&
          !snapshot->Options().geom_consistency &&
          snapshot->Width() == image.GetWidth() &&
          snapshot->Height() == image.GetHeight() &&
          snapshot->Camera().Matches(image));
}
}  // namespace

void GeoPhotoInput::Validate(const PatchMatchOptions& options,
                             const PatchMatch::Problem& problem) const {
  Require(options.geom_consistency && problem.images && reference_ &&
          !problem.resident_bank && problem.resident_ref_layer == -1 &&
          problem.resident_src_layers.empty() && !problem.resident_ref_normal &&
          !problem.photo_grayscale.bank &&
          problem.photo_grayscale.layers.empty() && !problem.depth_maps &&
          !problem.normal_maps && !sources_.empty() &&
          sources_.size() == problem.src_image_idxs.size());
  // Compare the complete intended GEO configuration, not the PHOTO producer's
  // options. Reuse the exhaustive bit-exact option wire (no struct padding).
  // This adapter has no workspace/config paths: supply the same neutral path
  // to both encodings in this call so only options participate in equality.
  Require(OptimizedPhotoOptions(geo_options_, ".", ".") ==
          OptimizedPhotoOptions(options, ".", "."));
  // The owned solver supports one explicit device, not an automatic/multi-GPU
  // list. Validate without changing context or allocating on an unknown GPU.
  size_t consumed = 0;
  const int device = std::stoi(options.gpu_index, &consumed);
  Require(consumed == options.gpu_index.size() && device >= 0 &&
          device == device_);
  ValidateSnapshot(reference_, problem.ref_image_idx, problem, device_);
  for (size_t i = 0; i < sources_.size(); ++i) {
    ValidateSnapshot(sources_[i], problem.src_image_idxs[i], problem, device_);
  }
}

GeoPhotoInput::GeoPhotoInput(const PatchMatchOptions& options,
                             const PatchMatch::Problem& problem,
                             Snapshot reference,
                             std::vector<Snapshot> ordered_sources)
    : geo_options_(options),
      reference_(std::move(reference)),
      sources_(std::move(ordered_sources)),
      device_(reference_ ? reference_->DeviceIndex() : -1) {
  Validate(options, problem);  // ALL binding refusals before GPU allocations
  int current = -1;
  CUDA_SAFE_CALL(cudaGetDevice(&current));
  Require(current == device_);
  size_t width = 0, height = 0;
  for (const auto& source : sources_) {
    width = std::max(width, source->Width());
    height = std::max(height, source->Height());
  }
  source_texture_bytes_ = CheckedMultiply(
      CheckedMultiply(CheckedMultiply(width, height), sources_.size()),
      sizeof(float));
  OwnedCudaStream stream;
  GpuMat<float> staging(width, height, sources_.size());
  source_staging_bytes_ = CheckedMultiply(
      CheckedMultiply(staging.GetPitch(), height), sources_.size());
  ScopedStreamCompletion completion(stream.Get());
  CUDA_SAFE_CALL(cudaMemset2DAsync(staging.GetPtr(),
                                   staging.GetPitch(),
                                   0,
                                   width * sizeof(float),
                                   height * sources_.size(),
                                   stream.Get()));
  for (size_t layer = 0; layer < sources_.size(); ++layer) {
    sources_[layer]->WithDeviceMaps(stream, [&](auto depth, auto) {
      auto* destination = reinterpret_cast<char*>(staging.GetPtr()) +
                          layer * height * staging.GetPitch();
      CUDA_SAFE_CALL(cudaMemcpy2DAsync(destination,
                                       staging.GetPitch(),
                                       depth.ptr,
                                       depth.pitch,
                                       depth.width * sizeof(float),
                                       depth.height,
                                       cudaMemcpyDeviceToDevice,
                                       stream.Get()));
    });
  }
  // Identical addressing, point sampling, element type and padding to the
  // ordinary private GEO depth texture. Original orientation, no arithmetic.
  cudaTextureDesc desc{};
  desc.addressMode[0] = desc.addressMode[1] = desc.addressMode[2] =
      cudaAddressModeBorder;
  desc.filterMode = cudaFilterModePoint;
  desc.readMode = cudaReadModeElementType;
  desc.normalizedCoords = false;
  source_depth_ =
      CudaArrayLayeredTexture<float>::FromGpuMat(desc, staging, stream.Get());
  completion.Wait();  // staging may be freed only after the final array copy
}

GeoPhotoInput::~GeoPhotoInput() noexcept {
  // Consumer contract: PatchMatchCuda drains before releasing this binding.
  // Setup has already fenced; no adapter-owned work remains in flight.
  int previous = -1;
  if (cudaGetDevice(&previous) != cudaSuccess ||
      cudaSetDevice(device_) != cudaSuccess) {
    std::terminate();
  }
  source_depth_.reset();
  if (cudaSetDevice(previous) != cudaSuccess) std::terminate();
}

void GeoPhotoInput::CopyReferenceMaps(OwnedCudaStream& stream,
                                      GpuMat<float>& depth,
                                      GpuMat<float>& normal) const {
  Require(stream.DeviceIndex() == device_ &&
          depth.GetWidth() == reference_->Width() &&
          depth.GetHeight() == reference_->Height() && depth.GetDepth() == 1 &&
          normal.GetWidth() == reference_->Width() &&
          normal.GetHeight() == reference_->Height() && normal.GetDepth() == 3);
  stream.Activate();
  reference_->WithDeviceMaps(stream, [&](auto input_depth, auto input_normal) {
    for (const auto& copy : {std::make_pair(input_depth, &depth),
                             std::make_pair(input_normal, &normal)}) {
      CUDA_SAFE_CALL(cudaMemcpy2DAsync(copy.second->GetPtr(),
                                       copy.second->GetPitch(),
                                       copy.first.ptr,
                                       copy.first.pitch,
                                       copy.first.width * sizeof(float),
                                       copy.first.height * copy.first.depth,
                                       cudaMemcpyDeviceToDevice,
                                       stream.Get()));
    }
  });
}

size_t GeoPhotoInput::SnapshotRetainedBytes() const {
  size_t bytes = reference_->RetainedBytes();
  for (size_t i = 0; i < sources_.size(); ++i) {
    if (sources_[i] == reference_) continue;
    if (std::find(sources_.begin(), sources_.begin() + i, sources_[i]) ==
        sources_.begin() + i) {
      bytes = CheckedAdd(bytes, sources_[i]->RetainedBytes());
    }
  }
  return bytes;
}
}  // namespace colmap::mvs::owned
#endif
