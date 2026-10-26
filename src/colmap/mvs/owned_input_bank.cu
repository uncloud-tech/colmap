// SPDX-License-Identifier: BSD-3-Clause
#include "colmap/mvs/cuda_texture.h"
#include "colmap/mvs/owned_cuda_execution.h"
#include "colmap/mvs/owned_input_bank.h"

#include <algorithm>
#include <limits>
#include <utility>

namespace colmap::mvs::owned {
namespace {
cudaTextureDesc TextureDescription(cudaTextureFilterMode filter,
                                   cudaTextureReadMode read) {
  cudaTextureDesc desc{};
  desc.addressMode[0] = cudaAddressModeBorder;
  desc.addressMode[1] = cudaAddressModeBorder;
  desc.addressMode[2] = cudaAddressModeBorder;
  desc.filterMode = filter;
  desc.readMode = read;
  return desc;
}
}  // namespace

struct ResidentInputBank::Impl {
  int device;
  size_t width;
  size_t height;
  std::vector<std::pair<size_t, size_t>> dimensions;
  // Stream dies last; point view dies before its array owner.
  std::unique_ptr<OwnedCudaStream> stream;
  std::unique_ptr<CudaArrayLayeredTexture<uint8_t>> grayscale;
  std::unique_ptr<CudaArrayLayeredTexture<float>> depth;
  std::unique_ptr<CudaArrayTextureView> reference_grayscale;

  Impl(const HostInputBank& host, int device_index)
      : device(device_index), width(host.Width()), height(host.Height()) {
    THROW_CHECK_GE(device, 0);
    THROW_CHECK_GT(width, 0);
    THROW_CHECK_GT(height, 0);
    THROW_CHECK_GT(host.Layers().size(), 0);
    THROW_CHECK_LE(width, std::numeric_limits<size_t>::max() / height);
    THROW_CHECK_LE(width * height,
                   std::numeric_limits<size_t>::max() / sizeof(float));
    THROW_CHECK_LE(host.Layers().size(),
                   static_cast<size_t>(std::numeric_limits<int>::max()));
    CUDA_SAFE_CALL(cudaSetDevice(device));
    stream = std::make_unique<OwnedCudaStream>();
    try {
      grayscale = std::make_unique<CudaArrayLayeredTexture<uint8_t>>(
          TextureDescription(cudaFilterModeLinear, cudaReadModeNormalizedFloat),
          width,
          height,
          host.Layers().size());
      depth = std::make_unique<CudaArrayLayeredTexture<float>>(
          TextureDescription(cudaFilterModePoint, cudaReadModeElementType),
          width,
          height,
          host.Layers().size());
      reference_grayscale = grayscale->CreateView(
          TextureDescription(cudaFilterModePoint, cudaReadModeNormalizedFloat));
      // Bound staging to one padded layer of each type. Every copy fences
      // before staging can be reused/freed, including failed partial uploads.
      std::vector<uint8_t> gray_layer(width * height);
      std::vector<float> depth_layer(width * height);
      for (size_t i = 0; i < host.Layers().size(); ++i) {
        const auto& layer = host.Layers()[i];
        const size_t w = layer.depth.GetWidth();
        const size_t h = layer.depth.GetHeight();
        THROW_CHECK_GT(w, 0);
        THROW_CHECK_GT(h, 0);
        THROW_CHECK_LE(w, width);
        THROW_CHECK_LE(h, height);
        THROW_CHECK_EQ(layer.grayscale.Width(), w);
        THROW_CHECK_EQ(layer.grayscale.Height(), h);
        const auto& gray = layer.grayscale.RowMajorData();
        THROW_CHECK_EQ(gray.size(), w * h);
        THROW_CHECK_EQ(layer.depth.GetData().size(), w * h);
        std::fill(gray_layer.begin(), gray_layer.end(), uint8_t{0});
        std::fill(depth_layer.begin(), depth_layer.end(), 0.0f);
        for (size_t row = 0; row < h; ++row) {
          std::copy_n(
              gray.data() + row * w, w, gray_layer.data() + row * width);
          std::copy_n(layer.depth.GetPtr() + row * w,
                      w,
                      depth_layer.data() + row * width);
        }
        grayscale->CopyLayerFromHost(i, gray_layer.data(), stream->Get());
        depth->CopyLayerFromHost(i, depth_layer.data(), stream->Get());
        dimensions.emplace_back(w, h);
      }
      stream->Synchronize();
    } catch (...) {
      stream->DrainBeforeDestruction();
      throw;
    }
  }
  ~Impl() {
    // Caller has already joined all borrowers. Activate the allocation device
    // even when retirement happens on a different host thread/device context.
    stream->DrainBeforeDestruction();
  }
};

ResidentInputBank::ResidentInputBank(const HostInputBank& host,
                                     int device_index)
    : impl_(std::make_unique<Impl>(host, device_index)) {}
ResidentInputBank::~ResidentInputBank() = default;
ResidentInputBank::ResidentInputBank(ResidentInputBank&&) noexcept = default;
ResidentInputBank& ResidentInputBank::operator=(ResidentInputBank&&) noexcept =
    default;
size_t ResidentInputBank::Width() const { return impl_->width; }
size_t ResidentInputBank::Height() const { return impl_->height; }
size_t ResidentInputBank::NumLayers() const { return impl_->dimensions.size(); }
int ResidentInputBank::DeviceIndex() const { return impl_->device; }
uint64_t ResidentInputBank::GrayscaleTexture() const {
  return impl_->grayscale->GetObj();
}
uint64_t ResidentInputBank::ReferenceGrayscaleTexture() const {
  return impl_->reference_grayscale->GetObj();
}
uint64_t ResidentInputBank::DepthTexture() const {
  return impl_->depth->GetObj();
}
size_t ResidentInputBank::LayerWidth(size_t layer) const {
  return impl_->dimensions.at(layer).first;
}
size_t ResidentInputBank::LayerHeight(size_t layer) const {
  return impl_->dimensions.at(layer).second;
}
}  // namespace colmap::mvs::owned
