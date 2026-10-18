// SPDX-License-Identifier: BSD-3-Clause
#include "colmap/mvs/cuda_texture.h"
#include "colmap/mvs/gpu_mat_ref_image.h"
#include "colmap/mvs/owned_cuda_execution.h"

#include <gtest/gtest.h>

namespace colmap::mvs::owned {
namespace {
__global__ void Sample(cudaTextureObject_t direct, cudaTextureObject_t bank,
                       const int* layers, GpuMatView<float> result) {
  const int i = threadIdx.x;
  if (i >= result.GetWidth()) return;
  // Includes negative border, fractional edge, padded region and far border.
  const float x = -1.0f + 0.25f * i;
  const float y = (i % 5) * 0.5f - 0.5f;
  result.Set(0, i, 0, tex2DLayered<float>(direct, x, y, 0));
  result.Set(0, i, 1, tex2DLayered<float>(bank, x, y, layers[0]));
}

cudaTextureDesc Description(cudaTextureFilterMode filter,
                            cudaTextureReadMode read) {
  cudaTextureDesc desc{};
  desc.addressMode[0] = desc.addressMode[1] = desc.addressMode[2] =
      cudaAddressModeBorder;
  desc.filterMode = filter;
  desc.readMode = read;
  return desc;
}

TEST(ResidentInputBankSampler, PaddingMappingAndReferenceInitialization) {
  OwnedCudaStream stream;
  const std::vector<uint8_t> gray = {17, 80, 240, 90, 1, 255};
  const std::vector<uint8_t> padded = {17, 80, 240, 0, 0,
                                       90, 1, 255, 0, 0,
                                       0, 0, 0, 0, 0,
                                       0, 0, 0, 0, 0};
  auto direct = CudaArrayLayeredTexture<uint8_t>::FromHostArray(
      Description(cudaFilterModeLinear, cudaReadModeNormalizedFloat),
      3, 2, 1, gray.data(), stream.Get());
  auto bank = std::make_unique<CudaArrayLayeredTexture<uint8_t>>(
      Description(cudaFilterModeLinear, cudaReadModeNormalizedFloat), 5, 4, 3);
  // Nonidentity mapping with deliberately different neighboring layers.
  std::vector<uint8_t> other(20, 123);
  bank->CopyLayerFromHost(0, other.data(), stream.Get());
  bank->CopyLayerFromHost(1, other.data(), stream.Get());
  bank->CopyLayerFromHost(2, padded.data(), stream.Get());
  GpuMat<int> mapping(1, 1);
  const int layer = 2;
  mapping.CopyToDevice(&layer, sizeof(int), stream.Get());
  GpuMat<float> sampled(32, 1, 2);
  ScopedStreamCompletion completion(stream.Get());
  Sample<<<1, 32, 0, stream.Get()>>>(direct->GetObj(), bank->GetObj(),
                                  mapping.GetPtr(), sampled.View());
  auto values = sampled.CopyToMat(stream.Get());
  for (size_t i = 0; i < 32; ++i) {
    EXPECT_EQ(values.GetData()[i], values.GetData()[32 + i]);
  }

  auto point = bank->CreateView(
      Description(cudaFilterModePoint, cudaReadModeNormalizedFloat));
  GpuMatRefImage private_ref(3, 2);
  GpuMatRefImage resident_ref(3, 2);
  private_ref.Filter(gray.data(), 2, 1, 3.0f, 0.2f, stream.Get());
  resident_ref.FilterResident(point->GetObj(), 2, 2, 1, 3.0f, 0.2f,
                              stream.Get());
  EXPECT_EQ(private_ref.image->CopyToMat(stream.Get()).GetData(),
            resident_ref.image->CopyToMat(stream.Get()).GetData());
  EXPECT_EQ(private_ref.sum_image->CopyToMat(stream.Get()).GetData(),
            resident_ref.sum_image->CopyToMat(stream.Get()).GetData());
  EXPECT_EQ(private_ref.squared_sum_image->CopyToMat(stream.Get()).GetData(),
            resident_ref.squared_sum_image->CopyToMat(stream.Get()).GetData());
  completion.Wait();
}

TEST(ResidentInputBankSampler, DepthElementTypeAndPadding) {
  OwnedCudaStream stream;
  const std::vector<float> depth = {1.25f, 7.0f, 2.5f, 19.0f, 4.0f, 8.0f};
  const std::vector<float> padded = {1.25f, 7.0f, 2.5f, 0, 0,
                                     19.0f, 4.0f, 8.0f, 0, 0,
                                     0, 0, 0, 0, 0,
                                     0, 0, 0, 0, 0};
  const auto desc = Description(cudaFilterModePoint, cudaReadModeElementType);
  auto direct = CudaArrayLayeredTexture<float>::FromHostArray(
      desc, 3, 2, 1, depth.data(), stream.Get());
  auto bank = std::make_unique<CudaArrayLayeredTexture<float>>(desc, 5, 4, 3);
  bank->CopyLayerFromHost(2, padded.data(), stream.Get());
  GpuMat<int> mapping(1, 1);
  const int layer = 2;
  mapping.CopyToDevice(&layer, sizeof(int), stream.Get());
  GpuMat<float> sampled(32, 1, 2);
  ScopedStreamCompletion completion(stream.Get());
  Sample<<<1, 32, 0, stream.Get()>>>(direct->GetObj(), bank->GetObj(),
                                  mapping.GetPtr(), sampled.View());
  auto values = sampled.CopyToMat(stream.Get());
  for (size_t i = 0; i < 32; ++i) {
    EXPECT_EQ(values.GetData()[i], values.GetData()[32 + i]);
  }
  completion.Wait();
}
}  // namespace
}  // namespace colmap::mvs::owned
