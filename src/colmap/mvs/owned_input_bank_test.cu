// SPDX-License-Identifier: BSD-3-Clause
#include "colmap/mvs/cuda_texture.h"
#include "colmap/mvs/gpu_mat_ref_image.h"
#include "colmap/mvs/owned_cuda_execution.h"
#include "colmap/mvs/owned_input_bank.h"
#include "colmap/mvs/patch_match.h"
#include "colmap/mvs/patch_match_cuda_owned.h"
#include "colmap/util/testing.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <future>
#include <mutex>
#include <stdexcept>

#include <gtest/gtest.h>

namespace colmap::mvs::owned {
namespace {
::testing::AssertionResult SameFloatBits(const std::vector<float>& expected,
                                         const std::vector<float>& actual) {
  static_assert(sizeof(float) == sizeof(uint32_t));
  if (expected.size() != actual.size()) {
    return ::testing::AssertionFailure() << "Different float-vector sizes";
  }
  for (size_t i = 0; i < expected.size(); ++i) {
    uint32_t expected_bits = 0;
    uint32_t actual_bits = 0;
    std::memcpy(&expected_bits, &expected[i], sizeof(expected_bits));
    std::memcpy(&actual_bits, &actual[i], sizeof(actual_bits));
    if (expected_bits != actual_bits) {
      return ::testing::AssertionFailure()
             << "Different float bits at element " << i << ": " << expected_bits
             << " vs " << actual_bits;
    }
  }
  return ::testing::AssertionSuccess();
}

// Host-only guard: float equality would miss signed-zero differences and reject
// identical NaN payloads. The GPU comparisons below require byte identity.
TEST(ResidentGrayscaleBank, FloatBitsComparisonGuard) {
  const std::array<uint32_t, 2> bits = {0u, 0x7fc12345u};
  std::vector<float> expected(2);
  std::vector<float> actual(2);
  std::memcpy(expected.data(), bits.data(), bits.size() * sizeof(uint32_t));
  std::memcpy(actual.data(), bits.data(), bits.size() * sizeof(uint32_t));
  EXPECT_TRUE(SameFloatBits(expected, actual));
  const uint32_t negative_zero = 0x80000000u;
  std::memcpy(&actual[0], &negative_zero, sizeof(negative_zero));
  EXPECT_FALSE(SameFloatBits(expected, actual));
  std::memcpy(actual.data(), bits.data(), bits.size() * sizeof(uint32_t));
  const uint32_t other_nan = 0x7fc12346u;
  std::memcpy(&actual[1], &other_nan, sizeof(other_nan));
  EXPECT_FALSE(SameFloatBits(expected, actual));
  actual.pop_back();
  EXPECT_FALSE(SameFloatBits(expected, actual));
  EXPECT_TRUE(SameFloatBits({}, {}));
}

__global__ void Sample(cudaTextureObject_t direct,
                       cudaTextureObject_t bank,
                       const int* layers,
                       GpuMatView<float> result) {
  const int i = threadIdx.x;
  if (i >= result.GetWidth()) return;
  // Includes negative border, fractional edge, padded region and far border.
  const float x = -1.0f + 0.25f * i;
  const float y = (i % 11) * 0.5f - 1.0f;
  result.Set(0, i, 0, SampleSourceGrayscale(direct, x, y, 0, nullptr));
  result.Set(0, i, 1, SampleSourceGrayscale(bank, x, y, 0, layers));
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
  const std::vector<uint8_t> padded = {17, 80, 240, 0, 0, 90, 1, 255, 0, 0,
                                       0,  0,  0,   0, 0, 0,  0, 0,   0, 0};
  auto direct = CudaArrayLayeredTexture<uint8_t>::FromHostArray(
      Description(cudaFilterModeLinear, cudaReadModeNormalizedFloat),
      3,
      2,
      1,
      gray.data(),
      stream.Get());
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
  Sample<<<1, 32, 0, stream.Get()>>>(
      direct->GetObj(), bank->GetObj(), mapping.GetPtr(), sampled.View());
  auto values = sampled.CopyToMat(stream.Get());
  for (size_t i = 0; i < 32; ++i) {
    EXPECT_EQ(values.GetData()[i], values.GetData()[32 + i]);
  }

  auto point = bank->CreateView(
      Description(cudaFilterModePoint, cudaReadModeNormalizedFloat));
  GpuMatRefImage private_ref(3, 2);
  GpuMatRefImage resident_ref(3, 2);
  private_ref.Filter(gray.data(), 2, 1, 3.0f, 0.2f, stream.Get());
  resident_ref.FilterResident(
      point->GetObj(), 2, 2, 1, 3.0f, 0.2f, stream.Get());
  EXPECT_EQ(private_ref.image->CopyToMat(stream.Get()).GetData(),
            resident_ref.image->CopyToMat(stream.Get()).GetData());
  EXPECT_EQ(private_ref.sum_image->CopyToMat(stream.Get()).GetData(),
            resident_ref.sum_image->CopyToMat(stream.Get()).GetData());
  EXPECT_EQ(private_ref.squared_sum_image->CopyToMat(stream.Get()).GetData(),
            resident_ref.squared_sum_image->CopyToMat(stream.Get()).GetData());
  completion.Wait();
}

// This fixture deliberately has no depth/normal files. The second job's private
// padded extent (3x2) is smaller than the shared union (5x4).
TEST(ResidentGrayscaleBank, OrderedPhotoSourcesAndLifetime) {
  const auto root = CreateTestDir();
  const float K[9] = {10, 0, 1, 0, 10, 1, 0, 0, 1};
  const float R[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  const float T[3] = {0, 0, 0};
  std::vector<Image> images;
  std::vector<GrayscaleInput> inputs;
  for (int i = 0; i < 3; ++i) {
    const int w = i == 2 ? 5 : 3;
    const int h = i == 2 ? 4 : 2;
    const auto path = root / (std::to_string(i) + ".png");
    Bitmap bitmap(w, h, true);
    for (int y = 0; y < h; ++y) {
      for (int x = 0; x < w; ++x) {
        ASSERT_TRUE(bitmap.SetPixel(
            x, y, BitmapColor<uint8_t>(i * 70 + 17 + x + y * 3)));
      }
    }
    ASSERT_TRUE(bitmap.Write(path));
    images.emplace_back(path, w, h, K, R, T);
    inputs.push_back({images.back(), -1, CaptureFileIdentity(path)});
  }
  HostGrayscaleBank host(
      {{inputs[2], inputs[0], inputs[1]}, {inputs[1], inputs[0]}});
  int device = -1;
  CUDA_SAFE_CALL(cudaGetDevice(&device));
  auto resident = std::make_shared<ResidentGrayscaleBank>(host, device);
  auto binding = resident->BindJob(1);
  EXPECT_EQ(binding.layers, (std::vector<int>{2, 1}));
  EXPECT_EQ(resident->Width(), 5);
  EXPECT_EQ(resident->Height(), 4);
  std::vector<Image> requested = {images[1], images[0]};
  EXPECT_NO_THROW(resident->ValidateSources(binding.layers, requested, device));
  EXPECT_THROW(resident->ValidateSources({1, 2}, requested, device),
               std::exception);  // wrong SAME-SIZE images
  EXPECT_THROW(resident->ValidateSources({2}, requested, device),
               std::exception);
  EXPECT_THROW(resident->ValidateSources({2, 99}, requested, device),
               std::exception);
  EXPECT_THROW(resident->ValidateSources(binding.layers, requested, device + 1),
               std::exception);
  auto wrong_dimensions = requested;
  wrong_dimensions[0].Downsize(1, 1);
  EXPECT_THROW(
      resident->ValidateSources(binding.layers, wrong_dimensions, device),
      std::exception);

  // Exercise the real PHOTO solver twice, preserving exact source order and
  // private reference preparation. No map files or source bitmaps in shared
  // mode.
  for (size_t job = 0; job < 2; ++job) {
    // Host job 0 includes image 0, so use image 2 as the private reference and
    // job 1's two ordered sources. Both permutations borrow the same upload.
    PatchMatchOptions options;
    options.gpu_index = std::to_string(device);
    options.geom_consistency = false;
    options.depth_min = 1;
    options.depth_max = 5;
    options.num_iterations = 2;
    options.num_samples = 2;
    options.filter = false;
    options.random_seed = 0;
    std::vector<Image> private_images = images;
    for (auto& image : private_images) {
      Bitmap bitmap;
      ASSERT_TRUE(bitmap.Read(image.GetPath(), false));
      image.SetBitmap(std::move(bitmap));
    }
    PatchMatch::Problem problem;
    problem.ref_image_idx = 2;
    problem.src_image_idxs =
        job == 0 ? std::vector<int>{1, 0} : std::vector<int>{0, 1};
    problem.images = &private_images;
    PatchMatch direct(options, problem, PatchMatchExecution::kOwnedCuda);
    direct.Run();
    problem.photo_grayscale = resident->BindJob(1);
    if (job == 1)
      std::reverse(problem.photo_grayscale.layers.begin(),
                   problem.photo_grayscale.layers.end());
    for (int idx : problem.src_image_idxs) {
      // Restore metadata-only images: SetBitmap rejects empty dimensions.
      private_images[idx] = images[idx];
      ASSERT_TRUE(private_images[idx].GetBitmap().IsEmpty());
    }
    options.owned_cuda_graphs = true;
    PatchMatch shared(options, problem, PatchMatchExecution::kOwnedCuda);
    EXPECT_NO_THROW(shared.Check());
    shared.Run();
    EXPECT_TRUE(SameFloatBits(direct.GetDepthMap().GetData(),
                              shared.GetDepthMap().GetData()));
    EXPECT_TRUE(SameFloatBits(direct.GetNormalMap().GetData(),
                              shared.GetNormalMap().GetData()));
    EXPECT_TRUE(SameFloatBits(direct.GetSelProbMap().GetData(),
                              shared.GetSelProbMap().GetData()));

    auto invalid = problem;
    std::reverse(invalid.photo_grayscale.layers.begin(),
                 invalid.photo_grayscale.layers.end());
    PatchMatch wrong_order(options, invalid, PatchMatchExecution::kOwnedCuda);
    EXPECT_THROW(wrong_order.Check(), std::exception);
    invalid = problem;
    invalid.resident_ref_layer = 0;
    PatchMatch mixed(options, invalid, PatchMatchExecution::kOwnedCuda);
    EXPECT_THROW(mixed.Check(), std::exception);
    invalid = problem;
    std::vector<DepthMap> maps;
    invalid.depth_maps = &maps;
    EXPECT_THROW(
        (PatchMatch(options, invalid, PatchMatchExecution::kOwnedCuda)),
        std::exception);
    PatchMatch stock(options, problem);
    EXPECT_THROW(stock.Check(), std::exception);
    options.geom_consistency = true;
    options.filter = true;
    PatchMatch geo(options, problem, PatchMatchExecution::kOwnedCuda);
    EXPECT_THROW(geo.Check(), std::exception);
  }

  OwnedCudaStream stream;
  // All borrowers retain the resident through completion, not merely enqueue.
  std::weak_ptr<const ResidentGrayscaleBank> lifetime = resident;
  resident.reset();
  for (size_t job = 0; job < 2; ++job) {
    auto ordered = binding.bank->BindJob(job);
    const auto& indices = host.SourceLayers(job);
    for (size_t ordinal = 0; ordinal < indices.size(); ++ordinal) {
      const auto& layer = *host.Layers().at(indices[ordinal]);
      std::vector<uint8_t> private_pixels(layer.width * layer.height);
      for (size_t row = 0; row < layer.height; ++row) {
        std::copy_n(layer.pixels.data() + row * layer.row_stride,
                    layer.width,
                    private_pixels.data() + row * layer.width);
      }
      auto direct = CudaArrayLayeredTexture<uint8_t>::FromHostArray(
          Description(cudaFilterModeLinear, cudaReadModeNormalizedFloat),
          layer.width,
          layer.height,
          1,
          private_pixels.data(),
          stream.Get());
      GpuMat<int> mapping(1, 1);
      mapping.CopyToDevice(&ordered.layers[ordinal], sizeof(int), stream.Get());
      GpuMat<float> sampled(32, 1, 2);
      ScopedStreamCompletion completion(stream.Get());
      Sample<<<1, 32, 0, stream.Get()>>>(direct->GetObj(),
                                         ordered.bank->GrayscaleTexture(),
                                         mapping.GetPtr(),
                                         sampled.View());
      auto values = sampled.CopyToMat(stream.Get());
      for (size_t i = 0; i < 32; ++i) {
        EXPECT_EQ(values.GetData()[i], values.GetData()[32 + i]);
      }
      completion.Wait();
    }
  }
  EXPECT_FALSE(lifetime.expired());
  stream.Synchronize();
  // Two actual owners hold the allocation after the caller drops its binding.
  // One runs successfully, the other is cancelled during graph submission.
  Bitmap reference;
  ASSERT_TRUE(reference.Read(images[2].GetPath(), false));
  images[2].SetBitmap(std::move(reference));
  PatchMatchOptions options;
  options.gpu_index = std::to_string(device);
  options.geom_consistency = false;
  options.depth_min = 1;
  options.depth_max = 5;
  options.random_seed = 0;
  options.num_iterations = 2;
  options.num_samples = 2;
  options.filter = false;
  options.owned_cuda_graphs = true;
  PatchMatch::Problem problem;
  problem.images = &images;
  problem.ref_image_idx = 2;
  problem.src_image_idxs = {1, 0};
  problem.photo_grayscale = binding;
  auto success = std::make_unique<PatchMatch>(
      options, problem, PatchMatchExecution::kOwnedCuda);
  auto cancelled = std::make_unique<PatchMatch>(
      options, problem, PatchMatchExecution::kOwnedCuda);
  auto coordinator = std::make_shared<GraphLaunchControl>(1);
  success->PrepareOwned();
  cancelled->PrepareOwned(32, coordinator, 0);
  problem.photo_grayscale = {};
  binding = {};
  EXPECT_FALSE(lifetime.expired());
  success->RunPreparedOwned();
  success.reset();
  EXPECT_FALSE(lifetime.expired());
  coordinator->Cancel(std::make_exception_ptr(std::runtime_error("cancelled")));
  EXPECT_THROW(cancelled->RunPreparedOwned(), std::exception);
  cancelled.reset();  // fences even on the exceptional/cancelled path
  EXPECT_TRUE(lifetime.expired());
  std::filesystem::remove_all(root);
}

// Concurrent CPU submission is intentional; kernel overlap is not required.
TEST(ResidentGrayscaleBank, ConcurrentPhotoConsumersOwnBindings) {
  const auto root = CreateTestDir();
  const float K[9] = {10, 0, 1, 0, 10, 1, 0, 0, 1};
  const float R[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  const float T[3] = {0, 0, 0};
  std::vector<Image> images;
  std::vector<GrayscaleInput> inputs;
  for (int i = 0; i < 3; ++i) {
    const int w = i == 2 ? 5 : 3;
    const int h = i == 2 ? 4 : 2;
    const auto path = root / (std::to_string(i) + ".png");
    Bitmap bitmap(w, h, true);
    for (int y = 0; y < h; ++y) {
      for (int x = 0; x < w; ++x) {
        ASSERT_TRUE(bitmap.SetPixel(
            x, y, BitmapColor<uint8_t>(i * 70 + 17 + x + y * 3)));
      }
    }
    ASSERT_TRUE(bitmap.Write(path));
    images.emplace_back(path, w, h, K, R, T);
    inputs.push_back({images.back(), -1, CaptureFileIdentity(path)});
    ASSERT_TRUE(bitmap.Read(path, false));
    images.back().SetBitmap(std::move(bitmap));
  }
  // Retain the existing fixture's padded union and two opposite source orders.
  HostGrayscaleBank host({{inputs[2], inputs[0], inputs[1]},
                          {inputs[1], inputs[0]},
                          {inputs[0], inputs[1]}});
  int device = -1;
  CUDA_SAFE_CALL(cudaGetDevice(&device));
  for (const bool graphs : {false, true}) {
    SCOPED_TRACE(graphs ? "graph submission" : "stream submission");
    PatchMatchOptions options;
    options.gpu_index = std::to_string(device);
    options.geom_consistency = false;
    options.depth_min = 1;
    options.depth_max = 5;
    options.num_iterations = 2;
    options.num_samples = 2;
    options.filter = false;
    options.random_seed = 0;
    options.owned_cuda_graphs = graphs;
    std::array<std::unique_ptr<PatchMatch>, 2> controls;
    std::array<std::unique_ptr<PatchMatch>, 2> consumers;
    auto resident = std::make_shared<ResidentGrayscaleBank>(host, device);
    std::weak_ptr<const ResidentGrayscaleBank> lifetime = resident;
    for (size_t job = 0; job < consumers.size(); ++job) {
      // Each constructor snapshots its own reference/solver inputs. The serial
      // control consumes private bitmaps with identical options/order/seed.
      auto private_images = images;
      PatchMatch::Problem problem;
      problem.images = &private_images;
      problem.ref_image_idx = 2;
      problem.src_image_idxs =
          job == 0 ? std::vector<int>{1, 0} : std::vector<int>{0, 1};
      controls[job] = std::make_unique<PatchMatch>(
          options, problem, PatchMatchExecution::kOwnedCuda);
      controls[job]->Run();
      problem.photo_grayscale = resident->BindJob(job + 1);
      for (const int idx : problem.src_image_idxs) {
        const auto& image = images[idx];
        private_images[idx] = Image(image.GetPath(),
                                    image.GetWidth(),
                                    image.GetHeight(),
                                    image.GetK(),
                                    image.GetR(),
                                    image.GetT());
      }
      consumers[job] = std::make_unique<PatchMatch>(
          options, problem, PatchMatchExecution::kOwnedCuda);
      consumers[job]->PrepareOwned();
    }  // No caller Problem or copied binding survives this scope.
    resident.reset();
    ASSERT_FALSE(lifetime.expired());

    std::mutex mutex;
    std::condition_variable ready;
    size_t arrived = 0;
    // Explicit async futures join on destruction, including partial launch
    // failure. A missing/failed peer times out rather than stranding a worker.
    // No GPU event waits or spin barriers are used to force kernel overlap.
    std::array<std::future<void>, 2> workers;
    for (size_t job = 0; job < workers.size(); ++job) {
      workers[job] = std::async(std::launch::async, [&, job] {
        CUDA_SAFE_CALL(cudaSetDevice(device));
        {
          std::unique_lock<std::mutex> lock(mutex);
          ++arrived;
          ready.notify_all();
          if (!ready.wait_for(lock, std::chrono::seconds(10), [&] {
                return arrived == consumers.size();
              })) {
            throw std::runtime_error("PHOTO submission rendezvous timed out");
          }
        }
        consumers[job]->RunPreparedOwned();
      });
    }
    std::array<bool, 2> completed{};
    for (size_t job = 0; job < workers.size(); ++job) {
      // Join BOTH workers even if one throws; do not read or destroy a running
      // consumer on an assertion/exception path.
      EXPECT_NO_THROW({
        workers[job].get();
        completed[job] = true;
      });
    }
    for (size_t job = 0; job < consumers.size(); ++job) {
      if (!completed[job]) continue;
      const auto expected_depth = controls[job]->GetDepthMap();
      ASSERT_EQ(expected_depth.GetData().size(), 20);
      EXPECT_TRUE(std::any_of(expected_depth.GetData().begin(),
                              expected_depth.GetData().end(),
                              [](float depth) { return depth > 0; }));
      EXPECT_TRUE(SameFloatBits(expected_depth.GetData(),
                                consumers[job]->GetDepthMap().GetData()));
      EXPECT_TRUE(SameFloatBits(controls[job]->GetNormalMap().GetData(),
                                consumers[job]->GetNormalMap().GetData()));
      EXPECT_TRUE(SameFloatBits(controls[job]->GetSelProbMap().GetData(),
                                consumers[job]->GetSelProbMap().GetData()));
    }
    // RunPreparedOwned's completion contract fences each stream. Destroy the
    // first completed consumer while the second production binding still owns
    // the bank, then verify final release without any extra strong test owner.
    consumers[0].reset();
    EXPECT_FALSE(lifetime.expired());
    consumers[1].reset();
    EXPECT_TRUE(lifetime.expired());
  }
  std::filesystem::remove_all(root);
}

TEST(ResidentInputBankSampler, DepthElementTypeAndPadding) {
  OwnedCudaStream stream;
  const std::vector<float> depth = {1.25f, 7.0f, 2.5f, 19.0f, 4.0f, 8.0f};
  const std::vector<float> padded = {1.25f, 7.0f, 2.5f, 0, 0, 19.0f, 4.0f,
                                     8.0f,  0,    0,    0, 0, 0,     0,
                                     0,     0,    0,    0, 0, 0};
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
  Sample<<<1, 32, 0, stream.Get()>>>(
      direct->GetObj(), bank->GetObj(), mapping.GetPtr(), sampled.View());
  auto values = sampled.CopyToMat(stream.Get());
  for (size_t i = 0; i < 32; ++i) {
    EXPECT_EQ(values.GetData()[i], values.GetData()[32 + i]);
  }
  completion.Wait();
}
}  // namespace
}  // namespace colmap::mvs::owned
