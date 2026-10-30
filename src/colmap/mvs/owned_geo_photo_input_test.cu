// SPDX-License-Identifier: BSD-3-Clause
#include "colmap/mvs/owned_geo_photo_input.h"
#include "colmap/mvs/patch_match_cuda_owned.h"

#include <array>
#include <cstring>
#include <stdexcept>

#include <gtest/gtest.h>

namespace colmap::mvs::owned {
namespace {
void ExpectBits(const std::vector<float>& expected,
                const std::vector<float>& actual) {
  ASSERT_EQ(expected.size(), actual.size());
  EXPECT_EQ(
      std::memcmp(
          expected.data(), actual.data(), expected.size() * sizeof(float)),
      0);
}

__global__ void ReadSourceDepth(cudaTextureObject_t texture,
                                GpuMatView<float> output) {
  const int col = threadIdx.x;
  const int row = threadIdx.y;
  const int layer = blockIdx.z;
  if (col < output.GetWidth() && row < output.GetHeight()) {
    output.Set(row,
               col,
               layer,
               tex2DLayered<float>(texture, col - 0.5f, row - 0.5f, layer));
  }
}

TEST(GeoPhotoInput, FreshCrossOwnerMapsMatchPrivateGeoAndDrainOnCancellation) {
  const float K[9] = {10, 0, 2, 0, 10, 2, 0, 0, 1};
  const float R[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  const float T[3] = {0, 0, 0};
  std::vector<Image> images;
  for (int i = 0; i < 3; ++i) {
    const int width = i == 1 ? 3 : 5;
    const int height = i == 1 ? 3 : 4;
    images.emplace_back("photo-" + std::to_string(i), width, height, K, R, T);
    Bitmap bitmap(width, height, false);
    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        ASSERT_TRUE(bitmap.SetPixel(
            x, y, BitmapColor<uint8_t>(17 + i * 40 + x + y * 3)));
      }
    }
    images.back().SetBitmap(std::move(bitmap));
  }
  int device = -1;
  CUDA_SAFE_CALL(cudaGetDevice(&device));
  for (const bool graphs : {false, true}) {
    SCOPED_TRACE(graphs ? "graph" : "stream");
    PatchMatchOptions options;
    options.gpu_index = std::to_string(device);
    options.geom_consistency = false;
    options.depth_min = 1;
    options.depth_max = 5;
    options.sigma_spatial = 5;
    options.num_iterations = 2;
    options.num_samples = 2;
    options.filter = false;
    options.random_seed = 0;
    options.owned_cuda_graphs = graphs;
    std::vector<DepthMap> depths(3);
    std::vector<NormalMap> normals(3);
    std::array<std::shared_ptr<const PhotoOutputSnapshot>, 3> snapshots;
    for (int ref = 0; ref < 3; ++ref) {
      PatchMatch::Problem photo;
      photo.images = &images;
      photo.ref_image_idx = ref;
      photo.src_image_idxs = {(ref + 2) % 3, (ref + 1) % 3};
      auto producer = std::make_unique<PatchMatchCuda>(options, photo);
      producer->Run();
      depths[ref] = producer->GetDepthMap();
      normals[ref] = producer->GetNormalMap();
      snapshots[ref] = producer->SnapshotPhotoOutput();
      EXPECT_TRUE(snapshots[ref]->Camera().Matches(images[ref]));
      // A successful later version must not mutate the published old copy.
      producer->Run();
      OwnedCudaStream reader;
      GpuMat<float> depth(images[ref].GetWidth(), images[ref].GetHeight());
      GpuMat<float> normal(images[ref].GetWidth(), images[ref].GetHeight(), 3);
      snapshots[ref]->WithDeviceMaps(reader, [&](auto d, auto n) {
        for (const auto& copy :
             {std::make_pair(d, &depth), std::make_pair(n, &normal)}) {
          CUDA_SAFE_CALL(cudaMemcpy2DAsync(copy.second->GetPtr(),
                                           copy.second->GetPitch(),
                                           copy.first.ptr,
                                           copy.first.pitch,
                                           copy.first.width * sizeof(float),
                                           copy.first.height * copy.first.depth,
                                           cudaMemcpyDeviceToDevice,
                                           reader.Get()));
        }
      });
      ExpectBits(depths[ref].GetData(),
                 depth.CopyToMat(reader.Get()).GetData());
      ExpectBits(normals[ref].GetData(),
                 normal.CopyToMat(reader.Get()).GetData());
    }  // All PHOTO owners released. This is the global phase barrier.

    options.geom_consistency = true;
    options.filter = true;
    options.filter_min_num_consistent = 1;
    // Exercise source permutation, mixed dimensions and repeated ordinals.
    PatchMatch::Problem geo;
    geo.images = &images;
    geo.ref_image_idx = 2;
    geo.src_image_idxs = {1, 0, 1};
    geo.depth_maps = &depths;
    geo.normal_maps = &normals;
    PatchMatchCuda control(options, geo);
    control.Run();
    geo.depth_maps = nullptr;
    geo.normal_maps = nullptr;
    std::vector<std::shared_ptr<const PhotoOutputSnapshot>> sources = {
        snapshots[1], snapshots[0], snapshots[1]};
    auto input =
        std::make_shared<GeoPhotoInput>(options, geo, snapshots[2], sources);
    EXPECT_EQ(input->SourceTextureLogicalBytes(), 5 * 4 * 3 * sizeof(float));
    EXPECT_GE(input->SourceStagingBytes(), input->SourceTextureLogicalBytes());
    EXPECT_EQ(input->ReferenceCopyLogicalBytes(), 5 * 4 * 4 * sizeof(float));
    EXPECT_EQ(input->SnapshotRetainedBytes(),
              snapshots[0]->RetainedBytes() + snapshots[1]->RetainedBytes() +
                  snapshots[2]->RetainedBytes());

    // Actual source texture, including all padded texels and external borders,
    // against the private fresh PHOTO host control (no synthetic map ledger).
    {
      OwnedCudaStream reader;
      GpuMat<float> sampled(7, 6, 3);
      ScopedStreamCompletion completion(reader.Get());
      ReadSourceDepth<<<dim3(1, 1, 3), dim3(8, 8), 0, reader.Get()>>>(
          input->SourceDepthTexture(), sampled.View());
      const auto values = sampled.CopyToMat(reader.Get());
      std::vector<float> expected(7 * 6 * 3, 0.0f);
      for (size_t layer = 0; layer < 3; ++layer) {
        const auto& depth = depths[geo.src_image_idxs[layer]];
        for (size_t row = 0; row < depth.GetHeight(); ++row) {
          for (size_t col = 0; col < depth.GetWidth(); ++col) {
            expected[layer * 7 * 6 + (row + 1) * 7 + col + 1] =
                depth.Get(row, col);
          }
        }
      }
      ExpectBits(expected, values.GetData());
      completion.Wait();
    }

    // Null/missing/wrong order/dimension/camera bindings must not be admitted.
    EXPECT_THROW(GeoPhotoInput(options, geo, nullptr, sources),
                 std::invalid_argument);
    EXPECT_THROW(GeoPhotoInput(options, geo, snapshots[0], sources),
                 std::invalid_argument);
    auto wrong = sources;
    wrong[0].reset();
    EXPECT_THROW(GeoPhotoInput(options, geo, snapshots[2], wrong),
                 std::invalid_argument);
    wrong = sources;
    wrong.pop_back();
    EXPECT_THROW(GeoPhotoInput(options, geo, snapshots[2], wrong),
                 std::invalid_argument);
    wrong = sources;
    std::swap(wrong[0], wrong[1]);
    EXPECT_THROW(GeoPhotoInput(options, geo, snapshots[2], wrong),
                 std::invalid_argument);
    auto changed_images = images;
    auto changed_geo = geo;
    changed_geo.images = &changed_images;
    float changed_K[9];
    std::memcpy(changed_K, K, sizeof(K));
    changed_K[0] += 1;
    changed_images[0] = Image("photo-0", 5, 4, changed_K, R, T);
    EXPECT_THROW(input->Validate(options, changed_geo), std::invalid_argument);
    changed_images = images;
    const float changed_T[3] = {1, 0, 0};
    changed_images[2] = Image("photo-2", 5, 4, K, R, changed_T);
    EXPECT_THROW(input->Validate(options, changed_geo), std::invalid_argument);
    changed_images = images;
    changed_images[1] = Image("photo-1", 4, 3, K, R, T);
    EXPECT_THROW(input->Validate(options, changed_geo), std::invalid_argument);
    auto mixed = geo;
    mixed.depth_maps = &depths;
    EXPECT_THROW(GeoPhotoInput(options, mixed, snapshots[2], sources),
                 std::invalid_argument);
    auto wrong_options = options;
    wrong_options.geom_consistency = false;
    EXPECT_THROW((PatchMatchCuda(wrong_options, geo, 32, input)),
                 std::invalid_argument);
    wrong_options = options;
    wrong_options.gpu_index = std::to_string(device + 1);
    EXPECT_THROW((PatchMatchCuda(wrong_options, geo, 32, input)),
                 std::invalid_argument);

    // Bind the intended GEO invocation, not the earlier PHOTO settings (which
    // deliberately differed in geom_consistency/filter). Refuse changes through
    // the real consumer constructor, before it allocates any GPU resources.
    for (const auto field : {&PatchMatchOptions::depth_min,
                             &PatchMatchOptions::depth_max,
                             &PatchMatchOptions::sigma_spatial,
                             &PatchMatchOptions::geom_consistency_regularizer,
                             &PatchMatchOptions::filter_min_ncc}) {
      auto changed = options;
      changed.*field += 0.25;
      EXPECT_THROW((PatchMatchCuda(changed, geo, 32, input)),
                   std::invalid_argument);
    }
    for (const auto field : {&PatchMatchOptions::random_seed,
                             &PatchMatchOptions::num_iterations,
                             &PatchMatchOptions::filter_min_num_consistent}) {
      auto changed = options;
      ++(changed.*field);
      EXPECT_THROW((PatchMatchCuda(changed, geo, 32, input)),
                   std::invalid_argument);
    }
    wrong_options = options;
    wrong_options.filter = false;
    EXPECT_THROW((PatchMatchCuda(wrong_options, geo, 32, input)),
                 std::invalid_argument);
    wrong_options = options;
    wrong_options.owned_cuda_graphs = !options.owned_cuda_graphs;
    EXPECT_THROW((PatchMatchCuda(wrong_options, geo, 32, input)),
                 std::invalid_argument);

    auto consumer = std::make_unique<PatchMatchCuda>(options, geo, 32, input);
    // Check initialization before filtering can hide a failed handoff behind
    // an all-zero final result. Orientation and all three normal planes matter.
    ExpectBits(depths[2].GetData(), consumer->GetDepthMap().GetData());
    ExpectBits(normals[2].GetData(), consumer->GetNormalMap().GetData());
    auto cancelled = std::make_unique<PatchMatchCuda>(options, geo, 32, input);
    std::weak_ptr<const GeoPhotoInput> lifetime = input;
    std::weak_ptr<const PhotoOutputSnapshot> source_lifetime = snapshots[1];
    snapshots = {};
    sources.clear();
    wrong.clear();
    input.reset();
    consumer->Run();
    ExpectBits(control.GetDepthMap().GetData(),
               consumer->GetDepthMap().GetData());
    ExpectBits(control.GetNormalMap().GetData(),
               consumer->GetNormalMap().GetData());
    ExpectBits(control.GetSelProbMap().GetData(),
               consumer->GetSelProbMap().GetData());
    EXPECT_EQ(control.GetConsistentImageIdxs(),
              consumer->GetConsistentImageIdxs());
    // Rerun both actual consumers: mutable solver buffers must not leak back
    // into the retained PHOTO snapshot or change the adapter's next Run().
    control.Run();
    consumer->Run();
    ExpectBits(control.GetDepthMap().GetData(),
               consumer->GetDepthMap().GetData());
    ExpectBits(control.GetNormalMap().GetData(),
               consumer->GetNormalMap().GetData());
    ExpectBits(control.GetSelProbMap().GetData(),
               consumer->GetSelProbMap().GetData());
    EXPECT_EQ(control.GetConsistentImageIdxs(),
              consumer->GetConsistentImageIdxs());
    consumer.reset();
    EXPECT_FALSE(lifetime.expired());
    EXPECT_FALSE(source_lifetime.expired());
    if (graphs) {
      auto coordinator = std::make_shared<GraphLaunchControl>(1);
      cancelled->SetGraphCoordinator(coordinator, 0);
      coordinator->Cancel(
          std::make_exception_ptr(std::runtime_error("cancelled")));
    } else {
      cancelled->SetSweepObserver(
          [](int, int) { throw std::runtime_error("cancelled"); });
    }
    EXPECT_THROW(cancelled->Run(), std::exception);
    cancelled.reset();  // destructor drains before dropping final binding
    EXPECT_TRUE(lifetime.expired());
    EXPECT_TRUE(source_lifetime.expired());
  }
}
}  // namespace
}  // namespace colmap::mvs::owned
