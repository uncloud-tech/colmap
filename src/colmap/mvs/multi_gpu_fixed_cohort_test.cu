// SPDX-License-Identifier: BSD-3-Clause
// Native two-device gates. A skipped test is NOT multi-GPU qualification.
#include "colmap/mvs/multi_gpu_fixed_cohort.h"
#include "colmap/mvs/owned_fixed_cohort_validation.h"
#include "colmap/mvs/patch_match_cuda_owned.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <stdexcept>

#include <gtest/gtest.h>

namespace colmap::mvs::owned {
namespace {
struct Fixture {
  std::vector<Image> images;
  std::vector<FixedCohortReference> refs;
  FixedCohortBudget budget;
  std::vector<FixedCohortImageShape> shapes;
  std::vector<PhasePlanDevice> devices;
};

Fixture MakeFixture() {
  Fixture f;
  const float K[9] = {64, 0, 32, 0, 64, 24, 0, 0, 1};
  const float R[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  for (int i = 0; i < 4; ++i) {
    const int width = i % 2 ? 60 : 64, height = i % 2 ? 44 : 48;
    const float T[3] = {-0.125f * i, 0, 0};
    f.images.emplace_back("plane-" + std::to_string(i), width, height, K, R, T);
    Bitmap bitmap(width, height, false);
    for (int y = 0; y < height; ++y) for (int x = 0; x < width; ++x) {
      uint32_t value = x + 2 * i + y * 101;
      value = (value ^ 61u) ^ (value >> 16);
      value *= 9u;
      value ^= value >> 4;
      value *= 0x27d4eb2du;
      value ^= value >> 15;
      if (!bitmap.SetPixel(x, y, BitmapColor<uint8_t>(value & 255)))
        throw std::runtime_error("fixture pixel failed");
    }
    f.images.back().SetBitmap(std::move(bitmap));
    f.shapes.push_back({static_cast<uint64_t>(width), static_cast<uint64_t>(height)});
  }
  const std::vector<int> order{3, 0, 2, 1};
  std::vector<FixedCohortJobShape> jobs;
  for (size_t i = 0; i < order.size(); ++i) {
    FixedCohortReference ref;
    ref.reference_image_idx = order[i];
    ref.photo_sources = {order[(i + 1) % 4], order[(i + 2) % 4], order[(i + 1) % 4]};
    ref.geo_sources = ref.photo_sources;
    auto& options = ref.photo_options;
    options.gpu_index = "0";
    options.geom_consistency = false;
    options.filter = false;
    options.depth_min = 1;
    options.depth_max = 5;
    options.sigma_spatial = 5;
    options.num_iterations = 2;
    options.num_samples = 2;
    options.random_seed = 0;
    options.num_threads = 1;
    ref.geo_options = options;
    ref.geo_options.geom_consistency = true;
    ref.geo_options.filter = true;
    ref.geo_options.filter_min_num_consistent = 1;
    jobs.push_back({ref.reference_image_idx, ref.photo_sources, ref.geo_sources});
    f.refs.push_back(ref);
  }
  FixedCohortBudget declarations;
  declarations.max_active_owners = 1;
  declarations.process = {16ull << 20, 16ull << 20};
  declarations.unknown = {16ull << 20, 16ull << 20};
  declarations.unknown_provenance = "tiny native fixture allowance, not calibrated physical-fit proof";
  declarations.headroom_reserve = {1ull << 20, 1ull << 20};
  f.budget = BuildFixedCohortLogicalInventory(f.shapes, jobs, declarations);
  for (int d = 0; d < 2; ++d) {
    const auto observed = ObserveFixedCohortCapacities(d);
    PhasePlanDevice device;
    device.ordinal = d;
    device.uuid = observed.device_uuid;
    device.capacity_bytes = std::min<uint64_t>(observed.available.device, 256ull << 20);
    f.devices.push_back(device);
  }
  return f;
}

void ExpectBits(const FixedCohortResult::Maps& a, const FixedCohortResult::Maps& b) {
  for (const auto& pair : {std::make_pair(&a.depth.GetData(), &b.depth.GetData()),
                           std::make_pair(&a.normal.GetData(), &b.normal.GetData())}) {
    ASSERT_FALSE(pair.first->empty());
    ASSERT_EQ(pair.first->size(), pair.second->size());
    EXPECT_EQ(std::memcmp(pair.first->data(), pair.second->data(), pair.first->size() * sizeof(float)), 0);
  }
}

TEST(MultiGpuFixedCohort, TwoDevicePhotoGeoExactUniqueWritersAndBarrier) {
  int count = 0;
  if (cudaGetDeviceCount(&count) != cudaSuccess || count < 2)
    GTEST_SKIP() << "requires two actual visible CUDA devices";
  cudaDeviceProp a{}, b{};
  ASSERT_EQ(cudaGetDeviceProperties(&a, 0), cudaSuccess);
  ASSERT_EQ(cudaGetDeviceProperties(&b, 1), cudaSuccess);
  if (a.major != 8 || (a.minor != 6 && a.minor != 9) || a.major != b.major || a.minor != b.minor)
    GTEST_SKIP() << "primary exactness gate requires identical qualified sm86/89 devices; heterogeneous/Blackwell not inferred";
  auto f = MakeFixture();
  const auto plan = PlanMultiGpuFixedCohort(f.shapes, f.refs, f.budget, f.devices, 1ull << 30, 1);
  std::atomic<int> begins[2][4]{};
  std::atomic<int> outputs[2][4]{};
  std::atomic<bool> useful[2][2]{};
  std::atomic<bool> photo_joined{false};
  const auto result = RunMultiGpuFixedCohortPhotoGeo(f.images, f.refs, f.budget, plan,
      NativeRunDeadline::FromBudget(std::chrono::seconds(120)), 32, nullptr,
      [&](bool geo, int device, size_t job, const char* event) {
        if (std::string(event) == "joined") {
          if (!geo) photo_joined = true;
          return;
        }
        if (std::string(event) == "begin") {
          ++begins[geo][job];
          useful[geo][device] = true;
          if (geo) {
            EXPECT_TRUE(photo_joined.load());
            for (const auto& output : outputs[0]) EXPECT_EQ(output.load(), 1);
          }
        } else if (std::string(event) == "output") ++outputs[geo][job];
      });
  for (const bool geo : {false, true}) {
    for (size_t i = 0; i < 4; ++i) {
      EXPECT_EQ(begins[geo][i].load(), 1);
      EXPECT_EQ(outputs[geo][i].load(), 1);
    }
    EXPECT_TRUE(useful[geo][0].load());
    EXPECT_TRUE(useful[geo][1].load());
  }
  const auto single = RunFixedCohortPhotoGeo(f.images, f.refs, f.budget,
      {1ull << 30, 256ull << 20}, NativeRunDeadline::FromBudget(std::chrono::seconds(120)), PairDispatch::kSerial);
  for (size_t i = 0; i < f.refs.size(); ++i) {
    ExpectBits(single.photo[i], result.photo[i]);
    ExpectBits(single.geo[i], result.geo[i]);
    size_t valid = 0;
    for (size_t y = 0; y < result.geo[i].depth.GetHeight(); ++y) {
      for (size_t x = 0; x < result.geo[i].depth.GetWidth(); ++x) {
        const float depth = result.geo[i].depth.Get(y, x);
        if (!(depth > 0)) continue;
        EXPECT_TRUE(std::isfinite(depth));
        float normal[3];
        result.geo[i].normal.GetSlice(y, x, normal);
        float norm = 0;
        for (float n : normal) { EXPECT_TRUE(std::isfinite(n)); norm += n * n; }
        EXPECT_GT(norm, 0);
        ++valid;
      }
    }
    EXPECT_GT(valid, 0) << "all-filtered maps are not useful reconstruction proof";
  }
}

TEST(MultiGpuFixedCohort, ObserverFailureAndCancellationDrainAllDevicesWithoutResult) {
  int count = 0;
  if (cudaGetDeviceCount(&count) != cudaSuccess || count < 2)
    GTEST_SKIP() << "requires two actual visible CUDA devices";
  auto f = MakeFixture();
  const auto plan = PlanMultiGpuFixedCohort(f.shapes, f.refs, f.budget, f.devices, 1ull << 30, 1);
  for (const bool failed_geo : {false, true}) {
    for (const bool cancel_request : {false, true}) {
      std::atomic<bool> cancelled{false};
      EXPECT_THROW(RunMultiGpuFixedCohortPhotoGeo(f.images, f.refs, f.budget, plan,
          NativeRunDeadline::FromBudget(std::chrono::seconds(120)), 32, &cancelled,
          [&](bool geo, int, size_t, const char* event) {
            if (geo == failed_geo && std::string(event) == "output") {
              if (cancel_request) cancelled = true;
              else throw std::runtime_error("injected device failure");
            }
          }), std::runtime_error);
      // Fail with an actual solver alive after useful GPU work, in each phase.
      // No partial FixedCohortResult is returned by any failed attempt.
    }
  }
  EXPECT_NO_THROW(RunMultiGpuFixedCohortPhotoGeo(f.images, f.refs, f.budget, plan,
      NativeRunDeadline::FromBudget(std::chrono::seconds(120)), 32));
}
}  // namespace
}  // namespace colmap::mvs::owned
