// SPDX-License-Identifier: BSD-3-Clause
// Native two-device gates. A skipped test is NOT multi-GPU qualification.
#include "colmap/mvs/multi_gpu_fixed_cohort.h"
#include "colmap/mvs/owned_fixed_cohort_validation.h"
#include "colmap/mvs/patch_match_cuda_owned.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <iostream>
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

Fixture MakeFixture(size_t clusters = 1) {
  Fixture f;
  // Reference 2 sees only adjacent cameras. At Z=4, a 0.125 baseline
  // gives at most 1.791 degrees: no true plane point can pass the unchanged
  // 3-degree triangulation filter. A 0.25 baseline gives 3.576 degrees on
  // each principal ray. Keep the rendered texture tied to the same geometry:
  // f=64, Z=4 => disparity = f * baseline / Z = 4 pixels per camera.
  constexpr float kCameraBaseline = 0.25f;
  constexpr int kPlaneDisparity = 4;
  static_assert(64 * kCameraBaseline / 4 == kPlaneDisparity);
  const float K[9] = {64, 0, 32, 0, 64, 24, 0, 0, 1};
  const float R[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  for (size_t i = 0; i < 4 * clusters; ++i) {
    const int camera = i % 4;
    const int width = i % 2 ? 60 : 64, height = i % 2 ? 44 : 48;
    const float T[3] = {-kCameraBaseline * camera, 0, 0};
    f.images.emplace_back("plane-" + std::to_string(i), width, height, K, R, T);
    Bitmap bitmap(width, height, false);
    for (int y = 0; y < height; ++y) for (int x = 0; x < width; ++x) {
      uint32_t value = x + kPlaneDisparity * camera + y * 101;
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
  std::vector<int> order;
  for (size_t c = 0; c < clusters; ++c)
    for (int camera : {3, 0, 2, 1}) order.push_back(4 * c + camera);
  std::vector<FixedCohortJobShape> jobs;
  for (size_t i = 0; i < order.size(); ++i) {
    FixedCohortReference ref;
    ref.reference_image_idx = order[i];
    const size_t base = i / 4 * 4;
    ref.photo_sources = {order[base + (i + 1) % 4], order[base + (i + 2) % 4],
                         order[base + (i + 1) % 4]};
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

size_t PositiveDepthCount(const FixedCohortResult::Maps& maps) {
  return std::count_if(maps.depth.GetData().begin(), maps.depth.GetData().end(),
                       [](float depth) { return std::isfinite(depth) && depth > 0; });
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
    const auto& ref = f.refs[i];
    SCOPED_TRACE("job=" + std::to_string(i) + " reference=" +
                 std::to_string(ref.reference_image_idx));
    std::cout << "MultiGpuFixture job=" << i
              << " reference=" << ref.reference_image_idx
              << " image=" << f.images[ref.reference_image_idx].GetPath()
              << " single_photo_positive=" << PositiveDepthCount(single.photo[i])
              << " multi_photo_positive=" << PositiveDepthCount(result.photo[i])
              << " single_geo_positive=" << PositiveDepthCount(single.geo[i])
              << " multi_geo_positive=" << PositiveDepthCount(result.geo[i])
              << " geo_sources=";
    for (int source : ref.geo_sources) std::cout << source << ',';
    std::cout << std::endl;
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

// The 16 independent copies retain the corrected plane, ordered duplicates,
// filter=3 degrees, and original numerical options. More input identities, not
// larger/resized images or changed arithmetic, force bank turnover.
MultiGpuFixedCohortPlan TightCohortPlan(const Fixture& f) {
  const auto full = PlanMultiGpuFixedCohort(f.shapes, f.refs, f.budget,
                                           f.devices, 1ull << 30, 1);
  uint64_t private_device = 0;
  for (const auto& group : full.phases[1].groups)
    for (const auto& cohort : group.cohorts)
      private_device = std::max(private_device, cohort.private_peak_raw.device);
  auto devices = f.devices;
  for (auto& device : devices)
    device.capacity_bytes = CheckedAdd(
        MarginBytes(CheckedAdd(f.budget.process.device + f.budget.unknown.device,
                               private_device)), f.budget.headroom_reserve.device);
  return PlanMultiGpuFixedCohort(f.shapes, f.refs, f.budget, devices, 1ull << 30, 1);
}

TEST(MultiGpuFixedCohort, SequentialAdmittedBanksMatchSingleAndFullBankControls) {
  int count = 0;
  if (cudaGetDeviceCount(&count) != cudaSuccess || count < 2)
    GTEST_SKIP() << "requires two actual visible CUDA devices";
  cudaDeviceProp a{}, b{};
  ASSERT_EQ(cudaGetDeviceProperties(&a, 0), cudaSuccess);
  ASSERT_EQ(cudaGetDeviceProperties(&b, 1), cudaSuccess);
  if (a.major != 8 || (a.minor != 6 && a.minor != 9) ||
      a.major != b.major || a.minor != b.minor)
    GTEST_SKIP() << "requires identical qualified sm86/89 devices";
  auto f = MakeFixture(16);
  const auto full = PlanMultiGpuFixedCohort(f.shapes, f.refs, f.budget,
                                           f.devices, 1ull << 30, 1);
  const auto tight = TightCohortPlan(f);
  for (size_t d = 0; d < 2; ++d) {
    ASSERT_EQ(full.phases[0].groups[d].cohorts.size(), 1);
    ASSERT_GT(tight.phases[0].groups[d].cohorts.size(), 1);
    // Full resident working set cannot fit THIS GPU's cap, independently of
    // the other device or the ample invocation-wide host allowance.
    EXPECT_GT(CheckedAdd(MarginBytes(full.phases[0].groups[d].peak_raw.device),
                         tight.phases[0].groups[d].device.reserve_bytes),
              tight.phases[0].groups[d].device.capacity_bytes);
  }
  std::atomic<bool> bank_live[2]{};
  std::atomic<size_t> builds[2]{}, drains[2][2]{};
  std::atomic<uint64_t> uploaded[2]{};
  std::atomic<int> begins[2][64]{}, outputs[2][64]{};
  std::atomic<bool> photo_joined{false};
  const auto result = RunMultiGpuFixedCohortPhotoGeo(f.images, f.refs, f.budget, tight,
      NativeRunDeadline::FromBudget(std::chrono::seconds(300)), 32, nullptr,
      [&](bool geo, int device, size_t job, const char* event) {
        const std::string name(event);
        if (name == "joined") {
          for (size_t d = 0; d < 2; ++d) {
            EXPECT_FALSE(bank_live[d].load());
            EXPECT_EQ(drains[geo][d].load(), tight.phases[geo].groups[d].cohorts.size());
          }
          if (!geo) photo_joined = true;
          return;
        }
        if (name == "bank_ready") {
          EXPECT_FALSE(geo);
          EXPECT_FALSE(bank_live[device].exchange(true));
          ++builds[device];
          for (const auto& cohort : tight.phases[0].groups[device].cohorts)
            if (cohort.jobs.front() == job) uploaded[device] += cohort.bank_raw.device;
        } else if (name == "cohort_drained") {
          if (!geo) EXPECT_TRUE(bank_live[device].exchange(false));
          const auto& cohort = tight.phases[geo].groups[device].cohorts.at(drains[geo][device]++);
          EXPECT_EQ(job, cohort.jobs.front());
          for (size_t member : cohort.jobs) EXPECT_EQ(outputs[geo][member].load(), 1);
        } else if (name == "begin") {
          if (!geo) EXPECT_TRUE(bank_live[device].load());
          else EXPECT_TRUE(photo_joined.load());
          ++begins[geo][job];
        } else if (name == "output") ++outputs[geo][job];
      });
  for (size_t d = 0; d < 2; ++d) {
    uint64_t expected_upload = 0;
    for (const auto& cohort : tight.phases[0].groups[d].cohorts)
      expected_upload += cohort.bank_raw.device;
    EXPECT_EQ(builds[d].load(), tight.phases[0].groups[d].cohorts.size());
    EXPECT_EQ(uploaded[d].load(), expected_upload);
    std::cout << "CohortTurnover device=" << d << " actual_bank_builds=" << builds[d]
              << " actual_upload_payload_bytes=" << uploaded[d] << std::endl;
  }
  const auto control = RunMultiGpuFixedCohortPhotoGeo(f.images, f.refs, f.budget, full,
      NativeRunDeadline::FromBudget(std::chrono::seconds(300)), 32);
  const auto single = RunFixedCohortPhotoGeo(f.images, f.refs, f.budget,
      {1ull << 30, 256ull << 20}, NativeRunDeadline::FromBudget(std::chrono::seconds(300)),
      PairDispatch::kSerial);
  for (size_t job = 0; job < f.refs.size(); ++job) {
    SCOPED_TRACE(job);
    for (bool geo : {false, true}) {
      EXPECT_EQ(begins[geo][job].load(), 1);
      EXPECT_EQ(outputs[geo][job].load(), 1);
    }
    ExpectBits(single.photo[job], result.photo[job]);
    ExpectBits(single.geo[job], result.geo[job]);
    ExpectBits(control.photo[job], result.photo[job]);
    ExpectBits(control.geo[job], result.geo[job]);
    EXPECT_GT(PositiveDepthCount(result.geo[job]), 0);
  }
}

TEST(MultiGpuFixedCohort, CancellationAtCohortRetirementStopsNewBanksAndDrains) {
  int count = 0;
  if (cudaGetDeviceCount(&count) != cudaSuccess || count < 2)
    GTEST_SKIP() << "requires two actual visible CUDA devices";
  auto f = MakeFixture(16);
  const auto plan = TightCohortPlan(f);
  for (bool request_cancel : {false, true}) {
    std::atomic<bool> cancelled{false}, triggered{false};
    std::atomic<size_t> builds[2]{};
    EXPECT_THROW(RunMultiGpuFixedCohortPhotoGeo(f.images, f.refs, f.budget, plan,
        NativeRunDeadline::FromBudget(std::chrono::seconds(300)), 32, &cancelled,
        [&](bool geo, int device, size_t, const char* event) {
          EXPECT_FALSE(geo);  // No GEO may consume a partially produced exchange.
          if (std::string(event) == "bank_ready") ++builds[device];
          if (device == 0 && std::string(event) == "cohort_drained" && !triggered.exchange(true)) {
            if (request_cancel) cancelled = true;
            else throw std::runtime_error("injected cohort retirement failure");
          }
        }), std::runtime_error);
    EXPECT_TRUE(triggered.load());
    EXPECT_EQ(builds[0].load(), 1);  // No second bank on the cancelling device.
  }
  // A fresh invocation succeeds after failed workers and banks have drained;
  // no process-global cancellation latch or retained owner survives the call.
  EXPECT_NO_THROW(RunMultiGpuFixedCohortPhotoGeo(f.images, f.refs, f.budget, plan,
      NativeRunDeadline::FromBudget(std::chrono::seconds(300)), 32));
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
