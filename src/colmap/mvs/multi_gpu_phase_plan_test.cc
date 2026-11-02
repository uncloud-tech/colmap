// SPDX-License-Identifier: BSD-3-Clause
// CPU-only: g++ -std=c++17 -Wall -Wextra -Werror -Isrc
// src/colmap/mvs/multi_gpu_phase_plan{,_test}.cc -o ./phase-plan-test
// && ./phase-plan-test (do not define NDEBUG).
#include "colmap/mvs/multi_gpu_phase_plan.h"

#include <cassert>
#include <iostream>
#include <limits>
#include <stdexcept>

using namespace colmap::mvs::owned;

namespace {
PhasePlanRequest Fixture() {
  PhasePlanRequest r;
  r.images = {{"a", 10, 10}, {"b", 20, 5}, {"c", 5, 20}};
  r.devices = {{0, "gpu-a", 10000, 0, 10000, {10, 20}},
               {2, "gpu-b", 10000, 0, 10000, {10, 20}}};
  r.host_capacity_bytes = 30000;
  r.common_host_raw = 100;
  r.active_owners_ceiling = 2;
  r.jobs = {{91, 2, {0, 0}, {30, 40}, 100},
            {17, 0, {1}, {30, 40}, 100},
            {62, 1, {1, 1}, {30, 40}, 100},
            {4, 2, {0}, {30, 40}, 100}};
  return r;
}
void AssignmentAndAffinity() {
  const auto r = Fixture();
  const auto p = BuildMultiGpuPhasePlan(r);
  assert(p.groups.size() == 2 && !p.geometry);
  assert(p.groups[0].device.ordinal == 0 && p.groups[1].device.ordinal == 2);
  // Balanced round robin would replicate both sources on both GPUs (800
  // padded bytes). Affinity retains just 100 bytes on each GPU (200 total).
  assert((p.groups[0].cohorts[0].jobs == std::vector<size_t>{91, 4}));
  assert((p.groups[1].cohorts[0].jobs == std::vector<size_t>{17, 62}));
  for (const auto& g : p.groups) {
    assert(g.assigned_work == 200 && g.cohorts.size() == 1);
    assert(g.cohorts[0].active_owners == 2);
    assert(g.cohorts[0].bank_raw.host == 100);
    assert(g.cohorts[0].bank_raw.device == 100);
    assert(g.peak_raw.host == 170 && g.peak_raw.device == 200);
  }
  assert(p.host_peak_raw == 440);
  assert((r.jobs[0].sources == std::vector<size_t>{0, 0}));
  const auto again = BuildMultiGpuPhasePlan(r);
  assert(again.groups[0].cohorts[0].jobs == p.groups[0].cohorts[0].jobs);
  assert(again.groups[1].cohorts[0].jobs == p.groups[1].cohorts[0].jobs);
}
void AffinityDoesNotStarveUsefulDevices() {
  auto r = Fixture();
  r.jobs.resize(2);
  for (size_t i = 2; i < 12; ++i)
    r.jobs.push_back({100 + i, 2, {0, 0}, {30, 40}, 100});
  const auto p = BuildMultiGpuPhasePlan(r);
  assert(p.groups[0].assigned_work == 600);
  assert(p.groups[1].assigned_work == 600);
  // Zero work proxies still seed owners using actual ownership, not work==0.
  for (auto& job : r.jobs) job.work = 0;
  const auto zero = BuildMultiGpuPhasePlan(r);
  assert(zero.groups[0].cohorts[0].jobs.front() == 91);
  assert(zero.groups[1].cohorts[0].jobs.front() == 17);
}
void IndependentGeometryAndIdleDevices() {
  auto r = Fixture();
  r.geometry = true;
  r.devices.push_back({7, "gpu-idle", 10000, 0, 10000, {10, 20}});
  r.jobs.resize(2);
  const auto p = BuildMultiGpuPhasePlan(r);
  assert(p.geometry && p.groups.size() == 3);
  assert((p.groups[0].cohorts[0].jobs == std::vector<size_t>{91}));
  assert((p.groups[1].cohorts[0].jobs == std::vector<size_t>{17}));
  assert(p.groups[2].cohorts.empty() && p.groups[2].assigned_work == 0);
  assert(p.groups[2].peak_raw.host == 10);
  for (size_t g = 0; g < 2; ++g) {
    assert(p.groups[g].cohorts[0].active_owners == 1);
    assert(p.groups[g].cohorts[0].bank_raw.host == 0);
    assert(p.groups[g].cohorts[0].bank_raw.device == 0);
  }
  r.jobs = Fixture().jobs;
  r.devices.resize(2);
  const auto all = BuildMultiGpuPhasePlan(r);
  assert((all.groups[0].cohorts[0].jobs == std::vector<size_t>{91, 62}));
  assert((all.groups[1].cohorts[0].jobs == std::vector<size_t>{17, 4}));
}
void CohortDrainingAndOwnerAdmission() {
  auto r = Fixture();
  r.devices.resize(1);
  r.devices[0].capacity_bytes = 165;  // ceil(1.03 * 160).
  r.jobs = {{9, 2, {0}, {10, 40}, 1}, {3, 2, {1}, {10, 40}, 1}};
  const auto p = BuildMultiGpuPhasePlan(r);
  const auto& g = p.groups[0];
  assert(g.cohorts.size() == 2);
  assert((g.cohorts[0].jobs == std::vector<size_t>{9}));
  assert((g.cohorts[1].jobs == std::vector<size_t>{3}));
  assert(g.peak_raw.host == 120 && g.peak_raw.device == 160);
  // Same bank admits both jobs, but not both owners concurrently.
  r.jobs[1].sources = {0, 0};
  const auto limited = BuildMultiGpuPhasePlan(r);
  assert(limited.groups[0].cohorts.size() == 1);
  assert(limited.groups[0].cohorts[0].jobs.size() == 2);
  assert(limited.groups[0].cohorts[0].active_owners == 1);
}
void ExtendingBankCanReduceCompleteOwnerPeak() {
  auto r = Fixture();
  r.devices.resize(1);
  r.devices[0].capacity_bytes = 536;
  r.images[1] = {"b", 11, 10};
  r.jobs = {{1, 2, {0}, {1000, 200}, 1},
            {2, 2, {0}, {1000, 200}, 1},
            {3, 2, {1}, {1000, 200}, 1}};
  const auto p = BuildMultiGpuPhasePlan(r);
  assert(p.groups[0].cohorts.size() == 1);
  assert(p.groups[0].cohorts[0].active_owners == 1);
  assert(p.groups[0].peak_raw.host == 1230);
  assert(p.groups[0].peak_raw.device == 440);
  assert(p.host_peak_raw == 1330);
}
template <typename Exception = std::invalid_argument>
void Refuses(const PhasePlanRequest& r) {
  bool refused = false;
  try {
    BuildMultiGpuPhasePlan(r);
  } catch (const Exception&) {
    refused = true;
  }
  if (!refused)
    std::cerr << "Expected refusal: host capacity=" << r.host_capacity_bytes
              << ", jobs=" << r.jobs.size() << '\n';
  assert(refused);
}
void PaddingAndResourcewiseOwnerPeaks() {
  auto r = Fixture();
  r.devices.resize(1);
  r.jobs = {{2, 0, {1, 2, 1}, {80, 10}, 1},
            {8, 0, {2, 1}, {10, 90}, 1},
            {1, 0, {1}, {50, 30}, 1}};
  const auto p = BuildMultiGpuPhasePlan(r);
  const auto& c = p.groups[0].cohorts[0];
  assert((c.jobs == std::vector<size_t>{2, 8, 1}));
  // The reference is excluded, duplicates count once in the PHOTO bank.
  // The 20x5 and 5x20 sources need two 20x20 padded layers.
  assert(c.bank_raw.host == 800 && c.bank_raw.device == 800);
  // The largest host and device charges belong to different owners.
  assert(c.private_peak_raw.host == 130 && c.private_peak_raw.device == 120);
  assert(c.active_owners == 2);
  assert(p.groups[0].peak_raw.host == 940);
  assert(p.groups[0].peak_raw.device == 940);
  assert((r.jobs[0].sources == std::vector<size_t>{1, 2, 1}));
}
void MarginAndSeparateReserves() {
  auto r = Fixture();
  r.geometry = true;
  r.devices.resize(1);
  r.devices[0].fixed_raw = {};
  r.devices[0].capacity_bytes = 110;
  r.devices[0].reserve_bytes = 7;
  r.devices[0].host_share_raw = 100;
  r.common_host_raw = 0;
  r.jobs = {{42, 0, {}, {100, 100}, 1}};
  r.host_capacity_bytes = 114;
  r.host_reserve_bytes = 11;
  const auto exact = BuildMultiGpuPhasePlan(r);
  assert(exact.host_peak_raw == 100);
  assert(exact.groups[0].peak_raw.device == 100);
  // One 3% margin, not per-owner + per-device + invocation margins.
  --r.host_capacity_bytes;
  Refuses(r);
  ++r.host_capacity_bytes;
  --r.devices[0].capacity_bytes;
  Refuses(r);
  ++r.devices[0].capacity_bytes;
  r.jobs[0].private_raw.device = 101;
  Refuses(r);  // ceil(104.03) + 7 = 112, beyond available cap.
  r.jobs[0].private_raw.device = 100;
  r.jobs[0].private_raw.host = 101;
  Refuses(r);  // RAW per-device host share, never another margin.
  r.jobs[0].private_raw = {0, 1};
  r.devices[0].capacity_bytes = 9;  // ceil(1.03) + 7 = 9.
  assert(BuildMultiGpuPhasePlan(r).groups[0].peak_raw.device == 1);
  r.devices[0].capacity_bytes = 8;
  Refuses(r);
}
void UnequalCapsAndAggregateHost() {
  auto r = Fixture();
  r.geometry = true;
  r.devices[0].capacity_bytes = 50;
  r.jobs = {{9, 0, {0}, {10, 100}, 1}, {3, 0, {}, {10, 5}, 1}};
  const auto p = BuildMultiGpuPhasePlan(r);
  assert((p.groups[1].cohorts[0].jobs == std::vector<size_t>{9}));
  assert((p.groups[0].cohorts[0].jobs == std::vector<size_t>{3}));
  // Combined VRAM is enough, but no individual GPU can own the singleton.
  r.devices[1].capacity_bytes = 110;
  Refuses(r);
  r = Fixture();
  r.geometry = true;
  r.devices[0].fixed_raw = r.devices[1].fixed_raw = {};
  r.jobs.resize(2);
  // Only GPU 1 can own the device-heavy job; only GPU 0 has the host
  // share for the host-heavy job. They cannot be serialized on one GPU.
  r.jobs[0].private_raw = {50, 100};
  r.jobs[1].private_raw = {100, 5};
  r.devices[0].capacity_bytes = 50;
  r.devices[0].host_share_raw = 100;
  r.devices[1].host_share_raw = 50;
  r.common_host_raw = 100;
  r.host_capacity_bytes = 258;
  assert(BuildMultiGpuPhasePlan(r).host_peak_raw == 250);
  r.host_capacity_bytes = 257;
  Refuses(r);  // Both static shares fit separately, total host does not.
  r.host_capacity_bytes = 10000;
  r.devices[0].host_share_raw = 99;
  Refuses(r);
}
void InvalidDeclarationsAndOverflow() {
  const auto check = [](auto mutate) {
    auto r = Fixture();
    mutate(r);
    Refuses(r);
  };
  check([](auto& r) { r.devices.clear(); });
  check([](auto& r) { r.active_owners_ceiling = 0; });
  check([](auto& r) { r.images[1].identity = r.images[0].identity; });
  check([](auto& r) { r.images[0].identity.clear(); });
  check([](auto& r) { r.images[0].width = 0; });
  check([](auto& r) { r.images[0].height = 0; });
  check([](auto& r) { r.jobs[1].id = r.jobs[0].id; });
  check([](auto& r) { r.jobs[0].reference = r.images.size(); });
  check([](auto& r) { r.jobs[0].sources.push_back(r.images.size()); });
  check([](auto& r) { r.devices[0].ordinal = -1; });
  check([](auto& r) { r.devices[1].ordinal = r.devices[0].ordinal; });
  check([](auto& r) { r.devices[1].uuid = r.devices[0].uuid; });
  check([](auto& r) { r.devices[0].uuid.clear(); });
  check([](auto& r) { r.devices[0].reserve_bytes = 10001; });
  check([](auto& r) { r.host_reserve_bytes = 30001; });
  check([](auto& r) { r.devices[0].fixed_raw.host = 10001; });
  check([](auto& r) { r.devices[0].fixed_raw.device = 10001; });
  const auto overflow = [](auto mutate) {
    auto r = Fixture();
    mutate(r);
    Refuses<std::overflow_error>(r);
  };
  constexpr auto maximum = std::numeric_limits<uint64_t>::max();
  overflow([=](auto& r) { r.images[0].width = maximum; });
  overflow([=](auto& r) { r.jobs[0].work = maximum; });
  overflow([=](auto& r) { r.common_host_raw = maximum; });
  overflow([=](auto& r) { r.devices[0].fixed_raw.device = maximum; });
  overflow([=](auto& r) { r.jobs[0].private_raw.device = maximum; });
  // Individually representable images whose padded union product overflows.
  overflow([=](auto& r) {
    r.images[0] = {"a", maximum / 2, 1};
    r.images[1] = {"b", 1, 3};
    r.jobs[0].sources = {0, 1};
  });
  // Near-maximum admission must not overflow merely from multiplying raw by
  // 103. 100 * floor(UINT64_MAX/103) has a representable margined value.
  auto r = Fixture();
  r.geometry = true;
  r.devices.resize(1);
  r.devices[0].fixed_raw = {};
  r.devices[0].capacity_bytes = maximum;
  r.jobs.resize(1);
  r.jobs[0].private_raw = {0, 100 * (maximum / 103)};
  assert(BuildMultiGpuPhasePlan(r).groups[0].cohorts[0].active_owners == 1);
  r.jobs.clear();
  const auto empty = BuildMultiGpuPhasePlan(r);
  assert(empty.groups[0].cohorts.empty());
  assert(empty.host_peak_raw == r.common_host_raw);
}
}  // namespace
int main() {
  AssignmentAndAffinity();
  AffinityDoesNotStarveUsefulDevices();
  IndependentGeometryAndIdleDevices();
  CohortDrainingAndOwnerAdmission();
  ExtendingBankCanReduceCompleteOwnerPeak();
  PaddingAndResourcewiseOwnerPeaks();
  MarginAndSeparateReserves();
  UnequalCapsAndAggregateHost();
  InvalidDeclarationsAndOverflow();
  std::cout << "phase plan tests passed\n";
}
