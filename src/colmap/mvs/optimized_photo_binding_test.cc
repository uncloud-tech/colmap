// SPDX-License-Identifier: BSD-3-Clause
#include "colmap/mvs/optimized_photo_reservation.h"

#include <cassert>
#include <functional>

int main() {
  using namespace colmap::mvs;
  const std::string digest(64, 'a');
  const auto rejects = [](const std::function<void()>& body) {
    bool rejected = false;
    try {
      body();
    } catch (const std::exception&) {
      rejected = true;
    }
    assert(rejected);
  };
  // Synthetic delivery objects, not calibrated model numbers or a claimed k.
  for (const int n : {2, 3}) {
    std::string bytes = "COLMAP_OPTIMIZED_PHOTO_V3\n" + std::to_string(n) +
                        " " + std::to_string(n) + " " + std::to_string(n) +
                        "\n";
    std::vector<PhotoPhaseV1Binding> bindings;
    for (int ref = 0; ref < n; ++ref) {
      const auto v2 = "COLMAP_OPTIMIZED_PHOTO_V2\n" + digest +
                      "\n310000 100 200000 900000\n" + std::to_string(ref) +
                      " 2 4 5\n1\n" + digest + " 2f61\n";
      bytes +=
          owned::Hex("/c" + std::to_string(ref)) + "\n" + owned::Hex(v2) + "\n";
      bindings.push_back({"photo-phase-v1",
                          std::string(40, 'b'),
                          digest,
                          digest,
                          {310000, 100},
                          {200000, 900000},
                          ref,
                          {4, 5}});
    }
    // Independently worked aggregate totals: process once + N private owners.
    const owned::ByteCharge caps = n == 2 ? owned::ByteCharge{710000, 1800100}
                                          : owned::ByteCharge{910000, 2700100};
    const auto launch = AdmitBoundOptimizedPhotoLaunch(
        bytes, "/c0", std::to_string(n), caps, bindings);
    assert(launch.admitted_n == n && launch.jobs.size() == size_t(n));
    const OptimizedPhotoWorkload workload{300000, 30000, 20000, 100, 100, 2};
    const auto legacy_absent_binding_floor =
        LegacyAbsentBindingPhotoFloor(workload);
    assert(legacy_absent_binding_floor.device >
           bindings.front().private_charge.device);
    assert(100 + n * legacy_absent_binding_floor.device > caps.device);
    for (size_t i = 0; i < launch.jobs.size(); ++i)
      CheckOptimizedPhotoReservation(
          launch.jobs[i].forecast, workload, &bindings[i]);
    rejects([&] {
      AdmitBoundOptimizedPhotoLaunch(bytes, "/c0", std::to_string(n), caps);
    });
    for (int mutation = 0; mutation < 20; ++mutation) {
      auto bad = bindings;
      auto& b = bad.back();
      if (mutation == 0) b.profile = "current-phase-v1";
      if (mutation == 1) b.lineage_sha.clear();
      if (mutation == 2) b.lineage_sha = std::string(40, 'B');
      if (mutation == 3) b.fingerprint = std::string(63, 'a');
      if (mutation == 4) b.fingerprint = std::string(64, 'g');
      if (mutation == 5) b.options_sha256 = std::string(64, 'b');
      if (mutation == 6) ++b.process.host;
      if (mutation == 7) ++b.process.device;
      if (mutation == 8) ++b.private_charge.host;
      if (mutation == 9) ++b.private_charge.device;
      if (mutation == 10) ++b.reference_image_idx;
      if (mutation == 11) b.source_image_idxs = {5, 4};
      if (mutation == 12) bad.pop_back();
      if (mutation == 13) bad.push_back(bad.front());
      if (mutation == 14) b.lineage_sha = std::string(39, 'b');
      if (mutation == 15) b.lineage_sha = std::string(41, 'b');
      if (mutation == 16) b.lineage_sha = std::string(40, 'g');
      if (mutation == 17) b.fingerprint = std::string(65, 'a');
      if (mutation == 18) b.fingerprint = std::string(64, 'A');
      if (mutation == 19) b.fingerprint.clear();
      rejects([&] {
        AdmitBoundOptimizedPhotoLaunch(
            bytes, "/c0", std::to_string(n), caps, bad);
      });
    }
    for (const owned::ByteCharge lower :
         {owned::ByteCharge{caps.host - 1, caps.device},
          owned::ByteCharge{caps.host, caps.device - 1}}) {
      rejects([&] {
        AdmitBoundOptimizedPhotoLaunch(
            bytes, "/c0", std::to_string(n), lower, bindings);
      });
    }
  }
  // A synthetic 100x100 workload: the historical mask/96WH layout floor
  // exceeds the delivered phase charge. It must not silently limit PHOTO.
  const OptimizedPhotoWorkload workload{300000, 30000, 20000, 100, 100, 2};
  const auto legacy_absent_binding_floor =
      LegacyAbsentBindingPhotoFloor(workload);
  const PhotoPhaseV1Binding phase{"photo-phase-v1",
                                  std::string(40, 'b'),
                                  digest,
                                  digest,
                                  {310000, 100},
                                  {200000, 900000},
                                  0,
                                  {1, 2}};
  OptimizedPhotoForecast forecast{
      digest, phase.process, phase.private_charge, 0, {1, 2}, {}};
  assert(legacy_absent_binding_floor.device > phase.private_charge.device);
  assert(legacy_absent_binding_floor.host > phase.private_charge.host);
  assert(phase.private_charge.host < workload.cache);
  CheckOptimizedPhotoReservation(forecast, workload, &phase);
  rejects([&] { CheckOptimizedPhotoReservation(forecast, workload); });
  // An invalid delivery must never unlock the lower floor.
  auto invalid = phase;
  invalid.fingerprint.clear();
  rejects(
      [&] { CheckOptimizedPhotoReservation(forecast, workload, &invalid); });
  auto small_process = phase;
  small_process.process.host = workload.cache - 1;
  auto small_forecast = forecast;
  small_forecast.process = small_process.process;
  rejects([&] {
    CheckOptimizedPhotoReservation(small_forecast, workload, &small_process);
  });
  auto undersized_cache = workload;
  undersized_cache.cache = workload.retained - 1;
  rejects([&] {
    CheckOptimizedPhotoReservation(forecast, undersized_cache, &phase);
  });
  rejects([&] { CheckOptimizedPhotoReservation(forecast, undersized_cache); });
  // The legacy seam retains its exact logical envelope, on each side.
  forecast.private_charge = legacy_absent_binding_floor;
  CheckOptimizedPhotoReservation(forecast, workload);
  --forecast.private_charge.host;
  rejects([&] { CheckOptimizedPhotoReservation(forecast, workload); });
  ++forecast.private_charge.host;
  --forecast.private_charge.device;
  rejects([&] { CheckOptimizedPhotoReservation(forecast, workload); });

  const auto v2 = "COLMAP_OPTIMIZED_PHOTO_V2\n" + digest +
                  "\n1 2 3 4\n0 1 2\n1\n" + digest + " 2f61\n";
  assert(AdmitBoundOptimizedPhotoLaunch(v2, "/c0", "1", {4, 6}).admitted_n ==
         1);
}
