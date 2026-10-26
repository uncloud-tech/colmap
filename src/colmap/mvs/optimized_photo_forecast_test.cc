// SPDX-License-Identifier: BSD-3-Clause
#include "colmap/mvs/optimized_photo_forecast.h"

#include "colmap/mvs/owned_pair_run.h"

#include <cassert>
int main() {
  using namespace colmap::mvs;
  const std::string digest(64, 'a');
  const auto bytes = "COLMAP_OPTIMIZED_PHOTO_V2\n" + digest +
                     "\n1 2 3 4\n0 2 4 2\n1\n" + digest + " 2f61\n";
  const auto batch = ParseOptimizedPhotoLaunch(
      "COLMAP_OPTIMIZED_PHOTO_V3\n2 2 2\n" + owned::Hex("/c0") + "\n" +
          owned::Hex(bytes) + "\n" + owned::Hex("/c1") + "\n" +
          owned::Hex(std::string("COLMAP_OPTIMIZED_PHOTO_V2\n") + digest +
                     "\n1 2 3 4\n1 2 4 2\n1\n" + digest + " 2f61\n") +
          "\n",
      "/unused",
      "2",
      {20, 20});
  assert(batch.admitted_n == 2 && batch.jobs.size() == 2);
  const auto launch_bytes =
      "COLMAP_OPTIMIZED_PHOTO_V3\n2 2 2\n" + owned::Hex("/c0") + "\n" +
      owned::Hex(bytes) + "\n" + owned::Hex("/c1") + "\n" +
      owned::Hex(std::string("COLMAP_OPTIMIZED_PHOTO_V2\n") + digest +
                 "\n1 2 3 4\n1 2 4 2\n1\n" + digest + " 2f61\n") +
      "\n";
  // V3 extends the existing wire with verbatim typed per-job objects.
  const auto binding = [&](int ref) {
    return "[BINDING]\nprofile=photo-phase-v1\nlineage_sha=" +
           std::string(40, 'b') + "\nfingerprint=" + digest +
           "\noptions_sha256=" + digest +
           "\nprocess=1 2\nprivate_charge=3 4\nreference_image_idx=" +
           std::to_string(ref) +
           "\nsource_image_idxs=4 2\nsource_commit=" + std::string(40, 'c') +
           "\n[END]\n";
  };
  const auto bound = ParseOptimizedPhotoLaunch(
      launch_bytes + binding(0) + binding(1), "/c0", "2", {20, 20});
  assert(bound.bindings.size() == 2);
  assert(bound.bindings[1].reference_image_idx == 1);
  assert(bound.bindings[0].source_commit == std::string(40, 'c'));
  // Binding cardinality follows scheduled jobs, independently of ceiling k.
  auto third_forecast = bytes;
  third_forecast.replace(third_forecast.find("\n0 2 4 2\n"), 9, "\n3 2 4 2\n");
  auto three_jobs = launch_bytes;
  three_jobs.replace(three_jobs.find("2 2 2"), 5, "2 2 3");
  three_jobs += owned::Hex("/c3") + "\n" + owned::Hex(third_forecast) + "\n";
  const auto three_bindings = binding(0) + binding(1) + binding(3);
  const auto fewer_owners = ParseOptimizedPhotoLaunch(
      three_jobs + three_bindings, "/c0", "2", {20, 20});
  assert(fewer_owners.admitted_n == 2);
  assert(fewer_owners.jobs.size() == 3 && fewer_owners.bindings.size() == 3);
  auto larger_ceiling = launch_bytes;
  larger_ceiling.replace(larger_ceiling.find("2 2 2"), 5, "3 2 2");
  const auto fewer_jobs = ParseOptimizedPhotoLaunch(
      larger_ceiling + binding(0) + binding(1), "/c0", "2", {20, 20});
  assert(fewer_jobs.admitted_n == 2);
  assert(fewer_jobs.jobs.size() == 2 && fewer_jobs.bindings.size() == 2);
  const auto refuses_named = [&](const std::string& text,
                                 const std::string& diagnostic) {
    bool refused = false;
    try {
      ParseOptimizedPhotoLaunch(text, "/c0", "2", {20, 20});
    } catch (const std::invalid_argument& e) {
      refused = std::string(e.what()).find(diagnostic) != std::string::npos;
    }
    assert(refused);
  };
  // Matching k is neither necessary (above) nor sufficient (below).
  refuses_named(three_jobs + binding(0) + binding(1),
                "PHOTO binding section count mismatch");
  refuses_named(larger_ceiling + three_bindings,
                "PHOTO binding section count mismatch");
  auto insufficient_ceiling = three_jobs;
  insufficient_ceiling.replace(insufficient_ceiling.find("2 2 3"), 5, "1 2 3");
  refuses_named(insufficient_ceiling + three_bindings,
                "exceeds the supplied launch-contract ceiling k=1");
  const auto rejects =
      [&](std::string text, std::string n, owned::ByteCharge caps) {
        bool refused = false;
        try {
          ParseOptimizedPhotoLaunch(text, "/c0", n, caps);
        } catch (const std::exception&) {
          refused = true;
        }
        assert(refused);
      };
  rejects(launch_bytes, "3", {20, 20});  // EXACT CLI binding, never clamp.
  rejects(launch_bytes, "2", {6, 20});  // Host top-N, not just one fitting job.
  rejects(launch_bytes, "2", {20, 9});  // Device top-N independently.
  rejects(launch_bytes + "\n", "2", {20, 20});
  auto over_k = launch_bytes;
  over_k.replace(over_k.find("2 2 2"), 5, "1 2 2");
  rejects(over_k, "2", {20, 20});
  rejects("COLMAP_OPTIMIZED_PHOTO_V3\n2 2 1\n" + owned::Hex("/c0") + "\n" +
              owned::Hex(bytes) + "\n",
          "2",
          {20, 20});
  rejects(bytes, "2", {20, 20});  // V2 cannot authorize concurrency.
  auto wire = ParseOptimizedPhotoForecast(bytes);
  assert(wire.process.host == 1 && wire.private_charge.device == 4);
  assert(wire.files.at("/a") == digest);
  assert(wire.reference_image_idx == 0);
  assert((wire.source_image_idxs == std::vector<int>{4, 2}));
  ValidateOptimizedPhotoOrder(wire, 0, {4, 2});
  for (const auto& sources : {std::vector<int>{2, 4},
                              std::vector<int>{4},
                              std::vector<int>{4, 2, 3}}) {
    bool refused = false;
    try {
      ValidateOptimizedPhotoOrder(wire, 0, sources);
    } catch (const std::invalid_argument&) {
      refused = true;
    }
    assert(refused);
  }
  bool wrong_reference = false;
  try {
    ValidateOptimizedPhotoOrder(wire, 1, {4, 2});
  } catch (const std::invalid_argument&) {
    wrong_reference = true;
  }
  assert(wrong_reference);
  for (const auto* order : {"0 2 4 4",
                            "0 2 4 0",
                            "0 2 4",
                            "0 0",
                            "00 1 4",
                            "0 1 2147483648",
                            "0 1 4 ",
                            "0 1 -1"}) {
    bool refused = false;
    try {
      ParseOptimizedPhotoForecast("COLMAP_OPTIMIZED_PHOTO_V2\n" + digest +
                                  "\n1 2 3 4\n" + order + "\n1\n" + digest +
                                  " 2f61\n");
    } catch (const std::invalid_argument&) {
      refused = true;
    }
    assert(refused);
  }
  bool old_wire = false;
  try {
    ParseOptimizedPhotoForecast("COLMAP_OPTIMIZED_PHOTO_V1\n" + digest +
                                "\n1 2 3 4\n1\n" + digest + " 2f61\n");
  } catch (const std::invalid_argument&) {
    old_wire = true;
  }
  assert(old_wire);
  // The production PHOTO route uses the existing one-permit rolling owner
  // protocol, including postjoin retirement; it is not an unconstrained Run().
  owned::PairCoordinator coordinator(
      {10, 10},
      wire.process,
      size_t{1},
      std::vector<owned::ByteCharge>{wire.private_charge},
      std::chrono::milliseconds{500});
  owned::RunRollingOwners(
      coordinator,
      [&](size_t job) {
        coordinator.BeginPreparation(job);
        coordinator.ReadyAndWait(job);
        coordinator.SolverDone(job);
        coordinator.BeginOutput(job);
        coordinator.EndOutput(job);
        coordinator.AwaitOutputs(job);
      },
      [](size_t, const char*) {});
  assert(coordinator.ReservedBytes().host == 0);
  assert(coordinator.ReservedBytes().device == 0);
  for (const auto& bad : {bytes + "extra\n",
                          std::string("bad\n"),
                          "COLMAP_OPTIMIZED_PHOTO_V2\n" + digest +
                              "\n0 2 3 4\n0 2 4 2\n1\n" + digest + " 2f61\n"}) {
    bool rejected = false;
    try {
      ParseOptimizedPhotoForecast(bad);
    } catch (const std::invalid_argument&) {
      rejected = true;
    }
    assert(rejected);
  }
}
