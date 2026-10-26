// SPDX-License-Identifier: BSD-3-Clause
#include "colmap/mvs/optimized_photo_delivery.h"

#include <cassert>

int main() {
  using namespace colmap::mvs;
  // TEST-EVIDENCE: synthetic canonical charges and options digest, NOT real
  // attested model numbers. The closure digest alone is build-recomputed.
  const std::string digest(64, 'a');
  const PhotoPhaseV1Binding good{"photo-phase-v1",
                                 kPhotoPhaseV1Lineage,
                                 kPhotoPhaseV1Fingerprint,
                                 digest,
                                 {310000, 100},
                                 {200000, 900000},
                                 0,
                                 {4, 5},
                                 kPhotoPhaseV1SourceCommit};
  const PhotoPhaseV1Job job{
      "native-options", {310000, 100}, {200000, 900000}, 0, {4, 5}};
  const auto hash = [&](const std::string_view options) {
    assert(options == "native-options");
    return digest;
  };
  auto adapter =
      MakeTrustedPhotoPhaseV1Delivery("launch-bytes", {good}, {job}, hash);
  assert(adapter("launch-bytes").size() == 1);
  bool refused = false;
  try {
    adapter("different-launch-bytes");
  } catch (const std::invalid_argument& e) {
    refused = std::string(e.what()) == "PHOTO delivery launch bytes mismatch";
  }
  assert(refused);
  auto bad = good;
  bad.profile = "current-phase-v1";
  refused = false;
  try {
    MakeTrustedPhotoPhaseV1Delivery(
        "launch-bytes", {bad}, {job}, hash)("launch-bytes");
  } catch (const std::invalid_argument& e) {
    refused = std::string(e.what()) == "PHOTO delivery unknown profile";
  }
  assert(refused);
  const auto reject = [&](std::vector<PhotoPhaseV1Binding> bindings,
                          std::vector<PhotoPhaseV1Job> jobs,
                          const std::string& reason) {
    bool rejected = false;
    try {
      MakeTrustedPhotoPhaseV1Delivery(
          "launch-bytes", bindings, jobs, hash)("launch-bytes");
    } catch (const std::invalid_argument& e) {
      rejected = e.what() == reason;
    }
    assert(rejected);
  };
  // Legacy delivery omits the per-build revision and must fail closed.
  bad = good;
  bad.source_commit.clear();
  reject({bad}, {job}, "source_commit-revision-mismatch");
  for (const auto& revision : {std::string(39, 'a'),
                               std::string(41, 'a'),
                               std::string(40, 'A'),
                               std::string(40, 'g'),
                               std::string(40, 'b')}) {
    bad = good;
    bad.source_commit = revision;
    reject({bad}, {job}, "source_commit-revision-mismatch");
  }
  // Measured hazard: a9ff74f9f and 0611de098 shared identical closure bytes.
  // Even a matching closure cannot authorize a different build revision.
  bad = good;
  bad.source_commit = "a9ff74f9f729127d02ccf39fd2ec707ef1d593a6";
  assert(bad.source_commit != kPhotoPhaseV1SourceCommit);
  reject({bad}, {job}, "source_commit-revision-mismatch");
  reject({}, {job}, "PHOTO delivery missing");
  reject({good}, {}, "PHOTO delivery missing");
  reject({good, good}, {job}, "PHOTO delivery count or order mismatch");
  for (const auto& value : {std::string(39, 'a'),
                            std::string(41, 'a'),
                            std::string(40, 'A'),
                            std::string(40, 'g')}) {
    bad = good;
    bad.lineage_sha = value;
    reject({bad}, {job}, "PHOTO delivery malformed lineage");
  }
  for (const auto& value : {std::string(),
                            std::string(63, 'a'),
                            std::string(65, 'a'),
                            std::string(64, 'A'),
                            std::string(64, 'g')}) {
    bad = good;
    bad.fingerprint = value;
    reject({bad}, {job}, "PHOTO delivery malformed fingerprint");
    bad = good;
    bad.options_sha256 = value;
    reject({bad}, {job}, "PHOTO delivery malformed options");
  }
  bad = good;
  bad.fingerprint = std::string(64, 'b');
  reject({bad}, {job}, "lineage-source_commit-incoherent");
  // The delivered historical constant MUST NOT pass against a changed closure.
  if (std::string(kPhotoPhaseV1Fingerprint) !=
      "c8468ff4f513c49cae1f07b42dd2b7b5c5058b4e2517d7dfe770919f7513f604") {
    bad.fingerprint =
        "c8468ff4f513c49cae1f07b42dd2b7b5c5058b4e2517d7dfe770919f7513f604";
    reject({bad}, {job}, "lineage-source_commit-incoherent");
  }
  bad = good;
  bad.lineage_sha = std::string(40, 'b');
  reject({bad}, {job}, "PHOTO delivery lineage incompatible");
  bad = good;
  bad.options_sha256 = std::string(64, 'b');
  reject({bad}, {job}, "PHOTO delivery options mismatch");
  bad = good;
  bad.reference_image_idx = 1;
  reject({bad}, {job}, "PHOTO delivery count or order mismatch");
  bad = good;
  bad.source_image_idxs = {5, 4};
  reject({bad}, {job}, "PHOTO delivery per-reference mismatch");
  bad.source_image_idxs = {4};
  reject({bad}, {job}, "PHOTO delivery per-reference mismatch");
  for (int field = 0; field < 4; ++field) {
    bad = good;
    uint64_t* charge[] = {&bad.process.host,
                          &bad.process.device,
                          &bad.private_charge.host,
                          &bad.private_charge.device};
    ++*charge[field];
    reject({bad}, {job}, "PHOTO delivery canonical charge mismatch");
    *charge[field] = 0;
    reject({bad}, {job}, "PHOTO delivery absent or zero charges");
  }
  // Independently supplied model/native evidence is also required to be usable.
  auto absent = job;
  absent.process = {};
  reject({good}, {absent}, "PHOTO delivery absent or zero charges");

  // Exercise the adapter through the existing production parse seam, not just
  // a direct callback. No source/options/charge evidence comes from the wire.
  const auto v2 = "COLMAP_OPTIMIZED_PHOTO_V2\n" + digest +
                  "\n310000 100 200000 900000\n0 2 4 5\n1\n" + digest +
                  " 2f61\n";
  const auto wire = "COLMAP_OPTIMIZED_PHOTO_V3\n1 1 1\n" + owned::Hex("/c0") +
                    "\n" + owned::Hex(v2) + "\n";
  auto trusted = MakeTrustedPhotoPhaseV1Delivery(wire, {good}, {job}, hash);
  auto parsed =
      ParseOptimizedPhotoLaunch(wire, "/c0", "1", {510000, 900100}, trusted);
  assert(parsed.bindings.size() == 1 && parsed.launch.admitted_n == 1);
  bool executed = false;
  parsed.WithCheckedInputs(
      0, 0, {4, 5}, {300000, 30000, 20000, 100, 100, 2}, [&] {
        executed = true;
      });
  assert(executed);
  // Delivery order is checked against independent effective order.
  auto second = good;
  second.reference_image_idx = 1;
  auto second_job = job;
  second_job.reference_image_idx = 1;
  reject({second, good},
         {job, second_job},
         "PHOTO delivery count or order mismatch");
  bool missing_hash = false;
  try {
    MakeTrustedPhotoPhaseV1Delivery(wire, {good}, {job}, {})(wire);
  } catch (const std::invalid_argument& e) {
    missing_hash = std::string(e.what()) == "PHOTO delivery missing";
  }
  assert(missing_hash);

  // RUNNER delivery objects, in launch order, use the same attestation and
  // production admission route. TEST-EVIDENCE only; charges already margined.
  const auto wire2 = "COLMAP_OPTIMIZED_PHOTO_V3\n2 2 2\n" + owned::Hex("/c0") +
                     "\n" + owned::Hex(v2) + "\n" + owned::Hex("/c1") + "\n" +
                     owned::Hex("COLMAP_OPTIMIZED_PHOTO_V2\n" + digest +
                                "\n310000 100 200000 900000\n1 2 4 5\n1\n" +
                                digest + " 2f61\n") +
                     "\n";
  PhotoPhaseV1RunnerDelivery runner = [&](const std::string& bytes) {
    assert(bytes == wire2);
    return PhotoPhaseV1DeliveryInputs{
        wire2, {good, second}, {job, second_job}, hash};
  };
  auto from_runner = MakeTrustedPhotoPhaseV1Delivery(runner);
  auto admitted = ParseOptimizedPhotoLaunch(
      wire2, "/c0", "2", {710000, 1800100}, from_runner);
  assert(admitted.launch.admitted_n == 2);
  assert(admitted.bindings.size() == 2);
  for (size_t i = 0; i < 2; ++i) {
    bool checked = false;
    admitted.WithCheckedInputs(
        i, i == 0 ? 0 : 1, {4, 5}, {300000, 30000, 20000, 100, 100, 2}, [&] {
          checked = true;
        });
    assert(checked);
  }
  // Exact caps above prove no consumer margin was added to delivered charges.
  runner = [&](const std::string&) {
    return PhotoPhaseV1DeliveryInputs{
        "different-launch", {good, second}, {job, second_job}, hash};
  };
  refused = false;
  try {
    ParseOptimizedPhotoLaunch(wire2,
                              "/c0",
                              "2",
                              {710000, 1800100},
                              MakeTrustedPhotoPhaseV1Delivery(runner));
  } catch (const std::invalid_argument& e) {
    refused = std::string(e.what()) == "PHOTO delivery launch bytes mismatch";
  }
  assert(refused);
  assert(!MakeTrustedPhotoPhaseV1Delivery(PhotoPhaseV1RunnerDelivery{}));
  const auto section = [&](const PhotoPhaseV1Binding& b) {
    return "[BINDING]\nprofile=" + b.profile +
           "\nlineage_sha=" + b.lineage_sha + "\nfingerprint=" + b.fingerprint +
           "\noptions_sha256=" + b.options_sha256 +
           "\nprocess=310000 100\nprivate_charge=200000 900000"
           "\nreference_image_idx=" +
           std::to_string(b.reference_image_idx) +
           "\nsource_image_idxs=4 5\nsource_commit=" + b.source_commit +
           "\n[END]\n";
  };
  const auto wire_delivery = MakeWirePhotoPhaseV1Delivery(
      "/c0",
      "2",
      {710000, 1800100},
      [](const std::filesystem::path&) { return "native-options"; },
      hash);
  const auto with_binding = wire2 + section(good) + section(second);
  const auto supplied = ParseOptimizedPhotoLaunch(
      with_binding, "/c0", "2", {710000, 1800100}, wire_delivery);
  assert(supplied.launch.admitted_n == 2 && supplied.bindings.size() == 2);
  assert(wire_delivery(wire2).empty());
  // Absence must refuse as absent delivery even before forecast admission.
  const auto no_room = MakeWirePhotoPhaseV1Delivery(
      "/c0",
      "2",
      {1, 1},
      [](const std::filesystem::path&) { return "native-options"; },
      hash);
  assert(no_room(wire2).empty());

  const auto rejects_wire = [&](const std::string& text,
                                const std::string& expected) {
    bool rejected = false;
    try {
      ParseOptimizedPhotoLaunch(
          text, "/c0", "2", {710000, 1800100}, wire_delivery);
    } catch (const std::invalid_argument& e) {
      rejected = std::string(e.what()).find(expected) != std::string::npos;
    }
    assert(rejected);
  };
  rejects_wire(wire2, "PHOTO V3 requires delivered photo-phase-v1 binding");
  rejects_wire(wire2 + "[BINDING]\nunknown=value\n[END]\n",
               "PHOTO malformed binding section: unknown key unknown");
  rejects_wire(wire2 + "[BINDING]\nprofile=x\n[END]\n",
               "PHOTO malformed binding section: missing");
  rejects_wire(wire2 + section(good), "PHOTO binding section count mismatch");
  for (const auto& mutation : std::vector<std::pair<std::string, std::string>>{
           {"process=310000 100", "process=0310000 100"},
           {"process=310000 100", "process=18446744073709551616 100"},
           {"private_charge=200000 900000", "private_charge=1"},
           {"reference_image_idx=0", "reference_image_idx=-1"},
           {"source_image_idxs=4 5", "source_image_idxs=4 2147483648"},
           {"source_image_idxs=4 5", "source_image_idxs=4 5 "},
           {"profile=photo-phase-v1", "profile=photo-phase-v1\nprofile=x"},
           {"[END]\n", ""},
           {"[END]\n", "[END]"}}) {
    auto malformed = section(good);
    malformed.replace(
        malformed.find(mutation.first), mutation.first.size(), mutation.second);
    rejects_wire(wire2 + malformed + section(second),
                 "PHOTO malformed binding section:");
  }
  auto retired = good;
  retired.fingerprint =
      "d26041fafe7860b2653bafe6702ca97043881b14d953097112bcb4b7bde0117e";
  rejects_wire(wire2 + section(retired) + section(second),
               "lineage-source_commit-incoherent");
  // V2 bytes remain exclusively in the old parser; no delivery is synthesized.
  assert(wire_delivery(v2).empty());
  const auto legacy = ParseOptimizedPhotoLaunch(
      v2, "/c0", "1", {510000, 900100}, wire_delivery);
  assert(legacy.bindings.empty() && legacy.launch.admitted_n == 1);
  assert(legacy.launch.jobs[0].forecast.options_sha256 == digest);
}
