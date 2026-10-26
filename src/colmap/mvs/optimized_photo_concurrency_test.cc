// SPDX-License-Identifier: BSD-3-Clause
// Standalone CPU seam for the PHOTO N1..Nk in-process dispatch contract:
//   g++ -std=c++17 -pthread -Isrc optimized_photo_concurrency_test.cc -o /tmp/t
//
// Covers ONLY the contract/guard/receipt layer: admission of a supplied PHOTO
// ceiling k, exact refusals (never clamping), receipt fields and the
// single-controller ownership/selection boundary under N owners. Kernel,
// scheduler and numerics are untouched by this slice, so no device is needed.
#include "colmap/mvs/optimized_dispatch.h"
#include "colmap/mvs/owned_selection_boundary.h"

#include <atomic>
#include <cassert>
#include <cstdio>
#include <deque>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

using colmap::mvs::AdmitOptimizedConcurrency;
using colmap::mvs::OptimizedConcurrencyLimits;
using colmap::mvs::OptimizedConcurrencyObservation;
using colmap::mvs::OptimizedDispatchReceipt;
using colmap::mvs::OptimizedPhotoMaxConcurrency;
using colmap::mvs::ParseOptimizedConcurrency;
using colmap::mvs::ValidateOptimizedStereoConcurrency;
using namespace colmap::mvs::owned;

// Contract violations are invalid_argument; compiled bounds are length_error;
// lifecycle misuse is logic_error. Every one of them is a refusal.
void Rejects(const std::function<void()>& body) {
  bool rejected = false;
  try {
    body();
  } catch (const std::invalid_argument&) {
    rejected = true;
  } catch (const std::length_error&) {
    rejected = true;
  } catch (const std::logic_error&) {
    rejected = true;
  }
  assert(rejected);
}

OptimizedConcurrencyLimits Photo(const std::string& concurrency,
                                 int ceiling,
                                 size_t job_count,
                                 bool full_bank = true,
                                 bool cohort = false) {
  OptimizedConcurrencyLimits limits;
  limits.geom_consistency = false;
  limits.concurrency = concurrency;
  limits.full_bank_forecast = full_bank;
  limits.cohort_forecast = cohort;
  limits.photo_ceiling = ceiling;
  limits.job_count = job_count;
  return limits;
}

OptimizedConcurrencyLimits Geo(const std::string& concurrency,
                               size_t job_count = 0,
                               bool full_bank = true,
                               bool cohort = false) {
  OptimizedConcurrencyLimits limits;
  limits.geom_consistency = true;
  limits.concurrency = concurrency;
  limits.full_bank_forecast = full_bank;
  limits.cohort_forecast = cohort;
  limits.photo_ceiling = 0;
  limits.job_count = job_count;
  return limits;
}

// Deterministic ordered source-draw vector with the shape of the compiled
// sample loop: one drawn source index per sample. The contract under test
// depends only on this resulting ordered vector, not on the device draw.
std::vector<int> Draws(const std::vector<int>& sources,
                       int samples,
                       uint64_t seed) {
  std::vector<int> draws;
  uint64_t state = seed;
  for (int sample = 0; sample < samples; ++sample) {
    state = state * 6364136223846793005ull + 1442695040888963407ull;
    draws.push_back(
        sources[static_cast<size_t>((state >> 33) % sources.size())]);
  }
  return draws;
}

// Observation of one owner's selection stream in the shape the compiled
// distinct-source cache exposes: slot index per draw plus hit/miss.
struct Trace {
  std::vector<int> slots;
  std::vector<int> hits;
  std::vector<int> final_slots;
  bool operator==(const Trace& other) const {
    return slots == other.slots && hits == other.hits &&
           final_slots == other.final_slots;
  }
};

void Accumulate(OwnedRollingSelectionCache& cache,
                const std::vector<int>& draws,
                Trace* trace) {
  for (const int source : draws) {
    bool hit = false;
    trace->slots.push_back(cache.Select(source, &hit));
    trace->hits.push_back(hit ? 1 : 0);
  }
  trace->final_slots = cache.Slots();
}

Trace Golden(const std::vector<int>& draws) {
  Trace trace;
  OwnedRollingSelectionCache cache;
  Accumulate(cache, draws, &trace);
  return trace;
}

bool AllHits(const Trace& trace) {
  for (const int hit : trace.hits) {
    if (!hit) return false;
  }
  return !trace.hits.empty();
}

}  // namespace

int main() {
  // Shared production PHOTO slots can turn over after join without retaining
  // the preceding generation's selection state; total jobs need not equal N.
  {
    OwnedSharedController shared(1, kMaxSelectionSlots, 3);
    for (int generation = 0; generation < 3; ++generation) {
      assert(shared.Begin(0) == static_cast<size_t>(generation));
      shared.WithSelection(0, [&](auto& selection) {
        assert(selection.Size() == 0);
        bool hit = true;
        assert(selection.Select(100 + generation, &hit) == 0);
        assert(!hit);
      });
      shared.End(0);
    }
    Rejects([&] { shared.Begin(0); });
    shared.Seal();
    assert(shared.Generations() == 3);
    assert(shared.SerializedSelections() == 3);
  }
  using namespace colmap::mvs;

  // ---- canonical CLI concurrency parsing --------------------------------
  assert(ParseOptimizedConcurrency("1") == 1);
  assert(ParseOptimizedConcurrency("2") == 2);
  assert(ParseOptimizedConcurrency("5") == 5);
  assert(ParseOptimizedConcurrency("15") == 15);
  for (const auto* refused :
       {"", "0", "02", " 2", "2 ", "+2", "-1", "2x", "2.0", "two"}) {
    Rejects([&] { (void)ParseOptimizedConcurrency(refused); });
  }

  // ---- PHOTO N1..Nk admission from a SUPPLIED ceiling -------------------
  // Compiled-not-qualified metadata stays 1: no PHOTO N>1 is claimed today.
  assert(OptimizedPhotoMaxConcurrency(false, false, 0) == 1);
  assert(OptimizedPhotoMaxConcurrency(true, false, 0) == 1);
  assert(OptimizedPhotoMaxConcurrency(true, false, 1) == 1);
  assert(OptimizedPhotoMaxConcurrency(true, false, 2) == 2);
  assert(OptimizedPhotoMaxConcurrency(true, false, 3) == 3);
  assert(OptimizedPhotoMaxConcurrency(true, false, 5) == 5);
  // A supplied ceiling above the compiled rolling ceiling does not raise it.
  assert(OptimizedPhotoMaxConcurrency(true, false, 9) == 5);
  assert(OptimizedPhotoMaxConcurrency(true, true, 4) == 1);

  // N1 is admitted exactly as today, with or without any forecast/ceiling.
  for (const bool full_bank : {false, true}) {
    for (const bool cohort : {false, true}) {
      for (const int ceiling : {0, 1, 5}) {
        assert(AdmitOptimizedConcurrency(
                   Photo("1", ceiling, 1, full_bank, cohort)) == 1);
      }
    }
  }
  // N>1 needs the full-bank forecast, no cohort and a supplied ceiling >= N.
  assert(AdmitOptimizedConcurrency(Photo("2", 2, 4)) == 2);
  assert(AdmitOptimizedConcurrency(Photo("3", 3, 6)) == 3);
  assert(AdmitOptimizedConcurrency(Photo("5", 5, 8)) == 5);
  assert(AdmitOptimizedConcurrency(Photo("2", 5, 8)) == 2);
  // Never clamp: an N above the supplied ceiling refuses, it is not reduced.
  Rejects([] { (void)AdmitOptimizedConcurrency(Photo("3", 2, 8)); });
  Rejects([] { (void)AdmitOptimizedConcurrency(Photo("2", 1, 8)); });
  // An unsupplied ceiling (0) refuses N>1 instead of estimating k here.
  Rejects([] { (void)AdmitOptimizedConcurrency(Photo("2", 0, 8)); });
  Rejects([] { (void)AdmitOptimizedConcurrency(Photo("2", -1, 8)); });
  // Never clamp: N above the compiled rolling structural ceiling refuses.
  for (const auto* refused : {"6", "7", "10"}) {
    Rejects([&] { (void)AdmitOptimizedConcurrency(Photo(refused, 9, 16)); });
  }
  // Forecast / cohort gates.
  Rejects(
      [] { (void)AdmitOptimizedConcurrency(Photo("2", 2, 8, false, false)); });
  Rejects(
      [] { (void)AdmitOptimizedConcurrency(Photo("2", 2, 8, true, true)); });
  // N>distinct-job-count refuses (never clamped down to the job count).
  Rejects([] { (void)AdmitOptimizedConcurrency(Photo("2", 2, 1)); });
  Rejects([] { (void)AdmitOptimizedConcurrency(Photo("3", 3, 2)); });
  // job_count 0 means "not bound at this call site"; the runner binds it.
  assert(AdmitOptimizedConcurrency(Photo("2", 2, 0)) == 2);
  // Non-canonical / out-of-contract CLI values refuse before any payload.
  for (const auto* refused : {"", "0", "02", " 2", "2 ", "+2", "-1", "2x"}) {
    Rejects([&] { (void)AdmitOptimizedConcurrency(Photo(refused, 5, 8)); });
  }
  // The CLI diagnostic names the admitted ceiling for BOTH phases, so a PHOTO
  // refusal is actionable and the resident CLI tests' contract stays literal.
  for (const auto& limits : {Photo("3", 1, 8), Geo("6", 0, true, false)}) {
    std::string what;
    try {
      (void)AdmitOptimizedConcurrency(limits);
    } catch (const std::invalid_argument& error) {
      what = error.what();
    }
    assert(what.find("optimized_only requires PHOTO concurrency1") !=
           std::string::npos);
    assert(what.find("no fallback") != std::string::npos);
  }

  // ---- GEO admission is unchanged ---------------------------------------
  assert(AdmitOptimizedConcurrency(Geo("1", 0, false, true)) == 1);
  assert(AdmitOptimizedConcurrency(Geo("1", 0, true, false)) == 1);
  assert(AdmitOptimizedConcurrency(Geo("2", 0, true, false)) == 2);
  assert(AdmitOptimizedConcurrency(Geo("5", 0, true, false)) == 5);
  Rejects([] { (void)AdmitOptimizedConcurrency(Geo("2", 0, false, false)); });
  Rejects([] { (void)AdmitOptimizedConcurrency(Geo("2", 0, true, true)); });
  Rejects([] { (void)AdmitOptimizedConcurrency(Geo("6", 0, true, false)); });
  Rejects([] { (void)AdmitOptimizedConcurrency(Geo("2", 1, true, false)); });
  // The production 4-argument entry point keeps its exact previous behavior.
  ValidateOptimizedStereoConcurrency(false, "1", false, false);
  ValidateOptimizedStereoConcurrency(true, "2", true, false);
  Rejects([] { ValidateOptimizedStereoConcurrency(false, "2", true, false); });
  Rejects([] { ValidateOptimizedStereoConcurrency(true, "2", true, true); });
  Rejects([] { ValidateOptimizedStereoConcurrency(true, "6", true, false); });

  // ---- receipts: N1 bytes identical to today ----------------------------
  // Literals captured from the pre-slice wire; changing them is a wire break.
  const std::string kPhotoN1Bytes =
      "{\"schema_version\":1,\"phase\":\"photo\",\"backend\":\"owned_cuda\","
      "\"tile\":16,\"cache\":true,\"lookahead\":true,\"lifetime\":"
      "\"preallocated_rotation\",\"fusion_mode\":null}\n";
  const std::string kGeoN1Bytes =
      "{\"schema_version\":1,\"phase\":\"geo\",\"backend\":\"owned_cuda\","
      "\"tile\":32,\"cache\":true,\"lookahead\":false,\"lifetime\":"
      "\"preallocated_rotation\",\"fusion_mode\":null}\n";
  const std::string kGeoN2Bytes =
      "{\"schema_version\":1,\"phase\":\"geo\",\"backend\":\"owned_cuda\","
      "\"tile\":32,\"cache\":true,\"lookahead\":false,\"lifetime\":"
      "\"preallocated_rotation\",\"fusion_mode\":null,"
      "\"max_concurrent_solvers\":2}\n";
  assert(OptimizedDispatchReceipt("photo", 16, 15, true, true) ==
         kPhotoN1Bytes);
  assert(OptimizedDispatchReceipt("geo", 32, 7, true, false) == kGeoN1Bytes);
  assert(OptimizedDispatchReceipt("geo", 32, 7, true, false, 2) == kGeoN2Bytes);
  // N1 must not gain concurrency fields even when telemetry is offered.
  Rejects([] {
    OptimizedConcurrencyObservation observation;
    observation.requested_n = 2;
    (void)OptimizedDispatchReceipt(
        "photo", 16, 15, true, true, 1, &observation);
  });
  assert(OptimizedDispatchReceipt("photo", 16, 15, true, true) ==
         kPhotoN1Bytes);

  // ---- receipts: PHOTO N>1 exposes requested/admitted/observed N ---------
  OptimizedConcurrencyObservation n2;
  n2.requested_n = 2;
  n2.admitted_n = 2;
  n2.observed_n = 2;
  n2.owner_intervals = {{0, 41}, {3, 44}};
  const std::string photo_n2 =
      OptimizedDispatchReceipt("photo", 16, 15, true, true, 2, &n2);
  assert(photo_n2.find("\"phase\":\"photo\"") != std::string::npos);
  assert(photo_n2.find("\"rolling_active_jobs\":2") != std::string::npos);
  assert(photo_n2.find("\"requested_n\":2") != std::string::npos);
  assert(photo_n2.find("\"admitted_n\":2") != std::string::npos);
  assert(photo_n2.find("\"observed_n\":2") != std::string::npos);
  assert(photo_n2.find("\"owner_intervals\":[{\"begin_us\":0,\"end_us\":41},"
                       "{\"begin_us\":3,\"end_us\":44}]") != std::string::npos);
  // Observed N is descriptive telemetry, not a physical-overlap claim.
  OptimizedConcurrencyObservation observed_one;
  observed_one.requested_n = 2;
  observed_one.admitted_n = 2;
  observed_one.observed_n = 1;
  observed_one.owner_intervals = {{0, 41}, {41, 44}};
  assert(OptimizedDispatchReceipt("photo", 16, 15, true, true, 2, &observed_one)
             .find("\"observed_n\":1") != std::string::npos);
  // Rolling active jobs must bind EXACTLY to the CLI value: no divergence.
  Rejects([&] {
    auto diverged = n2;
    diverged.requested_n = 3;
    (void)OptimizedDispatchReceipt("photo", 16, 15, true, true, 2, &diverged);
  });
  Rejects([&] {
    auto diverged = n2;
    diverged.admitted_n = 3;
    (void)OptimizedDispatchReceipt("photo", 16, 15, true, true, 2, &diverged);
  });
  Rejects([&] {
    auto bad = n2;
    bad.observed_n = 0;
    (void)OptimizedDispatchReceipt("photo", 16, 15, true, true, 2, &bad);
  });
  Rejects([&] {
    auto bad = n2;
    bad.observed_n = 3;
    (void)OptimizedDispatchReceipt("photo", 16, 15, true, true, 2, &bad);
  });
  Rejects([&] {
    auto bad = n2;
    bad.owner_intervals.pop_back();
    (void)OptimizedDispatchReceipt("photo", 16, 15, true, true, 2, &bad);
  });
  Rejects([&] {
    auto bad = n2;
    bad.owner_intervals = {{5, 4}, {3, 44}};
    (void)OptimizedDispatchReceipt("photo", 16, 15, true, true, 2, &bad);
  });
  Rejects([&] {
    auto bad = n2;
    bad.owner_intervals = {{9, 40}, {3, 44}};
    (void)OptimizedDispatchReceipt("photo", 16, 15, true, true, 2, &bad);
  });
  // PHOTO N>1 requires the descriptive fields; silence is not evidence.
  Rejects([] { OptimizedDispatchReceipt("photo", 16, 15, true, true, 2); });
  // GEO N>1 keeps its exact previous bytes, so telemetry is refused there.
  Rejects([&] { OptimizedDispatchReceipt("geo", 32, 7, true, false, 2, &n2); });
  // Compiled capability and range gates still apply at N>1.
  Rejects(
      [&] { OptimizedDispatchReceipt("photo", 8, 15, true, true, 2, &n2); });
  Rejects(
      [&] { OptimizedDispatchReceipt("photo", 16, 16, true, true, 2, &n2); });
  Rejects(
      [&] { OptimizedDispatchReceipt("photo", 16, 15, false, true, 2, &n2); });
  Rejects([&] {
    auto six = n2;
    six.requested_n = six.admitted_n = six.observed_n = 6;
    six.owner_intervals.assign(6, {0, 1});
    (void)OptimizedDispatchReceipt("photo", 16, 15, true, true, 6, &six);
  });
  Rejects([&] {
    auto bad = n2;
    (void)OptimizedDispatchReceipt("fusion", 0, 0, false, false, 2, &bad);
  });

  // ---- rolling distinct-source selection cache (compiled semantics) ------
  {
    OwnedRollingSelectionCache cache;
    assert(cache.Capacity() == kMaxSelectionSlots);
    assert(kMaxSelectionSlots == 15);
    for (int source = 0; source < 15; ++source) {
      bool hit = true;
      assert(cache.Select(1000 + source, &hit) == source);
      assert(!hit);
    }
    assert(cache.Size() == 15);
    // A repeat is a hit on the same slot and never grows the cache.
    bool hit = false;
    assert(cache.Select(1007, &hit) == 7);
    assert(hit);
    assert(cache.Size() == 15);
    assert(cache.Hits() == 1);
    assert(cache.Misses() == 15);
    // At most one slot per processed sample and <= 15 samples: a 16th distinct
    // source is a contract violation, never a silent wrap.
    Rejects([&] {
      bool ignored = false;
      (void)cache.Select(4242, &ignored);
    });
  }
  {
    OwnedRollingSelectionCache small(3);
    for (int source : {4, 5, 6}) {
      bool hit = true;
      (void)small.Select(source, &hit);
      assert(!hit);
    }
    bool hit = false;
    assert(small.Select(5, &hit) == 1);
    assert(hit);
    Rejects([&] {
      bool ignored = false;
      (void)small.Select(7, &ignored);
    });
    Rejects([] { OwnedRollingSelectionCache zero(0); });
    Rejects([] { OwnedRollingSelectionCache over(16); });
  }

  // ---- unguarded sharing is NOT race safe (deterministic interleaving) ---
  // Two owners with distinct references through ONE unpartitioned cache: the
  // alternating interleaving contaminates both selection streams, so no test
  // can prove the raw shared LRU race safe. That is why this slice partitions
  // and serializes at the selection boundary instead of claiming a safe race.
  const std::vector<int> hazard_a{1, 2, 3, 4, 5, 6};
  const std::vector<int> hazard_b{101, 102, 103, 104, 105, 106};
  const std::vector<int> hazard_draws_a = Draws(hazard_a, 15, 11);
  const std::vector<int> hazard_draws_b = Draws(hazard_b, 15, 29);
  const Trace hazard_golden_a = Golden(hazard_draws_a);
  const Trace hazard_golden_b = Golden(hazard_draws_b);
  {
    OwnedRollingSelectionCache shared;
    Trace a;
    Trace b;
    for (size_t i = 0; i < hazard_draws_a.size(); ++i) {
      Accumulate(shared, {hazard_draws_a[i]}, &a);
      Accumulate(shared, {hazard_draws_b[i]}, &b);
    }
    a.final_slots = shared.Slots();
    b.final_slots = shared.Slots();
    assert(!(a == hazard_golden_a));
    assert(!(b == hazard_golden_b));
  }
  // The same interleaving through the boundary reproduces both goldens.
  {
    OwnedSharedController guarded(2);
    guarded.Begin(0);
    guarded.Begin(1);
    Trace a;
    Trace b;
    for (size_t i = 0; i < hazard_draws_a.size(); ++i) {
      guarded.WithSelection(0, [&](OwnedRollingSelectionCache& cache) {
        Accumulate(cache, {hazard_draws_a[i]}, &a);
      });
      guarded.WithSelection(1, [&](OwnedRollingSelectionCache& cache) {
        Accumulate(cache, {hazard_draws_b[i]}, &b);
      });
    }
    assert(a == hazard_golden_a);
    assert(b == hazard_golden_b);
    guarded.End(0);
    guarded.End(1);
    guarded.Seal();
  }

  // ---- selection boundary and single-controller ownership under N -------
  {
    OwnedSelectionBoundary boundary(2);
    assert(boundary.Owners() == 2);
    Rejects([&] { boundary.Run(2, [] {}); });  // not a declared owner
    assert(boundary.SerializedCalls() == 0);
    boundary.Run(0, [] {});
    assert(boundary.SerializedCalls() == 1);
    // A throwing body releases the lease; the boundary stays usable.
    try {
      boundary.Run(0, [] { throw std::logic_error("body"); });
    } catch (const std::logic_error&) {
    }
    boundary.Run(1, [] {});
    assert(boundary.SerializedCalls() == 3);
    // A nested lease would deadlock, so it refuses instead of hanging.
    Rejects([&] { boundary.Run(0, [&] { boundary.Run(1, [] {}); }); });
    boundary.Run(0, [] {});
    assert(boundary.SerializedCalls() == 5);
    Rejects([] { OwnedSelectionBoundary none(0); });
    Rejects([] { OwnedSelectionBoundary wide(6); });
  }
  {
    OwnedSharedController controller(2);
    assert(controller.Owners() == 2);
    assert(controller.Begin(0) == 0);
    assert(controller.Begin(1) == 1);
    Rejects([&] { (void)controller.Begin(0); });  // duplicate owner
    Rejects([&] { (void)controller.Begin(2); });  // not a declared owner
    Rejects([&] { controller.Seal(); });          // live generations
    assert(controller.Observed() == 2);
    controller.End(0);
    controller.End(1);
    // A third generation on a two-generation controller refuses: the compiled
    // controller is single-use today, so no reuse is fabricated.
    Rejects([&] { (void)controller.Begin(0); });
    controller.Seal();
    Rejects([&] { (void)controller.Begin(1); });
    const auto intervals = controller.Intervals();
    assert(intervals.size() == 2);
    for (const auto& interval : intervals) {
      assert(interval.begin_micros <= interval.end_micros);
    }
    OptimizedConcurrencyObservation observation;
    observation.requested_n = 2;
    observation.admitted_n = 2;
    observation.observed_n = static_cast<int>(controller.Observed());
    observation.owner_intervals = intervals;
    const auto receipt =
        OptimizedDispatchReceipt("photo", 32, 15, true, true, 2, &observation);
    assert(receipt.find("\"observed_n\":2") != std::string::npos);
    assert(receipt.find("\"rolling_active_jobs\":2") != std::string::npos);
    assert(receipt.find("\"owner_intervals\":[{\"begin_us\":") !=
           std::string::npos);
  }

  // N owners, distinct references, hits and a full 15-slot fill, real threads.
  {
    constexpr int kRounds = 200;
    const std::vector<int> owner_sources[2] = {
        {1, 2, 3, 4, 5, 6, 7, 8}, {101, 102, 103, 104, 105, 106, 107, 108}};
    const std::vector<int> owner_draws[2] = {Draws(owner_sources[0], 15, 11),
                                             Draws(owner_sources[1], 15, 29)};
    const Trace golden[2] = {Golden(owner_draws[0]), Golden(owner_draws[1])};
    OwnedSharedController controller(2);
    std::atomic<int> ready{0};
    std::vector<size_t> generations(2);
    for (size_t owner = 0; owner < 2; ++owner) {
      generations[owner] = controller.Begin(owner);
      assert(generations[owner] == owner);
    }
    std::vector<std::thread> owners;
    for (size_t owner = 0; owner < 2; ++owner) {
      owners.emplace_back([&, owner] {
        ++ready;
        while (ready.load() < 2) std::this_thread::yield();
        for (int round = 0; round < kRounds; ++round) {
          Trace round_trace;
          controller.WithSelection(
              owner, [&](OwnedRollingSelectionCache& cache) {
                Accumulate(cache, owner_draws[owner], &round_trace);
              });
          if (round == 0) {
            assert(round_trace == golden[owner]);
          } else {
            // The same ordered draws against the retained owner-private cache:
            // identical slots, every draw a hit, never another owner's slot.
            assert(round_trace.slots == golden[owner].slots);
            assert(AllHits(round_trace));
            assert(round_trace.final_slots == golden[owner].final_slots);
          }
        }
      });
    }
    for (auto& owner : owners) owner.join();
    controller.End(0);
    controller.End(1);
    controller.Seal();
    assert(controller.Observed() == 2);
    assert(controller.Generations() == 2);
  }

  // Receipt telemetry from the REAL coordinator event stream: N permits admit
  // N owners concurrently, the observer reports what was admitted/joined, and
  // the PHOTO N>1 receipt accepts that observation. N1 keeps observed N == 1.
  for (const size_t permits : {size_t{2}, size_t{1}}) {
    OwnedConcurrencyObserver observer(static_cast<int>(permits),
                                      static_cast<int>(permits));
    constexpr size_t kJobs = 4;
    PairCoordinator coordinator(
        {2 + kJobs * 10, 3 + kJobs * 20},
        {2, 3},
        permits,
        std::vector<ByteCharge>(kJobs, {10, 20}),
        std::chrono::milliseconds(2000),
        [&](const PairEvent& event) { observer(event); });
    std::deque<size_t> active;
    const auto pump = [&] {
      while (true) {
        const size_t job = coordinator.AdmitNext();
        if (job == std::numeric_limits<size_t>::max()) break;
        active.push_back(job);
      }
    };
    pump();
    size_t joined = 0;
    while (!active.empty()) {
      const size_t job = active.front();
      active.pop_front();
      coordinator.BeginPreparation(job);
      coordinator.ReadyAndWait(job);
      coordinator.SolverDone(job);
      coordinator.BeginOutput(job);
      coordinator.EndOutput(job);
      coordinator.AwaitOutputs(job);
      coordinator.Retire(job, true);
      coordinator.Joined(job);
      ++joined;
      pump();
    }
    assert(joined == kJobs);
    coordinator.Finish();
    assert(observer.Observed() == permits);
    const auto observation = observer.Observation();
    assert(observation.requested_n == static_cast<int>(permits));
    assert(observation.admitted_n == static_cast<int>(permits));
    assert(observation.observed_n == static_cast<int>(permits));
    assert(observation.owner_intervals.size() == kJobs);
    for (const auto& interval : observation.owner_intervals) {
      assert(interval.begin_micros <= interval.end_micros);
    }
    const auto receipt =
        OptimizedDispatchReceipt("photo",
                                 16,
                                 15,
                                 true,
                                 true,
                                 static_cast<int>(permits),
                                 permits > 1 ? &observation : nullptr);
    if (permits > 1) {
      assert(receipt.find("\"observed_n\":" + std::to_string(permits)) !=
             std::string::npos);
      assert(receipt.find("\"rolling_active_jobs\":" +
                          std::to_string(permits)) != std::string::npos);
    } else {
      // N1 keeps the historical bytes even though 4 generations ran serially.
      assert(receipt == kPhotoN1Bytes);
    }
  }

  // Owner-private state is unreachable outside a live generation.
  {
    OwnedSharedController controller(1);
    controller.Begin(0);
    Rejects([&] {
      controller.WithSelection(1, [](OwnedRollingSelectionCache&) {});
    });
    controller.End(0);
    Rejects([&] {
      controller.WithSelection(0, [](OwnedRollingSelectionCache&) {});
    });
    controller.Seal();
  }

  // N1 identity of the selection model path: one owner produces exactly the
  // same stream as the unguarded single-owner baseline.
  {
    const std::vector<int> draws = Draws(hazard_a, 15, 11);
    const Trace baseline = Golden(draws);
    OwnedSharedController controller(1);
    assert(controller.Observed() == 0);
    controller.Begin(0);
    Trace trace;
    controller.WithSelection(0, [&](OwnedRollingSelectionCache& cache) {
      Accumulate(cache, draws, &trace);
    });
    controller.End(0);
    controller.Seal();
    assert(trace == baseline);
    assert(controller.Observed() == 1);
    assert(controller.Generations() == 1);
    OptimizedConcurrencyObservation observation;
    observation.observed_n = static_cast<int>(controller.Observed());
    observation.owner_intervals = controller.Intervals();
    // N1 keeps the historical single-solver receipt bytes: the observation is
    // telemetry only and never rewrites the N1 wire.
    assert(OptimizedDispatchReceipt("photo", 16, 15, true, true) ==
           kPhotoN1Bytes);
    assert(observation.owner_intervals.size() == 1);
  }

  std::printf("optimized_photo_concurrency_test: PASS\n");
  return 0;
}
