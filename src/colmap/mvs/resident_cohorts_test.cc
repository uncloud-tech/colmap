// SPDX-License-Identifier: BSD-3-Clause

// CPU structural/arithmetic tests for the resident cohort validated-plan
// boundary (resident_cohorts.h). The REAL planner (PlanCohorts) and the REAL
// typed boundary validator are exercised with a standalone test estimator;
// no payloads, no CUDA, no coordinator, no wire. All expected arithmetic
// values below are independently derived literals, never recomputed through
// the code under test.
#include "colmap/mvs/owned_cohort.h"
#include "colmap/mvs/resident_cohorts.h"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include <gtest/gtest.h>

namespace colmap {
namespace mvs {
namespace {

using owned::ByteCharge;
using owned::CohortPlan;
using owned::CohortTerms;

// A shared deterministic scenario used by several tests. Selection of
// original resolved bank job ids (strictly increasing, identity order):
//   selection = {7, 11, 15}
// Per-job complete private forecasts (host, device):
//   7 -> (300, 200);  11 -> (250, 150);  15 -> (400, 300)
// Planner union-bank estimates per contiguous selection slice (host, dev):
//   (1 job)  7 -> (800, 300)  11 -> (700, 250)  15 -> (900, 400)
//   (7,11)   -> (1100, 420)   (11,15) -> (1200, 520)  (7,11,15) -> (1400,600)
// Terms (canonical composition): process (500,600), queue_transient
// (120,0) = queue + bounded normal charge, explicit-legacy reviewed_bank
// (0,0), explicitly-{0,0} unclassified (the canonical allowance sits
// inside the allowance-inclusive bank charge). Capacity (2300,1500).
// Concurrency 2.
//
// PlanCohorts packs {7,11} (required host 1100+620+300=2030) and closes
// before adding 15 (required host 1400+620+400=2430 > 2300), leaving {15}.
std::vector<size_t> TestSelection() { return {7, 11, 15}; }

std::vector<ByteCharge> TestForecasts() {
  return {{300, 200}, {250, 150}, {400, 300}};
}

ByteCharge TestCapacityScalar() { return {2300, 1500}; }

// Test bank estimator with the union semantics tabulated above.
owned::CohortBankEstimator TestEstimator() {
  return [](size_t first, size_t count) -> ByteCharge {
    static const ByteCharge kSingleton[3] = {{800, 300}, {700, 250},
                                             {900, 400}};
    static const ByteCharge kUnions[2] = {{1100, 420}, {1400, 600}};
    // (11,15) union is only needed by planners that pack that slice; the
    // greedy FIFO packing never forms it in the scenarios below.
    if (count == 1) return kSingleton[first];
    if (first == 0 && count == 2) return kUnions[0];
    if (first == 1 && count == 2) return {1200, 520};
    if (first == 0 && count == 3) return kUnions[1];
    throw std::invalid_argument("test estimator: untabulated slice");
  };
}

CohortTerms TestTerms() {
  CohortTerms terms;
  terms.process = {500, 600};
  terms.queue_transient = {120, 0};
  terms.unclassified = {30, 0};
  // EXPLICIT legacy estimate-only packing contract: the reviewed bank
  // reservation is deliberately and explicitly zero here (its zero is the
  // documented legacy policy, not an implicit allowance); the reviewed
  // reservation policy is covered by a dedicated test below. CANONICAL
  // COMPOSITION: the unclassified term is explicitly {0,0} — the allowance
  // is inside the allowance-inclusive bank charge.
  terms.unclassified = {0, 0};
  terms.reviewed_bank = {0, 0};
  return terms;
}

// Distinct ORIGINAL identity records per problem (one reference input
// identity each, distinct exact paths). The boundary must RETAIN these
// original records in the returned object; identical placeholder records
// could never demonstrate retention.
std::vector<owned::BankJobIdentities> TestIdentities(size_t count) {
  std::vector<owned::BankJobIdentities> identities(count);
  for (size_t k = 0; k < count; ++k) {
    owned::BankLayerIdentity reference;
    const std::string prefix = "invocation/problem-" + std::to_string(k);
    reference.image.path = prefix + "-image-original.bin";
    reference.depth.path = prefix + "-depth-original.bin";
    reference.normal.path = prefix + "-normal-original.bin";
    identities[k].push_back(reference);
  }
  return identities;
}

TEST(ResidentCohortBoundary, ValidatesRealPlannerPlanToTypedBoundary) {
  const auto selection = TestSelection();
  const auto plan = owned::PlanCohorts(selection, TestForecasts(), TestTerms(),
                                       TestCapacityScalar(), TestEstimator());
  // The scenario must produce the expected two FIFO cohorts {7,11}, {15}.
  ASSERT_EQ(plan.cohorts.size(), 2u);
  EXPECT_EQ(plan.cohorts[0].first, 0u);
  EXPECT_EQ(plan.cohorts[0].count, 2u);
  EXPECT_EQ(plan.job_count, 3u);

  const resident::ValidatedCohortPlan validated =
      resident::ValidateCohortPlan(plan, selection, TestForecasts(),
                                   TestTerms(), TestCapacityScalar(),
                                   /*max_concurrent_solvers=*/2,
                                   TestEstimator(),
                                   TestIdentities(selection.size()));

  // The boundary is NOT execution-authorized, by type, even when fully
  // validated: per-cohort schema/binding/build/minima gates remain.
  EXPECT_FALSE(validated.ExecutionAuthorized());
  EXPECT_EQ(resident::ValidatedCohortPlan::kExecutionAuthorized, false);

  // Original FIFO identity order preserved exactly: no recapture, no sorting.
  ASSERT_EQ(validated.Cohorts().size(), 2u);
  EXPECT_EQ(validated.Cohorts()[0].original_jobs, (std::vector<size_t>{7, 11}));
  EXPECT_EQ(validated.Cohorts()[1].original_jobs, (std::vector<size_t>{15}));

  // Independent literals (see the scenario tabulation above):
  // cohort 0: bank estimate (1100,420); under the explicit legacy policy the
  // dedicated bank charge IS the plan estimate.
  EXPECT_EQ(validated.Cohorts()[0].bank_estimate.host, 1100u);
  EXPECT_EQ(validated.Cohorts()[0].bank_estimate.device, 420u);
  EXPECT_EQ(validated.Cohorts()[0].ledger_bank_charge.host, 1100u);
  EXPECT_EQ(validated.Cohorts()[0].ledger_bank_charge.device, 420u);
  // MAX membership (planner evidence) is per-side maxima of member privates:
  EXPECT_EQ(validated.Cohorts()[0].member_private_max.host, 300u);
  EXPECT_EQ(validated.Cohorts()[0].member_private_max.device, 200u);
  // Admitted concurrency (mandatory per cohort): min(declared 2, members 2).
  EXPECT_EQ(validated.Cohorts()[0].active_jobs, 2u);
  // The per-side largest-2 SUM (distinct from the MAX): (550,350).
  EXPECT_EQ(validated.Cohorts()[0].rolling_reserve.host, 550u);
  EXPECT_EQ(validated.Cohorts()[0].rolling_reserve.device, 350u);
  EXPECT_EQ(validated.Cohorts()[0].ledger_terms.host, 620u);
  EXPECT_EQ(validated.Cohorts()[0].ledger_terms.device, 600u);
  // COMPLETE per-cohort ledger: bank 1100 + terms 620 + reserve 550 = 2270
  // host; 420 + 600 + 350 = 1370 device.
  EXPECT_EQ(validated.Cohorts()[0].ledger.host, 2270u);
  EXPECT_EQ(validated.Cohorts()[0].ledger.device, 1370u);
  EXPECT_TRUE(validated.Cohorts()[0].ledger_fits_capacity);

  // cohort 1 is a singleton: active_jobs = min(2,1) = 1, so the reserve is
  // the single member (400,300); ledger 900+620+400=1920 / 400+600+300=1300.
  EXPECT_EQ(validated.Cohorts()[1].bank_estimate.host, 900u);
  EXPECT_EQ(validated.Cohorts()[1].bank_estimate.device, 400u);
  EXPECT_EQ(validated.Cohorts()[1].member_private_max.host, 400u);
  EXPECT_EQ(validated.Cohorts()[1].active_jobs, 1u);
  EXPECT_EQ(validated.Cohorts()[1].rolling_reserve.host, 400u);
  EXPECT_EQ(validated.Cohorts()[1].rolling_reserve.device, 300u);
  EXPECT_EQ(validated.Cohorts()[1].ledger.host, 1920u);
  EXPECT_EQ(validated.Cohorts()[1].ledger.device, 1300u);
  EXPECT_TRUE(validated.Cohorts()[1].ledger_fits_capacity);

  // Explicit original-job mapping (selection-ordinal space), in selection
  // order: 7 -> cohort 0 ordinal 0, 11 -> cohort 0 ordinal 1,
  // 15 -> cohort 1 ordinal 0.
  ASSERT_EQ(validated.Mapping().size(), 3u);
  EXPECT_EQ(validated.Mapping()[0].original_job, 7u);
  EXPECT_EQ(validated.Mapping()[0].selection_ordinal, 0u);
  EXPECT_EQ(validated.Mapping()[0].cohort, 0u);
  EXPECT_EQ(validated.Mapping()[0].cohort_local_ordinal, 0u);
  EXPECT_EQ(validated.Mapping()[1].original_job, 11u);
  EXPECT_EQ(validated.Mapping()[1].selection_ordinal, 1u);
  EXPECT_EQ(validated.Mapping()[1].cohort, 0u);
  EXPECT_EQ(validated.Mapping()[1].cohort_local_ordinal, 1u);
  EXPECT_EQ(validated.Mapping()[2].original_job, 15u);
  EXPECT_EQ(validated.Mapping()[2].selection_ordinal, 2u);
  EXPECT_EQ(validated.Mapping()[2].cohort, 1u);
  EXPECT_EQ(validated.Mapping()[2].cohort_local_ordinal, 0u);

  // OWNED IMMUTABLE identity retention: the returned object holds an exact
  // copy of the ORIGINAL records, distinct per problem; caller-side
  // mutations after validation cannot touch the retained copy.
  auto mutated_identities = TestIdentities(selection.size());
  const resident::ValidatedCohortPlan retained =
      resident::ValidateCohortPlan(plan, selection, TestForecasts(),
                                   TestTerms(), TestCapacityScalar(), 2,
                                   TestEstimator(), mutated_identities);
  mutated_identities[1][0].image.path = "recaptured-after-validation.bin";
  mutated_identities[2].clear();
  ASSERT_EQ(retained.InvocationIdentities().size(), 3u);
  EXPECT_EQ(retained.InvocationIdentities()[1][0].image.path,
            "invocation/problem-1-image-original.bin");
  EXPECT_EQ(retained.InvocationIdentities()[2][0].normal.path,
            "invocation/problem-2-normal-original.bin");
  EXPECT_EQ(retained.InvocationIdentities()[0][0].depth.path,
            "invocation/problem-0-depth-original.bin");
}

TEST(ResidentCohortBoundary, ReviewedReservationPolicyLedgerEvidence) {
  // Same selection/forecasts/estimator, but an explicit NONZERO reviewed bank
  // reservation: the packing policy charges the RESERVATION, not the
  // estimate, and a single cohort of all three jobs forms (required host
  // grows only when the member private MAX grows).
  const auto selection = TestSelection();
  CohortTerms terms = TestTerms();
  terms.reviewed_bank = {2000, 1400};
  const ByteCharge capacity{3600, 2600};
  const auto plan = owned::PlanCohorts(selection, TestForecasts(), terms,
                                       capacity, TestEstimator());
  ASSERT_EQ(plan.cohorts.size(), 1u);
  ASSERT_EQ(plan.cohorts[0].original_jobs.size(), 3u);

  const resident::ValidatedCohortPlan validated =
      resident::ValidateCohortPlan(plan, selection, TestForecasts(), terms,
                                   capacity, /*max_concurrent_solvers=*/2,
                                   TestEstimator(),
                                   TestIdentities(selection.size()));
  EXPECT_FALSE(validated.ExecutionAuthorized());

  // Independent literals: planned bank estimate = union of all three
  // (1400,600); the policy bank charge is the full reservation (2000,1400);
  // the rolling reserve is the per-side largest-2 member SUM (700,500); the
  // COMPLETE ledger is 2000+500+120+700 = 3320 host,
  // 1400+600+500 = 2500 device (canonical composition, no unclassified
  // term).
  EXPECT_EQ(validated.Cohorts()[0].bank_estimate.host, 1400u);
  EXPECT_EQ(validated.Cohorts()[0].bank_estimate.device, 600u);
  EXPECT_EQ(validated.Cohorts()[0].ledger_bank_charge.host, 2000u);
  EXPECT_EQ(validated.Cohorts()[0].ledger_bank_charge.device, 1400u);
  EXPECT_EQ(validated.Cohorts()[0].member_private_max.host, 400u);
  EXPECT_EQ(validated.Cohorts()[0].member_private_max.device, 300u);
  EXPECT_EQ(validated.Cohorts()[0].active_jobs, 2u);
  EXPECT_EQ(validated.Cohorts()[0].rolling_reserve.host, 700u);
  EXPECT_EQ(validated.Cohorts()[0].rolling_reserve.device, 500u);
  EXPECT_EQ(validated.Cohorts()[0].ledger_terms.host, 620u);
  EXPECT_EQ(validated.Cohorts()[0].ledger_terms.device, 600u);
  EXPECT_EQ(validated.Cohorts()[0].ledger.host, 3320u);
  EXPECT_EQ(validated.Cohorts()[0].ledger.device, 2500u);
  EXPECT_TRUE(validated.Cohorts()[0].ledger_fits_capacity);
}

TEST(ResidentCohortBoundary, UnfundedAdmittedConcurrencyRefusedHard) {
  // The MAX-fitting pair admits an unfunded declared concurrency: the
  // largest-2 SUM (600,600) makes the COMPLETE ledger (1800+500+600=2900
  // host; 800+300+600=1700 device) exceed BOTH cap sides of (2800,1400).
  // The hard gate refuses; there is no throttle, warning or degraded mode.
  const std::vector<size_t> selection{1, 2};
  const std::vector<ByteCharge> forecasts{{300, 300}, {300, 300}};
  CohortTerms terms;
  terms.process = {500, 300};
  terms.queue_transient = {0, 0};   // explicit zero queue allowance
  terms.unclassified = {0, 0};      // canonical: allowance-inclusive bank only
  terms.reviewed_bank = {0, 0};
  const ByteCharge capacity{2800, 1400};
  const auto estimator = [](size_t first, size_t count) -> ByteCharge {
    // Independent union semantics: singleton job1 (900,400), job2
    // (1000,350); the {1,2} union shares a layer: (1800,800).
    static const ByteCharge kSingleton[2] = {{900, 400}, {1000, 350}};
    if (count == 1) return kSingleton[first];
    if (first == 0 && count == 2) return {1800, 800};
    throw std::invalid_argument("test estimator: untabulated slice");
  };
  const auto plan = owned::PlanCohorts(selection, forecasts, terms, capacity,
                                       estimator);
  ASSERT_EQ(plan.cohorts.size(), 1u);
  // The unfunded concurrency is refused with the exact ledger literals
  // (canonical composition: 1800+500+600 = 2900 host; 800+300+600 = 1700
  // device; the former unclassified term is gone from the enforced
  // ledger).
  try {
    resident::ValidateCohortPlan(plan, selection, forecasts, terms, capacity, 2,
                                 estimator, TestIdentities(selection.size()));
    FAIL() << "an unfunded cohort ledger must refuse";
  } catch (const std::invalid_argument& error) {
    const std::string message = error.what();
    EXPECT_NE(message.find("COMPLETE ledger does not fit"), std::string::npos);
    EXPECT_NE(message.find("no throttling"), std::string::npos);
    EXPECT_NE(message.find("ledger host=2900"), std::string::npos);
    EXPECT_NE(message.find("device=1700"), std::string::npos);
  }
}

TEST(ResidentCohortBoundary, UnfundedConcurrencyRefusedPerSide) {
  // Both sides are independent: a host-only overfunding refuses (device
  // fits), and a device-only overfunding refuses (host fits).
  const std::vector<size_t> selection{1, 2};
  const std::vector<ByteCharge> forecasts{{300, 300}, {300, 300}};
  CohortTerms terms;
  terms.process = {500, 300};
  terms.queue_transient = {0, 0};
  terms.unclassified = {0, 0};  // canonical: allowance-inclusive bank only
  terms.reviewed_bank = {0, 0};
  const auto estimator = [](size_t first, size_t count) -> ByteCharge {
    static const ByteCharge kSingleton[2] = {{900, 400}, {1000, 350}};
    if (count == 1) return kSingleton[first];
    if (first == 0 && count == 2) return {1800, 800};
    throw std::invalid_argument("test estimator: untabulated slice");
  };
  const auto plan = owned::PlanCohorts(selection, forecasts, terms,
                                       ByteCharge{2800, 1400}, estimator);
  // Host-only failure: device cap raised, host cap unchanged.
  EXPECT_THROW(resident::ValidateCohortPlan(
                   plan, selection, forecasts, terms, ByteCharge{2800, 1800},
                   2, estimator, TestIdentities(2)),
               std::invalid_argument);
  // Device-only failure: host cap raised, device cap unchanged.
  EXPECT_THROW(resident::ValidateCohortPlan(
                   plan, selection, forecasts, terms, ByteCharge{3050, 1400},
                   2, estimator, TestIdentities(2)),
               std::invalid_argument);
  // EXACT-cap acceptance on both sides (equality fits).
  const resident::ValidatedCohortPlan exact_cap =
      resident::ValidateCohortPlan(plan, selection, forecasts, terms,
                                   ByteCharge{2900, 1700}, 2, estimator,
                                   TestIdentities(2));
  EXPECT_EQ(exact_cap.Cohorts()[0].ledger.host, 2900u);
  EXPECT_EQ(exact_cap.Cohorts()[0].ledger.device, 1700u);
  EXPECT_TRUE(exact_cap.Cohorts()[0].ledger_fits_capacity);
}

TEST(ResidentCohortBoundary, NonAdjacentLargestKReserveFitsExactCap) {
  // Out-of-order owner completion must never be assumed locally adjacent:
  // members (100,50),(1,50),(100,50) with concurrency 2 require the
  // per-side largest-2 SUM host = 100+100 = 200 (NON-adjacent members),
  // not a contiguous window peak (101). The complete ledger
  // (800+500+200=1500 host; 700+500+100=1300 device) fits BOTH capacities
  // exactly (1500,1300).
  const std::vector<size_t> selection{3, 4, 5};
  const std::vector<ByteCharge> forecasts{{100, 50}, {1, 50}, {100, 50}};
  CohortTerms terms;
  terms.process = {500, 500};
  terms.queue_transient = {0, 0};
  terms.unclassified = {0, 0};
  terms.reviewed_bank = {0, 0};
  const ByteCharge capacity{1500, 1300};
  const auto estimator = [](size_t first, size_t count) -> ByteCharge {
    // Independent union semantics: singleton banks (300,300),(350,300),
    // (300,300); pair union (3,4): (350,450); union (3..5) all three:
    // (800,700).
    static const ByteCharge kSingleton[3] = {{300, 300}, {350, 300},
                                             {300, 300}};
    if (count == 1) return kSingleton[first];
    if (first == 0 && count == 2) return {350, 450};
    if (first == 0 && count == 3) return {800, 700};
    throw std::invalid_argument("test estimator: untabulated slice");
  };
  const auto plan =
      owned::PlanCohorts(selection, forecasts, terms, capacity, estimator);
  ASSERT_EQ(plan.cohorts.size(), 1u);
  ASSERT_EQ(plan.cohorts[0].original_jobs.size(), 3u);

  const resident::ValidatedCohortPlan validated =
      resident::ValidateCohortPlan(plan, selection, forecasts, terms,
                                   capacity, /*max_concurrent_solvers=*/2,
                                   estimator, TestIdentities(3));
  EXPECT_EQ(validated.Cohorts()[0].active_jobs, 2u);
  EXPECT_EQ(validated.Cohorts()[0].rolling_reserve.host, 200u);
  EXPECT_EQ(validated.Cohorts()[0].rolling_reserve.device, 100u);
  EXPECT_EQ(validated.Cohorts()[0].ledger.host, 1500u);
  EXPECT_EQ(validated.Cohorts()[0].ledger.device, 1300u);
  EXPECT_TRUE(validated.Cohorts()[0].ledger_fits_capacity);
}

TEST(ResidentCohortBoundary, LargestKSumOverflowRefused) {
  // The largest-admitted-jobs SUM itself must overflow-check: members
  // (1e19,1e18),(9e18,1e18) with concurrency 2 require 1.9e19 host > the
  // uint64 range; the boundary refuses the required reserve arithmetically
  // (never a silent wrap), while every singleton and the MAX membership fit.
  const std::vector<size_t> selection{1, 2};
  using Huge = ByteCharge;
  const std::vector<ByteCharge> forecasts{
      Huge{10000000000000000000ULL, 1000000000000000000ULL},
      Huge{9000000000000000000ULL, 1000000000000000000ULL}};
  CohortTerms terms;
  terms.process = {1, 1};
  terms.queue_transient = {0, 0};
  terms.unclassified = {0, 0};
  terms.reviewed_bank = {0, 0};
  const ByteCharge capacity{12000000000000000000ULL, 12000000000000000000ULL};
  const auto estimator = [](size_t, size_t count) -> ByteCharge {
    return count == 1 ? ByteCharge{1, 1} : ByteCharge{2, 2};
  };
  const auto plan =
      owned::PlanCohorts(selection, forecasts, terms, capacity, estimator);
  ASSERT_EQ(plan.cohorts.size(), 1u);
  EXPECT_THROW(resident::ValidateCohortPlan(plan, selection, forecasts, terms,
                                            capacity, 2, estimator,
                                            TestIdentities(2)),
               std::overflow_error);
  // Device-only overflow: same shape, both privates carry huge DEVICE sides.
  const std::vector<ByteCharge> device_huge{
      {1000000000000000000ULL, 10000000000000000000ULL},
      {1000000000000000000ULL, 9000000000000000000ULL}};
  const auto device_plan =
      owned::PlanCohorts(selection, device_huge, terms, capacity, estimator);
  ASSERT_EQ(device_plan.cohorts.size(), 1u);
  EXPECT_THROW(resident::ValidateCohortPlan(device_plan, selection,
                                            device_huge, terms, capacity, 2,
                                            estimator, TestIdentities(2)),
               std::overflow_error);
}

TEST(ResidentCohortBoundary, LargestKSumIgnoresIrrelevantOverflowingMembers) {
  // With concurrency 1, the REQUIRED largest-1 SUM (5e18) fits and must
  // validate even though summing the WHOLE member list would overflow
  // (4 x 5e18 = 2e19): irrelevant members' overflow can never veto a plan.
  const std::vector<size_t> selection{1, 2, 3, 4};
  const std::vector<ByteCharge> forecasts(
      4, ByteCharge{5000000000000000000ULL, 1000000000000000000ULL});
  CohortTerms terms;
  terms.process = {1, 1};
  terms.queue_transient = {0, 0};
  terms.unclassified = {0, 0};
  terms.reviewed_bank = {0, 0};
  const ByteCharge capacity{12000000000000000000ULL, 12000000000000000000ULL};
  const auto estimator = [](size_t, size_t count) -> ByteCharge {
    return ByteCharge{100 * count, 100 * count};
  };
  const auto plan =
      owned::PlanCohorts(selection, forecasts, terms, capacity, estimator);
  ASSERT_EQ(plan.cohorts.size(), 1u);
  ASSERT_EQ(plan.cohorts[0].original_jobs.size(), 4u);
  const resident::ValidatedCohortPlan validated =
      resident::ValidateCohortPlan(plan, selection, forecasts, terms,
                                   capacity, /*max_concurrent_solvers=*/1,
                                   estimator, TestIdentities(4));
  EXPECT_EQ(validated.Cohorts()[0].active_jobs, 1u);
  EXPECT_EQ(validated.Cohorts()[0].rolling_reserve.host,
            5000000000000000000ULL);
  EXPECT_EQ(validated.Cohorts()[0].ledger.host,
            400 + 1 + 5000000000000000000ULL);
  EXPECT_TRUE(validated.Cohorts()[0].ledger_fits_capacity);
}

TEST(ResidentCohortBoundary, LargerConcurrencyThanCohortSumsFullSlice) {
  // Declared concurrency above a cohort's member count: active_jobs is
  // capped by the member count (min(k, count)); cohort 0 {7,11} keeps its
  // largest-2 SUM (550,350), cohort 1 stays the singleton (400,300).
  const auto selection = TestSelection();
  const auto plan = owned::PlanCohorts(selection, TestForecasts(), TestTerms(),
                                       TestCapacityScalar(), TestEstimator());
  ASSERT_EQ(plan.cohorts.size(), 2u);
  const resident::ValidatedCohortPlan validated =
      resident::ValidateCohortPlan(plan, selection, TestForecasts(),
                                   TestTerms(), TestCapacityScalar(),
                                   /*max_concurrent_solvers=*/5,
                                   TestEstimator(),
                                   TestIdentities(selection.size()));
  EXPECT_EQ(validated.MaxConcurrentSolvers(), 5u);
  EXPECT_EQ(validated.Cohorts()[0].active_jobs, 2u);
  EXPECT_EQ(validated.Cohorts()[0].rolling_reserve.host, 550u);
  EXPECT_EQ(validated.Cohorts()[0].rolling_reserve.device, 350u);
  EXPECT_EQ(validated.Cohorts()[1].active_jobs, 1u);
  EXPECT_EQ(validated.Cohorts()[1].rolling_reserve.host, 400u);
  EXPECT_EQ(validated.Cohorts()[1].rolling_reserve.device, 300u);
}

TEST(ResidentCohortBoundary, NonCanonicalFixedTermsRefuseExplicitly) {
  // CANONICAL COMPOSITION enforcement (Sol 2255):
  //  - a separate unclassified ledger term double-bills when the bank
  //    charge is allowance-inclusive (or misses the allowance margin when
  //    it is not): NONZERO terms.unclassified refuses explicitly, never
  //    silently dropped and never silently folded twice;
  //  - the device ledger is bank + process + rolling; a nonzero DEVICE
  //    queue/unclassified fixed term is unsupported and REJECTS, never
  //    silently dropped;
  //  - a ONE-SIDED reviewed reservation is wire-valid but cannot be
  //    expressed by the frozen all-or-neither planner policy: the
  //    planner-based boundary refuses EXPLICITLY with the adapter
  //    obligation named (never silently normalized);
  //  - the wire-valid BOTH-SIDES reviewed reservation still validates.
  const auto selection = TestSelection();
  const auto capacity = TestCapacityScalar();

  // Submit the CANONICAL planner record with the non-canonical terms: the
  // explicit composition checks must fire BEFORE provenance/planner work
  // (the frozen all-or-neither planner itself cannot even build a record
  // under a one-sided reservation).
  const auto canonical_plan =
      owned::PlanCohorts(selection, TestForecasts(), TestTerms(), capacity,
                         TestEstimator());
  auto assert_refused = [&](CohortTerms terms, const std::string& needle) {
    try {
      resident::ValidateCohortPlan(
          canonical_plan, selection, TestForecasts(), terms, capacity, 2,
          TestEstimator(), TestIdentities(3));
      FAIL() << "must refuse: " << needle;
    } catch (const std::invalid_argument& error) {
      EXPECT_NE(std::string(error.what()).find(needle), std::string::npos)
          << error.what();
    } catch (...) {
      FAIL() << "expected the exact canonical-composition refusal";
    }
  };

  CohortTerms extra_unclassified = TestTerms();
  extra_unclassified.unclassified = {30, 0};
  assert_refused(extra_unclassified, "allowance-inclusive");

  CohortTerms device_queue = TestTerms();
  device_queue.queue_transient = {120, 40};
  assert_refused(device_queue, "device queue/unclassified fixed term");

  CohortTerms one_sided = TestTerms();
  one_sided.reviewed_bank = {2000, 0};
  assert_refused(one_sided, "one-sided reviewed bank reservation");

  // Both-sided reviewed reservation still validates against a capacity
  // its policy can carry (see ReviewedReservationPolicyLedgerEvidence);
  // per-side canonical allowance translation remains the flagged adapter
  // obligation.
  const ByteCharge reviewed_capacity{3600, 2600};
  CohortTerms reviewed = TestTerms();
  reviewed.reviewed_bank = {2000, 1400};
  resident::ValidateCohortPlan(
      owned::PlanCohorts(selection, TestForecasts(), reviewed,
                         reviewed_capacity, TestEstimator()),
      selection, TestForecasts(), reviewed, reviewed_capacity, 2,
      TestEstimator(), TestIdentities(3));
}

// Refusal harness: every case below submits a plan record (or input) that
// the REAL planner did not produce from the declared inputs. The provenance
// check refuses all of them without ever indexing selection through
// plan-provided ordinals.
class ResidentCohortBoundaryRefusal : public ::testing::Test {
 protected:
  void SetUp() override {
    const auto selection = TestSelection();
    plan_ = owned::PlanCohorts(selection, TestForecasts(), TestTerms(),
                               TestCapacityScalar(), TestEstimator());
  }
  void ExpectRefused(const owned::CohortPlan& tampered) const {
    EXPECT_THROW(
        resident::ValidateCohortPlan(tampered, TestSelection(),
                                     TestForecasts(), TestTerms(),
                                     TestCapacityScalar(), 2, TestEstimator(),
                                     TestIdentities(TestSelection().size())),
        std::invalid_argument)
        << "a plan the real planner did not produce must refuse, exactly";
  }
  owned::CohortPlan plan_;
};

TEST_F(ResidentCohortBoundaryRefusal, SortedOrForeignMembershipRefused) {
  auto tampered = plan_;
  tampered.cohorts[0].original_jobs = {11, 7};  // sorted, not FIFO identity
  ExpectRefused(tampered);
}

TEST_F(ResidentCohortBoundaryRefusal, DuplicatedMembershipRefused) {
  auto tampered = plan_;
  tampered.cohorts[1].original_jobs = {7, 15};
  ExpectRefused(tampered);
}

TEST_F(ResidentCohortBoundaryRefusal, OverlongEntryRefusedWithoutOOB) {
  // A hand-made overlong cohort (count beyond the selection) must refuse;
  // the boundary never indexes the selection through plan-provided
  // ordinals, so a malformed record cannot cause out-of-bounds access.
  auto tampered = plan_;
  tampered.cohorts[0].count = 99;
  tampered.cohorts[0].original_jobs.assign(99, 7);
  ExpectRefused(tampered);
}

TEST_F(ResidentCohortBoundaryRefusal, MissingCohortRefused) {
  auto tampered = plan_;
  tampered.cohorts.pop_back();
  ExpectRefused(tampered);
}

TEST_F(ResidentCohortBoundaryRefusal, GapInCoverageRefused) {
  auto tampered = plan_;
  tampered.cohorts[1].first = 1;
  ExpectRefused(tampered);
}

TEST_F(ResidentCohortBoundaryRefusal, UnderstatedMemberMaxRefused) {
  auto tampered = plan_;
  tampered.cohorts[0].member_private = {250, 150};
  ExpectRefused(tampered);
}

TEST_F(ResidentCohortBoundaryRefusal, OverstatedMemberMaxRefused) {
  auto tampered = plan_;
  tampered.cohorts[0].member_private = {350, 250};
  ExpectRefused(tampered);
}

TEST_F(ResidentCohortBoundaryRefusal, TermsTotalMismatchRefused) {
  auto tampered = plan_;
  tampered.cohorts[0].terms_total.host = 9000;
  ExpectRefused(tampered);
}

TEST_F(ResidentCohortBoundaryRefusal, RequiredMismatchRefused) {
  auto tampered = plan_;
  tampered.cohorts[1].required.device = 999;
  ExpectRefused(tampered);
}

TEST_F(ResidentCohortBoundaryRefusal, BankAboveReviewedReservationRefused) {
  // Under the reviewed reservation policy, a record whose bank estimate
  // exceeds the explicit reservation cannot come from a real planning run;
  // a fabricated or corrupted record refuses at the boundary.
  auto tampered = plan_;
  tampered.cohorts[0].bank_estimate = {1300, 420};
  CohortTerms terms = TestTerms();
  terms.reviewed_bank = {1000, 400};
  EXPECT_THROW(resident::ValidateCohortPlan(
                   tampered, TestSelection(), TestForecasts(), terms,
                   TestCapacityScalar(), 2, TestEstimator(),
                   TestIdentities(TestSelection().size())),
               std::invalid_argument);
}

TEST_F(ResidentCohortBoundaryRefusal, PlanFromDifferentInputsRefused) {
  // A plan the real planner produced but FROM DIFFERENT INPUTS (a shrunk
  // capacity here) is stale for this validation call and must refuse: no
  // fabricated preflight provenance. The shrunk budget fails inside the
  // real planner's own preflight first (runtime_error); the boundary never
  // reaches an equality-confirmed acceptance for stale records.
  EXPECT_THROW(resident::ValidateCohortPlan(
                   plan_, TestSelection(), TestForecasts(), TestTerms(),
                   ByteCharge{2000, 1000}, 2, TestEstimator(),
                   TestIdentities(3)),
               std::runtime_error);
  // A record produced under ANY other planning inputs but structurally
  // valid under the current preflight still refuses via provenance
  // mismatch (invalid_argument): the submitted record must be byte-equal
  // to the fresh planner output.
  owned::CohortPlan stale_but_valid = plan_;
  stale_but_valid.cohorts[0].bank_estimate = plan_.cohorts[0].bank_estimate;
  stale_but_valid.cohorts.resize(1);
  stale_but_valid.cohorts[0] = plan_.cohorts[1];
  stale_but_valid.cohorts.push_back(plan_.cohorts[0]);
  EXPECT_THROW(resident::ValidateCohortPlan(
                   stale_but_valid, TestSelection(), TestForecasts(),
                   TestTerms(), TestCapacityScalar(), 2, TestEstimator(),
                   TestIdentities(3)),
               std::invalid_argument);
}

TEST_F(ResidentCohortBoundaryRefusal, MissingEstimatorRefused) {
  EXPECT_THROW(resident::ValidateCohortPlan(
                   plan_, TestSelection(), TestForecasts(), TestTerms(),
                   TestCapacityScalar(), 2, /*bank_estimator=*/{},
                   TestIdentities(3)),
               std::invalid_argument);
}

TEST(ResidentCohortBoundary, IncompleteExplicitInputsRefused) {
  const auto selection = TestSelection();
  const auto forecast = TestForecasts();
  const auto terms = TestTerms();
  const auto capacity = TestCapacityScalar();
  const auto identities = TestIdentities(selection.size());
  const auto plan = owned::PlanCohorts(selection, forecast, terms, capacity,
                                       TestEstimator());
  ASSERT_EQ(plan.cohorts.size(), 2u);

  // Zero private forecast sides.
  auto zero_host = forecast;
  zero_host[1] = {0, 150};
  EXPECT_THROW(resident::ValidateCohortPlan(plan, selection, zero_host, terms,
                                            capacity, 2, TestEstimator(),
                                            identities),
               std::invalid_argument);
  auto zero_device = forecast;
  zero_device[2] = {400, 0};
  EXPECT_THROW(resident::ValidateCohortPlan(plan, selection, zero_device,
                                            terms, capacity, 2,
                                            TestEstimator(), identities),
               std::invalid_argument);

  // Forecast count mismatch.
  auto short_forecasts = forecast;
  short_forecasts.pop_back();
  EXPECT_THROW(resident::ValidateCohortPlan(plan, selection, short_forecasts,
                                            terms, capacity, 2,
                                            TestEstimator(), identities),
               std::invalid_argument);

  // Non-strict FIFO selection identity order is refused (planner and
  // boundary alike: the identity is the immutable FIFO order).
  auto reordered = selection;
  std::swap(reordered[0], reordered[1]);
  EXPECT_THROW(owned::PlanCohorts(reordered, forecast, terms, capacity,
                                  TestEstimator()),
               std::invalid_argument);

  // Missing or count-mismatched ORIGINAL identity snapshot.
  EXPECT_THROW(resident::ValidateCohortPlan(plan, selection, forecast, terms,
                                            capacity, 2, TestEstimator(), {}),
               std::invalid_argument);
  EXPECT_THROW(resident::ValidateCohortPlan(plan, selection, forecast, terms,
                                            capacity, 2, TestEstimator(),
                                            TestIdentities(2)),
               std::invalid_argument);

  // Every problem's identity record set must be nonempty (at least the
  // reference input identity).
  auto empty_problem = TestIdentities(selection.size());
  empty_problem[2].clear();
  EXPECT_THROW(resident::ValidateCohortPlan(plan, selection, forecast, terms,
                                            capacity, 2, TestEstimator(),
                                            empty_problem),
               std::invalid_argument);

  // Zero capacity side.
  EXPECT_THROW(resident::ValidateCohortPlan(plan, selection, forecast, terms,
                                            ByteCharge{2300, 0}, 2,
                                            TestEstimator(), identities),
               std::invalid_argument);

  // Zero process side.
  CohortTerms zero_process = terms;
  zero_process.process = {500, 0};
  EXPECT_THROW(resident::ValidateCohortPlan(plan, selection, forecast,
                                            zero_process, capacity, 2,
                                            TestEstimator(), identities),
               std::invalid_argument);

  // Zero declared concurrency.
  EXPECT_THROW(resident::ValidateCohortPlan(plan, selection, forecast, terms,
                                            capacity, 0, TestEstimator(),
                                            identities),
               std::invalid_argument);
}

}  // namespace
}  // namespace mvs
}  // namespace colmap