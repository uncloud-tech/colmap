// SPDX-License-Identifier: BSD-3-Clause
#pragma once

// Typed INTERNAL validated-plan boundary for resident FIFO cohorts
// (next-cohort milestone). This is a structural and arithmetic VALIDATOR
// plus typed read-only plan container ONLY:
//
//   * It re-runs the REAL owned FIFO cohort planner (owned_cohort.h
//     ::PlanCohorts) on the ACTUAL planning inputs the caller supplies, and
//     requires the submitted plan to be EXACTLY the plan that planner
//     produced. This is the preflight-provenance contract: an arbitrary
//     hand-made record cannot substitute for the real singleton/complete-
//     plan preflight, and no provenance is faked here (no origin markers,
//     no filesystem recapture, no second estimator).
//
//   * THE RESULT IS EXPLICITLY NOT EXECUTION-AUTHORIZED. The type carries
//     the fact as a constant: no conversion into an execution-authorized
//     form exists at this boundary. Before any payload, a FUTURE caller
//     must run the real per-cohort schema/binding/build-identity/minima
//     gates specific to each cohort (the accepted full-bank resident
//     forecast wire cannot certify cohort plans: its full-bank aggregate
//     caps and grouping terms do not bind per-cohort slices) and only then
//     convert this plan into its own execution-authorized form. No public
//     unguarded executable path is exposed by this header. The global
//     snapshot/metadata, bounded normal owner/future/transient minima and
//     the whole-invocation binding/identity gates are FUTURE INTEGRATION
//     OBLIGATIONS of that caller: they are required inputs there and are
//     NOT proven by the generic charge inputs this boundary validates.
//
//   * Planner MAX membership and rolling private SUM stay DISTINCT, with
//     the MANDATORY admitted-concurrency contract (no informational mode,
//     no dynamic throttle): the per-cohort rolling reserve is the SUM of the
//     largest min(declared_concurrency, member_count) member private
//     charges, INDEPENDENTLY PER SIDE (out-of-order owner completion may
//     make the concurrently-resident owners non-adjacent, so a contiguous
//     window is NOT the reserve), and the per-cohort COMPLETE ledger must
//     fit the CLI caps on both sides as a HARD GATE. Cohorts are STRICTLY
//     SEQUENTIAL (each cohort's bank and owners retire before the next
//     starts), so ledgers are validated individually and are NEVER
//     aggregated across cohorts.
//
//   * Invocation-identity custody: the caller must pass the AUTHORITATIVE
//     ORIGINAL resolved invocation identity snapshot (one nonempty record
//     set per selected problem, in native problem order), captured once
//     before any planning read. The boundary RETAINS AN OWNED IMMUTABLE
//     COPY of exactly those records in the returned object (no filesystem
//     access, no recapture) so the validated plan substantiates its own
//     identity claim; fresh per-file revalidation around estimator
//     traversals, per-cohort bank construction and every later read remains
//     the duty of the caller/executor surfaces that already own that
//     machinery. Caller-side mutations after validation never affect the
//     retained copy.
//
//   * Mapping preservation: global model image indices live inside the per-
//     job resolved problems (BankInput::image_idx) and inside the retained
//     identity records; the original bank job id order is carried through
//     unchanged: no sorting, no recapture, no renumbering. Cohort-local
//     ordinals and per-cohort host-bank layer ordinals remain a strictly
//     executor-level mapping (HostInputBank::ReferenceLayer / SourceLayers
//     keyed by the cohort-local ordinal); this boundary preserves the
//     original bank job ids so no global image index and no bank job/layer
//     ordinal can be silently swapped. Cohorts are never summed, re-sorted
//     or reordered.

#include "colmap/mvs/owned_cohort.h"
#include "colmap/mvs/owned_input_bank.h"
#include "colmap/mvs/owned_pair_validation.h"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace colmap {
namespace mvs {
namespace resident {

using owned::ByteCharge;

// One validated cohort: the verified planner entry plus the DISTINCT
// admitted-concurrency arithmetic (per-side largest-k private SUM) and the
// HARD per-cohort complete-ledger capacity gate.
struct ValidatedCohortEntry {
  // Contiguous FIFO slice in selection-ordinal space, exactly as planned.
  size_t first = 0;
  size_t count = 0;
  // Original resolved bank job ids in FIFO order (identity preserved; the
  // global-image-index to bank-layer mapping stays executor-level).
  std::vector<size_t> original_jobs;
  // Native metadata bank estimate for the slice from the plan. Its role in
  // the admission policy is re-verified below; the estimator OUTPUT itself
  // is not re-derived at this metadata boundary (the planner runs with the
  // same estimator, so the estimate is the real planner's output), and the
  // per-cohort schema/binding/minima gates must re-qualify it later.
  ByteCharge bank_estimate;
  // MAX membership evidence (the planner's packing arithmetic): per-side
  // maximum of the member private forecasts, re-derived independently.
  ByteCharge member_private_max;
  // Explicit fixed terms as planned/re-verified, CANONICAL COMPOSITION:
  // host = process + queue/host (queue_transient.host carries the canonical
  // queue plus the bounded normal pipeline charge, with the whole-invocation
  // snapshot/plan basis already inside the queue); device = process.device.
  // The bank charge is allowance-INCLUSIVE: the explicit out-of-formula
  // allowance is composed with the margin-free raw BEFORE the single
  // canonical 3% margin exactly once inside the bank charge; there is NO
  // second unclassified ledger term (double-billing refuses, see below).
  ByteCharge ledger_terms;
  // MANDATORY admitted concurrency: active_jobs >= 1 and <= member_count.
  // The declared run-level concurrency is capped per cohort by its member
  // count; a cohort wire would declare this per-cohort value.
  size_t active_jobs = 0;
  // The per-cohort rolling reserve: SUM of the largest `active_jobs` member
  // private charges, INDEPENDENTLY PER SIDE (non-adjacent owners admitted;
  // contiguous locality is never assumed), computed with checked additions.
  // DISTINCT from member_private_max by construction.
  ByteCharge rolling_reserve;
  // The bank charge dedicated under the plan policy: the explicit reviewed
  // reservation when configured, else the plan's bank estimate (legacy
  // estimate-only packing contract).
  ByteCharge ledger_bank_charge;
  // The cohort's COMPLETE ledger (the admission unit; strictly sequential
  // cohorts are validated individually, never summed; canonical
  // composition matching the frozen cohort wire):
  //   host   = bank_charge + process + queue (incl. normal charge and the
  //            whole-invocation basis) + rolling_reserve
  //   device = bank_charge + process + rolling_reserve
  ByteCharge ledger;
  // HARD GATE record: `ledger` must fit the total CLI capacity on BOTH
  // sides (an unfunded cohort is refused, never throttled or warned).
  bool ledger_fits_capacity = false;
};

// Fully validated typed read-only cohort plan. All members are private; the
// validated values cannot be mutated through public accessors, and public
// construction is denied: obtain instances only through ValidateCohortPlan.
class ValidatedCohortPlan {
 public:
  // The boundary NEVER authorizes execution: the future per-cohort gates
  // (wire schema/binding for each cohort, build-identity compatibility,
  // per-cohort bank estimates, private/normal minima and caps) are owed
  // explicitly by the integrating caller, which also owns any wire
  // extension. No conversion to an execution-authorized type exists here.
  static constexpr bool kExecutionAuthorized = false;

  // Explicit original bank job id -> position mapping element, in SELECTION
  // order (no sorting anywhere).
  struct MappingEntry {
    size_t original_job = 0;
    size_t selection_ordinal = 0;
    size_t cohort = 0;
    size_t cohort_local_ordinal = 0;
  };

  // Always false, by type. Present so code cannot accidentally confuse this
  // boundary record with an executable authorization.
  bool ExecutionAuthorized() const { return kExecutionAuthorized; }

  size_t JobCount() const { return job_count_; }
  size_t MaxConcurrentSolvers() const { return max_concurrent_solvers_; }
  ByteCharge Capacity() const { return capacity_; }
  const owned::CohortTerms& Terms() const { return terms_; }
  const std::vector<ValidatedCohortEntry>& Cohorts() const { return cohorts_; }
  // Explicit original bank job id -> position mapping, in SELECTION order
  // (no sorting anywhere).
  const std::vector<MappingEntry>& Mapping() const { return mapping_; }
  // The OWNED IMMUTABLE COPY of the authoritative ORIGINAL invocation
  // identity snapshot: one nonempty record set per selected problem, in
  // problem order. Pure in-memory retention of the caller's original
  // records; no filesystem access and no recapture belong here.
  const std::vector<owned::BankJobIdentities>& InvocationIdentities() const {
    return invocation_identities_;
  }

 private:
  friend ValidatedCohortPlan ValidateCohortPlan(
      const owned::CohortPlan&,
      const std::vector<size_t>&,
      const std::vector<ByteCharge>&,
      const owned::CohortTerms&,
      ByteCharge,
      size_t,
      const owned::CohortBankEstimator&,
      const std::vector<owned::BankJobIdentities>&);
  ValidatedCohortPlan() = default;

  size_t job_count_ = 0;
  size_t max_concurrent_solvers_ = 0;
  ByteCharge capacity_;
  owned::CohortTerms terms_;
  std::vector<ValidatedCohortEntry> cohorts_;
  std::vector<MappingEntry> mapping_;
  std::vector<owned::BankJobIdentities> invocation_identities_;
};

namespace detail {

inline void CohortBoundaryRequire(bool ok, const std::string& message) {
  if (!ok) {
    throw std::invalid_argument("cohort boundary: " + message);
  }
}

// Per-side SUMs of the largest `k` charges, INDEPENDENTLY for host and
// device (owners completing out of order may be non-adjacent; nothing may
// assume contiguous locality). Only the selected largest k values are added
// (checked additions), so an overflow in the irrelevant remainder of the
// member list can never occur while the required reserve itself fits.
inline ByteCharge LargestKSum(const std::vector<ByteCharge>& values,
                              size_t k,
                              const char* const context) {
  CohortBoundaryRequire(!values.empty(),
                        std::string(context) + ": empty member list");
  CohortBoundaryRequire(k >= 1, std::string(context) + ": zero sum window");
  if (k > values.size()) {
    k = values.size();
  }
  std::vector<uint64_t> host;
  std::vector<uint64_t> device;
  host.reserve(values.size());
  device.reserve(values.size());
  for (const ByteCharge& value : values) {
    host.push_back(value.host);
    device.push_back(value.device);
  }
  ByteCharge sum{0, 0};
  if (k == values.size()) {
    for (const uint64_t v : host) {
      sum.host = owned::CheckedAdd(sum.host, v);
    }
    for (const uint64_t v : device) {
      sum.device = owned::CheckedAdd(sum.device, v);
    }
    return sum;
  }
  std::partial_sort(host.begin(),
                    host.begin() + static_cast<long>(k),
                    host.end(),
                    std::greater<uint64_t>());
  std::partial_sort(device.begin(),
                    device.begin() + static_cast<long>(k),
                    device.end(),
                    std::greater<uint64_t>());
  for (size_t j = 0; j < k; ++j) {
    sum.host = owned::CheckedAdd(sum.host, host[j]);
    sum.device = owned::CheckedAdd(sum.device, device[j]);
  }
  return sum;
}

// Exact structural equality of two planner records (provenance check).
inline bool PlannerPlansIdentical(const owned::CohortPlan& expected,
                                  const owned::CohortPlan& submitted) {
  if (expected.job_count != submitted.job_count ||
      expected.cohorts.size() != submitted.cohorts.size()) {
    return false;
  }
  for (size_t c = 0; c < expected.cohorts.size(); ++c) {
    const auto& a = expected.cohorts[c];
    const auto& b = submitted.cohorts[c];
    if (a.first != b.first || a.count != b.count ||
        a.original_jobs != b.original_jobs ||
        a.bank_estimate.host != b.bank_estimate.host ||
        a.bank_estimate.device != b.bank_estimate.device ||
        a.member_private.host != b.member_private.host ||
        a.member_private.device != b.member_private.device ||
        a.terms_total.host != b.terms_total.host ||
        a.terms_total.device != b.terms_total.device ||
        a.required.host != b.required.host ||
        a.required.device != b.required.device) {
      return false;
    }
  }
  return true;
}

}  // namespace detail

// Typed boundary validation. See the header contract above. Summary:
//   inputs (all fail closed): nonempty strictly-increasing selection; one
//   nonzero-both-sides complete private forecast per selected job; nonempty
//   ORIGINAL identity snapshot with exactly one NONEMPTY record set per
//   selected problem in problem order; nonzero-both-sides capacity;
//   nonzero-both-sides process charge; declared concurrency >= 1.
//   queue_transient/unclassified/reviewed_bank must be passed EXPLICITLY
//   (no implicit zero allowances). CANONICAL COMPOSITION: terms.unclassified
//   must be explicitly {0,0} (the allowance is folded into the allowance-
//   inclusive bank charge before the single canonical margin, exactly once;
//   a second unclassified ledger term double-bills and refuses);
//   terms.queue_transient.device must be 0 (the device ledger is bank +
//   process + rolling; unsupported device fixed terms REJECT, never
//   silently drop); terms.queue_transient.host = queue + normal charge;
//   reviewed_bank 0/0 = the legacy estimate-only packing contract. The bank
//   estimator is REQUIRED (real planner provenance).
//   provenance: the REAL planner runs on these inputs first; its singleton
//   and complete-plan preflight (and admission arithmetic) act as the only
//   accepted preflight evidence, and the submitted plan record must equal
//   the freshly planned result EXACTLY.
//   arithmetic: per cohort, the planner MAX membership is re-derived, the
//   admitted-concurrency reserve is the per-side largest-k SUM, and the
//   cohort's COMPLETE ledger is compared against the TOTAL capacity as a
//   hard gate on both sides — sequentially independent cohorts, never
//   summed across cohorts.
inline ValidatedCohortPlan ValidateCohortPlan(
    const owned::CohortPlan& plan,
    const std::vector<size_t>& selection,
    const std::vector<ByteCharge>& forecasts,
    const owned::CohortTerms& terms,
    const ByteCharge capacity,
    const size_t max_concurrent_solvers,
    const owned::CohortBankEstimator& bank_estimator,
    const std::vector<owned::BankJobIdentities>& invocation_identities) {
  detail::CohortBoundaryRequire(!selection.empty(),
                                "validation requires a nonempty selection");
  detail::CohortBoundaryRequire(forecasts.size() == selection.size(),
                                "forecast count " +
                                    std::to_string(forecasts.size()) +
                                    " does not match the selection count " +
                                    std::to_string(selection.size()));
  for (size_t k = 0; k < selection.size(); ++k) {
    detail::CohortBoundaryRequire(forecasts[k].host && forecasts[k].device,
                                  "zero private forecast (bank_job_index=" +
                                      std::to_string(selection[k]) + ")");
    if (k) {
      detail::CohortBoundaryRequire(
          selection[k] > selection[k - 1],
          "selection is not a strictly ordered FIFO identity (fail at "
          "selection_ordinal=" +
              std::to_string(k) + ")");
    }
  }
  detail::CohortBoundaryRequire(
      !invocation_identities.empty(),
      "the ORIGINAL resolved invocation identity snapshot is REQUIRED");
  detail::CohortBoundaryRequire(
      invocation_identities.size() == selection.size(),
      "identity snapshot count " +
          std::to_string(invocation_identities.size()) +
          " does not match the selection count " +
          std::to_string(selection.size()) +
          " (the snapshot must be the original captured invocation record "
          "set, one per selected problem, in problem order)");
  for (size_t k = 0; k < invocation_identities.size(); ++k) {
    detail::CohortBoundaryRequire(
        !invocation_identities[k].empty(),
        "the identity record set of problem " + std::to_string(k) +
            " is empty (every admitted job carries at least its reference "
            "input identity)");
  }
  detail::CohortBoundaryRequire(capacity.host && capacity.device,
                                "zero budget capacity");
  detail::CohortBoundaryRequire(terms.process.host && terms.process.device,
                                "zero process charge (both sides required; "
                                "no implicit zero allowances)");
  // CANONICAL ALLOWANCE COMPOSITION (per the frozen cohort wire): the
  // explicit out-of-formula allowance is composed with the margin-free raw
  // bank BEFORE the single canonical margin, exactly once, INSIDE the bank
  // charge. A separate unclassified ledger term would double-bill (or miss
  // the allowance margin) and is REFUSED, never silently dropped or
  // invented.
  detail::CohortBoundaryRequire(
      !terms.unclassified.host && !terms.unclassified.device,
      "canonical composition: the bank charge is allowance-inclusive "
      "(margin(raw + allowance) once); a separate unclassified ledger term "
      "double-bills and refuses — pass terms.unclassified = {0,0} "
      "explicitly and fold the allowance into the reviewed bank "
      "before this call (parent/Photogram ruling on record)");
  // Device fixed terms are unsupported by the enforced device ledger
  // composition (device = bank_charge + process + rolling_reserve); a
  // nonzero device queue/unclassified input is REJECTED, not silently
  // dropped.
  detail::CohortBoundaryRequire(
      !terms.queue_transient.device,
      "canonical composition: the device ledger is bank_charge + process + "
      "rolling_reserve; a nonzero device queue/unclassified fixed term is "
      "unsupported and refuses (never silently dropped)");
  // ONE-SIDED reviewed reservation: the frozen FIFO planner policy is
  // all-or-neither across both sides. A wire-valid one-sided reviewed R is
  // only consumed by the wire-level executor gates; this planner-based
  // boundary refuses it EXPLICITLY (never silently) until the per-side
  // adapter is agreed with parent/Photogram.
  detail::CohortBoundaryRequire(
      !(terms.reviewed_bank.host && !terms.reviewed_bank.device) &&
          !(!terms.reviewed_bank.host && terms.reviewed_bank.device),
      "one-sided reviewed bank reservation refuses at this planner-based "
      "boundary (the frozen FIFO planner policy is all-or-neither; the "
      "wire-valid per-side R is consumed by the wire-level executor gates "
      "instead; per-side adapter is a flagged parent/Photogram obligation, "
      "never silently normalized)");
  detail::CohortBoundaryRequire(max_concurrent_solvers >= 1,
                                "declared rolling concurrency must be >= 1");
  detail::CohortBoundaryRequire(
      static_cast<bool>(bank_estimator),
      "the real bank estimator is REQUIRED (planner preflight provenance)");

  // PROVENANCE: the real planner runs on the actual planning inputs. Its
  // singleton/complete-plan preflight, packing and admission arithmetic are
  // the only accepted preflight evidence; nothing invented here.
  const owned::CohortPlan planned =
      owned::PlanCohorts(selection, forecasts, terms, capacity, bank_estimator);
  detail::CohortBoundaryRequire(
      detail::PlannerPlansIdentical(planned, plan),
      "the submitted plan is not exactly the plan the real FIFO cohort "
      "planner produced from these inputs (hand-made, corrupted or stale "
      "records refuse; preflight provenance is never faked)");
  // Defensive bounds: every planned cohort must fit the remaining selection
  // before any ordinal-derived indexing elsewhere ever runs (bounds first).
  size_t covered = 0;
  for (const auto& entry : planned.cohorts) {
    detail::CohortBoundaryRequire(
        entry.first == covered,
        "cohort does not start at the dense FIFO coverage boundary "
        "(selection_ordinal=" +
            std::to_string(covered) + ")");
    detail::CohortBoundaryRequire(
        entry.count >= 1 && entry.count <= selection.size() - covered,
        "cohort count violates the remaining selection bounds "
        "(covered=" +
            std::to_string(covered) + ", count=" + std::to_string(entry.count) +
            ") — overlong entries refuse");
    covered += entry.count;
  }

  // Per-cohort arithmetic, from the REAL planner's record.
  ValidatedCohortPlan validated;
  validated.job_count_ = selection.size();
  validated.max_concurrent_solvers_ = max_concurrent_solvers;
  validated.capacity_ = capacity;
  validated.terms_ = terms;
  validated.invocation_identities_ = invocation_identities;
  validated.mapping_.reserve(selection.size());
  for (size_t c = 0; c < planned.cohorts.size(); ++c) {
    const auto& entry = planned.cohorts[c];
    const size_t first = entry.first;
    const size_t count = entry.count;
    std::vector<ByteCharge> member_forecasts;
    member_forecasts.reserve(count);
    ByteCharge member_max{0, 0};
    for (size_t k = 0; k < count; ++k) {
      const ByteCharge forecast = forecasts[first + k];
      member_forecasts.push_back(forecast);
      if (forecast.host > member_max.host) member_max.host = forecast.host;
      if (forecast.device > member_max.device) {
        member_max.device = forecast.device;
      }
    }
    detail::CohortBoundaryRequire(
        member_max.host == entry.member_private.host &&
            member_max.device == entry.member_private.device,
        "planner member_private does not equal the per-side MAX of its "
        "member private forecasts (planner invariant violation)");

    const ByteCharge bank_charge =
        owned::detail::CohortBankCharge(terms, entry.bank_estimate);
    if (terms.reviewed_bank.host || terms.reviewed_bank.device) {
      detail::CohortBoundaryRequire(
          entry.bank_estimate.host <= terms.reviewed_bank.host &&
              entry.bank_estimate.device <= terms.reviewed_bank.device,
          "cohort " + std::to_string(c) +
              " bank estimate exceeds the explicit reviewed bank reservation "
              "(estimate host=" +
              std::to_string(entry.bank_estimate.host) +
              " device=" + std::to_string(entry.bank_estimate.device) +
              "; reservation host=" + std::to_string(terms.reviewed_bank.host) +
              " device=" + std::to_string(terms.reviewed_bank.device) + ")");
    }
    // The cohort's COMPLETE ledger (admission unit; strictly sequential
    // cohorts are validated individually, never aggregated across cohorts)
    // in the CANONICAL composition: host = bank_charge + process +
    // queue/host; device = bank_charge + process. The bank charge is
    // allowance-inclusive (the single canonical margin already covers the
    // composed allowance); no extra allowance term is invented.
    ByteCharge ledger = {
        owned::CheckedAdd(
            bank_charge.host,
            owned::CheckedAdd(terms.process.host, terms.queue_transient.host)),
        owned::CheckedAdd(bank_charge.device, terms.process.device)};

    // MANDATORY admitted concurrency: active_jobs in [1, member_count]. A
    // cohort whose member count is below the declared run-level concurrency
    // admits exactly member_count owners; the wire declares this per-cohort
    // value.
    const size_t active_jobs = std::min(max_concurrent_solvers, count);
    const ByteCharge rolling_reserve = detail::LargestKSum(
        member_forecasts, active_jobs, "cohort rolling reserve");
    ValidatedCohortEntry validated_entry;
    validated_entry.first = first;
    validated_entry.count = count;
    validated_entry.original_jobs = entry.original_jobs;
    validated_entry.bank_estimate = entry.bank_estimate;
    validated_entry.member_private_max = member_max;
    validated_entry.active_jobs = active_jobs;
    validated_entry.rolling_reserve = rolling_reserve;
    validated_entry.ledger_terms = {
        owned::CheckedAdd(terms.process.host, terms.queue_transient.host),
        terms.process.device};
    validated_entry.ledger_bank_charge = bank_charge;
    validated_entry.ledger =
        ledger = {owned::CheckedAdd(ledger.host, rolling_reserve.host),
                  owned::CheckedAdd(ledger.device, rolling_reserve.device)};
    // HARD GATE, both sides, per cohort independently. Underfunded
    // admitted concurrency refuses: there is no dynamic throttle, no
    // degraded mode and no informational reserve.
    const bool fits =
        ledger.host <= capacity.host && ledger.device <= capacity.device;
    validated_entry.ledger_fits_capacity = fits;
    detail::CohortBoundaryRequire(
        fits,
        "cohort " + std::to_string(c) +
            " COMPLETE ledger does not fit the total CLI capacity (hard "
            "admitted-concurrency gate; no throttling): ledger host=" +
            std::to_string(ledger.host) +
            " device=" + std::to_string(ledger.device) +
            "; capacity host=" + std::to_string(capacity.host) +
            " device=" + std::to_string(capacity.device) +
            "; bank_charge host=" + std::to_string(bank_charge.host) +
            " device=" + std::to_string(bank_charge.device) +
            "; active_jobs=" + std::to_string(active_jobs) +
            "; rolling_reserve host=" + std::to_string(rolling_reserve.host) +
            " device=" + std::to_string(rolling_reserve.device) +
            "; ledger_terms host=" +
            std::to_string(validated_entry.ledger_terms.host) +
            " device=" + std::to_string(validated_entry.ledger_terms.device));
    validated.cohorts_.push_back(std::move(validated_entry));

    for (size_t k = 0; k < count; ++k) {
      ValidatedCohortPlan::MappingEntry mapping_entry;
      mapping_entry.original_job = selection[first + k];
      mapping_entry.selection_ordinal = first + k;
      mapping_entry.cohort = c;
      mapping_entry.cohort_local_ordinal = k;
      validated.mapping_.push_back(mapping_entry);
    }
  }
  return validated;
}

}  // namespace resident
}  // namespace mvs
}  // namespace colmap