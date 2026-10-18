// SPDX-License-Identifier: BSD-3-Clause
// Native per-cohort charge floors + singleton feasibility for the cohort
// wire gates. This translation unit depends on the dense-stereo bank
// estimator (HostInputBank::RawEstimate / EstimateNormalAllowance, which
// open image headers via OpenImageIO); it is compiled only where that
// library is available and is never stubbed into place.

#include "colmap/mvs/resident_cohort_wire.h"

#include "colmap/mvs/owned_resident_minima.h"
#include "colmap/util/logging.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <vector>

namespace colmap {
namespace mvs {
namespace cohort {
namespace {

using owned::CheckedAdd;

// Revalidates the ORIGINAL captured identity records of ONE original
// invocation problem against the current filesystem, before or after an
// estimator traversal. The ORIGINAL snapshot (caller-owned, never a
// recapture) is the only accepted identity; a changed object, size or
// timestamps refuses.
void CheckSliceIdentitiesUnchanged(
    const std::vector<owned::BankJobIdentities>& invocation_identities,
    size_t first_original_ordinal, size_t count) {
  for (size_t k = 0; k < count; ++k) {
    const auto& snapshot = invocation_identities.at(first_original_ordinal + k);
    for (const auto& layer : snapshot) {
      owned::CheckFileIdentityUnchanged(layer.image);
      owned::CheckFileIdentityUnchanged(layer.depth);
      owned::CheckFileIdentityUnchanged(layer.normal);
    }
  }
}

// Singleton (one-member) hypothesis floors for one invocation problem.
struct SingletonFloors {
  uint64_t bank_host_floor;
  uint64_t bank_device_floor;
  uint64_t normal_charge_floor;
};

SingletonFloors ComputeSingletonFloors(const owned::BankJob& job,
                                       const size_t max_future,
                                       const uint64_t allowance_bytes) {
  const std::vector<owned::BankJob> slice{job};
  const owned::BankRawLimits raw = owned::HostInputBank::RawEstimate(slice);
  const owned::NormalAllowance normal =
      owned::HostInputBank::EstimateNormalAllowance(slice, max_future);
  SingletonFloors floors;
  // CANONICAL COMPOSITION, exactly once: the wire-declared whole-invocation
  // unclassified allowance composes with the margin-free raw BEFORE the
  // single canonical 3% margin; there is no second allowance ledger term.
  floors.bank_host_floor = owned::MarginBytes(
      CheckedAdd(raw.bounded_host_raw_bytes, allowance_bytes));
  floors.bank_device_floor = owned::MarginBytes(raw.device_raw_bytes);
  floors.normal_charge_floor =
      CheckedAdd(normal.future_bytes, normal.transient_bytes);
  return floors;
}

}  // namespace

void VerifyResidentCohortForecastNativeFloors(
    const CohortForecastWire& wire,
    const std::vector<owned::BankJob>& invocation_jobs,
    const std::vector<owned::BankJobIdentities>& invocation_identities,
    const uint64_t host_budget_bytes,
    const uint64_t device_budget_bytes) {
  THROW_CHECK_EQ(invocation_jobs.size(), invocation_identities.size())
      << "cohort native floors: the invocation jobs and the ORIGINAL "
         "identity snapshot must be one per problem, in problem order";
  THROW_CHECK_EQ(wire.member_count_total, invocation_jobs.size())
      << "cohort native floors: the wire members must cover exactly this "
         "whole invocation";
  // Execution-side bounded FIFO queue contract.
  THROW_CHECK(wire.max_future == 1 || wire.max_future == 2)
      << "cohort native floors support max_future 1 or 2 only (the bounded "
         "normal FIFO queue is the only execution scope), got "
      << wire.max_future;
  THROW_CHECK(!wire.unclassified_provenance.empty())
      << "cohort native floors: the unclassified allowance provenance is "
         "mandatory (including an explicit declared zero)";

  // Singleton feasibility preflight, EVERY member, before any payload: the
  // canonical composition with the native singleton floor bank charge.
  for (size_t j = 0; j < invocation_jobs.size(); ++j) {
    CheckSliceIdentitiesUnchanged(invocation_identities, j, 1);
    const SingletonFloors floors = ComputeSingletonFloors(
        invocation_jobs.at(j), static_cast<size_t>(wire.max_future),
        wire.unclassified_allowance_bytes);
    CheckSliceIdentitiesUnchanged(invocation_identities, j, 1);
    // The member's own cohort carries the whole-invocation basis in its
    // queue (parser gate); the singleton hypothesis uses the declared
    // whole-invocation basis, the process term, the singleton bank floor,
    // the NATIVE singleton normal charge and the member's wire private
    // charge.
    const auto& member = [&]() -> const CohortWireMember& {
      for (const auto& cohort : wire.cohorts) {
        for (const auto& m : cohort.members) {
          if (m.member_ordinal == j) {
            return m;
          }
        }
      }
      throw std::invalid_argument(
          "cohort native floors: no wire member for invocation ordinal " +
          std::to_string(j));
    }();
    const uint64_t singleton_ledger_host = [
        =,
        &wire] {
      uint64_t sum = wire.global_snapshot_plan_host_bytes;
      sum = CheckedAdd(sum, wire.summary.process_once_host_bytes);
      sum = CheckedAdd(sum, floors.bank_host_floor);
      sum = CheckedAdd(sum, floors.normal_charge_floor);
      return CheckedAdd(sum, member.private_host_bytes);
    }();
    const uint64_t singleton_ledger_device = [
        =] {
      uint64_t sum = wire.summary.process_once_device_bytes;
      sum = CheckedAdd(sum, floors.bank_device_floor);
      return CheckedAdd(sum, member.private_device_bytes);
    }();
    THROW_CHECK_LE(singleton_ledger_host, host_budget_bytes)
        << "cohort native floors: invocation ordinal " << j << " is not "
        << "feasible alone (singleton host ledger floor "
        << singleton_ledger_host << " exceeds the host cap "
        << host_budget_bytes << "): the all-singleton preflight refuses, "
        << "no throttling";
    THROW_CHECK_LE(singleton_ledger_device, device_budget_bytes)
        << "cohort native floors: invocation ordinal " << j << " is not "
        << "feasible alone (singleton device ledger floor "
        << singleton_ledger_device << " exceeds the device cap "
        << device_budget_bytes << "): the all-singleton preflight refuses, "
        << "no throttling";
  }

  // Per-cohort floors: allowance-inclusive bank charge, native normal
  // allowances (EXACT equality) and per-member private floors.
  for (const auto& cohort : wire.cohorts) {
    std::vector<owned::BankJob> cohort_jobs;
    cohort_jobs.reserve(cohort.members.size());
    for (const auto& member : cohort.members) {
      cohort_jobs.push_back(
          invocation_jobs.at(member.member_ordinal));  // original order
    }
    size_t cohort_first = cohort.members.front().member_ordinal;
    CheckSliceIdentitiesUnchanged(invocation_identities, cohort_first,
                                  cohort_jobs.size());
    const owned::BankRawLimits raw =
        owned::HostInputBank::RawEstimate(cohort_jobs);
    const owned::NormalAllowance normal =
        owned::HostInputBank::EstimateNormalAllowance(
            cohort_jobs, static_cast<size_t>(wire.max_future));
    CheckSliceIdentitiesUnchanged(invocation_identities, cohort_first,
                                  cohort_jobs.size());
    // The allowance folds into the raw BEFORE the single canonical margin,
    // exactly once (compose-once rule; never re-margined).
    const uint64_t bank_host_floor = owned::MarginBytes(
        CheckedAdd(raw.bounded_host_raw_bytes,
                   wire.unclassified_allowance_bytes));
    THROW_CHECK_GE(cohort.bank_charge_host_bytes, bank_host_floor)
        << "cohort " << cohort.ordinal
        << " bank host charge is below the native allowance-composed raw "
        << "minimum for this cohort slice";
    const uint64_t bank_device_floor =
        owned::MarginBytes(raw.device_raw_bytes);
    THROW_CHECK_GE(cohort.bank_charge_device_bytes, bank_device_floor)
        << "cohort " << cohort.ordinal
        << " bank device charge is below the native raw minimum for this "
        << "cohort slice";
    THROW_CHECK_EQ(cohort.normal_owner_bytes, normal.owner_bytes)
        << "cohort " << cohort.ordinal
        << " normal owner allowance disagrees with the native logical "
           "recompute";
    THROW_CHECK_EQ(cohort.normal_future_bytes, normal.future_bytes)
        << "cohort " << cohort.ordinal
        << " normal future allowance disagrees with the native logical "
           "recompute";
    THROW_CHECK_EQ(cohort.normal_transient_bytes, normal.transient_bytes)
        << "cohort " << cohort.ordinal
        << " normal transient allowance disagrees with the native logical "
           "recompute";
    THROW_CHECK_GE(cohort.normal_charge_host_bytes,
                   CheckedAdd(normal.future_bytes, normal.transient_bytes))
        << "cohort " << cohort.ordinal
        << " normal charge is below the required future+transient "
        << "allowance";
    for (size_t m = 0; m < cohort.members.size(); ++m) {
      const owned::BankJob& job = cohort_jobs.at(m);
      const auto& reference = job.images.at(job.reference_image_idx);
      const auto private_minimum = owned::ResidentPrivateLogicalMinimum(
          reference.GetWidth(), reference.GetHeight(),
          job.source_image_idxs.size(), /*include_owner_normal=*/true);
      THROW_CHECK_GE(cohort.members.at(m).private_host_bytes,
                     private_minimum.host)
          << "cohort " << cohort.ordinal << " member " << m
          << " private host charge is below the native logical minimum";
      THROW_CHECK_GE(cohort.members.at(m).private_device_bytes,
                     private_minimum.device)
          << "cohort " << cohort.ordinal << " member " << m
          << " private device charge is below the native logical minimum";
    }
  }
}

}  // namespace cohort
}  // namespace mvs
}  // namespace colmap