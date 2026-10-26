// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "colmap/mvs/owned_pair_coordinator.h"
#include "colmap/mvs/owned_pair_validation.h"

#include <functional>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace colmap {
namespace mvs {
namespace owned {

// Explicit outside-formula admission terms for opt-in contiguous FIFO GPU
// cohorts of resolved jobs (backlog story 04). Every term is reviewed and
// explicit; nothing is invented or hardcoded here. The bank estimate comes
// from the native metadata estimators (the current 3%-margin contract); this
// planner never touches payloads and runs before any host/GPU allocation.
//
// Every singleton's complete private working set must fit against the TOTAL
// host/device envelope (its own full bank including construction/staging +
// fixed process + queue/transient + complete private job + explicit
// unclassified allowance) before any payload allocation; this is
// deliberately NOT a bank-only device cap. Rolling admission may still
// reduce the active concurrency below the cohort size.
struct CohortTerms {
  // Fixed reviewed process charge; disjoint from private forecasts.
  ByteCharge process;
  // Explicit allowance for the bounded normal loader queue and other
  // transient pipeline buffers; disjoint from bank and private forecasts.
  ByteCharge queue_transient;
  // Explicit allowance outside the native formula contract (unclassified
  // remainder, reported by the runtime memory model adapter).
  ByteCharge unclassified;
  // ReviewED fixed per-cohort bank reservation. NONZERO enables the actual
  // reservation policy used at execution: the union bank estimate must fit
  // INSIDE this reservation (the reservation, not the estimate, is what the
  // ledger deducts from capacity), and the reservation + explicit terms +
  // largest member private forecast must fit the capacity. Zero keeps the
  // legacy estimate-only contract (bank estimate directly in the total).
  // Either way EVERY singleton's fit is checked before any packing decision;
  // the estimate-side checks still constrain every candidate cohort.
  ByteCharge reviewed_bank;
};

// One planned contiguous FIFO cohort over the resolved job selection.
struct CohortPlanEntry {
  // Position and length of the contiguous FIFO slice in the selection
  // (selection-ordinal space; event/generation offsets come from `first`).
  size_t first = 0;
  size_t count = 0;
  // Original resolved job indices in FIFO order (explicit mapping).
  std::vector<size_t> original_jobs;
  // Native metadata bank estimate for the slice, including the single 3%
  // margin (retained payloads + metadata + construction/staging allowance;
  // device layout estimate).
  ByteCharge bank_estimate;
  // Largest member complete private forecast (reviewed bound).
  ByteCharge member_private;
  // Exact admission arithmetic as admitted: bank + process + queue/transient
  // + private (terms_total), then + unclassified (required). Each term is
  // exposed separately so the runtime memory model adapter can verify.
  ByteCharge terms_total;
  ByteCharge required;
};

struct CohortPlan {
  std::vector<CohortPlanEntry> cohorts;
  size_t job_count = 0;
  // Original selection id -> (cohort ordinal, cohort-local job ordinal).
  // Linear scan; -1 when the id is not part of the plan.
  int CohortOf(size_t original_job) const { return Find(original_job).first; }
  int CohortOrdinalOf(size_t original_job) const {
    return Find(original_job).second;
  }

 private:
  std::pair<int, int> Find(size_t original_job) const {
    for (size_t c = 0; c < cohorts.size(); ++c) {
      const auto& jobs = cohorts[c].original_jobs;
      for (size_t k = 0; k < jobs.size(); ++k) {
        if (jobs[k] == original_job) {
          return {static_cast<int>(c), static_cast<int>(k)};
        }
      }
    }
    return {-1, -1};
  }
};

// Native-slice estimator: (first, count) in selection-ordinal FIFO order.
// Production uses the real bank estimators (Estimate/EstimateBounded); tests
// may inject an independent reference.
using CohortBankEstimator =
    std::function<ByteCharge(size_t first, size_t count)>;

namespace detail {

// The reviewed fixed reservation (nonzero) is what execution deducts from
// the capacity and what the bank estimate must fit inside: explicit matching
// of the planned arithmetic to the reservation policy.
inline const ByteCharge& CohortBankCharge(const CohortTerms& terms,
                                          const ByteCharge& bank) {
  return terms.reviewed_bank.host || terms.reviewed_bank.device
             ? terms.reviewed_bank
             : bank;
}

// Checked admission arithmetic: bank charge + process + queue/transient +
// private (+ unclassified, outside the formula contract but inside the
// budget).
inline ByteCharge CohortRequired(const ByteCharge& bank,
                                 const CohortTerms& terms,
                                 const ByteCharge& private_job,
                                 const char* context) {
  const ByteCharge& bank_charge = CohortBankCharge(terms, bank);
  ByteCharge required;
  try {
    required.host = CheckedAdd(
        CheckedAdd(CheckedAdd(CheckedAdd(bank_charge.host, terms.process.host),
                              terms.queue_transient.host),
                   private_job.host),
        terms.unclassified.host);
    required.device = CheckedAdd(
        CheckedAdd(
            CheckedAdd(CheckedAdd(bank_charge.device, terms.process.device),
                       terms.queue_transient.device),
            private_job.device),
        terms.unclassified.device);
  } catch (const std::overflow_error&) {
    throw std::overflow_error(std::string(context) +
                              ": cohort admission arithmetic overflow");
  }
  return required;
}

// True when required fits BOTH sides. Never silently weakens a failure: the
// caller turns a false result into the explicit oversized diagnostic below.
inline bool CohortFits(const ByteCharge& required, const ByteCharge& capacity) {
  return required.host <= capacity.host && required.device <= capacity.device;
}

// True when the union bank estimate fits inside the actual reservation (or
// the capacity-side fit in legacy mode) plus explicit terms and the largest
// member private forecast. Cohorts pack on the FULL actual policy so later
// admissions cannot fail what planning already claimed fit.
inline bool CohortMembershipFits(const ByteCharge& bank,
                                 const CohortTerms& terms,
                                 const ByteCharge& private_job,
                                 const ByteCharge& capacity) {
  if (terms.reviewed_bank.host || terms.reviewed_bank.device) {
    if (bank.host > terms.reviewed_bank.host ||
        bank.device > terms.reviewed_bank.device) {
      return false;
    }
  }
  const ByteCharge required =
      CohortRequired(bank, terms, private_job, "cohort membership");
  return CohortFits(required, capacity);
}

// Exact limiting-resource diagnostic for a rejected singleton/pair. The
// limiting resource is the side with the larger positive deficit across the
// membership and budget checks; all components are always reported.
inline std::string CohortOversizedDiagnostic(const char* stage,
                                             size_t original_job,
                                             size_t selection_ordinal,
                                             const ByteCharge& bank,
                                             const CohortTerms& terms,
                                             const ByteCharge& private_job,
                                             const ByteCharge& capacity) {
  const ByteCharge required =
      CohortRequired(bank, terms, private_job, "cohort diagnostic");
  const bool policy = terms.reviewed_bank.host || terms.reviewed_bank.device;
  const int64_t bank_host_deficit =
      policy && bank.host > terms.reviewed_bank.host
          ? static_cast<int64_t>(bank.host - terms.reviewed_bank.host)
          : 0;
  const int64_t bank_device_deficit =
      policy && bank.device > terms.reviewed_bank.device
          ? static_cast<int64_t>(bank.device - terms.reviewed_bank.device)
          : 0;
  const int64_t host_deficit =
      required.host > capacity.host
          ? static_cast<int64_t>(required.host - capacity.host)
          : 0;
  const int64_t device_deficit =
      required.device > capacity.device
          ? static_cast<int64_t>(required.device - capacity.device)
          : 0;
  const int64_t worst_host = std::max(bank_host_deficit, host_deficit);
  const int64_t worst_device = std::max(bank_device_deficit, device_deficit);
  const char* limiting = worst_host || worst_device
                             ? (worst_host >= worst_device ? "host" : "device")
                             : "host";
  std::string message = "cohort: ";
  message += stage;
  message +=
      " oversized bank_job_index=" + std::to_string(original_job) +
      " selection_ordinal=" + std::to_string(selection_ordinal) +
      " bank_host=" + std::to_string(bank.host) +
      " bank_device=" + std::to_string(bank.device) +
      " reviewed_bank_host=" + std::to_string(terms.reviewed_bank.host) +
      " reviewed_bank_device=" + std::to_string(terms.reviewed_bank.device) +
      " bank_deficit_host=" + std::to_string(bank_host_deficit) +
      " bank_deficit_device=" + std::to_string(bank_device_deficit) +
      " process_host=" + std::to_string(terms.process.host) +
      " process_device=" + std::to_string(terms.process.device) +
      " queue_host=" + std::to_string(terms.queue_transient.host) +
      " queue_device=" + std::to_string(terms.queue_transient.device) +
      " private_host=" + std::to_string(private_job.host) +
      " private_device=" + std::to_string(private_job.device) +
      " unclassified_host=" + std::to_string(terms.unclassified.host) +
      " unclassified_device=" + std::to_string(terms.unclassified.device) +
      " required_host=" + std::to_string(required.host) +
      " required_device=" + std::to_string(required.device) +
      " budget_host=" + std::to_string(capacity.host) +
      " budget_device=" + std::to_string(capacity.device) +
      " deficit_host=" + std::to_string(host_deficit) +
      " deficit_device=" + std::to_string(device_deficit) +
      " limiting_resource=" + limiting;
  return message;
}

}  // namespace detail

// Partition resolved jobs into contiguous FIFO cohorts whose union
// reference+source banks, together with the explicit terms and every
// member's complete private forecast, fit the TOTAL capacity.
//
// Preflight of ALL singletons happens before any cohort packing decision;
// the first oversized singleton aborts planning with the exact limiting-
// resource diagnostic, before any payload/CUDA allocation anywhere.
//
// `selection` is the immutable FIFO original job id list (identity order,
// strictly increasing); `forecasts` holds one complete reviewed private
// forecast (host/device) per selected job, same order. The bank estimator
// receives (first, count) in selection-ordinal space and returns the native
// metadata estimate for that contiguous slice.
inline CohortPlan PlanCohorts(const std::vector<size_t>& selection,
                              const std::vector<ByteCharge>& forecasts,
                              const CohortTerms& terms,
                              ByteCharge capacity,
                              const CohortBankEstimator& estimate_slice) {
  if (selection.empty()) {
    throw std::invalid_argument("cohort plan: empty selection");
  }
  if (forecasts.size() != selection.size()) {
    throw std::invalid_argument(
        "cohort plan: forecast/selection identity count mismatch");
  }
  if (!estimate_slice) {
    throw std::invalid_argument("cohort plan: missing bank estimator");
  }
  for (size_t k = 0; k < selection.size(); ++k) {
    if (!forecasts[k].host || !forecasts[k].device) {
      throw std::invalid_argument(
          "cohort plan: zero private forecast for bank_job_index=" +
          std::to_string(selection[k]));
    }
    if (k && selection[k] <= selection[k - 1]) {
      throw std::invalid_argument(
          "cohort plan: selection is not a strictly ordered FIFO identity "
          "(identity fail at bank_job_index=" +
          std::to_string(selection[k]) + ")");
    }
  }
  // Explicit preflight: every singleton against the TOTAL envelope, before
  // any payload/CUDA allocation and before any packing decision, under the
  // FULL actual reservation policy (reviewed bank charge, not a raw bank
  // estimate) when one is configured.
  for (size_t k = 0; k < selection.size(); ++k) {
    const ByteCharge bank = estimate_slice(k, 1);
    if (!detail::CohortMembershipFits(bank, terms, forecasts[k], capacity)) {
      throw std::runtime_error(detail::CohortOversizedDiagnostic(
          "preflight", selection[k], k, bank, terms, forecasts[k], capacity));
    }
  }
  // Greedy contiguous FIFO packing: extend the current cohort while the
  // union bank + terms + largest member private forecast fits the FULL
  // actual reservation policy; close and continue at the first non-fitting
  // extension. The tail cohort may be smaller than the budget allows.
  CohortPlan plan;
  plan.job_count = selection.size();
  size_t first = 0;
  while (first < selection.size()) {
    size_t end = first + 1;
    ByteCharge bank = estimate_slice(first, end - first);
    ByteCharge cohort_private = forecasts[first];
    ByteCharge required =
        detail::CohortRequired(bank, terms, cohort_private, "cohort packing");
    while (end < selection.size()) {
      const ByteCharge candidate_bank = estimate_slice(first, end + 1 - first);
      // Per-side maxima are independent: the binding side decides
      // membership, never the cheapest profile.
      ByteCharge candidate_private = cohort_private;
      if (forecasts[end].host > candidate_private.host) {
        candidate_private.host = forecasts[end].host;
      }
      if (forecasts[end].device > candidate_private.device) {
        candidate_private.device = forecasts[end].device;
      }
      const ByteCharge candidate_required = detail::CohortRequired(
          candidate_bank, terms, candidate_private, "cohort packing");
      if (!detail::CohortMembershipFits(
              candidate_bank, terms, candidate_private, capacity)) {
        // All singletons already preflighted, so the current cohort is
        // nonempty; close it and continue with the next FIFO segment.
        break;
      }
      bank = candidate_bank;
      cohort_private = candidate_private;
      required = candidate_required;
      ++end;
    }
    CohortPlanEntry entry;
    entry.first = first;
    entry.count = end - first;
    entry.original_jobs.assign(selection.begin() + static_cast<long>(first),
                               selection.begin() + static_cast<long>(end));
    entry.bank_estimate = bank;
    entry.member_private = cohort_private;
    const ByteCharge bank_charge = detail::CohortBankCharge(terms, bank);
    entry.terms_total = {
        CheckedAdd(CheckedAdd(CheckedAdd(bank_charge.host, terms.process.host),
                              terms.queue_transient.host),
                   cohort_private.host),
        CheckedAdd(
            CheckedAdd(CheckedAdd(bank_charge.device, terms.process.device),
                       terms.queue_transient.device),
            cohort_private.device)};
    entry.required =
        detail::CohortRequired(bank, terms, cohort_private, "cohort packing");
    plan.cohorts.push_back(std::move(entry));
    first = end;
  }
  return plan;
}

}  // namespace owned
}  // namespace mvs
}  // namespace colmap