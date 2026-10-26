// SPDX-License-Identifier: BSD-3-Clause
// Native ledger minima computation for the resident forecast gates. This
// translation unit depends on the dense-stereo bank estimator
// (HostInputBank::Estimate/EstimateBounded/EstimateNormalAllowance, which
// opens image headers via OpenImageIO); it is compiled only where that
// library is available and is never stubbed into place.

#include "colmap/mvs/owned_resident_minima.h"
#include "colmap/mvs/resident_forecast.h"
#include "colmap/util/logging.h"

#include <algorithm>

namespace colmap {
namespace mvs {

ResidentLedgerMinima ComputeResidentLedgerMinima(
    const std::vector<owned::BankJob>& jobs,
    const ResidentForecastNormals mode,
    const size_t max_future) {
  // Bounded-only first execution gate: eager mode is legal on the wire and
  // in the CPU estimators but never an execution minima source.
  THROW_CHECK(mode == ResidentForecastNormals::kBounded)
      << "Resident forecast ledger minima are a bounded-only execution "
         "surface; eager mode refuses";
  ResidentLedgerMinima minima;
  const owned::BankRawLimits raw = owned::HostInputBank::RawEstimate(jobs);
  minima.bank_host_raw_bytes = raw.bounded_host_raw_bytes;
  minima.bank_device_raw_bytes = raw.device_raw_bytes;
  const owned::BankLimits bank = owned::HostInputBank::EstimateBounded(jobs);
  minima.bank_host_bytes = bank.host_bytes;
  minima.bank_device_bytes = bank.device_bytes;
  const owned::NormalAllowance allowance =
      owned::HostInputBank::EstimateNormalAllowance(jobs, max_future);
  minima.normal_owner_bytes = allowance.owner_bytes;
  minima.normal_future_bytes = allowance.future_bytes;
  minima.normal_transient_bytes = allowance.transient_bytes;
  minima.private_minima.reserve(jobs.size());
  for (const auto& job : jobs) {
    const auto& reference = job.images.at(job.reference_image_idx);
    const auto private_minimum =
        owned::ResidentPrivateLogicalMinimum(reference.GetWidth(),
                                             reference.GetHeight(),
                                             job.source_image_idxs.size(),
                                             /*include_owner_normal=*/true);
    minima.private_minima.push_back(
        {private_minimum.host, private_minimum.device});
  }
  return minima;
}

}  // namespace mvs
}  // namespace colmap
