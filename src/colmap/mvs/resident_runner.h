// SPDX-License-Identifier: BSD-3-Clause
#pragma once

// Opt-in bounded standard-CLI resident execution backend for slice B:
// geometry-on-existing-photo, COLMAP workspace, single explicit device,
// FULL-BANK resident rolling with a bounded reference-normal FIFO.
//
// No cohort/private-photo/static-3/benchmark selector, no Python switch,
// no automatic fresh photo phase, no resume, no default promotion: this
// backend runs ONLY when the CLI explicitly opts in with a resident
// forecast, and stock remains the default execution route.
//
// Execution is BOUNDED-ONLY: the forecast parser reads every legal wire,
// but this runner refuses eager declarations (the bounded FIFO producer is
// the only first-gate execution scope) and refuses provisional/absent/
// unknown fingerprints (the compiled closure digest is REQUIRED; there is
// no native registry -- the external versioned registry lives only in the
// canonical Python tooling).
//
// Gate discipline, all before any payload:
//   1. one authoritative invocation snapshot: a single
//      ResolveOwnedBankContractWithJobs traversal feeds BOTH the wire
//      binding bytes and the ledger minima (invocation file identities are
//      captured once and validated through bank construction);
//   2. ParseResidentForecast with a REAL sha function (required);
//   3. binding (exact regenerated snapshot bytes + reference blocks);
//   4. parameters (executable qualification, complete, rolling declared,
//      bounded, both fingerprints == the compiled digest);
//   5. ledger terms (planner/rolling consistency, single-process shape,
//      canonical 2 GiB process floor, canonical margins/labels);
//   6. native minima recompute + charge gate (bank raw + allowance
//      composition, normal allowance equality, private minima);
//   7. aggregate budget gate (FitsCaps: bank + process + metadata + normal
//      queue + rolling SUM against the caller's caps; the planner MAX terms
//      are consistency checks, never budget additions);
//   8. full-invocation coordinator admission (bank folded in, then
//      dedicated outside the rolling ledger).
//
// Failure model: the first error cancels every bounded waiter; owners drain
// and join on every error path (RunRollingOwners); the bounded producer is
// wired to prompt, non-joining cancellation on any coordinator failure and
// joins only in parent cleanup, before bank teardown. Cancellation is
// independent of the optional observer.

#include "colmap/mvs/normal_map.h"
#include "colmap/mvs/owned_input_bank.h"
#include "colmap/mvs/owned_normal_lookahead.h"
#include "colmap/mvs/owned_pair_coordinator.h"
#include "colmap/mvs/owned_pair_run.h"
#include "colmap/mvs/owned_pair_validation.h"
#include "colmap/mvs/owned_resident_minima.h"
#include "colmap/mvs/patch_match.h"
#include "colmap/mvs/resident_forecast.h"
#include "colmap/util/file.h"
#include "colmap/util/logging.h"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "colmap/mvs/generated/resident_allocation_fingerprint.h"

namespace colmap {
namespace mvs {
namespace resident {

// Single explicit device, bounded full-bank rolling admission values.
struct ResidentRunnerLimits {
  uint64_t host_budget_bytes = 0;
  uint64_t device_budget_bytes = 0;
  // Must equal the forecast's declared rolling active_jobs (verified).
  size_t max_concurrent_solvers = 0;
  int owned_tile_threads = 32;
  // Absolute native phase entry budget, shared unchanged by every cohort.
  // The CLI's phase guard also bounds synchronous setup, drains and joins.
  std::optional<NativeRunDeadline> execution_deadline;
};

// Legacy controls without an explicit phase policy keep this per-wait timeout.
constexpr std::chrono::milliseconds kResidentRunnerTimeoutMs{300000};

// Executes the bounded full-bank resident run under the forecast's verified
// ledger. `sha` is REQUIRED (the product CLI passes ComputeSHA256; a build
// without a crypto backend refuses execution honestly via the binding
// gate). The optional observer receives coordinator/owner lifecycle events
// only; cancellation never depends on it.
// Returns EXIT_SUCCESS; throws on any gate/admission/execution failure.
int RunResidentBank(
    const PatchMatchOptions& invocation_options,
    const std::filesystem::path& workspace_path,
    const std::string& workspace_format,
    const std::string& pmvs_option_name,
    const std::filesystem::path& config_path,
    const std::filesystem::path& forecast_path,
    const ResidentRunnerLimits& limits,
    ResidentSha256Fn sha,
    const std::function<void(size_t, const char*)>& observer = {});

namespace {

// The implementation requires the CUDA dense-stereo library; the
// declaration stays available so callers fail on availability explicitly.
#if defined(COLMAP_CUDA_ENABLED)

inline int RunResidentBankImpl(
    const PatchMatchOptions& invocation_options,
    const std::filesystem::path& workspace_path,
    const std::string& workspace_format,
    const std::string& pmvs_option_name,
    const std::filesystem::path& config_path,
    const std::filesystem::path& forecast_path,
    const ResidentRunnerLimits& limits,
    const ResidentSha256Fn sha,
    const std::function<void(size_t, const char*)>& observer) {
  THROW_CHECK(sha != nullptr)
      << "Resident execution requires SHA-256 verification (the product "
         "CLI supplies ComputeSHA256; crypto-less builds refuse)";
  THROW_CHECK_GT(limits.max_concurrent_solvers, 0u);
  THROW_CHECK(!forecast_path.empty());

  // 1. Parse the wire with the REAL hash (declared digest is verified).
  std::ifstream forecast_file(forecast_path, std::ios::binary);
  THROW_CHECK_FILE_OPEN(forecast_file, forecast_path);
  std::string forecast_bytes((std::istreambuf_iterator<char>(forecast_file)),
                             std::istreambuf_iterator<char>());
  const ResidentForecastWire wire = ParseResidentForecast(forecast_bytes, sha);

  // 2. ONE authoritative invocation snapshot: single resolution gives the
  // contract (binding bytes) AND jobs (ledger minima) in native order.
  auto format = workspace_format;
  StringToLower(&format);
  THROW_CHECK(format == "colmap" || format == "pmvs")
      << "Invalid `workspace_format` " << workspace_format
      << " - supported values are 'COLMAP' or 'PMVS'.";
  PatchMatchController resolution_controller(
      invocation_options, workspace_path, format, pmvs_option_name,
      config_path, PatchMatchExecution::kStock);
  auto resolved = resolution_controller.ResolveOwnedBankContractWithJobs();
  const std::string fresh_bytes = SerializeOwnedBankContract(resolved.contract);

  // 3-7. Every gate, in order, all BEFORE any payload allocation. The
  // ORIGINAL capture records (not a recapture) are the binding identity:
  // they are freshly revalidated around EVERY estimator traversal so a
  // same-shaped replacement between resolution and payload refuses.
  VerifyResidentForecastBinding(wire, fresh_bytes, sha);
  const auto bound_identities = ContractJobIdentities(resolved.contract);
  VerifyResidentForecastParameters(wire, kResidentAllocationFingerprint);
  VerifyResidentForecastLedgerTerms(wire, limits.max_concurrent_solvers);
  // Before the minima estimators (which open input file headers).
  CheckContractIdentitiesUnchanged(bound_identities);
  const auto minima = ComputeResidentLedgerMinima(
      resolved.jobs, ResidentForecastNormals::kBounded,
      static_cast<size_t>(wire.max_future));
  // After ALL estimator work, before any bank construction.
  CheckContractIdentitiesUnchanged(bound_identities);
  VerifyResidentForecastCharges(wire, minima);
  VerifyResidentForecastFitsCaps(wire, limits.host_budget_bytes,
                                 limits.device_budget_bytes);

  // Single explicit device (also enforced during resolution).
  const uint64_t device_value = owned::ParseBytes(invocation_options.gpu_index);
  THROW_CHECK_LT(device_value, 2147483648ull)
      << "Resident device index overflow: " << invocation_options.gpu_index;
  const int device_index = static_cast<int>(device_value);

  const size_t bank_job_count = resolved.jobs.size();
  const std::vector<owned::ByteCharge> forecasts = [&wire] {
    std::vector<owned::ByteCharge> values;
    values.reserve(wire.references.size());
    for (const auto& reference : wire.references) {
      values.push_back({reference.private_host_bytes,
                        reference.private_device_bytes});
    }
    return values;
  }();

  // Full-invocation admission validation BEFORE payload (bank folded into
  // the process reservation for the check, exactly like the accepted
  // driver, then dedicated outside the rolling ledger).
  const owned::ByteCharge bank_charge{wire.bank_host_bytes,
                                      wire.bank_device_bytes};
  const owned::ByteCharge admission_process{
      owned::CheckedAdd(
          wire.process_host_bytes,
          owned::CheckedAdd(wire.normal_charge_bytes, bank_charge.host)),
      owned::CheckedAdd(wire.process_device_bytes, bank_charge.device)};
  {
    owned::PairCoordinator admission(
        {limits.host_budget_bytes, limits.device_budget_bytes},
        admission_process,
        limits.max_concurrent_solvers,
        forecasts,
        kResidentRunnerTimeoutMs,
        {},
        limits.execution_deadline);
  }

  // The bank's identity snapshot IS the ORIGINAL contract record set: it
  // is never recaptured, so a same-shaped replacement after the binding
  // cannot become the bank's accepted input. The snapshot above was freshly
  // revalidated before bank construction; the bank keeps validating every
  // later read against these originals (never against a recapture).
  const auto identity_snapshot = ContractJobIdentities(resolved.contract);

  // Fixed ledger reservation includes the metadata dynamic arrays ONCE
  // (host side), consistent with the FitsCaps aggregate arithmetic
  // (bank + process + metadata + normal queue + rolling reserve); the
  // bank stays dedicated OUTSIDE the rolling ledger.
  const owned::ByteCharge ledger_process{
      owned::CheckedAdd(
          wire.process_host_bytes,
          owned::CheckedAdd(wire.metadata_charge_bytes,
                            wire.normal_charge_bytes)),
      wire.process_device_bytes};
  const auto ledger_capacity = [bank_charge](const uint64_t budget,
                                             const uint64_t bank) {
    if (budget < bank) {
      throw std::invalid_argument(
          "resident ledger budget below the dedicated bank charge");
    }
    return budget - bank;
  };
  const owned::ByteCharge ledger{
      ledger_capacity(limits.host_budget_bytes, bank_charge.host),
      ledger_capacity(limits.device_budget_bytes, bank_charge.device)};

  owned::HostInputBank host(std::move(resolved.jobs),
                            owned::BankLimits{bank_charge.host,
                                              bank_charge.device},
                            owned::BankNormals::kBounded,
                            identity_snapshot.data(),
                            identity_snapshot.size());
  owned::ResidentInputBank gpu(host, device_index);
  host.ReleaseUploadPayloads();

  // Bounded FIFO reference-normal producer over ALL layers in native
  // problem order. Declared AFTER the banks: its destructor joins the CPU
  // producer before bank teardown (the loader borrows the host bank).
  std::vector<size_t> pipeline_layers;
  pipeline_layers.reserve(bank_job_count);
  for (size_t generation = 0; generation < bank_job_count; ++generation) {
    pipeline_layers.push_back(host.ReferenceLayer(generation));
  }
  auto normal_pipeline = std::make_unique<owned::NormalLookahead>(
      std::move(pipeline_layers), static_cast<size_t>(wire.max_future),
      wire.normal_charge_bytes);
  normal_pipeline->Start(
      [&host](size_t layer, const std::atomic<bool>&) {
        return host.LoadReferenceNormal(layer);
      });

  // Coordinator sink: failure wiring is MANDATORY (blocked consumers/
  // producers must wake before owner drain/join), independent of the
  // optional observer.
  auto normal_pipeline_raw = normal_pipeline.get();
  const auto observer_storage = observer;
  const auto sink = [normal_pipeline_raw, &observer_storage](
                        const owned::PairEvent& e) {
    // MANDATORY ORDER: cancel first, observe after (Sol 3cdc fix 5; see
    // ResidentEventSink). A throwing observer is swallowed by the
    // coordinator's guarded sink only AFTER cancellation is published.
    ResidentEventSink(e.event == "cancelled",
                      [normal_pipeline_raw]() { normal_pipeline_raw->Cancel(); },
                      observer_storage ? &observer_storage : nullptr,
                      e.job, e.event.c_str());
  };
  owned::PairCoordinator coordinator(ledger,
                                     ledger_process,
                                     limits.max_concurrent_solvers,
                                     forecasts,
                                     kResidentRunnerTimeoutMs,
                                     sink,
                                     limits.execution_deadline);

  const auto run_job = [&](const size_t generation) {
    // The bounded FIFO handoff is unique per generation; the owner frame
    // keeps the reference normal alive until upload and owner retirement.
    mvs::NormalMap owned_normal = normal_pipeline->Consume(generation);
    PatchMatchController controller(
        invocation_options, workspace_path, format, pmvs_option_name,
        config_path, PatchMatchExecution::kOwnedCuda);
    PatchMatchController::OwnedPairHooks hooks;
    hooks.owned_tile_threads = limits.owned_tile_threads;
    hooks.check_inputs = [&](size_t, const PatchMatchOptions&,
                             const PatchMatch::Problem& problem) {
      // Owner-scope re-check: the ASSIGNED forecast charge must still
      // cover the native private minimum with the active owner normal.
      const auto& reference = problem.images->at(problem.ref_image_idx);
      const auto minimum = owned::ResidentPrivateLogicalMinimum(
          reference.GetWidth(), reference.GetHeight(),
          problem.src_image_idxs.size(), /*include_owner_normal=*/true);
      THROW_CHECK_GE(wire.references.at(generation).private_host_bytes,
                     minimum.host);
      THROW_CHECK_GE(wire.references.at(generation).private_device_bytes,
                     minimum.device);
    };
    controller.RunResidentOwnedPairJob(host, gpu, coordinator, generation,
                                       generation, hooks, &owned_normal);
  };

  // Returns (or throws) only after all owners drained and joined.
  const auto run_observer = observer
                                ? observer
                                : +[](size_t, const char*) {};
  owned::RunRollingOwners(coordinator, run_job, run_observer);

  // Ledger fully released on success (Finish ran inside RunRollingOwners).
  THROW_CHECK_EQ(coordinator.ReservedBytes().host, 0u)
      << "resident rolling ledger retained host credit after a successful "
         "drain";
  THROW_CHECK_EQ(coordinator.ReservedBytes().device, 0u)
      << "resident rolling ledger retained device credit after a "
         "successful drain";
  // Producer errors propagate AFTER the owner drain; destructor joins the
  // CPU producer before the banks tear down.
  normal_pipeline->Join();
  return EXIT_SUCCESS;
}

#else

// Non-CUDA builds never execute: the CLI refuses before this call runs,
// and any direct caller gets an explicit availability failure.
inline int RunResidentBankImpl(
    const PatchMatchOptions&, const std::filesystem::path&,
    const std::string&, const std::string&, const std::filesystem::path&,
    const std::filesystem::path&, const ResidentRunnerLimits&,
    const ResidentSha256Fn, const std::function<void(size_t, const char*)>&) {
  LOG(ERROR) << "The resident runner requires the dense-stereo library, "
                "which is built with CUDA support";
  return EXIT_FAILURE;
}

#endif  // COLMAP_CUDA_ENABLED

}  // namespace

inline int RunResidentBank(
    const PatchMatchOptions& invocation_options,
    const std::filesystem::path& workspace_path,
    const std::string& workspace_format,
    const std::string& pmvs_option_name,
    const std::filesystem::path& config_path,
    const std::filesystem::path& forecast_path,
    const ResidentRunnerLimits& limits,
    const ResidentSha256Fn sha,
    const std::function<void(size_t, const char*)>& observer) {
  return RunResidentBankImpl(invocation_options, workspace_path,
                             workspace_format, pmvs_option_name, config_path,
                             forecast_path, limits, sha, observer);
}

}  // namespace resident
}  // namespace mvs
}  // namespace colmap