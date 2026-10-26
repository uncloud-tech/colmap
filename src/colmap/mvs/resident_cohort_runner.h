// SPDX-License-Identifier: BSD-3-Clause
#pragma once

// Explicit opt-in cohort EXECUTOR for the standard CLI (next-cohort
// milestone): bounded geometry-on-EXISTING-photos, one explicit device,
// STRICTLY SEQUENTIAL FIFO cohorts from the distinct resident-cohort-
// forecast v1 wire, reusing the CURRENT concrete production machinery and
// no new generic injection:
//   full-bank resolution (once) -> HostInputBank/ResidentInputBank ->
//   bounded NormalLookahead producer -> PairCoordinator dynamic rolling
//   ledger -> RunRollingOwners -> drain/join -> teardown -> next cohort.
//
// The executor is the WIRE-level path: per-side reviewed R and the wire's
// per-cohort ledger composition (queue + bank_charge + normal charge +
// process + rolling host; bank + process + rolling device; the
// whole-invocation snapshot/plan basis once per active ledger; process
// once and identical across cohorts) are consumed directly from the
// parsed wire; no extra unclassified/b7 ledger term is invented.
//
// GATE DISCIPLINE, all BEFORE any payload (cohort 0 never builds before
// every gate below has passed for the WHOLE invocation):
//   1. one authoritative invocation snapshot: a single
//      ResolveOwnedBankContractWithJobs traversal feeds the wire binding
//      bytes and the ledger minima; the ORIGINAL captured identity
//      records are retained (never recaptured) and revalidated around
//      every estimator traversal and every later load;
//   2. ParseResidentCohortForecast (DISTINCT magic; a full-bank wire
//      refuses here; hash REQUIRED for execution);
//   3. binding (exact regenerated whole-invocation contract bytes + every
//      member block against the resolved problems);
//   4. parameters (executable cohort label, complete, rolling declared,
//      bounded mode, max_future 1..2, both fingerprints TEXT-equal and ==
//      the COMPILED cohort-scope allocation fingerprint of this build; an
//      unknown/refusing digest fails closed — there is no native
//      registry, and the full-bank 778 entry certifies nothing);
//   5. output/input path discipline (CPU, fail closed): every member's
//      resolved output depth/normal paths must NOT exist yet, and no
//      member output may alias another member's output or any admitted
//      input file;
//   6. native floors (VerifyResidentCohortForecastNativeFloors):
//      allowance-inclusive per-cohort bank charge floors under the single
//      canonical margin, EXACT native normal allowances, per-member
//      private floors with the active owner normal, and the native
//      singleton feasibility of EVERY member before payload;
//   7. hard FitsCaps (VerifyResidentCohortForecastFitsCaps): EVERY
//      cohort's complete ledger (top-k SUM reserve, never the planner
//      MAX; sequential cohorts validated independently, never summed)
//      fits the CLI caps on both sides, active_jobs <= the declared CLI
//      limit and member count.
//
// Per-cohort execution (strictly sequential; failure aborts remaining
// cohorts): fresh ledger (coordinator capacity minus the wire bank
// charge; process folded with the wire queue and normal charge once) and
// a fresh HostInputBank/ResidentInputBank pair per cohort (ORIGINAL
// identity slice; never a recapture, never a sort) — owners, producers
// and uploads are drained, joined and DESTROYED before the next cohort
// starts. The bounded FIFO producer hands its unique reference normal to
// the cohort-local generation; global model image indices, original bank
// job ordinals and host-bank layer order are preserved end to end
// (cohort-local ordinals map through ReferenceLayer/SourceLayers only).
// Cancellation is published through the mandatory ResidentEventSink
// ordering (cancel FIRST, observer after): a throwing (or absent)
// observer can never disable cancellation or strand an owner; the
// producer is joined only in parent cleanup. A failed cohort leaves the
// ledger/banks destroyed and refuses the remaining cohorts (no advancing
// after failure).
//
// CLI: explicit opt-in only (a separate cohort forecast flag); stock
// remains the default execution route; no auto-photo, no resume, no
// default promotion, no protocol/schema invention beyond the frozen v1
// wire. Owner-fixed tiles stay untouched. No GPU work belongs to any
// local writer; native qualification happens at the dev1 build.

#include "colmap/mvs/generated/resident_allocation_fingerprint.h"
#include "colmap/mvs/normal_map.h"
#include "colmap/mvs/owned_cohort.h"
#include "colmap/mvs/owned_input_bank.h"
#include "colmap/mvs/owned_normal_lookahead.h"
#include "colmap/mvs/owned_pair_coordinator.h"
#include "colmap/mvs/owned_pair_run.h"
#include "colmap/mvs/owned_pair_validation.h"
#include "colmap/mvs/patch_match.h"
#include "colmap/mvs/resident_cohort_wire.h"
#include "colmap/mvs/resident_forecast.h"
#include "colmap/mvs/resident_runner.h"
#include "colmap/mvs/workspace.h"
#include "colmap/util/file.h"
#include "colmap/util/logging.h"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <memory>
#include <set>
#include <string>
#include <vector>

namespace colmap {
namespace mvs {
namespace resident {

// Reuses the accepted resident runner limits (caps + the verified
// concurrency); the per-cohort active_jobs come from the wire and must
// additionally not exceed max_concurrent_solvers (the CLI limit).
using ResidentCohortLimits = ResidentRunnerLimits;

// CPU output-path discipline (fail closed BEFORE any payload): pass the
// WHOLE invocation's resolved jobs (the executor calls this once for the
// full invocation; the alias scope is complete, not per-cohort):
//   * every job's output depth/normal path must not exist yet;
//   * every output must not alias ANY admitted input of the WHOLE
//     invocation — image, depth AND normal files of every job — nor any
//     other output (including the same job's own depth vs normal pair).
// Canonical path comparisons from the owned validation helpers; a missing
// input object is tolerated here because identity revalidation already
// refuses it.
inline void CohortOutputPathsRefuseExistingOrAliased(
    const std::vector<owned::BankJob>& invocation_jobs) {
  std::vector<std::filesystem::path> inputs;
  inputs.reserve(invocation_jobs.size() * 6);
  for (const auto& job : invocation_jobs) {
    THROW_CHECK(!job.depth_map_path.empty() && !job.normal_map_path.empty())
        << "cohort member output paths must be resolved";
    for (const auto& input : job.inputs) {
      // The image input carries the model-image name/relative path; include
      // its resolved bitmap path in the complete alias scope.
      THROW_CHECK(!input.image.GetPath().empty())
          << "cohort member input image path must be resolved";
      inputs.push_back(input.image.GetPath());
      inputs.push_back(input.depth_path);
      inputs.push_back(input.normal_path);
    }
  }
  std::vector<std::filesystem::path> outputs;
  outputs.reserve(invocation_jobs.size() * 2);
  for (const auto& job : invocation_jobs) {
    outputs.push_back(job.depth_map_path);
    outputs.push_back(job.normal_map_path);
  }
  for (const auto& output : outputs) {
    THROW_CHECK(!std::filesystem::exists(output))
        << "cohort output already exists: " << output;
    for (const auto& input : inputs) {
      THROW_CHECK(!owned::PathsAlias(input, output))
          << "cohort output aliases an admitted input: " << output << " vs "
          << input;
    }
  }
  for (size_t a = 0; a < outputs.size(); ++a) {
    for (size_t b = a + 1; b < outputs.size(); ++b) {
      THROW_CHECK(!owned::PathsAlias(outputs[a], outputs[b]))
          << "cohort output aliases another output (including the same "
             "job's depth vs normal pair): "
          << outputs[a] << " vs " << outputs[b];
    }
  }
}

namespace {

#if defined(COLMAP_CUDA_ENABLED)

inline int RunResidentCohortBankImpl(
    const PatchMatchOptions& invocation_options,
    const std::filesystem::path& workspace_path,
    const std::string& workspace_format,
    const std::string& pmvs_option_name,
    const std::filesystem::path& config_path,
    const std::filesystem::path& forecast_path,
    const ResidentCohortLimits& limits,
    const ResidentSha256Fn sha,
    const std::function<void(size_t, const char*)>& observer) {
  THROW_CHECK(sha != nullptr)
      << "cohort execution requires SHA-256 verification (the product CLI "
         "supplies ComputeSHA256; crypto-less builds refuse)";
  THROW_CHECK_GT(limits.max_concurrent_solvers, 0u);
  THROW_CHECK(!forecast_path.empty());

  // 1. Parse the DISTINCT cohort wire with the REAL hash (hash-optional
  // parse is never a credential; the binding gate below requires sha).
  std::ifstream forecast_file(forecast_path, std::ios::binary);
  THROW_CHECK_FILE_OPEN(forecast_file, forecast_path);
  std::string forecast_bytes((std::istreambuf_iterator<char>(forecast_file)),
                             std::istreambuf_iterator<char>());
  const cohort::CohortForecastWire wire =
      cohort::ParseResidentCohortForecast(forecast_bytes, sha);

  // One authoritative invocation snapshot (metadata only, no CUDA).
  auto format = workspace_format;
  StringToLower(&format);
  THROW_CHECK(format == "colmap" || format == "pmvs")
      << "Invalid `workspace_format` " << workspace_format
      << " - supported values are 'COLMAP' or 'PMVS'.";
  PatchMatchController resolution_controller(invocation_options,
                                             workspace_path,
                                             format,
                                             pmvs_option_name,
                                             config_path,
                                             PatchMatchExecution::kStock);
  auto resolved = resolution_controller.ResolveOwnedBankContractWithJobs();
  const std::string fresh_bytes = SerializeOwnedBankContract(resolved.contract);

  // 3. Whole-invocation binding (real hash required) + member coverage.
  cohort::VerifyResidentCohortForecastBinding(wire, fresh_bytes, sha);
  // The ORIGINAL captured identity records, retained for the whole run
  // (never recaptured; the bank binds its reads to THESE originals).
  const auto identity_snapshot = ContractJobIdentities(resolved.contract);

  // Single explicit device.
  const uint64_t device_value = owned::ParseBytes(invocation_options.gpu_index);
  THROW_CHECK_LT(device_value, 2147483648ull)
      << "cohort device index overflow: " << invocation_options.gpu_index;
  const int device_index = static_cast<int>(device_value);

  // 5. Output/input path discipline (WHOLE invocation, fail closed,
  // complete alias scope incl. same-job depth/normal pairs).
  CohortOutputPathsRefuseExistingOrAliased(resolved.jobs);

  // 4. Execution parameters (executable cohort label; compiled cohort-
  // scope allocation fingerprint of THIS build; unknown digest fails).
  cohort::VerifyResidentCohortForecastParameters(
      wire, kResidentAllocationFingerprint);

  // 6. Native floors around never-recaptured originals (allowance-
  // inclusive bank charge, exact normal allowances, private floors,
  // singleton feasibility).
  CheckContractIdentitiesUnchanged(identity_snapshot);
  cohort::VerifyResidentCohortForecastNativeFloors(wire,
                                                   resolved.jobs,
                                                   identity_snapshot,
                                                   limits.host_budget_bytes,
                                                   limits.device_budget_bytes);
  CheckContractIdentitiesUnchanged(identity_snapshot);

  // 7. HARD FitsCaps: every cohort complete ledger (top-k SUM reserve)
  // against the CLI caps, active_jobs <= the CLI limit and members.
  cohort::VerifyResidentCohortForecastFitsCaps(wire,
                                               limits.host_budget_bytes,
                                               limits.device_budget_bytes,
                                               limits.max_concurrent_solvers);

  const auto run_observer = observer ? observer : +[](size_t, const char*) {};
  // 8. Strictly sequential cohort execution; a failure aborts the
  // remaining cohorts after the failing one drains and destroys.
  for (size_t c = 0; c < wire.cohorts.size(); ++c) {
    const auto& cohort = wire.cohorts.at(c);
    const size_t first = cohort.members.front().member_ordinal;
    const size_t count = cohort.member_count;
    // The ORIGINAL resolved jobs in FIFO order (never recaptured/sorted);
    // slicing preserves global problem order.
    std::vector<owned::BankJob> cohort_jobs(
        resolved.jobs.begin() + static_cast<long>(first),
        resolved.jobs.begin() + static_cast<long>(first + count));
    // Bank charge from the wire (parser-verified policy + native floor).
    const owned::BankLimits bank_charge{cohort.bank_charge_host_bytes,
                                        cohort.bank_charge_device_bytes};

    // Fresh ledger per cohort: capacity minus the DEDICATED bank charge
    // (the bank lives outside the rolling ledger until teardown);
    // process folds the wire process + queue (whole-invocation basis
    // inside) + the bounded normal charge exactly once.
    THROW_CHECK_LE(bank_charge.host_bytes, limits.host_budget_bytes)
        << "cohort bank charge above the host budget";
    THROW_CHECK_LE(bank_charge.device_bytes, limits.device_budget_bytes)
        << "cohort bank charge above the device budget";
    const owned::ByteCharge ledger_capacity{
        limits.host_budget_bytes - bank_charge.host_bytes,
        limits.device_budget_bytes - bank_charge.device_bytes};
    const owned::ByteCharge ledger_process{
        owned::CheckedAdd(owned::CheckedAdd(cohort.process_host_bytes,
                                            cohort.queue_host_bytes),
                          cohort.normal_charge_host_bytes),
        cohort.process_device_bytes};
    std::vector<owned::ByteCharge> member_forecasts;
    for (const auto& member : cohort.members) {
      member_forecasts.push_back(
          {member.private_host_bytes, member.private_device_bytes});
    }
    // Per-cohort admitted concurrency (1 <= k <= members, <= CLI limit).
    const size_t active_jobs =
        std::min(static_cast<size_t>(cohort.rolling_active_jobs), count);

    // Fresh per-cohort bank with the ORIGINAL identity slice (bounded
    // reference-normal mode; construction validates every admitted
    // member's captured identity; never recaptured, never sorted).
    owned::HostInputBank host(std::move(cohort_jobs),
                              bank_charge,
                              owned::BankNormals::kBounded,
                              identity_snapshot.data() + first,
                              count);
    // Bounded FIFO producer over the cohort's reference layers in native
    // problem order; declared AFTER the host bank, joined before teardown.
    std::vector<size_t> pipeline_layers;
    pipeline_layers.reserve(count);
    for (size_t local = 0; local < count; ++local) {
      pipeline_layers.push_back(host.ReferenceLayer(local));
    }
    auto normal_pipeline = std::make_unique<owned::NormalLookahead>(
        std::move(pipeline_layers),
        static_cast<size_t>(wire.max_future),
        cohort.normal_charge_host_bytes);
    normal_pipeline->Start([&host](size_t layer, const std::atomic<bool>&) {
      return host.LoadReferenceNormal(layer);
    });

    auto normal_pipeline_raw = normal_pipeline.get();
    const auto observer_storage = observer;
    const auto sink = [normal_pipeline_raw,
                       &observer_storage](const owned::PairEvent& e) {
      // MANDATORY ORDER: cancel first, observe after; a throwing observer
      // never disables cancellation; cancellation is observer-independent.
      ResidentEventSink(
          e.event == "cancelled",
          [normal_pipeline_raw]() { normal_pipeline_raw->Cancel(); },
          observer_storage ? &observer_storage : nullptr,
          e.job,
          e.event.c_str());
    };
    owned::PairCoordinator coordinator(ledger_capacity,
                                       ledger_process,
                                       active_jobs,
                                       member_forecasts,
                                       kResidentRunnerTimeoutMs,
                                       sink,
                                       limits.execution_deadline);

    // GPU upload of every cohort layer (all uploads drained before this
    // owner loop; the bank is destroyed together with the host bank before
    // the next cohort runs, and metadata is preserved on Release).
    owned::ResidentInputBank gpu(host, device_index);
    host.ReleaseUploadPayloads();
    const auto run_job = [&](const size_t generation) {
      // Unique bounded FIFO handoff per cohort-local ordinal (the original
      // bank job ordinal is generation + first; no renumbering anywhere).
      mvs::NormalMap owned_normal = normal_pipeline->Consume(generation);
      PatchMatchController controller(invocation_options,
                                      workspace_path,
                                      format,
                                      pmvs_option_name,
                                      config_path,
                                      PatchMatchExecution::kOwnedCuda);
      PatchMatchController::OwnedPairHooks hooks;
      hooks.owned_tile_threads = limits.owned_tile_threads;
      hooks.check_inputs = [&](size_t,
                               const PatchMatchOptions&,
                               const PatchMatch::Problem& problem) {
        const auto& reference = problem.images->at(problem.ref_image_idx);
        const auto minimum =
            owned::ResidentPrivateLogicalMinimum(reference.GetWidth(),
                                                 reference.GetHeight(),
                                                 problem.src_image_idxs.size(),
                                                 /*include_owner_normal=*/true);
        THROW_CHECK_GE(cohort.members.at(generation).private_host_bytes,
                       minimum.host);
        THROW_CHECK_GE(cohort.members.at(generation).private_device_bytes,
                       minimum.device);
      };
      // The SUBSET bank job index is the cohort-local GENERATION (the
      // per-cohort bank holds exactly this cohort's jobs); the ORIGINAL
      // invocation ordinal is `first + generation` and is used for
      // wire/telemetry binding only, never for subset-bank indexing.
      controller.RunResidentOwnedPairJob(
          host, gpu, coordinator, generation, generation, hooks, &owned_normal);
    };

    // Returns (or throws) only after all owners drained and joined.
    owned::RunRollingOwners(coordinator, run_job, run_observer);
    THROW_CHECK_EQ(coordinator.ReservedBytes().host, 0u)
        << "cohort " << c << " rolling ledger retained host credit";
    THROW_CHECK_EQ(coordinator.ReservedBytes().device, 0u)
        << "cohort " << c << " rolling ledger retained device credit";
    // Producer errors propagate after the owner drain; the destructor
    // joins the CPU producer before the banks tear down.
    normal_pipeline->Join();
    normal_pipeline.reset();
    // End of the bank-owning scope: host bank and resident bank are
    // destroyed here, strictly BEFORE the next cohort is admitted.
  }
  return EXIT_SUCCESS;
}

#else

// Non-CUDA builds never execute: the CLI refuses before this call runs.
inline int RunResidentCohortBankImpl(
    const PatchMatchOptions&,
    const std::filesystem::path&,
    const std::string&,
    const std::string&,
    const std::filesystem::path&,
    const std::filesystem::path&,
    const ResidentCohortLimits&,
    const ResidentSha256Fn,
    const std::function<void(size_t, const char*)>&) {
  LOG(ERROR) << "The cohort runner requires the dense-stereo library, "
                "which is built with CUDA support";
  return EXIT_FAILURE;
}

#endif  // COLMAP_CUDA_ENABLED

}  // namespace

inline int RunResidentCohortBank(
    const PatchMatchOptions& invocation_options,
    const std::filesystem::path& workspace_path,
    const std::string& workspace_format,
    const std::string& pmvs_option_name,
    const std::filesystem::path& config_path,
    const std::filesystem::path& forecast_path,
    const ResidentCohortLimits& limits,
    const ResidentSha256Fn sha,
    const std::function<void(size_t, const char*)>& observer = {}) {
  return RunResidentCohortBankImpl(invocation_options,
                                   workspace_path,
                                   workspace_format,
                                   pmvs_option_name,
                                   config_path,
                                   forecast_path,
                                   limits,
                                   sha,
                                   observer);
}

}  // namespace resident
}  // namespace mvs
}  // namespace colmap