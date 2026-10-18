// SPDX-License-Identifier: BSD-3-Clause
#pragma once

// Native consumer of the resident-cohort-forecast v1 wire contract: the
// DISTINCT cohort wire (frozen spec docs/RESIDENT_COHORT_WIRE_EXTENSION.md,
// photogram 9fb4957; canonical emitter/serializer is the sole Python side).
//
// This is a DUAL of that frozen spec, not a second emitter, estimator,
// registry or admission authority: the parser and gates below only
// RECOGNIZE and VERIFY; every executable decision is derived and compared
// against native state in the plan-boundary/executor surfaces that stay
// separate. The DISTINCT magic is structural: the accepted full-bank
// resident-forecast wire can never certify a cohort forecast and vice
// versa, with no version negotiation and no fullbank/capacity-token
// compatibility path.
//
// Wire-internal arithmetic the parser enforces (mirroring the canonical
// Python parser exactly, recomputed with checked additions):
//   * per-cohort complete ledger line equality (the ADMISSION UNIT):
//       ledger_host   = queue + bank_charge + normal charge + process
//                       + rolling
//       ledger_device = bank_charge + process + rolling
//   * rolling reserve must equal the SUM of the largest active_jobs member
//     private charges, INDEPENDENTLY PER SIDE (out-of-order owner
//     completion admits non-adjacent owners; contiguous locality is never
//     assumed) — a declared number is only provable by recomputation;
//   * active_jobs is the MANDATORY per-cohort admitted concurrency:
//     1 <= active_jobs <= member_count (0 and over-limit refuse; the CLI
//     cap comparison happens in the FitCaps gate, since caps are CLI-only
//     and must never appear on the wire);
//   * summary accounting sums must equal the per-cohort sums EXACTLY, and
//     preflight_max must equal the per-side MAX over the per-cohort
//     ledgers. The old aggregate `capacity host/device` token is REMOVED
//     from the frozen layout and is REFUSED if it appears (sums of
//     sequential cohort ledgers are never an admission quantity);
//   * the whole-invocation snapshot/plan basis is reserved ONCE inside
//     every cohort's queue charge (the persistent global metadata never
//     becomes a per-cohort subset): any cohort queue below the declared
//     global basis refuses;
//   * cohorts are contiguous increasing FIFO windows over the original
//     job ordinals, ordered, non-overlapping, covering every invocation
//     job exactly once; zero-member cohorts, duplicates, gaps and
//     reorderings refuse;
//   * bank_charge is the reviewed fixed R on any side where R is declared
//     (and reviewed_bank >= bank_estimate holds on both sides); otherwise
//     the canonical estimate is the charge; a mixed producer refuses;
//   * process is charged ONCE per active ledger (count == 1) with an
//     identical term in every cohort;
//   * the summary's aggregate terms are accounting only, never admission.
//
// EXECUTION IS NOT EXPOSED HERE. There is no CLI surface, no registry and
// no admission boolean: the provisional cohort label parses (it is a legal
// emitter output) but only the execution parameter gate accepts the
// executable label, which REQUIRES the native build's compiled cohort
// allocation fingerprint (a fresh, cohort-scoped closure scope; the
// full-bank 778c1c4 entry does NOT certify cohorts and no cohort closure
// is registered natively) and the actual binding/contract/identity/
// minima state of the integrating caller. Provisional wires are never
// execution-qualified. The wire's contract_sha256 + contract_raw bind the
// WHOLE invocation; the binding gate rejects anything the freshly
// regenerated native invocation does not reproduce byte for byte.

#include "colmap/mvs/resident_forecast.h"
#include "colmap/mvs/resident_wire_lexical.h"

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace colmap {
namespace mvs {
namespace cohort {

// Distinct cohort wire magic (never negotiates with `resident-forecast`).
inline constexpr char kCohortForecastMagic[] = "resident-cohort-forecast";
inline constexpr int kCohortForecastVersion = 1;

// Cohort labels only (a full-bank label never parses as a cohort label).
inline constexpr const char* kCohortQualificationProvisional =
    "metadata_only_cpu_estimate_cohort_provisional";
inline constexpr const char* kCohortQualificationExecutable =
    "metadata_only_cpu_estimate_cohort";

// One member block: an original invocation problem (bank job ordinal)
// with its complete private forecast charge. The per-reference fields
// intentionally repeat the full-bank wire's binding identity data so the
// binding gate can compare every member against the freshly resolved
// contract problems byte for byte.
struct CohortWireMember {
  size_t member_ordinal = 0;  // ORIGINAL invocation ordinal (never re-sorted)
  int reference_image_idx = -1;
  std::string reference_name;  // exact model-image name bytes
  uint64_t input_count = 0;
  uint64_t private_host_bytes = 0;
  uint64_t private_device_bytes = 0;
};

// One cohort block with its complete ledger (the admission unit).
struct CohortWireCohort {
  size_t ordinal = 0;
  size_t member_count = 0;
  std::vector<CohortWireMember> members;
  uint64_t bank_estimate_host_bytes = 0;
  uint64_t bank_estimate_device_bytes = 0;
  uint64_t bank_charge_host_bytes = 0;
  uint64_t bank_charge_device_bytes = 0;
  uint64_t reviewed_bank_host_bytes = 0;
  uint64_t reviewed_bank_device_bytes = 0;
  uint64_t normal_owner_bytes = 0;
  uint64_t normal_future_bytes = 0;
  uint64_t normal_transient_bytes = 0;
  uint64_t normal_charge_host_bytes = 0;
  uint64_t queue_host_bytes = 0;
  std::string process_mode;
  uint64_t process_count = 0;
  uint64_t process_host_bytes = 0;
  uint64_t process_device_bytes = 0;
  uint64_t planner_max_member_host_bytes = 0;
  uint64_t planner_max_member_device_bytes = 0;
  uint64_t rolling_active_jobs = 0;
  uint64_t rolling_host_bytes = 0;
  uint64_t rolling_device_bytes = 0;
  uint64_t ledger_host_bytes = 0;
  uint64_t ledger_device_bytes = 0;
};

// Accounting-only summary (never an admission quantity).
struct CohortWireSummary {
  uint64_t global_snapshot_plan_host_bytes = 0;
  uint64_t queue_sum_host_bytes = 0;
  uint64_t bank_sum_host_bytes = 0;
  uint64_t bank_sum_device_bytes = 0;
  uint64_t normal_sum_host_bytes = 0;
  uint64_t private_sum_host_bytes = 0;
  uint64_t private_sum_device_bytes = 0;
  uint64_t process_once_host_bytes = 0;
  uint64_t process_once_device_bytes = 0;
  // MAX over the per-cohort ledgers (the plan's sequential peak); the
  // aggregate `capacity` token does not exist in the frozen layout.
  uint64_t preflight_max_host_bytes = 0;
  uint64_t preflight_max_device_bytes = 0;
};

struct CohortForecastWire {
  int version = kCohortForecastVersion;
  // 64 lowercase hex chars, as carried on the wire, of the raw contract.
  std::string contract_sha256;
  int contract_version = 1;
  // The FULL-invocation raw native contract bytes, verbatim.
  std::string contract_raw;
  std::string qualification;
  std::string completeness;
  std::string concurrency_basis;
  std::string mode;
  int max_future = 0;
  int image_channels = 0;
  uint64_t unclassified_allowance_bytes = 0;
  std::string unclassified_provenance;
  bool has_build_allocation_fingerprint = false;
  // 64-character lowercase hex digest TEXT (never raw 32 bytes), or empty.
  std::string build_allocation_fingerprint;
  bool has_source_fingerprint = false;
  std::string source_fingerprint;
  uint64_t global_snapshot_plan_host_bytes = 0;
  size_t cohort_count = 0;
  size_t member_count_total = 0;
  std::vector<CohortWireCohort> cohorts;
  CohortWireSummary summary;
};

// Strict cohort-wire parser. Throws std::invalid_argument for schema
// violations and std::runtime_error for I/O-shaped problems, naming the
// offending line (mirroring the strict canonical Python parser; no
// version negotiation, no unknown keys, truncated/empty/padded lines and
// trailing content refuse). Reuses the shared strict forecast-wire
// lexical primitives (moved verbatim from the full-bank parser).
CohortForecastWire ParseResidentCohortForecast(
    const std::string& bytes, ResidentSha256Fn sha = nullptr);

// Binding gate: the wire must describe EXACTLY this whole invocation.
//  - `sha` is REQUIRED (the nullptr hash-optional parser path is never an
//    execution credential);
//  - `contract_raw` must equal the caller's freshly regenerated invocation
//    snapshot byte for byte, and its SHA must match contract_sha256;
//  - the cohort member blocks must cover EVERY resolved contract problem
//    exactly once, in order, and each member block must equal its problem
//    (reference_image_idx, exact reference-name bytes, input count).
void VerifyResidentCohortForecastBinding(
    const CohortForecastWire& wire,
    const std::string& fresh_contract_bytes,
    ResidentSha256Fn sha);

// Execution parameter gate (the only execution-facing gate; provisional
// wires are never execution-qualified):
//  - qualification must be EXACTLY the executable cohort label;
//  - completeness == complete; concurrency_basis == rolling_reserve_declared;
//  - mode bounded (the bounded FIFO queue is the only cohort execution
//    scope; eager readable but never executed);
//  - both fingerprints PRESENT, textually equal to each other and to the
//    compiled cohort allocation fingerprint of THIS native build (the
//    fresh cohort-scoped closure scope; there is no native registry, the
//    full-bank entry certifies nothing here, and absent fingerprints
//    refuse).
void VerifyResidentCohortForecastParameters(
    const CohortForecastWire& wire,
    const std::string& compiled_cohort_allocation_fingerprint);

// HARD FitsCaps gate (mandatory admitted concurrency; caps are CLI-only
// and never on the wire): every cohort's complete ledger must fit the
// caller's caps on BOTH sides INDEPENDENTLY (cohorts are strictly
// sequential and drained between; ledgers are never summed across
// cohorts), active_jobs already bounded by the parser to
// [1, member_count] and additionally not above the declared CLI
// concurrency of this run, and preflight_max must equal the MAX over the
// parsed ledgers (defense in depth; the parser re-derives it exactly).
// Any single unfunded ledger refuses the whole forecast: there is no
// dynamic throttle, degraded mode or informational reserve.
void VerifyResidentCohortForecastFitsCaps(const CohortForecastWire& wire,
                                          uint64_t host_budget_bytes,
                                          uint64_t device_budget_bytes,
                                          size_t cli_max_concurrency);

// NATIVE per-cohort charge floors + singleton feasibility (the missing
// production gate; requires the linked dense-stereo bank estimator; never
// stubbed and never compiled elsewhere). Before ANY payload, per cohort:
//   * the allowance-INCLUSIVE bank charge (canonical composition: the
//     wire-declared whole-invocation unclassified allowance composes with
//     the margin-free RAW per-cohort bank total BEFORE the SINGLE
//     canonical 3% margin; a separate allowance ledger term does not
//     exist) must be >= margin(cohort bounded host raw + allowance) /
//     margin(cohort device raw);
//   * the per-cohort normal owner/future/transient allowances must EQUAL
//     the native logical recompute for that cohort's member slice under
//     the wire's max_future, and the charge must cover future + transient;
//   * every member's private charge must be >= the native complete
//     private logical minimum (bounded mode includes the active owner
//     normal);
//   * EVERY member must be feasible ALONE (native singleton preflight):
//     the singleton ledger floor global_basis + process + singleton
//     allowance-inclusive bank floor + native singleton normal charge +
//     the member's wire private charge must fit the CLI caps on BOTH
//     sides.
// The declared whole-invocation snapshot/plan basis itself stays a
// wire-declared, provenance-backed allowance (queue >= basis is the
// parser gate); no native metadata formula exists and none is invented.
// The ORIGINAL invocation identity snapshot is revalidated (never
// recaptured) around every estimator traversal; the estimator floor
// reads the same immutable admitted inputs the executor will bind.
void VerifyResidentCohortForecastNativeFloors(
    const CohortForecastWire& wire,
    const std::vector<owned::BankJob>& invocation_jobs,
    const std::vector<owned::BankJobIdentities>& invocation_identities,
    uint64_t host_budget_bytes, uint64_t device_budget_bytes);

}  // namespace cohort
}  // namespace mvs
}  // namespace colmap