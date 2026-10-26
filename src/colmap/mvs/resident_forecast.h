// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "colmap/mvs/owned_input_bank.h"
#include "colmap/mvs/resident_contract.h"

#include <array>
#include <cstdint>
#include <functional>
#include <string>
#include <string_view>
#include <vector>

namespace colmap {
namespace mvs {

// Native consumer of the resident-forecast v1 wire contract (line protocol),
// emitted verbatim by the canonical Python serializer (photogram commit
// d155346, docs/RESIDENT_FORECAST_WIRE_PROPOSAL.md). This is a DUAL of that
// fixed schema, not a second registry, estimator or admission authority:
// the parser and gates below only RECOGNIZE and VERIFY; every executable
// decision is derived and compared against native state (compiled allocation
// fingerprint, freshly resolved contract bytes, logical minima, CLI caps).
//
// EXECUTION IS BOUNDED-ONLY (first gate): the parser correctly reads eager
// wires (they are legal emitter outputs) but every executable gate refuses
// an eager declaration; the bounded normal FIFO producer is the only
// first-gate execution scope. Python's eager accounting is untouched.
//
// Wire grammar (exact; mirrors the committed serializer):
//   fixed key/order; ASCII; '\n'-terminated final line; single-space tokens;
//   no empty/padded lines; canonical decimal integers (no signs beyond the
//   int32 fields' optional '-', no leading zeros); every string is
//   (decimal byte length, lowercase hex bytes); `contract_sha256` is a bare
//   64-char lowercase hex token; fingerprints are the literal token `absent`
//   or (64, hex of the 64-char digest TEXT). Integers below the Python
//   parser's declared bit widths are rejected identically:
//   unsigned 0..2^64-1 (with the named minima), max_future/image_channels
//   1..255, int32 sign-capable fields as in the contract.
constexpr int kResidentForecastVersion = 1;

// ComputeSHA256-compatible hashing hook (std::string(const std::string_view&)).
using ResidentSha256Fn = std::string (*)(const std::string_view&);

inline constexpr const char* kResidentForecastExecutableQualification =
    "metadata_only_cpu_estimate";
inline constexpr const char* kResidentForecastProvisionalQualification =
    "metadata_only_cpu_estimate_provisional";
inline constexpr const char* kResidentForecastComplete = "complete";
inline constexpr const char* kResidentForecastRollingDeclared =
    "rolling_reserve_declared";
inline constexpr const char* kResidentForecastModeEager = "eager";
inline constexpr const char* kResidentForecastModeBounded = "bounded";
inline constexpr const char* kResidentForecastProcessOnce = "once";

// Effective bank normal mode declared by the wire.
enum class ResidentForecastNormals { kEager, kBounded };

// One per-reference charge record, in wire order (== native contract
// problem order, verified by the binding gate).
struct ResidentForecastReference {
  int reference_image_idx = -1;
  // Exact bytes of the model image name (never UTF-8 gated or re-ordered).
  std::string reference_name;
  uint64_t input_count = 0;
  uint64_t private_host_bytes = 0;
  uint64_t private_device_bytes = 0;
};

struct ResidentForecastWire {
  int version = kResidentForecastVersion;
  // 64 lowercase hex chars, as carried on the wire, of the raw contract bytes.
  std::string contract_sha256;
  // The raw native `owned-bank-contract` V1 bytes, verbatim.
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
  // 64-character lowercase hex Digest TEXT (not raw 32 bytes), or empty.
  std::string build_allocation_fingerprint;
  bool has_source_fingerprint = false;
  std::string source_fingerprint;
  uint64_t bank_host_bytes = 0;
  uint64_t bank_device_bytes = 0;
  uint64_t normal_owner_bytes = 0;
  uint64_t normal_future_bytes = 0;
  uint64_t normal_transient_bytes = 0;
  uint64_t normal_charge_bytes = 0;
  std::string process_mode;
  uint64_t process_count = 0;
  uint64_t process_host_bytes = 0;
  uint64_t process_device_bytes = 0;
  std::string metadata_source;
  uint64_t metadata_raw_bytes = 0;
  uint64_t metadata_charge_bytes = 0;
  uint64_t planner_max_member_host_bytes = 0;
  uint64_t planner_max_member_device_bytes = 0;
  uint64_t rolling_active_jobs = 0;
  uint64_t rolling_host_bytes = 0;
  uint64_t rolling_device_bytes = 0;
  std::vector<ResidentForecastReference> references;
};

// Strict wire parser. Throws std::invalid_argument (or std::runtime_error
// for I/O-shaped problems) naming the offending line, mirroring the strict
// Python parser: unsupported versions, missing/unknown/out-of-order keys,
// wrong field counts, non-canonical or out-of-range integers, malformed or
// oversized hex strings, truncated/empty/padded lines, trailing content,
// executable qualification with absent or unequal fingerprints.
ResidentForecastWire ParseResidentForecast(const std::string& bytes,
                                           ResidentSha256Fn sha = nullptr);

// Native bank normal mode after the parser/gates have checked the label.
ResidentForecastNormals ResidentForecastNormalsMode(
    const ResidentForecastWire& wire);

// Native logical minima of the ledger terms, recomputed from the runner's
// OWN resolved invocation state (never trusted from the wire). Requires the
// fully linked dense-stereo library (the bank estimator opens image headers
// via OpenImageIO); never compiled or faked elsewhere. Bounded-only: the
// eager label (legal on the wire) refuses at this surface.
struct ResidentLedgerMinima {
  // Margin-free bank raw for this invocation's mode (bounded host raw, i.e.
  // eager raw minus retained reference normals exactly once) from the real
  // native estimator. The caller's explicit unclassified allowance is added
  // to THIS term before one canonical margin is applied; never re-margined.
  uint64_t bank_host_raw_bytes = 0;
  uint64_t bank_device_raw_bytes = 0;
  // Margined, allowance-free bank minimum (single 3% margin, as Estimate
  // applies it); a wire charge below this refuses regardless of allowance.
  uint64_t bank_host_bytes = 0;
  uint64_t bank_device_bytes = 0;
  uint64_t normal_owner_bytes = 0;
  uint64_t normal_future_bytes = 0;
  uint64_t normal_transient_bytes = 0;
  // Per reference, in native problem order: {host, device} logical minima.
  std::vector<std::array<uint64_t, 2>> private_minima;
};

ResidentLedgerMinima ComputeResidentLedgerMinima(
    const std::vector<owned::BankJob>& jobs,
    ResidentForecastNormals mode,
    size_t max_future);

// Binding gate: the wire must describe EXACTLY this invocation.
//  - `sha` is REQUIRED for execution: a nullptr (a build without a crypto
//    backend) refuses; the parser-only path stays hash-optional and clearly
//    documents that a declined hash is never an execution credential. The
//    declared `contract_sha256` must equal SHA-256 of the raw bytes and of
//    the freshly regenerated snapshot (ComputeSHA256 passes here);
//  - `contract_raw` must equal the caller's FRESHLY regenerated serialized
//    invocation snapshot byte for byte (a regenerated serialization is the
//    binding promise on the native side; a hash alone is not one);
//  - every reference block must equal the freshly resolved contract
//    problems (count, order, reference_image_idx, exact name bytes);
//  - the raw bytes must parse AND re-serialize canonically (rejected if the
//    stored contract was not canonical).
void VerifyResidentForecastBinding(const ResidentForecastWire& wire,
                                   const std::string& fresh_contract_bytes,
                                   ResidentSha256Fn sha);

// Parameter gate: executable-only shape plus fingerprint identity and the
// BOUNDED-ONLY first execution gate.
//  - qualification must be exactly the executable label (provisional
//    forecasts are explicitly non-executable);
//  - completeness == complete; concurrency_basis == rolling_reserve_declared;
//    mode must be bounded (the parser still reads eager wires; execution
//    refuses them rather than rewriting Python's eager accounting; unknown
//    labels refused);
//  - both fingerprints PRESENT, textually equal to each other and to the
//    compiled allocation fingerprint of THIS build (generated at build time
//    by scripts/resident/generate_resident_allocation_fingerprint.py; there
//    is no native registry).
void VerifyResidentForecastParameters(
    const ResidentForecastWire& wire,
    const std::string& compiled_allocation_fingerprint);

// Wire-internal ledger consistency: planner max_member == max of the
// per-reference privates; rolling host/device == sum of the largest
// active_jobs privates (overflow-checked); 1 <= active_jobs <=
// reference_count and active_jobs == `max_concurrent_solvers` (the actual
// CLI concurrency, not a wire-declared value); process mode/count/host/device
// must describe a single resident process for this run (mode 'once',
// count 1, positive charges) with the canonical 2 GiB floor per side
// (2147483648); metadata sources must be the ACTUAL canonical labels
// (caller_supplied_native_actual or explicit_unclassified; omitted_declared
// only on partial forecasts), with charges of exactly the single 3% margin
// of the declared raw; max_future/image channels validated here for gate
// independence from caller ordering. The planner max terms are consistency
// checks only: they are NEVER added into any budget total.
void VerifyResidentForecastLedgerTerms(const ResidentForecastWire& wire,
                                       size_t max_concurrent_solvers);

// Charge gate: every required charge must at least cover the native
// raw/margined minimum; the term values the native side can recompute
// deterministically must match exactly. Bounded-only execution: an eager
// declaration refuses here as well (no eager-mode charge rules exist).
//  - allowance provenance is MANDATORY (nonempty bytes even for a zero
//    allowance);
//  - bank host: wire charge >= MarginBytes(bounded host RAW + the declared
//    unclassified allowance); bank device: >= MarginBytes(device RAW).
//    The allowance is composed into the raw BEFORE the single canonical 3%
//    margin (exactly the emitter's composition) and never re-margined;
//  - normal owner/future/transient: EXACT equality with the recomputed
//    allowance for the wire-declared max_future, and the charge >=
//    future + transient (checked overflow);
//  - every per-reference private charge >= the native logical minimum for
//    that reference (bounded mode includes the active owner normal).
void VerifyResidentForecastCharges(const ResidentForecastWire& wire,
                                   const ResidentLedgerMinima& minima);

// Aggregate budget gate (the complete checked admission arithmetic): the
// declared charges must fit the CALLER's caps per side, with checked
// additions and NO double billing:
//   host   total = bank host + process host + metadata charge
//                  + normal charge + rolling host reserve
//   device total = bank device + process device + rolling device reserve
// (metadata charge and the bounded normal producer charge are host-side
// dynamic allocations; the planner max-member terms are consistency checks,
// never budget additions). Overflow refuses; any total above its cap is a
// refusal BEFORE payload, independent of every other gate.
void VerifyResidentForecastFitsCaps(const ResidentForecastWire& wire,
                                    uint64_t host_budget_bytes,
                                    uint64_t device_budget_bytes);

// ORIGINAL-record binding helpers (Sol 3cdc fix 4). The forecast binds the
// identity records the contract CAPTURED at resolution time; the runner's
// bank identity snapshot must be built FROM THOSE ORIGINAL RECORDS -
// never from a later recapture - and the originals must be freshly
// revalidated around every estimator traversal, so a same-shaped input
// replacement between resolution and payload can never be accepted.

// The invocation identity snapshot exactly as stored in the accepted
// contract: one record per admitted input, per problem, in problem/input
// order (image, depth, normal). No filesystem access: pure original-copy.
std::vector<owned::BankJobIdentities> ContractJobIdentities(
    const OwnedBankContract& contract);

// Fresh filesystem revalidation of the ORIGINAL bound identities (throws
// on any changed object, size, timestamps, or disappearance). Called
// before and after every estimator traversal and before bank
// construction; the bank additionally validates through its own
// snapshot-bound reads without ever recapturing.
void CheckContractIdentitiesUnchanged(
    const std::vector<owned::BankJobIdentities>& snapshot);

// Mandatory-order resident failure/telemetry sink: producer cancellation
// is published BEFORE any optional observer callback, so a throwing (and
// therefore coordinator-swallowed) observer can never disable cancellation
// (the pipeline's consumer waits have no timeout). `observer` may be null
// or empty (observer optional). Cancellation is observer-independent.
void ResidentEventSink(const bool cancelled_event,
                       const std::function<void()>& cancel,
                       const std::function<void(size_t, const char*)>* observer,
                       const size_t job,
                       const char* event);

}  // namespace mvs
}  // namespace colmap