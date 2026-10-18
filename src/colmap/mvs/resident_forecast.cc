// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/mvs/resident_forecast.h"

#include "colmap/mvs/owned_resident_minima.h"
#include "colmap/mvs/resident_wire_lexical.h"
#include "colmap/mvs/resident_contract.h"
#include "colmap/util/logging.h"

#include <algorithm>
#include <array>
#include <charconv>
#include <cstdint>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace colmap {
namespace mvs {
namespace {

using resident_lexical::ForecastLexer;
using resident_lexical::ParseByteString;
using resident_lexical::ParseFingerprint;
using resident_lexical::ParseShaToken;
using resident_lexical::ParseSignedInt32;
using resident_lexical::ParseUint;
using resident_lexical::ParseUintMax;
using resident_lexical::ParseUintMinimum;
using resident_lexical::ValidateFingerprintText;
using resident_lexical::TryHexDecode;

constexpr int64_t kExpectedContractVersion = 1;

}  // namespace

ResidentForecastWire ParseResidentForecast(const std::string& bytes,
                                           ResidentSha256Fn sha) {
  if (bytes.empty()) {
    throw std::invalid_argument("empty resident forecast");
  }
  if (bytes.back() != '\n') {
    throw std::runtime_error(
        "Malformed resident forecast: unterminated final line");
  }
  ForecastLexer lexer(bytes);
  ResidentForecastWire wire;

  const auto next_key = [&lexer](const std::string& expected_key) {
    if (!lexer.Next()) {
      lexer.Fail("expected " + expected_key + ", got end of forecast");
    }
    if (lexer.Key() != expected_key) {
      lexer.Fail("expected " + expected_key + ", got \"" +
                 lexer.LineText() + "\"");
    }
  };

  next_key("resident-forecast");
  if (lexer.TokenCount() != 1) lexer.Fail("header field count");

  next_key("version");
  if (lexer.TokenCount() != 2) lexer.Fail("version field count");
  const uint64_t version =
      ParseUint(lexer, lexer.Token(1), "version");
  if (version != kResidentForecastVersion) {
    lexer.Fail("unsupported resident forecast version " +
               std::to_string(version));
  }
  wire.version = static_cast<int>(version);

  next_key("contract_sha256");
  if (lexer.TokenCount() != 2) lexer.Fail("contract_sha256 field count");
  wire.contract_sha256 = ParseShaToken(lexer, lexer.Token(1));

  next_key("contract_version");
  if (lexer.TokenCount() != 2) lexer.Fail("contract_version field count");
  const uint64_t contract_version =
      ParseUint(lexer, lexer.Token(1), "contract_version");
  if (contract_version != kExpectedContractVersion) {
    lexer.Fail("unsupported contract version binding " +
               std::to_string(contract_version));
  }

  next_key("contract_raw_length");
  if (lexer.TokenCount() != 2) lexer.Fail("contract_raw_length field count");
  const uint64_t raw_length = ParseUintMinimum(
      lexer, lexer.Token(1), "contract_raw_length", 1);

  next_key("contract_raw");
  if (lexer.TokenCount() != 3) lexer.Fail("contract_raw field count");
  const std::string raw = ParseByteString(lexer, "contract_raw", 1, 2, /*line_final=*/true);
  if (raw.size() != raw_length) {
    lexer.Fail("contract_raw length does not match contract_raw_length");
  }
  if (sha != nullptr) {
    const std::string digest = sha(raw);
    if (digest != wire.contract_sha256) {
      lexer.Fail("contract_raw does not hash to the bound contract_sha256");
    }
  }
  wire.contract_raw = raw;

  next_key("qualification");
  wire.qualification = ParseByteString(lexer, "qualification", 1, 2, /*line_final=*/true);
  next_key("completeness");
  wire.completeness = ParseByteString(lexer, "completeness", 1, 2, /*line_final=*/true);
  next_key("concurrency_basis");
  wire.concurrency_basis = ParseByteString(lexer, "concurrency_basis", 1, 2, /*line_final=*/true);
  next_key("mode");
  wire.mode = ParseByteString(lexer, "mode", 1, 2, /*line_final=*/true);

  next_key("max_future");
  if (lexer.TokenCount() != 2) lexer.Fail("max_future field count");
  ParseUintMinimum(lexer, lexer.Token(1), "max_future", 1);
  // Python bounds the field to a uint8 width with a minimum of 1.
  // Python bounds max_future/image_channels to unsigned 8-bit values.
  wire.max_future = static_cast<int>(ParseUintMax(
      lexer, lexer.Token(1), "max_future", 255));

  next_key("image_channels");
  if (lexer.TokenCount() != 2) lexer.Fail("image_channels field count");
  ParseUintMinimum(lexer, lexer.Token(1), "image_channels", 1);
  wire.image_channels = static_cast<int>(ParseUintMax(
      lexer, lexer.Token(1), "image_channels", 255));

  next_key("unclassified_allowance_bytes");
  if (lexer.TokenCount() != 4) {
    lexer.Fail("unclassified_allowance_bytes must carry value + provenance");
  }
  wire.unclassified_allowance_bytes =
      ParseUint(lexer, lexer.Token(1), "unclassified_allowance_bytes");
  wire.unclassified_provenance = ParseByteString(
      lexer, "unclassified_allowance_provenance", 2, 3, /*line_final=*/true);

  next_key("build_allocation_fingerprint");
  ParseFingerprint(lexer, "build_allocation_fingerprint", lexer.LineNo(),
                   &wire.has_build_allocation_fingerprint,
                   &wire.build_allocation_fingerprint);
  next_key("source_fingerprint");
  ParseFingerprint(lexer, "source_fingerprint", lexer.LineNo(),
                   &wire.has_source_fingerprint,
                   &wire.source_fingerprint);
  // Both fingerprints always agree (the emitter carries the same digest);
  // `absent` is legal only for provisional emission.
  if (wire.qualification == kResidentForecastExecutableQualification) {
    if (!wire.has_build_allocation_fingerprint ||
        !wire.has_source_fingerprint) {
      lexer.Fail("an executable qualification cannot carry absent "
                 "fingerprints");
    }
  }
  if (wire.has_build_allocation_fingerprint !=
      wire.has_source_fingerprint) {
    lexer.Fail("build_allocation_fingerprint and source_fingerprint must "
               "carry the same allocation-closure digest");
  }
  if (wire.has_build_allocation_fingerprint &&
      wire.build_allocation_fingerprint != wire.source_fingerprint) {
    lexer.Fail("build_allocation_fingerprint and source_fingerprint must "
               "carry the same allocation-closure digest");
  }

  next_key("charges");
  if (lexer.TokenCount() != 1) lexer.Fail("charges field count");

  next_key("bank");
  if (lexer.TokenCount() != 5 || lexer.Token(1) != "host" ||
      lexer.Token(3) != "device") {
    lexer.Fail("bank charge field layout");
  }
  wire.bank_host_bytes = ParseUint(lexer, lexer.Token(2), "bank host");
  wire.bank_device_bytes = ParseUint(lexer, lexer.Token(4), "bank device");

  next_key("normal");
  if (lexer.TokenCount() != 9 || lexer.Token(1) != "owner" ||
      lexer.Token(3) != "future" || lexer.Token(5) != "transient" ||
      lexer.Token(7) != "charge") {
    lexer.Fail("normal charge field layout");
  }
  wire.normal_owner_bytes = ParseUint(lexer, lexer.Token(2), "normal owner");
  wire.normal_future_bytes =
      ParseUint(lexer, lexer.Token(4), "normal future");
  wire.normal_transient_bytes =
      ParseUint(lexer, lexer.Token(6), "normal transient");
  wire.normal_charge_bytes =
      ParseUint(lexer, lexer.Token(8), "normal charge");

  next_key("process");
  if (lexer.TokenCount() != 10 || lexer.Token(1) != "mode" ||
      lexer.Token(4) != "count" || lexer.Token(6) != "host" ||
      lexer.Token(8) != "device") {
    lexer.Fail("process charge field layout");
  }
  wire.process_mode = ParseByteString(lexer, "process mode", 2, 3, /*line_final=*/false);
  wire.process_count = ParseUint(lexer, lexer.Token(5), "process count");
  wire.process_host_bytes =
      ParseUint(lexer, lexer.Token(7), "process host");
  wire.process_device_bytes =
      ParseUint(lexer, lexer.Token(9), "process device");

  next_key("metadata");
  if (lexer.TokenCount() != 8 || lexer.Token(1) != "source" ||
      lexer.Token(4) != "raw" || lexer.Token(6) != "charge") {
    lexer.Fail("metadata charge field layout");
  }
  wire.metadata_source = ParseByteString(lexer, "metadata source", 2, 3, /*line_final=*/false);
  wire.metadata_raw_bytes =
      ParseUint(lexer, lexer.Token(5), "metadata raw");
  wire.metadata_charge_bytes =
      ParseUint(lexer, lexer.Token(7), "metadata charge");

  next_key("planner");
  if (lexer.TokenCount() != 6 || lexer.Token(1) != "max_member" ||
      lexer.Token(2) != "host" || lexer.Token(4) != "device") {
    lexer.Fail("planner charge field layout");
  }
  wire.planner_max_member_host_bytes =
      ParseUint(lexer, lexer.Token(3), "planner host");
  wire.planner_max_member_device_bytes =
      ParseUint(lexer, lexer.Token(5), "planner device");

  next_key("rolling");
  if (lexer.TokenCount() != 7 || lexer.Token(1) != "active_jobs" ||
      lexer.Token(3) != "host" || lexer.Token(5) != "device") {
    lexer.Fail("rolling charge field layout");
  }
  wire.rolling_active_jobs =
      ParseUint(lexer, lexer.Token(2), "rolling active_jobs");
  wire.rolling_host_bytes =
      ParseUint(lexer, lexer.Token(4), "rolling host");
  wire.rolling_device_bytes =
      ParseUint(lexer, lexer.Token(6), "rolling device");

  next_key("reference_count");
  if (lexer.TokenCount() != 2) lexer.Fail("reference_count field count");
  // Bounded before any reserve(): each reference block needs far more than
  // 32 bytes on the wire.
  const uint64_t reference_count = ParseUint(lexer, lexer.Token(1),
                                             "reference_count");
  if (reference_count > lexer.RemainingBytes() / 64) {
    lexer.Fail("declared reference_count exceeds the remaining forecast "
               "size");
  }
  wire.references.reserve(static_cast<size_t>(reference_count));

  for (size_t i = 0; i < reference_count; ++i) {
    next_key("reference");
    if (lexer.TokenCount() != 2) lexer.Fail("reference field count");
    const uint64_t ordinal =
        ParseUint(lexer, lexer.Token(1), "reference ordinal");
    if (ordinal != i) {
      lexer.Fail("reference ordinals must be sequential");
    }
    ResidentForecastReference reference;
    next_key("reference_image_idx");
    if (lexer.TokenCount() != 2) {
      lexer.Fail("reference_image_idx field count");
    }
    const int64_t image_idx = ParseSignedInt32(
        lexer, lexer.Token(1), "reference_image_idx");
    if (image_idx < 0) {
      lexer.Fail("negative reference_image_idx");
    }
    reference.reference_image_idx = static_cast<int>(image_idx);
    next_key("reference_name");
    if (lexer.TokenCount() != 3) lexer.Fail("reference_name field count");
    reference.reference_name =
        ParseByteString(lexer, "reference_name", 1, 2, /*line_final=*/true);
    next_key("input_count");
    if (lexer.TokenCount() != 2) lexer.Fail("input_count field count");
    reference.input_count = ParseUintMinimum(
        lexer, lexer.Token(1), "input_count", 1);
    next_key("private_host_bytes");
    if (lexer.TokenCount() != 2) lexer.Fail("private_host_bytes field count");
    reference.private_host_bytes =
        ParseUint(lexer, lexer.Token(1), "private_host_bytes");
    next_key("private_device_bytes");
    if (lexer.TokenCount() != 2) {
      lexer.Fail("private_device_bytes field count");
    }
    reference.private_device_bytes =
        ParseUint(lexer, lexer.Token(1), "private_device_bytes");
    wire.references.push_back(std::move(reference));
  }

  next_key("end");
  if (lexer.TokenCount() != 1) lexer.Fail("end field count");
  if (lexer.Next()) {
    lexer.Fail("trailing content after the forecast");
  }
  return wire;
}

ResidentForecastNormals ResidentForecastNormalsMode(
    const ResidentForecastWire& wire) {
  if (wire.mode == kResidentForecastModeBounded) {
    return ResidentForecastNormals::kBounded;
  }
  if (wire.mode == kResidentForecastModeEager) {
    return ResidentForecastNormals::kEager;
  }
  throw std::invalid_argument(
      "resident forecast declares unknown bank normal mode: " + wire.mode);
}

void VerifyResidentForecastBinding(
    const ResidentForecastWire& wire,
    const std::string& fresh_contract_bytes,
    const ResidentSha256Fn sha) {
  // A real hash function is REQUIRED for execution; nullptr refuses (the
  // parser-only path stays hash-optional and documented as such).
  THROW_CHECK(sha != nullptr)
      << "Resident forecast execution requires actual SHA-256 verification; "
         "a build without a crypto backend refuses execution";
  THROW_CHECK_EQ(sha(fresh_contract_bytes), wire.contract_sha256)
      << "Resident forecast binding: the regenerated invocation snapshot "
         "does not hash to the wire's contract digest";
  THROW_CHECK_EQ(wire.contract_raw, fresh_contract_bytes)
      << "Resident forecast binding: the forecast's raw contract bytes do "
         "not equal the freshly regenerated invocation snapshot";
  // The wire's reference blocks must equal the freshly resolved contract
  // problems exactly (count, order, index, exact name bytes, input count).
  const auto contract = ParseOwnedBankContract(fresh_contract_bytes);
  THROW_CHECK_EQ(wire.references.size(), contract.problems.size())
      << "Resident forecast binding: reference count and resolved problem "
         "count differ";
  for (size_t i = 0; i < wire.references.size(); ++i) {
    const auto& problem = contract.problems.at(i);
    THROW_CHECK_EQ(wire.references.at(i).reference_image_idx,
                   problem.reference_image_idx)
        << "Resident forecast binding: reference image index changed at "
        << i;
    THROW_CHECK_EQ(wire.references.at(i).reference_name,
                   problem.reference_name)
        << "Resident forecast binding: reference name bytes changed at "
        << i;
    THROW_CHECK_EQ(static_cast<uint64_t>(wire.references.at(i).input_count),
                   static_cast<uint64_t>(problem.inputs.size()))
        << "Resident forecast binding: input count changed at " << i;
  }
}

void VerifyResidentForecastParameters(
    const ResidentForecastWire& wire,
    const std::string& compiled_allocation_fingerprint) {
  THROW_CHECK_EQ(wire.qualification, kResidentForecastExecutableQualification)
      << "Resident forecast execution requires the executable "
         "qualification; provisional or unknown labels are refused";
  THROW_CHECK_EQ(wire.completeness, kResidentForecastComplete)
      << "Resident forecast execution refuses partial forecasts";
  THROW_CHECK_EQ(wire.concurrency_basis, kResidentForecastRollingDeclared)
      << "Resident forecast execution refuses bank-only scopes";
  // Bounded-only first execution gate: eager wires are correctly READ but
  // never executed; Python's eager accounting is untouched.
  THROW_CHECK_EQ(wire.mode, kResidentForecastModeBounded)
      << "Resident forecast execution is bounded-only; eager declarations "
         "refuse (unrecognized mode labels also refuse)";
  THROW_CHECK(wire.has_build_allocation_fingerprint &&
              wire.has_source_fingerprint)
      << "Resident forecast execution refuses absent fingerprints";
  THROW_CHECK_EQ(wire.build_allocation_fingerprint,
                 wire.source_fingerprint)
      << "Resident forecast fingerprints disagree";
  THROW_CHECK_EQ(wire.build_allocation_fingerprint,
                 compiled_allocation_fingerprint)
      << "Resident forecast allocation fingerprint does not match this "
         "build's compiled closure; unknown build sources are unqualified";
  // The wire's declared normal mode is the mode the bank is built with.
  ResidentForecastNormalsMode(wire);
}

void VerifyResidentForecastLedgerTerms(
    const ResidentForecastWire& wire,
    const size_t max_concurrent_solvers) {
  THROW_CHECK(wire.max_future == 1 || wire.max_future == 2)
      << "Resident forecast max_future must be the bounded FIFO queue "
         "contract value 1 or 2, got " << wire.max_future;
  THROW_CHECK_GE(wire.image_channels, 1);

  THROW_CHECK_GE(wire.rolling_active_jobs, 1)
      << "Resident forecast execution refuses a bank-only scope";
  THROW_CHECK_EQ(wire.rolling_active_jobs, max_concurrent_solvers)
      << "Resident forecast declared concurrency does not match the "
         "invocation's CLI concurrency";
  THROW_CHECK_LE(wire.rolling_active_jobs,
                 static_cast<uint64_t>(wire.references.size()))
      << "Resident forecast active_jobs exceeds the declared references";

  // Wire-internal ledger arithmetic, recomputed with checked operations:
  // planner max-member and the rolling reserve (sum of the largest
  // active_jobs private charges).
  uint64_t max_private_host = 0;
  uint64_t max_private_device = 0;
  std::vector<uint64_t> hosts;
  std::vector<uint64_t> devices;
  hosts.reserve(wire.references.size());
  devices.reserve(wire.references.size());
  for (const auto& reference : wire.references) {
    max_private_host = std::max(max_private_host, reference.private_host_bytes);
    max_private_device = std::max(max_private_device,
                                  reference.private_device_bytes);
    hosts.push_back(reference.private_host_bytes);
    devices.push_back(reference.private_device_bytes);
  }
  THROW_CHECK_EQ(wire.planner_max_member_host_bytes, max_private_host)
      << "Resident forecast planner max-member host charge disagrees with "
         "the declared private charges";
  THROW_CHECK_EQ(wire.planner_max_member_device_bytes, max_private_device)
      << "Resident forecast planner max-member device charge disagrees "
         "with the declared private charges";
  std::sort(hosts.begin(), hosts.end(),
            [](uint64_t a, uint64_t b) { return a > b; });
  std::sort(devices.begin(), devices.end(),
            [](uint64_t a, uint64_t b) { return a > b; });
  uint64_t rolling_host = 0;
  uint64_t rolling_device = 0;
  const size_t active = std::min<size_t>(
      static_cast<size_t>(wire.rolling_active_jobs), hosts.size());
  for (size_t i = 0; i < active; ++i) {
    rolling_host = owned::CheckedAdd(rolling_host, hosts.at(i));
    rolling_device = owned::CheckedAdd(rolling_device, devices.at(i));
  }
  THROW_CHECK_EQ(wire.rolling_host_bytes, rolling_host)
      << "Resident forecast rolling host reserve disagrees with the "
         "declared private charges";
  THROW_CHECK_EQ(wire.rolling_device_bytes, rolling_device)
      << "Resident forecast rolling device reserve disagrees with the "
         "declared private charges";

  // Process shape: exactly one resident process for this CLI run.
  THROW_CHECK_EQ(wire.process_mode, kResidentForecastProcessOnce)
      << "Resident forecast process charge must describe this single "
         "process ('once', count 1)";
  THROW_CHECK_EQ(wire.process_count, 1)
      << "Resident forecast process charge must describe this single "
         "process ('once', count 1)";
  THROW_CHECK_GE(wire.process_host_bytes, 2147483648ull)
      << "Resident forecast process host charge must cover the canonical "
         "2 GiB process floor";
  THROW_CHECK_GE(wire.process_device_bytes, 2147483648ull)
      << "Resident forecast process device charge must cover the canonical "
         "2 GiB process floor";

  // Metadata terms: omitted metadata must declare zero raw/charge and is
  // the only source that may appear on a partial forecast; complete
  // forecasts must declare a non-omitted source; non-omitted sources must
  // have a charge of exactly the single canonical 3% margin of raw.
  const bool omitted = wire.metadata_source == "omitted_declared";
  if (omitted) {
    THROW_CHECK_EQ(wire.metadata_raw_bytes, 0u)
        << "Resident forecast declares omitted metadata with a nonzero raw "
           "term";
    THROW_CHECK_EQ(wire.metadata_charge_bytes, 0u)
        << "Resident forecast declares omitted metadata with a nonzero "
           "charge";
    THROW_CHECK_EQ(wire.completeness, "partial")
        << "Resident forecast declares omitted metadata without the "
           "matching partial label";
  } else {
    THROW_CHECK(wire.metadata_source == "caller_supplied_native_actual" ||
              wire.metadata_source == "explicit_unclassified")
        << "Resident forecast declares an unknown metadata source";
    THROW_CHECK_EQ(wire.completeness, kResidentForecastComplete)
        << "Resident forecast partial completeness requires the omitted "
           "metadata declaration";
    THROW_CHECK_EQ(wire.metadata_charge_bytes,
                   owned::MarginBytes(wire.metadata_raw_bytes))
        << "Resident forecast metadata charge departs from the canonical "
           "single 3% margin";
  }
}

void VerifyResidentForecastCharges(const ResidentForecastWire& wire,
                                   const ResidentLedgerMinima& minima) {
  THROW_CHECK_EQ(minima.private_minima.size(), wire.references.size())
      << "Resident forecast ledger minima must cover every reference";
  // Bounded-only execution (first gate): refuses eager here as well; the
  // parser still reads eager wires, but no eager-mode charges exist.
  THROW_CHECK(ResidentForecastNormalsMode(wire) ==
              ResidentForecastNormals::kBounded)
      << "Resident forecast execution is bounded-only; eager declarations "
         "refuse before payload";
  // The allowance provenance is mandatory, even for a zero allowance.
  THROW_CHECK(!wire.unclassified_provenance.empty())
      << "Resident forecast unclassified allowance provenance is mandatory "
         "(including an explicit declared zero)";
  // Bank charges: the allowance composes with the margin-free RAW total
  // BEFORE the single canonical 3% margin, exactly as the canonical
  // estimator charges it; each margin is applied once (never re-margined).
  const uint64_t raw_with_allowance = owned::CheckedAdd(
      minima.bank_host_raw_bytes, wire.unclassified_allowance_bytes);
  THROW_CHECK_GE(wire.bank_host_bytes, owned::MarginBytes(raw_with_allowance))
      << "Resident forecast bank host charge is below the native "
         "allowance-composed raw minimum for this invocation";
  THROW_CHECK_GE(wire.bank_device_bytes,
                 owned::MarginBytes(minima.bank_device_raw_bytes))
      << "Resident forecast bank device charge is below the native raw "
         "minimum for this invocation";
  THROW_CHECK_EQ(wire.normal_owner_bytes, minima.normal_owner_bytes)
      << "Resident forecast normal owner allowance disagrees with the "
         "native logical recompute";
  THROW_CHECK_EQ(wire.normal_future_bytes, minima.normal_future_bytes)
      << "Resident forecast normal future allowance disagrees with the "
         "native logical recompute";
  THROW_CHECK_EQ(wire.normal_transient_bytes, minima.normal_transient_bytes)
      << "Resident forecast normal transient allowance disagrees with the "
         "native logical recompute";
  const uint64_t required = owned::CheckedAdd(
      minima.normal_future_bytes, minima.normal_transient_bytes);
  THROW_CHECK_GE(wire.normal_charge_bytes, required)
      << "Resident forecast normal charge is below the required "
         "future+transient allowance";
  for (size_t i = 0; i < wire.references.size(); ++i) {
    const auto& reference = wire.references.at(i);
    const auto& minimum = minima.private_minima.at(i);
    THROW_CHECK_GE(reference.private_host_bytes, minimum[0])
        << "Resident forecast private host charge is below the native "
           "logical minimum for reference " << i;
    THROW_CHECK_GE(reference.private_device_bytes, minimum[1])
        << "Resident forecast private device charge is below the native "
           "logical minimum for reference " << i;
  }
}

std::vector<owned::BankJobIdentities> ContractJobIdentities(
    const OwnedBankContract& contract) {
  // Exact ORIGINAL records: no filesystem access, no recapture, per
  // problem/input order, role order (image, depth, normal).
  std::vector<owned::BankJobIdentities> snapshot;
  snapshot.reserve(contract.problems.size());
  for (const auto& problem : contract.problems) {
    owned::BankJobIdentities entries;
    entries.reserve(problem.inputs.size());
    for (const auto& input : problem.inputs) {
      entries.push_back(owned::BankLayerIdentity{input.image_identity,
                                                 input.depth_identity,
                                                 input.normal_identity});
    }
    snapshot.push_back(std::move(entries));
  }
  return snapshot;
}

void CheckContractIdentitiesUnchanged(
    const std::vector<owned::BankJobIdentities>& snapshot) {
  for (const auto& entries : snapshot) {
    for (const auto& identity : entries) {
      owned::CheckFileIdentityUnchanged(identity.image);
      owned::CheckFileIdentityUnchanged(identity.depth);
      owned::CheckFileIdentityUnchanged(identity.normal);
    }
  }
}

void ResidentEventSink(const bool cancelled_event,
                       const std::function<void()>& cancel,
                       const std::function<void(size_t, const char*)>*
                           observer,
                       const size_t job,
                       const char* event) {
  // MANDATORY ORDER: cancellation first; a throwing observer (whose
  // exception the coordinator's guarded sink swallows) can never disable
  // the producer cancellation that unblocks the consumer waits.
  if (cancelled_event) {
    cancel();
  }
  if (observer != nullptr && *observer) {
    (*observer)(job, event);
  }
}

void VerifyResidentForecastFitsCaps(const ResidentForecastWire& wire,
                                    const uint64_t host_budget_bytes,
                                    const uint64_t device_budget_bytes) {
  // Complete checked aggregate ledger, NO double billing:
  //   host   = bank + process + metadata + normal queue + rolling reserve
  //   device = bank + process + rolling reserve
  // (the planner max-member terms are consistency checks, never additions).
  const uint64_t host_total = owned::CheckedAdd(
      owned::CheckedAdd(wire.bank_host_bytes, wire.process_host_bytes),
      owned::CheckedAdd(
          owned::CheckedAdd(wire.metadata_charge_bytes,
                            wire.normal_charge_bytes),
          wire.rolling_host_bytes));
  const uint64_t device_total = owned::CheckedAdd(
      owned::CheckedAdd(wire.bank_device_bytes, wire.process_device_bytes),
      wire.rolling_device_bytes);
  THROW_CHECK_LE(host_total, host_budget_bytes)
      << "Resident forecast host ledger total " << host_total
      << " exceeds the declared host budget cap " << host_budget_bytes;
  THROW_CHECK_LE(device_total, device_budget_bytes)
      << "Resident forecast device ledger total " << device_total
      << " exceeds the declared device budget cap " << device_budget_bytes;
}

}  // namespace mvs
}  // namespace colmap