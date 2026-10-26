// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/mvs/resident_cohort_wire.h"

#include "colmap/mvs/owned_pair_validation.h"
#include "colmap/mvs/resident_contract.h"
#include "colmap/util/logging.h"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

namespace colmap {
namespace mvs {
namespace cohort {
namespace {

using resident_lexical::ForecastLexer;
using resident_lexical::ParseByteString;
using resident_lexical::ParseFingerprint;
using resident_lexical::ParseShaToken;
using resident_lexical::ParseSignedInt32;
using resident_lexical::ParseUint;
using resident_lexical::ParseUintMax;
using resident_lexical::ParseUintMinimum;
using resident_lexical::TryHexDecode;
using resident_lexical::ValidateFingerprintText;

constexpr uint64_t kExpectedContractVersion = 1;

uint64_t CheckedSum(const char* const what,
                    const uint64_t a,
                    const uint64_t b,
                    const ForecastLexer& lexer) {
  try {
    return owned::CheckedAdd(a, b);
  } catch (const std::overflow_error&) {
    lexer.Fail(std::string(what) +
               " overflows uint64 (refusing, never "
               "wrapping)");
  }
}

}  // namespace

CohortForecastWire ParseResidentCohortForecast(const std::string& bytes,
                                               const ResidentSha256Fn sha) {
  if (bytes.empty()) {
    throw std::invalid_argument("empty cohort forecast");
  }
  if (bytes.back() != '\n') {
    throw std::runtime_error(
        "Malformed cohort forecast: unterminated final line");
  }
  ForecastLexer lexer(bytes);
  const auto next_key = [&lexer](const std::string& expected_key) {
    if (!lexer.Next()) {
      lexer.Fail("expected " + expected_key + ", got end of forecast");
    }
    if (lexer.Key() != expected_key) {
      lexer.Fail("expected " + expected_key + ", got \"" + lexer.LineText() +
                 "\"");
    }
  };

  CohortForecastWire wire;

  next_key(kCohortForecastMagic);
  if (lexer.TokenCount() != 1) {
    lexer.Fail("header field count");
  }
  next_key("version");
  if (lexer.TokenCount() != 2) lexer.Fail("version field count");
  const uint64_t version = ParseUint(lexer, lexer.Token(1), "version");
  if (version != static_cast<uint64_t>(kCohortForecastVersion)) {
    lexer.Fail("unsupported cohort forecast version " +
               std::to_string(version) +
               " (no version negotiation with the "
               "full-bank wire)");
  }
  wire.version = static_cast<int>(version);

  next_key("contract_sha256");
  if (lexer.TokenCount() != 2) lexer.Fail("contract_sha256 field count");
  wire.contract_sha256 = ParseShaToken(lexer, lexer.Token(1));

  next_key("contract_version");
  if (lexer.TokenCount() != 2) lexer.Fail("contract_version field count");
  if (ParseUint(lexer, lexer.Token(1), "contract_version") !=
      kExpectedContractVersion) {
    lexer.Fail("unsupported contract version binding");
  }

  next_key("contract_raw_length");
  if (lexer.TokenCount() != 2) lexer.Fail("contract_raw_length field count");
  const uint64_t raw_length =
      ParseUintMinimum(lexer, lexer.Token(1), "contract_raw_length", 1);

  next_key("contract_raw");
  if (lexer.TokenCount() != 3) lexer.Fail("contract_raw field count");
  const std::string raw =
      ParseByteString(lexer, "contract_raw", 1, 2, /*line_final=*/true);
  if (raw.size() != raw_length) {
    lexer.Fail("contract_raw length does not match contract_raw_length");
  }
  if (sha != nullptr) {
    // Hash-optional PARSE: a nullptr sha parses the wire but is never an
    // execution credential (the binding gate requires a real hash).
    const std::string digest = sha(raw);
    if (digest != wire.contract_sha256) {
      lexer.Fail("contract_raw does not hash to the bound contract_sha256");
    }
  }
  wire.contract_raw = raw;

  next_key("qualification");
  wire.qualification = ParseByteString(lexer,
                                       "qualification",
                                       1,
                                       2,
                                       /*line_final=*/true);
  if (wire.qualification != kCohortQualificationProvisional &&
      wire.qualification != kCohortQualificationExecutable) {
    lexer.Fail("unknown cohort qualification label");
  }
  next_key("completeness");
  wire.completeness =
      ParseByteString(lexer, "completeness", 1, 2, /*line_final=*/true);
  next_key("concurrency_basis");
  wire.concurrency_basis = ParseByteString(lexer,
                                           "concurrency_basis",
                                           1,
                                           2,
                                           /*line_final=*/true);
  if (wire.completeness != kResidentForecastComplete ||
      wire.concurrency_basis != kResidentForecastRollingDeclared) {
    lexer.Fail("a cohort wire cannot be partial or bank-only");
  }
  next_key("mode");
  wire.mode = ParseByteString(lexer, "mode", 1, 2, /*line_final=*/true);

  next_key("max_future");
  if (lexer.TokenCount() != 2) lexer.Fail("max_future field count");
  // Canonical Python parser parity at PARSE: positive uint8 only
  // (ParseUintMinimum refuses zero; ParseUintMax bounds the uint8 width);
  // the bounded FIFO queue is 1 or 2 ONLY for execution, and that
  // execution restriction lives in the execution gates.
  uint64_t max_future = 0;
  if (lexer.Token(1).size() > 3) {
    lexer.Fail("max_future must be a positive uint8");
  }
  max_future = ParseUintMinimum(lexer, lexer.Token(1), "max_future", 1);
  if (max_future > 255) {
    lexer.Fail("max_future must be a positive uint8");
  }
  wire.max_future = static_cast<int>(max_future);
  // Canonical Python parser parity at PARSE: any POSITIVE uint8 value is a
  // legal wire declaration (zero refuses here); the bounded FIFO queue is
  // 1 or 2 ONLY for execution, and that execution restriction lives in the
  // execution gates (never silently at parse).
  next_key("image_channels");
  if (lexer.TokenCount() != 2) lexer.Fail("image_channels field count");
  wire.image_channels = static_cast<int>(
      ParseUintMax(lexer, lexer.Token(1), "image_channels", 255));
  if (wire.image_channels < 1) {
    lexer.Fail("image_channels must be at least 1");
  }

  next_key("unclassified_allowance_bytes");
  if (lexer.TokenCount() != 4) {
    lexer.Fail("unclassified_allowance_bytes requires value + provenance");
  }
  wire.unclassified_allowance_bytes =
      ParseUint(lexer, lexer.Token(1), "unclassified_allowance_bytes");
  wire.unclassified_provenance = ParseByteString(
      lexer, "unclassified provenance", 2, 3, /*line_final=*/true);
  if (wire.unclassified_provenance.empty()) {
    lexer.Fail("unclassified allowance provenance is mandatory");
  }

  next_key("build_allocation_fingerprint");
  ParseFingerprint(lexer,
                   "build_allocation_fingerprint",
                   lexer.LineNo(),
                   &wire.has_build_allocation_fingerprint,
                   &wire.build_allocation_fingerprint);
  next_key("source_fingerprint");
  ParseFingerprint(lexer,
                   "source_fingerprint",
                   lexer.LineNo(),
                   &wire.has_source_fingerprint,
                   &wire.source_fingerprint);
  if (wire.qualification == kCohortQualificationExecutable &&
      (!wire.has_build_allocation_fingerprint ||
       !wire.has_source_fingerprint)) {
    lexer.Fail(
        "an executable cohort qualification cannot carry absent "
        "fingerprints");
  }
  if (wire.build_allocation_fingerprint != wire.source_fingerprint) {
    lexer.Fail(
        "cohort fingerprints must carry the same allocation-closure "
        "digest");
  }
  if (wire.has_build_allocation_fingerprint) {
    ValidateFingerprintText(lexer,
                            "cohort allocation fingerprint",
                            wire.build_allocation_fingerprint);
  }

  next_key("global_snapshot_plan");
  if (lexer.TokenCount() != 3 || lexer.Token(1) != "host") {
    lexer.Fail("expected the whole-invocation global_snapshot_plan basis");
  }
  wire.global_snapshot_plan_host_bytes =
      ParseUint(lexer, lexer.Token(2), "global_snapshot_plan host");

  next_key("cohort_count");
  if (lexer.TokenCount() != 2) lexer.Fail("cohort_count field count");
  wire.cohort_count = static_cast<size_t>(
      ParseUintMinimum(lexer, lexer.Token(1), "cohort_count", 1));

  size_t cursor = 0;
  bool process_bound = false;
  for (size_t ordinal = 0; ordinal < wire.cohort_count; ++ordinal) {
    next_key("cohort");
    if (lexer.TokenCount() != 2) lexer.Fail("cohort field count");
    if (ParseUint(lexer, lexer.Token(1), "cohort ordinal") != ordinal) {
      lexer.Fail("cohort ordinals must be sequential from 0");
    }
    CohortWireCohort cohort;
    cohort.ordinal = ordinal;

    next_key("member_count");
    if (lexer.TokenCount() != 2) lexer.Fail("member_count field count");
    cohort.member_count = static_cast<size_t>(
        ParseUintMinimum(lexer, lexer.Token(1), "member_count", 1));

    for (size_t m = 0; m < cohort.member_count; ++m) {
      next_key("member");
      if (lexer.TokenCount() != 2) lexer.Fail("member field count");
      CohortWireMember member;
      member.member_ordinal = static_cast<size_t>(
          ParseUint(lexer, lexer.Token(1), "member ordinal"));
      if (member.member_ordinal != cursor) {
        lexer.Fail(
            "cohort members must be contiguous increasing original "
            "ordinals (zero/duplicate/missing/reordered refuse)");
      }
      ++cursor;
      next_key("reference_image_idx");
      if (lexer.TokenCount() != 2) {
        lexer.Fail("reference_image_idx field count");
      }
      const int64_t image_idx =
          ParseSignedInt32(lexer, lexer.Token(1), "reference_image_idx");
      if (image_idx < 0) {
        lexer.Fail("reference_image_idx must be a nonnegative int32");
      }
      member.reference_image_idx = static_cast<int>(image_idx);
      next_key("reference_name");
      if (lexer.TokenCount() != 3) {
        lexer.Fail("reference_name field count");
      }
      member.reference_name =
          ParseByteString(lexer, "reference_name", 1, 2, /*line_final=*/true);
      if (member.reference_name.empty()) {
        lexer.Fail("reference_name must carry the exact model-image bytes");
      }
      next_key("input_count");
      if (lexer.TokenCount() != 2) lexer.Fail("input_count field count");
      member.input_count =
          ParseUintMinimum(lexer, lexer.Token(1), "input_count", 1);
      next_key("private_host_bytes");
      if (lexer.TokenCount() != 2) {
        lexer.Fail("private_host_bytes field count");
      }
      member.private_host_bytes =
          ParseUint(lexer, lexer.Token(1), "private_host_bytes");
      next_key("private_device_bytes");
      if (lexer.TokenCount() != 2) {
        lexer.Fail("private_device_bytes field count");
      }
      member.private_device_bytes =
          ParseUint(lexer, lexer.Token(1), "private_device_bytes");
      cohort.members.push_back(std::move(member));
    }

    for (const char* key : {"bank_estimate", "bank_charge", "reviewed_bank"}) {
      next_key(key);
      if (lexer.TokenCount() != 5 || lexer.Token(1) != "host" ||
          lexer.Token(3) != "device") {
        lexer.Fail(std::string(key) + " field layout");
      }
      const uint64_t host =
          ParseUint(lexer, lexer.Token(2), std::string(key) + " host");
      const uint64_t device =
          ParseUint(lexer, lexer.Token(4), std::string(key) + " device");
      if (std::string_view(key) == "bank_estimate") {
        cohort.bank_estimate_host_bytes = host;
        cohort.bank_estimate_device_bytes = device;
      } else if (std::string_view(key) == "bank_charge") {
        cohort.bank_charge_host_bytes = host;
        cohort.bank_charge_device_bytes = device;
      } else {
        cohort.reviewed_bank_host_bytes = host;
        cohort.reviewed_bank_device_bytes = device;
      }
    }
    // Reviewed reservation policy, per side: the charge is the reviewed R
    // where R is declared (and R >= the native estimate, both sides);
    // otherwise the canonical estimate is the charge. A producer that
    // mixes the policies (a declared R with a non-R charge on the same or
    // a mismatching side composition) refuses.
    for (int side = 0; side < 2; ++side) {
      const uint64_t reviewed = side == 0 ? cohort.reviewed_bank_host_bytes
                                          : cohort.reviewed_bank_device_bytes;
      const uint64_t estimate = side == 0 ? cohort.bank_estimate_host_bytes
                                          : cohort.bank_estimate_device_bytes;
      const uint64_t charge = side == 0 ? cohort.bank_charge_host_bytes
                                        : cohort.bank_charge_device_bytes;
      if (reviewed && estimate > reviewed) {
        lexer.Fail(
            "reviewed bank must be >= the native estimate (host or "
            "device side; zero means no reviewed R is declared)");
      }
      if ((reviewed && charge != reviewed) ||
          (!reviewed && charge != estimate)) {
        lexer.Fail(
            "bank charge must be the reviewed R when declared, else "
            "the canonical estimate (a mixed producer refuses)");
      }
    }

    next_key("normal");
    if (lexer.TokenCount() != 9 || lexer.Token(1) != "owner" ||
        lexer.Token(3) != "future" || lexer.Token(5) != "transient" ||
        lexer.Token(7) != "charge") {
      lexer.Fail("normal field layout");
    }
    cohort.normal_owner_bytes =
        ParseUint(lexer, lexer.Token(2), "normal owner");
    cohort.normal_future_bytes =
        ParseUint(lexer, lexer.Token(4), "normal future");
    cohort.normal_transient_bytes =
        ParseUint(lexer, lexer.Token(6), "normal transient");
    cohort.normal_charge_host_bytes =
        ParseUint(lexer, lexer.Token(8), "normal charge");

    next_key("queue");
    if (lexer.TokenCount() != 3 || lexer.Token(1) != "host") {
      lexer.Fail("queue field layout");
    }
    cohort.queue_host_bytes = ParseUint(lexer, lexer.Token(2), "queue host");
    if (cohort.queue_host_bytes < wire.global_snapshot_plan_host_bytes) {
      lexer.Fail(
          "every cohort ledger must carry the whole-invocation "
          "snapshot/plan charge (queue below the declared global "
          "basis; the persistent basis is never a per-cohort "
          "subset)");
    }

    next_key("process");
    if (lexer.TokenCount() != 10 || lexer.Token(1) != "mode" ||
        lexer.Token(4) != "count" || lexer.Token(6) != "host" ||
        lexer.Token(8) != "device") {
      lexer.Fail("process field layout");
    }
    cohort.process_mode =
        ParseByteString(lexer, "process mode", 2, 3, /*line_final=*/false);
    if (cohort.process_mode.empty()) {
      lexer.Fail("process mode is required");
    }
    cohort.process_count = ParseUint(lexer, lexer.Token(5), "process count");
    cohort.process_host_bytes =
        ParseUint(lexer, lexer.Token(7), "process host");
    cohort.process_device_bytes =
        ParseUint(lexer, lexer.Token(9), "process device");
    if (cohort.process_count != 1) {
      lexer.Fail(
          "process must be charged once per active ledger (the "
          "sequential cohort count never multiplies it)");
    }
    if (process_bound &&
        (cohort.process_mode != wire.cohorts.front().process_mode ||
         cohort.process_host_bytes != wire.cohorts.front().process_host_bytes ||
         cohort.process_device_bytes !=
             wire.cohorts.front().process_device_bytes)) {
      lexer.Fail(
          "process is charged once; every cohort must carry the "
          "identical process term");
    }
    process_bound = true;

    next_key("planner");
    if (lexer.TokenCount() != 6 || lexer.Token(1) != "max_member" ||
        lexer.Token(2) != "host" || lexer.Token(4) != "device") {
      lexer.Fail("planner field layout");
    }
    cohort.planner_max_member_host_bytes =
        ParseUint(lexer, lexer.Token(3), "planner max_member host");
    cohort.planner_max_member_device_bytes =
        ParseUint(lexer, lexer.Token(5), "planner max_member device");
    // The declared bound must be the per-side MAX of the member privates
    // (provable by recomputation; never added into any budget).
    {
      uint64_t max_host = 0;
      uint64_t max_device = 0;
      for (const auto& member : cohort.members) {
        if (member.private_host_bytes > max_host) {
          max_host = member.private_host_bytes;
        }
        if (member.private_device_bytes > max_device) {
          max_device = member.private_device_bytes;
        }
      }
      if (cohort.planner_max_member_host_bytes != max_host ||
          cohort.planner_max_member_device_bytes != max_device) {
        lexer.Fail(
            "planner max_member must be the per-side MAX of the "
            "member private charges (a bound is only provable by "
            "recomputation)");
      }
    }

    next_key("rolling");
    if (lexer.TokenCount() != 7 || lexer.Token(1) != "active_jobs" ||
        lexer.Token(3) != "host" || lexer.Token(5) != "device") {
      lexer.Fail("rolling field layout");
    }
    cohort.rolling_active_jobs =
        ParseUintMinimum(lexer, lexer.Token(2), "rolling active_jobs", 1);
    cohort.rolling_host_bytes =
        ParseUint(lexer, lexer.Token(4), "rolling host");
    cohort.rolling_device_bytes =
        ParseUint(lexer, lexer.Token(6), "rolling device");
    // MANDATORY admitted concurrency: 1 <= active_jobs <= member_count.
    if (cohort.rolling_active_jobs > cohort.member_count) {
      lexer.Fail(
          "active_jobs must not exceed the cohort member count (the "
          "admitted concurrency is bounded by the cohort)");
    }
    // The per-cohort rolling reserve must equal the SUM of the largest
    // active_jobs member private charges, INDEPENDENTLY PER SIDE
    // (out-of-order owner completion admits non-adjacent owners; only the
    // selected largest few values are added, so overflow refuses at the
    // REQUIRED reserve, never from an irrelevant remainder).
    {
      uint64_t host = 0;
      uint64_t device = 0;
      const size_t k = static_cast<size_t>(cohort.rolling_active_jobs);
      for (const int which : {0, 1}) {
        std::vector<uint64_t> values;
        values.reserve(cohort.members.size());
        for (const auto& member : cohort.members) {
          values.push_back(which == 0 ? member.private_host_bytes
                                      : member.private_device_bytes);
        }
        std::partial_sort(values.begin(),
                          values.begin() + static_cast<long>(k),
                          values.end(),
                          std::greater<uint64_t>());
        uint64_t sum = 0;
        for (size_t i = 0; i < k; ++i) {
          sum = CheckedSum("the largest active_jobs member private SUM",
                           sum,
                           values[i],
                           lexer);
        }
        if (which == 0) {
          host = sum;
        } else {
          device = sum;
        }
      }
      if (cohort.rolling_host_bytes != host ||
          cohort.rolling_device_bytes != device) {
        lexer.Fail(
            "rolling reserve must equal the SUM of the largest "
            "active_jobs member private charges in this cohort "
            "(per side independently; the declared reserve is the "
            "admitted concurrency, never a dynamic throttle)");
      }
    }

    next_key("ledger");
    if (lexer.TokenCount() != 5 || lexer.Token(1) != "host" ||
        lexer.Token(3) != "device") {
      lexer.Fail("ledger field layout");
    }
    cohort.ledger_host_bytes = ParseUint(lexer, lexer.Token(2), "ledger host");
    cohort.ledger_device_bytes =
        ParseUint(lexer, lexer.Token(4), "ledger device");
    // Per-cohort complete ledger equality (checked; this IS the admission
    // unit; sequential cohort ledgers are never aggregated):
    //   host   = queue + bank_charge + normal charge + process + rolling
    //   device = bank_charge + process + rolling
    const uint64_t expected_ledger_host =
        CheckedSum("ledger",
                   CheckedSum("ledger",
                              CheckedSum("ledger",
                                         cohort.queue_host_bytes,
                                         cohort.bank_charge_host_bytes,
                                         lexer),
                              CheckedSum("ledger",
                                         cohort.normal_charge_host_bytes,
                                         cohort.process_host_bytes,
                                         lexer),
                              lexer),
                   cohort.rolling_host_bytes,
                   lexer);
    const uint64_t expected_ledger_device =
        CheckedSum("ledger",
                   CheckedSum("ledger",
                              cohort.bank_charge_device_bytes,
                              cohort.process_device_bytes,
                              lexer),
                   cohort.rolling_device_bytes,
                   lexer);
    if (cohort.ledger_host_bytes != expected_ledger_host ||
        cohort.ledger_device_bytes != expected_ledger_device) {
      lexer.Fail(
          "the per-cohort ledger must equal queue + bank_charge + "
          "normal charge + process + rolling (host) and "
          "bank_charge + process + rolling (device) exactly: the "
          "ledger is the admission unit");
    }

    next_key("end_cohort");
    if (lexer.TokenCount() != 1) {
      lexer.Fail("end_cohort field count");
    }
    wire.cohorts.push_back(std::move(cohort));
  }
  wire.member_count_total = cursor;

  next_key("summary");
  if (lexer.TokenCount() != 1) lexer.Fail("summary field count");
  next_key("global_snapshot_plan");
  if (lexer.TokenCount() != 3 || lexer.Token(1) != "host") {
    lexer.Fail("summary global_snapshot_plan field layout");
  }
  if (ParseUint(lexer, lexer.Token(2), "summary global_snapshot_plan") !=
      wire.global_snapshot_plan_host_bytes) {
    lexer.Fail(
        "summary global_snapshot_plan disagrees with the declared "
        "global basis");
  }
  wire.summary.global_snapshot_plan_host_bytes =
      wire.global_snapshot_plan_host_bytes;

  next_key("queue_sum_host");
  if (lexer.TokenCount() != 2) {
    lexer.Fail("summary queue_sum_host field count");
  }
  wire.summary.queue_sum_host_bytes =
      ParseUint(lexer, lexer.Token(1), "queue_sum_host");

  next_key("bank_sum");
  if (lexer.TokenCount() != 5 || lexer.Token(1) != "host" ||
      lexer.Token(3) != "device") {
    lexer.Fail("summary bank_sum field layout");
  }
  wire.summary.bank_sum_host_bytes =
      ParseUint(lexer, lexer.Token(2), "bank_sum host");
  wire.summary.bank_sum_device_bytes =
      ParseUint(lexer, lexer.Token(4), "bank_sum device");

  next_key("normal_sum_host");
  if (lexer.TokenCount() != 2) {
    lexer.Fail("summary normal_sum_host field count");
  }
  wire.summary.normal_sum_host_bytes =
      ParseUint(lexer, lexer.Token(1), "normal_sum_host");

  next_key("private_sum");
  if (lexer.TokenCount() != 5 || lexer.Token(1) != "host" ||
      lexer.Token(3) != "device") {
    lexer.Fail("summary private_sum field layout");
  }
  wire.summary.private_sum_host_bytes =
      ParseUint(lexer, lexer.Token(2), "private_sum host");
  wire.summary.private_sum_device_bytes =
      ParseUint(lexer, lexer.Token(4), "private_sum device");

  next_key("process_once");
  if (lexer.TokenCount() != 5 || lexer.Token(1) != "host" ||
      lexer.Token(3) != "device") {
    lexer.Fail("summary process_once field layout");
  }
  wire.summary.process_once_host_bytes =
      ParseUint(lexer, lexer.Token(2), "process_once host");
  wire.summary.process_once_device_bytes =
      ParseUint(lexer, lexer.Token(4), "process_once device");
  // process_once must equal the first cohort's process charge on BOTH
  // sides (the once-per-active-ledger process exactly; Photogram 107711b
  // refuses either mismatch).
  if (wire.summary.process_once_host_bytes !=
          wire.cohorts.front().process_host_bytes ||
      wire.summary.process_once_device_bytes !=
          wire.cohorts.front().process_device_bytes) {
    lexer.Fail(
        "summary process_once must equal the first cohort's process "
        "charge on both sides (once per active ledger; never "
        "multiplied or understated)");
  }

  next_key("preflight_max");
  if (lexer.TokenCount() != 5 || lexer.Token(1) != "host" ||
      lexer.Token(3) != "device") {
    lexer.Fail(
        "summary preflight_max field layout (the aggregate "
        "capacity token is removed from the frozen layout and "
        "never parses)");
  }
  wire.summary.preflight_max_host_bytes =
      ParseUint(lexer, lexer.Token(2), "preflight_max host");
  wire.summary.preflight_max_device_bytes =
      ParseUint(lexer, lexer.Token(4), "preflight_max device");

  // Summary equality recomputation, checked: the accounting sums must
  // equal the per-cohort sums EXACTLY (accounting only, never admission);
  // preflight_max must equal the per-side MAX over the per-cohort ledgers.
  {
    uint64_t queue_sum = 0;
    uint64_t bank_sum_h = 0;
    uint64_t bank_sum_d = 0;
    uint64_t normal_sum_h = 0;
    uint64_t private_sum_h = 0;
    uint64_t private_sum_d = 0;
    uint64_t preflight_h = 0;
    uint64_t preflight_d = 0;
    for (const auto& cohort : wire.cohorts) {
      queue_sum =
          CheckedSum("queue_sum", queue_sum, cohort.queue_host_bytes, lexer);
      bank_sum_h = CheckedSum(
          "bank_sum host", bank_sum_h, cohort.bank_charge_host_bytes, lexer);
      bank_sum_d = CheckedSum("bank_sum device",
                              bank_sum_d,
                              cohort.bank_charge_device_bytes,
                              lexer);
      normal_sum_h = CheckedSum(
          "normal_sum", normal_sum_h, cohort.normal_charge_host_bytes, lexer);
      private_sum_h = CheckedSum(
          "private_sum host", private_sum_h, cohort.rolling_host_bytes, lexer);
      private_sum_d = CheckedSum("private_sum device",
                                 private_sum_d,
                                 cohort.rolling_device_bytes,
                                 lexer);
      if (cohort.ledger_host_bytes > preflight_h) {
        preflight_h = cohort.ledger_host_bytes;
      }
      if (cohort.ledger_device_bytes > preflight_d) {
        preflight_d = cohort.ledger_device_bytes;
      }
    }
    if (wire.summary.queue_sum_host_bytes != queue_sum ||
        wire.summary.bank_sum_host_bytes != bank_sum_h ||
        wire.summary.bank_sum_device_bytes != bank_sum_d ||
        wire.summary.normal_sum_host_bytes != normal_sum_h ||
        wire.summary.private_sum_host_bytes != private_sum_h ||
        wire.summary.private_sum_device_bytes != private_sum_d) {
      lexer.Fail(
          "the summary accounting sums must equal the per-cohort "
          "sums exactly (accounting only, never admission)");
    }
    if (wire.summary.preflight_max_host_bytes != preflight_h ||
        wire.summary.preflight_max_device_bytes != preflight_d) {
      lexer.Fail(
          "summary preflight_max must equal the MAX over the "
          "per-cohort ledgers (the plan's sequential peak; never a "
          "sum)");
    }
  }

  next_key("end");
  if (lexer.TokenCount() != 1) lexer.Fail("end field count");
  if (lexer.Next()) {
    lexer.Fail("trailing content after the cohort forecast");
  }
  return wire;
}

void VerifyResidentCohortForecastBinding(
    const CohortForecastWire& wire,
    const std::string& fresh_contract_bytes,
    const ResidentSha256Fn sha) {
  THROW_CHECK(sha != nullptr)
      << "cohort forecast binding requires actual SHA-256 verification; a "
         "build without a crypto backend refuses execution";
  THROW_CHECK_EQ(sha(fresh_contract_bytes), wire.contract_sha256)
      << "cohort forecast binding: the regenerated invocation snapshot does "
         "not hash to the wire's contract digest";
  THROW_CHECK_EQ(wire.contract_raw, fresh_contract_bytes)
      << "cohort forecast binding: the forecast's raw contract bytes do not "
         "equal the freshly regenerated whole-invocation snapshot";
  const auto contract = ParseOwnedBankContract(fresh_contract_bytes);
  THROW_CHECK_EQ(wire.member_count_total, contract.problems.size())
      << "cohort forecast binding: the cohort member blocks must cover "
         "every resolved problem exactly once";
  size_t member_index = 0;
  for (const auto& cohort : wire.cohorts) {
    for (const auto& member : cohort.members) {
      const auto& problem = contract.problems.at(member_index);
      THROW_CHECK_EQ(member.member_ordinal, member_index)
          << "cohort forecast binding: member ordinal ordering changed at "
          << member_index;
      THROW_CHECK_EQ(static_cast<uint64_t>(member.reference_image_idx),
                     static_cast<uint64_t>(problem.reference_image_idx))
          << "cohort forecast binding: reference image index changed at "
          << member_index;
      THROW_CHECK_EQ(member.reference_name, problem.reference_name)
          << "cohort forecast binding: reference name bytes changed at "
          << member_index;
      THROW_CHECK_EQ(member.input_count,
                     static_cast<uint64_t>(problem.inputs.size()))
          << "cohort forecast binding: input count changed at " << member_index;
      ++member_index;
    }
  }
}

void VerifyResidentCohortForecastParameters(
    const CohortForecastWire& wire,
    const std::string& compiled_cohort_allocation_fingerprint) {
  THROW_CHECK_EQ(wire.qualification, kCohortQualificationExecutable)
      << "cohort forecast execution requires the cohort-scoped executable "
         "qualification; provisional or unknown labels are never "
         "execution-qualified";
  THROW_CHECK_EQ(wire.completeness, kResidentForecastComplete)
      << "cohort forecast execution refuses partial forecasts";
  THROW_CHECK_EQ(wire.concurrency_basis, kResidentForecastRollingDeclared)
      << "cohort forecast execution refuses bank-only scopes";
  // Canonical process shape, like the accepted full-bank ledger gate: one
  // resident process, mode 'once', count 1, and the canonical 2 GiB floor
  // on BOTH sides (a self-consistent but understated process charge must
  // not pass execution admission).
  THROW_CHECK_EQ(wire.cohorts.front().process_mode, "once")
      << "cohort forecast execution requires the canonical single "
         "resident process mode 'once'";
  THROW_CHECK_EQ(wire.cohorts.front().process_count, 1u)
      << "cohort forecast execution requires process count 1 (once per "
         "active ledger)";
  THROW_CHECK_GE(wire.cohorts.front().process_host_bytes, 2147483648ull)
      << "cohort forecast execution requires the canonical 2 GiB host "
         "process floor";
  THROW_CHECK_GE(wire.cohorts.front().process_device_bytes, 2147483648ull)
      << "cohort forecast execution requires the canonical 2 GiB device "
         "process floor";
  THROW_CHECK_EQ(wire.mode, kResidentForecastModeBounded)
      << "cohort forecast execution is bounded-only; eager declarations "
         "refuse (unknown modes also refuse)";
  // Execution-side bounded FIFO queue contract (parser-parity: PARSE
  // accepts any positive uint8 declaration; execution requires 1 or 2).
  THROW_CHECK(wire.max_future == 1 || wire.max_future == 2)
      << "cohort forecast execution supports max_future 1 or 2 only "
         "(the bounded normal FIFO queue is the only execution scope), got "
      << wire.max_future;
  THROW_CHECK(wire.has_build_allocation_fingerprint &&
              wire.has_source_fingerprint)
      << "cohort forecast execution refuses absent fingerprints";
  THROW_CHECK_EQ(wire.build_allocation_fingerprint, wire.source_fingerprint)
      << "cohort forecast fingerprints disagree";
  THROW_CHECK(!compiled_cohort_allocation_fingerprint.empty())
      << "cohort forecast execution requires the compiled cohort-scoped "
         "allocation fingerprint of this build (no native registry; the "
         "full-bank closure certifies nothing here)";
  THROW_CHECK_EQ(wire.build_allocation_fingerprint,
                 compiled_cohort_allocation_fingerprint)
      << "cohort forecast allocation fingerprint does not match this "
         "build's compiled cohort-scoped closure; unknown build sources "
         "are unqualified";
}

void VerifyResidentCohortForecastFitsCaps(const CohortForecastWire& wire,
                                          const uint64_t host_budget_bytes,
                                          const uint64_t device_budget_bytes,
                                          const size_t cli_max_concurrency) {
  THROW_CHECK_GE(cli_max_concurrency, 1u)
      << "cohort forecast requires a declared CLI concurrency >= 1";
  for (size_t c = 0; c < wire.cohorts.size(); ++c) {
    const auto& cohort = wire.cohorts[c];
    // k per cohort: bounded by the member count (parser) and this run's
    // declared CLI concurrency (here); the adapter tail clamp must declare
    // exactly min(cli_max, member_count) in every cohort.
    THROW_CHECK_GE(cohort.rolling_active_jobs, 1u)
        << "cohort " << c
        << ": active_jobs must be >= 1 (no bank-only cohort wire exists)";
    THROW_CHECK_LE(cohort.rolling_active_jobs,
                   static_cast<uint64_t>(cli_max_concurrency))
        << "cohort " << c
        << ": active_jobs exceeds this run's declared CLI concurrency";
    // The parsed per-cohort ledger is the admission unit (its equality was
    // recomputed exactly by the parser from the wire's own terms, with the
    // global basis and process appearing once per active ledger). FitsCaps
    // is a HARD gate, per cohort independently, both sides.
    THROW_CHECK_LE(cohort.ledger_host_bytes, host_budget_bytes)
        << "cohort " << c << " COMPLETE ledger (" << cohort.ledger_host_bytes
        << " host, " << cohort.ledger_device_bytes
        << " device) exceeds the host cap " << host_budget_bytes
        << ": FitsCaps is a hard gate (no throttling, no degraded mode)";
    THROW_CHECK_LE(cohort.ledger_device_bytes, device_budget_bytes)
        << "cohort " << c << " COMPLETE ledger (" << cohort.ledger_host_bytes
        << " host, " << cohort.ledger_device_bytes
        << " device) exceeds the device cap " << device_budget_bytes
        << ": FitsCaps is a hard gate (no throttling, no degraded mode)";
  }
  // The plan-level sequential peak is the MAX over the parsed ledgers.
  THROW_CHECK_LE(wire.summary.preflight_max_host_bytes, host_budget_bytes)
      << "cohort forecast preflight_max host exceeds the host cap";
  THROW_CHECK_LE(wire.summary.preflight_max_device_bytes, device_budget_bytes)
      << "cohort forecast preflight_max device exceeds the device cap";
}

}  // namespace cohort
}  // namespace mvs
}  // namespace colmap