// SPDX-License-Identifier: BSD-3-Clause

// Native resident-cohort-forecast v1 wire parser + gate tests.
//
// Synthetic protocol fixtures bind the same tiny analytic contract used by
// resident_forecast_test.cc. README.json contains independently worked
// accounting assumptions and totals, not captured hardware measurements.
// Tests preserve strict truncation, ordering, identity, concurrency, SUM,
// ledger, preflight MAX, reviewed-charge and magic-confusion refusals.
//
// Not covered here (requires the linked native estimator / crypto build):
// ComputeResidentLedgerMinima recompute and per-cohort native minima
// gates; execution is NOT exposed anywhere (no CLI, no registry, no
// authorization from these tests). SHA-256 here uses a compact
// independent implementation validated by known NIST vectors and the
// fixture digests; production binds through ComputeSHA256 (crypto builds).
#include "colmap/mvs/resident_cohort_wire.h"
#include "colmap/mvs/resident_forecast.h"
#include "colmap/mvs/resident_contract.h"
#include "colmap/util/logging.h"

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <functional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <gtest/gtest.h>

#ifndef RESIDENT_COHORT_TESTDATA_DIR
#define RESIDENT_COHORT_TESTDATA_DIR ""
#endif
// The contract fixtures stay SHARED with the full-bank forecast tests (one
// canonical synthetic source of truth; no contract duplication).
#ifndef RESIDENT_COHORT_CONTRACT_TESTDATA_DIR
#define RESIDENT_COHORT_CONTRACT_TESTDATA_DIR RESIDENT_COHORT_TESTDATA_DIR
#endif

namespace colmap {
namespace mvs {
namespace cohort {
namespace {

std::string HexEncodeTest(const std::string& value) {
  static constexpr char kDigits[] = "0123456789abcdef";
  std::string out;
  out.reserve(value.size() * 2);
  for (const unsigned char c : value) {
    out.push_back(kDigits[c >> 4]);
    out.push_back(kDigits[c & 15]);
  }
  return out;
}

std::string ReadFileOrDieWithRoot(const std::string& root,
                                  const std::string& name) {
  const std::filesystem::path path = std::filesystem::path(root) / name;
  std::ifstream file(path, std::ios::binary);
  if (!file) {
    ADD_FAILURE() << "missing test fixture: " << path;
    return {};
  }
  return std::string((std::istreambuf_iterator<char>(file)),
                     std::istreambuf_iterator<char>());
}

std::string ReadFileOrDie(const std::string& name) {
  return ReadFileOrDieWithRoot(RESIDENT_COHORT_TESTDATA_DIR, name);
}

std::string Sha(const std::string_view& bytes) {
  // Independent SHA-256 (FIPS 180-4); validated by the fixture digests.
  const uint32_t k[64] = {
      0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1,
      0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
      0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786,
      0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
      0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147,
      0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
      0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
      0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
      0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a,
      0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
      0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};
  uint32_t h[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                   0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
  std::string data(bytes);
  const uint64_t bit_len = static_cast<uint64_t>(data.size()) * 8;
  data.push_back(static_cast<char>(0x80));
  while (data.size() % 64 != 56) data.push_back('\0');
  for (int i = 7; i >= 0; --i) {
    data.push_back(static_cast<char>((bit_len >> (i * 8)) & 0xff));
  }
  auto rotr = [](uint32_t x, int n) {
    return (x >> n) | (x << (32 - n));
  };
  for (size_t offset = 0; offset < data.size(); offset += 64) {
    uint32_t w[64];
    for (int i = 0; i < 16; ++i) {
      w[i] = (static_cast<uint32_t>(static_cast<unsigned char>(data[offset + i * 4])) << 24) |
             (static_cast<uint32_t>(static_cast<unsigned char>(data[offset + i * 4 + 1])) << 16) |
             (static_cast<uint32_t>(static_cast<unsigned char>(data[offset + i * 4 + 2])) << 8) |
             static_cast<uint32_t>(static_cast<unsigned char>(data[offset + i * 4 + 3]));
    }
    for (int i = 16; i < 64; ++i) {
      const uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
      const uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
      w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }
    uint32_t a = h[0], b = h[1], c = h[2], d = h[3];
    uint32_t e = h[4], f = h[5], g = h[6], hh = h[7];
    for (int i = 0; i < 64; ++i) {
      const uint32_t s1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
      const uint32_t ch = (e & f) ^ (~e & g);
      const uint32_t t1 = hh + s1 + ch + k[i] + w[i];
      const uint32_t s0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
      const uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
      const uint32_t t2 = s0 + maj;
      hh = g; g = f; f = e; e = d + t1;
      d = c; c = b; b = a; a = t1 + t2;
    }
    h[0] += a; h[1] += b; h[2] += c; h[3] += d;
    h[4] += e; h[5] += f; h[6] += g; h[7] += hh;
  }
  char out[65];
  for (int i = 0; i < 8; ++i) {
    std::snprintf(out + i * 8, 9, "%08x", h[i]);
  }
  return std::string(out, 64);
}

// Independent literals for the synthetic examples (README.json).
constexpr const char* kTwoPlusTwoFixtureSha =
    "440c673ca2ec58d5949b12defffb9640f4fcc6ce6d2616caf4c4aed3087e89ba";
constexpr const char* kAllSingletonFixtureSha =
    "9f684dc264230d9427e11e1b9bf59561e6da47dd508186966fb8e11dff613961";
constexpr const char* kFourContractSha =
    "8cf48120bbf9401c89a55fb59a9ec2b43d85dceab0d0a692116df1b19e161892";

class ResidentCohortWireTest : public ::testing::Test {
 protected:
  void SetUp() override {
    ASSERT_FALSE(std::string(RESIDENT_COHORT_TESTDATA_DIR).empty())
        << "RESIDENT_COHORT_TESTDATA_DIR is not configured";
    assert_wire_ = ReadFileOrDie("four-two-plus-two-provisional.cohort-wire");
    singleton_wire_ =
        ReadFileOrDie("four-all-singleton-provisional.cohort-wire");
    contract_ =
        ReadFileOrDieWithRoot(RESIDENT_COHORT_CONTRACT_TESTDATA_DIR,
                              "four.contract");
    ASSERT_EQ(Sha(assert_wire_), kTwoPlusTwoFixtureSha);
    ASSERT_EQ(Sha(singleton_wire_), kAllSingletonFixtureSha);
    ASSERT_EQ(Sha(contract_), kFourContractSha);
  }

  std::string assert_wire_;
  std::string singleton_wire_;
  std::string contract_;
};

TEST_F(ResidentCohortWireTest, FullBankWiresNeverParseAsCohortAndReverse) {
  // Distinct magic, no version negotiation, symmetric confusion refusal.
  const std::string fullbank_wire =
      "resident-forecast\nversion 1\ncontract_sha256 " +
      std::string(64, 'a') + "\n";
  EXPECT_THROW(ParseResidentCohortForecast(fullbank_wire),
               std::runtime_error);
  // A cohort wire is likewise never a full-bank wire (the full-bank
  // parser refuses the cohort magic symmetrically).
  EXPECT_THROW(ParseResidentForecast(assert_wire_, nullptr), std::exception);
}

TEST_F(ResidentCohortWireTest, ParsesTwoPlusTwoFixtureWithIndependentLiterals) {
  const CohortForecastWire wire =
      ParseResidentCohortForecast(assert_wire_, &Sha);
  EXPECT_EQ(wire.version, 1);
  EXPECT_EQ(wire.contract_sha256, kFourContractSha);
  EXPECT_EQ(wire.contract_version, 1);
  EXPECT_EQ(wire.contract_raw, contract_);
  EXPECT_EQ(wire.contract_raw.size(), 15947u);
  EXPECT_EQ(wire.qualification, "metadata_only_cpu_estimate_cohort_provisional");
  EXPECT_EQ(wire.completeness, "complete");
  EXPECT_EQ(wire.concurrency_basis, "rolling_reserve_declared");
  EXPECT_EQ(wire.mode, "bounded");
  EXPECT_EQ(wire.max_future, 2);
  EXPECT_EQ(wire.image_channels, 3);
  EXPECT_EQ(wire.unclassified_allowance_bytes, 0u);
  EXPECT_EQ(wire.unclassified_provenance, "synthetic_cohort_declared_terms");
  EXPECT_FALSE(wire.has_build_allocation_fingerprint);
  EXPECT_FALSE(wire.has_source_fingerprint);
  EXPECT_EQ(wire.global_snapshot_plan_host_bytes, 192u);
  EXPECT_EQ(wire.cohort_count, 2u);
  EXPECT_EQ(wire.member_count_total, 4u);
  ASSERT_EQ(wire.cohorts.size(), 2u);

  // Cohort 0 (members 0,1): independent literals for the synthetic example.
  EXPECT_EQ(wire.cohorts[0].ordinal, 0u);
  ASSERT_EQ(wire.cohorts[0].members.size(), 2u);
  EXPECT_EQ(wire.cohorts[0].members[0].member_ordinal, 0u);
  EXPECT_EQ(wire.cohorts[0].members[0].reference_image_idx, 0);
  EXPECT_EQ(wire.cohorts[0].members[0].reference_name, "synthetic_0.ppm");
  EXPECT_EQ(wire.cohorts[0].members[0].input_count, 4u);
  EXPECT_EQ(wire.cohorts[0].members[0].private_host_bytes, 445u);
  EXPECT_EQ(wire.cohorts[0].members[0].private_device_bytes, 5064u);
  EXPECT_EQ(wire.cohorts[0].members[1].member_ordinal, 1u);
  EXPECT_EQ(wire.cohorts[0].members[1].reference_name, "synthetic_1.ppm");
  EXPECT_EQ(wire.cohorts[0].members[1].private_host_bytes, 371u);
  EXPECT_EQ(wire.cohorts[0].members[1].private_device_bytes, 3601u);
  EXPECT_EQ(wire.cohorts[0].bank_estimate_host_bytes, 3749u);
  EXPECT_EQ(wire.cohorts[0].bank_estimate_device_bytes, 5707u);
  EXPECT_EQ(wire.cohorts[0].bank_charge_host_bytes, 3749u);
  EXPECT_EQ(wire.cohorts[0].bank_charge_device_bytes, 5707u);
  EXPECT_EQ(wire.cohorts[0].reviewed_bank_host_bytes, 0u);
  EXPECT_EQ(wire.cohorts[0].reviewed_bank_device_bytes, 0u);
  EXPECT_EQ(wire.cohorts[0].normal_owner_bytes, 144u);
  EXPECT_EQ(wire.cohorts[0].normal_future_bytes, 288u);
  EXPECT_EQ(wire.cohorts[0].normal_transient_bytes, 144u);
  EXPECT_EQ(wire.cohorts[0].normal_charge_host_bytes, 445u);
  EXPECT_EQ(wire.cohorts[0].queue_host_bytes, 320u);
  EXPECT_EQ(wire.cohorts[0].process_mode, "once");
  EXPECT_EQ(wire.cohorts[0].process_count, 1u);
  EXPECT_EQ(wire.cohorts[0].process_host_bytes, 2147483648u);
  EXPECT_EQ(wire.cohorts[0].process_device_bytes, 2147483648u);
  EXPECT_EQ(wire.cohorts[0].planner_max_member_host_bytes, 445u);
  EXPECT_EQ(wire.cohorts[0].planner_max_member_device_bytes, 5064u);
  EXPECT_EQ(wire.cohorts[0].rolling_active_jobs, 1u);
  EXPECT_EQ(wire.cohorts[0].rolling_host_bytes, 445u);
  EXPECT_EQ(wire.cohorts[0].rolling_device_bytes, 5064u);
  EXPECT_EQ(wire.cohorts[0].ledger_host_bytes, 2147488607u);
  EXPECT_EQ(wire.cohorts[0].ledger_device_bytes, 2147494419u);

  // Cohort 1 (members 2,3): independent literals for the synthetic example.
  ASSERT_EQ(wire.cohorts[1].members.size(), 2u);
  EXPECT_EQ(wire.cohorts[1].members[0].member_ordinal, 2u);
  EXPECT_EQ(wire.cohorts[1].members[0].reference_image_idx, 2);
  EXPECT_EQ(wire.cohorts[1].members[0].reference_name, "synthetic_2.ppm");
  EXPECT_EQ(wire.cohorts[1].members[0].input_count, 3u);
  EXPECT_EQ(wire.cohorts[1].members[0].private_host_bytes, 223u);
  EXPECT_EQ(wire.cohorts[1].members[0].private_device_bytes, 2744u);
  EXPECT_EQ(wire.cohorts[1].members[1].member_ordinal, 3u);
  EXPECT_EQ(wire.cohorts[1].members[1].input_count, 2u);
  EXPECT_EQ(wire.cohorts[1].members[1].private_host_bytes, 260u);
  EXPECT_EQ(wire.cohorts[1].members[1].private_device_bytes, 2058u);
  EXPECT_EQ(wire.cohorts[1].bank_estimate_host_bytes, 3716u);
  EXPECT_EQ(wire.cohorts[1].bank_estimate_device_bytes, 5707u);
  EXPECT_EQ(wire.cohorts[1].queue_host_bytes, 288u);
  EXPECT_EQ(wire.cohorts[1].planner_max_member_device_bytes, 2744u);
  EXPECT_EQ(wire.cohorts[1].rolling_active_jobs, 1u);
  EXPECT_EQ(wire.cohorts[1].ledger_host_bytes, 2147488172u);
  EXPECT_EQ(wire.cohorts[1].ledger_device_bytes, 2147492099u);

  // Summary: accounting sums EXACTLY; preflight_max is the MAX over the
  // per-cohort ledgers (the sequential peak); NO aggregate capacity token.
  EXPECT_EQ(wire.summary.global_snapshot_plan_host_bytes, 192u);
  EXPECT_EQ(wire.summary.queue_sum_host_bytes, 608u);
  EXPECT_EQ(wire.summary.bank_sum_host_bytes, 7465u);
  EXPECT_EQ(wire.summary.bank_sum_device_bytes, 11414u);
  EXPECT_EQ(wire.summary.normal_sum_host_bytes, 705u);
  EXPECT_EQ(wire.summary.private_sum_host_bytes, 705u);
  EXPECT_EQ(wire.summary.private_sum_device_bytes, 7808u);
  EXPECT_EQ(wire.summary.process_once_host_bytes, 2147483648u);
  EXPECT_EQ(wire.summary.process_once_device_bytes, 2147483648u);
  EXPECT_EQ(wire.summary.preflight_max_host_bytes, 2147488607u);
  EXPECT_EQ(wire.summary.preflight_max_device_bytes, 2147494419u);
}

TEST_F(ResidentCohortWireTest, ParsesAllSingletonFixture) {
  const CohortForecastWire wire =
      ParseResidentCohortForecast(singleton_wire_, &Sha);
  EXPECT_EQ(wire.cohort_count, 4u);
  EXPECT_EQ(wire.member_count_total, 4u);
  for (size_t c = 0; c < 4; ++c) {
    ASSERT_EQ(wire.cohorts[c].members.size(), 1u);
    EXPECT_EQ(wire.cohorts[c].members[0].member_ordinal, c);
  }
  EXPECT_EQ(wire.cohorts[0].ledger_host_bytes, 2147488361u);
  EXPECT_EQ(wire.cohorts[0].ledger_device_bytes, 2147494419u);
  EXPECT_EQ(wire.cohorts[3].ledger_host_bytes, 2147487696u);
  EXPECT_EQ(wire.cohorts[3].ledger_device_bytes, 2147490669u);
  EXPECT_EQ(wire.summary.preflight_max_host_bytes, 2147488361u);
  EXPECT_EQ(wire.summary.preflight_max_device_bytes, 2147494419u);
  EXPECT_EQ(wire.summary.queue_sum_host_bytes, 928u);
  EXPECT_EQ(wire.summary.bank_sum_host_bytes, 13623u);
  EXPECT_EQ(wire.summary.bank_sum_device_bytes, 21139u);
  EXPECT_EQ(wire.summary.normal_sum_host_bytes, 1299u);
  EXPECT_EQ(wire.summary.private_sum_host_bytes, 1299u);
  EXPECT_EQ(wire.summary.private_sum_device_bytes, 13467u);
}

TEST_F(ResidentCohortWireTest, ParsesWithoutShaButExecutesOnlyWithIt) {
  // Hash-optional PARSE (never an execution credential); malformed declared
  // digests still refuse at parse.
  EXPECT_NO_THROW(ParseResidentCohortForecast(assert_wire_, nullptr));
  std::string tampered = assert_wire_;
  const std::string prefix = "contract_sha256 ";
  const size_t at = tampered.find(prefix) + prefix.size();
  tampered[at] = 'a' == tampered[at] ? 'b' : 'a';
  EXPECT_THROW(ParseResidentCohortForecast(tampered, &Sha), std::runtime_error);
}

#ifndef RESIDENT_COHORT_LOCAL_PARSER_SLICE
// The native binding gate needs the REAL native contract parser at link
// time (ParseOwnedBankContract); the local stdlib-only parser/gate slice
// compiles without it and runs the binding coverage in the native build.
// Per-cohort ledger is the admission unit: binding the wire against the
// freshly regenerated contract (the whole-invocation contract bytes), the
// parameters gate (provisional refuses EXECUTION but parses), and the HARD
// FitsCaps gate with the README's accepted caps.
TEST_F(ResidentCohortWireTest, BindingAndGatesAcceptTheFrozenFixtures) {
  const CohortForecastWire wire =
      ParseResidentCohortForecast(assert_wire_, &Sha);
  VerifyResidentCohortForecastBinding(wire, contract_, &Sha);
  // Provisional labels PARSE and pass wire-internal gates but are NOT
  // execution-qualified: the execution parameter gate refuses them.
  EXPECT_THROW(VerifyResidentCohortForecastParameters(wire, "any-digest"),
               std::invalid_argument);
  // The committed provisional wires: every ledger fits the README caps
  // independently (40 GiB host / 32 GiB device, CLI-only).
  VerifyResidentCohortForecastFitsCaps(wire, 40ull * 1024 * 1024 * 1024,
                                       32ull * 1024 * 1024 * 1024,
                                       /*cli_max_concurrency=*/1);
  VerifyResidentCohortForecastFitsCaps(wire,
                                       2147488607u,
                                       2147494419u,
                                       /*cli_max_concurrency=*/1);

  const CohortForecastWire singletons =
      ParseResidentCohortForecast(singleton_wire_, &Sha);
  VerifyResidentCohortForecastBinding(singletons, contract_, &Sha);
  VerifyResidentCohortForecastFitsCaps(singletons,
                                       2147488361u,
                                       2147494419u,
                                       /*cli_max_concurrency=*/1);
}

TEST_F(ResidentCohortWireTest, FitsCapsIsHardPerCohortNeverThrottled) {
  const CohortForecastWire wire =
      ParseResidentCohortForecast(assert_wire_, &Sha);
  // One byte below the largest per-cohort ledger refuses (host side).
  EXPECT_THROW(VerifyResidentCohortForecastFitsCaps(wire,
                                                    2147488606u,
                                                    32ull * 1024 * 1024 * 1024,
                                                    /*cli_max_concurrency=*/1),
               std::invalid_argument);
  // One byte below the largest per-cohort ledger refuses (device side).
  EXPECT_THROW(VerifyResidentCohortForecastFitsCaps(wire,
                                                    40ull * 1024 * 1024 * 1024,
                                                    2147494418u,
                                                    /*cli_max_concurrency=*/1),
               std::invalid_argument);
}

#endif  // RESIDENT_COHORT_LOCAL_PARSER_SLICE

// The HARD per-cohort FitsCaps gate (stdlib-only; runs in the local
// parser/gate slice too): every cohort ledger fits the SAME caps
// independently, active_jobs<= the declared CLI concurrency; any single
// unfunded ledger refuses the whole forecast; no throttling.
TEST_F(ResidentCohortWireTest, FitsCapsGateAcceptsFixtureWithReadmeCaps) {
  const CohortForecastWire wire =
      ParseResidentCohortForecast(assert_wire_, &Sha);
  const uint64_t host_cap = 40ull * 1024 * 1024 * 1024;
  const uint64_t device_cap = 32ull * 1024 * 1024 * 1024;
  VerifyResidentCohortForecastFitsCaps(wire, host_cap, device_cap,
                                       /*cli_max_concurrency=*/1);
  // Exact-cap fit: the largest per-cohort ledger defines the minimum.
  VerifyResidentCohortForecastFitsCaps(wire,
                                       2147488607u,
                                       2147494419u,
                                       /*cli_max_concurrency=*/1);
  // active_jobs above the CLI limit refuses (k must be <= the CLI max and
  // <= member_count).
  EXPECT_THROW(VerifyResidentCohortForecastFitsCaps(wire, host_cap,
                                                    device_cap,
                                                    /*cli_max_concurrency=*/0),
               std::invalid_argument);
}

TEST_F(ResidentCohortWireTest, ParametersGateSeparatesProvisionalFromExecutable) {
  // Provisional cohort wires PARSE and pass the wire-internal gates but
  // are never execution-qualified. An executable-labeled wire with BOTH
  // fingerprints is required by the execution parameter gate, and only the
  // exact compiled cohort-scoped digest accepts (native has NO registry;
  // a full-bank fingerprint certifies nothing here).
  std::string executable = assert_wire_;
  const std::string qual_key = "qualification";
  const size_t qual_line = executable.find(qual_key + " 45 ");
  ASSERT_NE(qual_line, std::string::npos);
  const std::string label = "metadata_only_cpu_estimate_cohort";
  const std::string label_hex = HexEncodeTest(label);
  executable.replace(qual_line, executable.find('\n', qual_line) - qual_line,
                     qual_key + " " + std::to_string(label.size()) + " " + label_hex);
  // A real digest-text shape (exact 64 lowercase hex characters).
  const std::string digest(64, 'e');  // synthetic, not a build identity
  EXPECT_EQ(digest.size(), 64u);
  const std::string fp_line = "build_allocation_fingerprint absent";
  for (const char* field : {"build_allocation_fingerprint", "source_fingerprint"}) {
    const std::string needle = std::string(field) + " absent";
    const size_t at = executable.find(needle);
    ASSERT_NE(at, std::string::npos);
    executable.replace(at, needle.size(),
                       std::string(field) + " 64 " + HexEncodeTest(digest));
  }
  const CohortForecastWire exec_wire =
      ParseResidentCohortForecast(executable, &Sha);
  EXPECT_EQ(exec_wire.qualification, label);
  EXPECT_TRUE(exec_wire.has_build_allocation_fingerprint);
  EXPECT_EQ(exec_wire.build_allocation_fingerprint, digest);
  // Provisional fixture wire: parameters gate refuses (not executable).
  const CohortForecastWire provisional =
      ParseResidentCohortForecast(assert_wire_, &Sha);
  EXPECT_THROW(VerifyResidentCohortForecastParameters(provisional,
                                                      "any-digest"),
               std::invalid_argument);
  // Executable wire: only the exact compiled cohort-scoped digest accepts.
  EXPECT_THROW(VerifyResidentCohortForecastParameters(exec_wire,
                     std::string(64, 'd')),
               std::invalid_argument);
  VerifyResidentCohortForecastParameters(exec_wire, digest);
  // An empty compiled fingerprint (no cohort closure built) refuses.
  EXPECT_THROW(VerifyResidentCohortForecastParameters(exec_wire, ""),
               std::invalid_argument);
}

TEST_F(ResidentCohortWireTest, ProcessOnceParityParseRuleAndOneSidedReviewedR) {
  auto expect_parses_refused = [&](const std::string& mutated) {
    EXPECT_THROW(ParseResidentCohortForecast(mutated, &Sha), std::exception)
        << "a mutated wire must refuse: " << mutated.substr(0, 64);
  };
  // Summary process_once MUST equal the first cohort's process charge on
  // both sides (mutating either side refuses).
  {
    std::string wrong = assert_wire_;
    const std::string needle = "process_once host 2147483648 device 2147483648";
    const size_t at = wrong.find(needle);
    ASSERT_NE(at, std::string::npos);
    wrong.replace(at, needle.size(),
                  "process_once host 2147483647 device 2147483648");
    expect_parses_refused(wrong);
  }
  {
    std::string wrong = assert_wire_;
    const std::string needle = "process_once host 2147483648 device 2147483648";
    const size_t at = wrong.find(needle);
    ASSERT_NE(at, std::string::npos);
    wrong.replace(at, needle.size(),
                  "process_once host 2147483648 device 4294967296");
    expect_parses_refused(wrong);
  }
  // Parse parity with the canonical Python parser: max_future is any
  // POSITIVE uint8 at PARSE (zero refuses; a larger bounded queue is
  // readable; only EXECUTION restricts 1..2, in the execution gates).
  {
    std::string large_future = assert_wire_;
    const size_t at = large_future.find("max_future 2");
    ASSERT_NE(at, std::string::npos);
    large_future.replace(at, 12, "max_future 3");
    EXPECT_NO_THROW(ParseResidentCohortForecast(large_future, &Sha));
    std::string zero_future = large_future;
    zero_future.replace(zero_future.find("max_future 3"), 12, "max_future 0");
    expect_parses_refused(zero_future);
  }
  // ONE-SIDED reviewed R is wire-valid (per-side policy): host declares a
  // reviewed R of 4000; device stays the canonical estimate. All
  // derived values (bank charge, cohort ledger, summary bank_sum and
  // preflight_max) must shift coherently and parse.
  {
    std::string one_sided = assert_wire_;
    const uint64_t estimate_host = 3749;
    const uint64_t reviewed_host = 4000;
    const auto replace_first = [&](const std::string& needle,
                                   const std::string& update) {
      const size_t at = one_sided.find(needle);
      ASSERT_NE(at, std::string::npos) << needle;
      one_sided.replace(at, needle.size(), update);
    };
    replace_first("bank_charge host 3749 device 5707",
                  "bank_charge host 4000 device 5707");
    replace_first("reviewed_bank host 0 device 0",
                  "reviewed_bank host 4000 device 0");
    // cohort ledger host 2147488607 + (4000 - 3749) = 2147488858
    replace_first("ledger host 2147488607 device 2147494419",
                  "ledger host 2147488858 device 2147494419");
    // summary bank_sum host 7465 + 251 = 7716
    replace_first("bank_sum host 7465 device 11414",
                  "bank_sum host 7716 device 11414");
    // preflight_max host = 2147488858
    replace_first("preflight_max host 2147488607 device 2147494419",
                  "preflight_max host 2147488858 device 2147494419");
    const CohortForecastWire one_sided_wire = ParseResidentCohortForecast(one_sided, &Sha);
    EXPECT_EQ(one_sided_wire.cohorts[0].reviewed_bank_host_bytes, 4000u);
    EXPECT_EQ(one_sided_wire.cohorts[0].bank_charge_host_bytes, 4000u);
    EXPECT_EQ(one_sided_wire.cohorts[0].bank_charge_device_bytes,
              one_sided_wire.cohorts[0].bank_estimate_device_bytes);
    EXPECT_EQ(one_sided_wire.cohorts[0].ledger_host_bytes, 2147488858u);
    EXPECT_EQ(one_sided_wire.summary.bank_sum_host_bytes, 7716u);
    EXPECT_EQ(one_sided_wire.summary.preflight_max_host_bytes, 2147488858u);
    // The reviewed-R-vs-estimate policy still holds on the DECLARED side.
    EXPECT_THROW(ParseResidentCohortForecast(
                     one_sided.replace(
                         one_sided.find("reviewed_bank host 4000 device 0"),
                         std::string("reviewed_bank host 4000 device 0").size(),
                         "reviewed_bank host 3700 device 0"),
                     &Sha),
                 std::exception);
  }
}

TEST_F(ResidentCohortWireTest, MutatedWiresRefusePerRule) {
  auto expect_parses_refused = [&](const std::string& mutated) {
    EXPECT_THROW(ParseResidentCohortForecast(mutated, &Sha),
                 std::exception)
        << "a mutated wire must refuse: " << mutated.substr(0, 64);
  };
  // Truncated final line and trailing content.
  expect_parses_refused(assert_wire_.substr(0, assert_wire_.size() - 1));
  expect_parses_refused(assert_wire_ + "padding\n");
  // Member ordinal zero/gap/duplicate/sort rejections.
  {
    std::string gap = assert_wire_;
    const std::string needle = "\nmember 1\n";
    const size_t at = gap.find(needle);
    ASSERT_NE(at, std::string::npos);
    gap.replace(at, needle.size(), "\nmember 7\n");
    expect_parses_refused(gap);
  }
  {
    std::string zero_jobs = assert_wire_;
    const std::string needle = "member_count 2\n";
    const size_t at = zero_jobs.find(needle);
    ASSERT_NE(at, std::string::npos);
    zero_jobs.replace(at, needle.size(), "member_count 0\n");
    // Remove the first member block so a zero-member cohort is submitted
    // verbatim (the parser must refuse zero members closed by end_cohort).
    const size_t member_at = zero_jobs.find("\nmember 0\n", at);
    const size_t bank_at = zero_jobs.find("\nbank_estimate ", member_at);
    ASSERT_NE(member_at, std::string::npos);
    ASSERT_NE(bank_at, std::string::npos);
    zero_jobs.erase(member_at, bank_at - member_at);
    expect_parses_refused(zero_jobs);
  }
  // active_jobs must be a per-cohort admitted concurrency >= 1 (and <=
  // member_count): 0 refuses, over-limit refuses.
  {
    std::string zero = assert_wire_;
    const std::string needle = "rolling active_jobs 1 host 445 device 5064";
    const size_t at = zero.find(needle);
    ASSERT_NE(at, std::string::npos);
    zero.replace(at, needle.size(),
                 "rolling active_jobs 0 host 0 device 0");
    expect_parses_refused(zero);
  }
  {
    std::string over = assert_wire_;
    const std::string needle = "rolling active_jobs 1 host 445 device 5064";
    const size_t at = over.find(needle);
    ASSERT_NE(at, std::string::npos);
    over.replace(
        at, needle.size(), "rolling active_jobs 9 host 9000 device 50000");
    expect_parses_refused(over);
  }
  // A WRONG declared rolling SUM (not the largest active_jobs member privates)
  // refuses even when the active_jobs value itself is legal.
  {
    std::string wrong_sum = assert_wire_;
    const std::string needle = "rolling active_jobs 1 host 445 device 5064";
    const size_t at = wrong_sum.find(needle);
    ASSERT_NE(at, std::string::npos);
    wrong_sum.replace(
        at, needle.size(), "rolling active_jobs 1 host 446 device 5064");
    expect_parses_refused(wrong_sum);
  }
  // The per-cohort ledger line must equal the recomputed complete ledger.
  {
    std::string wrong_ledger = assert_wire_;
    const std::string needle = "ledger host 2147488607 device 2147494419";
    const size_t at = wrong_ledger.find(needle);
    ASSERT_NE(at, std::string::npos);
    wrong_ledger.replace(
        at, needle.size(), "ledger host 2147488608 device 2147494419");
    expect_parses_refused(wrong_ledger);
  }
  // Cohort queue below the whole-invocation snapshot/plan basis refuses.
  {
    std::string low_queue = assert_wire_;
    const std::string needle = "queue host 320";
    const size_t at = low_queue.find(needle);
    ASSERT_NE(at, std::string::npos);
    low_queue.replace(at, needle.size(), "queue host 191");
    expect_parses_refused(low_queue);
  }
  // The removed aggregate capacity token must never reappear.
  {
    std::string capacity_token = assert_wire_;
    capacity_token.insert(capacity_token.rfind("\npreflight_max"),
                          "\ncapacity host 4294976779 device 4294986518");
    expect_parses_refused(capacity_token);
  }
  // Summary arithmetic mismatches refuse: preflight_max below the true MAX.
  {
    std::string wrong_max = assert_wire_;
    const std::string needle =
        "preflight_max host 2147488607 device 2147494419";
    const size_t at = wrong_max.find(needle);
    ASSERT_NE(at, std::string::npos);
    wrong_max.replace(
        at, needle.size(), "preflight_max host 2147488171 device 2147492098");
    expect_parses_refused(wrong_max);
  }
  {
    std::string wrong_sum = assert_wire_;
    const std::string needle = "bank_sum host 7465 device 11414";
    const size_t at = wrong_sum.find(needle);
    ASSERT_NE(at, std::string::npos);
    wrong_sum.replace(at, needle.size(), "bank_sum host 7464 device 11414");
    expect_parses_refused(wrong_sum);
  }
  // A fabricated per-cohort declared MAX (planner max_member) mismatch is
  // a wire refusal: the declared member maxima must hold.
  {
    std::string wrong_max_member = assert_wire_;
    const std::string needle = "planner max_member host 445 device 5064";
    const size_t at = wrong_max_member.find(needle);
    ASSERT_NE(at, std::string::npos);
    wrong_max_member.replace(
        at, needle.size(), "planner max_member host 444 device 5064");
    expect_parses_refused(wrong_max_member);
  }
  // Reviewed bank and canonical estimate policy: a declared reviewed R
  // smaller than the estimate or a mismatching bank charge refuses.
  {
    std::string policy = assert_wire_;
    const std::string needle =
        "bank_estimate host 3749 device 5707\n"
        "bank_charge host 3749 device 5707\n"
        "reviewed_bank host 0 device 0\n";
    const size_t at = policy.find(needle);
    ASSERT_NE(at, std::string::npos);
    policy.replace(at,
                   needle.size(),
                   "bank_estimate host 3749 device 5707\n"
                   "bank_charge host 3750 device 5707\n"
                   "reviewed_bank host 0 device 0\n");
    expect_parses_refused(policy);
  }
  // Bare 64-hex fingerprint token is refused (v1 length-prefixed encoding
  // only), and a digest length mismatch refuses.
  {
    std::string bare = assert_wire_;
    const std::string needle = "build_allocation_fingerprint absent";
    const size_t at = bare.find(needle);
    ASSERT_NE(at, std::string::npos);
    bare.replace(at, needle.size(), "build_allocation_fingerprint " + std::string(64, 'f'));
    expect_parses_refused(bare);
  }
  {
    std::string short_fp = assert_wire_;
    const std::string needle = "build_allocation_fingerprint absent";
    const size_t at = short_fp.find(needle);
    ASSERT_NE(at, std::string::npos);
    short_fp.replace(at, needle.size(), "build_allocation_fingerprint 63 " + std::string(126, 'f'));
    expect_parses_refused(short_fp);
  }
  // Executable qualification with absent fingerprints refuses at parse
  // (the registry selection stays Python-side; native has no registry).
  {
    std::string executable = assert_wire_;
    const std::string needle =
        "qualification 45 6d657461646174615f6f6e6c795f6370755f657374696d6174655f636f686f72745f70726f766973696f6e616c";
    const size_t at = executable.find(needle);
    ASSERT_NE(at, std::string::npos);
    executable.replace(at, needle.size(), "qualification 33 6d657461646174615f6f6e6c795f6370755f657374696d6174655f636f686f7274");
    expect_parses_refused(executable);
  }
  // Unknown qualification label refuses.
  {
    std::string unknown = assert_wire_;
    const std::string needle =
        "qualification 45 6d657461646174615f6f6e6c795f6370755f657374696d6174655f636f686f72745f70726f766973696f6e616c";
    const size_t at = unknown.find(needle);
    ASSERT_NE(at, std::string::npos);
    unknown.replace(at, needle.size(), "qualification 3 616263");
    expect_parses_refused(unknown);
  }
}

TEST(ResidentCohortWireParser, BareHexOrMalformedDigestTokenRefused) {
  // A bare digest text or wrong-hex contract_sha256 never parses.
  const std::string bare =
      "resident-cohort-forecast\nversion 1\ncontract_sha256 " +
      std::string(64, 'z') + "\ncontract_version 1\n";
  EXPECT_THROW(ParseResidentCohortForecast(bare), std::exception);
}

}  // namespace
}  // namespace cohort
}  // namespace mvs
}  // namespace colmap