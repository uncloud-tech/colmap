// SPDX-License-Identifier: BSD-3-Clause
// Native resident-forecast v1 wire parser + gate tests.
//
// Positive vectors are independently constructed synthetic records in
// testdata/resident_forecast. Four tiny analytic images, invented identities
// and non-sorted input orders replace all captured workload data. Accounting
// assumptions and worked totals are in testdata/resident_cohort/README.json;
// these are protocol/gate examples, NOT physical measurements or build
// qualifications. Literal expectations and SHA-256 bindings pin exact bytes.
// SHA-256 here is independently checked against NIST vectors; production
// binds through ComputeSHA256 (crypto builds).
//
// Not covered here (requires the dense-stereo library with OpenImageIO and
// the real native estimator): ComputeResidentLedgerMinima's BankJob-based
// recompute; see resident_forecast_native_test.cc. The CLI runner mode is
// covered by the CLI tests.
#include "colmap/mvs/resident_forecast.h"

#include "colmap/mvs/optimized_dispatch.h"
#include "colmap/mvs/owned_resident_minima.h"
#include "colmap/mvs/resident_contract.h"
#include "colmap/util/logging.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <exception>
#include <fstream>
#include <functional>
#include <limits>
#include <string>
#include <string_view>
#include <vector>

#include <gtest/gtest.h>

#if __has_include("colmap/mvs/generated/resident_allocation_fingerprint.h")
// Generated at build by scripts/resident/generate_resident_allocation_
// fingerprint.py from the frozen ordered allocation closure. Only when the
// build wires the generator is the compiled digest visible.
#define RESIDENT_FORECAST_HAS_COMPILED_FINGERPRINT 1
#include "colmap/mvs/generated/resident_allocation_fingerprint.h"
#endif

namespace colmap {
namespace mvs {
namespace {

#ifndef RESIDENT_FORECAST_TESTDATA_DIR
#define RESIDENT_FORECAST_TESTDATA_DIR ""
#endif

std::string ReadFileOrDie(const std::string& name) {
  const std::filesystem::path path =
      std::filesystem::path(RESIDENT_FORECAST_TESTDATA_DIR) / name;
  std::ifstream file(path, std::ios::binary);
  if (!file) {
    ADD_FAILURE() << "missing test fixture: " << path;
    return std::string();
  }
  std::string bytes((std::istreambuf_iterator<char>(file)),
                    std::istreambuf_iterator<char>());
  return bytes;
}

// ---------------------------------------------------------------------------
// Test-side helpers.
// ---------------------------------------------------------------------------

// Full FIPS 180-4 SHA-256. Validated below against the known empty-string
// and "abc" digests plus the two committed fixture literals, so a
// transcription error cannot silently corrupt the gate vectors.
class TestSha256Helper {
 public:
  TestSha256Helper() {
    h_ = {0x6a09e667u,
          0xbb67ae85u,
          0x3c6ef372u,
          0xa54ff53au,
          0x510e527fu,
          0x9b05688cu,
          0x1f83d9abu,
          0x5be0cd19u};
    pending_bytes_ = 0;
    length_bits_ = 0;
  }

  void Update(std::string_view data) {
    length_bits_ += static_cast<uint64_t>(data.size()) * 8;
    const char* input = data.data();
    while (!data.empty()) {
      const size_t take =
          std::min(data.size(), 64 - static_cast<size_t>(pending_bytes_));
      std::memcpy(&block_[pending_bytes_], input, take);
      pending_bytes_ += static_cast<uint64_t>(take);
      input += take;
      data.remove_prefix(take);
      if (pending_bytes_ == 64) {
        Compress();
        pending_bytes_ = 0;
      }
    }
  }

  void Final(std::string* out) {
    static const char* kHexDigits = "0123456789abcdef";
    static const unsigned char kPad = 0x80;
    static const unsigned char kZero = 0x00;
    const uint64_t original_bits = length_bits_;
    Update(std::string_view(reinterpret_cast<const char*>(&kPad), 1));
    while (pending_bytes_ != 56) {
      Update(std::string_view(reinterpret_cast<const char*>(&kZero), 1));
    }
    unsigned char tail[8];
    for (int i = 0; i < 8; ++i) {
      tail[i] =
          static_cast<unsigned char>((original_bits >> (8 * (7 - i))) & 0xff);
    }
    Update(std::string_view(reinterpret_cast<const char*>(tail), 8));
    for (const uint32_t word : h_) {
      for (int shift = 24; shift >= 0; shift -= 8) {
        const unsigned char byte =
            static_cast<unsigned char>((word >> shift) & 0xff);
        out->push_back(kHexDigits[(byte >> 4) & 0xf]);
        out->push_back(kHexDigits[byte & 0xf]);
      }
    }
  }

 private:
  static uint32_t Ro(const uint32_t x, const int n) {
    return (x >> n) | (x << (32 - n));
  }
  void Compress() {
    static const uint32_t k[64] = {
        0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu,
        0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u, 0xd807aa98u, 0x12835b01u,
        0x243185beu, 0x550c7dc3u, 0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u,
        0xc19bf174u, 0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu,
        0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau, 0x983e5152u,
        0xa831c66du, 0xb00327c8u, 0xbf597fc7u, 0xc6e00bf3u, 0xd5a79147u,
        0x06ca6351u, 0x14292967u, 0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu,
        0x53380d13u, 0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
        0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u, 0xd192e819u,
        0xd6990624u, 0xf40e3585u, 0x106aa070u, 0x19a4c116u, 0x1e376c08u,
        0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu,
        0x682e6ff3u, 0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
        0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u};
    uint32_t w[64];
    for (int i = 0; i < 16; ++i) {
      w[i] = (static_cast<uint32_t>(block_[4 * i]) << 24) |
             (static_cast<uint32_t>(block_[4 * i + 1]) << 16) |
             (static_cast<uint32_t>(block_[4 * i + 2]) << 8) |
             static_cast<uint32_t>(block_[4 * i + 3]);
    }
    for (int i = 16; i < 64; ++i) {
      const uint32_t s0 =
          Ro(w[i - 15], 7) ^ Ro(w[i - 15], 18) ^ (w[i - 15] >> 3);
      const uint32_t s1 =
          Ro(w[i - 2], 17) ^ Ro(w[i - 2], 19) ^ (w[i - 2] >> 10);
      w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }
    uint32_t a = h_[0], b = h_[1], c = h_[2], d = h_[3];
    uint32_t e = h_[4], f = h_[5], g = h_[6], h = h_[7];
    for (int i = 0; i < 64; ++i) {
      const uint32_t s1 = Ro(e, 6) ^ Ro(e, 11) ^ Ro(e, 25);
      const uint32_t ch = (e & f) ^ (~e & g);
      const uint32_t temp1 = h + s1 + ch + k[i] + w[i];
      const uint32_t s0 = Ro(a, 2) ^ Ro(a, 13) ^ Ro(a, 22);
      const uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
      const uint32_t temp2 = s0 + maj;
      h = g;
      g = f;
      f = e;
      e = d + temp1;
      d = c;
      c = b;
      b = a;
      a = temp1 + temp2;
    }
    h_[0] += a;
    h_[1] += b;
    h_[2] += c;
    h_[3] += d;
    h_[4] += e;
    h_[5] += f;
    h_[6] += g;
    h_[7] += h;
  }

  std::array<uint32_t, 8> h_;
  std::array<unsigned char, 64> block_{};
  size_t pending_bytes_;
  uint64_t length_bits_;
};

// Independent SHA-256 for expected values (fixture shas, positive binding).
std::string TestSha256(const std::string_view& message) {
  TestSha256Helper helper;
  helper.Update(message);
  std::string digest;
  helper.Final(&digest);
  return digest;
}

// Deliberately wrong digest hook for refusal tests.
std::string TestShaWrong(const std::string_view&) {
  return std::string(64, '0');
}

std::string HexEncodeTest(const std::string& bytes) {
  static const char* kHexDigits = "0123456789abcdef";
  std::string out;
  out.reserve(bytes.size() * 2);
  for (const char byte : bytes) {
    out.push_back(kHexDigits[(static_cast<unsigned char>(byte) >> 4) & 0xf]);
    out.push_back(kHexDigits[static_cast<unsigned char>(byte) & 0xf]);
  }
  return out;
}

// Test-side wire builder: a faithful restatement of the committed wire
// schema (fixed keys, fixed order, sequential ordinals). Used for gate
// positives and struct-level mutations; the fixture round trip below pins
// every line against the exact committed bytes.
std::string LineInt(const std::string& key, uint64_t value) {
  return key + " " + std::to_string(value) + "\n";
}
std::string LineBytes(const std::string& key, const std::string& value) {
  return key + " " + std::to_string(value.size()) + " " + HexEncodeTest(value) +
         "\n";
}
std::string LineFingerprint(const std::string& key, const std::string* digest) {
  if (digest == nullptr) return key + " absent\n";
  return LineBytes(key, *digest);
}
std::string BuildWire(const ResidentForecastWire& wire) {
  std::string out = "resident-forecast\n";
  out += LineInt("version", 1);
  out += "contract_sha256 " + wire.contract_sha256 + "\n";
  out += LineInt("contract_version", 1);
  out += LineInt("contract_raw_length", wire.contract_raw.size());
  out += "contract_raw " + std::to_string(wire.contract_raw.size()) + " " +
         HexEncodeTest(wire.contract_raw) + "\n";
  out += LineBytes("qualification", wire.qualification);
  out += LineBytes("completeness", wire.completeness);
  out += LineBytes("concurrency_basis", wire.concurrency_basis);
  out += LineBytes("mode", wire.mode);
  out += LineInt("max_future", static_cast<uint64_t>(wire.max_future));
  out += LineInt("image_channels", static_cast<uint64_t>(wire.image_channels));
  out += "unclassified_allowance_bytes " +
         std::to_string(wire.unclassified_allowance_bytes) + " " +
         std::to_string(wire.unclassified_provenance.size()) + " " +
         HexEncodeTest(wire.unclassified_provenance) + "\n";
  out += LineFingerprint("build_allocation_fingerprint",
                         wire.has_build_allocation_fingerprint
                             ? &wire.build_allocation_fingerprint
                             : nullptr);
  out += LineFingerprint(
      "source_fingerprint",
      wire.has_source_fingerprint ? &wire.source_fingerprint : nullptr);
  out += "charges\n";
  out += "bank host " + std::to_string(wire.bank_host_bytes) + " device " +
         std::to_string(wire.bank_device_bytes) + "\n";
  out += "normal owner " + std::to_string(wire.normal_owner_bytes) +
         " future " + std::to_string(wire.normal_future_bytes) + " transient " +
         std::to_string(wire.normal_transient_bytes) + " charge " +
         std::to_string(wire.normal_charge_bytes) + "\n";
  out += "process mode " + std::to_string(wire.process_mode.size()) + " " +
         HexEncodeTest(wire.process_mode) + " count " +
         std::to_string(wire.process_count) + " host " +
         std::to_string(wire.process_host_bytes) + " device " +
         std::to_string(wire.process_device_bytes) + "\n";
  out += "metadata source " + std::to_string(wire.metadata_source.size()) +
         " " + HexEncodeTest(wire.metadata_source) + " raw " +
         std::to_string(wire.metadata_raw_bytes) + " charge " +
         std::to_string(wire.metadata_charge_bytes) + "\n";
  out += "planner max_member host " +
         std::to_string(wire.planner_max_member_host_bytes) + " device " +
         std::to_string(wire.planner_max_member_device_bytes) + "\n";
  out += "rolling active_jobs " + std::to_string(wire.rolling_active_jobs) +
         " host " + std::to_string(wire.rolling_host_bytes) + " device " +
         std::to_string(wire.rolling_device_bytes) + "\n";
  out += LineInt("reference_count", wire.references.size());
  for (size_t i = 0; i < wire.references.size(); ++i) {
    const auto& reference = wire.references[i];
    out += "reference " + std::to_string(i) + "\n";
    out += LineInt("reference_image_idx",
                   static_cast<uint64_t>(reference.reference_image_idx));
    out += LineBytes("reference_name", reference.reference_name);
    out += LineInt("input_count", reference.input_count);
    out += LineInt("private_host_bytes", reference.private_host_bytes);
    out += LineInt("private_device_bytes", reference.private_device_bytes);
  }
  out += "end\n";
  return out;
}

// Known-good literals for the synthetic contract fixtures, asserted
// against the actual file bytes so a silent fixture drift fails FIRST.
constexpr const char* kTwoContractSha =
    "07cbfee6016bdbd600caac1f77235de6673a17131fef1e3e191d970730f1cbb3";
constexpr const char* kFourContractSha =
    "8cf48120bbf9401c89a55fb59a9ec2b43d85dceab0d0a692116df1b19e161892";

class ResidentForecastFixture : public ::testing::Test {
 protected:
  void SetUp() override {
    ASSERT_FALSE(std::string(RESIDENT_FORECAST_TESTDATA_DIR).empty())
        << "RESIDENT_FORECAST_TESTDATA_DIR is not configured";
    two_contract_ = ReadFileOrDie("two.contract");
    four_contract_ = ReadFileOrDie("four.contract");
    two_wire_ = ReadFileOrDie("two.provisional.forecast");
    four_wire_ = ReadFileOrDie("four.executable.forecast");
    measured_wire_ = ReadFileOrDie("four.executable.measured.forecast");
    allowance_wire_ = ReadFileOrDie("four.executable.allowance.forecast");
    // The independent test SHA-256 must first agree with the known NIST
    // vectors; a transcription error cannot then corrupt gate vectors.
    ASSERT_EQ(
        TestSha256(std::string_view()),
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
    ASSERT_EQ(
        TestSha256("abc"),
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    // The committed wires must bind the committed contract bytes exactly.
    const auto two = ParseResidentForecast(two_wire_, &TestSha256);
    ASSERT_EQ(two.contract_sha256, kTwoContractSha);
    ASSERT_EQ(two.contract_raw, two_contract_);
    const auto four = ParseResidentForecast(four_wire_, &TestSha256);
    ASSERT_EQ(four.contract_sha256, kFourContractSha);
    ASSERT_EQ(four.contract_raw, four_contract_);
  }

  std::string two_contract_, four_contract_, two_wire_, four_wire_;
  std::string measured_wire_, allowance_wire_;
};

// ---------------------------------------------------------------------------
// Parse: positive fixture vectors (literal expected values).
// ---------------------------------------------------------------------------

TEST_F(ResidentForecastFixture, ParseProvisionalTwoWire) {
  const auto wire = ParseResidentForecast(two_wire_, &TestSha256);
  EXPECT_EQ(wire.version, 1);
  EXPECT_EQ(wire.contract_sha256, kTwoContractSha);
  EXPECT_EQ(wire.contract_raw, two_contract_);
  EXPECT_EQ(wire.qualification, "metadata_only_cpu_estimate_provisional");
  EXPECT_EQ(wire.completeness, "partial");
  EXPECT_EQ(wire.concurrency_basis, "rolling_reserve_declared");
  EXPECT_EQ(wire.mode, "bounded");
  EXPECT_EQ(wire.max_future, 2);
  EXPECT_EQ(wire.image_channels, 3);
  EXPECT_EQ(wire.unclassified_allowance_bytes, 0u);
  EXPECT_EQ(wire.unclassified_provenance, "synthetic_declared_terms");
  EXPECT_FALSE(wire.has_build_allocation_fingerprint);
  EXPECT_FALSE(wire.has_source_fingerprint);
  EXPECT_TRUE(wire.build_allocation_fingerprint.empty());
  EXPECT_TRUE(wire.source_fingerprint.empty());
  EXPECT_EQ(wire.bank_host_bytes, 3732u);
  EXPECT_EQ(wire.bank_device_bytes, 5707u);
  EXPECT_EQ(wire.normal_owner_bytes, 120u);
  EXPECT_EQ(wire.normal_future_bytes, 240u);
  EXPECT_EQ(wire.normal_transient_bytes, 120u);
  EXPECT_EQ(wire.normal_charge_bytes, 371u);
  EXPECT_EQ(wire.process_mode, "once");
  EXPECT_EQ(wire.process_count, 1u);
  EXPECT_EQ(wire.process_host_bytes, 2147483648u);
  EXPECT_EQ(wire.process_device_bytes, 2147483648u);
  EXPECT_EQ(wire.metadata_source, "omitted_declared");
  EXPECT_EQ(wire.metadata_raw_bytes, 0u);
  EXPECT_EQ(wire.metadata_charge_bytes, 0u);
  EXPECT_EQ(wire.planner_max_member_host_bytes, 371u);
  EXPECT_EQ(wire.planner_max_member_device_bytes, 3601u);
  EXPECT_EQ(wire.rolling_active_jobs, 1u);
  EXPECT_EQ(wire.rolling_host_bytes, 371u);
  EXPECT_EQ(wire.rolling_device_bytes, 3601u);
  ASSERT_EQ(wire.references.size(), 2u);
  EXPECT_EQ(wire.references[0].reference_image_idx, 1);
  EXPECT_EQ(wire.references[0].reference_name, "synthetic_1.ppm");
  EXPECT_EQ(wire.references[0].input_count, 3u);
  EXPECT_EQ(wire.references[0].private_host_bytes, 371u);
  EXPECT_EQ(wire.references[0].private_device_bytes, 3601u);
  EXPECT_EQ(wire.references[1].reference_image_idx, 2);
  EXPECT_EQ(wire.references[1].reference_name, "synthetic_2.ppm");
  EXPECT_EQ(wire.references[1].input_count, 3u);
  EXPECT_EQ(wire.references[1].private_host_bytes, 223u);
  EXPECT_EQ(wire.references[1].private_device_bytes, 2744u);
}

TEST_F(ResidentForecastFixture, ParseExecutableFourWire) {
  const auto wire = ParseResidentForecast(four_wire_, &TestSha256);
  EXPECT_EQ(wire.qualification, "metadata_only_cpu_estimate");
  EXPECT_EQ(wire.completeness, "complete");
  EXPECT_EQ(wire.concurrency_basis, "rolling_reserve_declared");
  EXPECT_EQ(wire.mode, "bounded");
  EXPECT_EQ(wire.max_future, 2);
  EXPECT_EQ(wire.image_channels, 3);
  EXPECT_TRUE(wire.has_build_allocation_fingerprint);
  EXPECT_TRUE(wire.has_source_fingerprint);
  // Synthetic digest TEXT, not a registered or qualified build.
  const std::string dummy(64, 'a');
  EXPECT_EQ(wire.build_allocation_fingerprint, dummy);
  EXPECT_EQ(wire.source_fingerprint, dummy);
  EXPECT_EQ(wire.bank_host_bytes, 4095u);
  EXPECT_EQ(wire.bank_device_bytes, 5707u);
  EXPECT_EQ(wire.normal_owner_bytes, 144u);
  EXPECT_EQ(wire.normal_future_bytes, 288u);
  EXPECT_EQ(wire.normal_transient_bytes, 144u);
  EXPECT_EQ(wire.normal_charge_bytes, 445u);
  EXPECT_EQ(wire.process_mode, "once");
  EXPECT_EQ(wire.process_count, 1u);
  EXPECT_EQ(wire.process_host_bytes, 2147483648u);
  EXPECT_EQ(wire.process_device_bytes, 2147483648u);
  EXPECT_EQ(wire.metadata_source, "explicit_unclassified");
  EXPECT_EQ(wire.metadata_raw_bytes, 0u);
  EXPECT_EQ(wire.metadata_charge_bytes, 0u);
  EXPECT_EQ(wire.planner_max_member_host_bytes, 445u);
  EXPECT_EQ(wire.planner_max_member_device_bytes, 5064u);
  EXPECT_EQ(wire.rolling_active_jobs, 1u);
  EXPECT_EQ(wire.rolling_host_bytes, 445u);
  EXPECT_EQ(wire.rolling_device_bytes, 5064u);
  ASSERT_EQ(wire.references.size(), 4u);
  const std::array<int, 4> expected_indices{{0, 1, 2, 3}};
  const std::array<const char*, 4> expected_names{{"synthetic_0.ppm",
                                                   "synthetic_1.ppm",
                                                   "synthetic_2.ppm",
                                                   "synthetic_3.ppm"}};
  const std::array<uint64_t, 4> expected_counts{{4, 3, 3, 2}};
  const std::array<uint64_t, 4> expected_hosts{{445u, 371u, 223u, 260u}};
  const std::array<uint64_t, 4> expected_devices{{5064u, 3601u, 2744u, 2058u}};
  for (size_t i = 0; i < 4; ++i) {
    EXPECT_EQ(wire.references[i].reference_image_idx, expected_indices[i]);
    EXPECT_EQ(wire.references[i].reference_name, expected_names[i]);
    EXPECT_EQ(wire.references[i].input_count, expected_counts[i]);
    EXPECT_EQ(wire.references[i].private_host_bytes, expected_hosts[i]);
    EXPECT_EQ(wire.references[i].private_device_bytes, expected_devices[i]);
  }
}

// Exercise the caller_supplied_native_actual declaration with an invented
// metadata inventory: 64 + 3*16 + 25 = 137 raw, 142 after one 3% margin.
// The schema label is not a claim that this fixture measured a native build.
TEST_F(ResidentForecastFixture, ParseMeasuredMetadataFourWire) {
  const auto wire = ParseResidentForecast(measured_wire_, &TestSha256);
  EXPECT_EQ(wire.qualification, "metadata_only_cpu_estimate");
  EXPECT_EQ(wire.completeness, "complete");
  EXPECT_EQ(wire.metadata_source, "caller_supplied_native_actual");
  EXPECT_EQ(wire.metadata_raw_bytes, 137u);
  EXPECT_EQ(wire.metadata_charge_bytes, 142u);
  EXPECT_EQ(owned::MarginBytes(137), 142u);
  EXPECT_EQ(wire.bank_host_bytes, 4095u);
  EXPECT_EQ(wire.bank_device_bytes, 5707u);
  EXPECT_EQ(wire.contract_sha256, kFourContractSha);
  EXPECT_EQ(wire.contract_raw, four_contract_);
  // Byte-exact round trip through the schema-faithful builder.
  EXPECT_EQ(BuildWire(wire), measured_wire_);
  // Ledger terms accept the measured declaration's margin equality.
  EXPECT_NO_THROW(VerifyResidentForecastLedgerTerms(wire, 1));
  // And refuse a perturbed charge departs from the canonical margin.
  auto mutated = wire;
  mutated.metadata_charge_bytes += 1;
  EXPECT_THROW(VerifyResidentForecastLedgerTerms(mutated, 1), std::exception);
}

// Positive-allowance wire: the explicit unclassified allowance (37) is
// composed into the bank host RAW before the single canonical 3% margin
// (4095 -> 4133); provenance recorded; device unchanged.
TEST_F(ResidentForecastFixture, ParseAllowanceFourWire) {
  const auto wire = ParseResidentForecast(allowance_wire_, &TestSha256);
  EXPECT_EQ(wire.qualification, "metadata_only_cpu_estimate");
  EXPECT_EQ(wire.completeness, "complete");
  EXPECT_EQ(wire.unclassified_allowance_bytes, 37u);
  EXPECT_EQ(wire.unclassified_provenance, "synthetic_gap_37_bytes");
  EXPECT_EQ(wire.bank_host_bytes, 4133u);
  EXPECT_EQ(wire.bank_device_bytes, 5707u);
  EXPECT_EQ(wire.metadata_source, "explicit_unclassified");
  EXPECT_EQ(BuildWire(wire), allowance_wire_);
  EXPECT_NO_THROW(VerifyResidentForecastLedgerTerms(wire, 1));
}

// Aggregate budget gate over the synthetic fixture charges (no double billing
// of the planner max-member terms; host metadata/normal charges included).
TEST_F(ResidentForecastFixture, CapsGateOnFixtureCharges) {
  const auto wire = ParseResidentForecast(four_wire_);
  // host total = 4095 + 2147483648 + 0 + 445 + 445 = 2147488633
  const uint64_t host_total = 2147488633ull;
  // device total = 5707 + 2147483648 + 5064 = 2147494419
  const uint64_t device_total = 2147494419ull;
  EXPECT_NO_THROW(
      VerifyResidentForecastFitsCaps(wire, host_total, device_total));
  EXPECT_THROW(
      VerifyResidentForecastFitsCaps(wire, host_total - 1, device_total),
      std::exception);
  EXPECT_THROW(
      VerifyResidentForecastFitsCaps(wire, host_total, device_total - 1),
      std::exception);
  // Max caps always admit the synthetic fixture charges; the aggregate is
  // checked, so NEAR-OVERFLOW charges refuse (checked addition) instead of
  // wrapping into an admission.
  EXPECT_NO_THROW(VerifyResidentForecastFitsCaps(
      wire, 18446744073709551615ull, 18446744073709551615ull));
  auto overcharged = wire;
  overcharged.bank_host_bytes = 18446744073709551615ull;
  EXPECT_THROW(
      VerifyResidentForecastFitsCaps(
          overcharged, 18446744073709551615ull, 18446744073709551615ull),
      std::exception);
}

TEST_F(ResidentForecastFixture, NormalsMode) {
  EXPECT_EQ(ResidentForecastNormalsMode(ParseResidentForecast(two_wire_)),
            ResidentForecastNormals::kBounded);
  EXPECT_EQ(ResidentForecastNormalsMode(ParseResidentForecast(four_wire_)),
            ResidentForecastNormals::kBounded);
  // Unrecognized mode labels must refuse.
  auto mutated = ParseResidentForecast(two_wire_);
  mutated.mode = "phased";
  EXPECT_THROW(ResidentForecastNormalsMode(mutated), std::exception);
  mutated.mode = "eager";
  EXPECT_EQ(ResidentForecastNormalsMode(mutated),
            ResidentForecastNormals::kEager);
}

// The parser must return EXACT native contract bytes, canonically
// re-serializable with the accepted V1 serializer.
// Synthetic record seam: deliberately unequal tiny dimensions and non-sorted
// input orders are independent of any captured dataset or filesystem.
TEST_F(ResidentForecastFixture, SyntheticContractRecords) {
  const auto contract = ParseOwnedBankContract(four_contract_);
  ASSERT_EQ(contract.problems.size(), 4u);
  const std::array<size_t, 4> widths{{4, 5, 3, 7}};
  const std::array<size_t, 4> heights{{3, 2, 2, 1}};
  const std::array<std::vector<int>, 4> orders{
      {{2, 0, 3, 1}, {3, 1, 2}, {1, 2, 0}, {3, 0}}};
  for (size_t p = 0; p < orders.size(); ++p) {
    const auto& problem = contract.problems[p];
    EXPECT_EQ(problem.reference_image_idx, p);
    ASSERT_EQ(problem.inputs.size(), orders[p].size());
    size_t references = 0;
    for (size_t i = 0; i < orders[p].size(); ++i) {
      const auto& input = problem.inputs[i];
      const int idx = orders[p][i];
      EXPECT_EQ(input.image_idx, idx);
      EXPECT_EQ(input.image_name, "synthetic_" + std::to_string(idx) + ".ppm");
      EXPECT_EQ(input.width, widths[idx]);
      EXPECT_EQ(input.height, heights[idx]);
      EXPECT_EQ(input.is_reference, idx == static_cast<int>(p));
      references += input.is_reference;
      EXPECT_EQ(input.image_identity.device, 0u);  // invented, never stat'ed
      EXPECT_EQ(input.image_identity.path, input.image_path);
      EXPECT_EQ(input.depth_header_depth, 1u);
      EXPECT_EQ(input.normal_header_depth, 3u);
    }
    EXPECT_EQ(references, 1u);
  }
}

TEST_F(ResidentForecastFixture, ContractRawIsCanonical) {
  for (const auto& contract_bytes : {two_contract_, four_contract_}) {
    const auto parse = ParseOwnedBankContract(contract_bytes);
    EXPECT_EQ(SerializeOwnedBankContract(parse), contract_bytes);
  }
}

// Parser output rebuilt through the schema-faithful test builder must
// reproduce the exact committed wire bytes (byte-exact round trip).
TEST_F(ResidentForecastFixture, ParseRebuildRoundTrip) {
  EXPECT_EQ(BuildWire(ParseResidentForecast(two_wire_)), two_wire_);
  EXPECT_EQ(BuildWire(ParseResidentForecast(four_wire_)), four_wire_);
}

// ---------------------------------------------------------------------------
// Parse: strict refusals (fixtures, mutated in place where the mutation is
// unambiguous, or rebuilt through the schema-faithful builder).
// ---------------------------------------------------------------------------

TEST_F(ResidentForecastFixture, RefusesEmptyInput) {
  EXPECT_THROW(ParseResidentForecast(std::string()), std::invalid_argument);
}

TEST_F(ResidentForecastFixture, RefusesTruncatedFinalLine) {
  // Dropping the final newline leaves the last line unterminated.
  EXPECT_THROW(
      ParseResidentForecast(four_wire_.substr(0, four_wire_.size() - 1)),
      std::exception);
}

TEST_F(ResidentForecastFixture, RefusesEmptyLine) {
  std::string mutated = two_wire_;
  mutated.insert(mutated.find("qualification"), "\n");
  EXPECT_THROW(ParseResidentForecast(mutated), std::exception);
}

TEST_F(ResidentForecastFixture, RefusesPaddedLine) {
  std::string mutated = two_wire_;
  mutated.insert(mutated.find("qualification"), " \n");
  EXPECT_THROW(ParseResidentForecast(mutated), std::exception);
}

TEST_F(ResidentForecastFixture, RefusesTrailingContent) {
  EXPECT_THROW(ParseResidentForecast(four_wire_ + "extra\n"), std::exception);
}

TEST_F(ResidentForecastFixture, RefusesUnknownKey) {
  std::string mutated = two_wire_;
  mutated.insert(mutated.find("completeness"), "bogus_key 1\n");
  EXPECT_THROW(ParseResidentForecast(mutated), std::exception);
}

TEST_F(ResidentForecastFixture, RefusesOutOfOrderKey) {
  // Swap the qualification/completeness lines (same-length lines).
  std::string mutated = two_wire_;
  const size_t a = mutated.find("qualification");
  const size_t a_end = mutated.find('\n', a) + 1;
  const size_t b = mutated.find("completeness");
  const size_t b_end = mutated.find('\n', b) + 1;
  const std::string qualification = mutated.substr(a, a_end - a);
  const std::string completeness = mutated.substr(b, b_end - b);
  mutated.replace(a, b_end - a, completeness + qualification);
  EXPECT_THROW(ParseResidentForecast(mutated), std::exception);
}

TEST_F(ResidentForecastFixture, RefusesWrongVersion) {
  std::string mutated = two_wire_;
  mutated.replace(mutated.find("version 1\n"), 9, "version 2");
  EXPECT_THROW(ParseResidentForecast(mutated), std::exception);
}

TEST_F(ResidentForecastFixture, RefusesWrongContractVersion) {
  std::string mutated = two_wire_;
  mutated.replace(mutated.find("contract_version 1"), 17, "contract_version 2");
  EXPECT_THROW(ParseResidentForecast(mutated), std::exception);
}

TEST_F(ResidentForecastFixture, RefusesLeadingZeroInteger) {
  std::string mutated = two_wire_;
  mutated.replace(mutated.find("max_future 2\n"), 12, "max_future 02");
  EXPECT_THROW(ParseResidentForecast(mutated), std::exception);
}

TEST_F(ResidentForecastFixture, RefusesSignedCharge) {
  // '-1' is not legal for the unsigned charge field.
  std::string mutated = two_wire_;
  mutated.replace(mutated.find("charge 371"),
                  std::string("charge 371").size(),
                  "charge -1");
  EXPECT_THROW(ParseResidentForecast(mutated), std::exception);
}

TEST_F(ResidentForecastFixture, RefusesPlusInteger) {
  std::string mutated = two_wire_;
  mutated.replace(mutated.find("image_channels 3"), 16, "image_channels +3");
  EXPECT_THROW(ParseResidentForecast(mutated), std::exception);
}

TEST_F(ResidentForecastFixture, RefusesUpperHex) {
  // One uppercase hex digit inside a byte-string payload refuses.
  std::string mutated = two_wire_;
  const size_t hex = mutated.find("qualification 38 ");
  mutated.replace(hex + 17, 2, "6D");
  EXPECT_THROW(ParseResidentForecast(mutated), std::exception);
}

TEST_F(ResidentForecastFixture, RefusesUppercaseShaToken) {
  std::string mutated = two_wire_;
  const size_t sha = mutated.find("contract_sha256 ");
  mutated.replace(sha + 16, 2, "7F");
  EXPECT_THROW(ParseResidentForecast(mutated), std::exception);
}

TEST_F(ResidentForecastFixture, RefusesShortShaToken) {
  std::string mutated = two_wire_;
  const size_t sha = mutated.find("contract_sha256 ");
  mutated.erase(sha + 16 + 16, 1);  // 63 hex chars remain
  EXPECT_THROW(ParseResidentForecast(mutated), std::exception);
}

TEST_F(ResidentForecastFixture, RefusesRawLengthMismatch) {
  std::string mutated = two_wire_;
  mutated.replace(mutated.find("contract_raw_length 8596"),
                  std::string("contract_raw_length 8596").size(),
                  "contract_raw_length 8595");
  EXPECT_THROW(ParseResidentForecast(mutated), std::exception);
}

TEST_F(ResidentForecastFixture, RefusesOversizedRawLength) {
  std::string mutated = two_wire_;
  mutated.replace(mutated.find("contract_raw_length 8596"),
                  std::string("contract_raw_length 8596").size(),
                  "contract_raw_length 18446744073709551615");
  EXPECT_THROW(ParseResidentForecast(mutated), std::exception);
}

TEST_F(ResidentForecastFixture, RefusesRawHashMismatch) {
  // Flip one hex digit of the raw block; with a sha the parse must refuse
  // the wire's declared contract digest binding.
  std::string mutated = two_wire_;
  const size_t length_start =
      mutated.find("contract_raw ") + strlen("contract_raw ");
  const size_t hex_start = mutated.find(' ', length_start) + 1;
  // Keep a valid hex payload and its declared length; only its hash changes.
  mutated[hex_start] = mutated[hex_start] == '6' ? '7' : '6';
  EXPECT_THROW(ParseResidentForecast(mutated, &TestSha256), std::exception);
}

TEST_F(ResidentForecastFixture, RefusesExecutableWithAbsentFingerprints) {
  auto wire = ParseResidentForecast(four_wire_);
  EXPECT_EQ(wire.qualification, "metadata_only_cpu_estimate");
  wire.has_build_allocation_fingerprint = false;
  wire.build_allocation_fingerprint.clear();
  wire.has_source_fingerprint = false;
  wire.source_fingerprint.clear();
  // Same raw/charges; only the labels/fingerprints changed.
  EXPECT_THROW(ParseResidentForecast(BuildWire(wire)), std::exception);
}

TEST_F(ResidentForecastFixture, RefusesDifferingFingerprints) {
  auto wire = ParseResidentForecast(four_wire_);
  wire.source_fingerprint.assign(64, 'b');
  EXPECT_THROW(ParseResidentForecast(BuildWire(wire)), std::exception);
}

TEST_F(ResidentForecastFixture, RefusesNonHexFingerprintPayload) {
  auto wire = ParseResidentForecast(four_wire_);
  wire.source_fingerprint.assign(64, 'z');
  EXPECT_THROW(ParseResidentForecast(BuildWire(wire)), std::exception);
}

TEST_F(ResidentForecastFixture, RefusesShortFingerprintPayload) {
  auto wire = ParseResidentForecast(four_wire_);
  wire.source_fingerprint.assign(63, 'a');
  EXPECT_THROW(ParseResidentForecast(BuildWire(wire)), std::exception);
}

TEST_F(ResidentForecastFixture, RefusesZeroMaxFuture) {
  std::string mutated = two_wire_;
  mutated.replace(mutated.find("max_future 2\n"), 12, "max_future 0");
  EXPECT_THROW(ParseResidentForecast(mutated), std::exception);
}

TEST_F(ResidentForecastFixture, RefusesZeroImageChannels) {
  std::string mutated = two_wire_;
  mutated.replace(mutated.find("image_channels 3"), 16, "image_channels 0");
  EXPECT_THROW(ParseResidentForecast(mutated), std::exception);
}

TEST_F(ResidentForecastFixture, RefusesMaxFutureAbove255) {
  std::string mutated = two_wire_;
  mutated.replace(mutated.find("max_future 2\n"), 12, "max_future 256");
  EXPECT_THROW(ParseResidentForecast(mutated), std::exception);
}

TEST_F(ResidentForecastFixture, RefusesReferenceOrdinalSkip) {
  // Renumbering a reference block away from the contiguous ordinal refuses.
  std::string mutated = two_wire_;
  mutated.replace(mutated.find("reference 1\n"), 11, "reference 2");
  EXPECT_THROW(ParseResidentForecast(mutated), std::exception);
}

TEST_F(ResidentForecastFixture, RefusesDuplicateReferenceOrdinal) {
  std::string mutated = two_wire_;
  mutated.replace(mutated.find("reference 1\n"), 11, "reference 0");
  EXPECT_THROW(ParseResidentForecast(mutated), std::exception);
}

TEST_F(ResidentForecastFixture, RefusesMissingReferenceBlock) {
  // Remove the final reference block; reference_count still says 2.
  std::string mutated = two_wire_;
  const size_t start = mutated.find("reference 1\n");
  const size_t end = mutated.find("end\n", start);
  mutated.erase(start, end - start);
  EXPECT_THROW(ParseResidentForecast(mutated), std::exception);
}

TEST_F(ResidentForecastFixture, RefusesNegativeReferenceImageIdx) {
  auto wire = ParseResidentForecast(two_wire_);
  wire.references[0].reference_image_idx = -1;
  EXPECT_THROW(ParseResidentForecast(BuildWire(wire)), std::exception);
}

TEST_F(ResidentForecastFixture, AcceptsInt32MaxReferenceImageIdx) {
  auto wire = ParseResidentForecast(two_wire_);
  wire.references[0].reference_image_idx = std::numeric_limits<int32_t>::max();
  EXPECT_NO_THROW(ParseResidentForecast(BuildWire(wire)));
}

// ORIGINAL-record binding helpers: the snapshot is an exact copy of the
// contract's stored identities, in problem/input order, with NO filesystem
// access and NO recapture (local harness: pure record semantics; the fresh
// revalidation path runs against real files in the native test surface).
TEST_F(ResidentForecastFixture, ContractJobIdentitiesAreOriginalRecords) {
  auto contract = OwnedBankContract();
  contract.version = kOwnedBankContractVersion;
  const std::function<const owned::BankFileIdentity(const char*)> identity =
      [](const char* name) {
        return owned::BankFileIdentity{/*path=*/std::filesystem::path(name),
                                       /*size=*/42,
                                       /*device=*/123,
                                       /*inode=*/456,
                                       /*modified_seconds=*/1,
                                       /*modified_nanoseconds=*/2,
                                       /*changed_seconds=*/3,
                                       /*changed_nanoseconds=*/4};
      };
  for (int problem = 0; problem < 2; ++problem) {
    OwnedBankContractProblem record;
    record.reference_name = "problem_" + std::to_string(problem);
    record.reference_image_idx = 0;
    const std::string prefix = "p" + std::to_string(problem);
    for (int input = 0; input < 2; ++input) {
      OwnedBankContractInput entry;
      const std::string suffix = prefix + "_i" + std::to_string(input);
      entry.image_path = identity((suffix + "_img").c_str()).path;
      entry.image_identity = identity((suffix + "_img").c_str());
      entry.depth_identity = identity((suffix + "_depth").c_str());
      entry.normal_identity = identity((suffix + "_normal").c_str());
      record.inputs.push_back(std::move(entry));
    }
    contract.problems.push_back(std::move(record));
  }
  const auto snapshot = ContractJobIdentities(contract);
  ASSERT_EQ(snapshot.size(), contract.problems.size());
  for (size_t i = 0; i < snapshot.size(); ++i) {
    ASSERT_EQ(snapshot[i].size(), contract.problems[i].inputs.size());
    for (size_t k = 0; k < snapshot[i].size(); ++k) {
      const auto& record = snapshot[i][k];
      const auto& original = contract.problems[i].inputs[k];
      EXPECT_TRUE(owned::SameIdentity(record.image, original.image_identity));
      EXPECT_TRUE(owned::SameIdentity(record.depth, original.depth_identity));
      EXPECT_TRUE(owned::SameIdentity(record.normal, original.normal_identity));
      EXPECT_EQ(record.image.path, original.image_identity.path);
    }
  }
  // ContractJobIdentities is a pure function: repeated calls return
  // identical records without touching the filesystem.
  const auto again = ContractJobIdentities(contract);
  EXPECT_EQ(again.size(), snapshot.size());
  for (size_t i = 0; i < again.size(); ++i) {
    for (size_t k = 0; k < again[i].size(); ++k) {
      EXPECT_TRUE(owned::SameIdentity(again[i][k].image, snapshot[i][k].image));
      EXPECT_TRUE(owned::SameIdentity(again[i][k].depth, snapshot[i][k].depth));
      EXPECT_TRUE(
          owned::SameIdentity(again[i][k].normal, snapshot[i][k].normal));
    }
  }
}

// ---------------------------------------------------------------------------
// Binding gate.
// ---------------------------------------------------------------------------

TEST_F(ResidentForecastFixture, BindingRequiresSha) {
  const auto wire = ParseResidentForecast(two_wire_);
  // A build without a hash backend must refuse execution, not silently
  // accept a declared digest it cannot check.
  EXPECT_THROW(VerifyResidentForecastBinding(wire, two_contract_, nullptr),
               std::exception);
  // Freshly re-serialized canonical contract bytes equal the fixture bytes.
  EXPECT_NO_THROW(
      VerifyResidentForecastBinding(wire, two_contract_, &TestSha256));
}

TEST_F(ResidentForecastFixture, BindingVerifiesFreshSha) {
  const auto wire = ParseResidentForecast(two_wire_);
  // The fresh bytes hash to the wire's declared digest (independent sha).
  EXPECT_NO_THROW(
      VerifyResidentForecastBinding(wire, two_contract_, &TestSha256));
  // A wrong/failing sha must refuse even though the bytes bind exactly.
  EXPECT_THROW(
      VerifyResidentForecastBinding(wire, two_contract_, &TestShaWrong),
      std::exception);
}

TEST_F(ResidentForecastFixture, BindingRefusesChangedFreshSnapshot) {
  const auto wire = ParseResidentForecast(two_wire_);
  // Length-preserving byte change: a same-shaped fresh snapshot change
  // must refuse before payload.
  std::string changed = two_contract_;
  const size_t name = changed.find("is_reference 1");
  ASSERT_NE(name, std::string::npos);
  changed.replace(name, 1, std::string(1, 'x'));
  EXPECT_THROW(VerifyResidentForecastBinding(wire, changed, nullptr),
               std::exception);
}

TEST_F(ResidentForecastFixture, BindingRefusesChangedIdentityOrInputOrder) {
  const auto wire = ParseResidentForecast(four_wire_, &TestSha256);
  auto changed = ParseOwnedBankContract(four_contract_);
  changed.problems[0].inputs[0].depth_identity.inode += 1;
  EXPECT_THROW(VerifyResidentForecastBinding(
                   wire, SerializeOwnedBankContract(changed), &TestSha256),
               std::exception);
  changed = ParseOwnedBankContract(four_contract_);
  std::swap(changed.problems[0].inputs[0], changed.problems[0].inputs[2]);
  EXPECT_THROW(VerifyResidentForecastBinding(
                   wire, SerializeOwnedBankContract(changed), &TestSha256),
               std::exception);
}

TEST_F(ResidentForecastFixture, BindingRefusesReferenceBlockMismatch) {
  const auto wire = ParseResidentForecast(two_wire_, &TestSha256);
  auto mutated = wire;
  mutated.references[0].reference_name = "synthetic_missing.ppm";
  EXPECT_THROW(
      VerifyResidentForecastBinding(mutated, two_contract_, &TestSha256),
      std::exception);
  mutated = wire;
  mutated.references.resize(1);
  EXPECT_THROW(
      VerifyResidentForecastBinding(mutated, two_contract_, &TestSha256),
      std::exception);
  mutated = wire;
  mutated.references[0].input_count += 1;
  EXPECT_THROW(
      VerifyResidentForecastBinding(mutated, two_contract_, &TestSha256),
      std::exception);
  mutated = wire;
  mutated.references[0].reference_image_idx += 1;
  EXPECT_THROW(
      VerifyResidentForecastBinding(mutated, two_contract_, &TestSha256),
      std::exception);
}

// ---------------------------------------------------------------------------
// Parameter gate.
// ---------------------------------------------------------------------------

std::string SomeCompiledFingerprint() {
#ifdef RESIDENT_FORECAST_HAS_COMPILED_FINGERPRINT
  return kResidentAllocationFingerprint;
#else
  return std::string(64, '0');
#endif
}

// Sentinel digest self-consistency (the REAL compiled-digest positive path
// is exercised by ParametersPositiveWithCompiledDigest under
// RESIDENT_FORECAST_HAS_COMPILED_FINGERPRINT).
TEST_F(ResidentForecastFixture, ParametersPositiveSentinelDigest) {
  const std::string compiled = SomeCompiledFingerprint();
  auto wire = ParseResidentForecast(four_wire_);
  wire.build_allocation_fingerprint = compiled;
  wire.source_fingerprint = compiled;
  EXPECT_NO_THROW(VerifyResidentForecastParameters(wire, compiled));
  // A different compiled fingerprint (unknown build) must refuse.
  EXPECT_THROW(VerifyResidentForecastParameters(wire, std::string(64, 'c')),
               std::exception);
}

#ifdef RESIDENT_FORECAST_HAS_COMPILED_FINGERPRINT
TEST(ResidentForecastCompiledFingerprint, HeaderShape) {
  // The generated header carries exactly one 64-char lowercase hex digest.
  const std::string digest = kResidentAllocationFingerprint;
  ASSERT_EQ(digest.size(), 64u);
  for (const char c : digest) {
    ASSERT_TRUE((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f')) << c;
  }
  // The positive fingerprint path is exercised with the REAL compiled
  // value by the fixture gate test below; there is no placeholder path.
}

TEST_F(ResidentForecastFixture, ParametersPositiveWithCompiledDigest) {
  const auto wire = ParseResidentForecast(four_wire_);
  auto mutated = wire;
  mutated.build_allocation_fingerprint = kResidentAllocationFingerprint;
  mutated.source_fingerprint = kResidentAllocationFingerprint;
  EXPECT_NO_THROW(VerifyResidentForecastParameters(
      mutated, kResidentAllocationFingerprint));
  // Any substitute digest refuses (unknown build).
  EXPECT_THROW(VerifyResidentForecastParameters(mutated, std::string(64, 'c')),
               std::exception);
}
#endif

TEST_F(ResidentForecastFixture, ParametersRefuseEagerExecution) {
  const auto wire = ParseResidentForecast(four_wire_);
  auto eager = wire;
  eager.mode = "eager";
  EXPECT_THROW(VerifyResidentForecastParameters(
                   eager, wire.build_allocation_fingerprint),
               std::exception);
}

TEST_F(ResidentForecastFixture, ParametersRefuseProvisionalWire) {
  const auto wire = ParseResidentForecast(two_wire_);
  EXPECT_THROW(
      VerifyResidentForecastParameters(wire, SomeCompiledFingerprint()),
      std::exception);
}

TEST_F(ResidentForecastFixture, ParametersRefuseAbsentOrUnequalFingerprints) {
  const std::string other(60, 'a');  // never equal to a 64-hex digest
  auto wire = ParseResidentForecast(four_wire_);
  wire.has_source_fingerprint = false;
  wire.source_fingerprint.clear();
  EXPECT_THROW(VerifyResidentForecastParameters(wire, other), std::exception);
  wire = ParseResidentForecast(four_wire_);
  wire.build_allocation_fingerprint.assign(64, 'b');
  EXPECT_THROW(VerifyResidentForecastParameters(wire, other), std::exception);
}

TEST_F(ResidentForecastFixture, ParametersRefuseBadCompletenessOrBasisOrMode) {
  auto wire = ParseResidentForecast(four_wire_);
  wire.completeness = "partial";
  EXPECT_THROW(
      VerifyResidentForecastParameters(wire, wire.build_allocation_fingerprint),
      std::exception);
  wire = ParseResidentForecast(four_wire_);
  wire.concurrency_basis = "bank_only";
  EXPECT_THROW(
      VerifyResidentForecastParameters(wire, wire.build_allocation_fingerprint),
      std::exception);
  wire = ParseResidentForecast(four_wire_);
  wire.mode = "phased";
  EXPECT_THROW(
      VerifyResidentForecastParameters(wire, wire.build_allocation_fingerprint),
      std::exception);
}

// ---------------------------------------------------------------------------
// Ledger-terms gate (wire-internal arithmetic + process shape).
// ---------------------------------------------------------------------------

TEST_F(ResidentForecastFixture, LedgerTermsPositive) {
  const auto wire = ParseResidentForecast(four_wire_);
  EXPECT_NO_THROW(
      VerifyResidentForecastLedgerTerms(wire, /*max_concurrent_solvers=*/1));
  const auto two = ParseResidentForecast(two_wire_);
  EXPECT_NO_THROW(VerifyResidentForecastLedgerTerms(two, 1));
  // Declared concurrency must match the actual CLI concurrency.
  EXPECT_THROW(VerifyResidentForecastLedgerTerms(wire, 2), std::exception);
}

TEST_F(ResidentForecastFixture, LedgerTermsExplicitPairBindsCliAndSum) {
  auto wire = ParseResidentForecast(four_wire_);
  std::vector<uint64_t> hosts;
  std::vector<uint64_t> devices;
  for (const auto& reference : wire.references) {
    hosts.push_back(reference.private_host_bytes);
    devices.push_back(reference.private_device_bytes);
  }
  ASSERT_GE(hosts.size(), 2u);
  std::sort(hosts.begin(), hosts.end(), std::greater<uint64_t>());
  std::sort(devices.begin(), devices.end(), std::greater<uint64_t>());
  wire.rolling_active_jobs = 2;
  wire.rolling_host_bytes = hosts[0] + hosts[1];
  wire.rolling_device_bytes = devices[0] + devices[1];
  EXPECT_NO_THROW(VerifyResidentForecastLedgerTerms(wire, 2));
  EXPECT_THROW(VerifyResidentForecastLedgerTerms(wire, 1), std::exception);
  // A pair declaration cannot carry an N1 max-member reservation.
  wire.rolling_host_bytes = wire.planner_max_member_host_bytes;
  EXPECT_THROW(VerifyResidentForecastLedgerTerms(wire, 2), std::exception);
}

// Extend the parsed contract with a fifth distinct reference and round-trip
// the complete declaration through all binding/parameter gates. These are
// declared CPU bounds, not physical measurements or hardware qualification.
TEST_F(ResidentForecastFixture, OptimizedN3N4N5BindDistinctTopNAndCaps) {
  auto contract = ParseOwnedBankContract(four_contract_);
  auto fifth = contract.problems.back();
  fifth.reference_image_idx = 4;
  fifth.reference_name = "synthetic_4.ppm";
  fifth.output_depth_path = "synthetic/output/4.depth";
  fifth.output_normal_path = "synthetic/output/4.normal";
  for (auto& input : fifth.inputs) {
    if (!input.is_reference) continue;
    input.image_idx = 4;
    input.image_name = fifth.reference_name;
    input.image_path = "synthetic/input/4.image";
    input.depth_path = "synthetic/input/4.depth";
    input.normal_path = "synthetic/input/4.normal";
    input.image_identity.path = input.image_path;
    input.depth_identity.path = input.depth_path;
    input.normal_identity.path = input.normal_path;
  }
  contract.problems.push_back(fifth);
  const auto five_contract = SerializeOwnedBankContract(contract);
  ASSERT_EQ(ParseOwnedBankContract(five_contract).problems.size(), 5u);
  for (const size_t concurrency : {3u, 4u, 5u}) {
    SCOPED_TRACE(concurrency);
    auto wire = ParseResidentForecast(four_wire_, &TestSha256);
    auto reference = wire.references.back();
    reference.reference_image_idx = fifth.reference_image_idx;
    reference.reference_name = fifth.reference_name;
    reference.input_count = fifth.inputs.size();
    wire.references.push_back(reference);
    wire.contract_raw = five_contract;
    wire.contract_sha256 = TestSha256(five_contract);
    ASSERT_EQ(wire.references.size(), 5u);
    // Independent rankings: host top3=2300, top4=2600, all5=2700;
    // device top3=24000, top4=30000, all5=31000.
    const uint64_t hosts[] = {900, 600, 300, 800, 100};
    const uint64_t devices[] = {7000, 9000, 8000, 6000, 1000};
    for (size_t i = 0; i < 5; ++i) {
      wire.references[i].private_host_bytes = hosts[i];
      wire.references[i].private_device_bytes = devices[i];
    }
    wire.planner_max_member_host_bytes = 900;
    wire.planner_max_member_device_bytes = 9000;
    wire.rolling_active_jobs = concurrency;
    wire.rolling_host_bytes = concurrency == 3   ? 2300
                              : concurrency == 4 ? 2600
                                                 : 2700;
    wire.rolling_device_bytes = concurrency == 3   ? 24000
                                : concurrency == 4 ? 30000
                                                   : 31000;
    // Parse the COMPLETE N3/N4/N5 declaration, including presence flags, rather
    // than exercising only a post-parse in-memory mutation.
    wire = ParseResidentForecast(BuildWire(wire), &TestSha256);
    ASSERT_TRUE(wire.has_build_allocation_fingerprint);
    ASSERT_TRUE(wire.has_source_fingerprint);
    ASSERT_EQ(wire.rolling_active_jobs, concurrency);
    EXPECT_NO_THROW(ValidateOptimizedStereoConcurrency(
        true, std::to_string(concurrency), true, false));
    EXPECT_NO_THROW(
        VerifyResidentForecastBinding(wire, five_contract, &TestSha256));
    EXPECT_NO_THROW(VerifyResidentForecastParameters(
        wire, wire.build_allocation_fingerprint));
    EXPECT_NO_THROW(VerifyResidentForecastLedgerTerms(wire, concurrency));
    EXPECT_THROW(VerifyResidentForecastLedgerTerms(wire, concurrency - 1),
                 std::exception);
    // Shared bank/process/normal producer charged once, no extra margin.
    const uint64_t host_cap = concurrency == 3   ? 2147490488ull
                              : concurrency == 4 ? 2147490788ull
                                                 : 2147490888ull;
    const uint64_t device_cap = concurrency == 3   ? 2147513355ull
                                : concurrency == 4 ? 2147519355ull
                                                   : 2147520355ull;
    EXPECT_NO_THROW(VerifyResidentForecastFitsCaps(wire, host_cap, device_cap));
    EXPECT_THROW(VerifyResidentForecastFitsCaps(wire, host_cap - 1, device_cap),
                 std::exception);
    EXPECT_THROW(VerifyResidentForecastFitsCaps(wire, host_cap, device_cap - 1),
                 std::exception);
    for (const bool missing_build : {true, false}) {
      auto absent = wire;
      if (missing_build)
        absent.has_build_allocation_fingerprint = false;
      else
        absent.has_source_fingerprint = false;
      EXPECT_THROW(ParseResidentForecast(BuildWire(absent), &TestSha256),
                   std::exception);
      EXPECT_THROW(VerifyResidentForecastParameters(
                       absent, wire.build_allocation_fingerprint),
                   std::exception);
    }
    auto bad = wire;
    --bad.rolling_host_bytes;
    EXPECT_THROW(VerifyResidentForecastLedgerTerms(bad, concurrency),
                 std::exception);
    bad = wire;
    --bad.rolling_device_bytes;
    EXPECT_THROW(VerifyResidentForecastLedgerTerms(bad, concurrency),
                 std::exception);
    // Checked top-N addition must reject overflow on either independent
    // ranking, even if an attacker declares the wrapped reserve.
    for (const bool host : {true, false}) {
      bad = wire;
      if (host) {
        bad.references[0].private_host_bytes = UINT64_MAX;
        bad.planner_max_member_host_bytes = UINT64_MAX;
        bad.rolling_host_bytes = concurrency == 3   ? 1399
                                 : concurrency == 4 ? 1699
                                                    : 1799;
      } else {
        bad.references[0].private_device_bytes = UINT64_MAX;
        bad.planner_max_member_device_bytes = UINT64_MAX;
        bad.rolling_device_bytes = concurrency == 3   ? 16999
                                   : concurrency == 4 ? 22999
                                                      : 23999;
      }
      bad = ParseResidentForecast(BuildWire(bad), &TestSha256);
      EXPECT_THROW(VerifyResidentForecastLedgerTerms(bad, concurrency),
                   std::exception);
      // Even an unlimited cap cannot make an overflowing aggregate fit.
      bad = wire;
      if (host)
        bad.bank_host_bytes = UINT64_MAX;
      else
        bad.bank_device_bytes = UINT64_MAX;
      bad = ParseResidentForecast(BuildWire(bad), &TestSha256);
      EXPECT_THROW(VerifyResidentForecastFitsCaps(bad, UINT64_MAX, UINT64_MAX),
                   std::exception);
    }
    // Fewer jobs than N must refuse, not silently clamp effective N.
    bad = wire;
    bad.references.resize(concurrency - 1);
    EXPECT_THROW(VerifyResidentForecastLedgerTerms(bad, concurrency),
                 std::exception);
    bad = wire;
    bad.references[1] = bad.references[0];
    EXPECT_THROW(VerifyResidentForecastBinding(bad, five_contract, &TestSha256),
                 std::exception);
  }
}

TEST_F(ResidentForecastFixture, LedgerTermsRefuseInconsistentPlanner) {
  auto wire = ParseResidentForecast(four_wire_);
  wire.planner_max_member_host_bytes += 1;
  EXPECT_THROW(
      VerifyResidentForecastLedgerTerms(wire, wire.rolling_active_jobs),
      std::exception);
}

TEST_F(ResidentForecastFixture, LedgerTermsRefuseInconsistentRolling) {
  auto wire = ParseResidentForecast(four_wire_);
  wire.rolling_device_bytes += 1;
  EXPECT_THROW(
      VerifyResidentForecastLedgerTerms(wire, wire.rolling_active_jobs),
      std::exception);
}

TEST_F(ResidentForecastFixture, LedgerTermsRefuseProcessShape) {
  auto wire = ParseResidentForecast(four_wire_);
  wire.process_mode = "per_owner";
  EXPECT_THROW(
      VerifyResidentForecastLedgerTerms(wire, wire.rolling_active_jobs),
      std::exception);
  wire = ParseResidentForecast(four_wire_);
  wire.process_count = 0;
  EXPECT_THROW(
      VerifyResidentForecastLedgerTerms(wire, wire.rolling_active_jobs),
      std::exception);
  wire = ParseResidentForecast(four_wire_);
  wire.process_host_bytes = 0;
  EXPECT_THROW(
      VerifyResidentForecastLedgerTerms(wire, wire.rolling_active_jobs),
      std::exception);
  wire = ParseResidentForecast(four_wire_);
  wire.process_device_bytes = 0;
  EXPECT_THROW(
      VerifyResidentForecastLedgerTerms(wire, wire.rolling_active_jobs),
      std::exception);
}

TEST_F(ResidentForecastFixture, LedgerTermsRefuseProcessBelowCanonicalFloor) {
  auto wire = ParseResidentForecast(four_wire_);
  wire.process_host_bytes = 2147483647u;
  EXPECT_THROW(
      VerifyResidentForecastLedgerTerms(wire, wire.rolling_active_jobs),
      std::exception);
  wire = ParseResidentForecast(four_wire_);
  wire.process_device_bytes = 2147483647u;
  EXPECT_THROW(
      VerifyResidentForecastLedgerTerms(wire, wire.rolling_active_jobs),
      std::exception);
  wire = ParseResidentForecast(four_wire_);
  wire.process_host_bytes = 2147483648u;
  wire.process_device_bytes = 2147483648u;
  EXPECT_NO_THROW(
      VerifyResidentForecastLedgerTerms(wire, wire.rolling_active_jobs));
}

TEST_F(ResidentForecastFixture, LedgerTermsRefuseBankOnlyScope) {
  auto wire = ParseResidentForecast(four_wire_);
  wire.rolling_active_jobs = 0;
  wire.rolling_host_bytes = 0;
  wire.rolling_device_bytes = 0;
  // The basis label still claims a rolling reserve: refuse regardless.
  wire.concurrency_basis = "rolling_reserve_declared";
  EXPECT_THROW(
      VerifyResidentForecastLedgerTerms(wire, wire.rolling_active_jobs),
      std::exception);
}

TEST_F(ResidentForecastFixture, LedgerTermsRefuseActiveJobsAboveReferences) {
  const auto wire = ParseResidentForecast(two_wire_);
  // Declared concurrency mismatch refusal (equality with the CLI value).
  EXPECT_THROW(VerifyResidentForecastLedgerTerms(wire, 3), std::exception);
  // active_jobs above the declared reference count refuses outright.
  auto mutated = wire;
  mutated.rolling_active_jobs = 5;
  mutated.rolling_host_bytes = 371u + 223u;
  mutated.rolling_device_bytes = 3601u + 2744u;
  EXPECT_THROW(VerifyResidentForecastLedgerTerms(mutated, 5), std::exception);
}

TEST_F(ResidentForecastFixture, LedgerTermsRefusePartialWithNonOmittedSource) {
  auto wire = ParseResidentForecast(four_wire_);
  wire.completeness = "partial";
  wire.metadata_source = "caller_supplied_native_actual";
  wire.metadata_raw_bytes = 1000;
  wire.metadata_charge_bytes = owned::MarginBytes(1000);
  // partial completeness is the omitted-metadata declaration only.
  EXPECT_THROW(VerifyResidentForecastLedgerTerms(wire, 1), std::exception);
}

TEST_F(ResidentForecastFixture, LedgerTermsAcceptsEagerMaxFutureDeclared) {
  auto wire = ParseResidentForecast(four_wire_);
  wire.mode = "eager";
  EXPECT_NO_THROW(VerifyResidentForecastLedgerTerms(wire, 1));
}

TEST_F(ResidentForecastFixture, LedgerTermsRefuseMaxFutureOutsideFifo) {
  auto wire = ParseResidentForecast(four_wire_);
  ASSERT_EQ(wire.max_future, 2);
  wire.max_future = 3;
  EXPECT_THROW(VerifyResidentForecastLedgerTerms(wire, 1), std::exception);
  wire.max_future = 0;
  EXPECT_THROW(VerifyResidentForecastLedgerTerms(wire, 1), std::exception);
}

TEST_F(ResidentForecastFixture, LedgerTermsValidateMetadataTerms) {
  // unknown metadata source refuses; known non-omitted sources must carry
  // charge == the single 3% margin (MarginBytes) of the declared raw bytes.
  auto wire = ParseResidentForecast(four_wire_);
  wire.metadata_source = "invented_source";
  EXPECT_THROW(VerifyResidentForecastLedgerTerms(wire, 1), std::exception);

  wire = ParseResidentForecast(four_wire_);
  wire.metadata_source = "caller_supplied_native_actual";
  wire.metadata_raw_bytes = 1000000;
  wire.metadata_charge_bytes = owned::MarginBytes(1000000);
  EXPECT_NO_THROW(VerifyResidentForecastLedgerTerms(wire, 1));
  wire.metadata_charge_bytes = owned::MarginBytes(1000000) - 1;
  EXPECT_THROW(VerifyResidentForecastLedgerTerms(wire, 1), std::exception);

  wire = ParseResidentForecast(four_wire_);
  wire.metadata_source = "explicit_unclassified";
  wire.metadata_raw_bytes = 999999;
  wire.metadata_charge_bytes = owned::MarginBytes(999999);
  EXPECT_NO_THROW(VerifyResidentForecastLedgerTerms(wire, 1));
  wire.metadata_charge_bytes = owned::MarginBytes(999999) + 1;
  EXPECT_THROW(VerifyResidentForecastLedgerTerms(wire, 1), std::exception);

  // omitted_declared is only legal on partial forecasts.
  const auto two = ParseResidentForecast(two_wire_);
  EXPECT_NO_THROW(VerifyResidentForecastLedgerTerms(two, 1));
  wire = ParseResidentForecast(four_wire_);
  wire.metadata_source = "omitted_declared";  // complete + omitted contradicts
  EXPECT_THROW(VerifyResidentForecastLedgerTerms(wire, 1), std::exception);
  wire.completeness = "partial";
  // partial + omitted must also carry zero raw/charge: nonzero raw refuses.
  wire.metadata_raw_bytes = 1;
  EXPECT_THROW(VerifyResidentForecastLedgerTerms(wire, 1), std::exception);
  wire.metadata_raw_bytes = 0;
  EXPECT_NO_THROW(VerifyResidentForecastLedgerTerms(wire, 1));
}

TEST_F(ResidentForecastFixture, LedgerTermsRefuseEmptyProvenance) {
  auto wire = ParseResidentForecast(four_wire_);
  wire.unclassified_provenance = std::string();
  // An empty provenance cannot even be serialized canonically by the
  // committed schema (trailing-space padded line), so rebuilding refuses.
  EXPECT_THROW(ParseResidentForecast(BuildWire(wire)), std::exception);
}

// ---------------------------------------------------------------------------
// Charge gate (hand-set native minima; the estimator recompute is covered by
// resident_forecast_native_test.cc in the dense-stereo build).
// ---------------------------------------------------------------------------

// Hand-set minima for the charge-gate logic tests. The RAW terms are set
// so the allowance-composed margined minimum stays 0 (raw 0 -> margin 0) —
// the composition itself is exercised by ChargesGateRefusesBankBelowMinimum
// and the allowance fixture below; the estimator-filling of the raw terms
// is the native minima test's surface.
ResidentLedgerMinima HandMinimaFor(const ResidentForecastWire& wire) {
  ResidentLedgerMinima minima;
  minima.bank_host_raw_bytes = 0;
  minima.bank_device_raw_bytes = 0;
  minima.bank_host_bytes = 1;
  minima.bank_device_bytes = 1;
  minima.normal_owner_bytes = wire.normal_owner_bytes;
  minima.normal_future_bytes = wire.normal_future_bytes;
  minima.normal_transient_bytes = wire.normal_transient_bytes;
  minima.private_minima.assign(wire.references.size(), {1, 1});
  return minima;
}

TEST_F(ResidentForecastFixture, ChargesGateAcceptsAdequateCharges) {
  const auto wire = ParseResidentForecast(four_wire_);
  EXPECT_NO_THROW(VerifyResidentForecastCharges(wire, HandMinimaFor(wire)));
}

TEST_F(ResidentForecastFixture, ChargesGateRefusesBankBelowMinimum) {
  // The host minimum composes the allowance into the RAW before the single
  // margin. For the synthetic allowance fixture (allowance 37, bank host
  // 4133), the largest acceptable RAW is derived independently:
  // MarginBytes(3975 + 37) == 4133.
  auto wire = ParseResidentForecast(allowance_wire_);
  ASSERT_EQ(wire.bank_host_bytes, 4133ull);
  auto minima = HandMinimaFor(wire);
  minima.bank_host_raw_bytes = 3975ull;
  EXPECT_NO_THROW(VerifyResidentForecastCharges(wire, minima));
  // One raw byte above the composed maximum refuses.
  minima.bank_host_raw_bytes = 3976ull;
  EXPECT_THROW(VerifyResidentForecastCharges(wire, minima), std::exception);
  // The device raw never takes the allowance (device charge unchanged by
  // the allowance): its composed maximum is MarginBytes^-1(5707).
  minima = HandMinimaFor(wire);
  minima.bank_device_raw_bytes = 5540ull;  // MarginBytes(5540) == 5707
  EXPECT_NO_THROW(VerifyResidentForecastCharges(wire, minima));
  minima.bank_device_raw_bytes = 5541ull;
  EXPECT_THROW(VerifyResidentForecastCharges(wire, minima), std::exception);
  // The zero-raw allowance-free minimum stays admission-safe for any wire.
  minima = HandMinimaFor(wire);  // raw 0, allowance-free minimum 0
  EXPECT_NO_THROW(VerifyResidentForecastCharges(wire, minima));
}

TEST_F(ResidentForecastFixture, ChargesGateRefusesNormalMismatch) {
  auto wire = ParseResidentForecast(four_wire_);
  auto minima = HandMinimaFor(wire);
  minima.normal_owner_bytes += 1;
  EXPECT_THROW(VerifyResidentForecastCharges(wire, minima), std::exception);
  minima = HandMinimaFor(wire);
  minima.normal_future_bytes += 1;
  EXPECT_THROW(VerifyResidentForecastCharges(wire, minima), std::exception);
  minima = HandMinimaFor(wire);
  minima.normal_transient_bytes += 1;
  EXPECT_THROW(VerifyResidentForecastCharges(wire, minima), std::exception);
}

TEST_F(ResidentForecastFixture, ChargesGateRefusesBoundedChargeBelowAllowance) {
  auto wire = ParseResidentForecast(four_wire_);
  // Bounded charge must cover future + transient, without overflowing.
  wire.normal_charge_bytes =
      wire.normal_future_bytes + wire.normal_transient_bytes;
  EXPECT_NO_THROW(VerifyResidentForecastCharges(wire, HandMinimaFor(wire)));
  wire.normal_charge_bytes -= 1;
  EXPECT_THROW(VerifyResidentForecastCharges(wire, HandMinimaFor(wire)),
               std::exception);
}

TEST_F(ResidentForecastFixture, ChargesGateRefusesEagerMode) {
  auto wire = ParseResidentForecast(four_wire_);
  wire.mode = "eager";
  // Bounded-only execution: an eager declaration refuses regardless of
  // its charges; no eager-mode accounting exists on the native side.
  EXPECT_THROW(VerifyResidentForecastCharges(wire, HandMinimaFor(wire)),
               std::exception);
}

TEST_F(ResidentForecastFixture, ChargesGateRefusesPrivateBelowMinimum) {
  auto wire = ParseResidentForecast(four_wire_);
  auto minima = HandMinimaFor(wire);
  minima.private_minima[2] = {wire.references[2].private_host_bytes + 1, 1};
  EXPECT_THROW(VerifyResidentForecastCharges(wire, minima), std::exception);
  minima = HandMinimaFor(wire);
  minima.private_minima[2] = {1, wire.references[2].private_device_bytes + 1};
  EXPECT_THROW(VerifyResidentForecastCharges(wire, minima), std::exception);
  minima = HandMinimaFor(wire);
  minima.private_minima[2] = {1, 1};
  EXPECT_NO_THROW(VerifyResidentForecastCharges(wire, minima));
}

TEST_F(ResidentForecastFixture, ChargesGateRefusesMinimaCountMismatch) {
  auto wire = ParseResidentForecast(four_wire_);
  auto minima = HandMinimaFor(wire);
  minima.private_minima.resize(wire.references.size() - 1);
  EXPECT_THROW(VerifyResidentForecastCharges(wire, minima), std::exception);
}

}  // namespace
}  // namespace mvs
}  // namespace colmap