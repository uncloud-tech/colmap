// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/mvs/resident_contract.h"

#include "colmap/mvs/owned_input_bank.h"
#include "colmap/util/logging.h"
#include "colmap/util/misc.h"

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

#include <gtest/gtest.h>

namespace colmap {
namespace mvs {
namespace {

PatchMatchOptions TestInvocationOptions() {
  PatchMatchOptions options;
  options.depth_min = -1.0;
  options.depth_max = -1.0;
  options.sigma_spatial = 2.0;  // hand-checkable double bit pattern
  options.window_radius = 5;
  options.window_step = 1;
  options.num_threads = 2;
  options.geom_consistency = true;
  options.gpu_index = "0";
  return options;
}

owned::BankFileIdentity TestIdentity(const std::filesystem::path& path,
                                     const uintmax_t size) {
  return {path, size, 123, 456, 10, 20, 30, 40};
}

OwnedBankContractInput TestInput(const int image_idx,
                                 const std::string& name,
                                 const bool is_reference) {
  OwnedBankContractInput input;
  input.is_reference = is_reference;
  input.image_idx = image_idx;
  input.image_name = name;
  input.width = 4;
  input.height = 2;
  for (int i = 0; i < 9; ++i) input.K[i] = static_cast<float>(i);
  for (int i = 0; i < 9; ++i) input.R[i] = static_cast<float>(9 + i);
  input.T = {0.5f, -0.5f, 1.5f};
  input.max_image_size = -1;
  input.image_path = "/ws/images/" + name;
  input.depth_path = "/ws/depth_maps/" + name + ".geometric.bin";
  input.normal_path = "/ws/normal_maps/" + name + ".geometric.bin";
  input.image_identity = TestIdentity(input.image_path, 100);
  input.depth_identity = TestIdentity(input.depth_path, 200);
  input.normal_identity = TestIdentity(input.normal_path, 300);
  input.depth_header_width = 4;
  input.depth_header_height = 2;
  input.depth_header_depth = 1;
  input.normal_header_width = 4;
  input.normal_header_height = 2;
  input.normal_header_depth = 3;
  return input;
}

// Two problems in native order. Problem 0's reference sits in the FIRST
// ordered-input slot and problem 1's in the SECOND slot, both of which native
// FlatHashSet iteration can and does produce; the contract must serialize
// this unspecial native order verbatim (no sorting/canonicalization).
OwnedBankContract TestContract() {
  OwnedBankContract contract;
  contract.build_cuda_enabled = false;
  contract.build_hip_enabled = false;
  contract.invocation_options = TestInvocationOptions();
  contract.stereo_folder = "stereo";

  OwnedBankContractProblem problem0;
  problem0.reference_name = "ref";
  problem0.reference_image_idx = 0;
  problem0.effective_options = contract.invocation_options;
  problem0.effective_options.depth_min = 1.5;
  problem0.effective_options.depth_max = 2.0;
  problem0.effective_options.filter_min_num_consistent = 1;
  problem0.output_depth_path = "/ws/stereo/depth_maps/ref.geometric.bin";
  problem0.output_normal_path = "/ws/stereo/normal_maps/ref.geometric.bin";
  problem0.inputs = {TestInput(0, "ref", true), TestInput(2, "b", false),
                     TestInput(1, "a", false)};

  OwnedBankContractProblem problem1;
  problem1.reference_name = "ref2";
  problem1.reference_image_idx = 1;
  problem1.effective_options = contract.invocation_options;
  problem1.effective_options.depth_min = 3.5;
  problem1.effective_options.depth_max = 4.0;
  problem1.output_depth_path = "/ws/stereo/depth_maps/ref2.geometric.bin";
  problem1.output_normal_path = "/ws/stereo/normal_maps/ref2.geometric.bin";
  problem1.inputs = {TestInput(3, "c", false), TestInput(1, "ref2", true),
                     TestInput(2, "b", false), TestInput(0, "ref", false)};

  contract.problems = {problem0, problem1};
  return contract;
}

std::string FindRequiredLine(const std::string& bytes, const std::string& key,
                             const size_t occurrence = 0) {
  const std::string needle = "\n" + key;
  size_t pos = std::string::npos;
  for (size_t i = 0; i <= occurrence; ++i) {
    pos = bytes.find(needle, pos == std::string::npos ? 0 : pos + 1);
    EXPECT_NE(pos, std::string::npos) << "missing key occurrence " << i;
    if (pos == std::string::npos) return "";
  }
  const size_t end = bytes.find('\n', pos + 1);
  return bytes.substr(pos + 1, end - (pos + 1));
}

std::string ReplaceLineOnce(const std::string& bytes,
                            const std::string& old_line,
                            const std::string& new_line) {
  const size_t pos = bytes.find("\n" + old_line + "\n");
  EXPECT_NE(pos, std::string::npos) << "missing line " << old_line;
  if (pos == std::string::npos) return bytes;
  return bytes.substr(0, pos + 1) + new_line + "\n" +
         bytes.substr(pos + old_line.size() + 2);
}

TEST(OwnedBankContract, ExactBitRepresentationsAndHexStrings) {
  const auto bytes = SerializeOwnedBankContract(TestContract());
  // Hand-computed IEEE-754 bit patterns (independent truth). The first
  // occurrence of each option key is the invocation options block (-1.0
  // defaults for the derived depth range), the second is problem 0's
  // effective block.
  EXPECT_EQ(FindRequiredLine(bytes, "opt_double depth_min"),
            "opt_double depth_min bff0000000000000");
  EXPECT_EQ(FindRequiredLine(bytes, "opt_double depth_min", 1),
            "opt_double depth_min 3ff8000000000000");
  EXPECT_EQ(FindRequiredLine(bytes, "opt_double depth_max", 1),
            "opt_double depth_max 4000000000000000");
  EXPECT_EQ(FindRequiredLine(bytes, "opt_double sigma_spatial"),
            "opt_double sigma_spatial 4000000000000000");
  EXPECT_EQ(FindRequiredLine(bytes, "opt_string gpu_index"),
            "opt_string gpu_index 1 30");
  EXPECT_EQ(bytes.substr(0, bytes.find('\n')), "owned-bank-contract");
  EXPECT_EQ(FindRequiredLine(bytes, "version"), "version 1");
  EXPECT_EQ(FindRequiredLine(bytes, "build_cuda_enabled"),
            "build_cuda_enabled 0");
  EXPECT_EQ(FindRequiredLine(bytes, "build_hip_enabled"),
            "build_hip_enabled 0");
  // Honest unqualified stability marker.
  EXPECT_EQ(FindRequiredLine(bytes, "allocation_compatibility"),
            "allocation_compatibility 20 "
            "756e7175616c69666965642d6370752d6f6e6c79");
  EXPECT_EQ(FindRequiredLine(bytes, "reference_name"),
            "reference_name 3 726566");
  // Calibration floats as exact 8-hex-digit bit patterns.
  EXPECT_EQ(
      FindRequiredLine(bytes, "calib_k"),
      "calib_k 00000000 3f800000 40000000 40400000 40800000 40a00000 "
      "40c00000 40e00000 41000000");
  EXPECT_EQ(
      FindRequiredLine(bytes, "calib_r"),
      "calib_r 41100000 41200000 41300000 41400000 41500000 41600000 "
      "41700000 41800000 41880000");
  EXPECT_EQ(FindRequiredLine(bytes, "calib_t"),
            "calib_t 3f000000 bf000000 3fc00000");
}

TEST(OwnedBankContract, RoundtripByteExactWithNativeOrder) {
  const auto contract = TestContract();
  const auto first = SerializeOwnedBankContract(contract);
  // Deterministic: repeated serialization gives identical bytes.
  const auto second = SerializeOwnedBankContract(contract);
  EXPECT_EQ(first, second);

  const auto parsed = ParseOwnedBankContract(first);
  EXPECT_EQ(SerializeOwnedBankContract(parsed), first);
  EXPECT_EQ(FindRequiredLine(SerializeOwnedBankContract(parsed),
                             "opt_double depth_min", 1),
            "opt_double depth_min 3ff8000000000000");
  // Input ordinals must NOT be renumbered or sorted anywhere.
  EXPECT_EQ(first, SerializeOwnedBankContract(ParseOwnedBankContract(first)));

  ASSERT_EQ(parsed.problems.size(), 2);
  // Problem 0 native order: ref, b, a.
  ASSERT_EQ(parsed.problems[0].inputs.size(), 3);
  EXPECT_EQ(parsed.problems[0].reference_name, "ref");
  EXPECT_EQ(parsed.problems[0].reference_image_idx, 0);
  EXPECT_TRUE(parsed.problems[0].inputs[0].is_reference);
  EXPECT_EQ(parsed.problems[0].inputs[0].image_name, "ref");
  EXPECT_EQ(parsed.problems[0].inputs[1].image_name, "b");
  EXPECT_EQ(parsed.problems[0].inputs[2].image_name, "a");
  // Per-problem effective options bind the derived depth range and clamped
  // filter minimum, distinct from the invocation options.
  EXPECT_EQ(parsed.problems[0].effective_options.depth_min, 1.5);
  EXPECT_EQ(parsed.problems[0].effective_options.depth_max, 2.0);
  EXPECT_EQ(parsed.problems[0].effective_options.filter_min_num_consistent, 1);
  EXPECT_EQ(parsed.invocation_options.depth_min, -1.0);
  EXPECT_EQ(parsed.invocation_options.num_threads, 2);
  // Problem 1 native order: c, ref2, b, ref (ref2 in the second slot).
  ASSERT_EQ(parsed.problems[1].inputs.size(), 4);
  EXPECT_FALSE(parsed.problems[1].inputs[0].is_reference);
  EXPECT_EQ(parsed.problems[1].inputs[0].image_name, "c");
  EXPECT_TRUE(parsed.problems[1].inputs[1].is_reference);
  EXPECT_EQ(parsed.problems[1].inputs[1].image_name, "ref2");
  EXPECT_EQ(parsed.problems[1].effective_options.depth_min, 3.5);
  // Invocation snapshot identity values survive the roundtrip.
  EXPECT_EQ(parsed.problems[0].inputs[0].image_identity.path.string(),
            "/ws/images/ref");
  EXPECT_EQ(parsed.problems[0].inputs[0].image_identity.size, 100u);
  EXPECT_EQ(parsed.problems[0].inputs[0].image_identity.device, 123u);
  EXPECT_EQ(parsed.problems[0].inputs[0].image_identity.inode, 456u);
  EXPECT_EQ(parsed.problems[0].inputs[0].image_identity.modified_seconds, 10);
  EXPECT_EQ(parsed.problems[0].inputs[0].image_identity.modified_nanoseconds,
            20);
  EXPECT_EQ(parsed.problems[0].inputs[0].image_identity.changed_seconds, 30);
  EXPECT_EQ(parsed.problems[0].inputs[0].image_identity.changed_nanoseconds,
            40);
  EXPECT_EQ(parsed.problems[0].inputs[0].depth_header_height, 2u);
  EXPECT_EQ(parsed.problems[0].inputs[1].normal_header_depth, 3u);
  // Stereo folder and build identity fields.
  EXPECT_EQ(parsed.stereo_folder, "stereo");
  EXPECT_FALSE(parsed.build_cuda_enabled);
  EXPECT_FALSE(parsed.build_hip_enabled);
  EXPECT_EQ(parsed.allocation_compatibility, "unqualified-cpu-only");
}

TEST(OwnedBankContract, OrderSensitiveCanonicalBytes) {
  const auto contract = TestContract();
  const auto base = SerializeOwnedBankContract(contract);
  // Reordering the ordered inputs changes the canonical bytes: native order
  // is serialized verbatim, never canonicalized or sorted.
  auto reordered = contract;
  std::swap(reordered.problems.front().inputs[0],
            reordered.problems.front().inputs[1]);
  EXPECT_NE(base, SerializeOwnedBankContract(reordered));
  // A changed option value changes the canonical bytes.
  auto changed_options = contract;
  changed_options.invocation_options.num_iterations = 7;
  EXPECT_NE(base, SerializeOwnedBankContract(changed_options));
  // A changed file identity (size) changes the canonical bytes.
  auto changed_identity = contract;
  changed_identity.problems[0]
      .inputs[1]
      .depth_identity.size += 1;
  EXPECT_NE(base, SerializeOwnedBankContract(changed_identity));
}

TEST(OwnedBankContract, ByteDifferenceLocator) {
  EXPECT_EQ(FirstOwnedBankContractByteDifference("abc", "abc"),
            std::string::npos);
  EXPECT_EQ(FirstOwnedBankContractByteDifference("abc", "abd"), 2u);
  EXPECT_EQ(FirstOwnedBankContractByteDifference("abc", "abcd"), 3u);
  EXPECT_EQ(FirstOwnedBankContractByteDifference("", "abc"), 0u);
}

TEST(OwnedBankContract, RejectsMalformedAndTruncatedContracts) {
  const auto valid = SerializeOwnedBankContract(TestContract());

  // Empty input.
  EXPECT_ANY_THROW(ParseOwnedBankContract(""));

  // Truncated final line (last '\n' removed).
  EXPECT_ANY_THROW(ParseOwnedBankContract(valid.substr(0, valid.size() - 1)));

  // Prefix truncations.
  EXPECT_ANY_THROW(ParseOwnedBankContract(valid.substr(0, 40)));

  // Trailing garbage after a complete contract.
  EXPECT_ANY_THROW(ParseOwnedBankContract(valid + "surprise 1\n"));
  // Trailing blank line.
  EXPECT_ANY_THROW(ParseOwnedBankContract(valid + "\n"));

  // Unsupported version.
  EXPECT_ANY_THROW(
      ParseOwnedBankContract(ReplaceLineOnce(valid, "version 1", "version 2")));

  // Wrong problem count.
  EXPECT_ANY_THROW(ParseOwnedBankContract(
      ReplaceLineOnce(valid, "problem_count 2", "problem_count 3")));

  // Non-sequential problem ordinal.
  EXPECT_ANY_THROW(ParseOwnedBankContract(
      ReplaceLineOnce(valid, "problem 0", "problem 1")));

  // Invalid hex content in a string field.
  EXPECT_ANY_THROW(ParseOwnedBankContract(ReplaceLineOnce(
      valid, "reference_name 3 726566", "reference_name 3 7265fg")));

  // Hex length does not match the declared byte length.
  EXPECT_ANY_THROW(ParseOwnedBankContract(ReplaceLineOnce(
      valid, "reference_name 3 726566", "reference_name 3 7265")));

  // Unknown line injected into the record layout.
  EXPECT_ANY_THROW(ParseOwnedBankContract(ReplaceLineOnce(
      valid, "version 1", "version 1\nunknown_key 42")));

  // Corrupted options line (short hex double token).
  EXPECT_ANY_THROW(ParseOwnedBankContract(ReplaceLineOnce(
      valid,
      "opt_double depth_min 3ff8000000000000",
      "opt_double depth_min 3ff800000000")));

  // Invalid boolean.
  EXPECT_ANY_THROW(ParseOwnedBankContract(
      ReplaceLineOnce(valid, "is_reference 1", "is_reference 2")));

  // Non-canonical line endings.
  std::string crlf;
  for (const char c : valid) {
    if (c == '\n') {
      crlf += "\r\n";
    } else {
      crlf += c;
    }
  }
  EXPECT_ANY_THROW(ParseOwnedBankContract(crlf));

  // Header replaced entirely (missing banner).
  EXPECT_ANY_THROW(ParseOwnedBankContract(valid.substr(valid.find("version "))));
}

TEST(OwnedBankContract, CanonicalStrictnessRejectsNoncanonicalSpellings) {
  const auto valid = SerializeOwnedBankContract(TestContract());

  // Leading-zero decimal spellings.
  EXPECT_ANY_THROW(ParseOwnedBankContract(
      ReplaceLineOnce(valid, "problem_count 2", "problem_count 02")));
  EXPECT_ANY_THROW(ParseOwnedBankContract(
      ReplaceLineOnce(valid, "input_count 3", "input_count 03")));
  EXPECT_ANY_THROW(ParseOwnedBankContract(
      ReplaceLineOnce(valid, "reference_image_idx 0", "reference_image_idx 00")));

  // Plus-signed and explicitly signed unsigned fields.
  EXPECT_ANY_THROW(ParseOwnedBankContract(
      ReplaceLineOnce(valid, "problem_count 2", "problem_count +2")));
  EXPECT_ANY_THROW(ParseOwnedBankContract(
      ReplaceLineOnce(valid, "version 1", "version +1")));

  // A negative integer can never appear in an unsigned field.
  EXPECT_ANY_THROW(ParseOwnedBankContract(ReplaceLineOnce(
      valid, "problem_count 2", "problem_count -1")));

  // Uppercase hex is non-canonical and rejected.
  EXPECT_ANY_THROW(ParseOwnedBankContract(ReplaceLineOnce(
      valid, "reference_name 3 726566", "reference_name 3 72656F")));

  // Zero-prefixed signed negative.
  EXPECT_ANY_THROW(ParseOwnedBankContract(ReplaceLineOnce(
      valid, "opt_int max_image_size -1", "opt_int max_image_size -01")));
}

TEST(OwnedBankContract, UnsignedIdentityMaxValueRoundtrips) {
  auto contract = TestContract();
  // Both identity fields that serialize as uintmax_t must round-trip their
  // full declared range, including the maximum unsigned value.
  for (auto& problem : contract.problems) {
    for (auto& input : problem.inputs) {
      for (auto* identity : {&input.image_identity, &input.depth_identity,
                             &input.normal_identity}) {
        identity->device = std::numeric_limits<uintmax_t>::max();
        identity->inode = std::numeric_limits<uintmax_t>::max();
      }
    }
  }
  const auto bytes = SerializeOwnedBankContract(contract);
  const auto parsed = ParseOwnedBankContract(bytes);
  EXPECT_EQ(SerializeOwnedBankContract(parsed), bytes);
  EXPECT_EQ(parsed.problems.front().inputs.front().image_identity.device,
            std::numeric_limits<uintmax_t>::max());
  EXPECT_EQ(parsed.problems.front().inputs.front().image_identity.inode,
            std::numeric_limits<uintmax_t>::max());
}

TEST(OwnedBankContract, BoundedCountsRejectInfeasibleDeclarations) {
  const auto valid = SerializeOwnedBankContract(TestContract());
  // A declared count that cannot possibly be backed by the remaining bytes
  // is rejected before any reserve(), including near-overflow spellings.
  EXPECT_ANY_THROW(ParseOwnedBankContract(ReplaceLineOnce(
      valid, "problem_count 2", "problem_count 999999999")));
  EXPECT_ANY_THROW(ParseOwnedBankContract(ReplaceLineOnce(
      valid, "input_count 3", "input_count 999999999999")));
}

}  // namespace
}  // namespace mvs
}  // namespace colmap