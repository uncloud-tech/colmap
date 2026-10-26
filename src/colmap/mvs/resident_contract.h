// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "colmap/mvs/owned_input_bank.h"
#include "colmap/mvs/patch_match_options.h"

#include <array>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace colmap {
namespace mvs {

// Versioned canonical contract for the resident owned-bank metadata slice.
// The contract records ONE native invocation snapshot: the resolved problems
// in native (FlatHashSet-preserving) source order, their ordered inputs, file
// identities, map headers, calibration and options, in exact representations.
// It is metadata only: no decoded payloads are serialized and the contract is
// a binding record, never a certification of executability or quality.
constexpr int kOwnedBankContractVersion = 1;

// Identity of one ordered input to one problem, in native resolution order.
struct OwnedBankContractInput {
  // True for the reference image (it appears in the ordered inputs exactly
  // once, at its native iteration position, matching the resolved BankJob).
  bool is_reference = false;
  int image_idx = -1;
  // Stable reference name (model image name), never reordered or canonicalized.
  std::string image_name;
  size_t width = 0;
  size_t height = 0;
  // Calibration metadata exactly as bound into the solver problem metadata.
  std::array<float, 9> K{};
  std::array<float, 9> R{};
  std::array<float, 3> T{};
  int max_image_size = -1;
  std::filesystem::path image_path;
  std::filesystem::path depth_path;
  std::filesystem::path normal_path;
  owned::BankFileIdentity image_identity{};
  owned::BankFileIdentity depth_identity{};
  owned::BankFileIdentity normal_identity{};
  // Float-map headers (width, height, channels), read header-only.
  size_t depth_header_width = 0;
  size_t depth_header_height = 0;
  size_t depth_header_depth = 0;
  size_t normal_header_width = 0;
  size_t normal_header_height = 0;
  size_t normal_header_depth = 0;
};

// One resolved problem in native problem order. All per-problem fields are
// the effective invocation values, including derived depth ranges, sigma
// defaults and the clamped filter_min_num_consistent.
struct OwnedBankContractProblem {
  std::string reference_name;
  int reference_image_idx = -1;
  PatchMatchOptions effective_options;
  std::filesystem::path output_depth_path;
  std::filesystem::path output_normal_path;
  std::vector<OwnedBankContractInput> inputs;
};

// Whole-invocation contract. The build identity uses only macros that exist
// in the compiled translation unit, and always states allocation
// compatibility unqualified; compiled source fingerprints are deliberately
// not fabricated here (a future generated build identity is a separate lane).
struct OwnedBankContract {
  int version = kOwnedBankContractVersion;
  bool build_cuda_enabled = false;
  bool build_hip_enabled = false;
  std::string allocation_compatibility = "unqualified-cpu-only";
  PatchMatchOptions invocation_options;
  std::string stereo_folder;
  std::vector<OwnedBankContractProblem> problems;
};

// Fill the invocation-invariant contract fields (version, build identity).
OwnedBankContract MakeOwnedBankContract(
    const PatchMatchOptions& invocation_options,
    const std::string& stereo_folder);

// Capture the enriched contract record for one resolved problem. The
// invocation snapshot (input file identities, map headers, calibration,
// dimensions and effective options) is captured here; reference and input
// names come from the model in native form. CPU-only: stat reads plus
// header-only map reads, no payload decode.
OwnedBankContractProblem CaptureOwnedBankContractProblem(
    const owned::BankJob& job, const std::vector<std::string>& image_names);

// Deterministic canonical serialization (versioned line protocol):
// every string/path as (byte length, hex bytes), every float/double as its
// exact bit representation in lowercase hex, all counts as decimal integers.
std::string SerializeOwnedBankContract(const OwnedBankContract& contract);

// Strict canonical parser. Rejects unsupported versions, truncated or
// trailing content, unknown or duplicate keys, malformed values, counts
// inconsistent with the record layout. Throws std::invalid_argument (or
// std::runtime_error for I/O-shaped problems) with the offending line.
OwnedBankContract ParseOwnedBankContract(const std::string& bytes);

// Index of the first differing byte, or string::npos when equal.
size_t FirstOwnedBankContractByteDifference(const std::string& a,
                                            const std::string& b);

}  // namespace mvs
}  // namespace colmap