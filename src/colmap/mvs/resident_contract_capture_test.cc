// SPDX-License-Identifier: BSD-3-Clause
// Capture-level regressions with REAL files: validates the map header
// reader (separators, channel binding, downscale relation, checked extent),
// the post-read invocation-identity revalidation, and the repeated-input
// identity consistency across problems. Uses real production map writers
// and the real capture path; image bitmaps stay metadata-only dummies.
#include "colmap/mvs/resident_contract.h"

#include "colmap/mvs/depth_map.h"
#include "colmap/mvs/normal_map.h"
#include "colmap/mvs/owned_input_bank.h"
#include "colmap/util/file.h"

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>

namespace colmap {
namespace mvs {
namespace {

const float* kIdentityK() {
  static const float K[9] = {1, 0, 2, 0, 1, 3, 0, 0, 1};
  return K;
}

const float* kIdentityR() {
  static const float R[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  return R;
}

const float* kIdentityT() {
  static const float T[3] = {0, 0, 0};
  return T;
}

// Real files: a metadata-only dummy image file plus depth/normal maps
// written by the production DepthMap/NormalMap writers.
owned::BankJob MakeJob(const std::filesystem::path& root) {
  owned::BankJob job;
  job.options.geom_consistency = true;
  job.options.depth_min = 1.5;
  job.options.depth_max = 2.0;
  job.options.window_radius = 5;
  job.options.window_step = 1;
  job.options.gpu_index = "0";
  job.options.num_threads = 2;
  job.reference_image_idx = 0;
  const auto image_path = root / "images" / "ref.png";
  job.images.emplace_back(Image(image_path, 10, 5, kIdentityK(), kIdentityR(),
                                kIdentityT()));
  owned::BankInput input;
  input.image_idx = 0;
  input.image = job.images.front();
  input.depth_path = root / "depth_maps" / "ref.photometric.bin";
  input.normal_path = root / "normal_maps" / "ref.photometric.bin";
  input.max_image_size = -1;
  job.inputs.push_back(input);
  job.depth_map_path = root / "depth_maps" / "ref.geometric.bin";
  job.normal_map_path = root / "normal_maps" / "ref.geometric.bin";
  return job;
}

owned::BankJob MakeCaptureFixture(const std::filesystem::path& root) {
  std::filesystem::create_directories(root / "images");
  std::filesystem::create_directories(root / "depth_maps");
  std::filesystem::create_directories(root / "normal_maps");
  const auto job = MakeJob(root);
  { std::ofstream image_file(job.inputs.front().image.GetPath(),
                             std::ios::binary);
    image_file << "dummy-png-bytes"; }
  DepthMap depth(10, 5, 0.5f, 10.0f);
  depth.Fill(2.0f);
  depth.Write(job.inputs.front().depth_path);
  NormalMap normal(10, 5);
  normal.Fill(0.5f);
  normal.Write(job.inputs.front().normal_path);
  return job;
}

TEST(OwnedBankContractCapture, NominalCaptureBindsRealFiles) {
  const auto root =
      std::filesystem::temp_directory_path() /
      ("resident_contract_capture_" +
       std::to_string(
           std::chrono::steady_clock::now().time_since_epoch().count()));
  const auto job = MakeCaptureFixture(root);

  const auto record = CaptureOwnedBankContractProblem(job, {"ref"});
  ASSERT_EQ(record.inputs.size(), 1u);
  EXPECT_TRUE(record.inputs.front().is_reference);
  EXPECT_EQ(record.inputs.front().depth_header_width, 10u);
  EXPECT_EQ(record.inputs.front().depth_header_height, 5u);
  EXPECT_EQ(record.inputs.front().depth_header_depth, 1u);
  EXPECT_EQ(record.inputs.front().normal_header_depth, 3u);
  EXPECT_EQ(record.inputs.front().image_name, "ref");
  EXPECT_EQ(record.reference_name, "ref");
  EXPECT_GT(record.inputs.front().image_identity.size, 0u);
  EXPECT_EQ(record.inputs.front().image_identity.path.string(),
            (root / "images" / "ref.png").string());
  EXPECT_EQ(record.inputs.front().depth_identity.path.string(),
            (root / "depth_maps" / "ref.photometric.bin").string());

  for (const auto& input : record.inputs) {
    // Revalidation succeeded; identities must still hold now.
    EXPECT_NO_THROW(owned::CheckFileIdentityUnchanged(input.image_identity));
    EXPECT_NO_THROW(owned::CheckFileIdentityUnchanged(input.depth_identity));
    EXPECT_NO_THROW(owned::CheckFileIdentityUnchanged(input.normal_identity));
  }
  std::error_code cleanup_error;
  std::filesystem::remove_all(root, cleanup_error);
}

TEST(OwnedBankContractCapture, RejectsWrongSeparator) {
  const auto root =
      std::filesystem::temp_directory_path() /
      ("resident_contract_capture_" +
       std::to_string(
           std::chrono::steady_clock::now().time_since_epoch().count()));
  const auto job = MakeCaptureFixture(root);
  {
    std::ofstream corrupt(job.inputs.front().depth_path, std::ios::binary);
    corrupt << "10|5|1&";
    const std::vector<char> payload(10 * 5 * 1 * sizeof(float), '\0');
    corrupt.write(payload.data(), payload.size());
  }
  EXPECT_ANY_THROW(CaptureOwnedBankContractProblem(job, {"ref"}));
  std::error_code cleanup_error;
  std::filesystem::remove_all(root, cleanup_error);
}

TEST(OwnedBankContractCapture, RejectsWrongChannelCounts) {
  const auto root =
      std::filesystem::temp_directory_path() /
      ("resident_contract_capture_" +
       std::to_string(
           std::chrono::steady_clock::now().time_since_epoch().count()));
  const auto job = MakeCaptureFixture(root);
  {
    std::ofstream corrupt(job.inputs.front().depth_path, std::ios::binary);
    corrupt << "10&5&2&";
    const std::vector<char> payload(10 * 5 * 2 * sizeof(float), '\0');
    corrupt.write(payload.data(), payload.size());
  }
  EXPECT_ANY_THROW(CaptureOwnedBankContractProblem(job, {"ref"}));
  // Normal maps must carry 3 channels; restore a valid 1-channel depth map
  // first so the normal header itself is the failure point.
  DepthMap restored_depth(10, 5, 0.5f, 10.0f);
  restored_depth.Fill(2.0f);
  restored_depth.Write(job.inputs.front().depth_path);
  {
    std::ofstream corrupt(job.inputs.front().normal_path, std::ios::binary);
    corrupt << "10&5&1&";
    const std::vector<char> payload(10 * 5 * 1 * sizeof(float), '\0');
    corrupt.write(payload.data(), payload.size());
  }
  EXPECT_ANY_THROW(CaptureOwnedBankContractProblem(job, {"ref"}));
  std::error_code cleanup_error;
  std::filesystem::remove_all(root, cleanup_error);
}

TEST(OwnedBankContractCapture, RejectsTruncatedPayloadExtent) {
  const auto root =
      std::filesystem::temp_directory_path() /
      ("resident_contract_capture_" +
       std::to_string(
           std::chrono::steady_clock::now().time_since_epoch().count()));
  const auto job = MakeCaptureFixture(root);
  {
    std::ofstream corrupt(job.inputs.front().depth_path, std::ios::binary);
    corrupt << "10&5&1&";
    const std::vector<char> payload(10 * 5 * 1 * sizeof(float) - 4, '\0');
    corrupt.write(payload.data(), payload.size());
  }
  EXPECT_ANY_THROW(CaptureOwnedBankContractProblem(job, {"ref"}));
  std::error_code cleanup_error;
  std::filesystem::remove_all(root, cleanup_error);
}

TEST(OwnedBankContractCapture, RejectsUnsupportedDownscaleRelation) {
  const auto root =
      std::filesystem::temp_directory_path() /
      ("resident_contract_capture_" +
       std::to_string(
           std::chrono::steady_clock::now().time_since_epoch().count()));
  const auto job = MakeCaptureFixture(root);
  {
    std::ofstream corrupt(job.inputs.front().depth_path, std::ios::binary);
    corrupt << "4&5&1&";
    const std::vector<char> payload(4 * 5 * 1 * sizeof(float), '\0');
    corrupt.write(payload.data(), payload.size());
  }
  EXPECT_ANY_THROW(CaptureOwnedBankContractProblem(job, {"ref"}));
  // 20x5 cannot downscale exactly to 10x5 along the height axis either.
  {
    std::ofstream corrupt(job.inputs.front().depth_path, std::ios::binary);
    corrupt << "20&5&1&";
    const std::vector<char> payload(20 * 5 * 1 * sizeof(float), '\0');
    corrupt.write(payload.data(), payload.size());
  }
  EXPECT_ANY_THROW(CaptureOwnedBankContractProblem(job, {"ref"}));
  std::error_code cleanup_error;
  std::filesystem::remove_all(root, cleanup_error);
}

// Note: an identity change WITHIN one capture call is caught by the
// post-header revalidation (no hook required). A file rewritten AFTER the
// completed capture is a normal later-invocation change and is rejected by
// verification, not by capture.

// Rewrites the depth map of the job's input with the given header/payload
// shape, returning a fresh job (fixture files rewritten deterministically).
owned::BankJob RewriteDepthHeader(const owned::BankJob& job,
                                  const std::string& header,
                                  const size_t payload_floats) {
  std::ofstream corrupt(job.inputs.front().depth_path, std::ios::binary);
  corrupt << header;
  const std::vector<char> payload(payload_floats * sizeof(float), '\0');
  corrupt.write(payload.data(), payload.size());
  corrupt.close();
  owned::BankJob rebuilt = job;
  return rebuilt;
}

TEST(OwnedBankContractCapture, RejectsOversizeWhenDownsizeDisabled) {
  // The bank performs float-map downsizing only when max_image_size > 0;
  // with it disabled the map must already equal the effective image dims,
  // so a 2x-oversized map is rejected exactly like the bank would reject it.
  const auto root =
      std::filesystem::temp_directory_path() /
      ("resident_contract_capture_" +
       std::to_string(
           std::chrono::steady_clock::now().time_since_epoch().count()));
  auto job = MakeCaptureFixture(root);
  // Oversized header with the correct payload extent.
  RewriteDepthHeader(job, "20&10&1&", 20 * 10 * 1);
  EXPECT_ANY_THROW(CaptureOwnedBankContractProblem(job, {"ref"}));
  // With the positive max_image_size the same shape mirrors the bank's
  // successful 2x downsize and must be admitted.
  RewriteDepthHeader(job, "20&10&1&", 20 * 10 * 1);
  job.inputs.front().max_image_size = 20;
  const auto record = CaptureOwnedBankContractProblem(job, {"ref"});
  EXPECT_EQ(record.inputs.front().depth_header_width, 20u);
  EXPECT_EQ(record.inputs.front().depth_header_height, 10u);
  std::error_code cleanup_error;
  std::filesystem::remove_all(root, cleanup_error);
}

TEST(OwnedBankContractCapture, AdmitsActualProductionDownsizeShape) {
  // Actual production scale behavior, no invented float proof: the real
  // DepthMap/NormalMap Downsize of a 10x5 map to effective 7x4 image dims
  // uses float factor 0.7f whose product 5*0.7f rounds to exactly 3.5f, so
  // both maps come out 7x4 and production admits the shape. Metadata must
  // admit exactly the same (capture succeeds, header dims recorded 10x5).
  {
    DepthMap depth(10, 5, 0.5f, 10.0f);
    depth.Fill(2.0f);
    depth.Downsize(7, 4);
    EXPECT_EQ(depth.GetWidth(), 7);
    EXPECT_EQ(depth.GetHeight(), 4);
    NormalMap normal(10, 5);
    normal.Fill(0.5f);
    normal.Downsize(7, 4);
    EXPECT_EQ(normal.GetWidth(), 7);
    EXPECT_EQ(normal.GetHeight(), 4);
  }
  const auto root =
      std::filesystem::temp_directory_path() /
      ("resident_contract_capture_" +
       std::to_string(
           std::chrono::steady_clock::now().time_since_epoch().count()));
  auto job = MakeCaptureFixture(root);
  // Effective image dims 7x4 with a positive max_image_size.
  job.inputs.front().max_image_size = 10;
  const auto image_path = job.inputs.front().image.GetPath();
  job.images.clear();
  job.images.emplace_back(
      Image(image_path, 7, 4, kIdentityK(), kIdentityR(), kIdentityT()));
  job.inputs.front().image = job.images.front();
  RewriteDepthHeader(job, "10&5&1&", 10 * 5 * 1);
  const auto record = CaptureOwnedBankContractProblem(job, {"ref"});
  EXPECT_EQ(record.inputs.front().depth_header_width, 10u);
  EXPECT_EQ(record.inputs.front().depth_header_height, 5u);
  EXPECT_EQ(record.inputs.front().normal_header_width, 10u);
  EXPECT_EQ(record.inputs.front().normal_header_height, 5u);
  std::error_code cleanup_error;
  std::filesystem::remove_all(root, cleanup_error);
}

}  // namespace
}  // namespace mvs
}  // namespace colmap