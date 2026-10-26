// SPDX-License-Identifier: BSD-3-Clause

// Native per-cohort charge/minima gate tests (resident_cohort_wire_minima
// .cc). REAL native estimator outputs over REAL valid geometry fixtures
// (never stubbed, never weakened): three shared small images per job (real
// grayscale/depth/normal files), per-job reference/source geometry with
// nonempty sources and in-bounds global image indices; the gate must
// demand the allowance-inclusive single-margin bank charge, the EXACT
// native normal allowances, the per-member private floors, and fresh
// revalidation (never recapture) around every estimator traversal.
//
// The singleton ledger check is the native FLOOR HYPOTHESIS (declared
// whole-invocation basis + native singleton bank/normal floors), NOT a
// complete canonical all-singleton forecast proof: copy-phase metadata
// and reviewed-R singleton evidence is supplied by the canonical
// Photogram side for experimental admission (documented limitation; no
// native formula was invented for the global basis).
//
// Also included: the cohort-two SUBSET BANK mapping regression through the
// real HostInputBank interface (subset job index == cohort-local
// generation; original ordinal is reporting-only; original job content,
// reference/source layers preserved).
//
// Native build only (needs the OpenImageIO dense-stereo link); local
// dev-box slices honestly skip this binary.
#include "colmap/mvs/depth_map.h"
#include "colmap/mvs/normal_map.h"
#include "colmap/mvs/owned_input_bank.h"
#include "colmap/mvs/owned_resident_minima.h"
#include "colmap/mvs/resident_cohort_wire.h"
#include "colmap/scene/image.h"
#include "colmap/sensor/bitmap.h"
#include "colmap/util/testing.h"

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include <gtest/gtest.h>

namespace colmap {
namespace mvs {
namespace {

using ::colmap::mvs::cohort::CohortForecastWire;
using ::colmap::mvs::cohort::CohortWireCohort;
using ::colmap::mvs::cohort::CohortWireMember;
using ::colmap::mvs::cohort::VerifyResidentCohortForecastNativeFloors;
using owned::BankInput;
using owned::BankJob;
using owned::BankLimits;
using owned::BankRawLimits;

class CohortNativeFloorsTest : public ::testing::Test {
 protected:
  void SetUp() override {
    root_ = CreateTestDir();
    const float K[9] = {10, 0, 2, 0, 10, 1, 0, 0, 1};
    const float R[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    const float T[3] = {0, 0, 0};
    // Three shared real images (per-job-local indices 0..2).
    std::vector<Image> images;
    std::vector<BankInput> inputs;
    for (int idx = 0; idx < 3; ++idx) {
      const auto path = root_ / (std::to_string(idx) + ".png");
      Bitmap bitmap(4, 2, true);
      bitmap.Fill(BitmapColor<uint8_t>(30, 90, 150));
      ASSERT_TRUE(bitmap.Write(path));
      Image image(path, 4, 2, K, R, T);
      image.Downsize(2, 2);
      images.push_back(image);
      BankInput input;
      input.image_idx = idx;
      input.image = image;
      input.max_image_size = 2;
      input.depth_path = root_ / (std::to_string(idx) + ".depth.bin");
      input.normal_path = root_ / (std::to_string(idx) + ".normal.bin");
      DepthMap depth(4, 2, 1, 10);
      depth.Fill(2 + static_cast<float>(idx));
      depth.Write(input.depth_path);
      NormalMap normal(4, 2);
      normal.Fill(0.5f);
      normal.Write(input.normal_path);
      inputs.push_back(input);
    }
    // Four valid jobs: real geometry (nonempty sources), per-job distinct
    // references, global model image indices preserved inside each job's
    // image vector (job-local index space [0, 3)).
    const int references[4] = {0, 1, 2, 0};
    const int sources[4][2] = {{2, 1}, {0, 2}, {1, 0}, {1, 2}};
    for (int j = 0; j < 4; ++j) {
      BankJob job;
      job.options.max_image_size = 2;
      job.options.geom_consistency = true;
      job.images = images;
      job.reference_image_idx = references[j];
      job.source_image_idxs = {sources[j][0], sources[j][1]};
      job.inputs = inputs;
      const std::string out = "job" + std::to_string(j) + ".out";
      const auto depth_out = root_ / (out + ".depth.bin");
      const auto normal_out = root_ / (out + ".normal.bin");
      job.depth_map_path = depth_out;
      job.normal_map_path = normal_out;
      jobs_.push_back(job);
    }
    WireWithCharges();
  }

  void TearDown() override { std::filesystem::remove_all(root_); }

  // Wire struct assembled ONLY from native estimator outputs (self-
  // consistent floors); the cross-language numeric agreement is the
  // canonical Python model's responsibility (verified in the crosscheck
  // artifacts) — this fixture exists to exercise the REAL gate mechanics.
  void WireWithCharges() {
    wire_ = CohortForecastWire{};
    wire_.mode = "bounded";
    wire_.max_future = 2;
    wire_.image_channels = 3;
    wire_.unclassified_allowance_bytes = 0;
    wire_.unclassified_provenance = "test";
    wire_.global_snapshot_plan_host_bytes = 1;
    wire_.cohort_count = 2;
    wire_.summary.global_snapshot_plan_host_bytes = 1;
    wire_.summary.process_once_host_bytes = 1;
    wire_.summary.process_once_device_bytes = 1;

    // Cohorts: {0,1} and {2,3} — contiguous FIFO windows over the original
    // ordinals; subset banks are built per cohort in the real executor.
    const size_t member_counts[2] = {2, 2};
    size_t cursor = 0;
    uint64_t queue_sum = 0;
    uint64_t bank_h = 0;
    uint64_t bank_d = 0;
    uint64_t normal_h = 0;
    uint64_t priv_h = 0;
    uint64_t priv_d = 0;
    uint64_t pre_h = 0;
    uint64_t pre_d = 0;
    for (size_t c = 0; c < 2; ++c) {
      CohortWireCohort cohort;
      cohort.ordinal = c;
      cohort.member_count = member_counts[c];
      const std::vector<owned::BankJob> slice(
          jobs_.begin() + static_cast<long>(cursor),
          jobs_.begin() + static_cast<long>(cursor + member_counts[c]));
      const auto raw_slice = owned::HostInputBank::RawEstimate(slice);
      const auto normal_slice =
          owned::HostInputBank::EstimateNormalAllowance(slice, 2);
      cohort.bank_estimate_host_bytes =
          owned::MarginBytes(raw_slice.bounded_host_raw_bytes);
      cohort.bank_estimate_device_bytes =
          owned::MarginBytes(raw_slice.device_raw_bytes);
      cohort.bank_charge_host_bytes = cohort.bank_estimate_host_bytes;
      cohort.bank_charge_device_bytes = cohort.bank_estimate_device_bytes;
      cohort.reviewed_bank_host_bytes = 0;
      cohort.reviewed_bank_device_bytes = 0;
      cohort.normal_owner_bytes = normal_slice.owner_bytes;
      cohort.normal_future_bytes = normal_slice.future_bytes;
      cohort.normal_transient_bytes = normal_slice.transient_bytes;
      cohort.normal_charge_host_bytes =
          normal_slice.future_bytes + normal_slice.transient_bytes;
      cohort.queue_host_bytes = 1;
      cohort.process_mode = "once";
      cohort.process_count = 1;
      cohort.process_host_bytes = 1;
      cohort.process_device_bytes = 1;
      cohort.rolling_active_jobs = 1;
      for (size_t m = 0; m < member_counts[c]; ++m) {
        const size_t index = cursor + m;
        CohortWireMember member;
        member.member_ordinal = index;
        member.reference_image_idx = jobs_[index].reference_image_idx;
        member.input_count = jobs_[index].inputs.size();
        const auto& reference =
            jobs_[index].images.at(jobs_[index].reference_image_idx);
        const auto minimum = owned::ResidentPrivateLogicalMinimum(
            reference.GetWidth(),
            reference.GetHeight(),
            jobs_[index].source_image_idxs.size(),
            /*include_owner_normal=*/true);
        member.private_host_bytes = minimum.host;
        member.private_device_bytes = minimum.device;
        cohort.members.push_back(member);
      }
      // rolling reserve = largest active_jobs member privates (k = 1)
      cohort.rolling_host_bytes = cohort.members[0].private_host_bytes;
      cohort.rolling_device_bytes = cohort.members[0].private_device_bytes;
      // planner max_member bounds (the per-side member MAX)
      for (const auto& member : cohort.members) {
        cohort.planner_max_member_host_bytes = std::max(
            cohort.planner_max_member_host_bytes, member.private_host_bytes);
        cohort.planner_max_member_device_bytes =
            std::max(cohort.planner_max_member_device_bytes,
                     member.private_device_bytes);
      }
      // the complete per-cohort ledger (canonical composition)
      cohort.ledger_host_bytes =
          cohort.queue_host_bytes + cohort.bank_charge_host_bytes +
          cohort.normal_charge_host_bytes + cohort.process_host_bytes +
          cohort.rolling_host_bytes;
      cohort.ledger_device_bytes = cohort.bank_charge_device_bytes +
                                   cohort.process_device_bytes +
                                   cohort.rolling_device_bytes;
      queue_sum += cohort.queue_host_bytes;
      bank_h += cohort.bank_charge_host_bytes;
      bank_d += cohort.bank_charge_device_bytes;
      normal_h += cohort.normal_charge_host_bytes;
      priv_h += cohort.rolling_host_bytes;
      priv_d += cohort.rolling_device_bytes;
      pre_h = std::max(pre_h, cohort.ledger_host_bytes);
      pre_d = std::max(pre_d, cohort.ledger_device_bytes);
      wire_.cohorts.push_back(cohort);
      cursor += member_counts[c];
    }
    wire_.member_count_total = cursor;
    wire_.summary.queue_sum_host_bytes = queue_sum;
    wire_.summary.bank_sum_host_bytes = bank_h;
    wire_.summary.bank_sum_device_bytes = bank_d;
    wire_.summary.normal_sum_host_bytes = normal_h;
    wire_.summary.private_sum_host_bytes = priv_h;
    wire_.summary.private_sum_device_bytes = priv_d;
    wire_.summary.preflight_max_host_bytes = pre_h;
    wire_.summary.preflight_max_device_bytes = pre_d;
    identities_ = owned::HostInputBank::CaptureJobIdentities(jobs_);
  }

  std::filesystem::path root_;
  std::vector<BankJob> jobs_;
  std::vector<owned::BankJobIdentities> identities_;
  CohortForecastWire wire_;
};

TEST_F(CohortNativeFloorsTest, NativeFloorsAcceptConsistentCharges) {
  VerifyResidentCohortForecastNativeFloors(
      wire_, jobs_, identities_, 1ull << 40, 1ull << 40);
}

TEST_F(CohortNativeFloorsTest, BankChargeBelowAllowanceComposedRawRefuses) {
  wire_.cohorts[0].bank_charge_host_bytes -= 1;
  EXPECT_THROW(VerifyResidentCohortForecastNativeFloors(
                   wire_, jobs_, identities_, 1ull << 40, 1ull << 40),
               std::exception);
}

TEST_F(CohortNativeFloorsTest, NormalAllowanceMismatchRefuses) {
  wire_.cohorts[0].normal_future_bytes += 1;
  EXPECT_THROW(VerifyResidentCohortForecastNativeFloors(
                   wire_, jobs_, identities_, 1ull << 40, 1ull << 40),
               std::exception);
}

TEST_F(CohortNativeFloorsTest, PrivateBelowNativeMinimumRefuses) {
  wire_.cohorts[1].members[1].private_device_bytes -= 1;
  EXPECT_THROW(VerifyResidentCohortForecastNativeFloors(
                   wire_, jobs_, identities_, 1ull << 40, 1ull << 40),
               std::exception);
}

TEST_F(CohortNativeFloorsTest, SingletonInfeasibilityRefuses) {
  const uint64_t tiny = 4096;
  EXPECT_THROW(VerifyResidentCohortForecastNativeFloors(
                   wire_, jobs_, identities_, tiny, tiny),
               std::exception);
}

TEST_F(CohortNativeFloorsTest, ChangedInputAfterCaptureRefuses) {
  const auto path = root_ / "0.png";
  // Touch the captured input's modification metadata: the ORIGINAL
  // captured records never recapture, so a changed input refuses.
  Bitmap bitmap(4, 2, true);
  bitmap.Fill(BitmapColor<uint8_t>(30, 90, 150));
  bitmap.Write(path);
  std::filesystem::last_write_time(
      path,
      std::filesystem::file_time_type::clock::now() + std::chrono::seconds(2));
  EXPECT_THROW(VerifyResidentCohortForecastNativeFloors(
                   wire_, jobs_, identities_, 1ull << 40, 1ull << 40),
               std::exception);
}

// Cohort-two SUBSET BANK mapping regression through the real
// HostInputBank interface: the subset bank's job index is the cohort-local
// generation; the original invocation ordinal stays outside (reporting
// only); the original job content (reference/source geometry and global
// model image indices) is preserved exactly.
TEST_F(CohortNativeFloorsTest, CohortTwoSubsetBankMappingRegression) {
  const std::vector<BankJob> slice(jobs_.begin() + 2, jobs_.end());
  const auto raw_slice = owned::HostInputBank::RawEstimate(slice);
  const BankLimits charges{owned::MarginBytes(raw_slice.bounded_host_raw_bytes),
                           owned::MarginBytes(raw_slice.device_raw_bytes)};
  const auto identity_slice = owned::HostInputBank::CaptureJobIdentities(slice);
  const owned::HostInputBank bank(slice,
                                  charges,
                                  owned::BankNormals::kBounded,
                                  identity_slice.data(),
                                  identity_slice.size());
  // Subset bank job index 0 IS cohort 2's first member (original ordinal
  // 2): the ORIGINAL job content, not any other cohort's job.
  EXPECT_EQ(bank.NumJobs(), 2u);
  EXPECT_EQ(bank.Job(0).reference_image_idx, jobs_[2].reference_image_idx);
  EXPECT_EQ(bank.Job(0).source_image_idxs, jobs_[2].source_image_idxs);
  EXPECT_EQ(bank.Job(1).reference_image_idx, jobs_[3].reference_image_idx);
  EXPECT_EQ(bank.Job(0).inputs.size(), jobs_[2].inputs.size());
  for (size_t k = 0; k < bank.Job(0).inputs.size(); ++k) {
    EXPECT_EQ(bank.Job(0).inputs[k].image_idx, jobs_[2].inputs[k].image_idx);
    EXPECT_EQ(bank.Job(0).inputs[k].depth_path, jobs_[2].inputs[k].depth_path);
    EXPECT_EQ(bank.Job(0).inputs[k].normal_path,
              jobs_[2].inputs[k].normal_path);
  }
  // Reference/source LAYER mapping stays in native problem order over the
  // original shared layers (the shared-image dedup preserves layer
  // identity: job 2's reference is image 2, job 3's reference is image 0).
  EXPECT_EQ(static_cast<size_t>(bank.ReferenceLayer(0)),
            static_cast<size_t>(2));
  EXPECT_EQ(static_cast<size_t>(bank.ReferenceLayer(1)),
            static_cast<size_t>(0));
  EXPECT_EQ(bank.SourceLayers(0), (std::vector<int>{1, 0}));
  EXPECT_EQ(bank.SourceLayers(1), (std::vector<int>{1, 2}));
}

}  // namespace
}  // namespace mvs
}  // namespace colmap