// SPDX-License-Identifier: BSD-3-Clause
// Native minima tests for the resident forecast gates. These exercise the
// REAL bank estimator path (HostInputBank::Estimate / EstimateBounded /
// EstimateNormalAllowance on real image/map files produced by the real
// writers); they require the dense-stereo library and never run on a
// stubbed substitute. Registered with the normal CMake test suite.
#include "colmap/mvs/resident_forecast.h"

#include "colmap/mvs/optimized_dispatch.h"
#include "colmap/mvs/owned_resident_minima.h"
#include "colmap/mvs/resident_contract.h"
#include "colmap/util/logging.h"
#include "colmap/util/testing.h"

#include <array>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include <gtest/gtest.h>

namespace colmap {
namespace mvs {
namespace {

// Two geometry jobs over three 4x2 layers with max_image_size 2 (the same
// shape family as the accepted bank tests): real PNG/depth/normal files.
class ResidentForecastNativeTest : public ::testing::Test {
 protected:
  void SetUp() override {
    root_ = CreateTestDir();
    const float K[9] = {10, 0, 2, 0, 10, 1, 0, 0, 1};
    const float R[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    const float T[3] = {0, 0, 0};
    for (int problem = 0; problem < 2; ++problem) {
      owned::BankJob job;
      job.options.geom_consistency = true;
      job.options.max_image_size = 2;
      job.reference_image_idx = problem;
      // Explicit ordered valid source sets: every source distinct, the
      // reference excluded, and every admitted layer covered exactly once
      // (three layers {0,1,2}; the reference occupies one of them).
      job.source_image_idxs =
          problem == 0 ? std::vector<int>{2, 1} : std::vector<int>{2, 0};
      for (const int idx : {0, 1, 2}) {
        const auto path = root_ / (std::to_string(idx) + ".png");
        Bitmap bitmap(4, 2, true);
        bitmap.Fill(BitmapColor<uint8_t>(30, 90, 150));
        ASSERT_TRUE(bitmap.Write(path));
        Image image(path, 4, 2, K, R, T);
        image.Downsize(2, 2);
        job.images.push_back(image);
        owned::BankInput input;
        input.image_idx = idx;
        input.image = image;
        input.max_image_size = 2;
        input.depth_path =
            root_ / (std::to_string(idx) + ".depth.bin");
        input.normal_path =
            root_ / (std::to_string(idx) + ".normal.bin");
        DepthMap depth(4, 2, 1, 10);
        depth.Fill(2);
        depth.Write(input.depth_path);
        NormalMap normal(4, 2);
        normal.Fill(0.5f);
        normal.Write(input.normal_path);
        job.inputs.push_back(input);
      }
      // Ordered-fixture validity: the reference set is excluded from the
      // sources and the admitted inputs cover layers exactly once.
      std::array<bool, 3> seen{};
      for (const int idx : job.source_image_idxs) {
        ASSERT_NE(idx, job.reference_image_idx);
        ASSERT_FALSE(seen.at(idx)) << "duplicate source layer";
        seen.at(idx) = true;
      }
      seen.at(job.reference_image_idx) = true;
      for (const bool layer : seen) {
        ASSERT_TRUE(layer) << "admitted input layer missing";
      }
      jobs_.push_back(std::move(job));
    }
  }

  std::filesystem::path root_;
  std::vector<owned::BankJob> jobs_;
};

// The margin helper must reproduce the canonical Python margin_bytes
// exactly (independent known values, computed from the committed formula:
// ceil(1.03 * raw)).
TEST(ResidentForecastMargin, MatchesCanonicalPythonMargins) {
  EXPECT_EQ(owned::MarginBytes(0), 0u);
  EXPECT_EQ(owned::MarginBytes(1), 2u);
  EXPECT_EQ(owned::MarginBytes(999999), 1029999u);
  EXPECT_EQ(owned::MarginBytes(1000000), 1030000u);
  EXPECT_EQ(owned::MarginBytes(36590), 37688u);
}

// The bounded private minimum includes the active owner normal exactly
// once; the eager minimum does not.
TEST(ResidentPrivateMinima, BoundedAddsOwnerNormalOnce) {
  const auto eager = owned::ResidentPrivateLogicalMinimum(
      4, 2, /*sources=*/2, /*include_owner_normal=*/false);
  const auto bounded = owned::ResidentPrivateLogicalMinimum(
      4, 2, /*sources=*/2, /*include_owner_normal=*/true);
  EXPECT_EQ(bounded.host, eager.host + 12u * 4u * 2u);
  EXPECT_EQ(bounded.device, eager.device);
  EXPECT_EQ(eager.host, 24u * 4u * 2u);
  // A=8, perimeter=8, S=2: init=97*A+48*8+8*4*S=1224;
  // steady=92*A+48*8+8*4*S=1184. Include the larger phase,
  // 2*area, 692 bytes per source and the 32-byte calibration allowance.
  EXPECT_EQ(eager.device, 1224u + 2u * 8u + 692u * 2u + 32u);
}

// The minima match the native estimator terms exactly and the margin is
// applied once: the margined minimum equals MarginBytes of the RAW, both
// with and without a caller unclassified allowance (host-side only).
TEST_F(ResidentForecastNativeTest, MinimaMatchNativeEstimatorTerms) {
  const auto minima = ComputeResidentLedgerMinima(
      jobs_, ResidentForecastNormals::kBounded, /*max_future=*/2);
  const auto bounded_estimate = owned::HostInputBank::EstimateBounded(jobs_);
  EXPECT_EQ(minima.bank_host_bytes, bounded_estimate.host_bytes);
  EXPECT_EQ(minima.bank_device_bytes, bounded_estimate.device_bytes);
  // Margined = MarginBytes(raw): single margin, allowance-free minima.
  EXPECT_EQ(minima.bank_host_bytes,
            owned::MarginBytes(minima.bank_host_raw_bytes));
  EXPECT_EQ(minima.bank_device_bytes,
            owned::MarginBytes(minima.bank_device_raw_bytes));
  const auto allowance =
      owned::HostInputBank::EstimateNormalAllowance(jobs_, 2);
  EXPECT_EQ(minima.normal_owner_bytes, allowance.owner_bytes);
  EXPECT_EQ(minima.normal_future_bytes, allowance.future_bytes);
  EXPECT_EQ(minima.normal_transient_bytes, allowance.transient_bytes);
  EXPECT_GT(minima.bank_host_raw_bytes, 0u);
  EXPECT_GT(minima.bank_device_raw_bytes, 0u);
  EXPECT_GT(minima.bank_host_bytes, 0u);
  EXPECT_GT(minima.bank_device_bytes, 0u);
  // The raw decomposition must hold: bounded raw = eager raw - retained.
  const auto raws = owned::HostInputBank::RawEstimate(jobs_);
  EXPECT_EQ(minima.bank_host_raw_bytes, raws.bounded_host_raw_bytes);
  EXPECT_EQ(minima.bank_device_raw_bytes, raws.device_raw_bytes);
  EXPECT_LT(raws.bounded_host_raw_bytes, raws.host_raw_bytes);
  // Private minima cover every reference in native problem order.
  ASSERT_EQ(minima.private_minima.size(), jobs_.size());
  EXPECT_GT(minima.private_minima[0][0], 0u);
}

// Bounded-only execution minima: eager mode refuses at the minima surface.
TEST_F(ResidentForecastNativeTest, MinimaRefuseEagerMode) {
  EXPECT_THROW(ComputeResidentLedgerMinima(
                   jobs_, ResidentForecastNormals::kEager,
                   /*max_future=*/2),
               std::exception);
}

// A wire assembled around the native minima is accepted by the charge gate
// at equality and refused below it (the ledger's charges must cover the
// native minima; the actual estimator output is checked for equality on
// the allowance terms).
TEST_F(ResidentForecastNativeTest, ChargeGateAgainstNativeMinima) {
  const auto minima = ComputeResidentLedgerMinima(
      jobs_, ResidentForecastNormals::kBounded, /*max_future=*/2);
  ResidentForecastWire wire;
  wire.qualification = "metadata_only_cpu_estimate";
  wire.completeness = "complete";
  wire.concurrency_basis = "rolling_reserve_declared";
  wire.mode = "bounded";
  wire.max_future = 2;
  wire.image_channels = 3;
  wire.metadata_source = "explicit_unclassified";
  wire.metadata_raw_bytes = 0;
  wire.metadata_charge_bytes = 0;
  wire.unclassified_allowance_bytes = 0;
  wire.unclassified_provenance = "caller_declared_zero_unclassified";
  wire.bank_host_bytes = minima.bank_host_bytes;
  wire.bank_device_bytes = minima.bank_device_bytes;
  wire.normal_owner_bytes = minima.normal_owner_bytes;
  wire.normal_future_bytes = minima.normal_future_bytes;
  wire.normal_transient_bytes = minima.normal_transient_bytes;
  wire.normal_charge_bytes =
      minima.normal_future_bytes + minima.normal_transient_bytes;
  wire.process_mode = "once";
  wire.process_count = 1;
  // Canonical single-process floor: 2 GiB per side.
  wire.process_host_bytes = 2147483648u;
  wire.process_device_bytes = 2147483648u;
  uint64_t max_host = 0;
  uint64_t max_device = 0;
  for (size_t i = 0; i < minima.private_minima.size(); ++i) {
    ResidentForecastReference reference;
    reference.reference_image_idx = jobs_.at(i).reference_image_idx;
    reference.reference_name = "test_reference_" + std::to_string(i);
    reference.input_count = 3;
    reference.private_host_bytes = minima.private_minima.at(i)[0] + 1;
    reference.private_device_bytes = minima.private_minima.at(i)[1] + 1;
    max_host = std::max(max_host, reference.private_host_bytes);
    max_device = std::max(max_device, reference.private_device_bytes);
    wire.references.push_back(reference);
  }
  wire.planner_max_member_host_bytes = max_host;
  wire.planner_max_member_device_bytes = max_device;
  wire.rolling_active_jobs = 1;
  wire.rolling_host_bytes = max_host;
  wire.rolling_device_bytes = max_device;
  EXPECT_NO_THROW(VerifyResidentForecastCharges(wire, minima));
  EXPECT_NO_THROW(VerifyResidentForecastLedgerTerms(wire, 1));
  // The complete checked aggregate fits exactly at its own total.
  const uint64_t host_total = minima.bank_host_bytes + 2147483648u + 0u +
      wire.normal_charge_bytes + wire.rolling_host_bytes;
  const uint64_t device_total = minima.bank_device_bytes + 2147483648u +
                                wire.rolling_device_bytes;
  EXPECT_NO_THROW(VerifyResidentForecastFitsCaps(wire, host_total,
                                                 device_total));
  EXPECT_THROW(VerifyResidentForecastFitsCaps(wire, host_total - 1,
                                              device_total),
               std::exception);
  EXPECT_THROW(VerifyResidentForecastFitsCaps(wire, host_total,
                                              device_total - 1),
               std::exception);
  // A margin-1 device charge over-cap refusal is covered above; a
  // marginal aggregate case: cap one byte below the total.
  EXPECT_THROW(VerifyResidentForecastFitsCaps(wire, host_total,
                                              device_total - 1),
               std::exception);

  // One byte below any native minimum refuses.
  auto refused = wire;
  refused.bank_host_bytes -= 1;
  EXPECT_THROW(VerifyResidentForecastCharges(refused, minima),
               std::exception);
  refused = wire;
  refused.references[1].private_device_bytes -= 2;
  EXPECT_THROW(VerifyResidentForecastCharges(refused, minima),
               std::exception);
}

// Five distinct real reference files and native estimator-backed admission
// charges (no CUDA execution). Raw contract binding is covered separately by
// OptimizedN3N4N5BindDistinctTopNAndCaps in resident_forecast_test.cc.
TEST_F(ResidentForecastNativeTest, OptimizedN3N4N5AgainstNativeMinima) {
  const float K[9] = {10, 0, 2, 0, 10, 1, 0, 0, 1};
  const float R[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  const float T[3] = {0, 0, 0};
  std::vector<owned::BankInput> extra_inputs;
  for (const int idx : {3, 4}) {
    const auto path = root_ / (std::to_string(idx) + ".png");
    Bitmap bitmap(4, 2, true);
    bitmap.Fill(BitmapColor<uint8_t>(30, 90, 150));
    ASSERT_TRUE(bitmap.Write(path));
    Image image(path, 4, 2, K, R, T);
    image.Downsize(2, 2);
    owned::BankInput input;
    input.image_idx = idx;
    input.image = image;
    input.max_image_size = 2;
    input.depth_path = root_ / (std::to_string(idx) + ".depth.bin");
    input.normal_path = root_ / (std::to_string(idx) + ".normal.bin");
    DepthMap depth(4, 2, 1, 10);
    depth.Fill(2);
    depth.Write(input.depth_path);
    NormalMap normal(4, 2);
    normal.Fill(0.5f);
    normal.Write(input.normal_path);
    extra_inputs.push_back(input);
  }
  // Keep the existing two-job fixture untouched; extend locally with new
  // references and identical immutable input bank membership.
  auto jobs = jobs_;
  jobs.resize(5, jobs_[0]);
  for (size_t i = 0; i < jobs.size(); ++i) {
    auto& job = jobs[i];
    job.reference_image_idx = static_cast<int>(i);
    job.source_image_idxs.clear();
    for (int source = 0; source < 5; ++source) {
      if (source != job.reference_image_idx)
        job.source_image_idxs.push_back(source);
    }
    for (const auto& input : extra_inputs) {
      job.images.push_back(input.image);
      job.inputs.push_back(input);
    }
    ASSERT_EQ(job.source_image_idxs.size(), 4u);
    ASSERT_EQ(job.inputs.size(), 5u);
  }
  const auto minima =
      ComputeResidentLedgerMinima(jobs, ResidentForecastNormals::kBounded, 2);
  ASSERT_EQ(minima.private_minima.size(), 5u);
  ResidentForecastWire wire;
  wire.qualification = kResidentForecastExecutableQualification;
  wire.completeness = kResidentForecastComplete;
  wire.concurrency_basis = kResidentForecastRollingDeclared;
  wire.mode = "bounded";
  wire.max_future = 2;
  wire.image_channels = 3;
  // This digest is an explicit test identity, not compiled-build qualification.
  wire.has_build_allocation_fingerprint = true;
  wire.has_source_fingerprint = true;
  wire.build_allocation_fingerprint = std::string(64, 'a');
  wire.source_fingerprint = wire.build_allocation_fingerprint;
  wire.metadata_source = "explicit_unclassified";
  wire.unclassified_provenance = "native_test_zero_allowance";
  wire.bank_host_bytes = minima.bank_host_bytes;
  wire.bank_device_bytes = minima.bank_device_bytes;
  wire.normal_owner_bytes = minima.normal_owner_bytes;
  wire.normal_future_bytes = minima.normal_future_bytes;
  wire.normal_transient_bytes = minima.normal_transient_bytes;
  wire.normal_charge_bytes =
      minima.normal_future_bytes + minima.normal_transient_bytes;
  wire.process_mode = "once";
  wire.process_count = 1;
  wire.process_host_bytes = 2147483648ull;
  wire.process_device_bytes = 2147483648ull;
  for (size_t i = 0; i < jobs.size(); ++i) {
    ResidentForecastReference reference;
    reference.reference_image_idx = static_cast<int>(i);
    reference.reference_name = std::to_string(i) + ".png";
    reference.input_count = 5;
    reference.private_host_bytes = minima.private_minima[i][0];
    reference.private_device_bytes = minima.private_minima[i][1];
    wire.references.push_back(reference);
  }
  // All five shapes/source counts are equal; literal multiplicity is the
  // independent expected top-N reserve. Bank and process are counted once.
  wire.planner_max_member_host_bytes = wire.references[0].private_host_bytes;
  wire.planner_max_member_device_bytes =
      wire.references[0].private_device_bytes;
  for (const size_t concurrency : {3u, 4u, 5u}) {
    wire.rolling_active_jobs = concurrency;
    wire.rolling_host_bytes = concurrency * wire.planner_max_member_host_bytes;
    wire.rolling_device_bytes =
        concurrency * wire.planner_max_member_device_bytes;
    EXPECT_NO_THROW(ValidateOptimizedStereoConcurrency(
        true, std::to_string(concurrency), true, false));
    EXPECT_NO_THROW(VerifyResidentForecastParameters(
        wire, wire.build_allocation_fingerprint));
    EXPECT_NO_THROW(VerifyResidentForecastLedgerTerms(wire, concurrency));
    EXPECT_NO_THROW(VerifyResidentForecastCharges(wire, minima));
    const auto host_cap = wire.bank_host_bytes + wire.process_host_bytes +
                          wire.normal_charge_bytes + wire.rolling_host_bytes;
    const auto device_cap = wire.bank_device_bytes + wire.process_device_bytes +
                            wire.rolling_device_bytes;
    EXPECT_NO_THROW(VerifyResidentForecastFitsCaps(wire, host_cap, device_cap));
    EXPECT_THROW(VerifyResidentForecastFitsCaps(wire, host_cap - 1, device_cap),
                 std::exception);
    EXPECT_THROW(VerifyResidentForecastFitsCaps(wire, host_cap, device_cap - 1),
                 std::exception);
    auto bad = wire;
    bad.has_build_allocation_fingerprint = false;
    EXPECT_THROW(VerifyResidentForecastParameters(
                     bad, wire.build_allocation_fingerprint),
                 std::exception);
    bad = wire;
    bad.has_source_fingerprint = false;
    EXPECT_THROW(VerifyResidentForecastParameters(
                     bad, wire.build_allocation_fingerprint),
                 std::exception);
    bad = wire;
    --bad.references[2].private_device_bytes;
    EXPECT_THROW(VerifyResidentForecastCharges(bad, minima), std::exception);
  }
}

}  // namespace
}  // namespace mvs
}  // namespace colmap