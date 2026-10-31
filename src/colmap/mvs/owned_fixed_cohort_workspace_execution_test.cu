// SPDX-License-Identifier: BSD-3-Clause
#include "colmap/mvs/generated/resident_allocation_fingerprint.h"
#include "colmap/mvs/owned_fixed_cohort_inventory.h"
#include "colmap/mvs/owned_fixed_cohort_workspace.h"
#include "colmap/mvs/patch_match_cuda_owned.h"
#include "colmap/mvs/workspace.h"
#include "colmap/util/testing.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>

#include <gtest/gtest.h>

namespace colmap::mvs::owned {
namespace {
struct WorkspaceFixture {
  struct Shape {
    std::string name;
    int width;
    int height;
  };
  std::string workspace, spec, fingerprint, revision, test_identities;
  int max_size = -1;
  int seconds = 0;
  uint64_t scratch = 0;
  ByteCharge capacity;
  std::array<std::array<FixedCohortBudget, 2>, 2> budgets;
  std::vector<Shape> shapes;
  std::vector<FixedCohortWorkspaceReference> references;
};
namespace synthetic_fixture {
#include "fixed_cohort_workspace_synthetic.h"
}
#ifdef COLMAP_FIXED_COHORT_EXTERNAL_FIXTURE
namespace external_fixture {
#include "fixed_cohort_workspace_external.h"
}
#endif

// Genuine rectified projections: Cx=i/8, f=64, Z=4 => disparity=2*i.
// The deterministic world-plane texture is sampled at x+2*i in camera i.
void WritePlane(const std::filesystem::path& path) {
  std::filesystem::create_directories(path / "sparse");
  std::filesystem::create_directories(path / "images");
  std::ofstream cameras(path / "sparse/cameras.txt");
  cameras << "1 PINHOLE 64 48 64 64 32 24\n";
  std::ofstream images(path / "sparse/images.txt");
  for (int i = 0; i < 4; ++i) {
    const auto name = "plane" + std::to_string(i) + ".png";
    images << i + 1 << " 1 0 0 0 " << -0.125 * i << " 0 0 1 " << name << "\n\n";
    Bitmap bitmap(64, 48, false);
    for (int y = 0; y < 48; ++y) {
      for (int x = 0; x < 64; ++x) {
        uint32_t texel = static_cast<uint32_t>(x + 2 * i + y * 101);
        texel = (texel ^ 61u) ^ (texel >> 16);
        texel *= 9u;
        texel ^= texel >> 4;
        texel *= 0x27d4eb2du;
        texel ^= texel >> 15;
        ASSERT_TRUE(bitmap.SetPixel(x, y, BitmapColor<uint8_t>(texel & 255)));
      }
    }
    ASSERT_TRUE(bitmap.Write(path / "images" / name));
  }
  std::ofstream points(path / "sparse/points3D.txt");
}

void ExpectMapBits(const FixedCohortResult::Maps& expected,
                   const FixedCohortResult::Maps& actual) {
  for (const auto& pair :
       {std::make_pair(&expected.depth.GetData(), &actual.depth.GetData()),
        std::make_pair(&expected.normal.GetData(), &actual.normal.GetData())}) {
    ASSERT_FALSE(pair.first->empty());
    ASSERT_EQ(pair.first->size(), pair.second->size());
    EXPECT_EQ(std::memcmp(pair.first->data(),
                          pair.second->data(),
                          pair.first->size() * sizeof(float)),
              0);
  }
}

void ExpectMeaningful(const FixedCohortResult::Maps& maps) {
  ASSERT_GT(maps.depth.GetWidth(), 0);
  ASSERT_GT(maps.depth.GetHeight(), 0);
  size_t valid = 0;
  for (size_t y = 0; y < maps.depth.GetHeight(); ++y) {
    for (size_t x = 0; x < maps.depth.GetWidth(); ++x) {
      const float depth = maps.depth.Get(y, x);
      // Native invalid pixels may carry sentinels; finite checks are for
      // scientifically valid positive-depth entries, not invalid encodings.
      if (!(depth > 0)) continue;
      ASSERT_TRUE(std::isfinite(depth));
      float normal[3];
      maps.normal.GetSlice(y, x, normal);
      float norm = 0;
      for (float n : normal) {
        ASSERT_TRUE(std::isfinite(n));
        norm += n * n;
      }
      EXPECT_GT(norm, 0);
      ++valid;
    }
  }
  EXPECT_GT(valid, 0)
      << "all-filtered/zero maps are not workspace execution proof";
}

// An independent native loader and private PHOTO->GEO chain, not a GEO oracle
// seeded from candidate PHOTO. The control never reads candidate result maps.
FixedCohortResult PrivateControl(const WorkspaceFixture& fixture,
                                 const PreparedFixedCohortWorkspace& prepared,
                                 NativeRunDeadline deadline) {
  NativeRunDeadlineGuard guard(deadline);
  Workspace::Options options;
  options.workspace_path = fixture.workspace;
  options.workspace_format = "COLMAP";
  options.image_as_rgb = false;
  options.max_image_size = fixture.max_size;
  CachedWorkspace workspace(options);
  auto images = workspace.GetModel().images;
  EXPECT_EQ(images.size(), prepared.Images().size());
  for (size_t i = 0; i < images.size(); ++i) {
    images[i].SetBitmap(workspace.GetBitmap(i).Clone());
    EXPECT_EQ(workspace.GetModel().GetImageName(i), prepared.ImageNames()[i]);
    EXPECT_EQ(images[i].GetBitmap().RowMajorData(),
              prepared.Images()[i].GetBitmap().RowMajorData());
    EXPECT_EQ(
        std::memcmp(
            images[i].GetK(), prepared.Images()[i].GetK(), 9 * sizeof(float)),
        0);
    EXPECT_EQ(
        std::memcmp(
            images[i].GetR(), prepared.Images()[i].GetR(), 9 * sizeof(float)),
        0);
    EXPECT_EQ(
        std::memcmp(
            images[i].GetT(), prepared.Images()[i].GetT(), 3 * sizeof(float)),
        0);
  }
  FixedCohortResult result;
  std::vector<DepthMap> depths(images.size());
  std::vector<NormalMap> normals(images.size());
  for (const auto& ref : fixture.references) {
    PatchMatch::Problem problem;
    problem.images = &images;
    problem.ref_image_idx = workspace.GetModel().GetImageIdx(ref.reference);
    for (const auto& name : ref.photo_sources)
      problem.src_image_idxs.push_back(workspace.GetModel().GetImageIdx(name));
    auto phase = ref.photo_options;
    phase.owned_cuda_graphs = false;
    PatchMatchCuda control(phase, problem);
    control.Run();
    result.photo.push_back({control.GetDepthMap(), control.GetNormalMap()});
    depths[problem.ref_image_idx] = result.photo.back().depth;
    normals[problem.ref_image_idx] = result.photo.back().normal;
  }
  for (const auto& ref : fixture.references) {
    PatchMatch::Problem problem;
    problem.images = &images;
    problem.ref_image_idx = workspace.GetModel().GetImageIdx(ref.reference);
    for (const auto& name : ref.geo_sources)
      problem.src_image_idxs.push_back(workspace.GetModel().GetImageIdx(name));
    problem.depth_maps = &depths;
    problem.normal_maps = &normals;
    auto phase = ref.geo_options;
    phase.owned_cuda_graphs = false;
    PatchMatchCuda control(phase, problem);
    control.Run();
    result.geo.push_back({control.GetDepthMap(), control.GetNormalMap()});
  }
  guard.Check();
  return result;
}

void ExerciseWorkspace(WorkspaceFixture fixture, bool external) {
  ASSERT_EQ(fixture.fingerprint, kResidentAllocationFingerprint);
  ::testing::Test::RecordProperty("literal_request", fixture.spec);
  ::testing::Test::RecordProperty("production_allocation_fingerprint",
                                  fixture.fingerprint);
  ::testing::Test::RecordProperty("producer_source_revision", fixture.revision);
  ::testing::Test::RecordProperty("test_source_identities",
                                  fixture.test_identities);
  // One original finite cap covers all policy/mode arms AND the reused control.
  const auto deadline =
      NativeRunDeadline::FromBudget(std::chrono::seconds(fixture.seconds));
  NativeRunDeadlineGuard guard(deadline);
  std::optional<FixedCohortResult> control;
  for (const bool graphs : {false, true}) {
    for (const size_t requested :
         {size_t{1}, size_t{2}, size_t{3}, size_t{4}, size_t{0}, size_t{5}}) {
      if (external && requested != 1 && requested != 2) continue;
      const bool automatic = requested == 0 || requested == 5;
      const size_t target = requested == 5 ? 3 : requested;
      // External minimum: stream active1 and graph active2. No campaign.
      if (external && graphs != (requested == 2)) continue;
      SCOPED_TRACE(graphs ? "graph" : "stream");
      SCOPED_TRACE(requested);
      auto refs = fixture.references;
      for (auto& ref : refs) {
        ref.photo_options.owned_cuda_graphs = graphs;
        ref.geo_options.owned_cuda_graphs = graphs;
      }
      std::optional<uint64_t> snapshot_bytes;
      std::vector<uint64_t> staging_bytes;
      for (size_t policy_index : {size_t{0}, size_t{1}}) {
        auto budget = fixture.budgets[policy_index][requested == 1 ? 0 : 1];
        const auto active = automatic ? SelectFixedCohortActiveOwners(
                                            budget, fixture.capacity, target)
                                            .count
                                      : requested;
        ASSERT_EQ(active, requested == 0 ? size_t{4} : target);
        budget.max_active_owners = active;
        SCOPED_TRACE(policy_index);
        guard.Check();
        auto prepared = PrepareFixedCohortWorkspace(
            fixture.workspace,
            fixture.max_size,
            refs,
            budget,
            fixture.capacity,
            fixture.scratch,
            deadline,
            active == 1 ? PairDispatch::kSerial : PairDispatch::kPair,
            32);
        ASSERT_EQ(prepared.Images().size(), fixture.shapes.size());
        for (const auto& shape : fixture.shapes) {
          const auto it = std::find(prepared.ImageNames().begin(),
                                    prepared.ImageNames().end(),
                                    shape.name);
          ASSERT_NE(it, prepared.ImageNames().end());
          const auto& image =
              prepared.Images()[it - prepared.ImageNames().begin()];
          ASSERT_EQ(image.GetWidth(), shape.width);
          ASSERT_EQ(image.GetHeight(), shape.height);
        }
        // Mutating the original caller budget cannot change the prepared
        // policy.
        const auto policy = budget.output_policy;
        auto& caller_budget = budget;
        caller_budget.output_policy = static_cast<FixedCohortOutputPolicy>(99);
        const auto actual = prepared.Run();
        caller_budget.output_policy = policy;
        EXPECT_EQ(prepared.OutputPolicy(), policy);
        EXPECT_EQ(actual.output_policy, policy);
        guard.Check();
        if (!control) control = PrivateControl(fixture, prepared, deadline);
        const bool retain = policy == FixedCohortOutputPolicy::kPhotoAndGeo;
        ASSERT_EQ(actual.photo.size(), retain ? 4 : 0);
        ASSERT_GT(actual.actual_snapshot_bytes, 0);
        if (snapshot_bytes) {
          EXPECT_EQ(actual.actual_snapshot_bytes, *snapshot_bytes);
          EXPECT_EQ(actual.actual_geo_staging_bytes, staging_bytes);
        } else {
          snapshot_bytes = actual.actual_snapshot_bytes;
          staging_bytes = actual.actual_geo_staging_bytes;
        }
        ASSERT_EQ(actual.geo.size(), 4);
        for (size_t i = 0; i < 4; ++i) {
          SCOPED_TRACE(i);
          ExpectMeaningful(control->photo[i]);
          ExpectMeaningful(control->geo[i]);
          if (retain) {
            ExpectMeaningful(actual.photo[i]);
            ExpectMapBits(control->photo[i], actual.photo[i]);
          }
          ExpectMeaningful(actual.geo[i]);
          ExpectMapBits(control->geo[i], actual.geo[i]);
        }
        // Same loaded invocation, now with an actually allocated shared union
        // and canonical raw inventory. Independent control generated its own
        // PHOTO outputs above, never candidate-seeded GEO. No timing claim.
        std::vector<FixedCohortImageShape> shapes;
        std::vector<FixedCohortJobShape> jobs;
        for (const auto& image : prepared.Images())
          shapes.push_back({image.GetWidth(), image.GetHeight()});
        for (const auto& ref : prepared.References())
          jobs.push_back(
              {ref.reference_image_idx, ref.photo_sources, ref.geo_sources});
        auto declarations = budget;
        declarations.photo_source_storage =
            FixedCohortPhotoSourceStorage::kShared;
        auto shared_budget =
            BuildFixedCohortLogicalInventory(shapes, jobs, declarations);
        if (automatic) {
          const auto selected = SelectFixedCohortActiveOwners(
              shared_budget, fixture.capacity, target);
          ASSERT_EQ(selected.count, active);
          shared_budget.max_active_owners = selected.count;
        }
        const auto shared = RunFixedCohortPhotoGeo(
            prepared.Images(),
            prepared.References(),
            shared_budget,
            fixture.capacity,
            deadline,
            active == 1 ? PairDispatch::kSerial : PairDispatch::kPair);
        EXPECT_EQ(shared.actual_photo_shared.host,
                  shared_budget.photo_shared.host);
        EXPECT_EQ(shared.actual_photo_shared.device,
                  shared_budget.photo_shared.device);
        const auto key =
            std::string(graphs ? "graph" : "stream") +
            (automatic ? "_auto_active" : "_active") + std::to_string(active) +
            (retain ? "_photo_and_geo_shared" : "_final_geo_only_shared");
        ::testing::Test::RecordProperty(key + "_selected_active_owners",
                                        std::to_string(active));
        ::testing::Test::RecordProperty(
            key + "_actual_photo_shared_host",
            std::to_string(shared.actual_photo_shared.host));
        ::testing::Test::RecordProperty(
            key + "_actual_photo_shared_device",
            std::to_string(shared.actual_photo_shared.device));
        ::testing::Test::RecordProperty(
            key + "_declared_photo_shared_host",
            std::to_string(shared_budget.photo_shared.host));
        ::testing::Test::RecordProperty(
            key + "_declared_photo_shared_device",
            std::to_string(shared_budget.photo_shared.device));
        ASSERT_EQ(shared.photo.size(), retain ? 4 : 0);
        ASSERT_EQ(shared.geo.size(), 4);
        EXPECT_EQ(shared.actual_snapshot_bytes, actual.actual_snapshot_bytes);
        EXPECT_EQ(shared.actual_geo_staging_bytes,
                  actual.actual_geo_staging_bytes);
        for (size_t i = 0; i < 4; ++i) {
          if (retain) {
            ExpectMeaningful(shared.photo[i]);
            ExpectMapBits(control->photo[i], shared.photo[i]);
          }
          ExpectMeaningful(shared.geo[i]);
          ExpectMapBits(control->geo[i], shared.geo[i]);
        }
        guard.Check();
      }
    }
  }
}

TEST(FixedCohortWorkspaceGpu, TexturedPlaneMatchesIndependentPrivatePhotoGeo) {
  if (!std::getenv("COLMAP_TEST_FIXED_COHORT_WORKSPACE_GPU"))
    GTEST_SKIP()
        << "explicit opt-in required; not real-dataset/performance proof";
  auto fixture = synthetic_fixture::MakeWorkspaceFixture();
  fixture.workspace = CreateTestDir().string();
  WritePlane(fixture.workspace);
  ASSERT_FALSE(::testing::Test::HasFatalFailure());
  ExerciseWorkspace(std::move(fixture), false);
}

TEST(FixedCohortWorkspaceGpu,
     ExternalWorkspaceMatchesIndependentPrivatePhotoGeo) {
  if (!std::getenv("COLMAP_TEST_FIXED_COHORT_WORKSPACE_GPU"))
    GTEST_SKIP() << "explicit opt-in required";
#ifdef COLMAP_FIXED_COHORT_EXTERNAL_FIXTURE
  auto fixture = external_fixture::MakeWorkspaceFixture();
  ASSERT_FALSE(fixture.workspace.empty());
  ExerciseWorkspace(std::move(fixture), true);
#else
  GTEST_SKIP() << "no explicit external literal request configured";
#endif
}
}  // namespace
}  // namespace colmap::mvs::owned
