// SPDX-License-Identifier: BSD-3-Clause
#include "colmap/mvs/owned_fixed_cohort_workspace.h"

#include "colmap/mvs/owned_fixed_cohort_validation.h"
#include "colmap/mvs/workspace.h"
#include "colmap/scene/synthetic.h"
#include "colmap/util/testing.h"

#include <fstream>
#include <limits>

#include <gtest/gtest.h>

namespace colmap::mvs::owned {
namespace {
class FixedCohortWorkspaceTest : public ::testing::Test {
 protected:
  void SetUp() override {
    path = CreateTestDir();
    std::filesystem::create_directories(path / "sparse");
    std::filesystem::create_directories(path / "images");
    SyntheticDatasetOptions options;
    options.num_rigs = 1;
    options.num_cameras_per_rig = 1;
    options.num_frames_per_rig = 3;
    options.camera_model_id = PinholeCameraModel::model_id;
    options.camera_params = {20, 20, 10, 5};
    options.camera_width = 20;
    options.camera_height = 10;
    Reconstruction reconstruction;
    SynthesizeDataset(options, &reconstruction);
    reconstruction.Write(path / "sparse");
    Model model;
    model.ReadFromCOLMAP(path);
    for (int i = 0; i < 3; ++i) {
      names.push_back(model.GetImageName(i));
      Bitmap bitmap(20, 10, true);
      for (int y = 0; y < 10; ++y)
        for (int x = 0; x < 20; ++x)
          ASSERT_TRUE(bitmap.SetPixel(
              x,
              y,
              BitmapColor<uint8_t>(20 + i * 40 + x * 2, y * 15, 200 - x * 4)));
      ASSERT_TRUE(bitmap.Write(path / "images" / names.back()));
    }
    for (int i : {2, 0, 1}) {
      FixedCohortWorkspaceReference reference;
      reference.reference = names[i];
      reference.photo_sources = {
          names[(i + 2) % 3], names[(i + 1) % 3], names[(i + 2) % 3]};
      reference.geo_sources = reference.photo_sources;
      reference.photo_options.depth_min = 1;
      reference.photo_options.depth_max = 10;
      reference.photo_options.sigma_spatial = 5;
      reference.photo_options.geom_consistency = false;
      reference.photo_options.gpu_index = "0";
      reference.photo_options.num_threads = 1;
      reference.geo_options = reference.photo_options;
      reference.geo_options.geom_consistency = true;
      references.push_back(reference);
    }
    budget.caller_inputs = {600, 0};
    budget.frozen_inputs = {600, 0};
    budget.host_outputs = {19200, 0};
    budget.snapshots = {0, 9600};
    budget.photo_private.assign(3, {10000000, 10000000});
    budget.geo_private = budget.photo_private;
    budget.geo_staging_device.assign(3, 2400);
    budget.unknown = {1000000, 1000000};
    budget.unknown_provenance =
        "test explicit unqualified codec/context allowance";
    budget.max_active_owners = 1;
  }
  PreparedFixedCohortWorkspace Prepare(int max_size = -1) {
    return PrepareFixedCohortWorkspace(
        path,
        max_size,
        references,
        budget,
        {1000000000, 1000000000},
        1000000,
        NativeRunDeadline::FromBudget(std::chrono::seconds(30)),
        PairDispatch::kSerial,
        32);
  }
  std::filesystem::path path;
  std::vector<std::string> names;
  std::vector<FixedCohortWorkspaceReference> references;
  FixedCohortBudget budget;
};

TEST_F(FixedCohortWorkspaceTest, NativePixelsCalibrationAndOrdinalIdentity) {
  auto prepared = Prepare();
  ASSERT_EQ(prepared.Images().size(), 3);
  EXPECT_EQ(prepared.References()[0].reference_image_idx, 2);
  EXPECT_EQ(prepared.References()[1].reference_image_idx, 0);
  EXPECT_EQ(prepared.References()[0].photo_sources,
            (std::vector<int>{1, 0, 1}));
  EXPECT_EQ(prepared.References()[0].geo_sources, (std::vector<int>{1, 0, 1}));
  EXPECT_EQ(prepared.References()[0].photo_options.depth_min, 1);
  Workspace::Options options;
  options.workspace_path = path;
  options.workspace_format = "COLMAP";
  options.image_as_rgb = false;
  CachedWorkspace native(options);
  for (int i = 0; i < 3; ++i) {
    EXPECT_EQ(prepared.Images()[i].GetBitmap().RowMajorData(),
              native.GetBitmap(i).RowMajorData());
    EXPECT_TRUE(prepared.Images()[i].GetBitmap().IsGrey());
    for (int k = 0; k < 9; ++k)
      EXPECT_EQ(prepared.Images()[i].GetK()[k],
                native.GetModel().images[i].GetK()[k]);
  }
}

TEST_F(FixedCohortWorkspaceTest, ResizingUsesNativeGrayscaleAndCalibration) {
  auto prepared = Prepare(10);
  Workspace::Options options;
  options.workspace_path = path;
  options.workspace_format = "COLMAP";
  options.image_as_rgb = false;
  options.max_image_size = 10;
  CachedWorkspace native(options);
  for (int i = 0; i < 3; ++i) {
    EXPECT_EQ(prepared.Images()[i].GetWidth(), 10);
    EXPECT_EQ(prepared.Images()[i].GetHeight(), 5);
    EXPECT_EQ(prepared.Images()[i].GetBitmap().RowMajorData(),
              native.GetBitmap(i).RowMajorData());
    EXPECT_FLOAT_EQ(prepared.Images()[i].GetK()[0], 10);
    EXPECT_FLOAT_EQ(prepared.Images()[i].GetK()[2], 5);
    EXPECT_FLOAT_EQ(prepared.Images()[i].GetK()[4], 10);
    EXPECT_FLOAT_EQ(prepared.Images()[i].GetK()[5], 2.5);
    for (int k = 0; k < 9; ++k)
      EXPECT_EQ(prepared.Images()[i].GetK()[k],
                native.GetModel().images[i].GetK()[k]);
  }
  EXPECT_EQ(prepared.References()[0].photo_options.max_image_size, -1);
}

TEST_F(FixedCohortWorkspaceTest, ShapeValidationDoesNotAuthorizeEmptyPixels) {
  Model model;
  model.ReadFromCOLMAP(path);
  auto prepared = Prepare();
  EXPECT_NO_THROW(
      ValidateFixedCohortShape(model.images, prepared.References(), budget));
  EXPECT_THROW(
      ValidateFixedCohortPixels(model.images, prepared.References(), budget),
      std::invalid_argument);
}

TEST_F(FixedCohortWorkspaceTest,
       RejectsOpenClosureAndInvalidResolvedOptionsBeforeIO) {
  // Missing bitmaps act as a real I/O witness: these metadata failures must be
  // reported BEFORE even opening their headers, not masked by missing files.
  for (const auto& name : names)
    std::filesystem::remove(path / "images" / name);
  auto saved = references;
  references.pop_back();
  try {
    Prepare();
    FAIL() << "open GEO closure accepted";
  } catch (const std::invalid_argument& error) {
    EXPECT_NE(std::string(error.what()).find("source producer"),
              std::string::npos);
  }
  references = saved;
  references[0].photo_options.max_image_size = 10;
  try {
    Prepare();
    FAIL() << "unresolved resizing accepted";
  } catch (const std::invalid_argument& error) {
    EXPECT_NE(std::string(error.what()).find("phase options"),
              std::string::npos);
  }
}

TEST_F(FixedCohortWorkspaceTest, MissingAndCorruptImagesRefuse) {
  const auto file = path / "images" / names[0];
  std::filesystem::remove(file);
  EXPECT_THROW(Prepare(), std::exception);
  {
    std::ofstream corrupt(file);
    corrupt << "not an image";
  }
  EXPECT_THROW(Prepare(), std::exception);
}

TEST_F(FixedCohortWorkspaceTest, HeaderMustMatchCameraEvenWhenResizing) {
  Bitmap bitmap(19, 10, true);
  bitmap.Fill(BitmapColor<uint8_t>(0, 0, 0));
  ASSERT_TRUE(bitmap.Write(path / "images" / names[0]));
  EXPECT_THROW(Prepare(10), std::invalid_argument);
}

TEST_F(FixedCohortWorkspaceTest, RefusesLogicalBoundsScratchCapAndReserve) {
  auto saved = budget;
  budget.caller_inputs.host = 599;
  EXPECT_THROW(Prepare(), std::invalid_argument);
  budget = saved;
  budget.geo_staging_device[0] = 1;
  EXPECT_THROW(Prepare(), std::invalid_argument);
  budget = saved;
  EXPECT_THROW(PrepareFixedCohortWorkspace(
                   path,
                   -1,
                   references,
                   budget,
                   {1000000000, 1000000000},
                   1,
                   NativeRunDeadline::FromBudget(std::chrono::seconds(30)),
                   PairDispatch::kSerial,
                   32),
               std::invalid_argument);
  EXPECT_THROW(PrepareFixedCohortWorkspace(
                   path,
                   -1,
                   references,
                   budget,
                   {1000000000, 1000000000},
                   1000000000,
                   NativeRunDeadline::FromBudget(std::chrono::seconds(30)),
                   PairDispatch::kSerial,
                   32),
               std::invalid_argument);
  budget.headroom_reserve = {1000000000, 0};
  EXPECT_THROW(Prepare(), std::invalid_argument);
}

TEST_F(FixedCohortWorkspaceTest, LoaderLifecycleHasExactlyOneAggregateMargin) {
  // Raw loader peak: original bank 600 + prepared bank 600 + scratch
  // 20,000,000 + unknown 1,000,000 = 21,001,200. One 3% = 21,631,236.
  EXPECT_NO_THROW(PrepareFixedCohortWorkspace(
      path,
      -1,
      references,
      budget,
      {21631236, 1000000000},
      20000000,
      NativeRunDeadline::FromBudget(std::chrono::seconds(30)),
      PairDispatch::kSerial,
      32));
  EXPECT_THROW(PrepareFixedCohortWorkspace(
                   path,
                   -1,
                   references,
                   budget,
                   {21631235, 1000000000},
                   20000000,
                   NativeRunDeadline::FromBudget(std::chrono::seconds(30)),
                   PairDispatch::kSerial,
                   32),
               std::invalid_argument);
}

TEST_F(FixedCohortWorkspaceTest, RealObservedHeadroomIsNotReplacedByCapacity) {
  budget.unknown.host =
      CheckedAdd(ObserveFixedCohortHostAvailable(), 1000000000);
  try {
    PrepareFixedCohortWorkspace(
        path,
        -1,
        references,
        budget,
        {std::numeric_limits<uint64_t>::max(), 1000000000},
        1000000,
        NativeRunDeadline::FromBudget(std::chrono::seconds(30)),
        PairDispatch::kSerial,
        32);
    FAIL() << "declared capacity replaced observed host headroom";
  } catch (const std::invalid_argument& error) {
    EXPECT_NE(std::string(error.what()).find("headroom"), std::string::npos);
  }
}

TEST_F(FixedCohortWorkspaceTest, InvalidDispatchDeadlineAndImageSizeRefuse) {
  EXPECT_THROW(Prepare(0), std::invalid_argument);
  EXPECT_THROW(PrepareFixedCohortWorkspace(
                   path,
                   -1,
                   references,
                   budget,
                   {1000000000, 1000000000},
                   1000000,
                   NativeRunDeadline::FromBudget(std::chrono::seconds(30)),
                   PairDispatch::kPair,
                   32),
               std::invalid_argument);
  const auto expired = NativeRunDeadline::FromBudget(
      std::chrono::milliseconds(1),
      NativeRunDeadline::Clock::now() - std::chrono::seconds(1));
  EXPECT_THROW(PrepareFixedCohortWorkspace(path,
                                           -1,
                                           references,
                                           budget,
                                           {1000000000, 1000000000},
                                           1000000,
                                           expired,
                                           PairDispatch::kSerial,
                                           32),
               std::runtime_error);
}

TEST_F(FixedCohortWorkspaceTest, DistortedWorkspaceRefuses) {
  Reconstruction reconstruction;
  reconstruction.Read(path / "sparse");
  const auto id = reconstruction.Cameras().begin()->first;
  auto& camera = reconstruction.Camera(id);
  camera.model_id = SimpleRadialCameraModel::model_id;
  camera.params = {20, 10, 5, 0.1};
  reconstruction.Write(path / "sparse");
  EXPECT_THROW(Prepare(), std::invalid_argument);
}

TEST_F(FixedCohortWorkspaceTest, ExistingNativeMapsRoundTripThroughWorkspace) {
  auto prepared = Prepare(10);
  // Representative native result buffers, NOT claimed GPU outputs. No writer
  // adapter is introduced: exercise the exact existing native consumption seam.
  std::filesystem::create_directories(path / "stereo" / "depth_maps");
  std::filesystem::create_directories(path / "stereo" / "normal_maps");
  FixedCohortResult::Maps maps{DepthMap(10, 5, 1, 10), NormalMap(10, 5)};
  maps.depth.Fill(2.5f);
  maps.normal.Fill(0.25f);
  for (const auto* phase : {"photometric", "geometric"}) {
    const auto filename = names[2] + "." + phase + ".bin";
    maps.depth.Write(path / "stereo" / "depth_maps" / filename);
    maps.normal.Write(path / "stereo" / "normal_maps" / filename);
    Workspace::Options options;
    options.workspace_path = path;
    options.workspace_format = "COLMAP";
    options.input_type = phase;
    options.max_image_size = 10;
    CachedWorkspace workspace(options);
    EXPECT_EQ(workspace.GetDepthMap(2).GetData(), maps.depth.GetData());
    EXPECT_EQ(workspace.GetNormalMap(2).GetData(), maps.normal.GetData());
  }
}

#if !defined(COLMAP_CUDA_ENABLED) || defined(COLMAP_HIP_ENABLED)
TEST_F(FixedCohortWorkspaceTest, CpuExecutionNeverPretendsSuccess) {
  auto prepared = Prepare();
  EXPECT_THROW(prepared.Run(), std::runtime_error);
}
#endif

}  // namespace
}  // namespace colmap::mvs::owned
