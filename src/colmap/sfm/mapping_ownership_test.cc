// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/scene/database_cache.h"
#include "colmap/scene/database_sqlite.h"
#include "colmap/scene/reconstruction.h"
#include "colmap/scene/synthetic.h"
#include "colmap/scene/track.h"
#include "colmap/sfm/observation_manager.h"

#include <cstdlib>
#include <memory>
#include <new>
#include <utility>
#include <vector>

#ifdef _WIN32
#include <malloc.h>
#endif

#include <gtest/gtest.h>

// This executable alone replaces allocation. The injector is disabled outside
// the precise operation under test, including all GTest assertions/reporting.
namespace {
thread_local int allocations_before_failure = -1;
void CheckAllocation() {
  if (allocations_before_failure == 0) throw std::bad_alloc();
  if (allocations_before_failure > 0) --allocations_before_failure;
}
struct FailAllocation {
  explicit FailAllocation(int index) { allocations_before_failure = index; }
  ~FailAllocation() { allocations_before_failure = -1; }
};
}  // namespace

void* operator new(std::size_t size) {
  CheckAllocation();
  if (void* ptr = std::malloc(size ? size : 1)) return ptr;
  throw std::bad_alloc();
}
void* operator new[](std::size_t size) { return ::operator new(size); }
void operator delete(void* ptr) noexcept { std::free(ptr); }
void operator delete[](void* ptr) noexcept { std::free(ptr); }
void operator delete(void* ptr, std::size_t) noexcept { std::free(ptr); }
void operator delete[](void* ptr, std::size_t) noexcept { std::free(ptr); }

void* operator new(std::size_t size, std::align_val_t alignment) {
  CheckAllocation();
  void* ptr = nullptr;
#ifdef _WIN32
  ptr = _aligned_malloc(size ? size : 1, static_cast<size_t>(alignment));
#else
  if (posix_memalign(&ptr, static_cast<size_t>(alignment), size ? size : 1)) {
    ptr = nullptr;
  }
#endif
  if (!ptr) throw std::bad_alloc();
  return ptr;
}
void* operator new[](std::size_t size, std::align_val_t alignment) {
  return ::operator new(size, alignment);
}
void operator delete(void* ptr, std::align_val_t) noexcept {
#ifdef _WIN32
  _aligned_free(ptr);
#else
  std::free(ptr);
#endif
}
void operator delete[](void* ptr, std::align_val_t alignment) noexcept {
  ::operator delete(ptr, alignment);
}
void operator delete(void* ptr, std::size_t, std::align_val_t alignment) noexcept {
  ::operator delete(ptr, alignment);
}
void operator delete[](void* ptr, std::size_t,
                       std::align_val_t alignment) noexcept {
  ::operator delete(ptr, alignment);
}

namespace colmap {
namespace {

// The stock ReconstructionEq matcher owns a copy of its RHS. Avoid introducing
// extra budgeted owners while checking refusal at a saturated ledger.
void ExpectSameModel(const Reconstruction& actual,
                     const Reconstruction& expected) {
  EXPECT_EQ(actual.Rigs(), expected.Rigs());
  EXPECT_EQ(actual.Cameras(), expected.Cameras());
  EXPECT_EQ(actual.Frames(), expected.Frames());
  EXPECT_EQ(actual.Images(), expected.Images());
  EXPECT_EQ(actual.Points3D(), expected.Points3D());
  EXPECT_EQ(actual.RegFrameIds(), expected.RegFrameIds());
  EXPECT_EQ(actual.NumRegImages(), expected.NumRegImages());
}

TEST(MappingTrackOwnership, CopiesMoveAndRetainedCapacity) {
  auto ledger = std::make_shared<MappingLedger>(4096, 0, 0);
  {
    Track source(ledger);
    source.Reserve(8);
    source.AddElement(1, 0);
    source.AddElement(1, 1);  // Distinct features in one image are valid.
    EXPECT_EQ(ledger->Used(), 8 * sizeof(TrackElement));
    {
      Track copy(source);
      EXPECT_EQ(copy, source);
      EXPECT_EQ(ledger->Used(), 10 * sizeof(TrackElement));
      Track moved(std::move(copy));
      EXPECT_EQ(moved, source);
      EXPECT_EQ(ledger->Used(), 10 * sizeof(TrackElement));
      moved.DeleteElement(0);
      EXPECT_EQ(ledger->Used(), 10 * sizeof(TrackElement));
      moved.Compress();
      EXPECT_EQ(ledger->Used(), 9 * sizeof(TrackElement));
      {
        FailAllocation fail(0);
        moved = moved;
        moved = std::move(moved);
      }
      EXPECT_EQ(moved.Length(), 1);
    }
    EXPECT_EQ(ledger->Used(), 8 * sizeof(TrackElement));
    source.DeleteElement(1, 0);
    source.DeleteElement(1, 1);
    EXPECT_EQ(ledger->Used(), 8 * sizeof(TrackElement));
    source.Compress();
    EXPECT_EQ(ledger->Used(), 0);
  }
  EXPECT_EQ(ledger->Used(), 0);
}

TEST(MappingTrackOwnership, UnbudgetedAssignmentKeepsLegacyCapacityReuse) {
  Track source;
  source.AddElement(1, 0);
  Track destination;
  destination.Reserve(8);
  const auto capacity = destination.Elements().capacity();
  destination = source;
  EXPECT_EQ(destination, source);
  EXPECT_EQ(destination.Elements().capacity(), capacity);
  auto ledger = std::make_shared<MappingLedger>(4096, 0, 0);
  Track owned(ledger);
  owned.AddElement(2, 0);
  EXPECT_GT(ledger->Used(), 0);
  owned = source;
  EXPECT_EQ(owned, source);
  EXPECT_EQ(ledger->Used(), 0);
}

TEST(MappingTrackOwnership, GrowthOverlapRefusesWithoutChangingTrack) {
  // Final 24 bytes fit, but old 16 + replacement 24 do not (one 3% margin).
  auto ledger = std::make_shared<MappingLedger>(33, 0, 0);
  Track track(ledger);
  track.Reserve(2);
  track.AddElement(3, 4);
  track.AddElement(3, 5);
  EXPECT_THROW(track.Reserve(3), std::runtime_error);
  EXPECT_EQ(track.Length(), 2);
  EXPECT_EQ(track.Element(0), TrackElement(3, 4));
  EXPECT_EQ(track.Element(1), TrackElement(3, 5));
  EXPECT_EQ(ledger->Used(), 2 * sizeof(TrackElement));
  EXPECT_THROW(track.AddElement(4, 0), std::runtime_error);
  EXPECT_EQ(track.Length(), 2);
  EXPECT_EQ(ledger->Used(), 2 * sizeof(TrackElement));
}

TEST(MappingTrackOwnership, CopyAssignmentRefusalKeepsBothOwners) {
  auto ledger = std::make_shared<MappingLedger>(33, 0, 0);
  Track source(ledger);
  source.Reserve(2);
  source.AddElement(1, 0);
  source.AddElement(1, 1);
  Track destination(ledger);
  destination.AddElement(2, 0);
  EXPECT_THROW(destination = source, std::runtime_error);
  EXPECT_EQ(destination.Length(), 1);
  EXPECT_EQ(destination.Element(0), TrackElement(2, 0));
  EXPECT_EQ(source.Length(), 2);
  EXPECT_EQ(ledger->Used(), 3 * sizeof(TrackElement));
  destination = std::move(source);
  EXPECT_EQ(destination.Length(), 2);
  EXPECT_EQ(ledger->Used(), 2 * sizeof(TrackElement));
}

TEST(MappingTrackOwnership, AllocationFailureDuringGrowthAndCopyRollsBack) {
  auto ledger = std::make_shared<MappingLedger>(4096, 0, 0);
  Track source(ledger);
  source.Reserve(2);
  source.AddElement(1, 0);
  source.AddElement(1, 1);
  Track destination(ledger);
  destination.AddElement(2, 0);
  const auto used = ledger->Used();
  int failures = 0;
  bool succeeded = false;
  for (int index = 0; index < 32 && !succeeded; ++index) {
    try {
      FailAllocation fail(index);
      destination = source;
      succeeded = true;
    } catch (const std::bad_alloc&) {
      ++failures;
    }
    EXPECT_EQ(source.Length(), 2);
    if (!succeeded) {
      EXPECT_EQ(destination.Length(), 1);
      EXPECT_EQ(destination.Element(0), TrackElement(2, 0));
      EXPECT_EQ(ledger->Used(), used);
    }
  }
  ASSERT_TRUE(succeeded);
  EXPECT_GE(failures, 2);  // Ticket and vector allocation.
  EXPECT_EQ(destination, source);
  failures = 0;
  succeeded = false;
  const auto before_growth = ledger->Used();
  for (int index = 0; index < 32 && !succeeded; ++index) {
    try {
      FailAllocation fail(index);
      source.Reserve(8);
      succeeded = true;
    } catch (const std::bad_alloc&) {
      ++failures;
    }
    EXPECT_EQ(source.Length(), 2);
    EXPECT_EQ(source.Element(0), TrackElement(1, 0));
    EXPECT_EQ(source.Element(1), TrackElement(1, 1));
    if (!succeeded) EXPECT_EQ(ledger->Used(), before_growth);
  }
  ASSERT_TRUE(succeeded);
  EXPECT_GE(failures, 2);
  EXPECT_EQ(ledger->Used(), 10 * sizeof(TrackElement));
  EXPECT_EQ(ledger->Peak(), 12 * sizeof(TrackElement));
}

TEST(MappingTrackOwnership, BulkAppendAdoptionAndCompressionRollback) {
  auto ledger = std::make_shared<MappingLedger>(4096, 0, 0);
  Track track(ledger);
  track.Reserve(8);
  track.AddElement(7, 0);
  track.AddElement(7, 1);
  track.DeleteElement(1);
  const auto before = ledger->Used();
  int failures = 0;
  bool succeeded = false;
  for (int index = 0; index < 32 && !succeeded; ++index) {
    try {
      FailAllocation fail(index);
      track.Compress();
      succeeded = true;
    } catch (const std::bad_alloc&) {
      ++failures;
    }
    EXPECT_EQ(track.Length(), 1);
    EXPECT_EQ(track.Element(0), TrackElement(7, 0));
    if (!succeeded) EXPECT_EQ(ledger->Used(), before);
  }
  ASSERT_TRUE(succeeded);
  EXPECT_GE(failures, 2);
  EXPECT_EQ(ledger->Used(), sizeof(TrackElement));
  track.AddElement(track.Element(0));  // Preserve aliased value across reserve.
  track.AddElements(track.Elements());
  EXPECT_EQ(track.Length(), 4);
  for (const auto& element : track.Elements()) {
    EXPECT_EQ(element, TrackElement(7, 0));
  }
  std::vector<TrackElement> incoming = {{3, 1}, {3, 2}};
  const auto before_replace = ledger->Used();
  failures = 0;
  succeeded = false;
  for (int index = 0; index < 32 && !succeeded; ++index) {
    try {
      FailAllocation fail(index);
      track.SetElements(incoming);
      succeeded = true;
    } catch (const std::bad_alloc&) {
      ++failures;
    }
    if (!succeeded) {
      EXPECT_EQ(track.Length(), 4);
      EXPECT_EQ(track.Element(0), TrackElement(7, 0));
      EXPECT_EQ(ledger->Used(), before_replace);
    }
  }
  ASSERT_TRUE(succeeded);
  EXPECT_GE(failures, 3);  // Caller vector copy, owner ticket and replacement.
  EXPECT_EQ(track.Elements(), incoming);
  EXPECT_EQ(ledger->Used(), 2 * sizeof(TrackElement));
}

TEST(MappingTrackOwnership, CompressionRefusalKeepsRetainedStorage) {
  auto ledger = std::make_shared<MappingLedger>(70, 0, 0);
  Track track(ledger);
  track.Reserve(8);
  track.AddElement(1, 0);
  EXPECT_THROW(track.Compress(), std::runtime_error);
  EXPECT_EQ(track.Length(), 1);
  EXPECT_EQ(track.Element(0), TrackElement(1, 0));
  EXPECT_EQ(ledger->Used(), 8 * sizeof(TrackElement));
  track.DeleteElement(0);
  EXPECT_EQ(ledger->Used(), 8 * sizeof(TrackElement));
  track.Compress();
  EXPECT_EQ(ledger->Used(), 0);
}

// Small real models exercise covered Point2D copies and internal pointer rebinding.
struct OwnedModel {
  std::shared_ptr<MappingLedger> ledger =
      std::make_shared<MappingLedger>(1 << 20, 0, 0);
  std::shared_ptr<Database> database = Database::Open(kInMemorySqliteDatabasePath);
  std::shared_ptr<DatabaseCache> cache;
  std::unique_ptr<Reconstruction> model = std::make_unique<Reconstruction>();

  OwnedModel() {
    SyntheticDatasetOptions synthetic;
    synthetic.num_rigs = 1;
    synthetic.num_cameras_per_rig = 1;
    synthetic.num_frames_per_rig = 2;
    synthetic.num_points3D = 3;
    SynthesizeDataset(synthetic, model.get(), database.get());
    DatabaseCache::Options options;
    options.mapping_budget = ledger;
    cache = DatabaseCache::Create(*database, options);
    model->Load(*cache);
    const auto id = model->Points3D().begin()->first;
    Track owned(ledger);
    owned.AddElements(model->Point3D(id).track.Elements());
    model->Point3D(id).track = std::move(owned);
  }
};

void ExpectBoundPointers(const Reconstruction& model) {
  for (const auto& [id, frame] : model.Frames()) {
    EXPECT_EQ(frame.RigPtr(), &model.Rig(frame.RigId()));
  }
  for (const auto& [id, image] : model.Images()) {
    EXPECT_EQ(image.CameraPtr(), &model.Camera(image.CameraId()));
    EXPECT_EQ(image.FramePtr(), &model.Frame(image.FrameId()));
  }
}

TEST(MappingModelOwnership, CloneRetainsIndependentCreditAndPointers) {
  OwnedModel fixture;
  auto& source = *fixture.model;
  const auto before = fixture.ledger->Used();
  auto clone = std::make_unique<Reconstruction>(source);
  EXPECT_EQ(clone->MappingBudget(), fixture.ledger);
  EXPECT_GT(fixture.ledger->Used(), before);
  const auto clone_credit = fixture.ledger->Used() - before;
  ExpectSameModel(*clone, source);
  ExpectBoundPointers(*clone);
  {
    FailAllocation fail(0);
    *clone = *clone;
  }
  EXPECT_EQ(fixture.ledger->Used(), before + clone_credit);
  fixture.model.reset();
  fixture.cache.reset();
  fixture.database.reset();
  EXPECT_EQ(fixture.ledger->Used(), clone_credit);
  ExpectBoundPointers(*clone);
  clone->TearDown();
  EXPECT_EQ(fixture.ledger->Used(), clone_credit);
  clone.reset();
  EXPECT_EQ(fixture.ledger->Used(), 0);
}

TEST(MappingModelOwnership, AssignmentReplacesIndependentRetainedOwners) {
  OwnedModel source;
  OwnedModel destination;
  const auto old_used = destination.ledger->Used();
  const auto source_used = source.ledger->Used();
  {
    Reconstruction retained(*destination.model);
    const auto retained_credit = destination.ledger->Used() - old_used;
    *destination.model = *source.model;
    EXPECT_EQ(destination.model->MappingBudget(), source.ledger);
    ExpectSameModel(*destination.model, *source.model);
    ExpectBoundPointers(*destination.model);
    ExpectBoundPointers(retained);
    EXPECT_GT(source.ledger->Used(), source_used);
    destination.cache.reset();
    destination.database.reset();
    EXPECT_EQ(destination.ledger->Used(), retained_credit);
  }
  EXPECT_EQ(destination.ledger->Used(), 0);
  destination.model.reset();
  EXPECT_EQ(source.ledger->Used(), source_used);
}

TEST(MappingModelOwnership, CloneRefusalLeavesSourceAndDestinationIntact) {
  OwnedModel fixture;
  Reconstruction destination;
  SyntheticDatasetOptions synthetic;
  synthetic.num_points3D = 2;
  SynthesizeDataset(synthetic, &destination);
  const Reconstruction original(destination);
  const auto before = fixture.ledger->Used();
  // Leave less than a clone's required payload, without allocating a huge model.
  auto full = fixture.ledger->Acquire(((uint64_t{1} << 20) * 100 / 103) - before);
  const auto occupied = fixture.ledger->Used();
  EXPECT_THROW(Reconstruction copy(*fixture.model), std::runtime_error);
  EXPECT_THROW(destination = *fixture.model, std::runtime_error);
  EXPECT_EQ(fixture.ledger->Used(), occupied);
  ExpectSameModel(destination, original);
  ExpectBoundPointers(destination);
  ExpectBoundPointers(*fixture.model);
  full.reset();
  destination = *fixture.model;
  EXPECT_EQ(destination.MappingBudget(), fixture.ledger);
  ExpectSameModel(destination, *fixture.model);
  ExpectBoundPointers(destination);
}

TEST(MappingModelOwnership, AllocationFailuresRollBackCloneAndAssignment) {
  OwnedModel fixture;
  Reconstruction destination;
  SyntheticDatasetOptions synthetic;
  synthetic.num_points3D = 2;
  SynthesizeDataset(synthetic, &destination);
  const Reconstruction original(destination);
  const auto before = fixture.ledger->Used();
  int failures = 0;
  bool succeeded = false;
  for (int index = 0; index < 512 && !succeeded; ++index) {
    try {
      FailAllocation fail(index);
      Reconstruction clone(*fixture.model);
      succeeded = true;
    } catch (const std::bad_alloc&) {
      ++failures;
    }
    EXPECT_EQ(fixture.ledger->Used(), before);
    ExpectBoundPointers(*fixture.model);
  }
  ASSERT_TRUE(succeeded);
  EXPECT_GT(failures, 2);
  failures = 0;
  succeeded = false;
  for (int index = 0; index < 512 && !succeeded; ++index) {
    try {
      FailAllocation fail(index);
      destination = *fixture.model;
      succeeded = true;
    } catch (const std::bad_alloc&) {
      ++failures;
    }
    if (!succeeded) {
      ExpectSameModel(destination, original);
      EXPECT_EQ(fixture.ledger->Used(), before);
    }
    ExpectBoundPointers(destination);
    ExpectBoundPointers(*fixture.model);
  }
  ASSERT_TRUE(succeeded);
  EXPECT_GT(failures, 2);
  ExpectSameModel(destination, *fixture.model);
}

TEST(MappingObservationOwnership, RefusalBeforeImageLinkKeepsStatistics) {
  OwnedModel fixture;
  ObservationManager observations(*fixture.model,
                                  fixture.cache->CorrespondenceGraph());
  const auto point_id = fixture.model->Points3D().begin()->first;
  const auto image_id = fixture.model->Point3D(point_id).track.Element(0).image_id;
  const auto index = fixture.model->Image(image_id).NumPoints2D() - 1;
  ASSERT_FALSE(fixture.model->Image(image_id).Point2D(index).HasPoint3D());
  const Reconstruction original(*fixture.model);
  const auto visible = observations.NumVisiblePoints3D(image_id);
  const auto score = observations.Point3DVisibilityScore(image_id);
  const auto count = fixture.model->Image(image_id).NumPoints3D();
  auto full = fixture.ledger->Acquire(
      ((uint64_t{1} << 20) * 100 / 103) - fixture.ledger->Used());
  const auto occupied = fixture.ledger->Used();
  EXPECT_THROW(observations.AddObservation(point_id, TrackElement(image_id, index)),
               std::runtime_error);
  ExpectSameModel(*fixture.model, original);
  EXPECT_EQ(fixture.model->Image(image_id).NumPoints3D(), count);
  EXPECT_EQ(observations.NumVisiblePoints3D(image_id), visible);
  EXPECT_EQ(observations.Point3DVisibilityScore(image_id), score);
  EXPECT_EQ(fixture.ledger->Used(), occupied);
  full.reset();
  observations.AddObservation(point_id, TrackElement(image_id, index));
  EXPECT_EQ(fixture.model->Image(image_id).Point2D(index).point3D_id, point_id);
  EXPECT_EQ(fixture.model->Image(image_id).NumPoints3D(), count + 1);
  EXPECT_EQ(fixture.model->Point3D(point_id).track.Length(), 3);
}

TEST(MappingObservationOwnership, AppendAllocationFailuresLeaveModelUnchanged) {
  OwnedModel fixture;
  ObservationManager observations(*fixture.model,
                                  fixture.cache->CorrespondenceGraph());
  const auto point_id = fixture.model->Points3D().begin()->first;
  const auto image_id = fixture.model->Point3D(point_id).track.Element(0).image_id;
  const auto index = fixture.model->Image(image_id).NumPoints2D() - 1;
  ASSERT_FALSE(fixture.model->Image(image_id).Point2D(index).HasPoint3D());
  const Reconstruction original(*fixture.model);
  const auto before = fixture.ledger->Used();
  const auto visible = observations.NumVisiblePoints3D(image_id);
  const auto score = observations.Point3DVisibilityScore(image_id);
  const auto pairs = observations.ImagePairs();
  const auto count = fixture.model->Image(image_id).NumPoints3D();
  int failures = 0;
  bool succeeded = false;
  for (int allocation = 0; allocation < 32 && !succeeded; ++allocation) {
    try {
      FailAllocation fail(allocation);
      observations.AddObservation(point_id, TrackElement(image_id, index));
      succeeded = true;
    } catch (const std::bad_alloc&) {
      ++failures;
    }
    if (!succeeded) {
      ExpectSameModel(*fixture.model, original);
      EXPECT_EQ(fixture.ledger->Used(), before);
      EXPECT_EQ(fixture.model->Image(image_id).NumPoints3D(), count);
      EXPECT_EQ(observations.NumVisiblePoints3D(image_id), visible);
      EXPECT_EQ(observations.Point3DVisibilityScore(image_id), score);
      ASSERT_EQ(observations.ImagePairs().size(), pairs.size());
      for (const auto& [pair_id, stats] : pairs) {
        EXPECT_EQ(observations.ImagePairs().at(pair_id).num_tri_corrs,
                  stats.num_tri_corrs);
      }
    }
  }
  ASSERT_TRUE(succeeded);
  EXPECT_GE(failures, 2);
  EXPECT_EQ(fixture.model->Image(image_id).Point2D(index).point3D_id, point_id);
  EXPECT_EQ(fixture.model->Point3D(point_id).track.Element(2),
            TrackElement(image_id, index));
}

}  // namespace
}  // namespace colmap
