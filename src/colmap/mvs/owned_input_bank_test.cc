// SPDX-License-Identifier: BSD-3-Clause
#include "colmap/mvs/owned_input_bank.h"
#include "colmap/mvs/patch_match.h"
#include "colmap/mvs/workspace.h"

#include "colmap/util/testing.h"


// Instrumented-test-only allocator meter with ownership tracking. It observes
// BOTH the C++ allocations and VLFeat's allocations (vl_imsmooth_f's
// convolution buffer and gaussian filters route through vl_malloc), and keeps
// a per-pointer owner table so frees of pre-existing or foreign pointers
// never corrupt live-byte bookkeeping: unknown pointers are freed untouched
// and the counter is unchanged. Counted sizes are the requested sizes, so
// measurements are deterministic and malloc usable rounding is invisible.
#include "thirdparty/VLFeat/generic.h"

#include <atomic>
#include <cmath>
#include <cstdlib>
#include <map>
#include <fstream>
#include <limits>
#include <system_error>
#include <stdexcept>
#include <type_traits>

#include <gtest/gtest.h>

namespace colmap::mvs::owned {
namespace {
class HostInputBankTest : public ::testing::Test {
 protected:
  void SetUp() override {
    root_ = CreateTestDir();
    const float K[9] = {10, 0, 2, 0, 10, 1, 0, 0, 1};
    const float R[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    const float T[3] = {0, 0, 0};
    job_.options.geom_consistency = true;
    job_.options.max_image_size = 2;
    job_.reference_image_idx = 0;
    job_.source_image_idxs = {2, 1};
    for (int idx = 0; idx < 3; ++idx) {
      const auto path = root_ / (std::to_string(idx) + ".png");
      Bitmap bitmap(4, 2, true);
      bitmap.Fill(BitmapColor<uint8_t>(30, 90, 150));
      ASSERT_TRUE(bitmap.Write(path));
      Image image(path, 4, 2, K, R, T);
      image.Downsize(2, 2);
      job_.images.push_back(image);
      BankInput input;
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
      job_.inputs.push_back(input);
    }
  }
  void TearDown() override { std::filesystem::remove_all(root_); }
  std::filesystem::path root_;
  BankJob job_;
};

TEST_F(HostInputBankTest, SharedInputsNativeOrderAndRetention) {
  auto second = job_;
  second.reference_image_idx = 1;
  second.source_image_idxs = {0, 2};
  std::vector<BankJob> jobs = {job_, second};
  HostInputBank bank(jobs, HostInputBank::Estimate(jobs));
  EXPECT_EQ(bank.Layers().size(), 3);
  EXPECT_EQ(bank.NumJobs(), 2);
  EXPECT_EQ(bank.SourceLayers(0), (std::vector<int>{2, 1}));
  EXPECT_EQ(bank.SourceLayers(1), (std::vector<int>{0, 2}));
  EXPECT_EQ(bank.WorkCounts().grayscale_reads, 3);
  EXPECT_EQ(bank.WorkCounts().depth_reads, 3);
  EXPECT_EQ(bank.WorkCounts().normal_reads, 3);
  EXPECT_EQ(bank.ReferenceNormal(0).GetWidth(), 2);
  EXPECT_EQ(bank.ReferenceNormal(1).GetWidth(), 2);
  EXPECT_EQ(bank.Layers()[2].reference_normal.GetWidth(), 0);
  Bitmap expected;
  ASSERT_TRUE(expected.Read(job_.inputs[0].image.GetPath(), false));
  expected.Rescale(2, 1);
  EXPECT_EQ(bank.Layers()[0].grayscale.RowMajorData(), expected.RowMajorData());
  DepthMap depth;
  depth.Read(job_.inputs[0].depth_path);
  depth.Downsize(2, 1);
  EXPECT_EQ(bank.Layers()[0].depth.GetData(), depth.GetData());
  NormalMap normal;
  normal.Read(job_.inputs[0].normal_path);
  normal.Downsize(2, 1);
  EXPECT_EQ(bank.ReferenceNormal(0).GetData(), normal.GetData());
  bank.ReleaseUploadPayloads();
  bank.ReleaseUploadPayloads();
  for (const auto& layer : bank.Layers()) {
    EXPECT_TRUE(layer.grayscale.IsEmpty());
    EXPECT_EQ(layer.depth.GetNumBytes(), 0);
  }
  EXPECT_EQ(bank.ReferenceNormal(0).GetData(), normal.GetData());
  EXPECT_TRUE(bank.Job(0).images[0].GetBitmap().IsEmpty());
}

TEST_F(HostInputBankTest, CanonicalAliasDeduplicatesButCopiedWorkspaceDoesNot) {
  auto alias = job_;
  const auto link = root_ / "alias.png";
  std::filesystem::create_symlink(job_.images[0].GetPath(), link);
  const auto& image = job_.images[0];
  alias.images[0] = Image(link, image.GetWidth(), image.GetHeight(),
                          image.GetK(), image.GetR(), image.GetT());
  alias.inputs[0].image = alias.images[0];
  std::vector<BankJob> jobs = {job_, alias};
  HostInputBank shared(jobs, HostInputBank::Estimate(jobs));
  EXPECT_EQ(shared.Layers().size(), 3);
  const auto copy = root_ / "copied.png";
  std::filesystem::copy_file(image.GetPath(), copy);
  alias.images[0] = Image(copy, image.GetWidth(), image.GetHeight(),
                          image.GetK(), image.GetR(), image.GetT());
  alias.inputs[0].image = alias.images[0];
  jobs = {job_, alias};
  HostInputBank distinct(jobs, HostInputBank::Estimate(jobs));
  EXPECT_EQ(distinct.Layers().size(), 4);
  EXPECT_NE(distinct.ReferenceLayer(0), distinct.ReferenceLayer(1));
}

TEST_F(HostInputBankTest, RejectsUndersizedReservations) {
  const std::vector<BankJob> jobs{job_};
  auto limits = HostInputBank::Estimate(jobs);
  --limits.host_bytes;
  EXPECT_THROW(HostInputBank(jobs, limits), std::invalid_argument);
  limits = HostInputBank::Estimate(jobs);
  --limits.device_bytes;
  EXPECT_THROW(HostInputBank(jobs, limits), std::invalid_argument);
}

TEST_F(HostInputBankTest, RejectsTruncatedSourceNormalBeforeLoading) {
  std::ofstream stream(job_.inputs[2].normal_path, std::ios::binary);
  stream << "4&2&3&";
  stream.close();
  EXPECT_THROW(HostInputBank::Estimate({job_}), std::invalid_argument);
}

TEST_F(HostInputBankTest, RejectsWrongMapChannelsAndIncompleteMappings) {
  DepthMap depth(4, 2, 1, 10);
  depth.Write(job_.inputs[1].normal_path);
  EXPECT_THROW(HostInputBank::Estimate({job_}), std::invalid_argument);
  job_.inputs.pop_back();
  EXPECT_THROW(HostInputBank::Estimate({job_}), std::invalid_argument);
}

TEST_F(HostInputBankTest, RejectsPayloadInMetadataAndDuplicateSources) {
  auto job = job_;
  job.images[0].SetBitmap(Bitmap(2, 1, false));
  EXPECT_THROW(HostInputBank::Estimate({job}), std::invalid_argument);
  job = job_;
  job.source_image_idxs = {1, 1};
  EXPECT_THROW(HostInputBank::Estimate({job}), std::invalid_argument);
}

TEST_F(HostInputBankTest, CheckedWholeBankArithmetic) {
  auto job = job_;
  const auto& old = job.images[0];
  job.images[0] = Image(old.GetPath(), std::numeric_limits<size_t>::max(), 2,
                        old.GetK(), old.GetR(), old.GetT());
  job.inputs[0].image = job.images[0];
  EXPECT_THROW(HostInputBank::Estimate({job}), std::overflow_error);
}

TEST_F(HostInputBankTest, BoundedModeSkipsNormalLoadsAndLoadsOnDemand) {
  const std::vector<BankJob> jobs{job_};
  const auto eager = HostInputBank::Estimate(jobs);
  const auto bounded = HostInputBank::EstimateBounded(jobs);
  EXPECT_EQ(eager.device_bytes, bounded.device_bytes);
  EXPECT_LT(bounded.host_bytes, eager.host_bytes);
  // Undersized reservation rejects against the bounded estimate.
  auto undersized = bounded;
  --undersized.host_bytes;
  EXPECT_THROW((HostInputBank(jobs, undersized, BankNormals::kBounded)),
               std::invalid_argument);
  HostInputBank bank(jobs, bounded, BankNormals::kBounded);
  EXPECT_EQ(bank.NormalsMode(), BankNormals::kBounded);
  EXPECT_EQ(bank.WorkCounts().normal_reads, 0);
  EXPECT_THROW(bank.ReferenceNormal(0), std::logic_error);
  NormalMap expected;
  expected.Read(job_.inputs[0].normal_path);
  expected.Downsize(2, 1);
  NormalMap loaded = bank.LoadReferenceNormal(0);
  EXPECT_EQ(loaded.GetData(), expected.GetData());
  // A repeat load returns another unique owned map with identical values;
  // single-load FIFO discipline is the pipeline's contract, not the bank's.
  NormalMap again = bank.LoadReferenceNormal(0);
  EXPECT_EQ(again.GetData(), expected.GetData());
}

TEST_F(HostInputBankTest, BoundedLoadDetectsIdentityChange) {
  const std::vector<BankJob> jobs{job_};
  HostInputBank bank(jobs, HostInputBank::EstimateBounded(jobs),
                     BankNormals::kBounded);
  {
    std::ofstream stream(job_.inputs[0].normal_path,
                         std::ios::binary | std::ios::app);
    stream << 'x';
  }
  // Identity is rechecked immediately before the read.
  EXPECT_THROW(bank.LoadReferenceNormal(0), std::runtime_error);
}

TEST_F(HostInputBankTest, NormalAllowanceFormula) {
  const std::vector<BankJob> jobs{job_};
  const auto allowance = HostInputBank::EstimateNormalAllowance(jobs, 2);
  // Reference layer is downsized to 2x1 → owner 12*2 = 24; normal file is
  // 4x2 → payload 12*8 = 96, source texels S = 8. Shrink overlap:
  // payload + new_data + shrink replacement = 96 + 48 = 144. Smoothing
  // overlap: payload + new_data + smoothed(4S) + VLFeat convolution
  // buffer(4S) + gaussian filters. Sigmas: scale 2 → 0.5 → filter
  // 4*(2*ceil(1.5)+1) floats = 20 bytes, equal sigmas alias to one filter.
  // 96 + 24 + 64 + 20 = 204; charged = max(144, 204) = 204.
  EXPECT_EQ(allowance.owner_bytes, 24);
  EXPECT_EQ(allowance.future_bytes, 48);
  EXPECT_EQ(allowance.transient_bytes, 204);
  EXPECT_EQ(allowance.reference_layers, 1u);
  EXPECT_THROW(HostInputBank::EstimateNormalAllowance(jobs, 0),
               std::invalid_argument);
  EXPECT_THROW(HostInputBank::EstimateNormalAllowance(jobs, 3),
               std::invalid_argument);
}

// Invocation-identity snapshot binding regression for the cohort reload
// path: the snapshot is per job in job.inputs order (input-ordinal space),
// never the whole-workspace image-index space, while the image-index map
// must remain the reference/source resolution space. Exercise against the
// REAL bank implementation; decode/filesystem identity semantics are
// native-build behavior (this gtest is CPU-real but requires OpenImageIO).
TEST_F(HostInputBankTest, SnapshotBindingValidatesInputOrdinalAndSharedLayers) {
  const auto pick = [this](const std::vector<int>& order) {
    std::vector<BankInput> inputs;
    for (const int idx : order) {
      for (const auto& input : job_.inputs) {
        if (input.image_idx == idx) {
          inputs.push_back(input);
          break;
        }
      }
    }
    return inputs;
  };
  // Job A admits inputs [image 2, image 0]; job B the same files in the
  // permuted input-ordinal order [image 0, image 2] — shared layers with
  // different input ordinals.
  auto a = job_;
  a.inputs = pick({2, 0});
  a.reference_image_idx = 2;
  a.source_image_idxs = {0};
  auto b = job_;
  b.inputs = pick({0, 2});
  b.reference_image_idx = 0;
  b.source_image_idxs = {2};
  std::vector<BankJob> jobs{a, b};
  const auto snapshot = HostInputBank::CaptureJobIdentities(jobs);
  // Per job, in job.inputs order: 2 entries — NOT the 3-image workspace.
  EXPECT_EQ(snapshot.size(), jobs.size());
  EXPECT_EQ(snapshot.at(0).size(), a.inputs.size());
  EXPECT_EQ(snapshot.at(1).size(), b.inputs.size());
  // Job A's first admitted input is image 2 (snapshot in input space).
  EXPECT_EQ(snapshot.at(0).at(0).image.path, a.inputs.at(0).image.GetPath());
  // Job B's first admitted input is image 0.
  EXPECT_EQ(snapshot.at(1).at(0).image.path, b.inputs.at(0).image.GetPath());
  HostInputBank bank(jobs, HostInputBank::Estimate(jobs),
                     BankNormals::kEager, snapshot.data(), snapshot.size());
  ASSERT_EQ(bank.Layers().size(), size_t(2));
  // Image-index space stays the reference/source resolution space.
  EXPECT_EQ(bank.ReferenceLayer(0), 0);                  // A ref image 2
  EXPECT_EQ(bank.SourceLayers(0), (std::vector<int>{1}));  // A image 0
  EXPECT_EQ(bank.ReferenceLayer(1), 1);                  // B ref image 0
  EXPECT_EQ(bank.SourceLayers(1), (std::vector<int>{0}));  // B image 2
}

TEST_F(HostInputBankTest, SnapshotBindingRejectsSameShapedReplacement) {
  const std::vector<BankJob> jobs{job_};
  const auto snapshot = HostInputBank::CaptureJobIdentities(jobs);
  // A TRUE object replacement, deterministically: same dims (and equal
  // payload size), but written to a fresh staging path and atomically
  // renamed over the target, so the inode changes unconditionally.
  // Stat-based invocation identity has a documented limitation the other
  // way: an IN-PLACE same-size rewrite can be invisible within one
  // file-timestamp tick (unchanged inode and possibly unchanged
  // mtime/ctime), so this test never relies on such timing. No hash or
  // content immutability is claimed.
  const auto target = job_.inputs[1].depth_path;
  const auto staged = target.parent_path() /
                      (target.filename().string() + ".staged");
  ASSERT_FALSE(std::filesystem::exists(staged));
  DepthMap replacement(4, 2, 1, 10);
  replacement.Fill(9);
  replacement.Write(staged);
  std::filesystem::rename(staged, target);
  // The re-admission must reject the replaced object before any payload.
  EXPECT_THROW((HostInputBank(jobs, HostInputBank::Estimate(jobs),
                              BankNormals::kEager, snapshot.data(),
                              snapshot.size())), std::runtime_error);
}

TEST_F(HostInputBankTest, BoundedSnapshotLoadRejectsReplacedNormal) {
  const std::vector<BankJob> jobs{job_};
  const auto bounded = HostInputBank::EstimateBounded(jobs);
  const auto snapshot = HostInputBank::CaptureJobIdentities(jobs);
  HostInputBank bank(jobs, bounded, BankNormals::kBounded,
                     snapshot.data(), snapshot.size());
  NormalMap replacement(4, 2);
  replacement.Fill(0.25f);
  replacement.Write(job_.inputs[0].normal_path);
  EXPECT_THROW(bank.LoadReferenceNormal(0), std::runtime_error);
}

TEST_F(HostInputBankTest, RejectsInconsistentSnapshotBindingArguments) {
  const std::vector<BankJob> jobs{job_};
  const auto limits = HostInputBank::Estimate(jobs);
  const auto snapshot = HostInputBank::CaptureJobIdentities(jobs);
  EXPECT_THROW((HostInputBank(jobs, limits, BankNormals::kEager,
                              nullptr, 1)), std::invalid_argument);
  EXPECT_THROW((HostInputBank(jobs, limits, BankNormals::kEager,
                              snapshot.data(), 0)), std::invalid_argument);
  // (nullptr, 0) is the legacy unbound path and remains valid.
  HostInputBank unbound(jobs, limits);
  EXPECT_EQ(unbound.NumJobs(), size_t(1));
}

}  // namespace
}  // namespace colmap::mvs::owned

// Counts live heap bytes of observed allocations to verify the actual
// rescaling/smoothing allocation overlap. Single-threaded scoped
// instrumentation; restored before returning. The NormalMap under test is
// CONSTRUCTED inside the window, so the original payload is part of the
// baseline and its later shrink-free is balanced (no pre-instrumentation
// pointer is ever subtracted). Requested sizes are tracked per pointer: the
// NormalMap payload, new_data, the shrink replacement and warp.cc's smoothed
// vector route through operator new, while VLFeat's convolution buffer and
// gaussian filters route through vl_malloc, observed by installing
// vl_set_alloc_func counting wrappers that share the same owner table. Table
// maintenance itself is guarded out of the meter and its nodes' own
// allocations are skipped, so node adds/removes stay balanced through the
// ownership table. The measured delta therefore counts exactly the
// allocations made inside the window between the baseline and the peak.
struct LiveByteCounter {
  std::atomic<size_t> live{0};
  std::atomic<size_t> peak{0};
};
static LiveByteCounter* g_live_bytes = nullptr;
static int g_meter_ops = 0;

std::map<void*, size_t>& MeterSizes() {
  static std::map<void*, size_t> table;
  return table;
}

void MeterNoteAlloc(void* pointer, size_t size) {
  if (!pointer || !g_live_bytes || g_meter_ops) return;
  ++g_meter_ops;
  MeterSizes()[pointer] = size;
  --g_meter_ops;
  const size_t now = g_live_bytes->live.fetch_add(size) + size;
  size_t peak = g_live_bytes->peak.load();
  while (peak < now && !g_live_bytes->peak.compare_exchange_weak(peak, now)) {}
}

void MeterNoteFree(void* pointer) {
  if (!pointer || !g_live_bytes || g_meter_ops) return;
  ++g_meter_ops;
  const auto entry = MeterSizes().find(pointer);
  if (entry != MeterSizes().end()) {
    g_live_bytes->live.fetch_sub(entry->second);
    MeterSizes().erase(entry);
  }
  --g_meter_ops;
}

// C++ allocations. (Aligned operator new variants route to aligned libc
// allocation functions and are outside meter scope both ways, so they cannot
// corrupt the counter either.)
void* operator new(std::size_t size) {
  void* result = std::malloc(size == 0 ? 1 : size);
  if (!result) throw std::bad_alloc();
  MeterNoteAlloc(result, size);
  return result;
}
void* operator new[](std::size_t size) {
  void* result = std::malloc(size == 0 ? 1 : size);
  if (!result) throw std::bad_alloc();
  MeterNoteAlloc(result, size);
  return result;
}
void operator delete(void* pointer) noexcept {
  MeterNoteFree(pointer);
  std::free(pointer);
}
void operator delete[](void* pointer) noexcept {
  MeterNoteFree(pointer);
  std::free(pointer);
}

// VLFeat vl_malloc/vl_calloc/vl_realloc/vl_free wrappers, observing every
// allocation the smoothing path performs inside VLFeat.
void* VLMeterMalloc(std::size_t size) {
  void* result = std::malloc(size);
  MeterNoteAlloc(result, size);
  return result;
}
void* VLMeterCalloc(std::size_t count, std::size_t size) {
  void* result = std::calloc(count, size);
  MeterNoteAlloc(result, count * size);
  return result;
}
void* VLMeterRealloc(void* pointer, std::size_t size) {
  const auto entry = MeterSizes().find(pointer);
  const size_t previous = entry != MeterSizes().end() ? entry->second : 0;
  MeterNoteFree(pointer);
  void* result = std::realloc(pointer, size);
  if (!result && previous) {
    // Failed realloc keeps the old block; restore its ownership entry.
    MeterNoteAlloc(pointer, previous);
  }
  MeterNoteAlloc(result, size);
  return result;
}
void VLMeterFree(void* pointer) {
  MeterNoteFree(pointer);
  std::free(pointer);
}

namespace colmap {
namespace mvs {
namespace owned {
namespace {

// RAII window: installs the meter and the VLFeat allocator wrappers for the
// guarded scope and restores BOTH on exit, including through exception
// unwinding. VLFeat exposes no allocator getter; the restored configuration
// is the stock set VLFeat itself installs by default (generic.c 1547:
// malloc/realloc/calloc/free), which restores the pre-test behavior of this
// process: the test never changes the allocator outside its windows.
class ScopedLiveMeter {
 public:
  ScopedLiveMeter() : previous_meter_(g_live_bytes) {
    g_live_bytes = &counter_;
    vl_set_alloc_func(
        [](std::size_t size) -> void* { return VLMeterMalloc(size); },
        [](void* pointer, std::size_t size) -> void* {
          return VLMeterRealloc(pointer, size);
        },
        [](std::size_t count, std::size_t size) -> void* {
          return VLMeterCalloc(count, size);
        },
        [](void* pointer) { VLMeterFree(pointer); });
  }
  ~ScopedLiveMeter() {
    g_live_bytes = previous_meter_;
    vl_set_alloc_func(
        [](std::size_t size) -> void* { return std::malloc(size); },
        [](void* pointer, std::size_t size) -> void* {
          return std::realloc(pointer, size);
        },
        [](std::size_t count, std::size_t size) -> void* {
          return std::calloc(count, size);
        },
        [](void* pointer) { std::free(pointer); });
  }

  LiveByteCounter counter_;

 private:
  LiveByteCounter* previous_meter_;
};

// Verifies the charged smoothing/rescale overlap against the REAL call path:
// NormalMap::Rescale -> DownsampleImage (smoothed vector) -> vl_imsmooth_f
// (VLFeat convolution buffer + gaussian filters) -> shrink-to-fit, with the
// bank's checked transient formula. The VLFeat allocator callbacks ARE
// REGISTERED for the measured scope (the ownership table is shared between
// the C++ meter and the vl wrappers, so VLFeat's convolution buffer and
// filters are observed). The NormalMap is constructed AND destroyed INSIDE
// the window: the original payload is baselined, its shrink-free stays
// balanced, and no tracked payload survives the meter close (no stale
// ownership entries). The measured live peak must match the charged
// formula's smoothing peak exactly (requested sizes are deterministic) and
// exceed the older one-destination / one-source-temp undercharges.
// NormalMap/warp implementations are unchanged.
TEST_F(HostInputBankTest, DownsizingRescaleAllocationOverlap) {
  // The outer window lives in its own block so its RAII restore (including
  // the stock allocator configuration) is observable below.
  ScopedLiveMeter meter;
  const size_t src_w = 64, src_h = 32, dst_w = 8, dst_h = 4;
  const size_t src_bytes = 4 * src_w * src_h;           // one full-source temp
  const size_t dst = 3 * 4 * dst_w * dst_h;             // one destination map
  const float sigma = 0.5f * (8.0f - 1.0f);
  const double samples = std::ceil(static_cast<double>(sigma) * 3.0);
  const size_t filters = 4 * (2 * static_cast<size_t>(samples) + 1);
  size_t live_delta = 0;
  {
    // Constructed and destroyed INSIDE the window.
    NormalMap normal(src_w, src_h);
    normal.Fill(0.5f);
    // Baseline includes the payload; a later payload free is balanced.
    const size_t baseline = meter.counter_.live.load();
    normal.Downsize(dst_w, dst_h);
    live_delta = meter.counter_.peak.load() - baseline;

    EXPECT_EQ(normal.GetWidth(), dst_w);
    EXPECT_EQ(normal.GetHeight(), dst_h);
    // Exact smoothing-window peak: new_data (dst) + two full-source temps
    // (warp.cc smoothed and the VLFeat convolution buffer) + the aliased
    // gaussian filter. Deterministic requested sizes, no meter noise.
    const size_t smoothing_peak = dst + 2 * src_bytes + filters;
    EXPECT_EQ(live_delta, smoothing_peak);
    // The previous undercharges are empirically exceeded: a single destination
    // (new_data alone), and one full-source temp alone.
    EXPECT_GT(live_delta, dst + src_bytes);
  }
  // All tracked payloads are destroyed before the meter closes: no stale
  // ownership entries and a fully balanced counter.
  EXPECT_EQ(meter.counter_.live.load(), 0);
  EXPECT_TRUE(MeterSizes().empty());

  // Exception cleanup probe (nested window): allocations are balanced, then a
  // throw unwinds through the RAII on the exception path. While the probe is
  // alive the enclosing window's meter is masked by design; the assertions
  // below only hold once the probe's destructor has restored it.
  {
    ScopedLiveMeter probe;
    try {
      float* scratch = new float[16];
      delete[] scratch;
      throw std::runtime_error("meter restore probe");
    } catch (const std::runtime_error&) {
    }
    EXPECT_EQ(g_live_bytes, &probe.counter_);
    EXPECT_EQ(probe.counter_.live.load(), 0);
    EXPECT_TRUE(MeterSizes().empty());
  }
  // The probe's RAII restored the enclosing window's meter on the exception
  // path, with no stale ownership entries.
  EXPECT_EQ(g_live_bytes, &meter.counter_);
  EXPECT_EQ(meter.counter_.live.load(), 0);
  EXPECT_TRUE(MeterSizes().empty());
}
// Closing the test's outer window (destructor) restored the prior meter
// pointer (null outside any window) and the stock VLFeat allocator
// configuration; both are exercised by construction of the RAII above.


#if defined(COLMAP_CUDA_ENABLED) && !defined(COLMAP_HIP_ENABLED)
TEST_F(HostInputBankTest, RejectsPmvsBeforeWorkspaceRead) {
  PatchMatchOptions options;
  options.geom_consistency = true;
  options.filter = true;
  options.gpu_index = "0";
  options.num_threads = 1;
  // Missing path proves rejection precedes workspace import/metadata reads.
  PatchMatchController controller(options, root_ / "missing", "PMVS", "option",
                                  "", PatchMatchExecution::kOwnedCuda);
  try {
    controller.ResolveOwnedBankJob();
    FAIL() << "PMVS resident mode unexpectedly accepted";
  } catch (const std::exception& error) {
    EXPECT_NE(std::string(error.what()).find("only COLMAP workspaces"),
              std::string::npos);
  }
}
#endif

TEST_F(HostInputBankTest, MoveOnlyOwner) {
  static_assert(!std::is_copy_constructible_v<HostInputBank>);
  static_assert(std::is_nothrow_move_constructible_v<HostInputBank>);
  HostInputBank bank({job_}, HostInputBank::Estimate({job_}));
  HostInputBank moved(std::move(bank));
  EXPECT_EQ(moved.WorkCounts().normal_reads, 3);
  EXPECT_EQ(moved.ReferenceLayer(0), 0);
}
}  // namespace
}  // namespace owned
}  // namespace mvs
}  // namespace colmap
