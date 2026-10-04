// SPDX-License-Identifier: BSD-3-Clause
#include "colmap/mvs/rotation_storage_plan.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef COLMAP_ROTATION_STORAGE_STANDALONE
#include <gtest/gtest.h>
#endif

namespace {
using namespace colmap::mvs::rotation_storage;
#define REQUIRE(x) do { if (!(x)) throw std::runtime_error( \
  std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": " #x); } while (false)

RotationBacking Owned(size_t bytes, int* releases = nullptr) {
  return {std::shared_ptr<void>(new uint8_t[bytes], [releases](void* p) {
            delete[] static_cast<uint8_t*>(p);
            if (releases) ++*releases;
          }), bytes};
}

void CheckLayoutsAndOwnership() {
  const auto p = MakePlan(3, 2, 61, 4);
  REQUIRE(p);
  REQUIRE(p->source_pitch == 12 && p->rotated_pitch == 8);
  REQUIRE(p->plane_bytes == 24 && p->backing_bytes == 1464);
  for (auto dims : {std::pair<size_t, size_t>{1, 9}, {9, 1}, {7, 7}, {13, 7}}) {
    for (size_t depth : {1, 61, 64}) {
      const auto a = MakePlan(dims.first, dims.second, depth, 4);
      const auto b = MakePlan(dims.second, dims.first, depth, 4);
      REQUIRE(a && b && a->plane_bytes == b->plane_bytes);
      REQUIRE(a->backing_bytes == b->backing_bytes);
    }
  }
  REQUIRE(!MakePlan(0, 2, 1, 4));
  REQUIRE(!MakePlan(2, 0, 1, 4));
  REQUIRE(!MakePlan(2, 2, 0, 4));
  REQUIRE(!MakePlan(2, 2, 1, 0));
  const size_t imax = std::numeric_limits<int>::max();
  REQUIRE(!MakePlan(imax + 1, 1, 1, 1));
  REQUIRE(!MakePlan(1, imax + 1, 1, 1));
  REQUIRE(!MakePlan(1, 1, imax + 1, 1));
  REQUIRE(!MakePlan(imax, 1, 1, 4));
  REQUIRE(!MakePlan(1, imax, 1, 4));
  REQUIRE(!MakePlan(imax / 4, imax / 4, imax, 4));
  REQUIRE(!MakePlan(2, 1, 1, std::numeric_limits<size_t>::max()));
  REQUIRE(!MakeView({}, 3, 2, 61, 4));

  int frees = 0;
  {
    auto backing = Owned(p->backing_bytes, &frees);
    auto source = MakeView(backing, 3, 2, 61, 4);
    REQUIRE(source);
    backing = {};
    auto dest = ReorientView(*source, 2, 3);
    REQUIRE(dest && dest->backing.allocation.get() == source->backing.allocation.get());
    REQUIRE(!ReorientView(*source, 1, 6));
    REQUIRE(frees == 0);
    auto workspace = MakeView(Owned(24), 2, 3, 1, 4);
    REQUIRE(workspace && ValidateReuse(*source, *dest, *workspace));
    auto bad = *dest;
    ++bad.layout.source_pitch;
    REQUIRE(!ValidateReuse(*source, bad, *workspace));
    REQUIRE(!ReorientView(bad, 3, 2));
    bad = *dest;
    --bad.backing.capacity_bytes;
    REQUIRE(!ValidateReuse(*source, bad, *workspace));
    REQUIRE(!MakeView(bad.backing, 2, 3, 61, 4));
    // An equal pointer with an unrelated control block is not shared ownership.
    bad = *dest;
    bad.backing.allocation = std::shared_ptr<void>(bad.backing.allocation.get(), [](void*) {});
    REQUIRE(!ValidateReuse(*source, bad, *workspace));
    auto wrong_shape = ReorientView(*workspace, 3, 2);
    REQUIRE(wrong_shape && !ValidateReuse(*source, *dest, *wrong_shape));
    auto overlap = *workspace;
    overlap.backing.allocation = std::shared_ptr<void>(
        static_cast<uint8_t*>(source->backing.allocation.get()) + 4, [](void*) {});
    REQUIRE(!ValidateReuse(*source, *dest, overlap));
    overlap.backing.allocation = std::shared_ptr<void>(
        source->backing.allocation, static_cast<uint8_t*>(source->backing.allocation.get()) + 4);
    REQUIRE(!ValidateReuse(*source, *dest, overlap));
    auto overflow = *workspace;
    overflow.backing.allocation = std::shared_ptr<void>(
        reinterpret_cast<void*>(std::numeric_limits<uintptr_t>::max() - 3), [](void*) {});
    REQUIRE(!ValidateReuse(*source, *dest, overflow));
    // Release all aliases, including intentionally invalid copies, then final owner.
    overlap = {}; bad = {}; source.reset();
    REQUIRE(frees == 0);
    dest.reset();
    REQUIRE(frees == 1);
  }
  REQUIRE(frees == 1);
}

std::vector<uint32_t> Oracle(const std::vector<uint32_t>& input,
                             size_t width, size_t height, size_t depth) {
  std::vector<uint32_t> result(input.size());
  for (size_t s = 0; s < depth; ++s)
    for (size_t y = 0; y < height; ++y)
      for (size_t x = 0; x < width; ++x)
        result[s * width * height + (width - 1 - x) * height + y] =
            input[s * width * height + y * width + x];
  return result;
}

void CheckBitsAndSchedule() {
  REQUIRE(Oracle({0, 1, 2, 3, 4, 5}, 3, 2, 1) ==
          std::vector<uint32_t>({2, 5, 1, 4, 0, 3}));
  const uint32_t special[] = {0, 0x80000000u, 0x7fc12345u, 0x7f800001u,
                              0x7f800000u, 0xff800000u, 0xffffffffu};
  for (auto dims : {std::pair<size_t, size_t>{3, 2}, {1, 9}, {9, 1}, {7, 7}, {13, 7}}) {
    for (size_t depth : {1, 61, 64}) {
      size_t width = dims.first, height = dims.second;
      const size_t count = width * height * depth;
      std::vector<uint32_t> original(count);
      for (size_t i = 0; i < count; ++i)
        original[i] = i < 7 ? special[i] : static_cast<uint32_t>(i * 2654435761u);
      std::vector<uint32_t> guarded(count + 2, 0xdeadbeefu);
      std::copy(original.begin(), original.end(), guarded.begin() + 1);
      std::vector<uint32_t> scratch(width * height + 2, 0xa5a5a5a5u);
      for (int turn = 0; turn < 4; ++turn) {
        const auto plan = MakePlan(width, height, depth, 4);
        REQUIRE(plan);
        const std::vector<uint32_t> before(guarded.begin() + 1, guarded.end() - 1);
        const auto expected = Oracle(before, width, height, depth);
        size_t next_layer = 0;
        bool want_rotate = true;
        ForEachLayer(*plan,
          [&](size_t layer, size_t offset) {
            REQUIRE(want_rotate && layer == next_layer && offset == layer * plan->plane_bytes);
            want_rotate = false;
            const auto* input = reinterpret_cast<const uint8_t*>(guarded.data() + 1) + offset;
            // CPU stand-in for the unchanged rotation kernel: byte copies only.
            for (size_t out_y = 0; out_y < width; ++out_y)
              for (size_t out_x = 0; out_x < height; ++out_x)
                std::memcpy(&scratch[1 + out_y * height + out_x],
                            input + (out_x * width + width - 1 - out_y) * 4, 4);
          },
          [&](size_t layer, size_t offset) {
            REQUIRE(!want_rotate && layer == next_layer);
            std::memcpy(reinterpret_cast<uint8_t*>(guarded.data() + 1) + offset,
                        scratch.data() + 1, plan->plane_bytes);
            want_rotate = true; ++next_layer;
          });
        REQUIRE(want_rotate && next_layer == depth);
        REQUIRE(std::equal(expected.begin(), expected.end(), guarded.begin() + 1));
        REQUIRE(guarded.front() == 0xdeadbeefu && guarded.back() == 0xdeadbeefu);
        REQUIRE(scratch.front() == 0xa5a5a5a5u && scratch.back() == 0xa5a5a5a5u);
        std::swap(width, height);
      }
      REQUIRE(std::equal(original.begin(), original.end(), guarded.begin() + 1));
    }
  }
  int calls = 0;
  try {
    ForEachLayer(*MakePlan(3, 2, 61, 4),
                 [&](size_t, size_t) { ++calls; },
                 [&](size_t, size_t) { ++calls; throw std::runtime_error("copy failure"); });
    REQUIRE(false);
  } catch (const std::runtime_error& e) {
    REQUIRE(std::string(e.what()) == "copy failure");
  }
  REQUIRE(calls == 2);
}
struct LifetimeLedger {
  struct Event { const char* name; size_t logical, known_physical; };
  size_t live = 0, known = 0, peak = 0, known_peak = 0;
  const char* peak_event = "empty";
  std::vector<Event> events;
  void Mark(const char* name) {
    events.push_back({name, live, known});
    if (live > peak) { peak = live; peak_event = name; }
    known_peak = std::max(known_peak, known);
  }
  std::shared_ptr<int> Allocate(const char* name, size_t logical,
                                std::optional<size_t> physical) {
    live += logical;
    known += physical.value_or(0); // arrays deliberately have UNKNOWN physical bytes
    Mark(name);
    return std::shared_ptr<int>(new int, [this, logical, physical](int* p) {
      delete p; live -= logical; known -= physical.value_or(0); Mark("release");
    });
  }
};

void CheckLifetimeModel(size_t sources, bool geometric, bool filter) {
  // Symbolic fixture, NOT a GPU estimate: 48-byte PRNG and 128-byte pitch alignment.
  // Actual CUDA pitches, sizeof(curandState), array/driver costs require tracing.
  size_t width = 3930, height = 2210;
  const size_t pixels = width * height, plane = pixels * 4;
  auto pitched = [](size_t w, size_t h, size_t layers, size_t element) {
    return ((w * element + 127) / 128) * 128 * h * layers;
  };
  LifetimeLedger ledger;
  auto alloc = [&](const char* label, size_t logical, std::optional<size_t> physical) {
    return ledger.Allocate(label, logical, physical);
  };
  {
    auto ref = alloc("ref_image", pixels, pitched(width, height, 1, 1));
    auto sum = alloc("ref_sum", plane, pitched(width, height, 1, 4));
    auto squared = alloc("ref_squared_sum", plane, pitched(width, height, 1, 4));
    { auto input = alloc("filter_input_texture", pixels, std::nullopt); }
    auto ref_texture = alloc("ref_texture", pixels, std::nullopt);
    auto images = alloc("src_images", pixels * sources, std::nullopt);
    auto depths = geometric ? alloc("src_depths", plane * sources, std::nullopt) : nullptr;
    std::vector<std::shared_ptr<int>> poses;
    for (int i = 0; i < 4; ++i) poses.push_back(alloc("pose", 43 * sources * 4, std::nullopt));
    auto random = alloc("random", pixels * 48, pitched(width, height, 1, 48));
    auto depth = alloc("depth", plane, pitched(width, height, 1, 4));
    auto normal = alloc("normal", plane * 3, pitched(width, height, 3, 4));
    auto beta_rows = [](size_t h) { return 32 + (h + 31) / 32; };
    auto beta_bytes = [&](size_t w, size_t h) { return w * beta_rows(h) * sources * 4; };
    auto sel = alloc("beta_scratch", beta_bytes(width, height),
                     pitched(width, beta_rows(height), sources, 4));
    auto prev = alloc("prev", plane * sources, plane * sources);
    auto cost = alloc("cost", plane * sources, plane * sources);
    const size_t max_dim = std::max(width, height);
    auto global = alloc("global", max_dim * sources * 2 * 4,
                        pitched(max_dim, sources, 2, 4));
    std::shared_ptr<int> mask; // zero placeholder, no bytes
    auto workspace = alloc("workspace", plane, plane);
    const size_t core = ledger.live - beta_bytes(width, height);
    const size_t remaining_single = std::max({pixels * 48, plane, plane * 3, pixels * 10});
    size_t expected_peak = ledger.peak;
    size_t control_peak = 0;
    REQUIRE(sel.get() != prev.get());
    for (int turn = 0; turn < 4; ++turn) {
      if (filter && turn == 3)
        mask = alloc("lazy_mask", pixels * sources, pitched(width, height, sources, 1));
      const size_t mask_live = (filter && turn == 3) ? pixels * sources : 0;
      expected_peak = std::max(expected_peak, core + beta_bytes(width, height) + mask_live + remaining_single);
      control_peak = std::max(control_peak, core - plane + beta_bytes(width, height) + mask_live + remaining_single);
      std::swap(width, height);
      { auto next = alloc("rotated_random", pixels * 48, pitched(width, height, 1, 48)); random.swap(next); }
      { auto next = alloc("rotated_depth", plane, pitched(width, height, 1, 4)); depth.swap(next); }
      { auto next = alloc("rotated_normal", plane * 3, pitched(width, height, 3, 4)); normal.swap(next); }
      {
        auto nr = alloc("rotated_ref", pixels, pitched(width, height, 1, 1));
        auto ns = alloc("rotated_sum", plane, pitched(width, height, 1, 4));
        auto nq = alloc("rotated_squared", plane, pitched(width, height, 1, 4));
        ref.swap(nr); sum.swap(ns); squared.swap(nq);
        { auto texture = alloc("rebound_texture", pixels, std::nullopt); ref_texture.swap(texture); }
      }
      ledger.Mark("before_scratch_release");
      sel.reset();
      ledger.Mark("after_scratch_release");
      REQUIRE(ledger.live == core + mask_live);
      {
        auto rotated_prev = prev;
        auto scratch_view = workspace;
        REQUIRE(rotated_prev.get() != scratch_view.get());
        prev.swap(rotated_prev);
        ledger.Mark("prev_alias_both_live");
        REQUIRE(ledger.live == core + mask_live);
      }
      // PR2 re-allocates scratch AFTER previous-probability rotation, before cost.
      sel = alloc("beta_scratch", beta_bytes(width, height),
                  pitched(width, beta_rows(height), sources, 4));
      const size_t before_alias = ledger.live;
      {
        auto rotated_cost = cost;
        auto scratch_view = workspace;
        REQUIRE(rotated_cost.get() != scratch_view.get());
        cost.swap(rotated_cost);
        ledger.Mark("cost_alias_both_live");
        REQUIRE(ledger.live == before_alias);
        REQUIRE(ledger.live == core + beta_bytes(width, height) + mask_live);
      }
      REQUIRE(ledger.live == before_alias);
      expected_peak = std::max(expected_peak, core + beta_bytes(width, height) + 2 * mask_live);
      // Original d70 cost rotation duplicates the S-plane map; it has no workspace.
      control_peak = std::max(control_peak, core - plane + beta_bytes(width, height) + mask_live + plane * sources);
      control_peak = std::max(control_peak, core - plane + beta_bytes(width, height) + 2 * mask_live);
      if (filter && turn == 3) {
        auto next = alloc("rotated_mask", pixels * sources, pitched(width, height, sources, 1));
        mask.swap(next);
      }
    }
    REQUIRE(ledger.peak == expected_peak);
    if (sources >= 61) REQUIRE(ledger.peak < control_peak);
    if (sources <= 10) REQUIRE(ledger.peak == control_peak + plane); // workspace can raise small-S peak
    if (filter && sources >= 61) REQUIRE(std::string(ledger.peak_event) == "rotated_mask");
    else REQUIRE(std::string(ledger.peak_event) == "rotated_random");
    REQUIRE(ledger.known_peak > 0);
    if (std::getenv("COLMAP_ROTATION_MODEL_DUMP")) {
      std::printf("PR2_MODEL_ONLY S=%zu geom=%d filter=%d core_logical=%zu peak_logical=%zu control_peak_logical=%zu known_physical_peak=%zu peak_event=%s\n",
                  sources, geometric, filter, core, ledger.peak, control_peak, ledger.known_peak, ledger.peak_event);
    }
  }
  REQUIRE(ledger.live == 0 && ledger.known == 0);
  REQUIRE(ledger.events.back().logical == 0);
}

void CheckTwoVolumesOneWorkspace() {
  const size_t width = 5, height = 3, sources = 61;
  const auto plan = *MakePlan(width, height, sources, 4);
  auto previous = std::make_shared<std::vector<uint32_t>>(width * height * sources);
  auto cost = std::make_shared<std::vector<uint32_t>>(previous->size());
  auto scratch = std::make_shared<std::vector<uint32_t>>(width * height);
  for (size_t i = 0; i < previous->size(); ++i) {
    (*previous)[i] = 0x7fc00000u ^ static_cast<uint32_t>(i * 97);
    (*cost)[i] = 0x80000000u ^ static_cast<uint32_t>(i * 193);
  }
  const auto original_cost = *cost;
  const auto original_previous = *previous;
  RotationBacking workspace_backing{std::shared_ptr<void>(scratch, scratch->data()), plan.plane_bytes};
  const auto workspace_view = MakeView(workspace_backing, height, width, 1, 4);
  for (auto data : {previous, cost}) {
    RotationBacking backing{std::shared_ptr<void>(data, data->data()), plan.backing_bytes};
    const auto source = MakeView(backing, width, height, sources, 4);
    const auto destination = ReorientView(*source, height, width);
    REQUIRE(ValidateReuse(*source, *destination, *workspace_view));
    const auto expected = Oracle(*data, width, height, sources);
    ForEachLayer(plan,
      [&](size_t layer, size_t) {
        for (size_t y = 0; y < height; ++y)
          for (size_t x = 0; x < width; ++x)
            (*scratch)[(width - 1 - x) * height + y] =
                (*data)[layer * width * height + y * width + x];
      },
      [&](size_t, size_t offset) {
        std::memcpy(reinterpret_cast<uint8_t*>(data->data()) + offset,
                    scratch->data(), plan.plane_bytes);
      });
    REQUIRE(*data == expected);
    if (data == previous) REQUIRE(*cost == original_cost);
    else REQUIRE(*previous == Oracle(original_previous, width, height, sources));
  }
}

void CheckPr2Memory() {
  for (size_t sources : {1, 10, 61, 64})
    for (bool geom : {false, true}) for (bool filter : {false, true})
      CheckLifetimeModel(sources, geom, filter);
}

void RunTests() {
  CheckLayoutsAndOwnership(); CheckBitsAndSchedule();
  CheckTwoVolumesOneWorkspace(); CheckPr2Memory();
}
}  // namespace

#ifdef COLMAP_ROTATION_STORAGE_STANDALONE
int main() {
  try { RunTests(); std::puts("rotation storage tests: PASS"); }
  catch (const std::exception& e) { std::fprintf(stderr, "%s\n", e.what()); return 1; }
}
#else
TEST(RotationStoragePlan, LayoutOwnershipAndBits) { EXPECT_NO_THROW(RunTests()); }
#endif
