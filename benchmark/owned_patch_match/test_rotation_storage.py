#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""Compile the actual CUDA-independent storage helper against a fake backend.

No CUDA compilation, GPU execution, or product lifetime guarantee is implied.
"""
import pathlib
import subprocess
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[2]
HEADER = ROOT / "src/colmap/mvs/owned_rotation_storage.h"

CPP = r'''
#include "src/colmap/mvs/owned_rotation_storage.h"
#include <array>
#include <cassert>
#include <climits>
#include <limits>
#include <string>
#include <type_traits>
using namespace colmap::mvs::owned;
using Role = RotationRole;

// Use runtime assertions (compilation intentionally does not define NDEBUG).
template <class F> void Reject(F f) {
  bool rejected = false;
  try { f(); } catch (const std::exception&) { rejected = true; }
  assert(rejected);
}
struct Backend {
  int attempts = 0, allocations = 0, frees = 0, drains = 0;
  int fail_at = -1;
  bool pending = false;
  void Drain() { ++drains; pending = false; }
};
struct Buffer {
  Backend& backend;
  const RotationSlotLayout layout;
  const RotationAllocationRecord record;
  std::vector<unsigned char> bytes;
  Buffer(Backend& b, const RotationSlotLayout& s, size_t pitch)
      : backend(b), layout(s), record(MakeRotationAllocationRecord(s, pitch)),
        bytes(record.pitched_span_bytes, 0xcd) { ++backend.allocations; }
  ~Buffer() { assert(!backend.pending); ++backend.frees; }
  unsigned char& At(size_t x, size_t y, size_t z) {
    assert(x < layout.width && y < layout.height && z < layout.depth);
    return bytes[(z * layout.height + y) * record.pitch_bytes +
                 x * layout.element_bytes];
  }
};
using Pair = OwnedRotationPair<Buffer>;
static_assert(!std::is_copy_constructible<Pair>::value);
static_assert(!std::is_move_constructible<Pair>::value);
static_assert(!std::is_copy_assignable<Pair>::value);
static_assert(!std::is_move_assignable<Pair>::value);

// Drain belongs to the enclosing owner, including its constructor catch.
// The helper deliberately does not know about streams or allocation backends.
struct Owner {
  Backend& backend;
  OwnedRotationLayout layout;
  std::array<Pair, 10> banks;
  std::unique_ptr<Buffer> workspace;
  explicit Owner(Backend& b, size_t w = 3, size_t h = 5, size_t s = 4)
      : backend(b), layout(BuildOwnedRotationLayout(w, h, s)) {
    try {
      for (const auto& slot : layout.slots) {
        int index = backend.attempts++;
        if (index == backend.fail_at) throw std::runtime_error("allocation");
        // Differing, nontrivial returned pitches; no shape/pitch inference.
        auto buffer = std::make_unique<Buffer>(
            backend, slot, slot.width * slot.element_bytes + 7 + index);
        backend.pending = true;
        if (slot.role == Role::kWorkspace) workspace = std::move(buffer);
        else banks[static_cast<size_t>(slot.role)].Set(slot.parity,
                                                     std::move(buffer));
      }
      backend.Drain();
    } catch (...) {
      backend.Drain();
      throw;
    }
  }
  ~Owner() { backend.Drain(); }
  Pair& Bank(Role r) { return banks[static_cast<size_t>(r)]; }
};
void Rotate(Buffer& src, Buffer& dst) {
  assert(&src != &dst);
  assert(src.layout.width == dst.layout.height);
  assert(src.layout.height == dst.layout.width);
  assert(src.layout.depth == dst.layout.depth);
  src.backend.pending = true;
  for (size_t z = 0; z < src.layout.depth; ++z)
    for (size_t y = 0; y < src.layout.height; ++y)
      for (size_t x = 0; x < src.layout.width; ++x)
        dst.At(src.layout.height - 1 - y, x, z) = src.At(x, y, z);
  src.backend.Drain();
}
void LayoutChecks() {
  auto l = BuildOwnedRotationLayout(3, 5, 4);
  assert(l.width == 3 && l.height == 5 && l.sources == 4);
  assert(l.slots.size() == 21);
  assert(l.logical_bytes == 15 * (146 + 26 * 4) + 8 * 5 * 4);
  const std::array<size_t, 11> depths{1,1,3,1,1,1,4,4,4,4,2};
  const std::array<size_t, 11> elements{48,4,4,1,4,4,4,4,4,1,4};
  const std::array<const char*, 11> names{
    "prng", "depth", "normal", "ref_image", "ref_sum", "ref_squared_sum",
    "selection_current", "selection_previous", "cost", "mask", "workspace"};
  size_t index = 0, sum = 0;
  for (unsigned role = 0; role < 11; ++role) {
    auto r = static_cast<Role>(role);
    assert(std::string(RotationRoleName(r)) == names[role]);
    for (unsigned p = 0; p < (role == 10 ? 1u : 2u); ++p) {
      const auto& s = l.At(r, p);
      assert(&s == &l.slots[index++]);
      assert(s.role == r && s.parity == p);
      assert(s.width == (role == 10 ? 5u : (p ? 5u : 3u)));
      assert(s.height == (role == 10 ? 4u : (p ? 3u : 5u)));
      assert(s.depth == depths[role] && s.element_bytes == elements[role]);
      assert(s.requested_span_bytes == s.width*s.height*s.depth*s.element_bytes);
      auto a = MakeRotationAllocationRecord(s, s.width*s.element_bytes + 13);
      assert(a.role == s.role && a.parity == p && a.width == s.width);
      assert(a.height == s.height && a.depth == s.depth);
      assert(a.element_bytes == s.element_bytes);
      assert(a.pitch_bytes == s.width*s.element_bytes + 13);
      assert(a.requested_span_bytes == s.requested_span_bytes);
      assert(a.pitched_span_bytes == a.pitch_bytes*s.height*s.depth);
      assert(std::string(a.lifetime_phase) == "prepared_through_outputs");
      sum += s.requested_span_bytes;
      Reject([&] { MakeRotationAllocationRecord(s, s.width*s.element_bytes - 1); });
    }
  }
  assert(sum == l.logical_bytes);
  auto large = BuildOwnedRotationLayout(6204, 4132, 8);
  assert(large.logical_bytes == 9075161568ULL);
  Reject([&] { l.At(Role::kWorkspace, 1); });
  Reject([&] { l.At(Role::kDepth, 2); });
  Reject([&] { l.At(static_cast<Role>(99), 0); });
  Reject([] { RotationRoleName(static_cast<Role>(-1)); });
  Reject([] { BuildOwnedRotationLayout(0, 1, 1); });
  Reject([] { BuildOwnedRotationLayout(1, 0, 1); });
  Reject([] { BuildOwnedRotationLayout(1, 1, 0); });
  for (int axis = 0; axis < 3; ++axis) {
    size_t dims[3]{1,1,1}; dims[axis] = size_t(INT_MAX) + 1;
    Reject([&] { BuildOwnedRotationLayout(dims[0], dims[1], dims[2]); });
  }
  Reject([] { BuildOwnedRotationLayout(INT_MAX, INT_MAX, INT_MAX); });
  // Individual spans fit size_t here, but their sum overflows on 64-bit hosts.
  if (sizeof(size_t) == 8)
    Reject([] { BuildOwnedRotationLayout(400000000, 400000000, 1); });
  auto slot = l.At(Role::kNormal, 0);
  Reject([&] { MakeRotationAllocationRecord(slot, std::numeric_limits<size_t>::max()); });
  // The first multiplication fits, but multilayer pitched span overflows.
  Reject([&] { MakeRotationAllocationRecord(
      slot, std::numeric_limits<size_t>::max() / slot.height); });
  for (int invalid = 0; invalid < 13; ++invalid) {
    auto bad = slot;
    switch (invalid) {
      case 0: bad.width = 0; break;
      case 1: bad.height = 0; break;
      case 2: bad.depth = 0; break;
      case 3: bad.element_bytes = 0; break;
      case 4: bad.parity = 2; break;
      case 5: bad.role = static_cast<Role>(100); break;
      case 6: bad.width = size_t(INT_MAX) + 1; break;
      case 7: ++bad.requested_span_bytes; break;
      case 8: bad.role = Role::kWorkspace; bad.parity = 1; break;
      case 9: bad.height = size_t(INT_MAX) + 1; break;
      case 10: bad.depth = size_t(INT_MAX) + 1; break;
      case 11: bad.depth = 2; break;
      case 12: bad.element_bytes = 8; break;
    }
    Reject([&] { MakeRotationAllocationRecord(bad, 4096); });
  }
}
void PairChecks() {
  Backend b;
  {
    Pair p;
    assert(p.CurrentParity() == 0);
    Reject([&] { p.Current(); }); Reject([&] { p.Alternate(); });
    Reject([&] { p.operator->(); }); Reject([&] { p.At(2); });
    Reject([&] { p.Flip(); });
    Reject([&] { p.Set(0, nullptr); });
    Reject([&] { p.Set(2, nullptr); });
    auto l = BuildOwnedRotationLayout(2, 2, 1);
    p.Set(0, std::make_unique<Buffer>(b, l.At(Role::kDepth, 0), 16));
    Reject([&] { p.Flip(); });
    Reject([&] { p.Set(0, std::make_unique<Buffer>(b, l.At(Role::kDepth, 0), 16)); });
    p.Set(1, std::make_unique<Buffer>(b, l.At(Role::kDepth, 1), 20));
    assert(p.At(0) != p.At(1)); // Square images still have separate owners.
    auto* original = p.Current(); auto* alternate = p.Alternate();
    int allocs = b.allocations, frees = b.frees;
    for (int i = 0; i < 20; ++i) {
      p.Flip();
      assert(p.CurrentParity() == unsigned((i + 1) % 2));
      assert(p.Current() == ((i % 2) ? original : alternate));
      assert(p.Alternate() == ((i % 2) ? alternate : original));
      assert(p.operator->() == p.Current());
    }
    assert(b.allocations == allocs && b.frees == frees);
  }
  assert(b.allocations == b.frees);
}
void LifetimeChecks() {
  for (int fail = 0; fail < 21; ++fail) {
    Backend b; b.fail_at = fail;
    Reject([&] { Owner o(b); });
    assert(b.attempts == fail + 1 && b.allocations == fail);
    assert(b.frees == fail && b.drains == 1 && !b.pending);
  }
  Backend b;
  {
    Owner o(b);
    assert(b.allocations == 21 && b.frees == 0);
    std::array<Buffer*, 4> selection{
      o.Bank(Role::kSelectionCurrent).At(0), o.Bank(Role::kSelectionCurrent).At(1),
      o.Bank(Role::kSelectionPrevious).At(0), o.Bank(Role::kSelectionPrevious).At(1)};
    for (int i = 0; i < 4; ++i)
      for (int j = i + 1; j < 4; ++j) assert(selection[i] != selection[j]);
    auto& normal = o.Bank(Role::kNormal);
    for (size_t z = 0; z < 3; ++z)
      for (size_t y = 0; y < 5; ++y)
        for (size_t x = 0; x < 3; ++x)
          normal.Current()->At(x,y,z) = static_cast<unsigned char>(z*50+y*3+x);
    auto initial_normal = normal.Current()->bytes;
    // Previous parity zero is the only initialized selection slot.
    for (auto& byte : selection[2]->bytes) byte = 50;
    for (int rotation = 0; rotation < 20; ++rotation) {
      unsigned p = unsigned(rotation % 2);
      auto& current = o.Bank(Role::kSelectionCurrent);
      auto& previous = o.Bank(Role::kSelectionPrevious);
      assert(current.CurrentParity() == p && previous.CurrentParity() == p);
      auto* c = current.Current(); auto* prev = previous.Current();
      assert(c != prev && c != previous.Alternate());
      // Backward overwrites every valid multilayer pixel before forward reads.
      for (size_t z = 0; z < c->layout.depth; ++z)
        for (size_t y = 0; y < c->layout.height; ++y)
          for (size_t x = 0; x < c->layout.width; ++x) {
            assert(prev->At(x,y,z) == (rotation ? rotation : 50));
            c->At(x,y,z) = static_cast<unsigned char>(rotation + 1);
          }
      if (rotation == 19) {
        assert(p == 1);
        auto* mask = o.Bank(Role::kMask).Current();
        for (size_t z = 0; z < mask->layout.depth; ++z)
          for (size_t y = 0; y < mask->layout.height; ++y)
            for (size_t x = 0; x < mask->layout.width; ++x)
              mask->At(x,y,z) = static_cast<unsigned char>(z*30+y*5+x);
      } else {
        for (unsigned parity = 0; parity < 2; ++parity)
          for (auto byte : o.Bank(Role::kMask).At(parity)->bytes) assert(byte == 0xcd);
      }
      Rotate(*c, *previous.Alternate());
      for (unsigned role = 0; role < 10; ++role) {
        auto& bank = o.banks[role];
        if (role != unsigned(Role::kSelectionCurrent) &&
            role != unsigned(Role::kSelectionPrevious) && role != unsigned(Role::kMask))
          Rotate(*bank.Current(), *bank.Alternate());
        bank.Flip();
      }
      if (rotation == 19)
        Rotate(*o.Bank(Role::kMask).Alternate(), *o.Bank(Role::kMask).Current());
      assert(b.allocations == 21 && b.frees == 0);
      assert(o.workspace->layout.width == 5 && o.workspace->layout.height == 4);
    }
    assert(normal.Current()->bytes == initial_normal);
    auto* final_mask = o.Bank(Role::kMask).Current();
    for (size_t z = 0; z < 4; ++z)
      for (size_t y = 0; y < 5; ++y)
        for (size_t x = 0; x < 3; ++x)
          assert(final_mask->At(x,y,z) == z*30+(2-x)*5+y);
    // Output consumers retain every bank. Pending output work drains at teardown.
    assert(o.Bank(Role::kSelectionPrevious).Current()->At(0,0,3) == 20);
    b.pending = true;
  }
  assert(b.frees == 21 && !b.pending);
  // Zero sweeps: no invented mask write; both masks remain uninitialized.
  Backend zero;
  { Owner o(zero); for (unsigned p = 0; p < 2; ++p)
      for (auto byte : o.Bank(Role::kMask).At(p)->bytes) assert(byte == 0xcd); }
  assert(zero.frees == 21);
  // Unfiltered sweeps flip mask owners without writing or rotating contents.
  Backend unfiltered;
  {
    Owner o(unfiltered);
    for (int i = 0; i < 20; ++i) {
      for (auto& bank : o.banks) bank.Flip();
      for (unsigned p = 0; p < 2; ++p)
        for (auto byte : o.Bank(Role::kMask).At(p)->bytes) assert(byte == 0xcd);
      assert(unfiltered.allocations == 21 && unfiltered.frees == 0);
    }
  }
  assert(unfiltered.frees == 21);
}
int main() { LayoutChecks(); PairChecks(); LifetimeChecks(); }
'''


class RotationStorageTest(unittest.TestCase):
    def compile_run(self, header=None, cpp=CPP):
        with tempfile.TemporaryDirectory(prefix="rotation-storage-") as tmp:
            root = pathlib.Path(tmp)
            if header is not None:
                target = root / "src/colmap/mvs/owned_rotation_storage.h"
                target.parent.mkdir(parents=True)
                target.write_text(header)
            source = root / "test.cc"
            source.write_text(cpp)
            binary = root / "test"
            compiled = subprocess.run(
                ["g++", "-std=c++17", "-Wall", "-Wextra", "-Werror", "-pedantic",
                 "-I", str(root), "-I", str(ROOT), str(source), "-o", str(binary)],
                capture_output=True, text=True, timeout=30,
            )
            self.assertEqual(compiled.returncode, 0, compiled.stderr)
            return subprocess.run([str(binary)], capture_output=True, text=True,
                                  timeout=10, cwd=tmp)

    def test_actual_helper(self):
        result = self.compile_run()
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_negative_helper_mutants(self):
        original = HEADER.read_text()
        mutants = {
            "wrong_parity": ("current_parity_ ^= 1;", "current_parity_ = 0;"),
            "alias_alternate": ("return At(current_parity_ ^ 1);", "return At(current_parity_);"),
            "early_free": ("current_parity_ ^= 1;", "slots_[current_parity_].reset(); current_parity_ ^= 1;"),
            "wrong_pitched_span": ("CheckedMultiply(returned_pitch, slot.height)",
                                   "CheckedMultiply(row_bytes, slot.height)"),
            "missing_mask": ("role_index <= 10", "role_index <= 8"),
        }
        for name, (old, new) in mutants.items():
            with self.subTest(mutant=name):
                self.assertEqual(original.count(old), 1, name)
                result = self.compile_run(original.replace(old, new))
                self.assertNotEqual(result.returncode, 0, name)

    def test_negative_composition_mutants(self):
        mutants = {
            "no_destructor_drain": ("~Owner() { backend.Drain(); }", "~Owner() {}"),
            "no_failure_drain": ("} catch (...) {\n      backend.Drain();", "} catch (...) {"),
            "selection_alias": ("auto* c = current.Current(); auto* prev = previous.Current();",
                                "auto* c = current.Current(); auto* prev = current.Current();"),
            "missing_final_mask_rotation": (
                "Rotate(*o.Bank(Role::kMask).Alternate(), *o.Bank(Role::kMask).Current());",
                "o.backend.Drain();"),
        }
        for name, (old, new) in mutants.items():
            with self.subTest(mutant=name):
                self.assertEqual(CPP.count(old), 1, name)
                result = self.compile_run(cpp=CPP.replace(old, new))
                self.assertNotEqual(result.returncode, 0, name)


if __name__ == "__main__":
    unittest.main()
