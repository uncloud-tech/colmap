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

CPP = (
    "\n"
    '#include "src/colmap/mvs/owned_rotation_storage.h"\n'
    "#include <array>\n"
    "#include <cassert>\n"
    "#include <climits>\n"
    "#include <limits>\n"
    "#include <string>\n"
    "#include <type_traits>\n"
    "using namespace colmap::mvs::owned;\n"
    "using Role = RotationRole;\n"
    "\n"
    "// Use runtime assertions (compilation intentionally does no"
    "t define NDEBUG).\n"
    "template <class F> void Reject(F f) {\n"
    "  bool rejected = false;\n"
    "  try { f(); } catch (const std::exception&) { rejected = tr"
    "ue; }\n"
    "  assert(rejected);\n"
    "}\n"
    "struct Backend {\n"
    "  int attempts = 0, allocations = 0, frees = 0, drains = 0;\n"
    "  int fail_at = -1;\n"
    "  bool pending = false;\n"
    "  void Drain() { ++drains; pending = false; }\n"
    "};\n"
    "struct Buffer {\n"
    "  Backend& backend;\n"
    "  const RotationSlotLayout layout;\n"
    "  const RotationAllocationRecord record;\n"
    "  std::vector<unsigned char> bytes;\n"
    "  Buffer(Backend& b, const RotationSlotLayout& s, size_t pit"
    "ch)\n"
    "      : backend(b), layout(s), record(MakeRotationAllocation"
    "Record(s, pitch)),\n"
    "        bytes(record.pitched_span_bytes, 0xcd) { ++backend.a"
    "llocations; }\n"
    "  ~Buffer() { assert(!backend.pending); ++backend.frees; }\n"
    "  unsigned char& At(size_t x, size_t y, size_t z) {\n"
    "    assert(x < layout.width && y < layout.height && z < layo"
    "ut.depth);\n"
    "    return bytes[(z * layout.height + y) * record.pitch_byte"
    "s +\n"
    "                 x * layout.element_bytes];\n"
    "  }\n"
    "};\n"
    "using Pair = OwnedRotationPair<Buffer>;\n"
    "static_assert(!std::is_copy_constructible<Pair>::value);\n"
    "static_assert(!std::is_move_constructible<Pair>::value);\n"
    "static_assert(!std::is_copy_assignable<Pair>::value);\n"
    "static_assert(!std::is_move_assignable<Pair>::value);\n"
    "\n"
    "// Drain belongs to the enclosing owner, including its const"
    "ructor catch.\n"
    "// The helper deliberately does not know about streams or al"
    "location backends.\n"
    "struct Owner {\n"
    "  Backend& backend;\n"
    "  OwnedRotationLayout layout;\n"
    "  std::array<Pair, 10> banks;\n"
    "  std::unique_ptr<Buffer> workspace;\n"
    "  explicit Owner(Backend& b, size_t w = 3, size_t h = 5, siz"
    "e_t s = 4)\n"
    "      : backend(b), layout(BuildOwnedRotationLayout(w, h, s)"
    ") {\n"
    "    try {\n"
    "      for (const auto& slot : layout.slots) {\n"
    "        int index = backend.attempts++;\n"
    "        if (index == backend.fail_at) throw std::runtime_err"
    'or("allocation");\n'
    "        // Differing, nontrivial returned pitches; no shape/"
    "pitch inference.\n"
    "        auto buffer = std::make_unique<Buffer>(\n"
    "            backend, slot, slot.width * slot.element_bytes +"
    " 7 + index);\n"
    "        backend.pending = true;\n"
    "        if (slot.role == Role::kWorkspace) workspace = std::"
    "move(buffer);\n"
    "        else banks[static_cast<size_t>(slot.role)].Set(slot."
    "parity,\n"
    "                                                     std::mo"
    "ve(buffer));\n"
    "      }\n"
    "      backend.Drain();\n"
    "    } catch (...) {\n"
    "      backend.Drain();\n"
    "      throw;\n"
    "    }\n"
    "  }\n"
    "  ~Owner() { backend.Drain(); }\n"
    "  Pair& Bank(Role r) { return banks[static_cast<size_t>(r)];"
    " }\n"
    "};\n"
    "void Rotate(Buffer& src, Buffer& dst) {\n"
    "  assert(&src != &dst);\n"
    "  assert(src.layout.width == dst.layout.height);\n"
    "  assert(src.layout.height == dst.layout.width);\n"
    "  assert(src.layout.depth == dst.layout.depth);\n"
    "  src.backend.pending = true;\n"
    "  for (size_t z = 0; z < src.layout.depth; ++z)\n"
    "    for (size_t y = 0; y < src.layout.height; ++y)\n"
    "      for (size_t x = 0; x < src.layout.width; ++x)\n"
    "        dst.At(src.layout.height - 1 - y, x, z) = src.At(x, "
    "y, z);\n"
    "  src.backend.Drain();\n"
    "}\n"
    "void LayoutChecks() {\n"
    "  auto l = BuildOwnedRotationLayout(3, 5, 4);\n"
    "  assert(l.width == 3 && l.height == 5 && l.sources == 4);\n"
    "  assert(l.slots.size() == 21);\n"
    "  assert(l.logical_bytes == 15 * (146 + 26 * 4) + 8 * 5 * 4)"
    ";\n"
    "  const std::array<size_t, 11> depths{1,1,3,1,1,1,4,4,4,4,2}"
    ";\n"
    "  const std::array<size_t, 11> elements{48,4,4,1,4,4,4,4,4,1"
    ",4};\n"
    "  const std::array<const char*, 11> names{\n"
    '    "prng", "depth", "normal", "ref_image", "ref_sum", "ref_'
    'squared_sum",\n'
    '    "selection_current", "selection_previous", "cost", "mask'
    '", "workspace"};\n'
    "  size_t index = 0, sum = 0;\n"
    "  for (unsigned role = 0; role < 11; ++role) {\n"
    "    auto r = static_cast<Role>(role);\n"
    "    assert(std::string(RotationRoleName(r)) == names[role]);"
    "\n"
    "    for (unsigned p = 0; p < (role == 10 ? 1u : 2u); ++p) {\n"
    "      const auto& s = l.At(r, p);\n"
    "      assert(&s == &l.slots[index++]);\n"
    "      assert(s.role == r && s.parity == p);\n"
    "      assert(s.width == (role == 10 ? 5u : (p ? 5u : 3u)));\n"
    "      assert(s.height == (role == 10 ? 4u : (p ? 3u : 5u)));"
    "\n"
    "      assert(s.depth == depths[role] && s.element_bytes == e"
    "lements[role]);\n"
    "      assert(s.requested_span_bytes == s.width*s.height*s.de"
    "pth*s.element_bytes);\n"
    "      auto a = MakeRotationAllocationRecord(s, s.width*s.ele"
    "ment_bytes + 13);\n"
    "      assert(a.role == s.role && a.parity == p && a.width =="
    " s.width);\n"
    "      assert(a.height == s.height && a.depth == s.depth);\n"
    "      assert(a.element_bytes == s.element_bytes);\n"
    "      assert(a.pitch_bytes == s.width*s.element_bytes + 13);"
    "\n"
    "      assert(a.requested_span_bytes == s.requested_span_byte"
    "s);\n"
    "      assert(a.pitched_span_bytes == a.pitch_bytes*s.height*"
    "s.depth);\n"
    '      assert(std::string(a.lifetime_phase) == "prepared_thro'
    'ugh_outputs");\n'
    "      sum += s.requested_span_bytes;\n"
    "      Reject([&] { MakeRotationAllocationRecord(s, s.width*s"
    ".element_bytes - 1); });\n"
    "    }\n"
    "  }\n"
    "  assert(sum == l.logical_bytes);\n"
    "  auto large = BuildOwnedRotationLayout(6204, 4132, 8);\n"
    "  assert(large.logical_bytes == 9075161568ULL);\n"
    "  Reject([&] { l.At(Role::kWorkspace, 1); });\n"
    "  Reject([&] { l.At(Role::kDepth, 2); });\n"
    "  Reject([&] { l.At(static_cast<Role>(99), 0); });\n"
    "  Reject([] { RotationRoleName(static_cast<Role>(-1)); });\n"
    "  Reject([] { BuildOwnedRotationLayout(0, 1, 1); });\n"
    "  Reject([] { BuildOwnedRotationLayout(1, 0, 1); });\n"
    "  Reject([] { BuildOwnedRotationLayout(1, 1, 0); });\n"
    "  for (int axis = 0; axis < 3; ++axis) {\n"
    "    size_t dims[3]{1,1,1}; dims[axis] = size_t(INT_MAX) + 1;"
    "\n"
    "    Reject([&] { BuildOwnedRotationLayout(dims[0], dims[1], "
    "dims[2]); });\n"
    "  }\n"
    "  Reject([] { BuildOwnedRotationLayout(INT_MAX, INT_MAX, INT"
    "_MAX); });\n"
    "  // Individual spans fit size_t here, but their sum overflo"
    "ws on 64-bit hosts.\n"
    "  if (sizeof(size_t) == 8)\n"
    "    Reject([] { BuildOwnedRotationLayout(400000000, 40000000"
    "0, 1); });\n"
    "  auto slot = l.At(Role::kNormal, 0);\n"
    "  Reject([&] { MakeRotationAllocationRecord(slot, std::numer"
    "ic_limits<size_t>::max()); });\n"
    "  // The first multiplication fits, but multilayer pitched s"
    "pan overflows.\n"
    "  Reject([&] { MakeRotationAllocationRecord(\n"
    "      slot, std::numeric_limits<size_t>::max() / slot.height"
    "); });\n"
    "  for (int invalid = 0; invalid < 13; ++invalid) {\n"
    "    auto bad = slot;\n"
    "    switch (invalid) {\n"
    "      case 0: bad.width = 0; break;\n"
    "      case 1: bad.height = 0; break;\n"
    "      case 2: bad.depth = 0; break;\n"
    "      case 3: bad.element_bytes = 0; break;\n"
    "      case 4: bad.parity = 2; break;\n"
    "      case 5: bad.role = static_cast<Role>(100); break;\n"
    "      case 6: bad.width = size_t(INT_MAX) + 1; break;\n"
    "      case 7: ++bad.requested_span_bytes; break;\n"
    "      case 8: bad.role = Role::kWorkspace; bad.parity = 1; b"
    "reak;\n"
    "      case 9: bad.height = size_t(INT_MAX) + 1; break;\n"
    "      case 10: bad.depth = size_t(INT_MAX) + 1; break;\n"
    "      case 11: bad.depth = 2; break;\n"
    "      case 12: bad.element_bytes = 8; break;\n"
    "    }\n"
    "    Reject([&] { MakeRotationAllocationRecord(bad, 4096); })"
    ";\n"
    "  }\n"
    "}\n"
    "void PairChecks() {\n"
    "  Backend b;\n"
    "  {\n"
    "    Pair p;\n"
    "    assert(p.CurrentParity() == 0);\n"
    "    Reject([&] { p.Current(); }); Reject([&] { p.Alternate()"
    "; });\n"
    "    Reject([&] { p.operator->(); }); Reject([&] { p.At(2); }"
    ");\n"
    "    Reject([&] { p.Flip(); });\n"
    "    Reject([&] { p.Set(0, nullptr); });\n"
    "    Reject([&] { p.Set(2, nullptr); });\n"
    "    auto l = BuildOwnedRotationLayout(2, 2, 1);\n"
    "    p.Set(0, std::make_unique<Buffer>(b, l.At(Role::kDepth, "
    "0), 16));\n"
    "    Reject([&] { p.Flip(); });\n"
    "    Reject([&] { p.Set(0, std::make_unique<Buffer>(b, l.At(R"
    "ole::kDepth, 0), 16)); });\n"
    "    p.Set(1, std::make_unique<Buffer>(b, l.At(Role::kDepth, "
    "1), 20));\n"
    "    assert(p.At(0) != p.At(1)); // Square images still have "
    "separate owners.\n"
    "    auto* original = p.Current(); auto* alternate = p.Altern"
    "ate();\n"
    "    int allocs = b.allocations, frees = b.frees;\n"
    "    for (int i = 0; i < 20; ++i) {\n"
    "      p.Flip();\n"
    "      assert(p.CurrentParity() == unsigned((i + 1) % 2));\n"
    "      assert(p.Current() == ((i % 2) ? original : alternate)"
    ");\n"
    "      assert(p.Alternate() == ((i % 2) ? alternate : origina"
    "l));\n"
    "      assert(p.operator->() == p.Current());\n"
    "    }\n"
    "    assert(b.allocations == allocs && b.frees == frees);\n"
    "  }\n"
    "  assert(b.allocations == b.frees);\n"
    "}\n"
    "void LifetimeChecks() {\n"
    "  for (int fail = 0; fail < 21; ++fail) {\n"
    "    Backend b; b.fail_at = fail;\n"
    "    Reject([&] { Owner o(b); });\n"
    "    assert(b.attempts == fail + 1 && b.allocations == fail);"
    "\n"
    "    assert(b.frees == fail && b.drains == 1 && !b.pending);\n"
    "  }\n"
    "  Backend b;\n"
    "  {\n"
    "    Owner o(b);\n"
    "    assert(b.allocations == 21 && b.frees == 0);\n"
    "    std::array<Buffer*, 4> selection{\n"
    "      o.Bank(Role::kSelectionCurrent).At(0), o.Bank(Role::kS"
    "electionCurrent).At(1),\n"
    "      o.Bank(Role::kSelectionPrevious).At(0), o.Bank(Role::k"
    "SelectionPrevious).At(1)};\n"
    "    for (int i = 0; i < 4; ++i)\n"
    "      for (int j = i + 1; j < 4; ++j) assert(selection[i] !="
    " selection[j]);\n"
    "    auto& normal = o.Bank(Role::kNormal);\n"
    "    for (size_t z = 0; z < 3; ++z)\n"
    "      for (size_t y = 0; y < 5; ++y)\n"
    "        for (size_t x = 0; x < 3; ++x)\n"
    "          normal.Current()->At(x,y,z) = static_cast<unsigned"
    " char>(z*50+y*3+x);\n"
    "    auto initial_normal = normal.Current()->bytes;\n"
    "    // Previous parity zero is the only initialized selectio"
    "n slot.\n"
    "    for (auto& byte : selection[2]->bytes) byte = 50;\n"
    "    for (int rotation = 0; rotation < 20; ++rotation) {\n"
    "      unsigned p = unsigned(rotation % 2);\n"
    "      auto& current = o.Bank(Role::kSelectionCurrent);\n"
    "      auto& previous = o.Bank(Role::kSelectionPrevious);\n"
    "      assert(current.CurrentParity() == p && previous.Curren"
    "tParity() == p);\n"
    "      auto* c = current.Current(); auto* prev = previous.Cur"
    "rent();\n"
    "      assert(c != prev && c != previous.Alternate());\n"
    "      // Backward overwrites every valid multilayer pixel be"
    "fore forward reads.\n"
    "      for (size_t z = 0; z < c->layout.depth; ++z)\n"
    "        for (size_t y = 0; y < c->layout.height; ++y)\n"
    "          for (size_t x = 0; x < c->layout.width; ++x) {\n"
    "            assert(prev->At(x,y,z) == (rotation ? rotation :"
    " 50));\n"
    "            c->At(x,y,z) = static_cast<unsigned char>(rotati"
    "on + 1);\n"
    "          }\n"
    "      if (rotation == 19) {\n"
    "        assert(p == 1);\n"
    "        auto* mask = o.Bank(Role::kMask).Current();\n"
    "        for (size_t z = 0; z < mask->layout.depth; ++z)\n"
    "          for (size_t y = 0; y < mask->layout.height; ++y)\n"
    "            for (size_t x = 0; x < mask->layout.width; ++x)\n"
    "              mask->At(x,y,z) = static_cast<unsigned char>(z"
    "*30+y*5+x);\n"
    "      } else {\n"
    "        for (unsigned parity = 0; parity < 2; ++parity)\n"
    "          for (auto byte : o.Bank(Role::kMask).At(parity)->b"
    "ytes) assert(byte == 0xcd);\n"
    "      }\n"
    "      Rotate(*c, *previous.Alternate());\n"
    "      for (unsigned role = 0; role < 10; ++role) {\n"
    "        auto& bank = o.banks[role];\n"
    "        if (role != unsigned(Role::kSelectionCurrent) &&\n"
    "            role != unsigned(Role::kSelectionPrevious) && ro"
    "le != unsigned(Role::kMask))\n"
    "          Rotate(*bank.Current(), *bank.Alternate());\n"
    "        bank.Flip();\n"
    "      }\n"
    "      if (rotation == 19)\n"
    "        Rotate(*o.Bank(Role::kMask).Alternate(), *o.Bank(Rol"
    "e::kMask).Current());\n"
    "      assert(b.allocations == 21 && b.frees == 0);\n"
    "      assert(o.workspace->layout.width == 5 && o.workspace->"
    "layout.height == 4);\n"
    "    }\n"
    "    assert(normal.Current()->bytes == initial_normal);\n"
    "    auto* final_mask = o.Bank(Role::kMask).Current();\n"
    "    for (size_t z = 0; z < 4; ++z)\n"
    "      for (size_t y = 0; y < 5; ++y)\n"
    "        for (size_t x = 0; x < 3; ++x)\n"
    "          assert(final_mask->At(x,y,z) == z*30+(2-x)*5+y);\n"
    "    // Output consumers retain every bank. Pending output wo"
    "rk drains at teardown.\n"
    "    assert(o.Bank(Role::kSelectionPrevious).Current()->At(0,"
    "0,3) == 20);\n"
    "    b.pending = true;\n"
    "  }\n"
    "  assert(b.frees == 21 && !b.pending);\n"
    "  // Zero sweeps: no invented mask write; both masks remain "
    "uninitialized.\n"
    "  Backend zero;\n"
    "  { Owner o(zero); for (unsigned p = 0; p < 2; ++p)\n"
    "      for (auto byte : o.Bank(Role::kMask).At(p)->bytes) ass"
    "ert(byte == 0xcd); }\n"
    "  assert(zero.frees == 21);\n"
    "  // Unfiltered sweeps flip mask owners without writing or r"
    "otating contents.\n"
    "  Backend unfiltered;\n"
    "  {\n"
    "    Owner o(unfiltered);\n"
    "    for (int i = 0; i < 20; ++i) {\n"
    "      for (auto& bank : o.banks) bank.Flip();\n"
    "      for (unsigned p = 0; p < 2; ++p)\n"
    "        for (auto byte : o.Bank(Role::kMask).At(p)->bytes) a"
    "ssert(byte == 0xcd);\n"
    "      assert(unfiltered.allocations == 21 && unfiltered.free"
    "s == 0);\n"
    "    }\n"
    "  }\n"
    "  assert(unfiltered.frees == 21);\n"
    "}\n"
    "int main() { LayoutChecks(); PairChecks(); LifetimeChecks();"
    " }\n"
)


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
                [
                    "g++",
                    "-std=c++17",
                    "-Wall",
                    "-Wextra",
                    "-Werror",
                    "-pedantic",
                    "-I",
                    str(root),
                    "-I",
                    str(ROOT),
                    str(source),
                    "-o",
                    str(binary),
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )
            self.assertEqual(compiled.returncode, 0, compiled.stderr)
            return subprocess.run(
                [str(binary)],
                capture_output=True,
                text=True,
                timeout=10,
                cwd=tmp,
            )

    def test_actual_helper(self):
        result = self.compile_run()
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_negative_helper_mutants(self):
        original = HEADER.read_text()
        mutants = {
            "wrong_parity": ("current_parity_ ^= 1;", "current_parity_ = 0;"),
            "alias_alternate": (
                "return At(current_parity_ ^ 1);",
                "return At(current_parity_);",
            ),
            "early_free": (
                "current_parity_ ^= 1;",
                "slots_[current_parity_].reset(); current_parity_ ^= 1;",
            ),
            "wrong_pitched_span": (
                "CheckedMultiply(returned_pitch, slot.height)",
                "CheckedMultiply(row_bytes, slot.height)",
            ),
            "missing_mask": ("role_index <= 10", "role_index <= 8"),
        }
        for name, (old, new) in mutants.items():
            with self.subTest(mutant=name):
                self.assertEqual(original.count(old), 1, name)
                result = self.compile_run(original.replace(old, new))
                self.assertNotEqual(result.returncode, 0, name)

    def test_negative_composition_mutants(self):
        mutants = {
            "no_destructor_drain": (
                "~Owner() { backend.Drain(); }",
                "~Owner() {}",
            ),
            "no_failure_drain": (
                "} catch (...) {\n      backend.Drain();",
                "} catch (...) {",
            ),
            "selection_alias": (
                "auto* c = current.Current(); auto* prev = previous.Current();",
                "auto* c = current.Current(); auto* prev = current.Current();",
            ),
            "missing_final_mask_rotation": (
                (
                    "Rotate(*o.Bank(Role::kMask).Alternate(), *o.Bank(Role::k"
                    "Mask).Current());"
                ),
                "o.backend.Drain();",
            ),
        }
        for name, (old, new) in mutants.items():
            with self.subTest(mutant=name):
                self.assertEqual(CPP.count(old), 1, name)
                result = self.compile_run(cpp=CPP.replace(old, new))
                self.assertNotEqual(result.returncode, 0, name)


if __name__ == "__main__":
    unittest.main()
