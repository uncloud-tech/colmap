# Packed PatchMatch consistency mask (B2a)

Independent implementation against `998ff4de`, without B1/B2c dependencies.
The requested B1 plan and storage-reuse APIs are absent from this baseline.
Per supervisor clarification, packing is unconditional, not a user-facing knob.
It changes representation only: eight source flags per byte, in pitched byte
planes. Each sweep thread exclusively owns all flags for its column's pixels;
no atomics or shared-byte writes across pixels are needed.

The numerical filter body, sweep order, allocation scopes, and synchronization
are unchanged. The existing final GPU rotation operates on packed byte planes;
it already restores the original orientation, so the host must **not** rotate
again. `GetConsistentImageIdxs` expands directly into the original sparse list
format, without allocating a full unpacked host mask. Source ordering and
column/row/count ordering are unchanged. Unused last-plane bits are initialized
to zero and excluded when expanding. Source counts greater than 64 are supported.

## Host validation

Standalone test (no CUDA, Eigen, glog, or GTest dependencies):

```sh
mkdir -p build/b2a
g++ -std=c++17 -O2 -DNDEBUG -Wall -Wextra -Werror -Isrc \
  src/colmap/mvs/packed_consistency_mask_test.cc -o build/b2a/mask-test
build/b2a/mask-test
g++ -std=c++17 -O2 -DNDEBUG -Wall -Wextra -Werror \
  -fsanitize=address,undefined -fno-omit-frame-pointer -Isrc \
  src/colmap/mvs/packed_consistency_mask_test.cc -o build/b2a/mask-test-sanitized
ASAN_OPTIONS=detect_leaks=1 UBSAN_OPTIONS=halt_on_error=1 \
  build/b2a/mask-test-sanitized
git diff --check
```

Both executions passed with g++ (Debian 12.2.0-14+deb12u1) 12.2.0:

```text
PASS: packed round-trip, padding, filtering, rotation, lists, and graph export equivalence (162 cases)
```

Coverage: S = 1, 7, 8, 9, 63, 64, 65, 129, 257; rectangular, single-row,
and single-column images; padded pitches; all-zero/all-one, alternating,
pseudorandom-pattern, first/last-source, and rejected-pixel masks; repeated
set/clear; source bounds and padding guards; dirty unused bits excluded from
lists; four consecutive counter-clockwise rotations; exact list equality to
an independent unpacked reference; sparse consistency-graph lookup equivalence;
and byte equality in the existing ASCII-header/little-endian-int32 export format.
The test models the unchanged graph lookup/export format, not an integration
execution of `ConsistencyGraph::Write` (Eigen/glog unavailable).
Checks remain active under `-DNDEBUG`.

Initial test compilation failed because the new helper did not exist, before
implementation. An initial executable in `/tmp` could not run (noexec mount);
the successful commands above use this worktree's ignored build directory.
`ctest --test-dir build --output-on-failure` could not run: `ctest` not installed.
No full COLMAP, CUDA compilation, GPU runtime, or seed-0 digest validation was
performed. Before accepting a GPU deployment, run the from-disk tile32 seed-0
632-depth/normal-digest gate **and** compare consistent-image lists/graph exports
against the unpacked baseline; the digest gate does not cover the mask itself.

## Expected memory reduction

At W=3930, H=2210, S=64, excluding CUDA pitch padding:

| Mask storage | Before (GiB) | After (GiB) | Saved (GiB) |
| --- | ---: | ---: | ---: |
| Steady | 0.517684 | 0.064711 | 0.452974 |
| During existing two-buffer rotation | 1.035368 | 0.129421 | 0.905947 |

Logical mask storage is exactly 8x smaller at S=64. Actual allocations depend on
CUDA pitch. For the supplied PR2+B1 measurements, subtracting these savings
suggests steady ~7.005 GiB and mask-rotation peak ~7.588 GiB; this topic alone does
not achieve the ~7.1 GiB target. Other transient allocations can become the peak.
These are analytical estimates, not measured GPU results.
