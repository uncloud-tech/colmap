# B2b: compact geometric PRNG (benchmark only)

**Explicit opt-in:** `COLMAP_PATCH_MATCH_COMPACT_PRNG=1`, geometric consistency,
and `random_seed == 0` are all required. Other values, photometric runs, and the
production default (`random_seed < 0`) retain the full map. This is NOT a claim
of general production byte-exactness: compact coordinate seeding is not
compatible with historical launch-geometry seeding. The seed-0 CUDA benchmark
(632 depth/normal digests, from-disk, tile32) remains a mandatory external gate.
No CUDA compiler or GPU validation was available here.

## Mapping proof and lifetime

`cuda_rotate.h` sends original `(r,c)` to `(W-1-c,r)`. Its successive inverse
row-zero mappings are:

| Orientation | Original row | Original column | Direction |
|---|---|---|---|
| 0 | 0 | c | top, forward |
| 1 | c | W-1 | right, forward |
| 2 | H-1 | W-1-c | bottom, reversed |
| 3 | H-1-c | 0 | left, reversed |

Store each perimeter pixel once, including shared corners. Initialize each
state with exactly `curand_init(PixelSeed(0, original_row, original_col),0,0)`.
Never reinitialize after a sweep: the updated corner state must survive into
its adjacent edge's sweep. Interior states are never observed in the geometric
leg. Photometric initialization still uses the original full map.

The compact path skips only redundant PRNG reinitialization and rotation
kernels at rotation time. It keeps the PRNG rotation scope and its original
`CUDA_SYNC_AND_CHECK` boundary. All other launches/order/scopes/synchronization
remain unchanged; no concurrency is introduced. The sweep kernel body is
unchanged; only its parameter type and launch argument differ. The row view's
opt-out branch delegates Get/Set unchanged to the full map. CUDA code generation
and byte-exact results still require the external gate.

This composes with B2c: leave its other rotation lifetimes alone and preserve
the PRNG conditional/sync boundary when integrating. Gate the combined branch.

## Expected memory saving

For W=3930, H=2210 and 48-byte cuRAND states: full storage is 416,894,400
bytes (about 0.388 GiB, before pitch); perimeter storage is 12,276 states,
589,248 bytes (<1 MB). Expected saving: approximately **0.388 GiB steady plus
0.388 GiB transient** (the PRNG rotation double-buffer). These are storage
estimates, not measured total-worker peak reductions: other live rotation
buffers determine the actual peak.

## Host validation

Standalone tests exercise the actual production indexing and row-view helper:
all dimensions 1..35 on each axis, singleton/degenerate shapes, square and
rectangular shapes, and 3930x2210; all four orientations; five full rotation
round trips with state updates; deduplicated corners; independent full-map
CCW rotation; opt-out delegation; and the full opt-in/seed/phase gate matrix.
Synthetic 64-bit states use PixelSeed and Mix, not cuRAND. These tests prove
mapping/state lifetime, not the GPU cuRAND implementation or output digests.
Checks remain active under NDEBUG. A deliberately incorrect non-reversed
bottom edge was also rejected by the suite.

Run from the repository root (g++ 12.2.0 was used):

```sh
mkdir -p .local-prng
g++ -std=c++17 -O2 -DNDEBUG -Wall -Wextra -Werror -Isrc \
  tests/host/compact_prng_test.cc -o .local-prng/compact_prng_test
.local-prng/compact_prng_test
g++ -std=c++17 -O2 -DNDEBUG -Wall -Wextra -Werror \
  -fsanitize=address,undefined -fno-omit-frame-pointer -Isrc \
  tests/host/compact_prng_test.cc -o .local-prng/compact_prng_test_sanitized
.local-prng/compact_prng_test_sanitized
g++ -std=c++17 -O2 -DNDEBUG -Wall -Wextra -Werror \
  -DCOLMAP_ROTATION_STORAGE_STANDALONE -Isrc \
  src/colmap/mvs/rotation_storage_plan_test.cc \
  -o .local-prng/rotation_storage_plan_test
.local-prng/rotation_storage_plan_test
```

All three runs passed. Full CTest could not run (`ctest` is not installed).
The first test compile intentionally preceded the missing helper. `/tmp` is
noexec here, so test executables were moved to the worktree. A first B1 test
compile without its standalone define found no GTest; the command above is
the corrected successful standalone invocation. No GPU build/test is claimed.
