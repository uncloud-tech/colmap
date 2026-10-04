# B1 rotation storage reuse — implementation and validation notes

This topic is based on fork main `998ff4de70344c0fcc5f10bf36f71924026dba95`.
It changes storage ownership/layout, not rotation/numerical kernels, options,
source ordering, precision or concurrency. It is not the PR2 bounded-beta lineage.

## Algorithm / ownership

Cost and both probability volumes use opt-in compact backing. A layer occupies
`W*H*sizeof(float)` bytes in either orientation. Cost rotates one layer into a
disjoint compact workspace, then copies that plane back on the same implicit
CUDA stream before starting the next layer. The workspace is allocated once per
problem. The existing final rotation synchronization is retained, not augmented.

Both probability metadata wrappers are prepared before probability submission;
main still rotates completed selection probabilities into a DISTINCT previous-
probability backing, then publishes the reoriented selection scratch wrapper.
The cost block prepares both metadata views before submission too. Shared owners
retain the backing through swaps; alias destruction does not free device memory.
Only final backing release calls cudaFree. Ordinary GpuMat construction and Rotate,
GpuMatView accessors, and cuda_rotate.h are unchanged. Compact storage is opt-in.

The scratch reuse proof is in the approved spec: main's backward loop writes every
valid row/layer for a column before either beta read; the forward pass writes the
probability before filtering reads it; invalid columns do not access scratch;
export reads prev. The relevant kernel prefix is unchanged from the base. A
poisoned-scratch host model exercises that ordering, not likelihood numerics.

## Host tests

```
g++ -std=c++17 -O2 -DNDEBUG -Wall -Wextra -Werror -pthread \
  -DCOLMAP_ROTATION_STORAGE_STANDALONE -Isrc \
  src/colmap/mvs/rotation_storage_plan_test.cc -o /tmp/rotation-test
/tmp/rotation-test
COLMAP_ROTATION_MODEL_DUMP=1 /tmp/rotation-test
```

CTest registration: `mvs/rotation_storage_plan_test`. Checks remain enabled under
NDEBUG. Coverage includes overflow/int limits, capacity/ownership validation,
exactly-once release, overlap rejection, ascending rotate/copy schedule, arbitrary
32-bit payloads including NaNs/signed zero, four rotations, guards, failed callback
propagation, poisoned scratch with every filter combination, and a lifetime ledger.

The ledger tests S=1/61/64, photo/geometry, filter on/off, both orientations,
reference filtering input, rebinding overlaps, PRNG/depth/normal rotations,
source/pose/global storage, and the late mask through cost and mask rotation.
Aliases add zero bytes. Teardown returns both tracked sums to zero. Filtered
S=61/64 cases deliberately have the remaining peak at mask rotation, not cost.

Model output is SYMBOLIC, not measured VRAM: the fixture uses W3930,H2210,
a 48-byte PRNG element and synthetic 128-byte pitch alignment. Known-physical
sums exclude unknown CUDA-array storage and all driver/context overhead. Actual
pitches and sizeof(curandState) must come from the GPU trace. Main's full scratch
volume remains, so these totals are not the d70/PR2 target-capacity estimates.

## Required external acceptance

At authoring time, Release/-Werror and ASan+UBSan host tests pass; direct source
checks confirm unchanged ordinary paths/kernel bodies and exactly one final
reuse synchronization. No local nvcc/CMake/CTest/GPU environment is available.
LSP's .h fallback is a C parser and is not CUDA compilation evidence.

Photogram owns:
1. CUDA build of the integrated topic (not just the unused template declarations).
2. Candidate clean replay onto exact upstream d7ffb9c plus the identical external
   minimal seed patch; full158/from-disk/seed0/tile32, 632 exact reference digests.
   The existing stock seeded reference/control is already established by the
   coordinator; do not rebuild/re-run a duplicate control without a reason.
3. Measured device peak and rotation-heavy throughput; no regression accepted.
4. Immediate separately reviewed PR2 integration after the product PR lands,
   then benchmark-lineage memory-goal measurements. Product-main measurements
   alone cannot establish the N4/32GB or N2/16GB goal.

Trace hooks and benchmark seed are excluded from this product topic. Its separate
diagnostic overlay must count compact BACKINGS once, not count alias wrappers.
Source-texture cache S1 stays on HOLD; smaller transients remain unchanged.
