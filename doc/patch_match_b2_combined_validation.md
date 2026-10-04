# Combined B2 integration (host validation only)

Base: e0fdc7b (PR2 row-tile beta scratch plus integrated B1 storage).
B2b aa50051b was already cherry-picked as 029ae48 when this worker began;
its tree was identical, so no duplicate cherry-pick was attempted.
B2a 85a2b022 became 8ef5734; B2c 0f33fc60 became 73cc03a.

## Conflict resolutions

B2a:
- gpu_mat.h: retain BOTH packed-mask and B1 rotation-storage includes.
- patch_match_cuda.h: retain B2b rand_state_edges_ and packed-mask comment.
- SweepFromTopToBottom: use PackedConsistencyMaskView while retaining PR2's
  beta_scratch and mutable prev_sel_prob_map parameters. Numerical body unchanged.
- Other mask hunks merged cleanly: GetSelProbMap still exports persistent prev;
  masks allocate ceil(S/8) planes, unpack source bits directly to sparse lists,
  and rotate stored byte planes. No full selection-probability map restored.

B2c:
- gpu_mat_prng.h: keep the seeded ordinary constructor, add the compact factory
  with random_seed=-1. Adapt factory implementation to dispatch the SAME initial
  kernels with the SAME launch geometry and seed as the ordinary constructor.
- patch_match_cuda.h: full PRNG member becomes GpuMat<curandState>, retaining
  B2b's separate perimeter member and B2a mask comment; use B2c workspace comment.
- Mask allocation combines CreateCompact with PackedConsistencyMaskDepth(S).
  Rotation uses MakeCompactView (preserving packed depth) and byte workspace.
- InitWorkspaceMemory retains B2b's exact gate; gate-off allocates/initializes
  the full map using the seeded compact factory, gate-on retains perimeter state.
- Rotate retains perimeter orientation-only/synchronization path; full-map path
  uses word-plane reuse without the dead destination initializer.
- PR2 beta scratch remains bounded and recreated for rotated dimensions;
  persistent prev/cost reuse remains B1. No numerical kernel body changes or
  additional concurrency were introduced during conflict resolution.

## Gate clarification approved by supervisor

B2b is opt-in and benchmark-scoped (seed-zero geometric phase only).
B2c is unconditional-but-internal, following B1's precedent, NOT a separate
benchmark opt-in. Ordinary/public GpuMat and GpuMatPRNG constructors and ordinary
GpuMat::Rotate remain unchanged. The apparent instruction conflict was escalated
and this distinction explicitly authorized; no assumptions remain unresolved.

## Full-map/default-path byte preservation

The gate-off path rotates the entire AoS state using ForEachWord. Each component
is gathered into disjoint scratch before scatter overwrites that component;
other components remain untouched until their turn. Thus every destination byte
is supplied by the full permutation, and no rotation-destination initialization
value is consumed. Removing that initializer does not remove the initial problem
seeding. The host word test covers 48-byte FULL opaque states as well as 16-/24-byte
states, checks every scratch index is written exactly once, compares each turn to
an independent full byte-copy oracle, checks guards, and restores the original
bytes after four turns across eight shapes. This is schedule proof, NOT actual
curandState ABI, CUDA launch, or GPU numerical validation.

## Executed host commands and output

Compiler: g++ (Debian 12.2.0-14+deb12u1) 12.2.0. All eight builds/runs passed;
no tests weakened or integration-only numerical changes made.

```text
+ g++ -std=c++17 -O2 -DNDEBUG -Wall -Wextra -Werror -Isrc src/colmap/mvs/packed_consistency_mask_test.cc -o evidence/b2-combined/packed_consistency_mask-opt
+ evidence/b2-combined/packed_consistency_mask-opt
PASS: packed round-trip, padding, filtering, rotation, lists, and graph export equivalence (162 cases)
PASS: packed_consistency_mask opt
+ g++ -std=c++17 -O2 -DNDEBUG -Wall -Wextra -Werror -Isrc -fsanitize=address,undefined -fno-omit-frame-pointer src/colmap/mvs/packed_consistency_mask_test.cc -o evidence/b2-combined/packed_consistency_mask-san
+ evidence/b2-combined/packed_consistency_mask-san
PASS: packed round-trip, padding, filtering, rotation, lists, and graph export equivalence (162 cases)
PASS: packed_consistency_mask san
+ g++ -std=c++17 -O2 -DNDEBUG -Wall -Wextra -Werror -Isrc tests/host/compact_prng_test.cc -o evidence/b2-combined/compact_prng-opt
+ evidence/b2-combined/compact_prng-opt
PASS: gate, perimeter bijection, four orientations, 5 round trips, full-map equality, opt-out delegation
PASS: compact_prng opt
+ g++ -std=c++17 -O2 -DNDEBUG -Wall -Wextra -Werror -Isrc -fsanitize=address,undefined -fno-omit-frame-pointer tests/host/compact_prng_test.cc -o evidence/b2-combined/compact_prng-san
+ evidence/b2-combined/compact_prng-san
PASS: gate, perimeter bijection, four orientations, 5 round trips, full-map equality, opt-out delegation
PASS: compact_prng san
+ g++ -std=c++17 -O2 -DNDEBUG -Wall -Wextra -Werror -Isrc -pthread -DCOLMAP_ROTATION_STORAGE_STANDALONE src/colmap/mvs/rotation_word_plan_test.cc -o evidence/b2-combined/rotation_word_plan-opt
+ evidence/b2-combined/rotation_word_plan-opt
rotation word tests passed (64 shape/type cases, four rotations each)
PASS: rotation_word_plan opt
+ g++ -std=c++17 -O2 -DNDEBUG -Wall -Wextra -Werror -Isrc -pthread -DCOLMAP_ROTATION_STORAGE_STANDALONE -fsanitize=address,undefined -fno-omit-frame-pointer src/colmap/mvs/rotation_word_plan_test.cc -o evidence/b2-combined/rotation_word_plan-san
+ evidence/b2-combined/rotation_word_plan-san
rotation word tests passed (64 shape/type cases, four rotations each)
PASS: rotation_word_plan san
+ g++ -std=c++17 -O2 -DNDEBUG -Wall -Wextra -Werror -Isrc -pthread -DCOLMAP_ROTATION_STORAGE_STANDALONE src/colmap/mvs/rotation_storage_plan_test.cc -o evidence/b2-combined/rotation_storage_plan-opt
+ evidence/b2-combined/rotation_storage_plan-opt
rotation storage tests: PASS
PASS: rotation_storage_plan opt
+ g++ -std=c++17 -O2 -DNDEBUG -Wall -Wextra -Werror -Isrc -pthread -DCOLMAP_ROTATION_STORAGE_STANDALONE -fsanitize=address,undefined -fno-omit-frame-pointer src/colmap/mvs/rotation_storage_plan_test.cc -o evidence/b2-combined/rotation_storage_plan-san
+ evidence/b2-combined/rotation_storage_plan-san
rotation storage tests: PASS
PASS: rotation_storage_plan san
```

`git diff --check e0fdc7b HEAD` passed. No CUDA toolchain available.
Pending: independent review; CUDA/HIP build; real PRNG ABI/state-byte comparison;
632-digest gate; N=2/N=4 aggregate device memory; rotation-heavy throughput.
No GPU correctness, measured memory savings, or throughput claim is made.

Integration effort: approximately 6 minutes of active worker time, including
inspection, two cherry-picks/conflict resolution, all eight host build/runs,
source audit and evidence preparation; no GPU time. Original topic development
and supervisor/orchestration effort are not included.
