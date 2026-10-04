# PR2-integrated B1 — separate benchmark-lineage review

Base: `d70fb56235904b2a8c72eadc6fe3d73cedb29e58` (PR2 + PR3 + existing benchmark seed).
This is NOT the main-semantics product PR, and not its exact-upstream validation
candidate. Do not merge/transplant the main probability path here. No seed,
specialized sweep selection, numerical kernel, calibration or source-order change.
No remote publication is authorized.

## Adaptation

Reuse the reviewed compact planner and GpuMat adapter verbatim. Only persistent
`prev_sel_prob_map_` and `cost_map_` opt into compact storage. Both rotate through
ONE persistent, disjoint compact float-plane workspace, sequentially in the
original map/layer order, with the existing final synchronization per rotation.

`sel_prob_map_` remains PR2's ordinary pitched beta scratch. Its initial and
rotated dimensions remain EXACTLY `width, 32 + ceil(height/32), S`. The rotation
sequence remains:

1. Existing PRNG, depth, normal and reference-image rotations/rebinding.
2. Existing `sel_prob_map_.reset()`.
3. Persistent prev rotation using its own backing and the shared workspace.
4. Existing beta scratch reallocation, unchanged arguments/policy.
5. Cost rotation using its own backing and the same workspace.
6. Existing calibration update and launch-configuration update.

Both metadata views are prepared before each reuse submission. Existing blocks
and swaps remain. There is no sel->prev copy, no scratch view reuse and no numeric
recomputation. Workspace is allocated once per problem. Other buffers stay on
their existing paths; this does not fix single-plane or mask transients.

## Local verification

The dependency-free rotation_storage_plan_test exercises layout/ownership/bit
preservation, disjoint workspace, four rotations and callback scheduling. A new
case rotates two different volumes sequentially through the same plane and proves
that each result and the other volume survive workspace reuse. PR2-specific
lifetime fixtures cover S1/10/61/64, photo/geometry and filtering off/on, including
exact beta reset -> prev -> beta allocation -> cost ordering and all other overlaps.
The main-only scratch recurrence model is intentionally not transplanted here.

```
g++ -std=c++17 -O2 -DNDEBUG -Wall -Wextra -Werror -pthread \
  -DCOLMAP_ROTATION_STORAGE_STANDALONE -Isrc \
  src/colmap/mvs/rotation_storage_plan_test.cc -o /tmp/pr2-rotation-test
/tmp/pr2-rotation-test
COLMAP_ROTATION_MODEL_DUMP=1 /tmp/pr2-rotation-test
```

Source contracts compare the complete numerical-kernel prefix to d70, pin both
beta allocation statements, check the actual reset/reuse/reallocation order and
verify the reviewed adapter/planner are unchanged. Seed, PR3 tile code, ordinary
rotation kernel and all non-target rotation blocks remain unchanged.

## Memory interpretation — no fit promise

The fixture uses W3930,H2210, 48-byte PRNG elements and synthetic 128-byte pitches;
these are test inputs, NOT observed CUDA footprints. It labels logical and known
physical totals separately and leaves array/driver physical storage unknown.

B1 removes the S-plane replacement allocations, but peak saving is NOT necessarily
a full S-plane volume: the persistent workspace and next-highest transient remain.
Filtered S61/64 fixtures peak at mask rotation. For small S (tested1/10), PRNG
already dominated and the extra persistent workspace can INCREASE peak by one
plane. No policy is added to hide that case; measure the complete workload.

CUDA compilation, all632 from-disk seed0/tile32 digests, measured device peaks and
throughput remain external acceptance gates. This d70 lineage supports the PR3
`--PatchMatchStereo.sweep_tile 32` option; unlike the stock-based product candidate,
pin it explicitly alongside `--PatchMatchStereo.random_seed 0`.

## Measurement path

A separate diagnostic overlay will register compact BACKINGS once, not count
metadata aliases twice, and preserve PR2-specific scratch-release/reallocation
markers. It is not part of this untraced candidate.

Device-wide samples alone cannot attribute per-worker peaks under MPS. First run
isolated N1 on the otherwise quiescent GPU, with baseline occupancy recorded before
worker launch, external device/PID sampling where available, and allocation-level
cudaMemGetInfo probes. Report absolute device peak AND baseline; an incremental
worker-attributed delta requires verified absence of other variable GPU clients.
If MPS exposes only server memory, do not relabel it as client memory.

Then measure N2/N3/N4 aggregate device peaks/exit statuses/full outputs. Never divide
an aggregate peak by N and call it measured per-worker memory. Trace totals explain
allocation lifetime but exclude driver overhead. Report sampled maxima honestly;
external polling can miss short transients and tracing perturbs timing. Use trace-
OFF runs for throughput and demonstrate OFF/ON byte neutrality separately.
