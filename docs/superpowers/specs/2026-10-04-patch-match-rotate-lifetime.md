# B1: byte-exact PatchMatch rotation storage reuse

Status: written spec and plan approved by photogram; direct execution authorized.
Product acceptance remains conditional on the coordinator's full GPU gates.
Source-texture cache S1 remains HOLD. Diagnostic instrumentation remains separate.

## 1. Purpose, evidence and acceptance

Eliminate the S-layer replacement allocations during cost/probability rotation
without recomputing values, changing kernels, reordering their logical launches,
or introducing concurrency. Preserve the full from-disk, seed-0, tile32 reference.

Coordinator's diagnostic sample (d70fb56, sm_89, S=61, photometric): tracked steady
5.13 GiB, tracked peak 7.106 GiB, observed device-wide peak 7.761 GiB. A full cost
or probability duplicate contributes ~1.974 GiB. Reported 40/40 map equivalence
is useful diagnostic evidence, NOT the required full 632-digest gate. Geometric
source depths add approximately 1.973 GiB in that sample.

Acceptance belongs to photogram: full158/632 exact digests, measured device peak,
and no throughput regression on rotation-heavy work. Neither tracked estimates
nor the smaller diagnostic sample establish an N4/32GB or N2/16GB fit. Consider
S=64, device overhead, filter masks and all remaining transients. The 48GB hardware
fallback remains independent; this design makes no guaranteed-capacity claim.

## 2. Branch and validation separation

Product topic: `feature/patch-match-rotate-lifetime`, draft PR against fork main
`998ff4de70344c0fcc5f10bf36f71924026dba95`. No trace or seed changes in the PR.

Main and PR2 have DIFFERENT selection-buffer semantics:

| Tree | Rotation semantics |
| --- | --- |
| main998ff4de | Rotate completed `sel_prob_map_` into `prev_sel_prob_map_`; provide full-size scratch `sel_prob_map_` for the next sweep |
| d70fb56 / PR2 | Rotate persistent `prev_sel_prob_map_`; release/reallocate bounded beta scratch |

The main product preserves the first row. A PR2 integration is a separate,
separately reviewed patch/branch, preserving PR2's scratch reset/reallocation.
It is not part of this initial implementation authorization.

Gateability: replay product commits onto exact upstream
`d7ffb9c4bd18bc9285256a91dbe3ce98a25c6248`, then apply the identical external minimal
seed patch to both control and candidate. Stop on conflicts; record product and
validation SHAs/trees/flags. Main alone does not contain PR2's memory savings:
measure the clean topic against its corresponding control, and label later PR2
integration results separately. No in-process versus from-disk comparison.

## 3. Approved execution boundary

- Unchanged CudaRotateKernel bodies, GpuMatView accessors and numerical kernels.
- Existing per-layer rotation kernel order stays ascending, with identical layer
  count and logical width/height; pitches/addresses may reflect compact storage.
- Additional SAME-STREAM D2D copies between those kernels are permitted.
- Existing outer rotation blocks, swaps, calibration updates and kernel order stay.
- Existing final rotation synchronization remains; no new explicit synchronization,
  streams, events, workers or same-device concurrency.
- Opt-in compact pitches and backing-ownership reuse are permitted. Ordinary
  GpuMat allocation, copy/rotation operations and CUDA call semantics remain
  unchanged for non-opted-in users. This means data/CUDA behavior, not identical
  compiled executable bytes after adding code.
- No FP16, recomputation, source-set changes, quality/default-option changes,
  probability-state deletion, PR2 reimplementation or calibration-topic work.

## 4. Compact storage and ownership

Only cost, sel-probability and previous-probability maps opt in on main. Add one
per-instance compact float workspace of one image plane. Other buffers keep their
ordinary allocations and existing rotation paths.

For element size E and dimensions W,H,S:

```
source_pitch = W * E
rotated_pitch = H * E
plane_bytes = W * H * E
backing_bytes = plane_bytes * S
workspace_bytes = plane_bytes
```

The source-layer interval `[s*plane_bytes, (s+1)*plane_bytes)` is invariant under
rotation. This is why the layout is compact: reinterpreting arbitrary cudaMallocPitch
storage is NOT safe, since pitch*height can change between orientations and cause
cross-layer overwrite or insufficient capacity.

Allocate backing with cudaMalloc, preserving required element alignment. Existing
view indexing works unchanged with compact row pitch. Validate positive dimensions,
all size_t products, and the int width/height/pitch limits consumed by CudaRotate.
Reject invalid reuse requests before allocation, mutation or CUDA submission.
Zero-sized consistency placeholders remain on the untouched ordinary path.

Host-only `RotationBacking` consists of a shared allocation handle plus its exact
capacity. The CUDA adapter supplies the deleter; host tests supply fake storage.
GpuMat's existing raw-pointer/view fields continue to address the data. Only the
compact path uses the shared owner; ordinary destruction retains its original
cudaFree call. No GPU context/device-selection changes are introduced by deleters.

A reoriented metadata wrapper references the same backing with swapped W/H and
updated pitch, preserving S and capacity. Creating/destroying aliases must issue
NO CUDA allocation/free. Existing local wrapper scopes and swaps remain; the last
owner releases the backing exactly once. Cost source/output alias intentionally;
the workspace never aliases either. A distinct compact destination is used for
main's sel->prev rotation.

Reserve metadata wrappers before mutating device data. Shared ownership keeps
source storage alive through failures. GPU errors terminate the problem via the
existing error policy; a partially executed rotation is never retried or exported
as valid data. No promise of rollback after a GPU failure.

## 5. Cost rotation algorithm

Allocate the single compact float workspace once per problem during workspace
initialization, not inside each rotation. Thus repeated rotations add no allocator
calls or allocator-induced synchronization. Include this persistent plane in all
memory bounds, including the untouched buffer rotations.

Within the original cost rotation block:

1. Create a swapped-shape metadata wrapper sharing cost's backing.
2. Create a swapped-shape workspace view sharing the workspace's backing.
3. For each layer s in original ascending order:
   - Invoke unchanged CudaRotate on the original layer into disjoint workspace.
   - Enqueue cudaMemcpyAsync of exactly plane_bytes from workspace to the same
     layer's backing interval, D2D on the SAME implicit default stream used by
     CudaRotate. Both buffers are compact, so a linear copy suffices.
4. Execute exactly the existing final CUDA_SYNC_AND_CHECK counterpart once.
5. Perform the existing pointer swap. Old wrappers leave scope at the same point;
   backing remains owned by the new wrapper.

Kernel and copy stream order prevents workspace overwrite before consumption and
prevents the next operation observing a partly copied layer. Distinct layers have
non-overlapping backing intervals, so ascending order is safe. Never call the
current rotation kernel directly on overlapping input/output pointers.

The existing ordinary `GpuMat::Rotate` implementation remains unchanged. A new
explicit reuse method implements this sequence and replaces only the opted-in
cost call. The host planner/executor seam must be consumed by this method, not
merely duplicate it in tests.

## 6. Main probability path and scratch proof

Keep the main operation order:

1. Reorient the existing previous-probability backing to the new shape using a
   metadata wrapper, instead of allocating a full replacement backing.
2. Run existing sel_prob_map_->Rotate(prev_sel_prob_map_.get()). Source/destination
   backings remain DISTINCT. This is the unmodified out-of-place rotation method.
3. Reorient sel's backing as next-sweep scratch instead of allocating a new backing.

Completed sel probabilities are retained in prev; no persistent recurrence is
removed. Rotation writes every valid destination element; old prev contents are
no longer needed once the preceding synchronized sweep has finished.

Source proof at main998ff4de, `patch_match_cuda.cu`:
- SweepFromTopToBottom lines 949–963: each valid column thread iterates ALL source
  layers and ALL rows backwards, writing beta to every valid sel element before
  reaching the forward sweep. Invalid column threads skip sel accesses.
- Lines 1047 and 1174: beta reads occur after that same thread's backward writes.
- Lines 1179 onward: that row/layer's final probability is written before filter
  reads (1208/1225). No neighboring-column thread reads this scratch.
- No later consumer reads sel after its probabilities have been copied into prev
  and the scratch wrapper is reoriented. GetSelProbMap exports prev (line 1337).

Consequently scratch's prior bytes are irrelevant. Its reuse needs no zeroing and
must not add initialization. Recheck this proof against the exact execution tree;
new/changed scratch consumers invalidate the proof and block implementation.
Pin it with a host read-before-write model for valid/invalid columns, all filter
specializations, odd dimensions and multiple sweeps using poisoned prior bytes.
The model documents ordering, not CUDA numerical equivalence; full632 is mandatory.

## 7. Remaining memory bound

Use an event/lifetime model, not only `steady + one plane`. Include physical known
footprints for pitched buffers and logical sizes separately for CUDA arrays:

- compact cost, sel and prev backings: three distinct volumes on main;
- persistent one-plane workspace; no double-counting metadata aliases;
- unchanged PRNG old/new overlap (depth1, ~0.388 GiB in the sample);
- unchanged depth old/new and three-channel normal old/new overlap;
- unchanged ref image, sum, squared-sum overlap PLUS old/new ref texture overlap
  during rebinding, and filter-input texture during initialization;
- source image/depth textures, four pose textures, global workspace;
- lazy consistency-mask allocation, its persistence during the final rotation,
  and its old/new rotation overlap, conditional on filtering;
- unchanged initialization and teardown timelines; allocator/context/module/
  descriptor overhead is not inferred from logical payload totals.

The filter mask matters: at W*H~8.69M and S=61 a uint8 mask is ~0.49 GiB, and two
can coexist during its unchanged rotation. The earlier 5.13-GiB steady breakdown
is not proof that this buffer can be omitted, particularly in the geometric phase.
B1 does not bound the whole remaining transient to a single float plane.

An upper-bound host ledger must follow these scopes, with S=1/61/64, both phases,
filter on/off, both orientations and zero placeholder masks. Measured device peaks
remain the acceptance criterion; no host model establishes an N4/N2 fit.

## 8. Files and host-testable interfaces

Project snake_case names are used (`rotation_storage_plan.h`, not hyphenated).

- NEW `src/colmap/mvs/rotation_storage_plan.h`: checked compact layout, shared
  backing vocabulary, ascending layer schedule and host memory-accounting seam.
- NEW `src/colmap/mvs/rotation_storage_plan_test.cc`: standalone-capable CPU tests.
- MODIFY `gpu_mat.h`: compact factories/aliases and explicit reuse rotation; no
  change to ordinary Rotate implementation or GpuMatView device accessors.
- MODIFY `patch_match_cuda.{h,cu}`: three opt-in initial allocations, one persistent
  workspace member/allocation, metadata destination factories and cost reuse call.
- MODIFY `mvs/CMakeLists.txt`: host-test registration/header listing.
- Notes/spec/plan: proof, acceptance and remaining-memory limitations.
- UNCHANGED `cuda_rotate.h`, kernel bodies, public options, trace files and seeds.

## 9. Diagnostic and PR2 follow-ups (separate changes)

Diagnostic adaptation counts allocation BACKINGS, not GpuMat wrappers. Register
one physical/compact allocation ID at allocation, retain it across alias wrappers,
and unregister only at final backing release. Alias/view events must not change
tracked bytes. Include the workspace. Ordinary pitched/array diagnostics retain
their existing accounting. Sampling failures still yield null device samples.

Prepare that overlay on a separate diagnostic/integration branch after product
interfaces stabilize, with separate hashes and review; never merge it into B1.
The same separation applies to the PR2 probability adaptation. Stop rather than
blindly applying the old instrumentation or resolving integration conflicts by
mixing main and PR2 semantics.

## 10. Review outcome required

Approve the written spec and companion implementation plan before source work.
Execution remains direct by this agent; no delegation, GPU work, rentals or host
changes implied. A draft PR is a review artifact, never an acceptance claim.
