# PatchMatch B1 Rotation Lifetime Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task, directly in the owning session. No delegation is authorized. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove S-volume replacement allocations from main's cost/probability rotations while preserving stored bits, numerical kernels and existing kernel order.

**Architecture:** Opt-in compact backings have orientation-invariant layer extents. Metadata aliases retain backing ownership; cost rotation uses a disjoint persistent one-plane workspace and interleaved same-stream copies. Main's probability source/destination remain distinct. Diagnostics count physical backings on a separate branch.

**Tech Stack:** C++17, existing CUDA/HIP runtime abstraction, unchanged CudaRotate kernels, standalone CPU tests/GTest, existing CMake.

**Spec:** `docs/superpowers/specs/2026-10-04-patch-match-rotate-lifetime.md`

**Status:** Spec and plan approved by photogram. Direct execution authorized, with a report at every task boundary and after Task 3. PR2 adaptation is the immediate separately reviewed follow-on after the product PR lands; memory-goal measurements belong to that benchmark lineage.

## Global Constraints

- Product base: `998ff4de70344c0fcc5f10bf36f71924026dba95`; branch `feature/patch-match-rotate-lifetime`; draft PR to fork main.
- Validation base: exact `d7ffb9c4bd18bc9285256a91dbe3ce98a25c6248`, identical external minimal seed patch in both arms; full158/632 digests, from-disk, random_seed=0, tile32.
- Preserve CudaRotateKernel/GpuMatView/numerical kernel bodies and ascending per-layer kernel order. Additional same-stream D2D copies are allowed.
- Keep the existing final synchronization; add no explicit synchronization, streams, events or worker/device concurrency. Allocate workspace once per problem, not per rotation.
- Default non-opted-in CUDA behavior remains unchanged. No CLI/default/precision/quality/calibration changes, recomputation or PR2 reimplementation.
- Trace hooks and benchmark seed remain OUT of the product PR. PR2 probability adaptation needs separate review and is not executed by this plan.
- S1 remains HOLD. No GPU work, host changes or rentals by this agent. Photogram owns compile/GPU acceptance; local host tests are not CUDA proof.
- All remaining transients are modeled; actual device peak and no throughput regression are required. No N4/N2 capacity promise.

## Review Focus

1. Odd/asymmetric dimensions and repeated orientation changes: no pitch-derived slice overlap or capacity overrun (Task 1).
2. Alias lifetime and failed construction: backing stays alive, freed once; overlapping workspace rejected before submission (Tasks 1–2).
3. Stale scratch contents and padded invalid columns: no scratch read precedes its same-column write, with filter branches covered (Task 3).
4. Late filter mask, reference rebinding and PRNG overlap: whole-problem peak does not quietly become `steady + one plane` (Task 3).
5. Validation/instrumentation lineage: main versus PR2 semantics, metadata aliases versus physical allocations, and exact control/candidate seeds stay distinguishable (Tasks 4–5).

---

## Task 1: Checked compact layout, ownership and scheduling seam

**Files:** NEW `src/colmap/mvs/rotation_storage_plan.h`, NEW `src/colmap/mvs/rotation_storage_plan_test.cc`; MODIFY `src/colmap/mvs/CMakeLists.txt`.

**Interfaces produced** (namespace `colmap::mvs::rotation_storage`):

- `struct RotationStoragePlan`: size_t fields `width`, `height`, `depth`, `element_bytes`, `source_pitch`, `rotated_pitch`, `plane_bytes`, `backing_bytes`.
- `std::optional<RotationStoragePlan> MakePlan(size_t width, size_t height, size_t depth, size_t element_bytes) noexcept`.
- `struct RotationBacking { std::shared_ptr<void> allocation; size_t capacity_bytes = 0; };`
- `struct RotationStorageView { RotationStoragePlan layout; RotationBacking backing; };`
- `std::optional<RotationStorageView> MakeView(const RotationBacking&, size_t width, size_t height, size_t depth, size_t element_bytes) noexcept`: rejects null backing, invalid layout or capacity mismatch.
- `std::optional<RotationStorageView> ReorientView(const RotationStorageView&, size_t width, size_t height) noexcept`: permits only current/swapped W/H, retains depth/type/capacity/backing ownership.
- `bool ValidateReuse(const RotationStorageView& source, const RotationStorageView& destination, const RotationStorageView& workspace) noexcept`: validates rotated destination, same source/destination ownership/address, exact one-plane workspace capacity/shape, disjoint workspace ownership/address interval. Reject pointer-range overflow.
- `template<class Rotate, class Copy> void ForEachLayer(const RotationStoragePlan&, Rotate&&, Copy&&)`: for s=0..depth-1, invoke rotate(s, s*plane_bytes), then copy(s, s*plane_bytes). No allocation, error swallowing or synchronization.

This header contains no CUDA headers/calls or production telemetry. Use std::shared_ptr's ownership comparison, not pointer equality alone, when validating shared ownership. Keep the full lifetime/peak simulator in tests, not production.

- [ ] After approval, read using-git-worktrees guidance, verify clean tracked trees/base SHA and create an isolated product worktree. Preserve existing diagnostic branch and untracked HOLD docs. Copy only these approved B1 docs into the product tree.
- [ ] Write failing CPU tests: W=3,H=2,S=61,E=4 gives pitches12/8, plane24, capacity1464; swapped view preserves capacity and owner; 1xN/Nx1/square/odd rectangles; S1/61/64; reject zero dimensions, size_t overflow and width/height/pitch beyond INT_MAX.
- [ ] Write ownership tests using fake owned buffers: alias deletion changes no allocation count; final release fires exactly once; failed view creation leaves original owner valid; distinct control blocks sharing an address do not qualify as the same owner; overlapping/undersized workspace fails before any operation.
- [ ] Write schedule/bit tests against a separate CPU coordinate oracle: explicit 3x2 fixture plus arbitrary uint32 payloads (NaN payloads, infinities, signed zero included without FP arithmetic), ascending rotate/copy pairs, four rotations restore every valid bit, guard bytes unchanged. Inject callback failure and verify no later callback executes; do not claim rollback.
- [ ] Run the standalone test before implementation; record the expected failing assertions/missing seam. Implement the interfaces and make the real schedule drive the CPU simulation.
- [ ] Register GTest and a standalone mode whose checks survive -DNDEBUG. Run `g++ -std=c++17 -O2 -DNDEBUG -Wall -Wextra -Werror -pthread -DCOLMAP_ROTATION_STORAGE_STANDALONE -Isrc src/colmap/mvs/rotation_storage_plan_test.cc -o <evidence>/rotation-storage-test` and execute it. Repeat with ASan+UBSan. Record actual results; never infer a CUDA build from them.
- [ ] Commit the independently host-tested seam and test registration.

## Task 2: Opt-in GpuMat backing and reuse adapter

**File:** MODIFY `src/colmap/mvs/gpu_mat.h` only.

**Consumes:** Task 1 layout/view/ownership/scheduling interfaces.

**Produces:** internal GpuMat methods:

- `static std::unique_ptr<GpuMat<T>> CreateCompact(size_t width, size_t height, size_t depth = 1)`.
- `std::unique_ptr<GpuMat<T>> MakeCompactView(size_t width, size_t height) const` (same/swapped shape, same depth).
- `void RotateWithStorageReuse(GpuMat<T>* output, GpuMat<T>* workspace)` in the existing CUDA/HIP compilation guard.

Private compact ownership/view state is empty for ordinary matrices. Keep existing raw pointer, pitch and const shape fields as the public/view representation; add a private constructor for valid compact views. Default constructor and ordinary Rotate body retain their original operations. Compact destruction releases the shared owner instead of calling cudaFree on every alias; ordinary destruction retains its original free/error-discard policy.

- [ ] Extend seam tests with the exact source/destination/workspace validation used by the adapter; require rejection before the first fake kernel/copy for each invalid arrangement.
- [ ] Implement CreateCompact: plan validation before CUDA calls, one cudaMalloc for backing_bytes, exception-safe shared deleter, then a metadata wrapper. Host allocation failure after cudaMalloc must not leak; factory construction/cleanup must have one owner at every point.
- [ ] Implement MakeCompactView using ReorientView; no CUDA calls and no mutation of existing metadata. All metadata allocations happen before rotation submission.
- [ ] Implement RotateWithStorageReuse using ValidateReuse and ForEachLayer. Rotate callback calls existing CudaRotate with original W/H/source pitch and workspace rotated pitch. Copy callback uses cudaMemcpyAsync of plane_bytes to that same layer's compact destination; omit explicit stream overrides so both operations use the existing implicit/default stream under the pinned --default-stream per-thread build. Keep exactly one final CUDA_SYNC_AND_CHECK, replacing rather than supplementing the ordinary cost rotation's final check.
- [ ] Leave cuda_rotate.h, GpuMatView accessors and ordinary GpuMat::Rotate byte-for-byte unchanged. Add no allocation/free inside the reuse method. Confirm existing cudaMemcpyAsync HIP mapping avoids another abstraction-file change.
- [ ] Run host seam suite and diff/operation audit. CUDA compilation is PENDING on the coordinator host; mark it so. Commit source-ready adapter, not a GPU-validated claim.

## Task 3: Main-only integration and complete lifetime model

**Files:** MODIFY `src/colmap/mvs/patch_match_cuda.h`, `patch_match_cuda.cu`, `rotation_storage_plan_test.cc`; update B1 notes.

**Consumes:** CreateCompact, MakeCompactView, RotateWithStorageReuse.

**Produces:** compact cost/sel/prev initialization, one per-instance `std::unique_ptr<GpuMat<float>> rotation_workspace_`, main-specific reuse operations and a documented whole-problem host memory bound.

- [ ] Add failing read/write-model tests for main's scratch proof: poison all prior bytes; for every valid column/layer write all backward messages before forward reads; final-probability writes precede filter reads; invalid padded columns never access scratch. Cover all photo/geom filtering combinations, odd dimensions, several sweeps and both orientations. Compare poison seeds to ensure prior contents are irrelevant. Keep numeric likelihood code OUT of this model.
- [ ] Recheck every sel/prev read/write/export against exact main source. Record backward-write lines949–963, beta reads1047/1174, probability write1179, filter reads1208/1225 and prev export1337 (base line numbers). If a read-before-write or additional consumer appears, STOP rather than add initialization/recompute.
- [ ] Change only the initial cost/sel/prev allocations to CreateCompact. Add one compact float workspace W*H*1 during workspace initialization. Preserve prev's existing FillWithScalar(0.5f) and all other numerical initialization calls/order.
- [ ] In main's probability section, replace new-prev allocation with prev->MakeCompactView(newW,newH), keep existing sel->Rotate(prev) call, replace new-sel allocation with sel->MakeCompactView(newW,newH). Require distinct physical source/destination backings. Preserve existing statement order.
- [ ] In the existing cost block, create a cost alias and correctly oriented workspace alias before work; call RotateWithStorageReuse, then perform the existing cost swap. Do not move the outer scope or any other rotation/calibration operation.
- [ ] Add failing event-ledger tests for both phases/filter modes and S1/61/64: each compact backing counted once across aliases; workspace persists; probability path has no replacement backing; cost has no S-plane replacement; include PRNG, depth, normals, ref buffers + rebinding textures, source textures, poses/global workspace, lazy mask and mask rotation. Test mask allocation before the final sweep and its lifetime THROUGH cost rotation. Account initialization/filter-input texture and teardown as well.
- [ ] Make the model pass from the actual specified scopes; expose per-event known physical/logical totals rather than asserting an invented fit. Include a case where mask rotation, not cost rotation, becomes the peak. Do not assign exact driver bytes to CUDA arrays.
- [ ] Run host tests/sanitizers, compare unchanged kernel bodies and ordinary paths to main, audit launch/copy sequence and synchronization counts. Commit integration and its proof/model. Full COLMAP/CUDA build remains externally pending.

## Task 4: Gateable product handoff and draft PR

**Files:** B1 proof/acceptance notes and source commits from Tasks1–3; no trace/seed additions.

- [ ] Review full diff versus main998ff4de for only scoped paths. Record product base/head/tree, patch SHA256, exact source/kernel hashes and all available host test evidence. Check preserved source/image/layer order and all untouched transient buffers.
- [ ] Prepare an isolated exact-upstream validation tree; replay only topic commits, STOP on conflicts, then apply the verified minimal seed patch outside the product branch. Verify reference/seed patch hashes and make control/candidate provenance explicit. Never edit coordinator-owned photogram files.
- [ ] Supply reproducible patches/build requests to photogram. Full CMake/CTest/CUDA builds and GPU execution are theirs; report unavailable local tools as unverified, not passed. Request consistent compiler flags including --use_fast_math and --default-stream per-thread and identical input/seed/invocation mode in both arms.
- [ ] Open the authorized DRAFT topic PR to fork main, preserving existing PR1/2/3. PR body labels source-ready/host-tested versus pending full632, measured device peak and rotation-heavy throughput; includes remaining-transient bounds and main-versus-PR2 distinction. Do not merge or mark accepted.
- [ ] On coordinator evidence, report exact full632 result, both phase/filter settings, S distribution including64, measured peak/time methodology and remaining overhead. Any digest mismatch or throughput regression blocks acceptance; no automatic precision/concurrency/lifetime expansion.

## Task 5: Separately reviewable diagnostic overlay (not the product PR)

**Branch/base:** a separate diagnostic branch from finalized B1 product HEAD, with a separate patch/target hash. Do NOT cherry-pick the entire d70 diagnostic patch into main or silently resolve probability-path conflicts.

**Files:** reuse/adapt diagnostic-only `patch_match_memory_trace.{h,_test.cc}` and hooks in GpuMat/texture/ref-image/PatchMatch code and CMake. Product commits do not acquire these files/hooks.

**Interface:** distinguish tagged physical backing identity from metadata view identity. Compact allocation registration happens once at CreateCompact's successful cudaMalloc; final deleter unregisters once after cudaFree. Alias/view events have zero allocation delta. Ordinary pitched/texture accounting and the env gate remain intact.

- [ ] Write failing host accounting tests: two orientation wrappers over one backing count once; destroying either nonfinal alias does not reduce live bytes; destroying the last owner reduces once; workspace counted separately; pointer reuse receives a new allocation ID; callback failure yields invalid/incomplete evidence rather than fabricated bytes.
- [ ] Port only reusable host formatter/tests and implement main/B1-specific probes from the reviewed source. Instrument compact physical allocation/deletion rather than GpuMat wrapper constructors/destructors. Retain per-problem observed device-wide and tracked-logical high-water, four-point rotation markers, error-null samples and completeness checks.
- [ ] Run host/sanitizer/JSON tests and exact-base patch replay; provide separate base/head/patch SHA to photogram for build, full632 OFF/ON and actual peak measurements. Default-disabled hooks add no diagnostic CUDA queries/output/sync.
- [ ] Keep PR2 integration outside this task. Supply an explicit follow-up note: reuse persistent prev via the same workspace on PR2, retain exact PR2 scratch reset/reallocation order, update separate memory model, then require separate review and the same acceptance gates.

## Self-review / handoff

Spec sections1–3 map to Tasks4/global constraints; sections4–5 to Tasks1–2;
section6 to Task3; section7 to Task3 plus coordinator measurements; sections8–9
to Tasks1–5. Review Focus cases have explicit owning tests. No extra numerical
algorithm or public option is introduced. New header uses project snake_case.

Two deliberate details beyond the initial sketch: persistent compact workspace
avoids per-rotation allocator calls, and compact-to-compact copy uses existing
cudaMemcpyAsync rather than cudaMemcpy2DAsync (same-stream D2D boundary unchanged).
The lazy filter mask can dominate the remaining transient and is not omitted.

Return both documents to photogram for review. Await approval before executing
Task1; direct execution stays the selected method. Preserve diagnostic HEAD and
S1 HOLD documents in the meantime.
