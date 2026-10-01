## Status and dependency

**Draft, fork-internal review only. Not an upstream submission or merge request.**

This PR is explicitly stacked on **#2, `feature/beta-checkpoint-tile` at `78f41b8c6cb2629775115ee2dc3b50f21a51c4f1`**. Its diff adds runtime-specialized tile dispatch on top of the beta-checkpoint implementation; it does not reintroduce beta checkpoints as unrelated changes. Review/merge the dependency first, then retarget/rebase this PR deliberately. PRs #1 and #2 are unchanged. This does not stack or merge the independent static-tiling PR #1.

### Provenance / exact-tree validation caveat

The GPU-tested implementation was developed on COLMAP4.2.0 (`be5e291`) with the preserved beta-split source. The fork's #2 is based on a newer tree. Its baseline `patch_match_cuda.cu` differs from the tested beta baseline only in the license header. After applying this scoped port, the CUDA implementation/header, options header, selector and host test match the tested versions after license headers. The options-validation changes were applied without reverting the newer fork's existing validation logic.

**The exact PR tree has passed the standalone CPU tests below, but has NOT been fully rebuilt or GPU-tested.** All GPU numbers below are evidence from the preserved implementation checkout/binary, not a claim that this new commit was executed. This remains a draft pending integration validation. No additional GPU campaign was run to prepare the PR.

## Design

One binary computes a tile at runtime, then dispatches once per problem to **compile-time-specialized initial-cost and sweep kernels** for8/16/32 columns per block.

- `RefLayout<B>` supplies producer indexing, consumer indexing and physical shared allocation pitch (`3*B`), avoiding the earlier generic padded-layout mismatch.
- Tile selection is cached after reference dimensions are available; validation, initial-cost launch and all rotated sweep launches use the same choice.
- `if constexpr` excludes halo-incompatible instantiations; a static assertion checks the layout, and a host guard checks cached tile equals specialization.
- Auto promotes its preferred tile to the smallest tile covering `window_radius`. Invalid forced tile/radius combinations are rejected rather than silently changed.
- The standalone, CUDA-independent selector remains testable without a device. Auto queries runtime SM count, uses the max image dimension and a K=2 coverage prior, and falls back to32 if SM count is unavailable. K=2 is **not universally calibrated**.
- Element-wise launch geometry and beta checkpoints are unchanged. No per-card specification table or linker fallback.

### CLI: unchanged default

`--PatchMatchStereo.sweep_tile`:

|Value|Behavior|
|---|---|
|32|Default, fixed32|
|16 or8|Forced specialization; must cover the halo|
|0|**Experimental auto**, with halo promotion|

Help text explicitly labels auto experimental. This PR does not flip the default.

## Correctness evidence

### Standalone CPU tests

Optimized (`-O2 -Wall -Wextra -Werror`) and ASan+UBSan builds each passed **131,272,128 pixel-identity checks**, both on the implementation checkout and again on the PR worktree.

Tests exercise production selection/address helpers across packed specializations and the legacy padded layout: B8/16/32, every halo-valid radius1..B, steps1/2, partial blocks, initialization, rolling rows, center lookup and patch traversal. Both directions of pitch-mismatch mutants are rejected. Selector tests cover36/82/170 SMs, swapped dimensions, threshold boundaries, invalid inputs, overflow-safe target arithmetic and promotion.

This is a host layout model, **not execution of the CUDA copy loop**. The test is registered with `COLMAP_ADD_TEST`, but only standalone builds ran; the benchmark build had `TESTS_ENABLED=OFF`.

### Existing GPU campaign

RTX5090,170 SMs, driver570.195.03,11 CPU threads, one watched worker at a time, explicit private EMPTY no-MPS pipe, warmups, fresh deterministic initialization and fresh outputs. **27 successful legs,54 maps retained and rehashed**; each candidate matched its fresh per-case beta-split baseline.

|Case|Modes exercised|
|---|---|
|max4000, actual3930×2210; radius5, step1, filterfalse|default32, forced16/8, auto→8; repeated32|
|max257, actual257×145; radius5, step2, filtertrue|forced8/16/32|
|max129; radius8, step2|forced8 at halo boundary|
|max129; radius9, step2|auto→16|
|max129; radius17, step2|auto→32|
|max65; radius20, step1|auto→32|
|max33; radius1, step1|forced8, partial blocks|
|max65; radius5, step1|forced8 under CUDA memcheck: **0 errors**|

Each leg runs20 sweeps, exercising original and rotated orientations with a cached tile. Warmups use max1000/radius5/step1. CLI invalid-tile and insufficient-forced-halo checks fail before GPU execution.

Full-size inventory digest: `52e4d309d9473f41c8af3882c4a116a8e22aa7e829acd98dc07ca554178fa5b7`. Definition: enumerate depth/normal `*.photometric.bin` files under each workspace's `stereo/`, sort by relative path, hash each file to SHA256 hex text, then SHA256 the newline-terminated list of those strings. Paths are not included in the outer payload. Full-size is7 legs, not8. The retained local files, not deleted historical3090 maps, support this PR's equivalence evidence. No independently reverified cross-card or unmodified-upstream claim.

## Performance: matched beta-split comparison

Sum of20 sweep timers, **including rotation**, not isolated kernel timing or end-to-end throughput:

|Configuration|First chain|Confirmation|
|---|---:|---:|
|Fresh rebuilt beta-split baseline32|34.2705s|34.2691s|
|Specialized default32|34.2381s|34.2399s|
|Specialized forced16|31.9706s|—|
|Specialized forced8|28.3933s|—|
|Auto→specialized8|28.3803s|—|

Tile32 is a practical tie (median−0.090%). Auto8 is **17.186% shorter** than the matched beta-split baseline. First process walls: baseline43.746s, specialized32 44.138s, auto37.967s. Only32 has two measurements; this is screening/validation, not a broad performance distribution.

**Do not conflate comparisons:** historical static tile timings (including ~17.6% improvement) used a **non-split** implementation on5090. These results use beta split. Matching baseline at tile32 does **not** prove zero dispatch/instantiation overhead at tile8. The earlier ~4–5% gap between split-based and non-split figures is **not a controlled A/B attribution to beta split**. No matched static-beta-split tile8 versus runtime-specialized tile8 overhead experiment was performed.

The rejected generic kernel measured39.0944/39.1001s at32 versus the old baseline34.2749/34.2300s. A fresh baseline rebuild under the identical executed kernel command measured34.2286s, ruling out the tested accidental-debug/build-configuration explanation. Specialization removes that observed regression; exact instruction/register/spill mechanisms remain unprofiled.

## Costs and build configuration

- Same Release kernel command: `-O3 -DNDEBUG`, `--use_fast_math`, `--default-stream per-thread`; no `-G`, `-lineinfo` or `-O0`.
- Global arch120 has an existing MVS **compute90 PTX-only override**. The preserved benchmark binary is **not usable on sm86/3090**.
- Incremental build144.42s; CUDA object122.858s versus preceding baseline56.448s (~2.18×, not a clean whole-project build comparison).
- Binary54,230,888 bytes versus generic45,778,776 bytes (~18.46% larger).
- 352 sweep and88 initial-cost instantiations after pruning invalid halo combinations.
- First discarded specialized warmup137.33s wall versus the earlier generic73.55s: known cold setup/JIT cost, single observations, not a controlled cold-cache benchmark. **Cold-start/on-demand-loading work is parked and not part of this PR.**

Benchmark binary SHA256: `cd4d3d4e162ba42e9f129739e8383d8ca8e42996b8093b57b626000e2644cc5b` (implementation checkout, not this ported commit).

## Outstanding gates / limitations

- [ ] Full build and integration validation of this exact newer-fork PR tree.
- [ ] `geom_consistency=true` runtime coverage before relying on that product path.
- [ ] Other GPUs/architectures, HIP and MPS concurrency validation as relevant.
- [ ] Full CTest integration; only standalone optimized/sanitized tests executed.
- [ ] Calibration evidence before treating max-axis K=2 as universal or changing default32.
- [ ] Broader performance distribution / matched static-beta tile8 control if claiming zero tile8 overhead.

The existing runtime radius support1..20 is unchanged (host layout tests extend through32). One small memcheck case is not exhaustive race/synchronization proof. No clang-format executable was available for a formatting pass. No upstream submission or merge is requested.

## Evidence retention

All maps/logs remain on the GPU host at `/home/spheron/specialized-dispatch/{verification,confirmation}/`; summary, source snapshot, executed compile command and per-file hashes are retained there. A/A evidence is in `/home/spheron/runtime-dispatch-implementation/aa-control/`. PR-preparation CPU logs and branch records are in `/home/spheron/specialized-dispatch/pr/`. No outputs were cleaned up, and no new rental, GPU campaign or cold-start experiment was performed for this PR.
