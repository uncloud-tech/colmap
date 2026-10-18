# Explicit optimized dense dispatch (experimental)

This integration is review code, not a qualified release or a performance/default
promotion. Ordinary stereo retains its historical launch-index PRNG behavior;
coordinate-based nonnegative seeds belong only to the explicit owned backend.
The owned PRNG separation and combined build identity require their own CPU,
CUDA and device validation; results from other builds do not transfer implicitly.

Legacy commands default to `--optimized_only false` and retain their existing
behavior. Strict execution is selected explicitly, not automatically from GPU
count, source count, or a claim of universal fastest performance. CUDA correctness,
quality and workload/hardware performance qualification remain separate gates.
Unknown qualification is **unknown**, not permission supplied by this interface.

Both lookahead modes remain genuine optimized owned/preallocated-rotation paths;
cache OFF and ordinary fallback are not accepted by strict dispatch. Neither
lookahead mode is universally fastest; select using qualified workload evidence.

## Stereo command

Both phases use `patch_match_stereo` with:

```
--optimized_only true --optimized_dispatch_receipt NEW.json
--owned_tile_threads 32 # 16 is also supported
--PatchMatchStereo.owned_distinct_source_cache true
--PatchMatchStereo.owned_tex_lookahead false # true is also supported
--PatchMatchStereo.window_radius 5 --PatchMatchStereo.window_step 1
--PatchMatchStereo.num_samples 15
# Filter is phase-specific (below).
--PatchMatchStereo.write_consistency_graph false
--PatchMatchStereo.allow_missing_files false
--PatchMatchStereo.max_image_size -1
--PatchMatchStereo.num_threads 1 --PatchMatchStereo.gpu_index 0
--owned_run_host_budget_bytes HOST --owned_run_device_budget_bytes DEVICE
--owned_run_max_concurrent_solvers 1
--owned_run_timeout_seconds POSITIVE_SECONDS
```

The explicit positive timeout is one absolute run budget, shared across all
solvers/cohorts and checked before receipt publication; cancellation still drains
owners before exit. It is never reset per solver/cohort.

Explicit positive thread counts other than 1 are permitted, subject to the
caller's CPU admission. Samples 1..15 are supported; S8/S20 are **source counts**,
not sample counts. Source selection, effective native source order, seed, quality
options, floating-point arithmetic and PRNG initialization are not overridden.
Unsupported/contradictory options refuse before a solver. No ordinary fallback.
Only COLMAP workspaces, native dense resolution, CUDA (not HIP) are supported.
Project-file overrides cannot be combined with these modes.

* GEO: `--PatchMatchStereo.geom_consistency true --PatchMatchStereo.filter true`,
  plus exactly one existing
  `--owned_run_forecast FILE` or `--owned_run_cohort_forecast FILE`. Existing
  SHA binding, qualification, allocation-fingerprint, resource minima, input
  identities, bounded-normal ownership, cancellation/drain and sequential
  cohort gates are unchanged. Tile selection is forwarded into the actual
  owners, rather than reporting requested cache flags at tile32.
* PHOTO: `--PatchMatchStereo.geom_consistency false --PatchMatchStereo.filter false`, plus
  `--owned_photo_forecast FILE --config_path SINGLE_REFERENCE.cfg`.
  One explicit reference per invocation; callers schedule the selected references
  without changing their native source vectors. The existing owned CUDA PHOTO
  specialization and preallocated alternate rotation storage are used through
  `RunOwnedPairJob`, the single-permit dynamic `PairCoordinator` and postjoin
  retirement. This is optimized **lifecycle/storage**, retaining ordinary inner
  arithmetic; cache/lookahead select the compiled tile16 or tile32 specialization.
  This is not evidence of faster PHOTO or byte-exactness on any untested hardware.

### Read-only planned native graph

```
optimized_dense_plan --workspace_path WORK --source_cap 8|20 \
  --num_samples 1..15 --gpu_index 0 --output_path NEW.json \
  [--cache_size_gb POSITIVE_FINITE_GB] [--random_seed -1_OR_NONNEGATIVE_INT] \
  [--owned_tex_lookahead true|false]
```

This metadata-only command uses native workspace/config/selection/depth resolution,
encoded RGB header checks, and hashes of config, sparse model and encoded RGB
bytes. It does not decode pixels/maps, create a GPU context, solve, write maps,
or license execution. The existing output parent must exist; publication is
atomic and never replaces an existing file/symlink. Inputs/output parents must
remain immutable during the query. Currently successful export requires a Linux
CUDA-MVS (not HIP), SHA256-enabled, revision-bearing build; a CPU build refuses
with actionable build prerequisites and no ordinary fallback. A CUDA-linked query
itself requires no device/context.

Schema v1 and argv are documented in `src/colmap/exe/optimized_dense_plan.h`.
`phases.photo.requested_options` and `phases.geo.requested_options` contain raw
invocation facts, retaining automatic depth/sigma values (`-1`), full resolution,
cache=true, selected lookahead=false|true, and PHOTO filter=false / GEO filter=true.
The intended specializations support tiles16/32; samples 1..15 are separate from source cap8/20.
Optional lookahead (default true for metadata compatibility) is raw launch policy
forwarded unchanged into both phases' requested and effective job options.
Optional cache/seed are raw launch policy: `cache_size_gb` sets
`PatchMatchOptions.cache_size` without rounding, and `random_seed` sets the stereo
seed, not any SfM seed. Omitting them preserves the existing native defaults
(including seed -1). Both phases' requested options and effective job options
expose these values. Automatic raw depth/sigma defaults remain -1; do not feed
resolved job depths back into invocation options. No model coefficients change.

Top-level `dispatch_support` advertises compiled source support, derived from
the same cache capability helper as CUDA dispatch, or is null if unsupported:

```json
{"backend":"owned_cuda","tiles":[16,32],"num_samples_min":1,"num_samples_max":15,"cache":true,"lookahead":true,"lookahead_modes":[false,true],"lifetime":"preallocated_rotation","status":"compiled_planned_not_qualified"}
```

`lookahead:true` here is capability, not the selected path; `lookahead_modes`
lists both supported modes. This is planned capability metadata, not executed
dispatch or hardware/quality/performance qualification. It must not be treated as proof of fastest execution.

Every selector must be `__auto__, cap` matching the declared cap. Actual source
vectors can be shorter; the command never truncates them. Explicit lists and
`__all__` refuse because their configured maximum is unknown.

Job `options`, `depth_min` and `depth_max` are effective/resolved facts.
`effective_options_wire_v1` is **informational**, not a native launch binding.
Never put resolved historical depths/sigma into argv: actual per-reference PHOTO
options export with the exact invocation options/config path remains the binding
authority. Preserve the original auto selector in singleton reference configs;
rehashing a serialized effective vector is not equivalent to selecting it once.
PHOTO wireV2 must verify the actual exact source order.

Histories explicitly identify fresh-controller PHOTO and fresh-controller GEO
after PHOTO, not stock Run's twice-transformed source history. Future PHOTO map
dependencies bind producer reference/kind/shape/output path, with no fabricated
hash. After the complete PHOTO barrier, the actual native GEO bank contract and
forecast must revalidate exact order and immutable map inputs. The existing
geom-only complete-bank gates are not weakened. Qualification tables remain
empty until real correctness/quality/comparative evidence exists; this graph is
not evidence of fit, eligibility, fastest execution, or automatic dense approval.

### PHOTO reviewed forecast wire v2

UTF-8/ASCII, every line LF-terminated, no trailing records:

```
COLMAP_OPTIMIZED_PHOTO_V2
OPTIONS_SHA256
PROCESS_HOST PROCESS_DEVICE PRIVATE_HOST PRIVATE_DEVICE
REFERENCE_GLOBAL_INDEX SOURCE_COUNT SOURCE_GLOBAL_INDEX...
INPUT_COUNT
FILE_SHA256 HEX_CANONICAL_ABSOLUTE_PATH
... exactly INPUT_COUNT file records ...
```

All four charges, source count and input count are positive canonical decimal integers;
indices are canonical nonnegative integers fitting native `int`. The reference
must not occur among sources and sources must be distinct. Hashes
are 64 lowercase hexadecimal characters, paths are lowercase hexadecimal encodings
of canonical absolute UTF-8 path bytes. Duplicate paths and embedded NUL refuse.

To obtain the exact option-binding bytes, call the same stereo command with the
same workspace, config and PatchMatchStereo options, but only
`--owned_photo_options_export NEW.options` (no strict execution, receipt,
forecast, or budget flags). This is metadata-only, no solver/device/payload.
SHA256 those exact exported bytes for OPTIONS_SHA256. They use a dedicated
`COLMAP_OPTIMIZED_PHOTO_OPTIONS_V1` options-only format: CUDA/HIP build booleans,
all 28 invocation options in fixed field order (exact double bits, decimal
integers/booleans, length/hex strings), then length/hex canonical workspace and
config paths. The complete owned-bank contract still requires captured problems;
it is neither fabricated nor relaxed for PHOTO. No copied defaults are required.

V1 forecasts are rejected. V2 binds the approved reference global model index
and the exact ordered source-index vector to the same immutable config/model
file manifest. Before decoding/solving, the actual production problem must match
both reference and complete source order; same membership in another order
refuses with a reapproval requirement. No sorting is performed. Global indices
are meaningful only alongside the bound model bytes. This options export does
NOT produce an effective graph or approve source order. The graph command above
provides planned native metadata; its native build/command integration and
cross-process reproducibility qualification remain acceptance gates. Do not
guess the native FlatHashSet order or treat a parsed forecast as proof that the
approved graph was produced correctly.

The manifest must contain config, every complete sparse model triplet present
(cameras/images/points3D, `.bin` and/or `.txt`), optional rigs/frames files
when present, and every actually used reference
and source image. Additional immutable files are allowed. All manifest file
bytes are SHA-checked against retained native stat identities; identities are
rechecked before decode, before solver, and after joined completion. The actual
used images are required in the manifest before decode, with native header/model
checks, complete-cache retention, and the existing owned logical workload floor.
Inputs must remain immutable until completion. Native resolution/source order
are never sorted or replaced to satisfy the forecast.

Charges are **reviewed physical bounds**, not measurements or a model generated
by this parser. Process must cover model/metadata/runtime overhead; private must
cover cache, both grayscale snapshots, staging, scratch, pitched rotation storage,
encoded-file hash scratch and overhead. Forecaster applies the single margin and
external admission/headroom controls. Native floors are necessary, not physical
fit certification. Positive integers do not supply missing qualification.

## Fusion command

```
stereo_fusion --workspace_path WORK --output_path OUT.ply
--optimized_only true --optimized_dispatch_receipt NEW.json
--StereoFusion.use_mmap true --StereoFusion.use_cache false
--StereoFusion.num_threads 1 --StereoFusion.max_image_size -1
```

Only native COLMAP mapped input mode with one CPU worker is supported. Existing
mapped RGB, mask, traversal and output logic is reused. External fusion forecasting,
physical admission, immutable inputs and process ownership remain required; this
switch does not grant them. Strict dispatch preflights every configured bitmap,
depth map and normal map before processing any input and refuses a missing
dependency without publishing a receipt. Legacy dispatch retains warn-and-skip.

## Completion receipt v1

Both commands require a new receipt path in an existing directory. Only after
successful work and joined owner teardown is a complete JSON file atomically
published without replacing another file. Refusal/failure creates no success
receipt. Publication failure is an error even if maps/output already completed;
do not automatically retry into those outputs. The receipt is actual dispatch
telemetry, not qualification, fastest-path evidence, or an approval token. Bind
its invocation/output identity through the caller's immutable approved plan.

PHOTO/GEO (`phase` is `photo` or `geo`; `tile` is the actual selected 16 or 32):

```json
{"schema_version":1,"phase":"photo","backend":"owned_cuda","tile":16,"cache":true,"lookahead":true,"lifetime":"preallocated_rotation","fusion_mode":null}
```

Fusion:

```json
{"schema_version":1,"phase":"fusion","backend":"cpu","tile":null,"cache":false,"lookahead":null,"lifetime":"mapped_inputs","fusion_mode":"mmap"}
```

`cache` means distinct-source sweep cache for stereo and LRU input caching for
fusion. `lookahead` means texture lookahead, not the resident normal producer.
Stereo receipts report the actual selected lookahead boolean (false or true),
not a constant true. Cache must remain enabled in either case.
No requested flag is reported as effective unless the supported actual dispatch
has completed. There is no pre-work success receipt.

## Validation gates

Dependency-free CPU seams (from repository root, no CUDA):

```
g++ -std=c++17 -pthread -Isrc src/colmap/mvs/optimized_dispatch_test.cc -o dispatch_test
./dispatch_test
g++ -std=c++17 -DOPTIMIZED_DENSE_PLAN_STANDALONE_TEST -Isrc \
  src/colmap/exe/optimized_dense_plan_io_test.cc -o plan_io_test
./plan_io_test
g++ -std=c++17 -pthread -Isrc src/colmap/mvs/optimized_photo_forecast_test.cc -o photo_forecast_test
./photo_forecast_test
# Also compile/run optimized_photo_options_test.cc, optimized_photo_paths_test.cc,
# and fusion_dependencies_test.cc with the same -std=c++17 -Isrc flags.
```

Also registered as `mvs/optimized_dispatch_test` and
`mvs/optimized_photo_forecast_test` (plus the three additional seams above) in
CTest. `exe/resident_cli_runner_test` includes successful production PHOTO
options export on CUDA/HIP dense-library builds, requiring no GPU execution.
`mvs/fusion_test` exercises the real `COLMAP_BINARY` CLI: missing bitmap/depth/
normal must refuse with no output/receipt, a complete strict invocation succeeds,
and legacy missing-input behavior is preserved. Run resident/cohort suites for GEO.
Before acceptance, build CPU and CUDA configurations and run native tests, then
qualify actual PHOTO/GEO S8 and S20 tile16/32 outputs against equivalent ordered
inputs/options/PRNG controls, sample-limit/unsupported/cancellation cases, mapped
fusion outputs, physical peaks and unprofiled performance. CPU contract tests
alone do not establish those GPU/quality/performance results.
