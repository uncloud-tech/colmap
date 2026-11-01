# Experimental installed photo-directory reconstruction

CMake installs `colmap-reconstruct` beside `colmap`. It requires Python 3.10+,
Linux, a **non-root user**, and a matching CUDA-enabled COLMAP build with its
runtime libraries available. This is not a standalone static binary. No
pycolmap, source checkout, external Python package or separate resource calculator
is needed. The launcher uses isolated Python and prefix-relative modules.

Supported experimental invocation (no internal byte declarations required):

```sh
colmap-reconstruct --experimental-local --photos /data/photos --workspace /data/new-run
```

Before any reconstruction or requested-workspace creation, this inspects inputs,
queries the selected native GPU/resources and displays the effective pipeline,
resolution semantics, options, output path and conservative policy. On a terminal,
type **yes** to approve; **no**, Enter or EOF refuses without reconstruction.
`--experimental-local` consents only to engineering estimates, **not** to running.

```sh
# Inspect only; never create the requested workspace/DB/sparse/maps:
colmap-reconstruct --experimental-local --plan-only --photos /data/photos --workspace /data/new-run
# Explicit noninteractive plan approval (piped answers are not accepted):
colmap-reconstruct --experimental-local --yes --photos /data/photos --workspace /data/new-run
```

Preview queries use an owned temporary directory under the resolved workspace
parent, cleaned on completion, rejection, failure, timeout or handled cancellation.
The workspace and its parent must be outside the photo tree. Preview does not use
or probe environment-selected, cached or default temporary locations. A CPU-only
build or unavailable CUDA device query reports **plan unavailable**, not a runnable
GPU plan or CPU fallback. Native query diagnostics are printed before temporary logs are removed.

Approval binds executable SHA256, input bytes/inventory/resolved targets and
selected options. Changed inputs/executable refuse before workspace creation;
rerun to review and approve a fresh plan. GPU UUID is checked at existing native
resource boundaries. Inputs must remain immutable throughout execution: this is
not a sandbox or protection against concurrent hostile writes. FREE is a racy
snapshot, not reserved capacity; it is reobserved at native boundaries without
asking again for insignificant fluctuations. Insufficient capacity refuses,
without changing quality or falling back. Runtime and whole-pipeline physical
fit remain **UNKNOWN**. Actual post-SfM geometry, depths, source graph and remaining
native dense options resolve later under the displayed contract, not a fabricated
final dense plan before SfM. Native CPU model/CUDA name/total/SM are not exported
by this query and are explicitly UNKNOWN; selected threads/GPU are not inferred
optimal hardware settings.

This selects `experimental-stereo-v1` conservative allowances for its supported
native allocation binding. This is **not** whole-pipeline physical-fit or
arbitrary-scene qualification. Unsupported/stale bindings explain the refusal;
install a supported native build, or supply all expert allowances explicitly.

Choose a **fresh** workspace under a directory you own. Existing paths (including
symlinks) refuse. Originals are not modified. `--timeout-seconds` covers the entire
invocation; cancellation/timeout terminates and drains owned child sessions.
Failure logs and partial outputs remain for diagnosis, without `result.json`.
Controller SIGKILL cannot be handled: use an external service/cgroup for
controller-death containment or hard memory enforcement. Native children are
trusted not to detach from their process groups.

## Actual path and limits

The entry runs feature extraction, exhaustive matching, mapper, sparse model
selection, undistortion, **experimental_fixed_cohort**, then mapped-input CPU
fusion. Multiple sparse models refuse unless `--sparse-model NAME` or explicit
`--sparse-model largest` is supplied. Largest means most registered images, with
lexicographic-name ties; mapper/model numbering is nondeterministic. At least two
registered images and a nonempty point model are required, not a quality claim.

**Supported-input limitation: CUDA SIFT currently truncates input widths to a
multiple of four. This frontend refuses affected inputs, rather than silently
losing columns.** Effective width is encoded height for EXIF orientations 6/8
(the native gravity rotation), otherwise encoded width. An odd height alone is
not refused. Native support for other effective widths remains a delivery
blocker; this is not general full-resolution photo-directory qualification.

Before extraction, native `photo_directory_summary` recursively inspects headers
in native filesystem-path discovery order. Empty inventories, corrupt or
unknown-dimension photos and malformed inspection results refuse, not skip.
Photo-only directories retain default native discovery and its order. When
ordinary sidecars (`.xmp`, text files, etc.) are present, extraction instead uses
`--image_list_path` containing exactly the inspected photos; sidecars are reported
but never passed to the image decoder. This newly supported case uses the native
image-list **UTF-8 byte string sort**, which can differ from filesystem-path
ordering for nested names (e.g. `a/z.jpg` versus `a.jpg`). The selected inventory,
effective order and excluded sidecars are recorded in
`planning/extraction-inventory.json` and displayed before extraction.

The list is UTF-8, one unchanged relative filename per LF-terminated line.
Internal spaces and Unicode are preserved; `#` is not a comment. For sidecar
filtering, filenames containing LF/NUL, invalid UTF-8, or leading/trailing ASCII
space/tab/CR cannot be represented by the native line reader and explicitly
refuse before extraction; no photo is renamed or silently skipped. Header
inspection is not a decode/fit guarantee; keep inputs immutable throughout the
invocation.

Extraction uses an explicit positive maximum edge covering **all originals**,
not SIFT's implicit 3200 fallback for nonpositive limits. GPU SIFT's normal first
octave upsamples by 2; there is no extraction thumbnail or CPU retry. Unsupported
GPU sizes/allocation failures refuse; there is no reduced retry.

Dense maps remain float32. `--max-image-size N` retains its expert meaning: an
optional edge cap on the **undistorted camera only**, never extraction. Default
`-1` disables this resource resize; it does **not** promise original dimensions.
Pinhole undistortion geometry (`blank_pixels=0`, `min_scale=0.2`, `max_scale=2`)
can change dimensions independently of the cap. Stereo and mapped fusion both
use `max_image_size=-1`, without another resize. Original/working dimensions and
these semantics are displayed before extraction in `planning/resolution-intent.json`;
`planning/photos.json` retains the native inventory. After SfM/undistortion,
`planning/resolution-effective.json` and stderr report actual native dense image
dimensions consumed by PHOTO/GEO and fusion. This later geometric resolution is
anticipated in the approved preview; it is not an unrequested resource resize.

Samples/seed/source cap/cache/lookahead/tile/GPU are explicit CLI options
(`--help`); existing native command defaults are unchanged.

The native typed dense-plan API resolves per-reference depth ranges, effective
options (including PRNG), source order and complete PHOTO producer closure.
The opt-in native `--resolve_workspace_plan 1` adapter consumes that same typed
API directly, without globalizing depths, sorting or reserializing sources through
another controller transformation. Effective lists, including duplicates, are
copied verbatim. All PHOTO jobs finish before GEO. Unsupported camera models,
missing producers/inputs or unsupported options refuse; no stock stereo fallback.
Sparse workspace and input bytes must remain immutable during reconstruction.

Native automatic FREE host/cgroup + selected CUDA-device observations determine
capacities, optionally lowered by `--host-budget-bytes` / `--device-budget-bytes`.
Automatic independent PHOTO/GEO owner counts and private/shared PHOTO storage use
the **current canonical native fixed-cohort inventory** and one aggregate 3%
margin. `--active-owners-ceiling` optionally bounds counts (0 means job count).
Input dimensions, effective source lists/options and PRNG layout feed the native
inventory; actual observed capacities feed admission. No second Python allocation
formula, free-minus-known filler or invented pixel coefficient is used. Counts
are memory-admissible limits, **not measured overlap or fastest settings**.

### Versioned resource policy and expert overrides

`experimental-stereo-v1` supplies the following RAW **conservative policy**
allowances, reproducing one bounded successful nine-photo configuration (explicit
12 GiB host/device caps and 1600-pixel dense resize, not full-resolution or
arbitrary raw-camera qualification):

| Term | Default |
| --- | --- |
| Loader scratch (host) | 512 MiB |
| Process (host / device) | 512 MiB / 512 MiB |
| Unknown additional overhead (host / device) | 1 GiB / 1 GiB |
| Headroom reserve (host / device) | 512 MiB / 512 MiB |

These values were declarations in that execution, **not measured usage or
calibrated bounds**. A new scene or device is not guaranteed to fit. They do not
scale from hardware size or pretend to forecast SfM or complete fusion physical usage. Native input-derived
allocations are predictions; FREE capacity is an observation (racy, not reserved);
these allowances are policy. Already-live usage is already excluded from FREE
and is **not subtracted again**. Process/unknown allowances cover additional
future overhead conservatively; they do not represent current live usage.
Native admission alone applies the single aggregate 3% margin.

Each `--loader-scratch-host-bytes`, `--process-{host,device}-bytes`,
`--unknown-{host,device}-bytes`, and `--reserve-{host,device}-bytes` remains an
optional expert override. Partial overrides preserve defaults for absent terms.
Explicit unknown overrides require nonblank `--unknown-provenance`; both unknown
sides cannot be zero. Explicit zero is not treated as absence. Supplying all
seven allowances preserves the legacy expert path without claiming default-policy
qualification for another binding. Budgets can only lower observed FREE, never
raise it; they are not additional reserves or proof of physical fit.

These are unqualified logical dense estimates, **not hard RSS/VRAM caps**.
SfM/decoding/fusion physical memory models remain unavailable. Before heavy
stages, `reconstruction_info --gpu_index N --deadline_seconds N` reuses the native
FREE host/cgroup and selected-device observer for reserve sanity checks; it is
not a physical stage forecast. Identity-only `reconstruction_info` creates no
CUDA context. Immediately before fusion, the installed frontend automatically hands
observed host capacity (limited by any user cap) to native fusion and the existing
direct-child RSS monitor. No extra byte flags are required. Native fusion again
limits that phase cap by its own observed host/cgroup FREE before allocation.

### Feature extraction admission

Before extraction the installed frontend passes a fresh observed host phase cap
(lowered by a user cap), raw host policy and reserve to native extraction. The
same phase cap scopes the direct-child RSS monitor; the prior monitor limit is
restored on success or failure. Extraction uses the inspected positive original
maximum edge, independently of the expert undistortion cap, with explicit GPU
SIFT and first octave -1. Native compensated maximum-dimension checks reject
backend clamping; CUDA SIFT's automatic memory-fit octave skipping is disabled
by default. Allocation failures propagate through the bounded controller and
drain workers, without a reduced-size or CPU retry. The multiple-of-four
working-width restriction above prevents the backend's separate column
truncation. There is no automatic feature-count change or quality reduction.

Native `FeatureExtraction.host_budget_bytes` defaults to zero, retaining legacy
behavior. Nonzero budgets support grayscale SIFT (CPU, covariant CPU or GPU) with
UINT8, non-deep 1–4-channel images and a build supporting the native host/cgroup
observer. Masks and non-SIFT extractors explicitly refuse in budgeted mode only.
The same open decoder header is admitted before pixels are allocated; dimensions
must fit Bitmap's int arithmetic. A valid one-pixel dimension remains positive
when bounded thumbnailing. The reader releases its database transaction while
waiting, so the writer can retire payloads without a transaction/credit deadlock.

For original pixel area P and actual rounded thumbnail area T, native admission
reserves the maximum of decode/conversion overlap (P for gray, 4P for RGB),
original + thumbnail when resized (P+T), gravity rotation (2T), and CPU grayscale
plus float staging (5T, prepaid rather than charged again). This per-job credit
travels through the existing reader/resizer/extractor/writer queues. It is held
until the job's actual payloads are destroyed, not just until extraction finishes.
All simultaneously queued and executing images therefore share one ledger.

SIFT adds credits before controlled output/scratch allocation using actual
feature counts and requested capacities, including old/new vector growth,
per-level CPU storage, descriptor transforms, GPU float download and database
keypoint serialization. `max_num_features` is **not** treated as a storage bound:
orientations and complete retained scale levels can exceed it. Budgeted GPU
keypoint download storage is job-local and actually freed before credit release;
legacy GPU buffer reuse is unchanged. Actual vector capacities are reconciled
where available. Scratch credits conservatively remain held through writer
retirement even after temporary storage is freed; this can refuse a job that
would fit a tighter lifetime analysis. Only new-job admission waits. Growth while
holding live credits refuses immediately rather than deadlocking all extractors.

Capacity is frozen once from native observed FREE, lowered by the supplied cap;
live allocations are never subtracted again. Native applies exactly one aggregate
ceil(3%) to known live credits plus raw policy, then adds reserve outside it.
Singleton no-fit, unsupported input, extraction/writer failure or cancellation
stops and drains all queues/workers, returning failure. The database can retain
honest partial state; it is not deleted, rolled back wholesale or labelled a
successful frontend reconstruction. No successful `result.json` is published.

These are **logical owned-payload checks, not a physical host/device cap**.
VLFeat pyramids and detector state, SiftGPU host/device workspace, decoder/filter
internals, SQLite page/cache storage, metadata, allocation rounding and library
bookkeeping remain explicit UNKNOWN policy. CPU SIFT workspace can dominate the
known pixels/output by gigabytes. Reusing default policy is not CPU/GPU overhead
calibration or a complete physical-fit claim. Device capability gates and the
existing native backend remain unchanged; real GPU compatibility requires GPU
validation, not these CPU-path tests.

### Exhaustive SIFT matching admission

The installed CUDA frontend passes fresh measured host/device ceilings, lowered
by user caps, and existing automatic policy/reserve terms to `exhaustive_matcher`.
The host monitor is phase-scoped and restored in `finally` on success or failure.
No extra expert declarations are required. Host policy is loader scratch + process
host + unknown host; **device** policy is process device + unknown device, never
host coefficients. These uncalibrated policy allowances carry their existing
per-term provenance in `resource-policy.json`; they are not measured workspaces.

Native `FeatureMatching.host_budget_bytes=0` retains legacy behavior. Nonzero
budgets support exhaustive non-guided/non-rig SIFT with the unchanged default CPU
FAISS or native CUDA GPU backend. Guided/rig/multiple-model/non-SIFT/GLSL/HIP modes
refuse explicitly in budgeted mode. CPU must not specify a device budget; native
CUDA requires one. GPU `-1` retains native all-visible-device selection; ordinals
must otherwise be distinct. Native observation records each selected ordinal and
UUID, constrains each device by its actual FREE, and can lower host FREE after
CUDA context initialization. No threshold, seed, RANSAC option, thread count,
feature count, pair coverage or pair generation/enqueue order is changed. GPU
DB rows exceeding `max_num_matches` refuse rather than truncating features;
there is no automatic CPU fallback.

Inventory uses SQLite rows/columns/type/BLOB length **without decoding BLOBs**:
SIFT descriptors are 128 uint8 bytes per feature; keypoints accept existing 2/4/6
float encodings and convert to the actual native keypoint layout. Inconsistent
shapes/orphan features refuse. Feature shapes must remain immutable; cache loads
recheck them before allocating. The conservative persistent reservation includes
all DB images, read/converted keypoint overlap and descriptor payload, FAISS float
build input/index/ID/centroid payload, and explicit detached cache/index overlap
for worker-held shared objects after LRU eviction. Shared cached data is not
charged independently for every pair. This is **not actual cache residency** and
can over-refuse, especially for unused images. It is not a cache eviction
optimization. Persistent credits retire only after workers, retained CPU indexes,
cache owners and CUDA buffers are actually destroyed.

CUDA uses M = round-up-to-32(min(max_num_matches, actual DB max rows)). Persistent
known device storage includes two 8M-by-4-float descriptor textures, two M-by-2
location textures, the M-by-M float dot texture, match rows and optional
M-by-ceil(M/8)-by-4-float cross-check texture (the source clamps channels to four).
The retained host `sift_buffer` is charged separately, including growth overlap;
it is not device memory. CUDA allocation/query/matching failures refuse, not
successful empty matches.

Per-pair known credits use actual descriptor/keypoint rows, stored-match rows
(which can exceed the configured GPU result limit), complete verifier point
vectors, corresponding points/ray/Jacobian/mask arrays, CPU float search/index
conversion or brute-force distance-matrix staging, and raw/inlier/result DB BLOB
storage. Credits travel through the existing matcher/verifier/output queues until
DB writing and payload destruction. Result vector **capacities**, not only sizes,
are reconciled at the writer; extra logical growth fails immediately if unavailable.
Estimator/library internal growth remains policy, not a physical allocation guard.

The producer is also the DB writer: on nonblocking admission contention it drains
one pending result, writes it under the existing DB serialization, retires its
payload, and retries the **same** pair. It never waits holding all credits for a
writer that cannot run. Existing cache DB locks are per operation, not held across
admission/draining. Exhaustive pair blocks and feature reads remain unchanged;
interleaved writes concern result tables, not frozen feature inputs. Failure,
no-fit, cancellation or writer/worker exceptions stop/wake all queues, join workers
and clear remaining payload owners, returning command status 1 (main retains
SIGINT status 130). Partial DB state is retained honestly, not labelled success.

Exactly one ceil(3%) is applied to live known + raw policy; reserve is outside it.
This is **logical payload admission, not physical fit**: FAISS training/search,
list capacities, RANSAC/solver/refinement scratch, metadata, SQLite pages, vector
allocator slack, texture handles/driver and allocator overhead remain UNKNOWN.
Geometric verification may differ across runs because default thread-local PRNG
state is consumed by worker scheduling; raw match equality and full geometry
field/inlier-order comparisons require separate effective-input/RNG binding.
No seeds or geometry fields are changed/canonicalized to force agreement.
Matching's full physical model/calibration and remaining mapper growth remain
future work; this slice does not qualify whole-pipeline fit or performance.

### Initial CPU mapping payload and correspondence-graph admission

The frontend passes a fresh observed, lower-capped **host** ceiling and existing
host policy/reserve terms to `mapper`, with phase-scoped RSS monitoring restored
in `finally` on both success and failure. The default local/global BA backends
remain CPU Ceres: `Mapper.ba_gpu_index` alone does **not** enable GPU BA. No
backend, solver options, seed, threading, image selection, registration order,
color extraction, track coverage or resolution is changed.

Native `Mapper.host_budget_bytes=0` preserves unbudgeted behavior. Nonzero budgets
support fresh SQLite-backed incremental mapping with CPU Ceres via `mapper` and
`pose_prior_mapper`, including the existing multiple-model behavior. Nonzero
budgets are unsupported by `hierarchical_mapper` and `point_triangulator`; both
commands reject them immediately after option parsing, before loading a database
or input model. Both incremental commands reject budgeted input-model continuation
before loading the model (or updating pose-prior covariance). Fresh pose-prior
mapping retains admission and controlled failure handling, with the existing
main signal handler owning interruption. Explicit GPU BA, other BA backends,
input-model continuation and preloaded-cache construction refuse before their
unsupported materialization; there is no automatic fallback. Budgeted host observation needs
a supported Linux/cgroup-v2 MVS build. `Mapper.policy_host_bytes` and
`Mapper.reserve_host_bytes` use the existing strict byte-option parser; the
frontend supplies policy automatically without new declared-byte requirements.

A read transaction spans metadata inventory and payload loading, including WAL
snapshots. Inventory validates full-width dimensions, scalar representation,
BLOB lengths, image references and optional matrix/pose/camera shapes before
keypoint or two-view payload allocation. Descriptors and raw matches are not
loaded by this mapper path. The conservative all-DB reservation covers keypoint
read/converted-coordinate overlap, cache Point2D values, two-view match BLOB and
converted vectors, the pre-sized geometry array and optional camera overlap.
Rows later filtered by the existing mapper are still reserved: this may
**over-refuse**, never truncate images/features/correspondences.

The graph additionally admits actual per-point adjacency vector capacities and
replacement overlap before growth. Flattening retains original adjacency until
all flat buffers and offsets are ready, then retires old storage and its credits.
A flattening refusal leaves original adjacency queryable, not a half-finalized
successful graph. Shared graph storage is charged once, even while cache,
mapper, triangulator and observation manager share it. Each initial model's
Point2D copy and observation counters/visibility pyramids are admitted separately.
Retained models keep their copy credits until actual owner destruction; credits
are conservatively **not** returned merely by `TearDown`, deleting vector elements
or writing a model. Model/graph payloads die before their credit holders.
Copies of budgeted reconstructions independently admit the clone's covered
Point2D payload before copying it; assignment prepares and rebinds the complete
replacement before a nonthrowing member-swap commit. Other cloned model payloads
remain UNKNOWN, except for explicitly opted-in Track storage described below.

A component-only `Track(ledger)` API admits requested vector replacement payloads
before allocation, including old/new overlap. Copies acquire independent credits;
moves transfer ownership. `Track(ledger)` does **not** permanently bind the object
to that ledger: copy/move assignment replaces the destination's ledger policy
with the source's, including clearing it for an unbudgeted source. Copy assignment
from an unbudgeted source allocates without admission, and subsequent growth
remains unbudgeted. Old destination storage is destroyed before its credits
retire. This source-policy assignment is accepted only for the opt-in value
component; future production attachment must audit/prevent unintended policy
shedding, not assume assignment retains the destination ledger. Erase retains
storage credit, and compression prepares a replacement before retiring old storage.
Allocator-provided capacity beyond the requested amount remains **UNKNOWN**, not
covered by the aggregate 3% margin.
Arbitrary growth through the compatible mutable `Elements()` vector reference and
caller allocations passed to `SetElements` are outside interception. This optional
Track ownership is **not attached to native mapper/triangulator Tracks**: merge
node preparation and observation-statistics transaction safety are deferred.
Therefore this component does not expand production Track-growth admission or
provide a Point3D-storage cap. `AddObservation` does finish any throwing track
append before linking the image; it does not enable native Track admission.

Admission applies exactly one aggregate `ceil(3%)` to live known + raw host
policy, with reserve outside. Capacity is `min(observed_FREE, supplied_cap)`;
already-live usage is excluded from FREE and is not subtracted again. This is
**initial payload and controlled graph-growth admission, NOT a whole-SfM hard
cap or physical-fit guarantee**. Point3D/track growth, triangulation temporaries,
uncovered reconstruction-copy payloads, RANSAC/Ceres/linear-solver workspace, full-resolution
color bitmap/decoder storage, metadata/hash tables, SQLite caches and allocator
internals remain UNKNOWN, covered only by uncalibrated additional policy and
external safeguards. Direct-child RSS monitoring is best-effort, not a process-
tree memory limit. No physical-fit, quality or performance qualification follows
from passing this partial ledger.

Loader/growth failures unwind owned payloads and return status 1 without the
success flush/project.ini path. Already-finished submodels or explicitly requested
snapshots from earlier progress can remain; these are not frontend completion.
Main remains the sole signal handler and returns 130 for SIGINT after cooperative
cleanup. Existing interruption behavior may retain a valid partial sparse model;
the frontend still refuses to publish a completed result after nonzero exit.
Retained output comparisons require matched DB geometry, effective options,
thread/RNG history and registration order. Unknown repeatability is not repaired
by sorting geometry, changing seeds or canonicalizing numerical differences.

### Undistortion admission

Immediately before undistortion, the frontend also passes an observed, lower-capped
host budget and the existing host policy/reserve allowances automatically. No new
user byte declarations are required. Native `image_undistorter` exposes
`--host_budget_bytes`, `--policy_host_bytes` and `--reserve_host_bytes`; a zero
budget preserves the unbudgeted path (policy/reserve must then be zero).

Budgeted execution supports COLMAP output with validated UINT8 image headers of
1–4 channels, matching model dimensions, ordinary perspective camera models
(including nonzero-distortion SIMPLE_RADIAL) and the supported spherical path.
ROI, dry-run and non-COLMAP output combinations are unsupported in this mode.
Header checks precede pixel allocation. The known RGB payload accounts for
source/output overlap, the full-resolution intermediate warp before substantial
downscaling, spherical cloning and grayscale-to-RGB conversion. Decoder, filter,
writer, model and allocator overhead remain policy allowances—not measured bounds.

Native admission freezes its observed host/cgroup capacity, limited by the supplied
cap. It admits ordered batches within `--num_threads`, reducing concurrent workers
when needed without changing resolution, warp arithmetic or image order. One
aggregate 3% margin covers concurrent known payload plus the once-per-phase policy;
reserve is added separately. A singleton that cannot fit refuses before decode.
Failures and cancellation drain admitted workers before returning failure; images
already written may remain, but successful sparse/config completion is not
published by a failed run. The frontend records failure rather than `result.json`.
Logical admission and direct-child RSS monitoring do not guarantee physical fit or
hard process-tree containment. SfM and complete physical stage forecasts remain
UNKNOWN.

### Fusion admission

This optional native logical budget supports only single-thread, full-resolution
COLMAP mmap fusion, without cache or external masks. Other budgeted settings
refuse. Distinct configured inputs contribute retained RGB plus atomic masks
(`4 * sum(width * height)`); the ordered initialization peak includes temporary
char and atomic mask coexistence (`max(4*sum(area), 4*prefix + 5*area)`). Mapped
depth/normal payload (`16*sum(area)`) is reported separately, never a resident
floor or free credit. Bitmap headers and complete map sizes are checked before
decoding. Input refusal precedes RGB/mask allocation.

Output point, visibility-container and visibility-integer vector capacities are
checked before growth, including old/new buffers and task/final merge overlap.
Unknown scratch, decoder, metadata, allocator and traversal costs remain explicit
policy: the existing raw loader-scratch + process-host + unknown-host allowances
are summed once. Native applies one aggregate 3% margin to accounted inputs,
outputs and policy, then adds the separate reserve. Runtime checks use this same
frozen capacity, not FREE-minus-already-live storage a second time. Budget
exhaustion drains workers and exits with code 3, without output serialization or
a success receipt. Unbudgeted native fusion retains its old behavior; set
`StereoFusion.host_budget_bytes` to opt in, with optional `policy_host_bytes` and
`reserve_host_bytes` in the same namespace. These logical checks and direct-child
RSS monitoring remain best-effort, not whole-process-tree isolation. No universal 8GB, physical-fit, quality,
throughput or performance-qualified claim is made. `--require-hard-memory-cap`
always refuses before creating outputs. External OS limits remain the user's
responsibility where required.

`planning/build.json` records the executable SHA and generated same-build
fingerprints. `planning/intent.json` preserves requested overrides (null means default).
`planning/resource-policy.json` and `result.json.resource_policy` record the
selected policy/version/allocation binding, every raw term's default/user origin
and provenance, and UNKNOWN whole-pipeline physical fit. The selected policy is
also printed before execution. `planning/resources-*.json` preserves native FREE
observations and effective lower-only capacities, including reserve refusals.
`planning/dense-plan.json` records planned inputs/options;
`planning/cohort.json` reports actual completed native backend, storage, owner
limits and exact per-reference effective options/source order. The frontend
compares these, validates complete finite map payloads, publishes without clobber,
checks actual fusion eligibility and consumes the mapped-fusion receipt before
writing `result.json`. Native exit alone or incomplete output is not success.

## Validation boundaries

Frozen dense-workspace exactness and fresh photo-directory validity are different
checks. Fresh mapper results cannot be compared by deterministic point count.
An already-undistorted photo fixture does not qualify raw-camera distortion or
reconstruction accuracy. This frontend does not promote experimental backend
performance or hardware qualification.

Selective frontend provenance: PR16 `701da8fd1` (commits `9ae0f642c`, `074ecb001`,
`705ee7ce2`, `701da8fd1`), not a wholesale merge. Its old PHOTO N1/N2 / GEO N1
producer/wire/resident dispatch is replaced, not installed. See the packaged
NOTICE for attribution.

Frontend CPU/installation regressions (synthetic native protocol, no GPU proof):

```sh
PYTHONPATH=python:python/tests python3 -m unittest \
  colmap_reconstruction.reconstruction_test test_reconstruction
```

They are also registered as CTest `reconstruction/frontend_test` on Linux test
builds. Native adapter/command checks require the real built executable:

```sh
COLMAP_TEST_EXECUTABLE=/prefix/bin/colmap PYTHONPATH=python python3 -m unittest \
  colmap_reconstruction.fixed_cohort_cli_test
```

The existing GPU tests are opt-in; their skip is not execution qualification.
