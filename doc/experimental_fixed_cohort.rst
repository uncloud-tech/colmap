Experimental fixed-cohort PHOTO → GEO
======================================

``colmap experimental_fixed_cohort --help`` is an opt-in native executable
entry point, independent of Python/pycolmap, source checkouts and test binaries.
It consumes an existing **undistorted COLMAP workspace** (``images/``,
``sparse/``), runs all PHOTO producers, then a finite closed set of GEO
consumers, and writes native binary depth/normal maps. It does not perform
SfM, fusion, meshing, consistency-graph output or change stock command defaults.
Execution requires a Linux native CUDA MVS build (not HIP). CPU MVS builds
support help and actual workspace preparation, never a stock execution fallback.

Configuration and phase options
-------------------------------

Use the existing ``stereo/patch-match.cfg`` two-line syntax. For example, for
three actual images named ``a.jpg``, ``b.jpg`` and ``nested/c.jpg``::

    a.jpg
    nested/c.jpg,b.jpg,nested/c.jpg
    b.jpg
    a.jpg,nested/c.jpg
    nested/c.jpg
    b.jpg,a.jpg

Every config reference is selected, in PHOTO config order. Sources are explicit,
ordered and may repeat; no hash-set conversion, sorting, deduplication or
``__auto__``/``__all__`` expansion occurs. Names must be relative with no dot or
parent components. ``--photo_config_path`` defaults to the workspace config;
``--geo_config_path`` defaults to the PHOTO config. Separate configs must select
the same references; GEO may use only these PHOTO producers. PHOTO may use
additional workspace images. All workspace images are loaded, not just sources.
Keep workspace files immutable for the full invocation.

The full existing ``PatchMatchStereo.*`` options are available. This first
entry uses **explicit global** positive ``depth_min``, ``depth_max`` and
``sigma_spatial``, an explicit positive ``num_threads`` and a single canonical
GPU index. It never infers a depth range or sigma. Global bounds are not a
substitute for differing per-reference bounds in earlier benchmark fixtures.
``PatchMatchStereo.max_image_size`` must stay -1; resize the loaded bank with
``--max_load_image_size`` (-1 or positive). PHOTO is always unfiltered and
without geometry; GEO is always filtered with geometry. Leave the shared
``geom_consistency`` and ``filter`` flags at 1. Unsupported native option
combinations, missing sources and nonclosed GEO coverage refuse before GPU work.
``--active_owners 1`` is the unchanged serial default; 2 permits two owners.
Manual higher counts are supported up to the number of configured jobs, using
the same canonical resource gates. Graphs are selected with
``--PatchMatchStereo.owned_cuda_graphs 1``. There is no performance or quality
promotion implied by these settings.

Invocation
----------

Set shell variables below to the actual workspace/config, **chosen depth/sigma**,
resource caps and independently justified raw allowances for your invocation.
The byte variables are integers, not GiB strings. There are no calibrated
allowance defaults. The output parent must already exist without symlink
components; the output root must not exist. Then run the installed executable::

    colmap experimental_fixed_cohort \
      --workspace_path "$WORKSPACE" --output_path "$FRESH_OUTPUT" \
      --photo_config_path "$PHOTO_CFG" --geo_config_path "$GEO_CFG" \
      --PatchMatchStereo.depth_min "$DEPTH_MIN" \
      --PatchMatchStereo.depth_max "$DEPTH_MAX" \
      --PatchMatchStereo.sigma_spatial "$SIGMA_SPATIAL" \
      --PatchMatchStereo.num_threads 1 --PatchMatchStereo.gpu_index 0 \
      --max_load_image_size 512 --active_owners 1 \
      --PatchMatchStereo.owned_cuda_graphs 0 \
      --output_policy final_geo_only --photo_source_storage private \
      --deadline_seconds 300 \
      --host_capacity_bytes "$HOST_CAP_BYTES" \
      --device_capacity_bytes "$DEVICE_CAP_BYTES" \
      --reserve_host_bytes "$HOST_RESERVE_BYTES" \
      --reserve_device_bytes "$DEVICE_RESERVE_BYTES" \
      --process_host_bytes "$PROCESS_HOST_BYTES" \
      --process_device_bytes "$PROCESS_DEVICE_BYTES" \
      --unknown_host_bytes "$UNKNOWN_HOST_BYTES" \
      --unknown_device_bytes "$UNKNOWN_DEVICE_BYTES" \
      --unknown_provenance "$ALLOWANCE_JUSTIFICATION" \
      --loader_scratch_host_bytes "$LOADER_SCRATCH_BYTES"

Add ``--prepare_only 1`` for CPU decoding, shape/options/closure and resource
checks only: **NO solver execution, reconstruction, results or output creation**.
Manual preparation remains GPU-free; the explicit observation mode below queries
CUDA even during preparation.
The output path argument is unused in that mode, including existing paths.
Preparation success does not certify CUDA capability or physical memory fit.

``final_geo_only`` writes only
``$FRESH_OUTPUT/stereo/{depth_maps,normal_maps}/NAME.geometric.bin``, including
GEO normal maps. It creates no PHOTO files. ``photo_and_geo`` additionally
writes ``NAME.photometric.bin`` for both map types. Both policies retain all
PHOTO device snapshots through final GEO drain. These are standard COLMAP
binary map layouts, in original reference/source order. Input workspace files
are never overwritten. The new output root is exclusively created, never
reused. Failures leave owned partial outputs for inspection and return nonzero;
there is no automatic deletion or claim of transactional publication.

Opt-in observed FREE capacities
-------------------------------

Add ``--observe_capacities 1 --auto_active_owners 1`` to derive the existing
selector's host/device ceilings from a real available-memory snapshot. The mode
requires automatic owner selection; it does not change the manual defaults.
Omit either or both ``--host_capacity_bytes`` / ``--device_capacity_bytes`` in
this mode, or supply explicit unsigned-byte caps: effective capacity is
``min(observed_free, supplied_cap)`` independently per resource. Zero caps are
real zero ceilings, not "auto"; malformed/negative/overflowing caps refuse.
Without observation both manual capacities remain required, even with automatic
owners. The optional ``--active_owners_ceiling`` retains its existing meaning.

The command reuses Linux ``MemAvailable`` constrained by remaining cgroup-v2
``memory.max - memory.current`` at the leaf **and every visible ancestor**.
It selects the single canonical ``PatchMatchStereo.gpu_index`` (a process-visible
CUDA ordinal), initializes that device/context as needed, and reads
``cudaMemGetInfo`` **free**, not total, memory. The sparse metadata model has
been released, and no workspace pixels, frozen copy, device banks, snapshots or
solvers have been allocated at this point. GPU ordinal, name, raw hexadecimal
CUDA UUID, observed host/device free bytes, device total bytes (evidence only),
effective capacity bytes and selected owner count are printed. Query errors or
unsupported builds/devices refuse explicitly, never as a memory shortage or a
silent manual/stock fallback. Linux native CUDA is required for observation;
CPU/HIP builds still support GPU-free manual preparation.

These are conservative **incremental FREE-envelope ceilings**, not total-memory
budgets. Already-live process/context usage is excluded from free; the existing
raw process and unknown allowances remain **additional conservative charges**.
They are not inferred live credits, not enlarged free memory, and not calibrated
context allowances. This can reject work that would physically fit. No extra
percentage margin is introduced: the existing logical demand retains exactly
one aggregate 3% margin, all unknown/provenance requirements, loader peak and
unallocated reserves. Later real loader/native phase headroom checks remain.
The snapshot is racy, not atomic across host/device, and not a reservation or
physical-fit guarantee; other allocations can invalidate it immediately.
The blocking device query is covered by the existing absolute deadline guard
using the original command deadline, not a fresh budget. External
process/resource limits remain necessary. There is no promise of 8 GiB
support or optimal throughput, no quality change, and no automatic eviction.

``--prepare_only 1 --observe_capacities 1 --auto_active_owners 1`` explicitly
permits that device/context query, then performs the existing CPU preparation.
It never constructs a solver/device bank, solves, writes maps or creates the
output root. Successful observation/preparation is not GPU numerical
qualification. To exercise actual installed CLI tests on a GPU runner, set
``COLMAP_TEST_EXECUTABLE`` and ``COLMAP_TEST_OBSERVED_CUDA=1``; optionally set
``COLMAP_TEST_CUDA_DEVICE`` to the selected process-visible ordinal. Run
``python -m unittest discover -s python/colmap_reconstruction -p fixed_cohort_cli_test.py``.
A CPU-only runner can instead set ``COLMAP_TEST_NO_CUDA=1`` to test the explicit
unsupported observation refusal. These opt-ins do not turn skipped GPU tests
into qualification.

Resource and qualification limits
---------------------------------

The native producer shares the validator's checked logical inventory and
``ResidentPrivateLogicalMinimum`` authority, not a new physical estimator.
Loaded and frozen pixels, selected host maps, all PHOTO snapshots and active
private solvers/staging are charged. Native admission applies **one aggregate
3% margin** plus a separate unallocated reserve. Process, loader scratch and
positive unknown overhead with provenance remain caller-declared raw bounds.
No free-memory-derived or unmeasured live-process credits are taken.
Linux/cgroup host and native GPU headroom observations do not impose hard caps;
apply appropriate external process/resource limits. Pitch/context/allocator
and codec behavior are not physically calibrated by this inventory.

Explicit shared PHOTO mode
--------------------------

``--photo_source_storage private`` is the unchanged CLI default. Explicitly use
``--photo_source_storage shared`` to bind the shared PHOTO grayscale union to
both canonical inventory/admission and execution. Other values except ``auto``
refuse before workspace loading or output/device side effects.
``--prepare_only 1`` uses the same
selection and reports its logical host/device union payload (zero for private),
without allocating a device bank or claiming physical VRAM measurement.

Opt-in ``--photo_source_storage auto`` requires ``--auto_active_owners 1``;
it is a selection request, not a third runtime bank mode. The native metadata
producer evaluates **both** private and shared inventories before pixel decode,
bank or solver allocation, under the same effective observed FREE/caps, user
ceiling, process/unknown declarations, single 3% margin and reserves. It supports
both common owner selection and ``--auto_phase_active_owners 1``. No second CUDA
capacity query is used to compare candidates.

Shared is chosen only if it admits at least as many PHOTO **and** GEO owners
with at least one strictly higher count, or if private cannot fit a complete
pipeline and shared can. Equal or incomparable feasible counts prefer private;
if only private fits it is retained. Neither fitting reports both capacity
reasons. Malformed/unsupported inputs, overflow (including the full effective
ceiling in either mode) and query errors fail closed, never trigger fallback.
Global padding, zero source overlap and the additional shared host payload can
erase sharing gains or make it infeasible.

The report records requested ``auto``, effective storage, selection reason and
chosen PHOTO/GEO counts. On successful execution it also records actual uploaded
union payload and the limits passed to each phase's coordinator (not measured
owner overlap). The selected canonical budget drives actual bank and owner
allocation as well as forecasts; there is no late OOM/partial-run fallback.
This is logical memory admissibility, not fastest execution, physical fit or a
general maximality claim. Observed FREE remains racy, incremental and unreserved;
no live credit, quality reduction or default promotion is implied.

For typed native API callers, ``FixedCohortBudget::photo_source_storage =
FixedCohortPhotoSourceStorage::kShared`` is the equivalent opt-in.
Pass this declaration through the existing
``BuildFixedCohortLogicalInventory`` (or Python ``forecast_fixed_cohort`` with
``PhotoSourceStorage.SHARED``) before admission. The runner actually constructs
the existing host/device grayscale bank from its frozen, loaded pixels; it does
not reread paths or alter cameras, PRNG, source order or duplicate ordinals.
Layers deduplicate only by invocation image ordinal, not by filename.

``photo_shared`` charges one globally padded uint8 host union and one device
array. Upload reads that same host union without a second payload staging copy.
Host padding remains live for binding validation; when supplied, loaded source
pixels are checked byte-for-byte. Explicit metadata-only borrowers use the bank
binding as their pixel authority (paths are labels, not file identities).
Per-owner PHOTO device source-array charges are removed, but all private solver,
layer mapping and reference storage remain. The per-owner host source-packing
allowance remains to cover ordered Image-copy validation transients, including
duplicates. Both unions survive all PHOTO borrowers/graphs and their failure
drains/joins, then are released before the GEO headroom observation. GEO storage,
all snapshots, returned maps and caller/frozen pixels retain their prior charges.

Global padding can outweigh deduplication, especially with zero overlap or
opposed aspect ratios. Shared mode therefore makes no savings/physical-fit
promise and never silently selects a cheaper forecast while allocating private
arrays. The private default is unchanged. Unknown allowances, reserve and the
single aggregate 3% margin are unchanged; CUDA array overhead is not certified
by these logical payload counts. Source-storage selection is independent of
active-owner selection below.

Automatic memory-admitted concurrency
--------------------------------------

Explicitly opt in with ``--auto_active_owners 1``. The command selects the largest
**memory-admissible** count from 1 through the configured job count, optionally
bounded by ``--active_owners_ceiling N`` (0, the default, means job count).
Do not combine automatic selection with a nondefault manual ``--active_owners``.
A ceiling without automatic selection, negative ceiling or nonpositive manual
count refuses. The existing default/manual 1/2 behavior is unchanged.

Selection uses the caller's explicit host/device capacities (or the opt-in
observed FREE ceilings described above) and the same raw canonical inventory, resourcewise top-N private charges, one aggregate 3% margin
and separate reserve as manual admission. Host and device bottlenecks can belong
to different jobs. It precedes pixel decoding, candidate-dependent payload
allocation and solver construction; there are no OOM trials. The normal command
prints ``active_owners``, automatic/manual selection and the limiting host/device
capacity, job count or user ceiling, including in ``--prepare_only 1``. If not
even one owner fits, it reports ``no singleton fits`` and the limiting resource.
Invalid bounds and arithmetic overflow refuse rather than selecting a lower N.

This is **not a measured throughput optimum**, physical-fit certification or
whole-machine capacity discovery. The full caller/frozen input and snapshot/output
floor stays retained. Decode scratch, actual free headroom and physical-copy
checks can still refuse independently; automatic selection never reduces quality,
resolution or sources to fit. No eviction, spill or phase mixing is introduced.
The existing rolling owner loop retains credits through destruction and parent
join; all PHOTO owners still drain before any GEO owner begins. Private/shared
PHOTO storage semantics and optional stream/graph execution remain unchanged.

Typed callers build the canonical inventory first, then call
``SelectFixedCohortActiveOwners(budget, capacity, ceiling)`` and bind its ``count``
to ``budget.max_active_owners`` before preparing/executing. Serial dispatch requires
count 1; concurrent ``PairDispatch::kPair`` supports counts >=2 only in this
fixed-cohort entry. Unrelated standard pair/triple contracts are unchanged.
Python ``forecast_fixed_cohort`` accepts the effective count for the same raw
inventory; capacities, selection and the single margin remain native concerns.

Independent phase owner counts (explicit opt-in)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Add ``--auto_phase_active_owners 1 --auto_active_owners 1`` to select PHOTO and
GEO counts independently under the **same** effective host/device capacities and
``--active_owners_ceiling``. Both manual capacities and ``--observe_capacities 1``
work. Defaults, manual counts and uniform automatic selection are unchanged.
Reports include ``PHOTO active_owners=...; limiting_reason=...`` and the matching
GEO line; the compatibility ``active_owners`` field is their maximum, not a
uniform count. Refusal names the phase if its complete singleton cannot fit.

Each search checks arithmetic at its effective ceiling before any capacity
short-circuit. Actual phase peaks use resourcewise top-N private charges for
that phase, plus the unchanged full retained inventory (and PHOTO-only shared
bank). One aggregate 3% margin and the existing reserve apply, never per-owner
margins. Snapshot, fixed-bank, unknown and process declarations receive no new
credits. Phase observations still perform the existing conservative headroom
checks; no new observation, live credit or capacity enlargement is introduced.
Observed FREE remains incremental, racy and unreserved. The largest logically
admissible counts are neither a throughput optimum nor a physical-fit guarantee.
All PHOTO owners complete and retire before GEO starts; no mixed phases,
eviction, residency changes or quality changes are enabled.

Typed callers use ``SelectFixedCohortPhaseActiveOwners`` and bind the PHOTO/GEO
counts to ``budget.phase_active_owners``, with their maximum in
``budget.max_active_owners`` for dispatch compatibility. Both zero phase counts
retain uniform behavior; partial/zero/out-of-range explicit counts are invalid.
Python ``forecast_fixed_cohort(..., phase_active_owners=(photo, geo))`` mirrors
the raw inventory and initializer, not native capacity selection.

One original deadline covers preparation, execution and output. Checks occur
between output writes, not as an interrupt guarantee inside blocking output
I/O. Native setup/drain uses the existing fail-closed deadline semantics.
This is experimental/unqualified: no full-scene throughput, quality or
physical-fit guarantee, no default promotion and no new reconstruction policy.
