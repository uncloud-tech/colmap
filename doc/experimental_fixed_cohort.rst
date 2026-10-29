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
``--active_owners 1`` is serial; 2 permits two owners. Graphs are selected with
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
      --output_policy final_geo_only --deadline_seconds 300 \
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
checks only: **NO GPU execution, reconstruction, results or output creation**.
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

One original deadline covers preparation, execution and output. Checks occur
between output writes, not as an interrupt guarantee inside blocking output
I/O. Native setup/drain uses the existing fail-closed deadline semantics.
This is experimental/unqualified: no full-scene throughput, quality or
physical-fit guarantee, no default promotion and no new reconstruction policy.
