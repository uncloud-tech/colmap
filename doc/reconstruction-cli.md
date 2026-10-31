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

Dense resolution defaults to full-resolution float32. Only an explicit
`--max-image-size N` resizes undistortion; there is no automatic quality reduction.
Feature extraction retains COLMAP's SIFT default maximum of 3200, independently
of dense resolution. Samples/seed/source cap/cache/lookahead/tile/GPU are explicit
CLI options (`--help`); existing native command defaults are unchanged.

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
scale from hardware size or pretend to forecast SfM/fusion. Native input-derived
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
CUDA context. These reserve checks and direct-child RSS monitoring (when a host
budget is supplied) are best-effort,
not whole-process-tree isolation. No universal 8GB, physical-fit, quality,
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
