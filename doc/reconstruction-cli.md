# Experimental installed photo-directory reconstruction

CMake installs `colmap-reconstruct` beside `colmap`. It requires Python 3.10+,
Linux, a **non-root user**, and a matching CUDA-enabled COLMAP build with its
runtime libraries available. This is not a standalone static binary. No
pycolmap, source checkout, external Python package or separate resource calculator
is needed. The launcher uses isolated Python and prefix-relative modules.

Example (all byte values below are **user-supplied placeholders**, not calibration):

```sh
colmap-reconstruct --experimental-local --photos /data/photos --workspace /data/new-run \
  --loader-scratch-host-bytes "$DECODE_SCRATCH" \
  --process-host-bytes "$PROCESS_HOST" --process-device-bytes "$PROCESS_DEVICE" \
  --unknown-host-bytes "$UNKNOWN_HOST" --unknown-device-bytes "$UNKNOWN_DEVICE" \
  --unknown-provenance 'my explicit uncalibrated allowance basis' \
  --reserve-host-bytes "$HOST_RESERVE" --reserve-device-bytes "$DEVICE_RESERVE"
```

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
Process, unknown overhead, decoder scratch and reserves remain explicit inputs;
no free-minus-known filler or invented calibration is used. Counts are
memory-admissible limits, **not measured overlap or fastest settings**.

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
fingerprints. `planning/dense-plan.json` records planned inputs/options;
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
