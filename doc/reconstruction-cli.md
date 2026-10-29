# Experimental installed reconstruction command

`colmap-reconstruct` runs the native feature extractor, exhaustive matcher,
incremental mapper, image undistorter, optimized PHOTO and GEO stereo, and
optimized mmap fusion. It starts from a directory of overlapping photographs,
not an existing sparse model or a benchmark fixture. There is **no stock fallback**.
This is an explicit experimental backend, not a claim of best performance,
calibrated memory consumption, or general reconstruction quality.

## Install

Use the existing COLMAP native build and its normal dependencies, including
CUDA/MVS and download support. Python 3.10+ and the NVIDIA driver's `nvidia-smi`
are required for execution. The current optimized path supports Linux/NVIDIA
CUDA (not HIP); actual native capability failures still refuse.

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$HOME/.local/colmap" -DGUI_ENABLED=OFF \
  -DCUDA_ENABLED=ON -DMVS_ENABLED=ON -DDOWNLOAD_ENABLED=ON
cmake --build build --parallel 2
cmake --install build
```

The native CMake install includes the stdlib-only package under
`share/colmap/python` and a prefix-relative `bin/colmap-reconstruct` launcher.
It invokes Python with `-I -B`; neither a Python wheel build nor NumPy,
pybind11, scikit-build-core, source Git, an estimator checkout, a lab checkout,
or `PYTHONPATH` is needed for installed use. The existing pycolmap wheel also
includes the frontend and console entry, but still requires a matching native
`colmap` executable; the wheel is **not** the minimal installation route.

```sh
env -u PYTHONPATH -u PYTHONHOME "$HOME/.local/colmap/bin/colmap-reconstruct" \
  --experimental-local --photos /path/to/photos --workspace /path/to/NEW-output \
  --host-budget-bytes 17179869184 --device-budget-bytes 10737418240 \
  --timeout-seconds 1800 --threads 4
```

Budgets in this example are illustrative, **not** a hardware or workload
recommendation. `--help` lists explicit cache, source cap, samples, seed, tile,
lookahead, resizing, absolute headroom and extra allowance controls. No quality
preset or automatic quality reduction is applied. Feature extraction preserves
the existing 3200-pixel working cap; undistortion defaults to no resize. Dense
stereo always consumes native-resolution undistorted inputs. PHOTO defaults to
N1; `--photo-concurrent-solvers 2` explicitly selects the existing native rolling
N2 runner. GEO and fusion remain N1. `--threads` bounds feature/SfM/undistortion
workers and common CPU thread-pool environment settings, not solver concurrency.
N2 prepares every registered reference in native graph order and submits one V3
phase with one binding/config/options snapshot per job, including jobs beyond
the two active slots. It never substitutes sequential N1 execution.

Fresh mapper runs can produce different components, model numbering and point
counts, even with the same code and photographs. By default, multiple sparse
models still refuse; a single model is selected only when it is the sole output.
`--sparse-model NAME` selects that exact directory name and refuses if absent.
For numbering-independent selection, explicitly pass `--sparse-model largest`:
this reserved policy selects the most registered images, breaking ties by the
lexicographically smallest directory name (e.g. `10` before `2`), not point count.
It does not merge components or guarantee every input photograph registers.
The existing native analyzer runs once per candidate for `largest`; malformed
counts, native analyzer failures, fewer than two registered images or no 3D points
in any candidate refuse rather than silently skipping a model. Exact/default
selection validates only the selected model. No models also refuses.
`planning/sparse-model.json` records the actual chosen name, analyzer counts and
policy (`largest`, `explicit`, or `single`). Every registered reference in
that model must appear in both native dense graphs. Source order is preserved:
PHOTO singleton configs retain the original native source-selection tokens;
GEO must match the fresh post-PHOTO native order. Completion requires all PHOTO
maps, successful native GEO, strict fusion and a readable nonempty PLY.

## Resource and ownership boundary

Ordinary execution requires a non-root user, a new workspace whose parent is
owned by that user, and explicit `--experimental-local` consent. Existing output
paths (including symlinks) refuse; there is no resume or implicit overwrite.
The existing privileged runner/cgroup authorization lane is not changed.

* Host/GPU budgets must fit current available memory minus the explicit absolute
  headroom. Host observations include visible cgroup-v2 ancestor limits. GPU
  observations resolve `--gpu-index` as a CUDA-visible ordinal via the Driver API
  and query `nvidia-smi` by that device's UUID, not its management index. Caller
  `CUDA_VISIBLE_DEVICES`/`CUDA_DEVICE_ORDER` are preserved for native dispatch;
  ordinal, UUID and visibility/order are recorded. Identity resolution failure
  refuses, with no management-index fallback.
* Canonical dense engineering forecasts use **one 3% margin**, once per disjoint
  term. Native PHOTO/GEO independently verify the wire, exact options/order,
  compiled fingerprints, minima and logical budgets. These are admission checks,
  **not physical RSS/VRAM allocator limits**. PHOTO N2 charges one process/cache
  plus the two largest private owner charges per resource (not all queued jobs,
  not a sequential peak, and no shared-gray credits). Extra GEO bank and metadata
  allowances default explicitly to unmeasured zero; callers can increase them.
* SfM/undistortion/fusion physical memory is **not modeled or certified by the
  stereo model**. Direct child RSS is sampled best-effort and overbudget children
  are stopped. Available memory is rechecked at stage boundaries. Concurrent
  workloads, driver allocations and hidden cgroup ancestors remain risks.
* One monotonic whole-job deadline applies across native stages. Cancellation
  and expiry send TERM, allow up to two seconds of drain, then KILL and reap the
  owned direct child, signaling only its new process group. Hash/map validation
  checks the same deadline/cancel state in bounded chunks; PLY checks between
  bounded vertex batches. Native stereo receives floored positive integer
  remaining seconds; less than one second refuses rather than extending time.
  COLMAP is trusted not
  to detach children. This is not a sandbox; SIGKILL/controller death, detached
  descendants and hard tree-wide RSS/VRAM limits require external containment.
  `--require-hard-memory-cap` therefore always refuses, rather than pretending an
  estimate or RLIMIT_AS is a CUDA memory cap.

Native stdout/stderr, build identity, exact dense plan, forecasts, PHOTO audit
records and resource observations remain in the new workspace on success or
failure. `result.json` means operational completion only, not scientific or
performance acceptance. No provider, rental, upload or cleanup actions occur.

## Supported model binding

`colmap reconstruction_info` exports the generated PHOTO source commit,
lineage/fingerprint and resident allocation fingerprint from the **same ELF**
that executes the job. The frontend records its ELF hash and refuses a changed
executable between stages. The source commit is provenance, not a whitelist:
future commits with the supported exact allocation content work without a
manual commit repin. Unknown PHOTO or GEO fingerprints always refuse.

The supported PHOTO closure is `9e749b59…`; the bounded full-bank GEO closure is
`0614ab2b…`. GEO's registry status is `experimental_compatibility`, not
`qualified`: it certifies formula/content support only. Its native wire label
`metadata_only_cpu_estimate` is distinct from comparative-performance or quality
qualification. A changed allocation closure needs actual formula review and
native validation, not an allow-unknown switch.

## Validation and limits

CPU tests exercise prefix-only installation/import isolation, canonical wire and
binding, source order, resource refusals, output validation, owned-process
cancellation/deadline and a **synthetic** complete native protocol. That fixture
is not CUDA/native numerical proof. Four-job N2/N1 synthetic tests check the
full PHOTO barrier, while a compiled CPU probe parses emitted V3 with the real
native headers and tests exact top-two resource caps and integer deadlines.
The synthetic Driver API visibility seam is not real GPU-identity proof.
Before release acceptance, validate a clean
native install outside the checkout, `reconstruction_info`, real options/wire
binding and a fresh small complete CUDA reconstruction with all selected-model
references and nonempty PLY. Native/GPU validation and independent source review
are separate acceptance prerequisites; no completed result is manufactured from
mock dispatch or from a private two-reference bypass.
