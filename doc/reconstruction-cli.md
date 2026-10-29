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
stereo always consumes native-resolution undistorted inputs. Stereo and fusion
remain N1; `--threads` bounds feature/SfM/undistortion workers and common CPU
thread-pool environment settings, not optimized solver concurrency.

If mapping produces more than one sparse model, execution refuses unless
`--sparse-model NAME` explicitly selects one. Every registered reference in
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
  headroom. Host observations include visible cgroup-v2 ancestor limits.
* Canonical dense engineering forecasts use **one 3% margin**, once per disjoint
  term. Native PHOTO/GEO independently verify the wire, exact options/order,
  compiled fingerprints, minima and logical budgets. These are admission checks,
  **not physical RSS/VRAM allocator limits**. Extra GEO bank and metadata
  allowances default explicitly to unmeasured zero; callers can increase them.
* SfM/undistortion/fusion physical memory is **not modeled or certified by the
  stereo model**. Direct child RSS is sampled best-effort and overbudget children
  are stopped. Available memory is rechecked at stage boundaries. Concurrent
  workloads, driver allocations and hidden cgroup ancestors remain risks.
* One monotonic whole-job deadline applies across native stages. Cancellation
  and expiry send TERM, allow up to two seconds of drain, then KILL and reap the
  owned direct child, signaling only its new process group. COLMAP is trusted not
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
`5daa3f6a…`. GEO's registry status is `experimental_compatibility`, not
`qualified`: it certifies formula/content support only. Its native wire label
`metadata_only_cpu_estimate` is distinct from comparative-performance or quality
qualification. A changed allocation closure needs actual formula review and
native validation, not an allow-unknown switch.

## Validation and limits

CPU tests exercise prefix-only installation/import isolation, canonical wire and
binding, source order, resource refusals, output validation, owned-process
cancellation/deadline and a **synthetic** complete native protocol. That fixture
is not CUDA/native numerical proof. Before release acceptance, validate a clean
native install outside the checkout, `reconstruction_info`, real options/wire
binding and a fresh small complete CUDA reconstruction with all selected-model
references and nonempty PLY. Native/GPU validation and independent source review
are separate acceptance prerequisites; no completed result is manufactured from
mock dispatch or from a private two-reference bypass.
