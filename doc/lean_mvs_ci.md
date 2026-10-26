# Experimental Linux MVS CI

PRs **targeting** `dev/streaming-workflow` use `lean-mvs.yml` instead of the
upstream platform, wheel and release matrices. Main/release push, release and
existing manual triggers are unchanged. No publishing or merge automation is
added. External-fork PRs do not execute on Blacksmith; their lean jobs are
skipped. Maintainers can manually dispatch a trusted repository revision.

Both jobs check out the actual PR head with full history and no persisted
checkout credentials. Fork/lineage and generated source fingerprint checks are
not bypassed.

* **CPU/format:** existing Ruff 0.15.20 and clang-format 22.1.5 check files
  changed against the PR base SHA (manual dispatch: selected revision's parent),
  without rewriting unrelated base-branch formatting; 13 real
  standalone MVS C++17 contract executables, with assertions enabled and strict
  warnings (only existing aggregate missing-field warnings are suppressed).
  Covers optimized dispatch, photo forecast/binding/dispatch/delivery/options/
  paths/prepared/concurrency, fusion dependencies, graph topology/coordinator,
  and deadline handling. Also runs fingerprint, Windows-byte and scoped
  generated-header install plumbing regressions. The latter is **not** a full
  SDK build. Run locally with `bash scripts/ci/run_mvs_cpu.sh`; set `RUNNER_TEMP`
  to an executable temporary directory, or it defaults to the workspace.
* **Native CUDA compile/link only:** the established digest-pinned CUDA 12.8.1
  Ubuntu 24.04 developer image is a normal GitHub job container. Git is installed
  before checkout; normal CMake generators and pinned FetchContent dependencies
  run on the checked-out source. A positive target list builds a fresh CLI,
  production MVS CUDA library and CUDA unit binaries for SM86 and SM89, with
  parallelism limited to eight. An optional normal-rotation test is included
  only if its source and registration exist. Failure logs are uploaded.

The native job has no physical GPU and executes no GPU binaries. Successful
compilation does not qualify CUDA runtime behavior, driver compatibility,
performance, numerical results, or reconstruction quality. Blacksmith/container
execution must be qualified by an actual Actions run; local CPU validation is
not evidence that the native job has run.
