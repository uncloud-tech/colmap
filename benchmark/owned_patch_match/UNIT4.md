# Finite TWO owned references — experimental source interface

This is an optional source-only prototype, not a promoted backend or scheduler.
Build/ABI/CUDA/exactness/performance qualification is a separate gate. The retained
ONE-A result belongs to checkpoint `d41c4915`, not automatically to this path.
No MPS, G6, geometry, pooling, adaptive N, or allocator rewrite is incorporated.
The stock CUDA hot translation unit and default Mat writer remain unchanged.

## Target and invocation

Default OFF CMake option `OWNED_PATCH_MATCH_PAIR_DRIVER_ENABLED` adds
`colmap_owned_pair`. It requires Linux, CUDA and MVS, and rejects HIP. No
subcommand is used. `--help`/`-h` is the sole optional standalone invocation.
Normal invocations require exactly one occurrence of each key below, as separate
`--key value` arguments. Unknown/duplicate/abbreviated/`--key=value` syntax rejects.

- `--dispatch serial|pair`
- `--job_a_project_path <INI>` / `--job_b_project_path <INI>`
- `--host_budget_bytes <uint64>` / `--device_budget_bytes <uint64>`
- `--job_a_host_charge_bytes <uint64>` / `--job_a_device_charge_bytes <uint64>`
- `--job_b_host_charge_bytes <uint64>` / `--job_b_device_charge_bytes <uint64>`
- `--process_host_charge_bytes <uint64>` / `--process_device_charge_bytes <uint64>`
- `--io_threads_limit <uint64>`

Byte/count tokens are unsigned ASCII decimal digits: no sign, whitespace,
scientific notation or overflow. Reservations/allowances are positive. The
process allowance is charged once; both job allowances are charged atomically
against separate host/device capacities, before owner preparation. No one-job
admission or fallback occurs. Serial still requires the whole both-resident
footprint. Wait timeouts are fixed at 300000 ms per coordinator wait, not an
end-to-end process deadline; the external runner must bound the whole process.

Each strict INI registers workspace_path, workspace_format, pmvs_option_name,
config_path, the existing PatchMatchStereo group, and process logging/random
settings. Process globals are read into private storage, compared between jobs,
then applied once before owner threads start. The pilot requires native-sized
COLMAP photometric radius5/step1, five iterations, filtering enabled, complete
inputs and no graph output. Each job config must resolve exactly one reference;
A/B references differ and select the same explicit nonnegative GPU index.
The original loader determines effective source order and depth ranges.

Explicit positive per-job num_threads must each be <= io_threads_limit <= CPU
affinity count. Preparation is serialized and each I/O pool is drained and
joined before ready; the cap therefore bounds aggregate active controller I/O
workers, not two independent simultaneously active pools. The two CUDA-owner
threads are separate. Library-internal codec workers are not established by
this count. Affinity is NOT CPU quota entitlement: external ancestor cpu.max,
cpuset and resource admission are still mandatory.

## Resources, ownership and failure

Canonicalized workspaces/output roots must not overlap or alias. Existing
outputs, partial outputs, dangling output symlinks, depth/normal directory
aliases and output paths escaping their workspace reject. Shared read-only
image/sparse inputs are allowed; no shared mutable bitmap cache is introduced.
Inputs and output-directory topology must remain immutable for the run.

After model loading but BEFORE any decode task, actual image headers are opened
with Bitmap-compatible orientation/alpha settings. Unsupported layouts and
file/model dimension mismatches reject. Complete actual grayscale W*H retention
must fit each private cache, avoiding eviction of a returned bitmap reference.
Checked logical host/device envelopes must fit the declared job charges.
Header opens, codec scratch, pitch/array/context/stack allocations and allocator
behavior are not physical memory bounds; external operating allowances and
monitors remain required. Source CUDA XORWOW size48 is a pinned audit assumption,
not a portable ABI claim.

Two actual owner threads own separate controllers, inputs, caches, prepared
engines and streams. Preparation token=1; output token=1. PrepareOwned constructs
and drains the backend before ready. Both jobs reach ready before solver release.
`serial` means **SOLVER-serial**: B starts after A solver completion, while A
exports MAY overlap B's solver. It is not a strictly serial pipeline. Otherwise
both modes use identical owners, preparation, ordering, options and lifetimes.
Depth precedes normal export. Both owners retain resources until both exports
finish, then clean up on their owner threads.

Input failures publish cancellation before unwinding, consume every submitted
future AND ThreadPool's independent checker queue via Wait, then destroy/join
the pool while captured inputs remain alive. Pair-only output uses checked
stream writes, flush and close, preserving exact stock layout and partial files
on failure. It does not promise crash-durable storage. Stock Mat::Write is not
changed. Backend cleanup uncertainty terminates rather than freeing live work;
unknown retirement never releases credits. Proven retirement only marks cleanup;
Finish releases the whole ledger AFTER both joins, once, then rethrows any first
failure. Missing retirement proof cannot mask that originating failure.

## Event schema 1

Every record is a single Linux <=4096-byte write INCLUDING leading and trailing
newlines. Leading framing isolates unrelated partial stdout; other ordinary logs
remain present. Parse only complete `OWNED_PAIR_EVENT ` lines. Telemetry failure
fails the batch; it is never degraded into success or silently re-emitted.

All records carry:
`schema batch seq monotonic_ns pid emitter_tid owner_tid job reference
 device_index dispatch event`.

- batch = process PID plus steady-clock start nanoseconds.
- seq is unique contiguous global EMISSION order; monotonic_ns is nondecreasing.
- job is A, B or batch; batch owner_tid=0, reference/device=-1.
- Before owner/model resolution, per-job owner_tid may be0 and reference/device
  may be-1. At ready all are resolved; owner_tid is the actual Linux CUDA-owner
  thread, not necessarily the thread emitting a coordinator record.
- Coordinator records additionally carry coord_seq and transition_ns. Callbacks
  run outside the state mutex and can emit out of logical order across threads.
  Reconstruct state/token/dispatch order using coord_seq, NOT print order;
  transition_ns must not exceed that record's emission timestamp. Per-job
  metadata/sweeps/output phases retain causal emission order.

Cheap preflight emits validated with budgets, individual job/process charges,
I/O settings, affinity, driver_executable and common random/log settings. This
field is deliberately not named driver; external driver_version/GPU identity
remain separate runner bindings. Constructor emits admitted per job. The main
thread emits ledger_reserved with actual logical totals; constructor denial
emits admission_denied with zero reserved totals and launches no owners.

Per-job metadata is a tagged sequence: effective_metadata_begin (width, height,
ref_path_hex, sources), effective_source (ordinal/index/path_hex), 25
 effective_option records (option/value_hex), then effective_metadata_end
(option_count). Hex strings encode raw bytes without spaces; empty hex is valid.
Double serialization uses max_digits10. Every sweep has sweep_complete with
iteration1..5 and sweep1..4. Output phases are depth_begin/depth_end then
normal_begin/normal_end, enclosed by coordinator output_start/output_end.

Coordinator events also expose prep_start/prep_end, ready/both_ready,
solver_release/solver_start/solver_end, cancelled, retired and ledger_release.
Actual-owner cleanup_complete precedes retirement. Batch owners_joined precedes
ledger_release (reserved_host=0 reserved_device=0), then successful batch_finished.
Any cancellation/failure forbids successful batch_finished. Error paths attempt
batch_failed, but broken output transport/process termination may prevent it;
absence of full success closure always fails validation.

Success carries pair_total_ns (in-process setup-inclusive window from event
emitter creation before INI/preflight through cleanup/joins/ledger release) and
ready_window_ns (both-ready transition through later solver-end transition),
plus serial_scope=solver_only. Neither is GPU overlap or throughput evidence.
External process/container wall time remains separately measured.

## CPU evidence and remaining qualification

Run `PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover
-s benchmark/owned_patch_match -p 'test_*.py' -v` (one shell line).
Tests include actual coordinator/owner runner, original stock/owned source guards,
real header/cache and extracted ThreadPool fault behavior, actual stock-vs-checked
writer bytes and short-write/flush/close failures, negative mutants and framed
events. Host image tests use installed OpenImageIO2.4.17.0; writer tests use host
Eigen/glog/Boost. CLI syntax tests still use fake host interfaces. These are NOT
product configuration, compiler/ABI, CUDA ordering or numerical qualification.

Separate authorized gates remain: isolated product build/codegen; B canonical
stock equivalence; same-new-path SOLVER-serial versus pair numerical neutrality;
then bounded memory/overlap/throughput measurement. No timing/promotion conclusion
follows from source tests or earlier ONE-A evidence.
