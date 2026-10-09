# Owned PatchMatch source prototype (units 1–3)

Base: `c6ab4f897c94f10b15bdecc442dcfce4688f1b76` (BSD-3-Clause COLMAP).
No external implementation code imported. This is an opt-in, CUDA-only,
**one-reference photo radius5/step1** source prototype, not a throughput result.
The original `patch_match_cuda.cu` is byte-identical to the base.

## API and optional validation entry

`PatchMatch(..., PatchMatchExecution::kOwnedCuda)` explicitly selects the owned
backend; omission remains stock. The same final per-instance selector on
`PatchMatchController` forwards through the existing `ProcessProblem` path.
Owned mode rejects unsupported configurations, multiple configured references,
and the already-completed-output skip. It snapshots input images and preserves
the actual reference/source vector order. The caller's original inputs may then
expire without invalidating getters. While the caller retains them, this adds
one host bitmap payload copy (Q bytes plus metadata), not free ownership.

The optional `colmap_owned_reference` target requires CUDA+MVS, not HIP, and is
controlled by the default-OFF **build option**
`OWNED_PATCH_MATCH_DRIVER_ENABLED`. There is no execution-mode CLI/environment
option. Its arguments are the existing stereo options directly (no
`patch_match_stereo` subcommand). `RunPatchMatchStereoWithExecution` shares the
existing OptionManager parser and controller implementation; no input loader or
parser is duplicated. Original exported entry functions retain their stock route.

Exactly once after successful owned backend construction it logs:

```
PATCH_MATCH_BACKEND actual=owned_cuda reference=<actual model index>
```

Require this marker, existing runtime Problem/options/depth prints, fresh output
paths, and each reference's independent oracle in any later validator. This
source has NOT yet passed a product build, CUDA compilation, GPU exactness,
resource/codegen, or timing gate. The original canonical binary is still the
control; adding a backend can change module footprint even with unchanged stock
kernel source. Same-new-path serial/pair neutrality and canonical-stock
equivalence are separate future gates. No pair driver exists here.

## Lifetime/stream contract

The owned backend supplies a 32-byte by-value calibration snapshot per launch;
no owned module camera writes. Kernels and called helper operations use the
instance's nonblocking stream, including the actual rotate kernel, PRNG init,
reference filter, layered copies, downloads and timer events. Original launch
geometry, RNG initialization, recurrence, arithmetic and texture conventions
remain unchanged. Stream copies initially block on their own stream to retain
safe pageable-host-buffer lifetimes; no transfer-overlap promise.

Destructor fencing is not enough for failed construction: the backend catches
partial initialization failures and drains before member destruction. Local
async owners/caller buffers are protected by `ScopedStreamCompletion` across
copy/filter/helper exceptions. A fence failure cannot certify safe destruction:
these fail-closed cleanup paths terminate rather than silently releasing live
resources and claiming success. No MPS lifecycle is involved.

Legacy helpers retain their bodies, new CUDA-only overloads are HIP-guarded, and
legacy/default-stream compiler behavior is unchanged. Allocation/free may still
serialize other streams. Pooling, allocator changes, pair dispatch, preparation/
output scheduling and worker-count calibration are explicitly deferred.

## CPU-only source/contract verification

```
python3 -m unittest discover -s benchmark/owned_patch_match -p 'test_*.py' -v
```

The suite checks canonical stock bytes, unchanged legacy helper tokens, exact
device algorithm tokens after reversing only camera plumbing, all four host
calibration rotations, execution rejection/default routing, actual wrapper input
snapshot constructor against fake input types, explicit stream call sites, and
actual stream/timer/constructor/destructor host code against fake CUDA APIs.
Three negative CPU mutants detect a legacy-stream event, missing constructor
fence and missing destructor fence. Fake CUDA and extracted host tests are
**not** CUDA ordering, device ABI, numerical exactness or spill/occupancy proof.
LSP is unavailable in this environment; tree-sitter-only header parsing does not
replace a compiler. Standalone g++ CPU tests do not link a CUDA runtime.

`derive_owned.py` records the one-time canonical sibling derivation and refuses
to overwrite existing files. Later edits must remain explicit reviewed diffs.
Full source/helper review is required before any separately authorized build.
