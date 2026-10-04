# B2c: remaining PatchMatch rotation storage reuse

Status: host-tested, CUDA/HIP compilation and GPU acceptance pending. This topic
is on the supplied B1/main base, not a silent PR2/B2a/B2b integration.

## Implementation and approved exceptions

All remaining maps opt into compact backing at initialization. Ordinary GpuMat
construction/Rotate and the ordinary PRNG constructor retain their operations.
Depth, three normal planes, reference intensity/sum/squared-sum and every mask
plane use B1's ascending rotate/copy schedule. A bounded typed workspace prefix
lets uint8 maps use the existing float workspace without a second allocation.
Reference texture rebinding, normal-vector rotation, calibration, scopes and final
synchronizations remain in their previous positions. No new streams or concurrency.

AoS PRNG cannot use one whole-state plane without increasing scratch from 4 to
48 bytes/pixel. Instead, the supervisor explicitly approved a linear-time raw
uint32 word-plane utility instead of a rectangular cycle-permutation entry point:

1. Gather and rotate one raw word per pixel into the existing disjoint workspace.
2. Scatter that word back with same-stream cudaMemcpy2DAsync, then advance word.
3. Retain the single final synchronization.

For 48-byte curandState this is 12 rotation launches + 12 scatters instead of one
whole-state rotation. It also removes the dead rotation-destination initializer
(supervisor-approved launch-order exception). The initial problem initializer is
unchanged, with the identical 32x16 launch configuration and id/seed coordinates.
Every destination state component is overwritten by the full W*H permutation;
no old destination initialization value is ever consumed. Each scratch word is
written exactly once before scatter reads it; scatter changes only that component,
so future components retain all their original bits. No existing numerical kernel
body changes. Rotation time/throughput is NOT measured here: strided scatter may
be expensive even if rotations are a small portion of total runtime.

## Integration interfaces

- B2b: use `GpuMat<State>::CreateCompact`, `MakeCompactView(newW,newH)`, and
  `RotateWithWordStorageReuse(output, workspace)` with a
  `rotation_workspace_->MakeCompactWorkspaceView<uint32_t>(newW,newH)` view.
  State must be trivially copyable, at least 4-byte aligned, and a multiple of 4
  bytes. The helper does not depend on curandState's size or fields. Adapt the
  initial B2b state initializer separately, preserving its approved semantics.
- B2a: mask reuse is per stored layer, with no knowledge of source-bit semantics.
  Byte-packed planes can use the byte workspace; uint32-packed planes can use a
  uint32 workspace and existing `RotateWithStorageReuse`. A layout that packs
  spatial rather than source dimensions requires separate integration review.
- HIP: adds the missing cudaMemcpy2DAsync-to-hipMemcpy2DAsync alias.

All shape, workspace capacity, shared ownership and address-overlap checks occur
before rotation submission. No device allocation is introduced by rotation.

## Expected eliminated replacement allocations

At W=3930, H=2210 (logical bytes; allocator padding excluded):

| Buffer | Bytes | GiB |
| --- | ---: | ---: |
| Mask, S=64 | 555859200 | 0.517684 |
| PRNG, sizeof(state)=48 | 416894400 | 0.388263 |
| Normal planes | 104223600 | 0.097066 |
| Depth | 34741200 | 0.032355 |
| Reference three maps | 78167700 | 0.072799 |

The persistent existing float workspace remains 34741200 bytes (0.032355 GiB).
No extra device scratch is added. These are per-buffer transient removals, NOT
additive peak savings: scopes differ, the lazy mask remains live, and reference
texture-array replacement/driver overhead are unchanged. B2a/B2b reduce the
underlying allocations independently. No capacity/7.1-GiB claim is made.

## Host evidence and reproduction

New test initially failed compilation because RotatedIndex/ForEachWord were
missing; the workspace-prefix test separately failed for missing MakeWorkspaceView.
After implementation both standalone suites passed:

```sh
mkdir -p evidence/b2c
for t in rotation_storage_plan rotation_word_plan; do
  g++ -std=c++17 -O2 -DNDEBUG -Wall -Wextra -Werror -pthread \
    -DCOLMAP_ROTATION_STORAGE_STANDALONE -Isrc \
    src/colmap/mvs/${t}_test.cc -o evidence/b2c/${t}-test
  evidence/b2c/${t}-test
  g++ -std=c++17 -O2 -DNDEBUG -Wall -Wextra -Werror -pthread \
    -fsanitize=address,undefined -fno-omit-frame-pointer \
    -DCOLMAP_ROTATION_STORAGE_STANDALONE -Isrc \
    src/colmap/mvs/${t}_test.cc -o evidence/b2c/${t}-san
  evidence/b2c/${t}-san
done
git diff --check
```

Results: `rotation storage tests: PASS` twice; `rotation word tests passed
(64 shape/type cases, four rotations each)` twice. New suite is also registered
with GTest/CMake. Coverage includes 1x1, 1xN, Nx1, square, asymmetric/odd sizes,
float raw bits (signed zero, infinities and NaN payloads), normal layers, byte
masks and source-packed byte/word mask representations, and opaque 48-, 16-,
24-byte state representations; each turn compares against an independent full
coordinate oracle, checks complete unique scratch writes and guards, and four
turns restore the original byte array. Production word schedule/index helpers
are used in the simulator. Workspace prefix capacity/ownership rejection and
B1's existing overlap/lifetime tests also run.

Source audit: existing PatchMatch numerical kernels, initial PRNG kernel, and
ordinary GpuMat::Rotate remain unchanged. No CUDA toolchain or configured CTest
build exists here; full CMake/CTest and device execution were NOT run. Opaque host
states are NOT actual curandState ABI/device tests. Required combined-branch gates:
CUDA/HIP build; real PRNG state-byte equivalence against ordinary rotation
(including ABI/padding review); seed-0 from-disk tile32 full 632 digests; actual
tracked/device-wide peak; rotation-heavy throughput and strided-scatter cost.
Review B2a/B2b integration separately before any acceptance claim.
