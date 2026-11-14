# Developer raw-product guard (Story24 / S24.1a)

This stdlib-only **developer tool**, not an installed reconstruction launcher,
checks a ceased `colmap-reconstruct` workspace offline. It neither runs a solve
nor promotes a candidate. A hash baseline is an oracle for equality, not proof
that a reconstruction is correct. No photographs or golden bodies ship here.

```sh
export PYTHONPATH="$PRODUCT_CHECKOUT/python"
python3 -m unittest colmap_reconstruction.byte_golden_test
# Run the existing installed colmap-reconstruct separately, with seed 0,
# threads 1, one explicit GPU and --output-policy photo_and_geo.
# Retain both completed workspaces; do not overwrite either.
time python3 -m colmap_reconstruction.byte_golden capture \
  /preserved/run-a /private/execution-qualification.json /private/golden.json
time python3 -m colmap_reconstruction.byte_golden capture \
  /preserved/run-b /private/execution-qualification.json /private/candidate.json
python3 -m colmap_reconstruction.byte_golden compare \
  /private/golden.json /private/candidate.json
```

Capture exclusively creates its output, outside the workspace. Compare is
read-only, returning 0 for product equality (even if the package differs), 1 for
product/binding differences, and 2 for malformed/refused input. JSON reports
`product_equal`, `package_equal`, `binding_equal` and changed paths per domain.
Record reconstruction wall time separately from the offline `time` measurements.
The guard revision is **not** the frontend/native revision that executed the solve.

## Recorded execution qualification

Before execution, the executor must freeze/attest the actual installation and
retain this small JSON receipt. Capture binds its complete content, schema and
SHA256 of canonical JSON content (sorted keys, compact separators, UTF-8).
Each hash below is lowercase SHA256 of actual artifact bytes, not a label:

```json
{
  "schema_version": 1,
  "frontend_source_commit": "<40 hex Git commit>",
  "frontend_source_tree": "<40 hex Git tree>",
  "frontend_artifacts": [
    {"name": "bin/colmap-reconstruct", "sha256": "<64 hex>"},
    {"name": "share/colmap/python/colmap_reconstruction/cli.py", "sha256": "<64 hex>"},
    {"name": "<each other executed Python artifact>", "sha256": "<64 hex>"}
  ],
  "native_source_commit": "<40 hex actual native commit>",
  "native_elf_sha256": "<64 hex actual native ELF>",
  "libraries": [{"name": "<library identity>", "sha256": "<64 hex>"}],
  "hardware": {"cpu": "<recorded CPU facts>", "gpu": "<recorded UUID/name/CC/driver>"}
}
```

Library and frontend lists must be nonempty with unique names; include all loaded
libraries and executed frontend modules. The launcher hash is **not** `cli.py`'s
hash. Frontend and native source identities may intentionally differ. Both arms
must use the same frozen receipt/installation/hardware: there is no cross-bin or
cross-hardware override. Keep timestamps, workspace paths and observer records
outside this receipt. Hardware/library completeness and historical authenticity
are executor qualification responsibilities, not facts this offline tool can
retroactively prove. It verifies source/ELF against existing `planning/build.json`
and rehashes the recorded native executable if still accessible; an installation
need not survive machine closure. It does not demand live GPUs or a source tree.

## Existing metadata and complete membership

Retain root `result.json` and `planning/{intent,approved-plan,build,dense-plan,
cohort,sparse-model,extraction-inventory,resolution-intent}.json`. The guard reads
actual encoded input order/hashes/dimensions and actual resolved image/reference
membership, per-phase source order (including duplicates), depth bounds, requested
and effective options. It checks the cohort's reference/sources/options against
the richer native plan, not against historical preset source sequences.
Seed 0, threads 1, fixed-cohort backend and PHOTO retention are mandatory.
Historical native b693's Mapper `random_seed=-1` cannot qualify.

Mandatory regular scientific bodies are database.db, five model binaries
(cameras/images/points3D/rigs/frames) in the selected sparse model and dense model,
every registered undistorted image, each reference's depth+normal PHOTO and GEO
in cohort, dense GEO publications, and points.ply/points.ply.vis. Hardlink aliases
are separate publications, each fully hashed. All other files and directories
are inventoried too; unknown extras default scientific, never disappear.
Scientific file bytes have **no normalization or tolerance**.

## Scientific binding versus raw operational inventory

All publications retain relative name/type, regular length/SHA256, directory
membership or symlink target. Links must resolve inside the tree. Traversal,
unsupported types, dangling/escaping links, malformed/duplicate JSON members and
missing required bodies refuse. Use immutable, ceased, trusted local trees;
this is not an adversarial concurrent-writer sandbox or authenticity service.

`project.ini` is never dropped: raw bytes are inventoried operationally, while
all scalar values except these explicit operational fields bind scientifically:
root database_path/image_path/output_path/log_path; Mapper host_budget_bytes,
policy_host_bytes/reserve_host_bytes. Mapper seed/thread must be 0/1. Other INI
keys (including new ones) remain exact. The intent split excludes only
photos/workspace/colmap/yes/timeout_seconds, host/device/disk budget and reserve
byte fields, loader_scratch_host_bytes/process_host_bytes/process_device_bytes/
unknown_host_bytes/unknown_device_bytes/unknown_provenance. All other effective
approved options remain exact. Automatic requested GPU selection is resolved to
the actual approved GPU; no resource-driven numerical options are changed.

The V1 options wire has strict header, field order/type/token/length validation.
Only its final workspace/config encoded strings bind to validated per-arm graph
paths rather than raw absolute path bytes. Every other wire byte, including
hex double bits, remains exact. Graph destinations bind to validated relative
publication paths; consumed input hashes are checked against actual bodies.
The complete raw wires remain in the operational dense-plan file inventory.
This is a new versioned guard, not a rewrite of any historical MISMATCH contract.

Recognized operational paths are exactly:

- root command-<digits>.stdout/.stderr and result.json;
- planning/{intent,approved-plan,build,dense-plan,cohort,sparse-model,
  extraction-inventory,resolution-intent,photos,resolution-effective,
  resource-policy,multi-gpu-plan}.json;
- planning/{fusion.receipt,extraction-images.txt,extraction-dimensions.txt};
- planning/{resources,extraction-selection}-<digits>.json;
- project.ini at any depth; dedicated profiler/ subtree (if present).

All are still exhaustively hashed. Other paths, including dense stereo cfgs and
run-colmap scripts, stay scientific/raw. Outer packets/traces outside this root
are not part of this workspace package; inventory those separately. If unknown
operational files differ, fail closed and independently review their producer
and exact fields before changing classification. Do not blanket-exempt JSON,
logs, binaries or filename extensions. Log/timestamp/path differences can leave
scientific products equal but must never be reported as whole-package equality.

## Limits and acceptance

The lean CPU CI job runs the stdlib guard suite directly, without a native
installation or GPU. Coverage includes distinct relative publications sharing a
basename and same-length body swaps between them.

Synthetic unit mutations cover the guard, not an installed reconstruction pair.
Actual selected five-photo execution, complete new workspace inventory, usable
private evidence, measured wall times and fresh read-only review are separate
acceptance gates owned by the parent. Changed numerics, hardware, input/options,
producer paths or scientific schema require explicit baseline qualification and
appropriate larger workload validation; compare never updates a baseline.
