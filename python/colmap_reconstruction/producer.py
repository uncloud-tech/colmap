# SPDX-License-Identifier: BSD-3-Clause
"""In-process PHOTO V3 staged-plan producer; no CLI/disk transport or GPU work.

The historical memory_model PHOTO API remains the cf789105 model record. This
adapter regenerates that inventory (including per-owner pose textures) against
a verified per-build closure and emits the nine-field ancestry-aware binding. Input
jobs must be independent native effective snapshots, NOT decoded forecasts.
All numbers are engineering arithmetic, never a physical allocation bound.
"""
import copy
import hashlib
import re
import struct
from pathlib import Path

from . import model as m

FINGERPRINT = '9e749b59aabc5433f7e475b4e660a1a62ad8d4d5fda083fec8fc803165a78dda'
RETIRED_FINGERPRINTS = {
    'd26041fafe7860b2653bafe6702ca97043881b14d953097112bcb4b7bde0117e': 'retired-fingerprint-d26041fa',
    'e61ec37792d6dfb491eedba0dc980c99d3e75c85f5ce731f7942854039c3cfb0': 'retired-fingerprint-e61ec377',
}
LINEAGE = m.PHOTO_PHASE_LINEAGE_SHA
# Historical allocation citations identify the reviewed formula derivation.
PROVENANCE = {
    'allocation_status': 'Reviewed PHOTO closure; same-build identity is checked separately from arithmetic. No physical-fit certification.',
    'effective_preparation': 'optimized_photo_runner.h:188-207,227-231; optimized_photo_prepared.h:12-23 (effective options, owner/generation, engine guards)',
    'input_pixels': 'optimized_photo_runner.h:39-67; owned_pair_image.h:15-42',
    'bound_reservation': 'optimized_photo_reservation.h:40-49 (cache>=q, process.host>=cache only)',
    'decoded_bank': 'PRODUCER POLICY: reserve 2*q preparation peak, NOT a bound-mode requirement; patch_match.cc:1171-1179,1240-1245,1271-1277 local decoded images coexist with owned SnapshotImages copy :46-48,74-80; ../sensor/bitmap.cc:143-148 copies pixel data; after preparation only one private copy survives',
    'host_scratch': 'Conservative PRODUCER POLICY: max(staging,24*a), magnitude 24*a NOT derived from output conversion; normal_map.cc:23-34 moves the downloaded map; optimized_photo_reservation.h:24 retains the policy; staging t only corroborated by patch_match_cuda_owned.cu:1920-1940',
    'initialization': 'patch_match_cuda_owned.cu:1833-1846,2180-2224,2243-2259; owned_rotation_storage.h:100-120; gpu_mat_ref_image.cu:80-85 (73*a+12*s*a)',
    'steady': 'patch_match_cuda_owned.cu:2260-2272; owned_prng_boundary.h:64-86; owned_selection_scratch.h:11-25,57-69; gpu_mat_ref_image.cu:80-85 (50*a+20*s*a; filter=false)',
    'boundary': 'owned_prng_boundary.h:19-32; patch_match_cuda_owned.cu:2244-2259 (48*boundary)',
    'workspace': 'patch_match_cuda_owned.cu:2222-2224 (2 floats*max(width,height)*s)',
    'reference_textures': 'patch_match_cuda_owned.cu:1869-1877 (two uint8 CUDA arrays)',
    'source_texture': 'patch_match_cuda_owned.cu:1907-1971 (max_source_width*max_source_height*s uint8)',
    'pose_textures': 'patch_match_cuda_owned.cu:1561-1580,2070-2117; patch_match_cuda_owned.h: poses_texture_ (4 owner-member arrays * (4+9+3+3+12+12)*s floats)',
    'process': 'PRODUCER POLICY: inherited 2 GiB host/device OWNED_PROCESS_CHARGE_BYTES, not derived from consumer; plus one controller cache optimized_photo_reservation.h:45-48',
    'margin': 'PRODUCER POLICY: ceil(103*raw/100) once per disjoint term; optimized_photo_delivery.h:11-13,125-127 consumer adds none',
}


def _fail(message):
    raise ValueError('PHOTO delivery ' + message)


def _reject_retired_fingerprint(value, *, compiled=False):
    if value in RETIRED_FINGERPRINTS:
        prefix = 'compiled fingerprint mismatch: ' if compiled else 'fingerprint mismatch: '
        _fail(prefix + RETIRED_FINGERPRINTS[value])


def _refuse_model_only(value):
    if isinstance(value, dict) and value.get('api_scope') == 'model-only-ancestry':
        _fail('model-only ancestry object is not a production delivery')


def _fields(value, required, name):
    if not isinstance(value, dict) or not set(required) <= value.keys():
        _fail('malformed ' + name + ': expected mapping with ' + ', '.join(required))


def _sha(data):
    return hashlib.sha256(data).hexdigest()


# Exact field order at the reviewed build's optimized_photo_options.h:52-95
# Paths and effective options are bound to actual native exports.
_OPTION_FIELDS = (
    ('int', 'max_image_size'), ('string', 'gpu_index'),
    ('double', 'depth_min'), ('double', 'depth_max'), ('int', 'random_seed'),
    ('int', 'window_radius'), ('int', 'window_step'), ('double', 'sigma_spatial'),
    ('double', 'sigma_color'), ('int', 'num_samples'),
    ('bool', 'owned_distinct_source_cache'), ('bool', 'owned_tex_lookahead'),
    # Opt-in graph/pool bindings: emitted ONLY when true, so default V1 byte
    # content (thread mode, ordinary allocation) stays unchanged.
    ('bool', 'owned_cuda_graphs'), ('bool', 'owned_async_pool'),
    ('double', 'ncc_sigma'), ('double', 'min_triangulation_angle'),
    ('double', 'incident_angle_sigma'), ('int', 'num_iterations'),
    ('bool', 'geom_consistency'), ('double', 'geom_consistency_regularizer'),
    ('double', 'geom_consistency_max_cost'), ('bool', 'filter'),
    ('double', 'filter_min_ncc'), ('double', 'filter_min_triangulation_angle'),
    ('int', 'filter_min_num_consistent'), ('double', 'filter_geom_consistency_max_cost'),
    ('bool', 'write_consistency_graph'), ('bool', 'allow_missing_files'),
    ('int', 'num_threads'), ('double', 'cache_size'))


# Mirrors the native `if (options.owned_cuda_graphs) boolean(...)` emission: a
# false opt-in adds no line at all, which is what keeps thread-mode bytes stable.
_CONDITIONAL_TRUE_OPTIONS = frozenset({'owned_cuda_graphs', 'owned_async_pool'})


def optimized_photo_options(effective, workspace, config):
    """Serialize the EFFECTIVE native snapshot, never controller defaults.

    Byte-compatible with OptimizedPhotoOptions(effective, workspace, config)
    for this CUDA/non-HIP build; paths use existing canonical filesystem paths.
    Resolution of depth bounds/GPU/sigma remains the native supplier's job.
    """
    if not isinstance(effective, dict) or set(effective) != {key for _, key in _OPTION_FIELDS}:
        _fail('malformed effective options snapshot')
    if effective['geom_consistency'] is not False:
        _fail('effective options are not photometric')
    lines = [b'COLMAP_OPTIMIZED_PHOTO_OPTIONS_V1', b'build_cuda_enabled 1', b'build_hip_enabled 0']
    def string_line(key, value):
        if not isinstance(value, str):
            _fail('malformed effective options string ' + key)
        data = value.encode('utf-8')
        return b'opt_string ' + key.encode() + b' ' + str(len(data)).encode() + b' ' + data.hex().encode()
    for kind, key in _OPTION_FIELDS:
        value = effective[key]
        if key in _CONDITIONAL_TRUE_OPTIONS and not value:
            continue
        if kind == 'string':
            lines.append(string_line(key, value))
            continue
        if kind == 'bool':
            if type(value) is not bool:
                _fail('malformed effective options bool ' + key)
            text = str(int(value))
        elif kind == 'int':
            if type(value) is not int or not -(1 << 31) <= value < (1 << 31):
                _fail('malformed effective options int ' + key)
            text = str(value)
        else:
            if type(value) not in (int, float):
                _fail('malformed effective options double ' + key)
            try:
                text = struct.pack('>d', value).hex()
            except (OverflowError, struct.error):
                _fail('malformed effective options double ' + key)
        lines.append(('opt_' + kind + ' ' + key + ' ' + text).encode())
    try:
        for key, path in [('workspace', workspace), ('config', config)]:
            lines.append(string_line(key, str(Path(path).resolve(strict=True))))
    except (OSError, TypeError, ValueError):
        _fail('effective options paths must exist and resolve canonically')
    return b'\n'.join(lines) + b'\n'


def compiled_binding(identity):
    """Bind a same-ELF native identity to the reviewed formula closure.

    Commit is provenance, not a compatibility whitelist. No source tree or Git
    is consulted at runtime. Unknown allocation content always refuses.
    """
    if not isinstance(identity, dict):
        _fail('native build identity required')
    constants = {key: identity.get(key) for key in
                 ('lineage_sha', 'fingerprint', 'source_commit')}
    if not isinstance(constants['source_commit'], str) or not re.fullmatch(
            '[0-9a-f]{40}', constants['source_commit']):
        _fail('malformed native source_commit')
    if constants['lineage_sha'] != LINEAGE:
        _fail('lineage not required ancestry base')
    _reject_retired_fingerprint(constants['fingerprint'], compiled=True)
    if constants['fingerprint'] != FINGERPRINT:
        _fail('compiled fingerprint mismatch')
    return constants


def _job(job, constants):
    # Exact native dimensions, no uniform-image assumption. Lists preserve order.
    ref = m._photo_index('reference_image_idx', job['reference_image_idx'])
    indices = job['source_image_idxs']
    if not isinstance(indices, list) or not indices:
        _fail('per-reference mismatch: empty source order')
    for idx in indices:
        m._photo_index('source_image_idx', idx)
    if len(set(indices)) != len(indices) or ref in indices:
        _fail('per-reference mismatch: duplicate indices')
    dims = job['source_dimensions']
    if not isinstance(dims, list) or len(dims) != len(indices):
        _fail('per-reference dimensions count mismatch')
    all_dims = [[job['width'], job['height']], *dims]
    for shape in all_dims:
        if not isinstance(shape, list) or len(shape) != 2:
            _fail('per-reference dimensions malformed')
        for value in shape:
            m._photo_exact_int('actual dimension', value, 1)
    q = sum(m._mul(w, h) for w, h in all_dims)
    terms = m.photo_phase_private_terms(
        job['width'], job['height'], len(indices), input_pixels=q,
        max_source_width=max(w for w, h in dims),
        max_source_height=max(h for w, h in dims))
    cache = m.photo_phase_cache_bytes(job['cache_size_gib'])
    if cache < q:
        _fail('cache must retain complete input set')
    # Shared phase inventory already includes the independently derived
    # per-owner pose textures in BOTH phases. Do not charge them a second time.
    private = dict(host=m._add(m.margin_bytes(terms['decoded_input_bank_raw_bytes']),
                              m.margin_bytes(terms['host_scratch_raw_bytes'])),
                   device=m.margin_bytes(terms['device_raw_bytes']))
    process = m.photo_phase_process_charge(controllers=1, cache_size_gib=job['cache_size_gib'])
    options = job['options_bytes']
    snapshot_keys = ('effective_options', 'workspace', 'config')
    if any(key in job for key in snapshot_keys):
        _fields(job, snapshot_keys, 'effective options serialization inputs')
        if options != optimized_photo_options(job['effective_options'], job['workspace'], job['config']):
            _fail('prepared effective options mismatch')
    if not isinstance(options, bytes) or not options:
        _fail('options mismatch: expected nonempty exact native bytes')
    required = [b'opt_bool geom_consistency 0', b'opt_bool filter 0',
                b'opt_int max_image_size -1',
                b'opt_double cache_size ' + struct.pack('>d', float(job['cache_size_gib'])).hex().encode()]
    lines = options.splitlines()
    if not lines or lines[0] != b'COLMAP_OPTIMIZED_PHOTO_OPTIONS_V1':
        _fail('options malformed native encoding')
    if b'opt_bool geom_consistency 1' in lines:
        _fail('effective options are not photometric')
    for line in required:
        key = b' '.join(line.split()[:2]) + b' '
        if [v for v in lines if v.startswith(key)] != [line]:
            _fail('options unsupported inventory or cache mismatch')
    binding = dict(profile='photo-phase-v1', **constants, options_sha256=_sha(options),
                   process=process, private_charge=private,
                   reference_image_idx=ref, source_image_idxs=list(indices))
    effective = dict(options_bytes=options, process=copy.deepcopy(process),
                     private_charge=copy.deepcopy(private), reference_image_idx=ref,
                     source_image_idxs=list(indices))
    return binding, effective, dict(actual_dimensions=all_dims, terms=terms)


def build_delivery(*, exact_launch_bytes, jobs, build_identity):
    """Supply native PhotoPhaseV1DeliveryInputs fields plus a separate audit record.

    Embedding bridge maps only exact_launch_bytes/bindings/effective_jobs to C++
    (and supplies native ComputeSHA256). producer_record is never a wire field.
    jobs/options/dimensions come from the sealed native plan; no forecast parsing.
    Scope is unfiltered PHOTO (geom_consistency=false, filter=false). Native
    supplier must enforce those effective options; their bytes are bound here.
    Only the header and four inventory-sensitive lines are checked locally;
    this is NOT an exhaustive OptimizedPhotoOptions grammar validator. The
    trusted native supplier must generate the full encoding independently.
    """
    _refuse_model_only(jobs)
    constants = compiled_binding(build_identity)
    if not isinstance(exact_launch_bytes, bytes) or not exact_launch_bytes:
        _fail('missing exact launch bytes')
    if not isinstance(jobs, list) or not jobs:
        _fail('missing effective jobs')
    for job in jobs:
        _refuse_model_only(job)
        _fields(job, ('reference_image_idx', 'source_image_idxs', 'width', 'height',
                      'source_dimensions', 'options_bytes', 'cache_size_gib'), 'native job')
        m._photo_index('reference_image_idx', job['reference_image_idx'])
        m.photo_phase_cache_bytes(job['cache_size_gib'])
    if len({j['reference_image_idx'] for j in jobs}) != len(jobs):
        _fail('count or order mismatch: duplicate references')
    if len({j['cache_size_gib'] for j in jobs}) != 1:
        _fail('controller cache mismatch')
    triples = [_job(j, constants) for j in jobs]
    record = dict(**constants, lineage_semantics='required ancestry base, NOT hashed revision',
                  model_source_commit=m.PHOTO_PHASE_SOURCE_COMMIT,
                  model_fingerprint=m.PHOTO_PHASE_SOURCE_FINGERPRINT,
                  provenance=copy.deepcopy(PROVENANCE),
                  source_prefix='src/colmap/mvs/', ancestry_verified_by='native build-time committed closure generator',
                  attestation_limit='Native equality checks prove consistency, NOT charge derivation; provenance and actual dimensions are the derivation evidence',
                  jobs=[t[2] for t in triples], native_jobs=copy.deepcopy(jobs),
                  calibration_certified=False, physical_fit_certified=False)
    return dict(exact_launch_bytes=exact_launch_bytes, bindings=[t[0] for t in triples],
                effective_jobs=[t[1] for t in triples], producer_record=record)


def validate_delivery(delivery, *, exact_launch_bytes, build_identity):
    """Mirror native predicates AND regenerate canonical charges from audit inputs."""
    _refuse_model_only(delivery)
    _fields(delivery, ('exact_launch_bytes', 'bindings', 'effective_jobs', 'producer_record'), 'outer delivery')
    if not isinstance(delivery['exact_launch_bytes'], bytes):
        _fail('malformed exact launch bytes')
    for key in ('bindings', 'effective_jobs'):
        if not isinstance(delivery[key], list):
            _fail('malformed ' + key + ': expected list')
    constants = compiled_binding(build_identity)
    if delivery['exact_launch_bytes'] != exact_launch_bytes:
        _fail('launch bytes mismatch')
    bindings, jobs = delivery['bindings'], delivery['effective_jobs']
    if not bindings or not jobs:
        _fail('missing')
    if len(bindings) != len(jobs):
        _fail('count or order mismatch')
    record = delivery['producer_record']
    _fields(record, ('native_jobs', 'source_commit', 'fingerprint', 'lineage_sha'), 'producer record')
    for key, value in constants.items():
        if record.get(key) != value:
            _fail('record ' + key + ' mismatch')
    canonical = build_delivery(exact_launch_bytes=exact_launch_bytes,
                               jobs=record['native_jobs'], build_identity=build_identity,
                               )
    if len(bindings) != len(canonical['bindings']):
        _fail('count or order mismatch')
    for i, (binding, job) in enumerate(zip(bindings, jobs)):
        if not isinstance(binding, dict):
            _fail('malformed binding: expected mapping')
        _fields(job, ('options_bytes', 'reference_image_idx', 'source_image_idxs'), 'effective job')
        if set(binding) != m.PHOTO_PHASE_DELIVERY_KEYS:
            _fail('missing or unknown binding keys (source_commit required)')
        if binding['profile'] != 'photo-phase-v1':
            _fail('unknown profile')
        for name, size in [('lineage_sha', 40), ('fingerprint', 64), ('options_sha256', 64)]:
            if not isinstance(binding[name], str) or not re.fullmatch('[0-9a-f]{%d}' % size, binding[name]):
                _fail('malformed ' + name)
        if not binding['source_commit']:
            _fail('missing/empty source_commit')
        if binding['source_commit'] != constants['source_commit']:
            _fail('source_commit mismatch with fingerprint revision: source_commit-revision-mismatch')
        _reject_retired_fingerprint(binding['fingerprint'])
        if binding['fingerprint'] != constants['fingerprint']:
            _fail('fingerprint mismatch')
        if binding['lineage_sha'] != constants['lineage_sha']:
            _fail('lineage not required ancestry base')
        for received in (binding, job):
            m._photo_index('reference_image_idx', received['reference_image_idx'])
            indices = received['source_image_idxs']
            if not isinstance(indices, list) or not indices:
                _fail('per-reference mismatch: malformed source indices')
            for idx in indices:
                m._photo_index('source_image_idx', idx)
        if binding['reference_image_idx'] != job['reference_image_idx'] or binding['reference_image_idx'] != canonical['bindings'][i]['reference_image_idx']:
            _fail('count or order mismatch')
        if binding['source_image_idxs'] != job['source_image_idxs'] or binding['source_image_idxs'] != canonical['bindings'][i]['source_image_idxs']:
            _fail('per-reference mismatch')
        if not isinstance(job['options_bytes'], bytes) or not job['options_bytes'] or binding['options_sha256'] != _sha(job['options_bytes']):
            _fail('options mismatch')
        for charge in ('process', 'private_charge'):
            for obj in (binding, job):
                pair = obj.get(charge)
                if not isinstance(pair, dict) or set(pair) != {'host', 'device'} or any(type(pair[k]) is not int or not 0 < pair[k] <= (1 << 64) - 1 for k in ('host', 'device')):
                    _fail('absent or zero charges')
            if binding[charge] != job[charge] or binding[charge] != canonical['bindings'][i][charge]:
                _fail('canonical charge mismatch')
        if job['options_bytes'] != canonical['effective_jobs'][i]['options_bytes']:
            _fail('options mismatch with native snapshot')
    if record != canonical['producer_record']:
        _fail('producer record/provenance mismatch')
    return True


def runner_delivery(**kwargs):
    """Callable for an embedding bridge; snapshot by value and bind exact bytes."""
    sealed = build_delivery(**kwargs)
    def supply(exact_launch_bytes):
        if exact_launch_bytes != sealed['exact_launch_bytes']:
            _fail('launch bytes mismatch')
        return copy.deepcopy(sealed)
    return supply
