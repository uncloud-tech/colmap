# SPDX-License-Identifier: BSD-3-Clause
"""Canonical resource arithmetic extracted from photogram 0b84800e.

Engineering estimates, not physical RSS/VRAM bounds. No provider APIs.
"""
import math
import os
import re

MODEL_VERSION = '1.4.0'

OWNED_FORMULA_SOURCE_COMMIT = '3b9099bb950016a741e37c12ac6a46f6c60f4028'

OWNED_NORMAL_BYTES_PER_TEXEL = 12      # 3 float channels

OWNED_GRAY_DEPTH_BYTES_PER_TEXEL = 5   # 1 grayscale + 4 float depth

OWNED_REFERENCE_BYTES_PER_TEXEL = 17   # 5 + 12 for reference layers

OWNED_UPLOAD_STAGING_BYTES_PER_PADDED_TEXEL = 5

OWNED_DEVICE_BYTES_PER_TEXEL = 5

OWNED_DEVICE_PER_LAYER_BYTES = 256

OWNED_DEVICE_FIXED_BYTES = 4096

OWNED_SOURCE_IDX_ENTRY_BYTES = 2 * 4   # 2 * sizeof(int) per source idx entry

OWNED_PROCESS_CHARGE_BYTES = 2 * 1024 ** 3   # jit-normal-memory-prep: 2147483648 host/device (= 2 * GIB)

OWNED_PLANNING_PAGE_BYTES = 4096

OWNED_PLANNING_PER_INPUT_EXTRA_BYTES = 512

OWNED_PLANNING_PER_LAYER_PAGES = 4

OWNED_NATIVE_ABISIZEOF = {
    'BankJob': 336, 'BankInput': 384, 'BankLayer': 544,
    'Image': 288, 'FileIdentity': 96,
}

class OwnedPhaseUnsupported(Exception):
    """Requested owned phase has no model terms; never fall back to stereo coeffs."""

def _add(a, b):
    a, b = int(a), int(b)
    if a < 0 or b < 0 or b > (1 << 64) - 1 - a:
        raise OverflowError('Resident bank byte sum overflow')
    return a + b

def _mul(a, b):
    a, b = int(a), int(b)
    if a < 0 or b < 0 or (a and b > ((1 << 64) - 1) // a):
        raise OverflowError('Resident bank byte product overflow')
    return a * b

def margin_bytes(raw):
    """Single 3% forecast margin, rounded up (C++ Margin integer semantics),

    mathematically equal to ceil(1.03 * raw); not a physical allocation bound.
    """
    raw = int(raw)
    if raw < 0:
        raise ValueError('margin input must be nonnegative')
    return _add(raw, _add(_mul(raw // 100, 3), (_mul(raw % 100, 3) + 99) // 100))

def _f32(x):
    """Round through IEEE binary32 exactly as the C++ float arithmetic does."""
    import struct as _struct
    return _struct.unpack('<f', _struct.pack('<f', float(x)))[0]

FLOAT32_EPSILON = 1.1920928955078125e-07

def _gaussian_filter_bytes(sigma):
    """VLFeat _vl_new_gaussian_filter charge (imopv.c): 4*(2*ceil(3*sigma)+1)."""
    samples = math.ceil(float(sigma) * 3.0)   # vl_ceil_d semantics (plain ceil)
    return _mul(_add(_mul(samples, 2), 1), 4)

def read_resident_map_header(path):
    """Parse one resident map `&`-header exactly like owned_input_bank.cc MapBytes.

    Returns (width, height, channels, payload_bytes); refuses nonpositive
    dimensions, zero/negative channels, or a payload that does not fit in the
    actual file (truncated).
    """
    size = os.path.getsize(path)
    with open(path, 'rb') as fh:
        head = fh.read(96)
    match = re.match(rb'\s*(\d+)\s*&\s*(\d+)\s*&\s*(\d+)\s*&', head)
    if not match:
        raise ValueError('Invalid resident map header: %s' % path)
    w, h, d = (int(x) for x in match.groups())
    if not w or not h or d <= 0:
        raise ValueError('Invalid resident map header: %s' % path)
    payload = _mul(_mul(_mul(w, h), d), 4)
    if len(head) < 1:
        raise ValueError('Invalid resident map header: %s' % path)
    header_len = match.end()
    if header_len + payload > size:
        raise ValueError('Truncated resident map: %s' % path)
    return w, h, d, payload

def _owned_layer_payload(dims, channels, path=None):
    """Payload (w, h, bytes) from an explicit dims pair or an actual map file."""
    if path:
        w, h, d, payload = read_resident_map_header(path)
        if d != channels or (dims and (w, h) != tuple(dims)):
            raise ValueError(
                'resident map file %s does not match the descriptor dims' % path)
        return w, h, payload
    if not dims:
        raise ValueError('resident payload dims or path required')
    w, h = (int(x) for x in dims)
    if w <= 0 or h <= 0:
        raise ValueError('resident payload dims must be positive')
    return w, h, _mul(_mul(w, h), channels * 4)

def _owned_input_identity(input_desc, key):
    """Verified layer identity per the explicit descriptor identity contract.

    Mirrors native Resolve, which deduplicates on the canonical resolved path
    triple IN ROLE ORDER (image, depth, normal) plus metadata (never on
    invented dimensions alone, and never role-agnostic):
      * resolved file paths: all three of image_path/depth_path/normal_path
        (or one `resolved_paths` triple, validated to have exactly 3 entries
        BEFORE any resolution) -> identity = the CANONICAL resolved path
        triple in (image, depth, normal) role order, strictly resolved
        (os.path.realpath(strict=True) matches std::filesystem::canonical:
        a claimed verified path that does not exist is refused, and role-#
        swapped files never merge even with equal headers);
      * `identity`: an explicit caller-certified identity token (the caller
        guarantees it labels one native file triple only);
      * otherwise the layer_key is an IMPLICIT UNVERIFIED caller assertion
        (convenience for the admission-free arithmetic estimator only; a
        repeated key there asserts — without verification — the same native
        file): the admission gate refuses it.
    Returns (identity_tag, contract_mode) where the paths tag is an ordered
    triple so a depth/normal role swap is a DIFFERENT identity.
    """
    triple = input_desc.get('resolved_paths')
    image_path = input_desc.get('image_path')
    depth_path = input_desc.get('depth_path')
    normal_path = input_desc.get('normal_path')
    if triple:
        if not isinstance(triple, (list, tuple)) or len(triple) != 3:
            raise ValueError(
                'ambiguous resident identity: `resolved_paths` must be the '
                'ordered (image, depth, normal) triple of exactly 3 paths; '
                'got %s' % (len(triple) if isinstance(triple, (list, tuple))
                            else type(triple).__name__))
        image_path, depth_path, normal_path = (
            str(triple[0]), str(triple[1]), str(triple[2]))
    provided = [p for p in (image_path, depth_path, normal_path) if p]
    if provided:
        if len(provided) != 3:
            raise ValueError(
                'ambiguous resident identity: give the full resolution-'
                'ordered image/depth/normal path triple (or none of them); '
                'got %d of 3' % len(provided))
        resolved = []
        for role, path in (('image', image_path), ('depth', depth_path),
                           ('normal', normal_path)):
            try:
                resolved.append(os.path.realpath(str(path), strict=True))
            except (OSError, ValueError) as exc:
                raise ValueError(
                    'claimed verified resolved identity %s path does not '
                    'resolve (%s); a verified resolved_paths identity must '
                    'exist (std::filesystem::canonical semantics)'
                    % (role, exc))
        # STRICTLY role-ordered (image, depth, normal): swapping the depth
        # and normal files is a different identity even with equal headers.
        return ('paths',) + tuple(resolved), 'resolved_paths'
    token = input_desc.get('identity')
    if token:
        return ('id', str(token)), 'identity_token'
    return ('key', str(key)), 'layer_key_token'

def _owned_resolve(jobs):
    """Layer/ownership layout resolution mirroring owned_input_bank.cc Resolve.

    Validation mirrors the C++ invalid_argument paths: complete geometry inputs
    (nonempty distinct sources, reference excluded), inputs exactly covering the
    required set, one max_image_size per job, layers deduplicated by the
    VERIFIED identity token (canonical resolved path triple / caller-certified
    token / key token under the documented identity contract) with conflicting
    keys or metadata refused, positive dims and in-range payload dims.
    Retained terms are computed in a second pass with the final is_reference
    flags, exactly like the native Resolve.
    """
    if not jobs:
        raise ValueError('Empty resident bank')
    layers = []       # first-occurrence order (native Resolve semantics)
    layer_by_key = {}
    layer_by_identity = {}
    identity_modes = set()
    key_aliases_merged = 0
    references_count = 0
    unverified_identity_layers = 0
    total_inputs = 0
    total_sources = 0
    per_job_inputs = []
    per_job_images = []
    images_count_defaulted = False
    for job in jobs:
        if not isinstance(job, dict):
            raise ValueError('owned job must be a descriptor dict')
        ref = job.get('reference_image_idx')
        sources = list(job.get('source_image_idxs') or [])
        inputs = list(job.get('inputs') or [])
        if not sources or sources.count(ref) or len(set(sources)) != len(sources):
            raise ValueError('Resident bank requires complete geometry inputs')
        required = set(sources) | {ref}
        seen = set()
        for image_idx in (i.get('image_idx') for i in inputs):
            if image_idx not in required or image_idx in seen:
                raise ValueError('Duplicate or incomplete resident input mapping')
            seen.add(image_idx)
        if len(inputs) != len(required):
            raise ValueError('Duplicate or incomplete resident input mapping')
        input_max = {i.get('max_image_size') for i in inputs}
        if len(input_max) != 1:
            raise ValueError('Resident job/input metadata mismatch')
        job_max = input_max.pop()
        if job_max is None:
            raise ValueError('Resident job/input metadata mismatch')
        total_inputs += len(inputs)
        total_sources += len(sources)
        per_job_inputs.append(len(inputs))
        # The native metadata term bills each job's ENTIRE images vector (the
        # loader copies the full model image list), which can be larger than
        # the required resident inputs. Descriptors carry `images_count`
        # explicitly; without it the term is only defaulted in modes that do
        # not read it, never in native_sizeof mode.
        if job.get('images_count') is not None:
            images_count = int(job['images_count'])
            minimum = max(1 + len(sources), len(inputs),
                          1 + max(i['image_idx'] for i in inputs))
            if images_count < minimum:
                raise ValueError(
                    'job.images_count %d is smaller than the images/inputs it '
                    'must contain (>= %d)' % (images_count, minimum))
        else:
            images_count = len(inputs)
            images_count_defaulted = True
        per_job_images.append(images_count)
        for input_desc in inputs:
            key = input_desc.get('layer_key')
            if not key:
                raise ValueError('Invalid resident input metadata')
            # Identity contract first (no file reads needed): partial or
            # ambiguous identities are refused before any header parsing.
            identity, identity_mode = _owned_input_identity(input_desc, key)
            identity_modes.add(identity_mode)
            w = int(input_desc.get('width') or 0)
            h = int(input_desc.get('height') or 0)
            if w <= 0 or h <= 0:
                raise ValueError('Invalid resident input metadata')
            # C++ order: the texel product overflows uint64 BEFORE the
            # bitmap-dimension-range check can convert it to a range error.
            _mul(w, h)
            if (w > (1 << 31) - 1) or (h > (1 << 31) - 1):
                raise ValueError('Resident image exceeds bitmap dimension range')
            spec = input_desc.get('image_spec')
            if not spec or len(spec) != 3 or any(int(x) <= 0 for x in spec):
                raise ValueError('Unreadable resident image header dims')
            depth_path = input_desc.get('depth_path')
            normal_path = input_desc.get('normal_path')
            dw, dh, dbytes = _owned_layer_payload(
                input_desc.get('depth_payload'), 1, depth_path)
            nw, nh, nbytes = _owned_layer_payload(
                input_desc.get('normal_payload'), 3, normal_path)
            # Binding: dedup by the verified identity token, with the user
            # key bound 1:1. A repeated key that labels a DIFFERENT verified
            # identity is REFUSED (the equal-header/different-file
            # counterexample: collapsing would undercharge retained payloads
            # and device layers instead of failing closed). The same
            # identity under a different label is merged onto the canonical
            # first key (native path-triple dedup; alias-safe: symlinks and
            # spelling variants canonicalize to the same identity).
            layer = layer_by_key.get(key) or layer_by_identity.get(identity)
            if layer is not None:
                if layer['identity'] != identity:
                    raise ValueError(
                        'layer key %r and %r label conflicting verified '
                        'identities; bind one layer_key per native identity '
                        '(resolved path triple + metadata), not dims alone'
                        % (key, layer['layer_key']))
                key_aliases_merged += 1 if key != layer['layer_key'] else 0
                conflicts = []
                if (layer['width'], layer['height']) != (w, h):
                    conflicts.append('dims')
                if layer['max_image_size'] != job_max:
                    conflicts.append('max_image_size')
                if tuple(layer['image_spec']) != tuple(spec):
                    conflicts.append('image_spec')
                if layer['depth_payload'][:2] != (dw, dh):
                    conflicts.append('depth_payload')
                if layer['normal_payload'][:2] != (nw, nh):
                    conflicts.append('normal_payload')
                if conflicts:
                    raise ValueError(
                        'resident layer identity conflict for %r: %s describe '
                        'different files; bind one layer_key per verified '
                        'native identity (resolved path triple + metadata)'
                        % (key, ', '.join(conflicts)))
            else:
                layer = {'layer_key': key, 'identity': identity,
                         'identity_mode': identity_mode,
                         'width': w, 'height': h,
                         'max_image_size': job_max, 'image_spec': tuple(spec),
                         'depth_payload': (dw, dh, dbytes),
                         'normal_payload': (nw, nh, nbytes),
                         'is_reference': False}
                layer_by_key[key] = layer
                layer_by_identity[identity] = layer
                layers.append(layer)
                if identity_mode == 'layer_key_token':
                    unverified_identity_layers += 1
            if input_desc.get('image_idx') == ref:
                if not layer['is_reference']:
                    layer['is_reference'] = True
                    references_count += 1
    # Native Resolve computes the retained terms in a SECOND pass over the
    # resolved layers, with the final is_reference flags (a layer first seen
    # as a source and later promoted must not undercount its 12 B/texel
    # normal): retained 17(ref)/5(source) and the bounded normal removal.
    normal_retained_once = 0
    retained = 0
    for layer in layers:
        area = _mul(layer['width'], layer['height'])
        retained = _add(retained, _mul(area, OWNED_REFERENCE_BYTES_PER_TEXEL
                                       if layer['is_reference'] else
                                       OWNED_GRAY_DEPTH_BYTES_PER_TEXEL))
        if layer['is_reference']:
            normal_retained_once = _add(normal_retained_once,
                                        _mul(area, 12))
    return {'layers': layers, 'references_count': references_count,
            'retained_bytes': retained,
            'normal_retained_once': normal_retained_once,
            'total_inputs': total_inputs, 'total_sources': total_sources,
            'per_job_inputs': per_job_inputs,
            'per_job_images': per_job_images,
            'images_count_defaulted': images_count_defaulted,
            'identity_contract_modes': sorted(identity_modes),
            'identity_key_aliases_merged': key_aliases_merged,
            'unverified_identity_layers': unverified_identity_layers}

def _owned_metadata(jobs, resolve, mode):
    """Metadata allowance: C++ structure, per-entry allowance selected by mode.

    'planning_envelope' (default) uses engineering planning allowances
    (4096 B pages; native_sizeof_measured
    false). 'native_sizeof' bills the probed C++ struct sizes of the qualified
    revision instead (ABI-specific).
    """
    layers = len(resolve['layers'])
    if mode == 'planning_envelope':
        page = OWNED_PLANNING_PAGE_BYTES
        metadata = _mul(len(jobs), page)
        metadata = _add(metadata, _mul(_mul(len(jobs), layers), page))
        metadata = _add(metadata, _mul(resolve['total_inputs'],
                                       page + OWNED_PLANNING_PER_INPUT_EXTRA_BYTES))
        metadata = _add(metadata, _mul(resolve['total_sources'],
                                       OWNED_SOURCE_IDX_ENTRY_BYTES))
        metadata = _add(metadata, _mul(layers, _mul(
            OWNED_PLANNING_PER_LAYER_PAGES, page)
            + OWNED_PLANNING_PER_INPUT_EXTRA_BYTES))
        return metadata, 'planning_envelope'
    if mode == 'native_sizeof':
        s = OWNED_NATIVE_ABISIZEOF
        metadata = _mul(len(jobs), s['BankJob'])
        # Full job.images vectors, not only the required resident inputs
        # (owned_input_bank.cc bills job.images.size(); review 2026-10-14).
        metadata = _add(metadata, _mul(sum(resolve['per_job_images']),
                                       s['Image']))
        if resolve['images_count_defaulted']:
            raise ValueError("metadata_mode 'native_sizeof' bills each job's "
                             'full images vector; descriptors must carry an '
                             'explicit images_count (no silent default)')
        metadata = _add(metadata, _mul(resolve['total_inputs'],
                                       s['BankInput'] + 512))
        metadata = _add(metadata, _mul(resolve['total_sources'],
                                       2 * 4))
        metadata = _add(metadata, _mul(layers, s['BankLayer']
                                       + 3 * s['FileIdentity'] + 512))
        return metadata, 'native_sizeof'
    raise ValueError("metadata_mode must be 'planning_envelope' or 'native_sizeof'")

def estimate_owned_bank(jobs, mode='eager', unclassified_allowance_bytes=0,
                        metadata_mode='planning_envelope'):
    """owned_input_bank.cc Estimate / EstimateBounded over verified descriptors.

    Eager raw = retained (17/5 B per texel) + metadata allowance + serial
    staging (max over layers of decode*4 + depth + normal + 34*area, floored by
    the padded 5*texel upload max). Bounded removes the retained reference
    normals (12 B/texel) EXACTLY ONCE from the raw total before the same single
    ceil(1.03) margin; every other term (in particular no construction/decode
    staging allowance) is attributed to normals or removed. `unclassified_
    allowance_bytes` is an explicit caller input only — there is no default and
    no module constant, because its provenance is unknown (measured gap,
    prior memory.stat unavailable).
    """
    if mode not in ('eager', 'bounded'):
        raise ValueError("mode must be 'eager' or 'bounded'")
    if not isinstance(unclassified_allowance_bytes, int) \
            or unclassified_allowance_bytes < 0:
        raise ValueError('unclassified_allowance_bytes must be a nonnegative '
                         'EXPLICIT integer input with recorded provenance')
    resolve = _owned_resolve(jobs)
    layers = resolve['layers']
    retained = resolve['retained_bytes']
    staging = 0
    width = height = 0
    for layer in layers:
        w, h = layer['width'], layer['height']
        width = max(width, w)
        height = max(height, h)
        area = _mul(w, h)
        sw, sh, nch = layer['image_spec']
        decode = _mul(_mul(sw, sh), nch + 1)
        dw, dh, dbytes = layer['depth_payload']
        nw, nh, nbytes = layer['normal_payload']
        staging = max(staging, _add(_add(_mul(decode, 4), dbytes),
                                    _add(nbytes, _mul(area, 34))))
    padded = _mul(_mul(width, height), len(layers))
    # Decode and upload are sequential (C++), so their temporary allowances
    # are a maximum, not a sum; the padded 5-bytes-per-texel upload staging is
    # the final max branch (staging = max(staging, 5 * padded_area)).
    staging = max(staging, _mul(_mul(width, height),
                                OWNED_UPLOAD_STAGING_BYTES_PER_PADDED_TEXEL))
    upload = _mul(_mul(width, height),
                  OWNED_UPLOAD_STAGING_BYTES_PER_PADDED_TEXEL)
    metadata, meta_label = _owned_metadata(jobs, resolve, metadata_mode)
    raw = _add(_add(retained, metadata), staging)
    raw_with_allowance = _add(raw, unclassified_allowance_bytes) \
        if unclassified_allowance_bytes else raw
    if raw_with_allowance < resolve['normal_retained_once']:
        raise OverflowError('Resident bank bounded raw underflow')
    bounded_raw = raw_with_allowance - resolve['normal_retained_once']
    device_raw = _add(_add(_mul(padded, OWNED_DEVICE_BYTES_PER_TEXEL),
                           _mul(len(layers), OWNED_DEVICE_PER_LAYER_BYTES)),
                      OWNED_DEVICE_FIXED_BYTES)
    eager_charge = margin_bytes(raw_with_allowance)
    result = {
        'model_version': MODEL_VERSION,
        'owned_phase_version': '1.0.0',
        'formula_source_commit': OWNED_FORMULA_SOURCE_COMMIT,
        'references': resolve['references_count'],
        'layers': len(layers), 'total_inputs': resolve['total_inputs'],
        'retained_12q_reference_normal_bytes': resolve['normal_retained_once'],
        'retained_bytes': retained,
        'retained_gray_depth_bytes': retained - resolve['normal_retained_once'],
        'staging_bytes': staging,
        'upload_staging_included': upload,
        'metadata_bytes': metadata,
        'images_count_defaulted': resolve['images_count_defaulted'],
        'identity_contract_modes': resolve['identity_contract_modes'],
        'identity_key_aliases_merged': resolve['identity_key_aliases_merged'],
        'unverified_identity_layers': resolve['unverified_identity_layers'],
        'metadata_mode': meta_label,
        'unclassified_allowance_bytes': unclassified_allowance_bytes,
        'unclassified_allowance_provenance': (
            'explicit caller input; classification unknown' if
            unclassified_allowance_bytes else 'none provided (native Estimate parity)'),
        'device_raw_bytes': device_raw,
        'device_charge_bytes': margin_bytes(device_raw),
        'margin_policy': 'single ceil(1.03) margin applied once per disjoint raw term; '
                         'the 3% is charged once; totals never re-margined',
    }
    if mode == 'eager':
        result['raw_bytes'] = raw_with_allowance
        result['host_charge_bytes'] = eager_charge
    else:
        result['raw_bytes'] = bounded_raw
        result['host_charge_bytes'] = margin_bytes(bounded_raw)
    return result

def _owned_downsize_layer(layer):
    """Whether Downsize rescales this layer's normal per the C++ condition."""
    w_target, h_target = layer['width'], layer['height']
    nw, nh, _ = layer['normal_payload']
    return layer['max_image_size'] > 0 and (nw > w_target or nh > h_target)

def _owned_transient_bytes(layer):
    """One in-flight bounded load peak, mirroring owned_input_bank.cc.

    No rescale: the normal file payload only. When rescaling: payload +
    max(24a, 12a + 8S + input-derived VLFeat gaussian filters) — shrink
    replacement overlap vs the two full-source smoothing temps; never payload
    alone.
    """
    area = _mul(layer['width'], layer['height'])
    owner = _mul(area, OWNED_NORMAL_BYTES_PER_TEXEL)
    nw, nh, payload = layer['normal_payload']
    if not _owned_downsize_layer(layer):
        return payload
    src_texels = _mul(nw, nh)
    scale_x = _f32(_f32(nw) / _f32(layer['width']))
    scale_y = _f32(_f32(nh) / _f32(layer['height']))
    sigma_x = _f32(max(FLOAT32_EPSILON, _f32(0.5 * (scale_x - 1.0))))
    sigma_y = _f32(max(FLOAT32_EPSILON, _f32(0.5 * (scale_y - 1.0))))
    filters = (_gaussian_filter_bytes(sigma_x)
               if sigma_x == sigma_y else
               _add(_gaussian_filter_bytes(sigma_x),
                    _gaussian_filter_bytes(sigma_y)))
    shrink = _add(payload, _mul(area, 24))     # payload + owner + owner
    smoothing = _add(_add(payload, owner),
                     _add(_mul(src_texels, 8), filters))
    return max(shrink, smoothing)

def estimate_owned_normal_allowance(jobs, max_future):
    """owned_input_bank.cc EstimateNormalAllowance (bounded FIFO contract).

    owner_bytes  = 12 * A (max reference-layer area): each active owner job's
                   distinct normal holder. The active-owner peak per job is
                   36*A = 24*A private scratch + 12*A distinct normal.
    future_bytes = max_future * owner (max_future restricted to 1 or 2).
    transient_bytes = max over reference layers of the checked load peak
                   (payload + max(24a, 12a+8S+filters) when rescaling).
    Never re-added or re-margined into a bank forecast; charged exactly once.
    """
    if max_future not in (1, 2):
        raise ValueError('bounded normal lookahead supports max_future of 1 or 2 only')
    resolve = _owned_resolve(jobs)
    owner = transient = 0
    reference_layers = 0
    for layer in resolve['layers']:
        if not layer['is_reference']:
            continue
        reference_layers += 1
        owner = max(owner, _mul(_mul(layer['width'], layer['height']),
                                OWNED_NORMAL_BYTES_PER_TEXEL))
        transient = max(transient, _owned_transient_bytes(layer))
    future = _mul(max_future, owner)
    return {
        'owner_bytes': owner, 'future_bytes': future,
        'transient_bytes': transient, 'reference_layers': reference_layers,
        'ready_plus_producer_charge_bytes': margin_bytes(_add(future, transient)),
        'max_future': max_future,
    }

CURRENT_OWNED_PROFILE = 'current-phase-v1'

CURRENT_OWNED_STATUS = 'engineering_formula_only'

CURRENT_OWNED_SOURCE_COMMIT = 'efa2593a275ceb314ba8703ff85d0e6728a03028'

PHOTO_PHASE_PROFILE = 'photo-phase-v1'

PHOTO_PHASE_LINEAGE_SHA = 'cf7891059c402c333b84b752de3acb8b084abdad'

PHOTO_PHASE_SOURCE_COMMIT = PHOTO_PHASE_LINEAGE_SHA

PHOTO_PHASE_SOURCE_FINGERPRINT = (
    'c8468ff4f513c49cae1f07b42dd2b7b5c5058b4e2517d7dfe770919f7513f604')

PHOTO_PHASE_SUPPORTED_PHASES = {
    'photo': 'photometric-only: geom_consistency=false, filter=false, one reference '
             'per invocation (RunOptimizedPairJob)',
}

PHOTO_PHASE_GRAYSCALE_BYTES_PER_TEXEL = 1        # uint8 bitmap in PHOTO

PHOTO_PHASE_STAGING_BYTES_PER_TEXEL = 1          # source grayscale staging

PHOTO_PHASE_HOST_SCRATCH_PIXELS = 24             # preserved result/reference scratch

PHOTO_PHASE_PRNG_STATE_BYTES = 48                # sizeof(curandState) XORWOW48

PHOTO_PHASE_NORMAL_BYTES_PER_TEXEL = 12          # 3 float channels

PHOTO_PHASE_DEPTH_BYTES_PER_TEXEL = 4

PHOTO_PHASE_REF_IMAGE_BYTES_PER_TEXEL = 1

PHOTO_PHASE_REF_MOMENT_BYTES_PER_TEXEL = 4       # sum_image / squared_sum_image

PHOTO_PHASE_SELECTION_BYTES_PER_SOURCE_TEXEL = 4 # one packed float plane

PHOTO_PHASE_WORKSPACE_FLOATS = 2                 # global_workspace_ depth

PHOTO_PHASE_REF_TEXTURE_PARITIES = 2             # both ref image textures exist

PHOTO_PHASE_BITMAP_BYTES_PER_TEXEL = 1           # uint8 layered texture texel

PHOTO_PHASE_SOURCE_TEXTURE_BYTES_PER_TEXEL = 1   # uint8 layered source texels

PHOTO_PHASE_TEXTURE_PIXELS_COEFFICIENT = (
    PHOTO_PHASE_REF_TEXTURE_PARITIES * PHOTO_PHASE_BITMAP_BYTES_PER_TEXEL)  # 2

PHOTO_PHASE_INITIALIZATION_PIXELS_COEFFICIENT = 73   # 48+4+12+1+4+4 (parity 0)

PHOTO_PHASE_STEADY_PIXELS_COEFFICIENT = 50           # 8+24+2+8+8 (both parities)

PHOTO_PHASE_INITIALIZATION_SOURCES_COEFFICIENT = 12  # selection 4 + prev 4 + cost 4

PHOTO_PHASE_STEADY_SOURCES_COEFFICIENT = 20          # 4 + 2*4 prev + 2*4 cost, NO mask

PHOTO_PHASE_POSE_TEXTURE_ROTATIONS = 4

PHOTO_PHASE_POSE_TEXTURE_FLOATS_PER_SOURCE = 4 + 9 + 3 + 3 + 12 + 12

PHOTO_PHASE_POSE_TEXTURE_FLOAT_BYTES = 4

PHOTO_PHASE_POSE_TEXTURE_PROVENANCE = (
    '0611de098:src/colmap/mvs/patch_match_cuda_owned.cu:2070-2117 '
    'InitTransforms; patch_match_cuda_owned.h:151 four solver-owned members; '
    'per-owner, retained in both initialization and steady phases')

PHOTO_PHASE_PROCESS_CHARGE_BYTES = OWNED_PROCESS_CHARGE_BYTES

PHOTO_PHASE_MARGIN_POLICY = (
    'single ceil(1.03) margin applied once per disjoint raw term (decoded input '
    'bank, staging/host scratch, controller cache, process charge, device phase '
    'maximum); totals are sums of already-margined terms and are never re-margined')

PHOTO_PHASE_ESTIMATE_CLASS = 'engineering_arithmetic_only_no_calibration'

PHOTO_PHASE_DELIVERY_KEYS = frozenset({
    'profile', 'fingerprint', 'lineage_sha', 'source_commit', 'options_sha256',
    'process', 'private_charge', 'reference_image_idx', 'source_image_idxs',
})

_NATIVE_INT_MAX = (1 << 31) - 1

def _photo_exact_int(name, value, minimum):
    if type(value) is not int or value < minimum:
        raise ValueError('%s must be an exact integer >= %d, got %r'
                         % (name, minimum, value))
    return value

def photo_phase_boundary_pixels(width, height):
    """PrngBoundaryLayout.Count(): canonical perimeter, corners stored once.

    Degenerate one-row/one-column shapes have a true border count of the other
    dimension (owned_prng_boundary.h). The compacted state row is a GpuMat width,
    so its byte pitch must stay inside the native int ABI: refused, never clamped.
    """
    width = _photo_exact_int('width', width, 1)
    height = _photo_exact_int('height', height, 1)
    count = (width if height == 1 else height if width == 1
             else _add(_mul(2, width), _mul(2, height)) - 4)
    if count > _NATIVE_INT_MAX // PHOTO_PHASE_PRNG_STATE_BYTES:
        raise OverflowError('photo PRNG boundary pitch exceeds native int ABI')
    return count

def photo_phase_cache_bytes(cache_size_gib):
    """``owned::CacheBytes``: exact gibibytes -> bytes, refusing unrepresentable."""
    if isinstance(cache_size_gib, bool) or not isinstance(cache_size_gib, (int, float)):
        raise ValueError('cache_size_gib must be a finite number, got %r'
                         % (cache_size_gib,))
    gib = float(cache_size_gib)
    if not math.isfinite(gib) or gib <= 0.0:
        raise ValueError('cache_size_gib must be finite and > 0, got %r'
                         % (cache_size_gib,))
    size = int(gib * 1073741824.0)
    if size < 1:
        raise ValueError('cache_size_gib %r retains less than one byte'
                         % (cache_size_gib,))
    return size

def photo_phase_private_terms(width, height, sources, *, profile=PHOTO_PHASE_PROFILE,
                              phase='photo', input_pixels=None,
                              max_source_width=None, max_source_height=None):
    """Raw PHOTO per-owner terms for ONE reference job with ``sources`` sources.

    ``input_pixels`` (q) defaults to a*1 + a*s, i.e. the uniform assumption that
    every source has the reference dimensions; ``max_source_width``/
    ``max_source_height`` default to (width, height). Pass the real values from the
    workload when they differ -- the assumption is reported, never hidden. Phases
    are NEVER summed: the owner charge is max(init, steady).
    """
    if profile != PHOTO_PHASE_PROFILE:
        raise ValueError('unsupported photo profile key %r (the model knows only %r; '
                         'current-phase-v1 is filtered GEO and does not cover PHOTO)'
                         % (profile, PHOTO_PHASE_PROFILE))
    if phase not in PHOTO_PHASE_SUPPORTED_PHASES:
        raise OwnedPhaseUnsupported(
            'phase %r has no photo-phase-v1 terms (supported: %s)'
            % (phase, ', '.join(sorted(PHOTO_PHASE_SUPPORTED_PHASES))))
    width = _photo_exact_int('width', width, 1)
    height = _photo_exact_int('height', height, 1)
    sources = _photo_exact_int('sources', sources, 1)
    boundary = photo_phase_boundary_pixels(width, height)
    area = _mul(width, height)
    max_w = width if max_source_width is None else _photo_exact_int(
        'max_source_width', max_source_width, 1)
    max_h = height if max_source_height is None else _photo_exact_int(
        'max_source_height', max_source_height, 1)
    uniform = input_pixels is None and max_source_width is None and max_source_height is None
    if input_pixels is None:
        input_pixels = _add(area, _mul(sources, area))
    else:
        input_pixels = _photo_exact_int('input_pixels', input_pixels, 1)
    max_dim = max(width, height)

    # host: decoded input bank (private, grayscale) + staging/result scratch
    bank = _mul(2, _mul(PHOTO_PHASE_GRAYSCALE_BYTES_PER_TEXEL, input_pixels))
    staging = _mul(PHOTO_PHASE_STAGING_BYTES_PER_TEXEL,
                   _mul(sources, _mul(max_w, max_h)))
    scratch = _mul(area, PHOTO_PHASE_HOST_SCRATCH_PIXELS)
    host_scratch = max(staging, scratch)

    # device: initialization inventory (parity 0 + full-image PRNG temporary)
    # coexisting with the compacted boundary, then the retained both-parity set.
    prng_boundary = _mul(boundary, PHOTO_PHASE_PRNG_STATE_BYTES)
    prng_full_image = _mul(area, PHOTO_PHASE_PRNG_STATE_BYTES)
    workspace = _mul(_mul(PHOTO_PHASE_WORKSPACE_FLOATS, 4), _mul(max_dim, sources))
    # Both CUDA-array textures are constructed during initialization and retained
    # for the job's life (members), so they belong to BOTH phases.
    reference_texture = _mul(area, PHOTO_PHASE_TEXTURE_PIXELS_COEFFICIENT)
    source_texture = _mul(PHOTO_PHASE_SOURCE_TEXTURE_BYTES_PER_TEXEL,
                          _mul(sources, _mul(max_w, max_h)))
    pose_textures = _mul(
        _mul(PHOTO_PHASE_POSE_TEXTURE_ROTATIONS,
             PHOTO_PHASE_POSE_TEXTURE_FLOATS_PER_SOURCE),
        _mul(sources, PHOTO_PHASE_POSE_TEXTURE_FLOAT_BYTES))
    initialization = _add(
        _add(_mul(area, PHOTO_PHASE_INITIALIZATION_PIXELS_COEFFICIENT),
             _mul(_mul(sources, area),
                  PHOTO_PHASE_INITIALIZATION_SOURCES_COEFFICIENT)),
        _add(_add(prng_boundary, workspace),
             _add(_add(reference_texture, source_texture), pose_textures)))
    steady = _add(
        _add(_mul(area, PHOTO_PHASE_STEADY_PIXELS_COEFFICIENT),
             _mul(_mul(sources, area), PHOTO_PHASE_STEADY_SOURCES_COEFFICIENT)),
        _add(_add(prng_boundary, workspace),
             _add(_add(reference_texture, source_texture), pose_textures)))
    device_raw = max(initialization, steady)
    return {
        'profile': profile, 'phase': phase,
        'lineage_sha': PHOTO_PHASE_LINEAGE_SHA,
        'shape': (width, height, sources),
        'estimate_class': PHOTO_PHASE_ESTIMATE_CLASS,
        'boundary_pixels': boundary,
        'reference_pixels': area, 'input_pixels': input_pixels,
        'input_bank_assumption': ('uniform: every source has the reference dimensions'
                                  if uniform else 'explicit input_pixels/padded dims'),
        # host raw terms
        'decoded_input_bank_raw_bytes': bank,
        'staging_raw_bytes': staging,
        'host_scratch_raw_bytes': host_scratch,
        'host_raw_bytes': _add(bank, host_scratch),
        # device raw terms (named so the decomposition is auditable)
        'initialization_prng_full_image_bytes': prng_full_image,
        'prng_boundary_bytes': prng_boundary,
        'initialization_normal_bytes': _mul(area, PHOTO_PHASE_NORMAL_BYTES_PER_TEXEL),
        'steady_normal_bytes': _mul(_mul(area, PHOTO_PHASE_NORMAL_BYTES_PER_TEXEL), 2),
        'initialization_depth_bytes': _mul(area, PHOTO_PHASE_DEPTH_BYTES_PER_TEXEL),
        'steady_depth_bytes': _mul(_mul(area, PHOTO_PHASE_DEPTH_BYTES_PER_TEXEL), 2),
        'initialization_ref_image_bytes': _mul(
            area, _add(PHOTO_PHASE_REF_IMAGE_BYTES_PER_TEXEL,
                       _mul(2, PHOTO_PHASE_REF_MOMENT_BYTES_PER_TEXEL))),
        'steady_ref_image_bytes': _mul(
            _mul(area, _add(PHOTO_PHASE_REF_IMAGE_BYTES_PER_TEXEL,
                            _mul(2, PHOTO_PHASE_REF_MOMENT_BYTES_PER_TEXEL))), 2),
        'initialization_selection_bytes': _mul(
            _mul(area, sources), PHOTO_PHASE_SELECTION_BYTES_PER_SOURCE_TEXEL),
        'steady_selection_bytes': _mul(
            _mul(area, sources), PHOTO_PHASE_SELECTION_BYTES_PER_SOURCE_TEXEL),
        'initialization_previous_cost_bytes': _mul(
            _mul(_mul(area, sources), PHOTO_PHASE_SELECTION_BYTES_PER_SOURCE_TEXEL), 2),
        'steady_previous_cost_bytes': _mul(
            _mul(_mul(area, sources), PHOTO_PHASE_SELECTION_BYTES_PER_SOURCE_TEXEL), 4),
        'consistency_mask_bytes': 0,     # filter=false for PHOTO: no mask parity
        'workspace_bytes': workspace,
        # CUDA-array textures: separate device allocations, both phases, both
        # parities for the reference; per-source layered uint8 for the sources.
        'reference_texture_bytes': reference_texture,
        'source_texture_bytes': source_texture,
        'pose_textures_raw_bytes': pose_textures,
        'pose_textures_provenance': PHOTO_PHASE_POSE_TEXTURE_PROVENANCE,
        'source_depth_texture_bytes': 0,  # geom_consistency=false for PHOTO
        'reference_texture_parities': PHOTO_PHASE_REF_TEXTURE_PARITIES,
        # rotation-inventory coefficients only (the textures are separate terms
        # above, so these stay exact for non-uniform source dimensions).
        'initialization_pixels_coefficient': PHOTO_PHASE_INITIALIZATION_PIXELS_COEFFICIENT,
        'steady_pixels_coefficient': PHOTO_PHASE_STEADY_PIXELS_COEFFICIENT,
        'initialization_sources_coefficient': PHOTO_PHASE_INITIALIZATION_SOURCES_COEFFICIENT,
        'steady_sources_coefficient': PHOTO_PHASE_STEADY_SOURCES_COEFFICIENT,
        'texture_pixels_coefficient': PHOTO_PHASE_TEXTURE_PIXELS_COEFFICIENT,
        'total_initialization_pixels_coefficient': _add(
            PHOTO_PHASE_INITIALIZATION_PIXELS_COEFFICIENT,
            PHOTO_PHASE_TEXTURE_PIXELS_COEFFICIENT),
        'total_steady_pixels_coefficient': _add(
            PHOTO_PHASE_STEADY_PIXELS_COEFFICIENT,
            PHOTO_PHASE_TEXTURE_PIXELS_COEFFICIENT),
        'initialization_device_raw_bytes': initialization,
        'steady_device_raw_bytes': steady,
        'device_raw_bytes': device_raw,
        'peak_phase': 'initialization' if initialization >= steady else 'steady',
        'margin_policy': PHOTO_PHASE_MARGIN_POLICY,
        'calibration_certified': False, 'physical_fit_certified': False,
    }

def photo_phase_private_charge(width, height, sources, *, cache_size_gib, **kwargs):
    """PHASE-AWARE per-owner ``{host, device}`` reservation for the PHOTO wire.

    host   = margin(decoded input bank) + margin(staging/host scratch)
    device = margin(max(init, steady))          # phases NEVER summed, never added
                                                # to the legacy absent-binding floor
    The controller LRU is NOT part of a per-owner charge: it is a once-per-
    controller resource and lives in ``photo_phase_process_charge`` (the same
    process-vs-private split the native wire uses). ``cache_size_gib`` is still
    required here only to enforce the native input-set floor ``cache >= q``
    (owned_photo_runner.h), which is a property of the workload, not a charge.

    These numbers are admissible ONLY with the binding-aware native reservation
    (PHOTO V3). With no binding present the legacy ``CheckOptimizedPhotoInputs``
    floor governs instead; that floor is computed and reported separately by
    ``photo_phase_legacy_absent_binding_floor`` and is strictly larger here.
    """
    terms = photo_phase_private_terms(width, height, sources, **kwargs)
    cache = photo_phase_cache_bytes(cache_size_gib)
    if cache < terms['input_pixels']:
        raise ValueError(
            'cache must retain the complete private PHOTO input set: '
            '%d bytes configured < %d input pixels (owned_photo_runner.h refuses '
            'this workload)' % (cache, terms['input_pixels']))
    return {'host': _add(margin_bytes(terms['decoded_input_bank_raw_bytes']),
                         margin_bytes(terms['host_scratch_raw_bytes'])),
            'device': margin_bytes(terms['device_raw_bytes'])}

def photo_phase_process_charge(host_raw_bytes=PHOTO_PHASE_PROCESS_CHARGE_BYTES,
                               device_raw_bytes=PHOTO_PHASE_PROCESS_CHARGE_BYTES,
                               *, controllers=1, cache_size_gib=None):
    """ONCE-PER-CONTROLLER ``{host, device}`` charge pair.

    host   = margin(process raw) + controllers * margin(cache)
    device = margin(process raw)
    The controller LRU belongs here and nowhere else, so N delivered reservations
    plus this pair describe exactly the ``photo_phase_k_caps`` arithmetic.
    """
    for name, value in (('host_raw_bytes', host_raw_bytes),
                        ('device_raw_bytes', device_raw_bytes)):
        _photo_exact_int(name, value, 0)
    controllers = _photo_exact_int('controllers', controllers, 1)
    host = margin_bytes(host_raw_bytes)
    if cache_size_gib is not None:
        host = _add(host, _mul(controllers,
                              margin_bytes(photo_phase_cache_bytes(cache_size_gib))))
    return {'host': host, 'device': margin_bytes(device_raw_bytes)}

def _photo_index(name, value):
    if type(value) is not int or not 0 <= value <= _NATIVE_INT_MAX:
        raise ValueError('%s must be a canonical nonnegative int in native int '
                         'range, got %r' % (name, value))
    return value

def estimate_owned_private_job(job, mode='eager', *, profile='legacy',
                               phase='filtered_geometry', resident=None):
    """One resident solver slot (owned_pair.cc:265-283 formulas).

    Host: eager 24a (normal retained in the bank) / bounded 36a = 24a private
    scratch + 12a active normal holder. Device:
    Legacy default: (146+26s)*a + 8*max(w,h)*s + 2a + 692s + 32.
    Explicit current-phase-v1: resident filtered-GEO engineering phase maximum,
    not legacy calibration. PHOTO/nonresident modes are unsupported. The raw
    arithmetic alone does not verify any source tree. Single margin per raw term.
    """
    if mode not in ('eager', 'bounded'):
        raise ValueError("mode must be 'eager' or 'bounded'")
    if profile not in ('legacy', CURRENT_OWNED_PROFILE):
        raise ValueError('unsupported owned algorithm profile')
    if profile == CURRENT_OWNED_PROFILE:
        if resident is not True or phase != 'filtered_geometry':
            raise OwnedPhaseUnsupported('current profile supports only resident filtered geometry')
    resolved = _owned_resolve([job])
    layer = None
    ref_layer_key = None
    for input_desc in job['inputs']:
        if input_desc['image_idx'] == job['reference_image_idx']:
            ref_layer_key = input_desc['layer_key']
            break
    for candidate in resolved['layers']:
        if candidate['layer_key'] == ref_layer_key:
            layer = candidate
    if layer is None:
        raise ValueError('reference layer not resolved')
    a = _mul(layer['width'], layer['height'])
    s = len(job['source_image_idxs'])
    w, h = layer['width'], layer['height']
    rotation = _add(_mul(_add(_mul(146, 1), _mul(26, s)), a),
                    _mul(_mul(8, max(w, h)), s))
    phase_terms = {}
    if profile == CURRENT_OWNED_PROFILE:
        # Full initialization PRNG overlaps boundary compaction, then is freed
        # before swapped owners/final mask. Per-owner max covers mixed phases;
        # no initialization serialization is assumed.
        # Both orientations use CudaRotate's signed-int row-pitch ABI.
        if max(w, h) > ((1 << 31) - 1) // 4:
            raise OverflowError('current selection scratch pitch exceeds native int ABI')
        boundary = (w if h == 1 else h if w == 1 else
                    _add(_mul(2, w), _mul(2, h)) - 4)
        if boundary > ((1 << 31) - 1) // 48:
            raise OverflowError('current PRNG boundary pitch exceeds native int ABI')
        common = _add(_mul(48, boundary), _mul(_mul(8, max(w, h)), s))
        initialization = _add(_mul(_add(73, _mul(12, s)), a), common)
        steady = _add(_mul(_add(50, _mul(21, s)), a), common)
        rotation = max(initialization, steady)
        phase_terms = {
            'profile': profile, 'phase': phase, 'resident': True,
            'formula_source_commit': CURRENT_OWNED_SOURCE_COMMIT,
            'source_binding_status': 'unbound_arithmetic',
            'qualification_status': CURRENT_OWNED_STATUS,
            'calibration_certified': False, 'physical_fit_certified': False,
            'initialization_rotation_raw_bytes': initialization,
            'steady_rotation_raw_bytes': steady,
            'boundary_pixels': boundary,
            'peak_phase': 'initialization' if initialization > steady else 'steady',
        }
    device_raw = _add(_add(_add(rotation, _mul(a, 2)), _mul(692, s)), 32)
    host_raw = _mul(a, 36 if mode == 'bounded' else 24)
    return {
        **phase_terms,
        'shape': (w, h, s),
        'host_raw_bytes': host_raw, 'host_charge_bytes': margin_bytes(host_raw),
        'device_raw_bytes': device_raw, 'device_charge_bytes': margin_bytes(device_raw),
        'mode': mode,
    }

COHORT_METADATA_SIZEOF_KEYS = (
    'BankJobIdentities', 'BankLayerIdentity', 'layer_expected',
    'snapshot_entry', 'allocation_granularity',
)

def estimate_cohort_metadata(*, members, inputs_per_job, layer_expected_count,
                             snapshot_entry_count, planner_high_water_bytes,
                             sizeofs, path_bytes=(), invocation_members=None,
                             invocation_inputs_per_job=None, copy_phases=1):
    """Raw host metadata for the cohort snapshot/plan arrays (single margin later).

    Two identity arrays coexist and are charged explicitly: the PERSISTENT
    invocation-wide snapshot (all selected jobs) and the per-subset bank copy
    (``copy_phases`` copies, conservatively declared). ``sizeofs`` is a REQUIRED
    caller-supplied descriptor of the ACTUAL native GNU13 sizes; this module ships
    no default or certified sizes. ``path_bytes`` are canonical path string
    lengths (dynamic storage), capacity-rounded by ``allocation_granularity``.
    """
    if not isinstance(sizeofs, dict):
        raise ValueError('cohort metadata sizeofs descriptor is required (an actual '
                         'native GNU13 descriptor); no default sizes exist here')
    missing = [key for key in COHORT_METADATA_SIZEOF_KEYS if key not in sizeofs]
    if missing:
        raise ValueError('cohort metadata sizeofs descriptor is incomplete (missing %s); '
                         'no default sizes here' % ', '.join(sorted(missing)))
    for key in COHORT_METADATA_SIZEOF_KEYS:
        value = sizeofs[key]
        if type(value) is not int or value < 0:
            raise ValueError('cohort metadata sizeof %s must be a nonnegative int' % key)
    granularity = sizeofs['allocation_granularity']
    if granularity < 1:
        raise ValueError('cohort metadata allocation_granularity must be >= 1')
    if type(planner_high_water_bytes) is not int or planner_high_water_bytes < 0:
        raise ValueError('planner_high_water_bytes must be a nonnegative int')
    if type(copy_phases) is not int or copy_phases < 1:
        raise ValueError('copy_phases must be an explicit integer >= 1')
    if invocation_members is None:
        invocation_members = members
    if invocation_inputs_per_job is None:
        invocation_inputs_per_job = inputs_per_job
    invocation_identity = _mul(len(invocation_members), sizeofs['BankJobIdentities'])
    invocation_identity = _add(
        invocation_identity,
        _mul(sum(invocation_inputs_per_job), sizeofs['BankLayerIdentity']))
    copy_identity = _mul(len(members), sizeofs['BankJobIdentities'])
    copy_identity = _add(copy_identity,
                         _mul(sum(inputs_per_job), sizeofs['BankLayerIdentity']))
    copy_identity = _mul(copy_phases, copy_identity)
    identity = _add(invocation_identity, copy_identity)
    paths = 0
    for size in path_bytes:
        if type(size) is not int or size < 0:
            raise ValueError('canonical path byte lengths must be nonnegative ints')
        paths = _add(paths, _mul(-(-size // granularity), granularity))
    snapshots = _mul(snapshot_entry_count, sizeofs['snapshot_entry'])
    dedup = _mul(layer_expected_count, sizeofs['layer_expected'])
    raw = _add(_add(_add(identity, paths), _add(snapshots, dedup)),
               planner_high_water_bytes)
    return {'raw_bytes': raw,
            'identity_array_bytes': identity,
            'invocation_identity_array_bytes': invocation_identity,
            'bank_copy_identity_array_bytes': copy_identity,
            'copy_phases': copy_phases,
            'path_dynamic_bytes': paths, 'snapshot_bytes': snapshots,
            'dedup_layer_expected_bytes': dedup,
            'planner_high_water_bytes': planner_high_water_bytes,
            'allocation_granularity': granularity,
            'sizeofs_source': 'caller_supplied_native_actual'}
