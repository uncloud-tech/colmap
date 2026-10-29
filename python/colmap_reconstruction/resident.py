# SPDX-License-Identifier: BSD-3-Clause
"""Strict native resident-contract V1 parser + canonical-formula adapter (CPU-only).

Consumes the native ``owned-bank-contract`` version 1 metadata bytes **verbatim** --
no re-derivation, no sorting, no option defaulting -- and feeds the existing
canonical estimators (`estimate_owned_bank`, `estimate_owned_normal_allowance`,
`estimate_owned_private_job`, `estimate_cohort_metadata`). No new math.

Native-boundary honesty (review round 2, commit 7667221 -> this revision):

* numbers are decoded with **native byte significance** (big-endian IEEE hex as the
  native serializer emits them: ``3ff0000000000000`` is ``1.0``);
* parsing is **native strict**: no leading zeros, int32 ranges for native ``int``
  fields, nonnegative indices, positive counts/dims, exact marker tokens, and
  length-prefixed **byte** strings (never a UTF-8 requirement -- native accepts
  arbitrary bytes);
* layer identity binds **image + depth + normal** stat identities **and K/R/T**,
  never the image alone;
* target (layer) dimensions are the contract's **effective image dims**, payload
  dims are the **map header** dims, and the **original decode dims are required
  from the caller** (the contract does not carry them) -- unknown resize terms are
  refused rather than silently approximated;
* accounting assumptions (unclassified allowance, metadata mode, active_jobs,
  image channels) are **explicit required inputs**, validated, surfaced in the
  output, and never silently zero;
* the returned record carries the **raw contract bytes**.

The contract is a binding record, never a certification:
``allocation_compatibility`` is preserved as recorded and every forecast carries
``admission_authorized=False``. Nothing here turns metadata into an execution
certificate or a GPU whitelist.
"""
from __future__ import annotations

import hashlib
import math
import struct

from . import model

__all__ = ['ContractError', 'OPTION_SCHEMA', 'parse_owned_bank_contract',
           'contract_jobs', 'contract_identity_differences', 'forecast_contract',
           'CONTRACT_IMAGE_CHANNELS_DEFAULT']

CONTRACT_MAGIC = 'owned-bank-contract'
CONTRACT_VERSION = 1
CONTRACT_IMAGE_CHANNELS_DEFAULT = 3  # JPEG/DSLR RGB; the caller must still confirm/override

# Fixed, exhaustive, version-bound order (mirrors SerializePatchMatchOptions).
OPTION_SCHEMA = (
    ('opt_int', 'max_image_size'),
    ('opt_string', 'gpu_index'),
    ('opt_double', 'depth_min'),
    ('opt_double', 'depth_max'),
    ('opt_int', 'random_seed'),
    ('opt_int', 'window_radius'),
    ('opt_int', 'window_step'),
    ('opt_double', 'sigma_spatial'),
    ('opt_double', 'sigma_color'),
    ('opt_int', 'num_samples'),
    ('opt_bool', 'owned_distinct_source_cache'),
    ('opt_bool', 'owned_tex_lookahead'),
    ('opt_double', 'ncc_sigma'),
    ('opt_double', 'min_triangulation_angle'),
    ('opt_double', 'incident_angle_sigma'),
    ('opt_int', 'num_iterations'),
    ('opt_bool', 'geom_consistency'),
    ('opt_double', 'geom_consistency_regularizer'),
    ('opt_double', 'geom_consistency_max_cost'),
    ('opt_bool', 'filter'),
    ('opt_double', 'filter_min_ncc'),
    ('opt_double', 'filter_min_triangulation_angle'),
    ('opt_int', 'filter_min_num_consistent'),
    ('opt_double', 'filter_geom_consistency_max_cost'),
    ('opt_bool', 'write_consistency_graph'),
    ('opt_bool', 'allow_missing_files'),
    ('opt_int', 'num_threads'),
    ('opt_double', 'cache_size'),
)

_I32_MIN, _I32_MAX = -(2 ** 31), 2 ** 31 - 1
_I64_MIN, _I64_MAX = -(2 ** 63), 2 ** 63 - 1
_U64_MAX = 2 ** 64 - 1


class ContractError(ValueError):
    """Strict contract-format violation (mirrors the native std::invalid_argument)."""


class MalformedDeclaration(ContractError):
    """A CALLER DECLARATION is malformed (decode dims/channels), not a geometry verdict.

    Kept distinct so a candidate-level 'invalid projected geometry' verdict is never used
    to label a caller's own malformed declaration.
    """


class _Lexer:
    def __init__(self, data: bytes):
        if not data:
            raise ContractError('empty contract')
        try:
            text = data.decode('ascii')
        except UnicodeDecodeError:
            raise ContractError('contract is not ASCII (canonical form is ASCII)') from None
        if not text.endswith('\n'):
            raise ContractError('contract must end with a newline (truncated?)')
        self.lines = text.split('\n')[:-1]
        self.index = 0

    def at_end(self):
        return self.index >= len(self.lines)

    def next(self):
        if self.at_end():
            return None
        line = self.lines[self.index]
        self.index += 1
        if line == '' or line != line.strip():
            raise ContractError('line %d: empty or padded line' % self.index)
        return line.split(' ')

    def expect(self, tokens, key):
        if tokens is None:
            raise ContractError('expected %r, got end of contract' % key)
        if len(tokens) != 1 or tokens[0] != key:
            raise ContractError('line %d: expected marker %r, got %r'
                                % (self.index, key, ' '.join(tokens)))
        return tokens

    def expect_kv(self, tokens, key):
        if tokens is None:
            raise ContractError('expected %r, got end of contract' % key)
        if tokens[0] != key:
            raise ContractError('line %d: expected %r, got %r'
                                % (self.index, key, tokens[0]))
        return tokens

    def fail(self, message):
        raise ContractError('line %d: %s' % (self.index, message))


def _decimal(token, line, *, signed=False, bits=64, minimum=None):
    """Native-strict decimal integer: no leading zeros, bounded by ``bits``."""
    body = token
    negative = False
    if signed and token.startswith('-'):
        negative = True
        body = token[1:]
    if not body.isdigit():
        line_fail('expected a %s decimal integer, got %r'
                  % ('signed' if signed else 'unsigned', token), line)
    if len(body) > 1 and body[0] == '0':
        line_fail('leading zeros are not canonical: %r' % token, line)
    value = int(token)
    low, high = (-(2 ** (bits - 1)), 2 ** (bits - 1) - 1) if signed else (0, 2 ** bits - 1)
    if value < low or value > high:
        line_fail('integer out of range for %s%d: %r' % ('int' if signed else 'uint',
                                                         bits, token), line)
    if minimum is not None and value < minimum:
        line_fail('value below the native minimum %d: %r' % (minimum, token), line)
    return value


def line_fail(message, line):
    raise ContractError('line %d: %s' % (line, message))


def _hex(token, line, expected_bytes):
    if len(token) != expected_bytes * 2:
        line_fail('expected %d hex bytes, got %r' % (expected_bytes, token), line)
    if any(c not in '0123456789abcdef' for c in token):
        line_fail('non-canonical (uppercase/invalid) hex: %r' % token, line)
    return bytes.fromhex(token)


def _string_bytes(tokens, line, label):
    """Length-prefixed byte string; native accepts arbitrary bytes (no UTF-8 gate)."""
    if len(tokens) != 3:
        line_fail('%s field count' % label, line)
    length = _decimal(tokens[1], line)
    raw = _hex(tokens[2], line, length)
    return raw


def _string_text(raw):
    return raw.decode('utf-8', 'surrogateescape')


def _bool(token, line):
    if token not in ('0', '1'):
        line_fail('bool must be 0 or 1, got %r' % token, line)
    return token == '1'


def _float32(token, line):
    """Native float hex -> float, using NATIVE byte significance (big-endian)."""
    raw = _hex(token, line, 4)
    return struct.unpack('>f', raw)[0]


def _double(token, line):
    raw = _hex(token, line, 8)
    return struct.unpack('>d', raw)[0]


def _options(lexer):
    options = {}
    for kind, key in OPTION_SCHEMA:
        tokens = lexer.expect_kv(lexer.next(), kind)
        line = lexer.index
        if len(tokens) < 2 or tokens[1] != key:
            line_fail('expected %s %s, got %r' % (kind, key, ' '.join(tokens)), line)
        if kind == 'opt_int':
            if len(tokens) != 3:
                line_fail('opt_int field count', line)
            options[key] = _decimal(tokens[2], line, signed=True, bits=32)
        elif kind == 'opt_bool':
            if len(tokens) != 3:
                line_fail('opt_bool field count', line)
            options[key] = _bool(tokens[2], line)
        elif kind == 'opt_double':
            if len(tokens) != 3:
                line_fail('opt_double field count', line)
            options[key] = _double(tokens[2], line)
        elif kind == 'opt_string':
            options[key] = _string_text(_string_bytes(tokens[1:], line, 'opt_string'))
    return options


def _identity(tokens, line, key):
    """``<key> <path_len> <path_hex> <size> <device> <inode> <mt_s> <mt_ns> <ct_s> <ct_ns>``."""
    if len(tokens) != 10 or tokens[0] != key:
        line_fail('%s must carry path + size + device + inode + 4 timestamps' % key, line)
    path_bytes = _string_bytes([tokens[0], tokens[1], tokens[2]], line, key)
    return {'path_bytes': path_bytes, 'path': _string_text(path_bytes),
            'size': _decimal(tokens[3], line),
            'device': _decimal(tokens[4], line),
            'inode': _decimal(tokens[5], line),
            'modified_seconds': _decimal(tokens[6], line, signed=True),
            'modified_nanoseconds': _decimal(tokens[7], line, signed=True),
            'changed_seconds': _decimal(tokens[8], line, signed=True),
            'changed_nanoseconds': _decimal(tokens[9], line, signed=True)}


def parse_owned_bank_contract(data: bytes) -> dict:
    """Strict canonical parse. Returns the structured record, raw bytes and sha256."""
    lexer = _Lexer(data)
    lexer.expect(lexer.next(), CONTRACT_MAGIC)
    tokens = lexer.expect_kv(lexer.next(), 'version')
    if len(tokens) != 2 or _decimal(tokens[1], lexer.index) != CONTRACT_VERSION:
        lexer.fail('unsupported contract version')
    tokens = lexer.expect_kv(lexer.next(), 'build_cuda_enabled')
    if len(tokens) != 2:
        lexer.fail('build_cuda_enabled field count')
    cuda = _bool(tokens[1], lexer.index)
    tokens = lexer.expect_kv(lexer.next(), 'build_hip_enabled')
    if len(tokens) != 2:
        lexer.fail('build_hip_enabled field count')
    hip = _bool(tokens[1], lexer.index)
    allocation = _string_text(_string_bytes(
        lexer.expect_kv(lexer.next(), 'allocation_compatibility'),
        lexer.index, 'allocation_compatibility'))
    lexer.expect(lexer.next(), 'invocation_options')
    invocation_options = _options(lexer)
    stereo_folder = _string_text(_string_bytes(
        lexer.expect_kv(lexer.next(), 'stereo_folder'), lexer.index, 'stereo_folder'))
    tokens = lexer.expect_kv(lexer.next(), 'problem_count')
    if len(tokens) != 2:
        lexer.fail('problem_count field count')
    problem_count = _decimal(tokens[1], lexer.index, minimum=1)
    problems = []
    seen_references = set()
    for number in range(problem_count):
        tokens = lexer.expect_kv(lexer.next(), 'problem')
        if len(tokens) != 2 or _decimal(tokens[1], lexer.index) != number:
            lexer.fail('problem indices must be contiguous from 0')
        tokens = lexer.expect_kv(lexer.next(), 'reference_image_idx')
        if len(tokens) != 2:
            lexer.fail('reference_image_idx field count')
        reference_image_idx = _decimal(tokens[1], lexer.index, signed=True, bits=32,
                                       minimum=0)
        if reference_image_idx in seen_references:
            lexer.fail('duplicate reference image_idx %d' % reference_image_idx)
        seen_references.add(reference_image_idx)
        reference_name = _string_text(_string_bytes(
            lexer.expect_kv(lexer.next(), 'reference_name'), lexer.index, 'reference_name'))
        output_depth_path = _string_text(_string_bytes(
            lexer.expect_kv(lexer.next(), 'output_depth_path'), lexer.index,
            'output_depth_path'))
        output_normal_path = _string_text(_string_bytes(
            lexer.expect_kv(lexer.next(), 'output_normal_path'), lexer.index,
            'output_normal_path'))
        lexer.expect(lexer.next(), 'problem_options')
        options = _options(lexer)
        tokens = lexer.expect_kv(lexer.next(), 'input_count')
        if len(tokens) != 2:
            lexer.fail('input_count field count')
        input_count = _decimal(tokens[1], lexer.index, minimum=1)
        inputs = []
        seen_indices = set()
        for ordinal in range(input_count):
            tokens = lexer.expect_kv(lexer.next(), 'input')
            if len(tokens) != 2 or _decimal(tokens[1], lexer.index) != ordinal:
                lexer.fail('input ordinals must be contiguous from 0')
            entry = {}
            tokens = lexer.expect_kv(lexer.next(), 'is_reference')
            if len(tokens) != 2:
                lexer.fail('is_reference field count')
            entry['is_reference'] = _bool(tokens[1], lexer.index)
            tokens = lexer.expect_kv(lexer.next(), 'image_idx')
            if len(tokens) != 2:
                lexer.fail('image_idx field count')
            entry['image_idx'] = _decimal(tokens[1], lexer.index, signed=True, bits=32,
                                         minimum=0)
            if entry['image_idx'] in seen_indices:
                lexer.fail('duplicate input image_idx %d' % entry['image_idx'])
            seen_indices.add(entry['image_idx'])
            raw_name = _string_bytes(lexer.expect_kv(lexer.next(), 'image_name'),
                                     lexer.index, 'image_name')
            entry['image_name_bytes'] = raw_name
            entry['image_name'] = _string_text(raw_name)
            tokens = lexer.expect_kv(lexer.next(), 'image_dims')
            if len(tokens) != 3:
                lexer.fail('image_dims field count')
            entry['width'] = _decimal(tokens[1], lexer.index, minimum=1)
            entry['height'] = _decimal(tokens[2], lexer.index, minimum=1)
            tokens = lexer.expect_kv(lexer.next(), 'max_image_size')
            if len(tokens) != 2:
                lexer.fail('max_image_size field count')
            entry['max_image_size'] = _decimal(tokens[1], lexer.index, signed=True, bits=32)
            for key, width in (('calib_k', 9), ('calib_r', 9), ('calib_t', 3)):
                tokens = lexer.expect_kv(lexer.next(), key)
                if len(tokens) != 1 + width:
                    lexer.fail('%s must carry %d floats' % (key, width))
                entry[key + '_hex'] = tuple(tokens[1:])
                entry[key] = tuple(_float32(t, lexer.index) for t in tokens[1:])
            raw_path = _string_bytes(lexer.expect_kv(lexer.next(), 'image_path'),
                                     lexer.index, 'image_path')
            entry['image_path_bytes'] = raw_path
            entry['image_path'] = _string_text(raw_path)
            entry['image_identity'] = _identity(lexer.expect_kv(lexer.next(), 'image_identity'),
                                                lexer.index, 'image_identity')
            raw_depth = _string_bytes(lexer.expect_kv(lexer.next(), 'depth_path'),
                                      lexer.index, 'depth_path')
            entry['depth_path'] = _string_text(raw_depth)
            entry['depth_identity'] = _identity(lexer.expect_kv(lexer.next(), 'depth_identity'),
                                                lexer.index, 'depth_identity')
            tokens = lexer.expect_kv(lexer.next(), 'depth_header')
            if len(tokens) != 4:
                lexer.fail('depth_header field count')
            entry['depth_header'] = tuple(_decimal(t, lexer.index, minimum=1)
                                          for t in tokens[1:])
            raw_normal = _string_bytes(lexer.expect_kv(lexer.next(), 'normal_path'),
                                       lexer.index, 'normal_path')
            entry['normal_path'] = _string_text(raw_normal)
            entry['normal_identity'] = _identity(lexer.expect_kv(lexer.next(), 'normal_identity'),
                                                 lexer.index, 'normal_identity')
            tokens = lexer.expect_kv(lexer.next(), 'normal_header')
            if len(tokens) != 4:
                lexer.fail('normal_header field count')
            entry['normal_header'] = tuple(_decimal(t, lexer.index, minimum=1)
                                           for t in tokens[1:])
            inputs.append(entry)
        references = [index for index, entry in enumerate(inputs) if entry['is_reference']]
        if len(references) != 1:
            lexer.fail('problem %d: exactly one reference input required, got %d'
                       % (number, len(references)))
        if inputs[references[0]]['image_idx'] != reference_image_idx:
            lexer.fail('problem %d: reference input image_idx %d != reference_image_idx %d'
                       % (number, inputs[references[0]]['image_idx'], reference_image_idx))
        problems.append({'reference_image_idx': reference_image_idx,
                         'reference_name': reference_name,
                         'output_depth_path': output_depth_path,
                         'output_normal_path': output_normal_path,
                         'options': options, 'inputs': inputs})
    if not lexer.at_end():
        lexer.fail('trailing content after the contract')
    return {'raw': data, 'sha256': hashlib.sha256(data).hexdigest(),
            'version': CONTRACT_VERSION,
            'build_cuda_enabled': cuda, 'build_hip_enabled': hip,
            'allocation_compatibility': allocation,
            'invocation_options': invocation_options,
            'stereo_folder': stereo_folder,
            'problems': problems}


def _layer_identity(entry: dict) -> str:
    """Bind image + depth + normal stat identities AND K/R/T (never image alone)."""
    parts = []
    for key in ('image_identity', 'depth_identity', 'normal_identity'):
        identity = entry[key]
        parts.extend([key, identity['path_bytes'].hex(), str(identity['size']),
                      str(identity['device']), str(identity['inode']),
                      str(identity['modified_seconds']), str(identity['modified_nanoseconds']),
                      str(identity['changed_seconds']), str(identity['changed_nanoseconds'])])
    for key in ('calib_k_hex', 'calib_r_hex', 'calib_t_hex'):
        parts.extend([key, *entry[key]])
    return 'contract-layer:' + hashlib.sha256('|'.join(parts).encode()).hexdigest()


def _decode_dims_for(entry: dict, image_decode_dims) -> tuple:
    """Original JPEG decode dims: required from the caller, never assumed."""
    if callable(image_decode_dims):
        dims = image_decode_dims(entry['image_name'])
    elif isinstance(image_decode_dims, dict):
        dims = image_decode_dims.get(entry['image_name'])
    else:
        dims = image_decode_dims
    if not isinstance(dims, (list, tuple)) or len(dims) != 2:
        raise MalformedDeclaration(
            'original decode dims are not carried by the contract and must be supplied '
            'per image (missing for %r); refusing the unknown resize term'
            % entry['image_name'])
    width, height = dims
    if any(type(value) is not int or value < 1 for value in dims):
        raise MalformedDeclaration('decode dims must be positive ints for %r'
                                  % entry['image_name'])
    if width < entry['width'] or height < entry['height']:
        raise MalformedDeclaration('decode dims %r smaller than the effective dims (%d, %d) for %r'
                            % (dims, entry['width'], entry['height'], entry['image_name']))
    return int(width), int(height)


def contract_jobs(contract: dict, *, image_decode_dims, image_channels,
                  images_count=None, size_projection=None) -> list:
    """Canonical-estimator descriptors for every problem, ordered as captured.

    ``width``/``height`` are the contract's EFFECTIVE image dims; the payload dims
    are the map-header dims; ``image_spec`` uses the caller-supplied ORIGINAL decode
    dims and channel count. Identity binds image+depth+normal stat and K/R/T.
    """
    if type(image_channels) is not int or image_channels < 1:
        raise MalformedDeclaration('image_channels must be an explicit positive int')
    jobs = []
    for problem in contract['problems']:
        inputs = []
        for entry in problem['inputs']:
            depth_w, depth_h, depth_c = entry['depth_header']
            normal_w, normal_h, normal_c = entry['normal_header']
            if depth_c != 1 or normal_c != 3:
                raise ContractError('input %s: map header channels must be depth=1/normal=3, '
                                    'got %d/%d' % (entry['image_name'], depth_c, normal_c))
            if depth_w < entry['width'] or depth_h < entry['height']:
                raise ContractError('input %s: depth header %dx%d is smaller than the '
                                    'effective dims %dx%d'
                                    % (entry['image_name'], depth_w, depth_h,
                                       entry['width'], entry['height']))
            decode_w, decode_h = _decode_dims_for(entry, image_decode_dims)
            if size_projection is None:
                effective_w, effective_h = entry['width'], entry['height']
                max_image_size = entry['max_image_size']
            else:
                # WHOLE-INVOCATION projection from the DECLARED ORIGINAL model geometry:
                # never from the dims as captured (that would double-resize)
                size = size_projection['max_image_size']
                originals = size_projection['model_dims']
                if entry['image_name'] not in originals:
                    raise ContractError('projection needs the declared original model dims for %r'
                                        % entry['image_name'])
                effective_w, effective_h = project_image_dims(*originals[entry['image_name']], size)
                if effective_w < 1 or effective_h < 1 or effective_w > INT_MAX \
                        or effective_h > INT_MAX:
                    raise ContractError('projecting %r to max_image_size %d yields an invalid '
                                        'geometry %rx%r (no clamp is invented)'
                                        % (entry['image_name'], size, effective_w, effective_h))
                max_image_size = size
            inputs.append({'image_idx': entry['image_idx'],
                           'identity': _layer_identity(entry),
                           'layer_key': entry['image_name'],
                           'width': effective_w, 'height': effective_h,
                           'max_image_size': max_image_size,
                           'image_spec': (decode_w, decode_h, image_channels),
                           'depth_payload': (depth_w, depth_h),
                           'normal_payload': (normal_w, normal_h)})
        job = {'reference_image_idx': problem['reference_image_idx'],
               'source_image_idxs': [entry['image_idx'] for entry in problem['inputs']
                                     if not entry['is_reference']],
               'max_image_size': (problem['options']['max_image_size']
                                  if size_projection is None
                                  else size_projection['max_image_size']),
               'inputs': inputs}
        if images_count is not None:
            if isinstance(images_count, dict):
                count = images_count.get(problem['reference_image_idx'])
            else:
                count = images_count
            if type(count) is not int or count < 1:
                raise ContractError('images_count must be an explicit positive int (or a '
                                    'mapping keyed by reference_image_idx)')
            job['images_count'] = count
        jobs.append(job)
    return jobs


def contract_identity_differences(a: dict, b: dict) -> list:
    """Ordered identity/option differences between two contracts (verification mode)."""
    differences = []
    if a['sha256'] != b['sha256']:
        differences.append('contract sha256 differs: %s vs %s' % (a['sha256'], b['sha256']))
    for _, name in OPTION_SCHEMA:
        if a['invocation_options'].get(name) != b['invocation_options'].get(name):
            differences.append('invocation option %s changed' % name)
    if len(a['problems']) != len(b['problems']):
        differences.append('problem count %d != %d' % (len(a['problems']), len(b['problems'])))
    for index, (left, right) in enumerate(zip(a['problems'], b['problems'])):
        if left['reference_image_idx'] != right['reference_image_idx']:
            differences.append('problem %d reference_image_idx changed' % index)
        if left['reference_name'] != right['reference_name']:
            differences.append('problem %d reference_name changed' % index)
        if left['options'] != right['options']:
            differences.append('problem %d effective options changed' % index)
        if len(left['inputs']) != len(right['inputs']):
            differences.append('problem %d input count %d != %d'
                               % (index, len(left['inputs']), len(right['inputs'])))
            continue
        for ordinal, (one, two) in enumerate(zip(left['inputs'], right['inputs'])):
            if one['image_idx'] != two['image_idx']:
                differences.append('problem %d input %d image_idx changed' % (index, ordinal))
            if one['image_name'] != two['image_name']:
                differences.append('problem %d input %d image_name changed' % (index, ordinal))
            if (one['width'], one['height']) != (two['width'], two['height']):
                differences.append('problem %d input %d effective dims changed' % (index, ordinal))
            if one['calib_k_hex'] != two['calib_k_hex'] or one['calib_r_hex'] != two['calib_r_hex'] \
                    or one['calib_t_hex'] != two['calib_t_hex']:
                differences.append('problem %d input %d calibration changed' % (index, ordinal))
            for key in ('image_identity', 'depth_identity', 'normal_identity'):
                if one[key] != two[key]:
                    differences.append('problem %d input %d %s changed' % (index, ordinal, key))
    return differences


def forecast_contract(contract: dict, *, expected_sha256=None, mode='eager',
                      max_future=2, unclassified_allowance_bytes=None,
                      image_channels=None, image_decode_dims=None, active_jobs=None,
                      bank_metadata_mode='planning_envelope', images_count=None,
                      metadata_sizeofs=None, metadata_planner_high_water=0,
                      metadata_path_bytes=None, metadata_unclassified_bytes=None,
                      metadata_omitted_reason=None,
                      process_mode='once', owners=None) -> dict:
    """Structured forecast through the EXISTING canonical formulas (no new math).

    Every accounting assumption is an explicit, validated input: the unclassified
    allowance, the image channel count, the original decode dims, the concurrency
    (``active_jobs``) and the metadata mode (measured sizeofs, explicit unclassified
    bytes, or an explicitly recorded omission reason). ``admission_authorized`` is
    always False: a metadata-only estimate, not an execution certificate.
    """
    if expected_sha256 is not None and contract['sha256'] != expected_sha256:
        raise ContractError('contract sha256 %s does not match the expected %s'
                            % (contract['sha256'], expected_sha256))
    if unclassified_allowance_bytes is None:
        raise ContractError('unclassified_allowance_bytes is required: the unexplained '
                            'construction allowance must be an explicit caller input')
    if type(unclassified_allowance_bytes) is not int or unclassified_allowance_bytes < 0:
        raise ContractError('unclassified_allowance_bytes must be an explicit nonnegative int')
    if image_channels is None:
        raise ContractError('image_channels is required: the contract carries no channel '
                            'count, so the caller must declare it')
    if active_jobs is None:
        raise ContractError('active_jobs is required: the concurrency of the private '
                            'reserves must be declared. Explicit 0 means a declared '
                            'bank-only scope; an unset value is refused rather than '
                            'silently billed as zero.')
    if type(active_jobs) is not int or active_jobs < 0:
        raise ContractError('active_jobs must be an explicit nonnegative int')
    if bank_metadata_mode not in ('planning_envelope', 'native_sizeof'):
        raise ContractError("bank_metadata_mode must be 'planning_envelope' or 'native_sizeof'")
    jobs = contract_jobs(contract, image_decode_dims=image_decode_dims,
                         image_channels=image_channels, images_count=images_count)
    if active_jobs > len(jobs):
        raise ContractError('active_jobs %d exceeds the %d members; concurrent-reserve slicing '
                            'would be misleading' % (active_jobs, len(jobs)))
    bank = model.estimate_owned_bank(jobs, mode=mode,
                                     unclassified_allowance_bytes=unclassified_allowance_bytes,
                                     metadata_mode=bank_metadata_mode)
    if bank['unverified_identity_layers']:
        raise ContractError('unverified identity layers: %d' % bank['unverified_identity_layers'])
    normal = model.estimate_owned_normal_allowance(jobs, max_future)
    privates = [model.estimate_owned_private_job(job, mode=mode) for job in jobs]
    modes = [name for name, value in (('measured_sizeofs', metadata_sizeofs is not None),
                                      ('explicit_unclassified',
                                       metadata_unclassified_bytes is not None),
                                      ('omitted_declared', metadata_omitted_reason is not None))
             if value]
    if len(modes) != 1:
        raise ContractError('exactly one metadata mode must be declared (measured sizeofs / '
                            'explicit unclassified bytes / omitted with a recorded reason); '
                            'got %r' % (modes,))
    if modes[0] == 'measured_sizeofs':
        if metadata_path_bytes is None:
            raise ContractError('measured sizeofs require metadata_path_bytes (canonical path '
                                'lengths) or an explicit omission is not allowed')
        metadata = model.estimate_cohort_metadata(
            members=jobs, inputs_per_job=[len(job['inputs']) for job in jobs],
            layer_expected_count=len(model._owned_resolve(jobs)['layers']),
            snapshot_entry_count=len(jobs),
            planner_high_water_bytes=metadata_planner_high_water,
            sizeofs=metadata_sizeofs, path_bytes=list(metadata_path_bytes))
        metadata_charge = model.margin_bytes(metadata['raw_bytes'])
    elif modes[0] == 'explicit_unclassified':
        if type(metadata_unclassified_bytes) is not int or metadata_unclassified_bytes < 0:
            raise ContractError('metadata_unclassified_bytes must be a nonnegative int')
        metadata = {'raw_bytes': metadata_unclassified_bytes,
                    'sizeofs_source': 'explicit_unclassified'}
        metadata_charge = model.margin_bytes(metadata_unclassified_bytes)
    else:
        if not isinstance(metadata_omitted_reason, str) or not metadata_omitted_reason.strip():
            raise ContractError('a metadata omission needs a recorded reason')
        metadata = {'raw_bytes': 0, 'sizeofs_source': 'omitted_declared',
                    'omission_reason': metadata_omitted_reason}
        metadata_charge = 0
    if process_mode == 'once':
        process = {'mode': 'once', 'count': 1,
                   'host_bytes': model.OWNED_PROCESS_CHARGE_BYTES,
                   'device_bytes': model.OWNED_PROCESS_CHARGE_BYTES}
    elif process_mode == 'per_owner':
        if type(owners) is not int or owners < 1:
            raise ContractError("process_mode 'per_owner' requires an explicit owners>=1")
        process = {'mode': 'per_owner', 'count': owners,
                   'host_bytes': owners * model.OWNED_PROCESS_CHARGE_BYTES,
                   'device_bytes': owners * model.OWNED_PROCESS_CHARGE_BYTES}
    else:
        raise ContractError("process_mode must be 'once' or 'per_owner'")
    per_reference = []
    for problem, private in zip(contract['problems'], privates):
        per_reference.append({'reference_name': problem['reference_name'],
                              'reference_image_idx': problem['reference_image_idx'],
                              'input_count': len(problem['inputs']),
                              'private_host_bytes': private['host_charge_bytes'],
                              'private_device_bytes': private['device_charge_bytes'],
                              'effective_options': dict(problem['options'])})
    partial = modes[0] == 'omitted_declared'
    return {'contract_sha256': contract['sha256'],
            'contract_raw_bytes': contract['raw'],
            'contract_version': contract['version'],
            'build_cuda_enabled': contract['build_cuda_enabled'],
            'build_hip_enabled': contract['build_hip_enabled'],
            'allocation_compatibility': contract['allocation_compatibility'],
            'admission_authorized': False,
            'completeness': 'partial' if partial else 'complete',
            'completeness_note': ('the metadata term is declared omitted; this charge '
                                  'combination is NOT a complete total'
                                  if partial else None),
            'complete_total_available': not partial,
            'concurrency_basis': 'bank_only' if active_jobs == 0
                                 else 'rolling_reserve_declared',
            'qualification': ('metadata_only_cpu_estimate_partial' if partial
                              else 'metadata_only_cpu_estimate'),
            'assumptions': {'unclassified_allowance_bytes': unclassified_allowance_bytes,
                            'metadata_mode': modes[0],
                            'metadata_omitted_reason': metadata_omitted_reason,
                            'active_jobs': active_jobs,
                            'image_channels': image_channels,
                            'decode_dims': 'caller_supplied_original' if not callable(image_decode_dims)
                                           else 'caller_supplied_callable'},
            'mode': mode,
            'reference_count': len(contract['problems']),
            'bank_estimate_host_bytes': bank['host_charge_bytes'],
            'bank_estimate_device_bytes': bank['device_charge_bytes'],
            'normal_owner_bytes': normal['owner_bytes'],
            'normal_future_bytes': normal['future_bytes'],
            'normal_transient_bytes': normal['transient_bytes'],
            'normal_charge_host_bytes': normal['ready_plus_producer_charge_bytes'],
            'max_future': max_future,
            'metadata_raw_bytes': metadata['raw_bytes'],
            'metadata_charge_host_bytes': metadata_charge,
            'metadata_source': metadata['sizeofs_source'],
            'process': process,
            'planner_max_member_host_bytes': max(item['host_charge_bytes'] for item in privates),
            'planner_max_member_device_bytes': max(item['device_charge_bytes'] for item in privates),
            'rolling_reserve_host_bytes': sum(sorted((item['host_charge_bytes'] for item in privates),
                                                     reverse=True)[:active_jobs]),
            'rolling_reserve_device_bytes': sum(sorted((item['device_charge_bytes']
                                                        for item in privates),
                                                       reverse=True)[:active_jobs]),
            'rolling_reserve_active_jobs': active_jobs,
            'per_reference': per_reference}


# ---------------------------------------------------------------------------
# Slice-B wire contract: resident-forecast v1 (line protocol, integers only).
#
# Versioned native field contract:
#   * fixed keys/order/counts, version-bound; decimal integers; length-prefixed
#     hex byte strings; reference names are the EXACT native contract model-image
#     name BYTES (surrogateescape, reversible); indices/order identical to the
#     contract;
#   * the raw contract bytes are carried (length + hex) so the native side can
#     compare a regenerated serialized snapshot exactly -- a hash alone cannot
#     satisfy the byte-preserved binding promise;
#   * the two fingerprint fields carry the SAME allocation-closure SHA-256 bytes
#     (generated at BUILD by the source writer from the frozen explicit closure);
#     `absent` is legal ONLY for provisional, non-executable forecacts. The
#     emitter refuses execution-qualified emission for a fingerprint that is not
#     in the supported formula/content registry;
#   * caps are deliberately absent (CLI/policy); there is no admission boolean.
#
# The runner compares both wire digests to the actual compiled digest, checks the
# exact contract binding and recomputes logical minima / the complete ledger. It
# never trusts a boolean and never loads a second mutable registry.
# ---------------------------------------------------------------------------

FORECAST_MAGIC = 'resident-forecast'
FORECAST_VERSION = 1
FORECAST_QUALIFICATION_PROVISIONAL = 'metadata_only_cpu_estimate_provisional'
FORECAST_QUALIFICATION_EXECUTABLE = 'metadata_only_cpu_estimate'
FORECAST_FINGERPRINT_ABSENT = 'absent'

# Versioned exact digest set, one entry per closure version. Old closures are
# preserved; nothing here is mutable at runtime and the native side never loads
# a second registry -- the runner compares the wire digests to its compiled one.
FORECAST_REGISTRY_EXPERIMENTAL = 'experimental_compatibility'
FORECAST_REGISTRY_QUALIFIED = 'qualified'
FORECAST_REGISTRY_STATUSES = (FORECAST_REGISTRY_EXPERIMENTAL, FORECAST_REGISTRY_QUALIFIED)
FORECAST_REGISTRY_SCOPE_FULL_BANK = 'full_bank_allocation'
FORECAST_REGISTRY_SCOPE_COHORT = 'cohort_allocation'

# Versioned exact digest set, one entry per closure version. Old closures are
# preserved; the native side never loads this registry. An entry certifies the
# FORMULA/SOURCE COMPATIBILITY of that exact closure only.
# Formula/content support only. Performance, quality and physical fit are NOT
# certified. The native runner independently checks this exact compiled closure.
SUPPORTED_ALLOCATION_FINGERPRINT = '0614ab2b8cbe1e7cbeec2cc90b51b10275d7eea39b9c3f20a6651d7bb2879ecb'
ALLOCATION_CLOSURE_REGISTRY = {
    SUPPORTED_ALLOCATION_FINGERPRINT: {
        'status': FORECAST_REGISTRY_EXPERIMENTAL,
        'scope': FORECAST_REGISTRY_SCOPE_FULL_BANK,
        'certifies': 'bounded full-bank formula/content compatibility only',
    },
}


def allocation_closure_fingerprint(entries) -> str:
    """SHA-256 over an ordered frozen closure: path + NUL + file bytes + NUL.

    ``entries`` is an ordered sequence of ``(repo_relative_path, file_bytes)``.
    Paths must be repo-relative (no absolute paths, no parent traversal); the
    ordered list is the caller's/source writer's frozen declaration. The
    generated header is excluded by the generator, not here, so the digest cannot
    recurse into itself.
    """
    digest = hashlib.sha256()
    ordered = list(entries)
    if not ordered:
        raise ContractError('a closure fingerprint needs at least one ordered entry')
    for path, content in ordered:
        if not isinstance(path, str) or not path:
            raise ContractError('closure entries need nonempty repo-relative paths')
        if path.startswith('/') or '..' in path.split('/') or '\\' in path:
            raise ContractError('closure paths must be repo-relative: %r' % path)
        if not isinstance(content, (bytes, bytearray)):
            raise ContractError('closure entries need file bytes for %r' % path)
        digest.update(path.encode('utf-8'))
        digest.update(b'\0')
        digest.update(bytes(content))
        digest.update(b'\0')
    return digest.hexdigest()


def _line_int(name, value):
    if type(value) is not int or value < 0:
        raise ContractError('%s must be a nonnegative int' % name)
    return '%s %d\n' % (name, value)


def _line_bytes(name, text):
    raw = text.encode('utf-8', 'surrogateescape')
    return '%s %d %s\n' % (name, len(raw), raw.hex())


def _line_fingerprint(name, value):
    if value is None:
        return '%s %s\n' % (name, FORECAST_FINGERPRINT_ABSENT)
    raw = bytes.fromhex(value)
    if len(raw) != 32:
        raise ContractError('%s must be a 32-byte (SHA-256) digest' % name)
    return _line_bytes(name, value)


def serialize_resident_forecast(forecast: dict, *, unclassified_provenance,
                                execution_qualified=False, registry=None,
                                compiled_allocation_fingerprint=None) -> bytes:
    """Serialize a ``forecast_contract`` result to the wire contract.

    ``execution_qualified`` requires the allocation-closure fingerprint to be in
    the versioned registry AND to equal the runner's compiled fingerprint. Unknown formula closures refuse; compatibility is not a performance or
    quality qualification.
    """
    if not isinstance(unclassified_provenance, str) or not unclassified_provenance.strip():
        raise ContractError('unclassified_provenance is required (never omit the allowance '
                            'provenance)')
    raw = forecast.get('contract_raw_bytes')
    if not isinstance(raw, (bytes, bytearray)):
        raise ContractError('the forecast must carry the raw contract bytes')
    active = forecast.get('qualification')
    if execution_qualified:
        registry = ALLOCATION_CLOSURE_REGISTRY if registry is None else registry
        if not isinstance(compiled_allocation_fingerprint, str):
            raise ContractError('execution-qualified emission needs the runner compiled '
                                'allocation fingerprint')
        entry = (registry or {}).get(compiled_allocation_fingerprint)
        if not entry:
            raise ContractError('allocation closure fingerprint %s is not in the reviewed '
                                'registry; execution-qualified emission refuses'
                                % compiled_allocation_fingerprint)
        if not isinstance(entry, dict) or entry.get('status') not in FORECAST_REGISTRY_STATUSES:
            raise ContractError('registry entry %s has an unknown or revoked status (%r); '
                                'refusing to emit an executable forecast'
                                % (compiled_allocation_fingerprint,
                                   entry.get('status') if isinstance(entry, dict) else entry))
        digest = compiled_allocation_fingerprint
        if forecast.get('completeness') != 'complete' or \
                forecast.get('concurrency_basis') != 'rolling_reserve_declared':
            raise ContractError('execution-qualified emission refuses partial or bank-only '
                                'forecasts')
        qualification = FORECAST_QUALIFICATION_EXECUTABLE
    else:
        # Provisional emission always carries the provisional label, including for
        # COMPLETE accounting: completeness stays the caller's fact, fingerprints stay
        # `absent`, and nothing executable can be laundered through this path. Native
        # execution refuses every non-executable label, so an unregistered forecast is
        # readable but never executable.
        digest = None
        qualification = FORECAST_QUALIFICATION_PROVISIONAL
    out = [FORECAST_MAGIC + '\n', 'version %d\n' % FORECAST_VERSION,
           'contract_sha256 %s\n' % forecast['contract_sha256'],
           'contract_version %d\n' % forecast['contract_version'],
           _line_int('contract_raw_length', len(raw)),
           'contract_raw %d %s\n' % (len(raw), bytes(raw).hex()),
           _line_bytes('qualification', qualification),
           _line_bytes('completeness', forecast['completeness']),
           _line_bytes('concurrency_basis', forecast['concurrency_basis']),
           _line_bytes('mode', forecast['mode']),
           _line_int('max_future', forecast['max_future']),
           _line_int('image_channels', forecast['assumptions']['image_channels']),
           '%s %d %s\n' % ('unclassified_allowance_bytes',
                           forecast['assumptions']['unclassified_allowance_bytes'],
                           _string_bytes_tokens(unclassified_provenance)),
           _line_fingerprint('build_allocation_fingerprint', digest),
           _line_fingerprint('source_fingerprint', digest),
           'charges\n',
           'bank host %d device %d\n' % (forecast['bank_estimate_host_bytes'],
                                         forecast['bank_estimate_device_bytes']),
           'normal owner %d future %d transient %d charge %d\n'
           % (forecast['normal_owner_bytes'], forecast['normal_future_bytes'],
              forecast['normal_transient_bytes'], forecast['normal_charge_host_bytes']),
           'process mode %d %s count %d host %d device %d\n'
           % (*_string_pair(forecast['process']['mode']), forecast['process']['count'],
              forecast['process']['host_bytes'], forecast['process']['device_bytes']),
           'metadata source %d %s raw %d charge %d\n'
           % (*_string_pair(forecast['metadata_source']), forecast['metadata_raw_bytes'],
              forecast['metadata_charge_host_bytes']),
           'planner max_member host %d device %d\n'
           % (forecast['planner_max_member_host_bytes'],
              forecast['planner_max_member_device_bytes']),
           'rolling active_jobs %d host %d device %d\n'
           % (forecast['rolling_reserve_active_jobs'],
              forecast['rolling_reserve_host_bytes'],
              forecast['rolling_reserve_device_bytes']),
           'reference_count %d\n' % forecast['reference_count']]
    for ordinal, item in enumerate(forecast['per_reference']):
        out.append('reference %d\n' % ordinal)
        out.append(_line_int('reference_image_idx', item['reference_image_idx']))
        out.append(_line_bytes('reference_name', item['reference_name']))
        out.append(_line_int('input_count', item['input_count']))
        out.append(_line_int('private_host_bytes', item['private_host_bytes']))
        out.append(_line_int('private_device_bytes', item['private_device_bytes']))
    out.append('end\n')
    return ''.join(out).encode('ascii')


def _string_pair(text):
    raw = text.encode('utf-8', 'surrogateescape')
    return len(raw), raw.hex()


def _string_bytes_tokens(text):
    length, hexed = _string_pair(text)
    return '%d %s' % (length, hexed)


def parse_resident_forecast(data: bytes) -> dict:
    """Strict parser for the wire contract (mirrors the native strictness)."""
    lexer = _Lexer(data)
    lexer.expect(lexer.next(), FORECAST_MAGIC)
    tokens = lexer.expect_kv(lexer.next(), 'version')
    if len(tokens) != 2 or _decimal(tokens[1], lexer.index) != FORECAST_VERSION:
        lexer.fail('unsupported forecast version')
    tokens = lexer.expect_kv(lexer.next(), 'contract_sha256')
    if len(tokens) != 2 or len(tokens[1]) != 64:
        lexer.fail('contract_sha256 must be 64 lowercase hex characters')
    contract_sha256 = tokens[1]
    _hex(contract_sha256, lexer.index, 32)
    tokens = lexer.expect_kv(lexer.next(), 'contract_version')
    if len(tokens) != 2 or _decimal(tokens[1], lexer.index) != CONTRACT_VERSION:
        lexer.fail('unsupported contract version binding')
    tokens = lexer.expect_kv(lexer.next(), 'contract_raw_length')
    if len(tokens) != 2:
        lexer.fail('contract_raw_length field count')
    raw_length = _decimal(tokens[1], lexer.index, minimum=1)
    tokens = lexer.expect_kv(lexer.next(), 'contract_raw')
    if len(tokens) != 3:
        lexer.fail('contract_raw field count')
    raw = _hex(tokens[2], lexer.index, _decimal(tokens[1], lexer.index))
    if len(raw) != raw_length:
        lexer.fail('contract_raw length does not match contract_raw_length')
    if hashlib.sha256(raw).hexdigest() != contract_sha256:
        lexer.fail('contract_raw does not hash to the bound contract_sha256')
    result = {'raw': data, 'contract_sha256': contract_sha256,
              'contract_raw_bytes': raw, 'contract_raw_length': raw_length,
              'contract_version': CONTRACT_VERSION}
    for key in ('qualification', 'completeness', 'concurrency_basis', 'mode'):
        result[key] = _string_text(_string_bytes(
            lexer.expect_kv(lexer.next(), key), lexer.index, key))
    tokens = lexer.expect_kv(lexer.next(), 'max_future')
    if len(tokens) != 2:
        lexer.fail('max_future field count')
    result['max_future'] = _decimal(tokens[1], lexer.index, minimum=1, bits=8,
                                    signed=False)
    tokens = lexer.expect_kv(lexer.next(), 'image_channels')
    if len(tokens) != 2:
        lexer.fail('image_channels field count')
    result['image_channels'] = _decimal(tokens[1], lexer.index, minimum=1, bits=8,
                                        signed=False)
    tokens = lexer.expect_kv(lexer.next(), 'unclassified_allowance_bytes')
    if len(tokens) != 4:
        lexer.fail('unclassified_allowance_bytes must carry value + provenance')
    result['unclassified_allowance_bytes'] = _decimal(tokens[1], lexer.index)
    result['unclassified_provenance'] = _string_text(
        _hex(tokens[3], lexer.index, _decimal(tokens[2], lexer.index)))
    for key in ('build_allocation_fingerprint', 'source_fingerprint'):
        tokens = lexer.expect_kv(lexer.next(), key)
        if len(tokens) == 2 and tokens[1] == FORECAST_FINGERPRINT_ABSENT:
            result[key] = None
        else:
            text = _string_text(_string_bytes(tokens, lexer.index, key))
            if len(text) != 64:
                lexer.fail('%s must be a 64-character lowercase-hex digest TEXT or %r'
                           % (key, FORECAST_FINGERPRINT_ABSENT))
            _hex(text, lexer.index, 32)
            result[key] = text
    if result['qualification'] == FORECAST_QUALIFICATION_EXECUTABLE and (
            result['build_allocation_fingerprint'] is None
            or result['source_fingerprint'] is None):
        lexer.fail('an executable qualification cannot carry absent fingerprints')
    if result['build_allocation_fingerprint'] != result['source_fingerprint']:
        lexer.fail('build_allocation_fingerprint and source_fingerprint must carry the '
                   'same allocation-closure digest')
    lexer.expect(lexer.next(), 'charges')
    tokens = lexer.expect_kv(lexer.next(), 'bank')
    if len(tokens) != 5 or tokens[1] != 'host' or tokens[3] != 'device':
        lexer.fail('bank charge field layout')
    result['bank_estimate_host_bytes'] = _decimal(tokens[2], lexer.index)
    result['bank_estimate_device_bytes'] = _decimal(tokens[4], lexer.index)
    tokens = lexer.expect_kv(lexer.next(), 'normal')
    if len(tokens) != 9 or tokens[1] != 'owner' or tokens[3] != 'future' \
            or tokens[5] != 'transient' or tokens[7] != 'charge':
        lexer.fail('normal charge field layout')
    for index, name in ((2, 'normal_owner_bytes'), (4, 'normal_future_bytes'),
                        (6, 'normal_transient_bytes'), (8, 'normal_charge_host_bytes')):
        result[name] = _decimal(tokens[index], lexer.index)
    tokens = lexer.expect_kv(lexer.next(), 'process')
    if len(tokens) != 10 or tokens[1] != 'mode' or tokens[4] != 'count' \
            or tokens[6] != 'host' or tokens[8] != 'device':
        lexer.fail('process charge field layout')
    result['process'] = {'mode': _string_text(_hex(tokens[3], lexer.index,
                                                    _decimal(tokens[2], lexer.index))),
                         'count': _decimal(tokens[5], lexer.index),
                         'host_bytes': _decimal(tokens[7], lexer.index),
                         'device_bytes': _decimal(tokens[9], lexer.index)}
    tokens = lexer.expect_kv(lexer.next(), 'metadata')
    if len(tokens) != 8 or tokens[1] != 'source' or tokens[4] != 'raw' or tokens[6] != 'charge':
        lexer.fail('metadata charge field layout')
    result['metadata_source'] = _string_text(_hex(tokens[3], lexer.index,
                                                  _decimal(tokens[2], lexer.index)))
    result['metadata_raw_bytes'] = _decimal(tokens[5], lexer.index)
    result['metadata_charge_host_bytes'] = _decimal(tokens[7], lexer.index)
    tokens = lexer.expect_kv(lexer.next(), 'planner')
    if len(tokens) != 6 or tokens[1] != 'max_member' or tokens[2] != 'host' \
            or tokens[4] != 'device':
        lexer.fail('planner charge field layout')
    result['planner_max_member_host_bytes'] = _decimal(tokens[3], lexer.index)
    result['planner_max_member_device_bytes'] = _decimal(tokens[5], lexer.index)
    tokens = lexer.expect_kv(lexer.next(), 'rolling')
    if len(tokens) != 7 or tokens[1] != 'active_jobs' or tokens[3] != 'host' \
            or tokens[5] != 'device':
        lexer.fail('rolling charge field layout')
    result['rolling_reserve_active_jobs'] = _decimal(tokens[2], lexer.index)
    result['rolling_reserve_host_bytes'] = _decimal(tokens[4], lexer.index)
    result['rolling_reserve_device_bytes'] = _decimal(tokens[6], lexer.index)
    tokens = lexer.expect_kv(lexer.next(), 'reference_count')
    if len(tokens) != 2:
        lexer.fail('reference_count field count')
    reference_count = _decimal(tokens[1], lexer.index)
    references = []
    for ordinal in range(reference_count):
        tokens = lexer.expect_kv(lexer.next(), 'reference')
        if len(tokens) != 2 or _decimal(tokens[1], lexer.index) != ordinal:
            lexer.fail('reference ordinals must be contiguous from 0')
        entry = {}
        tokens = lexer.expect_kv(lexer.next(), 'reference_image_idx')
        if len(tokens) != 2:
            lexer.fail('reference_image_idx field count')
        entry['reference_image_idx'] = _decimal(tokens[1], lexer.index, signed=True,
                                                bits=32, minimum=0)
        entry['reference_name'] = _string_text(_string_bytes(
            lexer.expect_kv(lexer.next(), 'reference_name'), lexer.index, 'reference_name'))
        for key in ('input_count', 'private_host_bytes', 'private_device_bytes'):
            tokens = lexer.expect_kv(lexer.next(), key)
            if len(tokens) != 2:
                lexer.fail('%s field count' % key)
            entry[key] = _decimal(tokens[1], lexer.index, minimum=1 if key == 'input_count'
                                  else 0)
        references.append(entry)
    lexer.expect(lexer.next(), 'end')
    if not lexer.at_end():
        lexer.fail('trailing content after the forecast')
    result['reference_count'] = reference_count
    result['per_reference'] = references
    return result
