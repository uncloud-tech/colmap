# SPDX-License-Identifier: BSD-3-Clause
"""Pinned 1af44f37 PHOTO V2/V3 wire adapter, not an authorization mechanism.

Grammar: optimized_photo_forecast.h:26-102,116-187,201-278. Linux/POSIX paths.
Provenance stays OUTSIDE the native wire grammar. Native snapshots are trusted
caller evidence; a label/hash is not proof of native origin or correct charges.
"""
import copy
import hashlib
import os
import re

from . import producer as p

U64_MAX = (1 << 64) - 1
INT_MAX = (1 << 31) - 1
BINDING_KEYS = ('profile', 'lineage_sha', 'fingerprint', 'source_commit',
                'options_sha256', 'process', 'private_charge',
                'reference_image_idx', 'source_image_idxs')


def _fail(reason):
    raise ValueError(reason)


def _lines(data):
    if not isinstance(data, bytes):
        _fail('wire-bytes-required')
    if data and not data.endswith(b'\n'):
        _fail('wire-unterminated-line')
    return data.split(b'\n')[:-1] if data else []


def _uint(value, name, maximum=U64_MAX, minimum=0):
    if type(value) is not int:
        _fail('wire-integer-type: ' + name)
    if not minimum <= value <= maximum:
        _fail('wire-number-range: ' + name)
    return value


def _numbers(raw, name, count=None, maximum=U64_MAX, minimum=0):
    if not re.fullmatch(rb'(0|[1-9][0-9]*)( (0|[1-9][0-9]*))*', raw):
        _fail('wire-noncanonical-number: ' + name)
    tokens = raw.split(b' ')
    if any(len(token) > 20 for token in tokens):
        _fail('wire-number-range: ' + name)
    values = [int(token) for token in tokens]
    if count is not None and len(values) != count:
        _fail('wire-number-count: ' + name)
    for value in values:
        _uint(value, name, maximum, minimum)
    return values


def _parse_section(lines, pos):
    values = {}
    while pos < len(lines):
        line = lines[pos]
        pos += 1
        if line == b'[END]':
            break
        if b'=' not in line:
            _fail('wire-binding-expected-key-value')
        key, value = line.split(b'=', 1)
        if key not in {name.encode() for name in BINDING_KEYS}:
            _fail('wire-binding-unknown-key')
        key = key.decode('ascii')
        if key in values:
            _fail('wire-binding-duplicate-key: ' + key)
        values[key] = value
    else:
        _fail('wire-binding-missing-terminator')
    for key in BINDING_KEYS:
        if key not in values:
            _fail('wire-binding-missing-key: ' + key)
        if not values[key]:
            _fail('wire-binding-empty-value: ' + key)
    binding = {}
    for key in BINDING_KEYS:
        raw = values[key]
        if key in ('process', 'private_charge'):
            host, device = _numbers(raw, key, count=2)
            binding[key] = dict(host=host, device=device)
        elif key == 'reference_image_idx':
            binding[key] = _numbers(raw, key, count=1, maximum=INT_MAX)[0]
        elif key == 'source_image_idxs':
            binding[key] = _numbers(raw, key, maximum=INT_MAX)
        else:
            binding[key] = raw.decode('utf-8', 'surrogateescape')
    return binding, pos


def parse_binding_section(data):
    """Parse a BODY (no [BINDING] prefix), including its final [END]\n.

    Unlike C++'s stream-prefix helper, this standalone API also refuses bytes
    after [END]. The envelope parser consumes consecutive framed sections.
    String semantics/digests are checked by the bound delivery, not this grammar.
    """
    lines = _lines(data)
    result, pos = _parse_section(lines, 0)
    if pos != len(lines):
        _fail('wire-section-trailing-bytes')
    return result


def emit_binding_section(binding):
    """Canonical typed section BODY; envelope adds the [BINDING] framing line."""
    if not isinstance(binding, dict):
        _fail('wire-binding-mapping-required')
    if set(binding) - set(BINDING_KEYS):
        _fail('wire-binding-unknown-key')
    if set(BINDING_KEYS) - set(binding):
        _fail('wire-binding-missing-key')
    lines = []
    for key in BINDING_KEYS:
        value = binding[key]
        if key in ('process', 'private_charge'):
            if not isinstance(value, dict) or set(value) != {'host', 'device'}:
                _fail('wire-number-count: ' + key)
            value = ' '.join(str(_uint(value[k], key)) for k in ('host', 'device'))
        elif key == 'reference_image_idx':
            value = str(_uint(value, key, INT_MAX))
        elif key == 'source_image_idxs':
            if not isinstance(value, list) or not value:
                _fail('wire-number-count: ' + key)
            value = ' '.join(str(_uint(idx, key, INT_MAX)) for idx in value)
        if not isinstance(value, str) or not value:
            _fail('wire-binding-empty-value: ' + key)
        if '\n' in value:
            _fail('wire-binding-line-injection: ' + key)
        lines.append(key.encode() + b'=' + value.encode('utf-8', 'surrogateescape'))
    result = b'\n'.join(lines) + b'\n[END]\n'
    if parse_binding_section(result) != binding:
        _fail('wire-binding-roundtrip-mismatch')
    return result


def _digest(value):
    if not isinstance(value, str) or not re.fullmatch('[0-9a-f]{64}', value):
        _fail('wire-malformed-digest')
    return value


def _absolute_path(path):
    if not isinstance(path, str) or '\0' in path or not path.startswith('/'):
        _fail('wire-noncanonical-absolute-path')
    # POSIX lexically-normal absolute names; a final slash is retained by C++.
    parts = path.split('/')[1:]
    if any(part in ('.', '..') for part in parts) or any(not part for part in parts[:-1]):
        _fail('wire-noncanonical-absolute-path')
    return path


def _unhex(raw):
    if not raw or len(raw) % 2 or not re.fullmatch(rb'[0-9a-f]+', raw):
        _fail('wire-invalid-hex')
    return bytes.fromhex(raw.decode('ascii'))


def _order(raw):
    values = _numbers(raw, 'v2-order')
    if len(values) < 3 or values[1] < 1 or values[1] != len(values) - 2:
        _fail('wire-v2-source-count')
    ref, sources = values[0], values[2:]
    for idx in [ref, *sources]:
        _uint(idx, 'v2-index', INT_MAX)
    if len(set([ref, *sources])) != len(sources) + 1:
        _fail('wire-v2-duplicate-reference-or-source')
    return ref, sources


def parse_v2_forecast(data):
    lines = _lines(data)
    if len(lines) < 5 or lines[0] != b'COLMAP_OPTIMIZED_PHOTO_V2':
        _fail('wire-invalid-v2-header')
    try:
        digest = _digest(lines[1].decode('ascii'))
    except UnicodeError:
        _fail('wire-malformed-digest')
    host, device, private_host, private_device = _numbers(lines[2], 'v2-charges', count=4, minimum=1)
    ref, sources = _order(lines[3])
    count = _numbers(lines[4], 'v2-file-count', count=1, minimum=1)[0]
    if count > len(data) // 67 or len(lines) != 5 + count:
        _fail('wire-v2-trailing-or-missing-files')
    files = {}
    for line in lines[5:]:
        if len(line) < 66 or line[64:65] != b' ':
            _fail('wire-invalid-v2-file')
        try:
            sha = _digest(line[:64].decode('ascii'))
        except UnicodeError:
            _fail('wire-malformed-digest')
        path = _absolute_path(os.fsdecode(_unhex(line[65:])))
        if path in files:
            _fail('wire-v2-duplicate-file')
        files[path] = sha
    return dict(options_sha256=digest, process=dict(host=host,device=device),
                private_charge=dict(host=private_host,device=private_device),
                reference_image_idx=ref, source_image_idxs=sources, files=files)


def emit_v2_forecast(binding, files):
    # Reuse section type checks, then enforce V2's additional semantic rules.
    emit_binding_section(binding)
    if not isinstance(files, dict) or not files:
        _fail('wire-v2-files-required')
    charges = [binding[key][side] for key in ('process','private_charge') for side in ('host','device')]
    for value in charges:
        _uint(value, 'v2-charges', minimum=1)
    sources = binding['source_image_idxs']
    order = [binding['reference_image_idx'], len(sources), *sources]
    lines = [b'COLMAP_OPTIMIZED_PHOTO_V2', _digest(binding['options_sha256']).encode(),
             ' '.join(map(str, charges)).encode(), ' '.join(map(str, order)).encode(),
             str(len(files)).encode()]
    for path in files:
        _absolute_path(path)
    for path in sorted(files, key=os.fsencode):
        lines.append(_digest(files[path]).encode() + b' ' + os.fsencode(_absolute_path(path)).hex().encode())
    result = b'\n'.join(lines) + b'\n'
    parse_v2_forecast(result)
    return result


def _admit(jobs, ceiling, active, capacity):
    _uint(ceiling, 'ceiling', min(INT_MAX,999999999), 1)
    _uint(active, 'active', min(INT_MAX,999999999), 1)
    if active > min(ceiling,5) or active > len(jobs):
        _fail('wire-concurrency-exceeds-ceiling-or-jobs')
    process = jobs[0]['forecast']['process']
    if any(job['forecast']['process'] != process for job in jobs):
        _fail('wire-process-charge-mismatch')
    if capacity is not None:
        if not isinstance(capacity,dict) or set(capacity) != {'host','device'}:
            _fail('wire-capacity-required')
        for side in ('host','device'):
            cap = _uint(capacity[side], 'capacity')
            owners = sorted((job['forecast']['private_charge'][side] for job in jobs),reverse=True)
            total = process[side] + sum(owners[:active])
            if total > U64_MAX:
                _fail('wire-total-overflow')
            if total > cap:
                _fail('wire-capacity-exceeded: '+side)


def parse_v3_launch(data, *, requested_active_jobs=None, capacity_bytes=None, options_bytes=None):
    """Parse emitted V3 framing, optional sections, and structural admission.

    No options bytes are present on the wire: the optional independent bytes
    argument models the native per-config hash check, NOT a source of options.
    Capacity checks are performed only when an independent capacity is supplied.
    """
    lines = _lines(data)
    if len(lines) < 2 or lines[0] != b'COLMAP_OPTIMIZED_PHOTO_V3':
        _fail('wire-invalid-v3-header')
    ceiling, active, count = _numbers(lines[1],'v3-envelope',count=3,minimum=1)
    if requested_active_jobs is not None and (type(requested_active_jobs) is not int or active != requested_active_jobs):
        _fail('wire-active-cli-mismatch')
    if count > len(data)//4 or len(lines) < 2+2*count:
        _fail('wire-invalid-job-count')
    jobs, references = [], set()
    for i in range(count):
        path = _absolute_path(os.fsdecode(_unhex(lines[2+2*i])))
        forecast = parse_v2_forecast(_unhex(lines[3+2*i]))
        if forecast['reference_image_idx'] in references:
            _fail('wire-duplicate-reference')
        references.add(forecast['reference_image_idx'])
        jobs.append(dict(config=path,forecast=forecast))
    bindings, pos = [], 2+2*count
    while pos < len(lines):
        if lines[pos] != b'[BINDING]':
            _fail('wire-v3-trailing-bytes')
        binding,pos = _parse_section(lines,pos+1)
        bindings.append(binding)
    if bindings and len(bindings) != count:
        _fail('wire-binding-section-count')
    for binding,job in zip(bindings,jobs):
        for key in ('options_sha256','process','private_charge','reference_image_idx','source_image_idxs'):
            if binding[key] != job['forecast'][key]:
                _fail('wire-binding-forecast-mismatch: '+key)
    if options_bytes is not None:
        if not isinstance(options_bytes,list) or len(options_bytes) != count:
            _fail('wire-options-count')
        for job,options in zip(jobs,options_bytes):
            if not isinstance(options,bytes) or hashlib.sha256(options).hexdigest() != job['forecast']['options_sha256']:
                _fail('wire-options-mismatch')
    _admit(jobs,ceiling,active,capacity_bytes)
    return dict(ceiling=ceiling,rolling_active_jobs=active,jobs=jobs,bindings=bindings)


def _serialize_v3(jobs,bindings,ceiling,active,capacity):
    # THE ONLY V3 serializer; native/fixture feeders differ solely in provenance.
    if not jobs or len(jobs) != len(bindings):
        _fail('wire-binding-section-count')
    _uint(ceiling,'ceiling',min(INT_MAX,999999999),1)
    _uint(active,'active',min(INT_MAX,999999999),1)
    lines = [b'COLMAP_OPTIMIZED_PHOTO_V3', f'{ceiling} {active} {len(jobs)}'.encode()]
    for job,binding in zip(jobs,bindings):
        if hashlib.sha256(job['options_bytes']).hexdigest() != binding['options_sha256']:
            _fail('wire-options-mismatch')
        lines.extend([os.fsencode(_absolute_path(job['config'])).hex().encode(),
                      emit_v2_forecast(binding,job['files']).hex().encode()])
    # All path/forecast pairs precede ALL named sections, never interleaved.
    for binding in bindings:
        lines.extend([b'[BINDING]',emit_binding_section(binding)[:-1]])
    payload = b'\n'.join(lines)+b'\n'
    parse_v3_launch(payload, requested_active_jobs=active, capacity_bytes=capacity,
                    options_bytes=[job['options_bytes'] for job in jobs])
    return payload


# Frozen bytes-only wrapper: no mutable dict aliases and no public rewrapping
# constructor. Integrity binds the provenance tag, ready flag, audit and payload.
from dataclasses import dataclass, field
import json


@dataclass(frozen=True,init=False)
class PhotoWireArtifact:
    provenance: str
    fire_ready: bool
    payload_bytes: bytes = field(repr=False)
    payload_sha256: str
    audit_json: bytes = field(repr=False)
    _seal: str = field(repr=False)

    def __init__(self,*args,**kwargs):
        _fail('wire-artifact-construction-requires-snapshots')

    @property
    def audit(self):
        return json.loads(self.audit_json)


def _seal(provenance,ready,payload_hash,audit):
    return hashlib.sha256(provenance.encode()+b'\0'+str(ready).encode()+b'\0'+payload_hash.encode()+b'\0'+audit).hexdigest()


def _verify_artifact(artifact):
    if type(artifact) is not PhotoWireArtifact:
        _fail('wire-artifact-required')
    # Verify payload before checking its provenance (including fixture refusal).
    if not isinstance(artifact.payload_bytes,bytes) or hashlib.sha256(artifact.payload_bytes).hexdigest() != artifact.payload_sha256:
        _fail('wire-payload-hash-mismatch')
    if (artifact.provenance not in ('native','fixture') or type(artifact.fire_ready) is not bool
            or not isinstance(artifact.audit_json,bytes) or not isinstance(artifact._seal,str)):
        _fail('wire-artifact-promotion-refused')
    if _seal(artifact.provenance,artifact.fire_ready,artifact.payload_sha256,artifact.audit_json) != artifact._seal:
        _fail('wire-artifact-promotion-refused')


def export_fire(artifact):
    _verify_artifact(artifact)
    if artifact.provenance != 'native' or not artifact.fire_ready:
        _fail('wire-fixture-not-fire-ready')
    return artifact.payload_bytes


def _emit_from_snapshots(jobs, *, provenance, ceiling, rolling_active_jobs,
                         capacity_bytes, build_identity):
    if type(jobs) is PhotoWireArtifact:
        _fail('wire-fixture-native-input-refused')
    if not isinstance(jobs,list) or not jobs:
        _fail('wire-native-snapshots-required')
    if not isinstance(capacity_bytes,dict) or set(capacity_bytes) != {'host','device'}:
        _fail('wire-capacity-required')
    jobs = copy.deepcopy(jobs)
    capacity_bytes = copy.deepcopy(capacity_bytes)
    model_jobs = []
    fields = ('reference_image_idx','source_image_idxs','width','height',
              'source_dimensions','cache_size_gib','options_bytes')
    expected_origin = 'fixture' if provenance == 'fixture' else 'native-snapshot'
    for job in jobs:
        if isinstance(job,dict) and job.get('api_scope') == 'model-only-ancestry':
            _fail('wire-model-only-ancestry-refused')
        if not isinstance(job,dict) or job.get('provenance') != expected_origin:
            _fail('wire-fixture-native-input-refused' if provenance == 'native' else 'wire-fixture-input-required')
        if not set((*fields,'workspace','config','files')) <= job.keys():
            _fail('wire-native-snapshot-fields-missing')
        if not isinstance(job['options_bytes'],bytes):
            _fail('wire-options-bytes-required')
        for key in ('workspace','config'):
            path = os.fsencode(_absolute_path(job[key]))
            expected = b'opt_string '+key.encode()+b' '+str(len(path)).encode()+b' '+path.hex().encode()
            matching = [line for line in job['options_bytes'].splitlines() if line.startswith(b'opt_string '+key.encode()+b' ')]
            if matching != [expected]:
                _fail('wire-options-path-mismatch: '+key)
        if not isinstance(job['files'],dict) or not job['files']:
            _fail('wire-v2-files-required')
        for path,sha in job['files'].items():
            _absolute_path(path)
            _digest(sha)
            if provenance == 'native' and sha == '0'*64:
                _fail('wire-placeholder-file-identity-refused')
        model_jobs.append({key:job[key] for key in fields})
    # Preparation only: this interim envelope is never exposed as a delivery.
    # The final artifact binds actual serialized bytes and producer evidence.
    prepared = p.build_delivery(exact_launch_bytes=b'PHOTO V3 SERIALIZATION PREPARATION ONLY',
        jobs=model_jobs,build_identity=build_identity)
    payload = _serialize_v3(jobs,prepared['bindings'],ceiling,rolling_active_jobs,capacity_bytes)
    audit_jobs = []
    for job,binding,record in zip(jobs,prepared['bindings'],prepared['producer_record']['jobs']):
        audit_jobs.append(dict(config=job['config'],workspace=job['workspace'],binding=binding,
            options_origin='fixture-placeholder-encoding' if provenance == 'fixture' else 'caller-supplied-native-effective',
            files_origin='fixture-placeholders-NOT-native-identities' if provenance == 'fixture' else 'caller-supplied-native-identities',
            indices_origin='declared-fixture-convention' if provenance == 'fixture' else 'caller-supplied-native-effective',
            dimensions_origin='fixture/manifest-NOT-runtime-effective' if provenance == 'fixture' else 'caller-supplied-effective-dimensions',
            fixture_context=job.get('fixture_context') if provenance == 'fixture' else None,
            options_bytes_hex=job['options_bytes'].hex(),files=job['files'],**record))
    audit = dict(jobs=audit_jobs,source_commit=prepared['producer_record']['source_commit'],fingerprint=p.FINGERPRINT,
        lineage_sha=p.LINEAGE,ceiling=ceiling,rolling_active_jobs=rolling_active_jobs,
        capacity_bytes=capacity_bytes,provenance_terms=prepared['producer_record']['provenance'],
        fire_ready_meaning='Typed native-input export readiness ONLY; not GPU authorization, qualification or a physical-fit guarantee',
        boundary='Caller native-origin assertions are trusted inputs, NOT authenticated provenance; stripping this wrapper bypasses Python export checks but not native options/file checks.')
    audit_bytes = json.dumps(audit,sort_keys=True,separators=(',',':')).encode()
    sha = hashlib.sha256(payload).hexdigest()
    ready = provenance == 'native'
    artifact = object.__new__(PhotoWireArtifact)
    values = dict(provenance=provenance,fire_ready=ready,payload_bytes=payload,
                  payload_sha256=sha,audit_json=audit_bytes,
                  _seal=_seal(provenance,ready,sha,audit_bytes))
    for key,value in values.items():
        object.__setattr__(artifact,key,value)
    return artifact


def emit_native_v3(**kwargs):
    """Native feeder: caller supplies actual native snapshots and file identities.

    No artifact can be promoted or rewrapped here; native inputs must be supplied
    anew. This is a structural contract, not authentication of a malicious caller.
    """
    return _emit_from_snapshots(provenance='native',**kwargs)
