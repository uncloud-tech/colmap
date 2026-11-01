#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""Synthetic native public protocol fixture; NOT native/GPU qualification.

SfM/model selection protocol selectively ported from PR16 701da8fd1.
Dense protocol exercises the current fixed-cohort entry, never old forecasts.
"""
import hashlib
import json
import os
from pathlib import Path
import struct
import sys

command = sys.argv[1]
args = dict(zip(sys.argv[2::2], sys.argv[3::2]))
if os.environ.get('FAKE_CALLS'):
    with open(os.environ['FAKE_CALLS'], 'a') as stream:
        stream.write(json.dumps([command, args]) + '\n')
models = json.loads(os.environ.get('FAKE_SPARSE_MODELS', '{"0":{"num_registered_images":2,"num_points3D":4}}'))
identity = dict(device_uuid='GPU-synthetic', schema_version=1, cuda_compiled=True, source_commit='a' * 40,
                allocation_fingerprint=os.environ.get('FAKE_ALLOCATION_FP', 'b64f7ca666241e74825827e560b0ca04017799658860708b237f1137cb818b53'),
                available_host_bytes=str(16 * 1024**3),
                available_device_bytes='0' if os.environ.get('FAKE_BAD_FREE') else str(8 * 1024**3))


def write(path, value):
    Path(path).write_text(json.dumps(value))


def graph(workspace):
    images = [dict(index=i, name='image%d.pgm' % i, path=str(workspace / 'images' / ('image%d.pgm' % i)),
                   width=3, height=2) for i in range(2)]
    phases = {}
    for phase in ('photo', 'geo'):
        geo = phase == 'geo'
        requested = dict(depth_min=-1, depth_max=-1, sigma_spatial=-1, gpu_index='0',
                         num_samples=15, num_threads=1, max_image_size=-1,
                         geom_consistency=geo, filter=geo, random_seed=37)
        jobs = [dict(reference=i, sources=[1-i, 1-i], options=dict(requested,
                     depth_min=i+1, depth_max=i+8, sigma_spatial=5)) for i in range(2)]
        phases[phase] = dict(requested_options=requested, jobs=jobs)
    if os.environ.get('FAKE_DENSE_RESIZE'):
        phases['geo']['jobs'][0]['options']['max_image_size'] = 1600
    return dict(schema_version=1, workspace_path=str(workspace), images=images,
                build=dict(revision='a' * 8), phases=phases,
                input_files=[dict(path=i['path'], sha256=hashlib.sha256(Path(i['path']).read_bytes()).hexdigest()) for i in images])


if command == 'reconstruction_info':
    identity['device_inventory'] = [dict(ordinal=0, usable=True, device_uuid=identity['device_uuid'],
        name='Synthetic CUDA', compute_capability_major=8, compute_capability_minor=6,
        total_device_bytes=str(8 * 1024**3), available_device_bytes=identity['available_device_bytes'], reason='')]
    print(os.environ.get('FAKE_IDENTITY_JSON', json.dumps(identity)))
elif command == 'photo_directory_summary':
    print(os.environ.get('FAKE_PHOTO_SUMMARY', json.dumps(dict(
        schema_version=1, image_path=args['--image_path'], photo_count=2,
        photos=[dict(path='image%d.pgm' % i, width=4004, height=3001, orientation=None) for i in range(2)],
        problems=[], non_image_files=[], header_only=True, orientation_applied=False,
        build=dict(revision='a' * 8, cuda_compiled=True)))))
elif command in ('feature_extractor', 'exhaustive_matcher'):
    if command == 'feature_extractor' and os.environ.get('FAKE_EXTRACTION_NOFIT'):
        print('CUDA SIFT size/allocation unsupported; workers drained', file=sys.stderr)
        raise SystemExit(3)
elif command == 'mapper':
    for name in models:
        (Path(args['--output_path']) / name).mkdir()
elif command == 'model_analyzer':
    counts = models[Path(args['--path']).name]
    if counts == 'native-failure':
        raise SystemExit(7)
    if counts == 'malformed-json':
        print('not JSON')
    else:
        print(json.dumps(counts))
elif command == 'image_undistorter':
    for kind in ('INPUT', 'WORKER', 'UNSUPPORTED'):
        if os.environ.get('FAKE_BAD_UNDISTORT_' + kind):
            print('undistortion refused/failed; workers drained: ' + kind, file=sys.stderr)
            raise SystemExit(3)
    workspace = Path(args['--output_path'])
    (workspace / 'images').mkdir(parents=True)
    (workspace / 'stereo').mkdir()
    for i in range(2):
        (workspace / 'images' / ('image%d.pgm' % i)).write_bytes(b'P5\n3 2\n255\nabcdef')
    (workspace / 'stereo/fusion.cfg').write_text('image0.pgm\nimage1.pgm\n')
elif command == 'optimized_dense_plan':
    write(args['--output_path'], graph(Path(args['--workspace_path'])))
elif command == 'experimental_fixed_cohort':
    if os.environ.get('FAKE_NOFIT'):
        print('no feasible singleton within observed capacity', file=sys.stderr)
        raise SystemExit(2)
    assert args['--resolve_workspace_plan'] == '1'
    assert args['--auto_active_owners'] == args['--auto_phase_active_owners'] == '1'
    assert args['--observe_capacities'] == '1'
    assert args['--photo_source_storage'] == 'auto'
    assert args['--PatchMatchStereo.depth_min'] == args['--PatchMatchStereo.depth_max'] == '-1'
    workspace = Path(args['--workspace_path'])
    output = Path(args['--output_path'])
    output.mkdir()
    for kind, channels in (('depth_maps', 1), ('normal_maps', 3)):
        (output / 'stereo' / kind).mkdir(parents=True)
        for i in range(2):
            (output / 'stereo' / kind / ('image%d.pgm.geometric.bin' % i)).write_bytes(
                ('3&2&%d&' % channels).encode() + struct.pack('<' + 'f' * (6 * channels), *([1.] * (6 * channels))))
    phases = {phase: content['jobs'] for phase, content in graph(workspace)['phases'].items()}
    if os.environ.get('FAKE_BAD_DEPTH'):
        phases['geo'][1]['options']['depth_min'] = 1
    if os.environ.get('FAKE_BAD_ORDER'):
        phases['geo'][1]['sources'] = [0]  # Silent deduplication is NOT parity.
    write(args['--report_path'], dict(status='completed', effective_backend='experimental_fixed_cohort',
          allocation_fingerprint='c' * 64 if os.environ.get('FAKE_BAD_FP') else identity['allocation_fingerprint'],
          effective_storage='private', photo_active_owners=2, geo_active_owners=2, phases=phases))
elif command == 'stereo_fusion':
    if os.environ.get('FAKE_BAD_FUSION_INPUT'):
        print('fusion logical host budget exhausted before decode', file=sys.stderr)
        raise SystemExit(3)
    if os.environ.get('FAKE_BAD_FUSION_OUTPUT'):
        print('fusion runtime output budget exhausted; workers drained', file=sys.stderr)
        raise SystemExit(3)
    if os.environ.get('FAKE_BAD_FUSION_UNSUPPORTED'):
        print('unsupported budgeted fusion options', file=sys.stderr)
        raise SystemExit(1)
    assert args['--optimized_only'] == 'true'
    assert args['--StereoFusion.use_mmap'] == 'true'
    workspace = Path(args['--workspace_path'])
    assert len(list((workspace / 'stereo/depth_maps').glob('*.geometric.bin'))) == 2
    Path(args['--output_path']).write_bytes(b'ply\nformat ascii 1.0\nelement vertex 1\nproperty float x\nproperty float y\nproperty float z\nend_header\n1 2 3\n')
    write(args['--optimized_dispatch_receipt'], dict(schema_version=1, phase='fusion', backend='wrong' if os.environ.get('FAKE_BAD_FUSION') else 'cpu',
          cache=False, lifetime='mapped_inputs', fusion_mode='mmap'))
else:
    raise SystemExit('unexpected native dispatch: ' + command)
