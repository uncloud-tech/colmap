#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
# mypy: disallow-untyped-defs=False, disallow-incomplete-defs=False
"""Synthetic native protocol fixture. NEVER native/GPU qualification evidence.

Options literal originates in canonical 0b84800e's native serializer fixture;
all paths, images, maps, geometry and hardware responses here are synthetic.
"""
import hashlib
import json
import os
from pathlib import Path
import struct
import sys

OPTIONS = 'COLMAP_OPTIMIZED_PHOTO_OPTIONS_V1\nbuild_cuda_enabled 1\nbuild_hip_enabled 0\nopt_int max_image_size -1\nopt_string gpu_index 1 30\nopt_double depth_min 3ff0000000000000\nopt_double depth_max 4024000000000000\nopt_int random_seed 0\nopt_int window_radius 5\nopt_int window_step 1\nopt_double sigma_spatial 4014000000000000\nopt_double sigma_color 3fc999999999999a\nopt_int num_samples 15\nopt_bool owned_distinct_source_cache 1\nopt_bool owned_tex_lookahead 0\nopt_double ncc_sigma 3fe3333333333333\nopt_double min_triangulation_angle 3ff0000000000000\nopt_double incident_angle_sigma 3feccccccccccccd\nopt_int num_iterations 5\nopt_bool geom_consistency 0\nopt_double geom_consistency_regularizer 3fd3333333333333\nopt_double geom_consistency_max_cost 4008000000000000\nopt_bool filter 0\nopt_double filter_min_ncc 3fb999999999999a\nopt_double filter_min_triangulation_angle 4008000000000000\nopt_int filter_min_num_consistent 2\nopt_double filter_geom_consistency_max_cost 3ff0000000000000\nopt_bool write_consistency_graph 0\nopt_bool allow_missing_files 0\nopt_int num_threads 8\nopt_double cache_size 3f847ae147ae147b\nopt_string workspace 18 2f666978747572652f776f726b7370616365\nopt_string config 17 2f666978747572652f636f6e6669672d37\n'
JOBS = int(os.environ.get('FAKE_REGISTERED_IMAGES', '2'))

IDENTITY = dict(schema_version=1, cuda_compiled=True, source_commit='a' * 40,
    lineage_sha='cf7891059c402c333b84b752de3acb8b084abdad',
    fingerprint='9e749b59aabc5433f7e475b4e660a1a62ad8d4d5fda083fec8fc803165a78dda',
    allocation_fingerprint='0614ab2b8cbe1e7cbeec2cc90b51b10275d7eea39b9c3f20a6651d7bb2879ecb')


def string(key, value):
    raw = os.fsencode(value)
    return '%s %d %s' % (key, len(raw), raw.hex())


def options(phase):
    result = []
    for line in OPTIONS.splitlines()[3:-2]:
        if line.startswith('opt_bool geom_consistency ') or line.startswith('opt_bool filter '):
            line = line.rsplit(' ', 1)[0] + (' 1' if phase == 'geo' else ' 0')
        if line.startswith('opt_int num_threads '):
            line = 'opt_int num_threads 1'
        result.append(line)
    return result


def decoded_options(phase):
    result = {}
    for line in options(phase):
        kind, name, value, *rest = line.split()
        result[name] = (bytes.fromhex(rest[0]).decode() if kind == 'opt_string'
            else struct.unpack('>d', bytes.fromhex(value))[0] if kind == 'opt_double'
            else bool(int(value)) if kind == 'opt_bool' else int(value))
    result['depth_min'] = -1
    result['depth_max'] = -1
    return result


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def map_path(ws, name, kind):
    return ws / ('stereo/' + kind + '_maps/' + name + '.photometric.bin')


def contract(ws):
    lines = ['owned-bank-contract', 'version 1', 'build_cuda_enabled 1', 'build_hip_enabled 0',
        string('allocation_compatibility', 'unqualified-cpu-only'), 'invocation_options',
        *options('geo'), string('stereo_folder', 'stereo'), 'problem_count %d' % JOBS]
    for ref in range(JOBS):
        name = 'image%d.jpg' % ref
        lines += ['problem %d' % ref, 'reference_image_idx %d' % ref,
            string('reference_name', name), string('output_depth_path', ws / ('depth%d' % ref)),
            string('output_normal_path', ws / ('normal%d' % ref)), 'problem_options',
            *options('geo'), 'input_count 2']
        for ordinal, idx in enumerate([ref, (ref + 1) % JOBS]):
            image_name = 'image%d.jpg' % idx
            lines += ['input %d' % ordinal, 'is_reference %d' % (idx == ref), 'image_idx %d' % idx,
                string('image_name', image_name), 'image_dims 3 2', 'max_image_size -1',
                'calib_k ' + ' '.join(['3f800000'] * 9),
                'calib_r ' + ' '.join(['3f800000'] * 9), 'calib_t 00000000 00000000 00000000']
            for key, path in [('image', ws / 'images' / image_name),
                              ('depth', map_path(ws, image_name, 'depth')),
                              ('normal', map_path(ws, image_name, 'normal'))]:
                stat = path.stat()
                lines += [string(key + '_path', path), string(key + '_identity', path) +
                    ' %d %d %d %d %d %d %d' % (stat.st_size, stat.st_dev, stat.st_ino,
                    stat.st_mtime_ns // 10**9, stat.st_mtime_ns % 10**9,
                    stat.st_ctime_ns // 10**9, stat.st_ctime_ns % 10**9)]
                if key != 'image':
                    lines.append(key + '_header 3 2 ' + ('1' if key == 'depth' else '3'))
    return ('\n'.join(lines) + '\n').encode()


def main():
    if Path(sys.argv[0]).name == 'nvidia-smi':
        print(32768)
        return
    command = sys.argv[1]
    args = dict(zip(sys.argv[2::2], sys.argv[3::2]))
    if os.environ.get('FAKE_CALLS'):
        with Path(os.environ['FAKE_CALLS']).open('a') as stream:
            stream.write(json.dumps([command, args]) + '\n')
    if command == 'reconstruction_info':
        if os.environ.get('FAKE_BAD_FINGERPRINT'):
            IDENTITY['fingerprint'] = '0' * 64
        print(json.dumps(IDENTITY))
    elif command == 'mapper':
        models = json.loads(os.environ.get('FAKE_SPARSE_MODELS', '{"0": null}'))
        for name in models:
            (Path(args['--output_path']) / name).mkdir()
    elif command == 'model_analyzer':
        if 'FAKE_SPARSE_MODELS' in os.environ:
            counts = json.loads(os.environ['FAKE_SPARSE_MODELS'])[Path(args['--path']).name]
            if counts == 'native-failure':
                raise SystemExit(9)
            if counts == 'malformed-json':
                print('{')
                return
            print(json.dumps(counts))
        else:
            print(json.dumps(dict(num_registered_images=JOBS, num_points3D=4)))
    elif command == 'image_undistorter':
        ws = Path(args['--output_path'])
        for folder in ['images', 'stereo/depth_maps', 'stereo/normal_maps']:
            (ws / folder).mkdir(parents=True)
        for i in range(JOBS):
            (ws / 'images' / ('image%d.jpg' % i)).write_bytes(b'synthetic image')
        (ws / 'stereo/patch-match.cfg').write_text(''.join(
            'image%d.jpg\n__auto__, 20\n' % i for i in range(JOBS)))
    elif command == 'optimized_dense_plan':
        ws = Path(args['--workspace_path'])
        images = [dict(index=i, name='image%d.jpg' % i,
            path=str(ws / 'images' / ('image%d.jpg' % i)), width=3, height=2) for i in range(JOBS)]
        phases = {}
        for phase in ['photo', 'geo']:
            phases[phase] = dict(requested_options=decoded_options(phase), jobs=[dict(reference=i,
                sources=[(i + 1) % JOBS]) for i in range(JOBS)])
        dependencies = [dict(reference=i, kind=kind, width=3, height=2,
            channels=1 if kind == 'depth' else 3, path=str(map_path(ws, 'image%d.jpg' % i, kind)))
            for i in range(JOBS) for kind in ['depth', 'normal']]
        paths = [ws / 'stereo/patch-match.cfg', *[Path(image['path']) for image in images]]
        Path(args['--output_path']).write_text(json.dumps(dict(schema_version=1,
            workspace_path=str(ws), build=dict(revision='a'*40), images=images, phases=phases,
            future_photo_dependencies=dependencies,
            input_files=[dict(path=str(path), sha256=sha(path)) for path in paths])))
    elif command == 'patch_match_stereo':
        ws = Path(args['--workspace_path'])
        if '--owned_photo_options_export' in args:
            lines = OPTIONS.splitlines()[:3] + options('photo') + [string('opt_string workspace', ws),
                string('opt_string config', args['--config_path'])]
            Path(args['--owned_photo_options_export']).write_text('\n'.join(lines) + '\n')
        elif '--owned_metadata_export' in args:
            Path(args['--owned_metadata_export']).write_bytes(contract(ws))
        elif '--owned_photo_forecast' in args:
            assert args['--optimized_only'] == 'true'
            timeout = args['--owned_run_timeout_seconds']
            assert timeout.isascii() and timeout.isdigit() and timeout[0] != '0'
            payload = Path(args['--owned_photo_forecast']).read_bytes().splitlines()
            assert payload[0] == b'COLMAP_OPTIMIZED_PHOTO_V3'
            ceiling, active, count = map(int, payload[1].split())
            assert active == int(args['--owned_run_max_concurrent_solvers'])
            for ordinal in range(count):
                config = Path(os.fsdecode(bytes.fromhex(payload[2 + 2 * ordinal].decode())))
                name = config.read_text().splitlines()[0]
                for kind, channels in [('depth', 1), ('normal', 3)]:
                    map_path(ws, name, kind).write_bytes(('3&2&%d&' % channels).encode() + struct.pack('<f', 1) * (6*channels))
            if os.environ.get('FAKE_PHOTO_FAILURE'):
                raise SystemExit(9)
        elif '--owned_run_forecast' in args:
            assert args['--optimized_only'] == 'true'
            timeout = args['--owned_run_timeout_seconds']
            assert timeout.isascii() and timeout.isdigit() and timeout[0] != '0'
            assert Path(args['--owned_run_forecast']).read_bytes().startswith(b'resident-forecast\n')
        else:
            raise AssertionError('stock fallback')
    elif command == 'stereo_fusion':
        assert args['--optimized_only'] == 'true'
        Path(args['--output_path']).write_bytes(b'ply\nformat ascii 1.0\nelement vertex 1\nproperty float x\nproperty float y\nproperty float z\nend_header\n0 0 0\n')
    elif command not in ['feature_extractor', 'exhaustive_matcher']:
        raise AssertionError(command)


if __name__ == '__main__':
    main()
