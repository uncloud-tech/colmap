# SPDX-License-Identifier: BSD-3-Clause
"""Experimental local reconstruction with the installed native optimized backend."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import re
import struct
import sys

from . import model, producer, resident, wire
from .process import NativeFailure, NativeProcess
from .resources import observe_resources


def positive_int(text):
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError('must be positive')
    return value


def nonnegative_int(text):
    value = int(text)
    if value < 0:
        raise argparse.ArgumentTypeError('must be nonnegative')
    return value


def positive_float(text):
    value = float(text)
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError('must be positive and finite')
    return value


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--colmap', default='colmap', help='trusted native executable; defaults to installed sibling')
    parser.add_argument('--experimental-local', action='store_true', help='consent to engineering estimates, not hard RSS/VRAM caps or performance qualification')
    parser.add_argument('--backend', choices=('optimized',), default='optimized', help='strict native optimized PHOTO/GEO/fusion; never stock fallback')
    parser.add_argument('--photos', required=True, type=Path)
    parser.add_argument('--workspace', required=True, type=Path, help='NEW exclusively owned output directory; existing paths refuse')
    parser.add_argument('--host-budget-bytes', required=True, type=positive_int)
    parser.add_argument('--device-budget-bytes', required=True, type=positive_int)
    parser.add_argument('--host-headroom-bytes', type=nonnegative_int, default=512 * 1024**2)
    parser.add_argument('--device-headroom-bytes', type=nonnegative_int, default=256 * 1024**2)
    parser.add_argument('--require-hard-memory-cap', action='store_true', help='unsupported in ordinary local mode: always refuses')
    parser.add_argument('--timeout-seconds', type=positive_float, default=3600)
    parser.add_argument('--threads', type=positive_int, default=1, help='SfM/feature/undistortion workers; optimized stereo and fusion remain N1')
    parser.add_argument('--gpu-index', type=nonnegative_int, default=0)
    parser.add_argument('--source-cap', type=int, choices=(8, 20), default=20)
    parser.add_argument('--num-samples', type=int, choices=range(1, 16), default=15)
    parser.add_argument('--random-seed', type=int, default=0)
    parser.add_argument('--cache-size-gib', type=positive_float, default=1)
    parser.add_argument('--max-image-size', type=int, default=-1, help='explicit undistortion resize only; -1 preserves size')
    parser.add_argument('--sparse-model', help='mapper model directory name; required if mapper produces multiple models')
    parser.add_argument('--tile', choices=(16, 32), type=int, default=16)
    parser.add_argument('--no-lookahead', action='store_true')
    parser.add_argument('--unclassified-allowance-bytes', type=nonnegative_int, default=0, help='explicit additional GEO bank raw allowance; unmeasured default 0')
    parser.add_argument('--metadata-allowance-bytes', type=nonnegative_int, default=0, help='explicit additional GEO metadata raw allowance; unmeasured default 0')
    return parser.parse_args(argv)


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def write_json(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write('\n')


class Reconstruction:
    """Single local invocation; the existing native CLI owns the algorithms."""
    def __init__(self, args, executable):
        self.args = args
        self.executable = executable
        self.executable_sha = digest(executable)
        self.workspace = args.workspace
        self.dense = self.workspace / 'dense'
        self.assets = self.workspace / 'planning'
        self.process = NativeProcess(self.workspace, args.timeout_seconds,
                                     host_budget_bytes=args.host_budget_bytes, threads=args.threads)

    def native(self, command, *arguments, capture=False):
        if digest(self.executable) != self.executable_sha:
            raise NativeFailure('native executable changed during reconstruction')
        return self.process.run([self.executable, command, *map(str, arguments)], capture=capture)

    def resources(self):
        observation = observe_resources(self.process, self.args)
        write_json(self.assets / ('resources-%03d.json' % self.process.sequence), observation)

    def stereo_argv(self, phase):
        argv = ['--workspace_path', str(self.dense), '--workspace_format', 'COLMAP']
        for key, value in self.graph['phases'][phase]['requested_options'].items():
            argv.extend(['--PatchMatchStereo.' + key,
                         str(value).lower() if type(value) is bool else str(value)])
        return argv

    def strict_flags(self, phase, forecast, ordinal):
        return ['--optimized_only', 'true', '--optimized_dispatch_receipt',
            str(self.assets / ('%s-%d.receipt' % (phase, ordinal))),
            '--owned_tile_threads', str(self.args.tile),
            '--owned_photo_forecast' if phase == 'photo' else '--owned_run_forecast', str(forecast),
            '--owned_run_host_budget_bytes', str(self.args.host_budget_bytes),
            '--owned_run_device_budget_bytes', str(self.args.device_budget_bytes),
            '--owned_run_max_concurrent_solvers', '1',
            '--owned_run_timeout_seconds', str(self.process.remaining())]

    def run(self):
        self.assets.mkdir()
        write_json(self.assets / 'intent.json', {key: str(value) if isinstance(value, Path) else value
                   for key, value in vars(self.args).items()})
        identity = json.loads(self.native('reconstruction_info', capture=True))
        if identity.get('schema_version') != 1 or identity.get('cuda_compiled') is not True:
            raise ValueError('native CUDA/MVS build required')
        producer.compiled_binding(identity)
        if identity.get('allocation_fingerprint') != resident.SUPPORTED_ALLOCATION_FINGERPRINT:
            raise ValueError('unsupported native allocation fingerprint; no stock fallback')
        self.identity = identity
        write_json(self.assets / 'build.json', dict(identity=identity, executable=self.executable,
                   executable_sha256=self.executable_sha))
        self.resources()
        self.prepare()
        self.plan()
        self.photo()
        self.geo()
        self.resources()
        self.native('stereo_fusion', '--workspace_path', self.dense,
            '--workspace_format', 'COLMAP', '--input_type', 'geometric',
            '--output_path', self.workspace / 'points.ply', '--optimized_only', 'true',
            '--optimized_dispatch_receipt', self.assets / 'fusion.receipt',
            '--StereoFusion.max_image_size', '-1', '--StereoFusion.use_mmap', 'true',
            '--StereoFusion.use_cache', 'false', '--StereoFusion.num_threads', '1')
        count = validate_ply(self.workspace / 'points.ply')
        self.process.remaining()
        write_json(self.workspace / 'result.json', dict(status='completed', vertices=count,
            registered_images=self.registered_images, photo_references=len(self.graph['phases']['photo']['jobs']),
            geo_references=len(self.graph['phases']['geo']['jobs']), backend='optimized',
            performance_qualified=False, hard_memory_cap=False))

    def prepare(self):
        a = self.args
        common = ['--database_path', self.workspace / 'database.db']
        self.native('feature_extractor', *common, '--image_path', a.photos,
            '--FeatureExtraction.max_image_size', '3200', '--FeatureExtraction.gpu_index', a.gpu_index,
            '--FeatureExtraction.num_threads', a.threads)
        self.resources()
        self.native('exhaustive_matcher', *common, '--FeatureMatching.gpu_index', a.gpu_index,
            '--FeatureMatching.num_threads', a.threads)
        sparse = self.workspace / 'sparse'
        sparse.mkdir()
        self.resources()
        self.native('mapper', *common, '--image_path', a.photos, '--output_path', sparse,
            '--Mapper.num_threads', a.threads, '--Mapper.ba_gpu_index', a.gpu_index)
        candidates = sorted(path.name for path in sparse.iterdir() if path.is_dir() and not path.is_symlink())
        selected = a.sparse_model
        if selected is None and len(candidates) == 1:
            selected = candidates[0]
        if selected not in candidates:
            raise ValueError('select --sparse-model from native models: ' + ', '.join(candidates))
        counts = json.loads(self.native('model_analyzer', '--path', sparse / selected,
                                       '--output_format', 'JSON', capture=True))
        self.registered_images = counts['num_registered_images']
        if type(self.registered_images) is not int or self.registered_images < 2 or counts['num_points3D'] < 1:
            raise ValueError('selected sparse reconstruction has insufficient registered images/points')
        write_json(self.assets / 'sparse-model.json', dict(name=selected, counts=counts))
        self.resources()
        self.native('image_undistorter', '--image_path', a.photos, '--input_path', sparse / selected,
            '--output_path', self.dense, '--max_image_size', a.max_image_size,
            '--num_patch_match_src_images', a.source_cap, '--num_threads', a.threads)

    def plan(self):
        a = self.args
        plan = self.assets / 'dense-plan.json'
        self.native('optimized_dense_plan', '--workspace_path', self.dense,
            '--output_path', plan, '--source_cap', a.source_cap, '--num_samples', a.num_samples,
            '--gpu_index', a.gpu_index, '--cache_size_gb', a.cache_size_gib,
            '--random_seed', a.random_seed, '--owned_tex_lookahead', str(not a.no_lookahead).lower())
        self.graph = json.loads(plan.read_bytes())
        if self.graph.get('schema_version') != 1 or self.graph.get('workspace_path') != str(self.dense):
            raise ValueError('invalid native dense plan')
        if not self.identity['source_commit'].startswith(self.graph['build']['revision']):
            raise ValueError('native plan/build revision mismatch')
        for phase in ('photo', 'geo'):
            jobs = self.graph['phases'][phase]['jobs']
            if len(jobs) != self.registered_images or len({j['reference'] for j in jobs}) != len(jobs):
                raise ValueError('dense graph does not cover every registered selected-model reference')
        self.images = {image['index']: image for image in self.graph['images']}
        for image in self.images.values():
            path = Path(image['path']).resolve(strict=True)
            if not path.is_relative_to(self.dense / 'images'):
                raise ValueError('native image outside owned dense workspace')
        self.input_files = {entry['path']: entry['sha256'] for entry in self.graph['input_files']}
        self.check_inputs(self.input_files)

    @staticmethod
    def check_inputs(files):
        if any(digest(path) != expected for path, expected in files.items()):
            raise ValueError('native input changed after planning')

    def photo(self):
        # Retain original source tokens; do not apply native hash-set transforms
        # again by serializing the resolved source order back into a config.
        raw = [line.strip() for line in (self.dense / 'stereo/patch-match.cfg').read_text().splitlines()
               if line.strip() and not line.strip().startswith('#')]
        if len(raw) % 2 or len(set(raw[::2])) != len(raw[::2]):
            raise ValueError('invalid native config pairs')
        pairs = dict(zip(raw[::2], raw[1::2]))
        for ordinal, job in enumerate(self.graph['phases']['photo']['jobs']):
            self.check_inputs(self.input_files)
            self.resources()
            config = self.assets / ('photo-%d.cfg' % ordinal)
            name = self.images[job['reference']]['name']
            with config.open('x') as stream:
                stream.write(name + '\n' + pairs[name] + '\n')
            argv = self.stereo_argv('photo') + ['--config_path', str(config)]
            options = self.assets / ('photo-%d.options' % ordinal)
            self.native('patch_match_stereo', *argv, '--owned_photo_options_export', options)
            image = self.images[job['reference']]
            files = dict(self.input_files, **{str(config): digest(config)})
            snapshot = dict(provenance='native-snapshot', reference_image_idx=job['reference'],
                source_image_idxs=job['sources'], width=image['width'], height=image['height'],
                source_dimensions=[[self.images[i]['width'], self.images[i]['height']] for i in job['sources']],
                cache_size_gib=self.args.cache_size_gib, options_bytes=options.read_bytes(),
                workspace=str(self.dense), config=str(config), files=files)
            artifact = wire.emit_native_v3(jobs=[snapshot], ceiling=1, rolling_active_jobs=1,
                capacity_bytes=dict(host=self.args.host_budget_bytes, device=self.args.device_budget_bytes),
                build_identity=self.identity)
            forecast = self.assets / ('photo-%d.forecast' % ordinal)
            with forecast.open('xb') as stream:
                stream.write(wire.export_fire(artifact))
            write_json(self.assets / ('photo-%d.audit.json' % ordinal), json.loads(artifact.audit_json))
            self.native('patch_match_stereo', *argv, *self.strict_flags('photo', forecast, ordinal))
        self.photo_files = {}
        for dependency in self.graph['future_photo_dependencies']:
            path = dependency['path']
            validate_photo_map(path, dependency)
            self.photo_files[path] = digest(path)

    def geo(self):
        self.check_inputs(self.input_files)
        self.check_inputs(self.photo_files)
        self.resources()
        contract_path = self.assets / 'geo.contract'
        argv = self.stereo_argv('geo')
        self.native('patch_match_stereo', *argv, '--owned_metadata_export', contract_path)
        contract = resident.parse_owned_bank_contract(contract_path.read_bytes())
        actual = [(j['reference_image_idx'], [i['image_idx'] for i in j['inputs'] if not i['is_reference']])
                  for j in contract['problems']]
        expected = [(j['reference'], j['sources']) for j in self.graph['phases']['geo']['jobs']]
        if actual != expected:
            raise ValueError('post-PHOTO native GEO effective source order changed')
        forecast = resident.forecast_contract(contract, mode='bounded', max_future=2,
            unclassified_allowance_bytes=self.args.unclassified_allowance_bytes,
            image_channels=3, image_decode_dims={i['name']: (i['width'], i['height']) for i in self.images.values()},
            active_jobs=1, images_count=len(self.images), metadata_unclassified_bytes=self.args.metadata_allowance_bytes)
        payload = resident.serialize_resident_forecast(forecast,
            unclassified_provenance='explicit experimental local CLI allowances; unmeasured',
            execution_qualified=True, compiled_allocation_fingerprint=self.identity['allocation_fingerprint'])
        path = self.assets / 'geo.forecast'
        with path.open('xb') as stream:
            stream.write(payload)
        self.native('patch_match_stereo', *argv, *self.strict_flags('geo', path, 0))
        self.check_inputs(self.photo_files)


def validate_photo_map(path, expected):
    if model.read_resident_map_header(path)[:3] != tuple(expected[k] for k in ('width', 'height', 'channels')):
        raise ValueError('PHOTO map dimensions differ from native plan')
    with Path(path).open('rb') as stream:
        header = stream.read(96)
        match = re.match(rb'\s*\d+\s*&\s*\d+\s*&\s*\d+\s*&', header)
        stream.seek(match.end())
        remaining = expected['width'] * expected['height'] * expected['channels'] * 4
        while remaining:
            chunk = stream.read(min(1024 * 1024, remaining))
            if not chunk or len(chunk) % 4 or any(not math.isfinite(v[0]) for v in struct.iter_unpack('<f', chunk)):
                raise ValueError('truncated/nonfinite PHOTO map')
            remaining -= len(chunk)
        if stream.read(1):
            raise ValueError('unexpected PHOTO map payload')


def validate_ply(path):
    """Read the native scalar-vertex PLY formats, including finite XYZ values."""
    with Path(path).open('rb') as stream:
        header = bytearray()
        while len(header) < 65536:
            line = stream.readline(65536 - len(header))
            if not line:
                break
            header.extend(line)
            if line.strip() == b'end_header':
                break
        lines = bytes(header).splitlines()
        if not lines or lines[0] != b'ply' or lines[-1] != b'end_header':
            raise ValueError('unreadable fused PLY header')
        vertices = [int(line.split()[2]) for line in lines if line.startswith(b'element vertex ')]
        if len(vertices) != 1 or vertices[0] <= 0:
            raise ValueError('empty fused PLY')
        properties = [line.split()[1:] for line in lines if line.startswith(b'property ')]
        if [prop[1] for prop in properties[:3]] != [b'x', b'y', b'z']:
            raise ValueError('unsupported fused PLY coordinates')
        types = {b'float': 'f', b'float32': 'f', b'double': 'd', b'float64': 'd',
                 b'uchar': 'B', b'uint8': 'B'}
        formats = [line.split()[1] for line in lines if line.startswith(b'format ')]
        if formats == [b'ascii']:
            for _ in range(vertices[0]):
                values = stream.readline().split()
                if len(values) != len(properties) or not all(math.isfinite(float(v)) for v in values):
                    raise ValueError('truncated/nonfinite fused PLY')
        elif formats in ([b'binary_little_endian'], [b'binary_big_endian']):
            layout = struct.Struct(('<' if formats[0] == b'binary_little_endian' else '>') +
                                   ''.join(types[prop[0]] for prop in properties))
            for _ in range(vertices[0]):
                payload = stream.read(layout.size)
                if len(payload) != layout.size:
                    raise ValueError('truncated fused PLY')
                if not all(math.isfinite(v) for v in layout.unpack(payload)):
                    raise ValueError('nonfinite fused PLY')
        else:
            raise ValueError('unsupported fused PLY format')
        return vertices[0]


def main(argv=None):
    args = parse_args(argv)
    created_workspace = False
    try:
        if not args.experimental_local:
            raise ValueError('explicit --experimental-local consent required')
        if args.require_hard_memory_cap:
            raise ValueError('hard RSS/VRAM caps unsupported; use an external cgroup/service, not this mode')
        if sys.platform != 'linux' or os.geteuid() == 0:
            raise ValueError('experimental local execution requires non-root Linux; privileged lane is unchanged')
        if args.random_seed < -1 or args.max_image_size == 0 or args.max_image_size < -1:
            raise ValueError('invalid seed or max-image-size')
        args.photos = args.photos.resolve(strict=True)
        if not args.photos.is_dir():
            raise ValueError('photos must be a directory')
        # mkdir is the atomic no-clobber check, including dangling symlinks.
        workspace = args.workspace.absolute()
        if workspace.exists() or workspace.is_symlink():
            raise ValueError('workspace already exists; choose a new path')
        parent = workspace.parent.resolve(strict=True)
        if parent.stat().st_uid != os.geteuid():
            raise ValueError('workspace parent must be owned by the current user')
        executable = shutil.which(args.colmap)
        if not executable:
            raise ValueError('native COLMAP executable not found')
        args.workspace = parent / workspace.name
        args.workspace.mkdir(mode=0o700)
        created_workspace = True
        print('EXPERIMENTAL: logical dense admission only, NOT hard RSS/VRAM caps. '
              'SfM/fusion physical terms unknown; monitored host RSS is best-effort.', file=sys.stderr)
        reconstruction = Reconstruction(args, str(Path(executable).resolve()))
        previous = {}
        def cancelled(signum, frame):
            # Do not raise between Popen and child assignment: remember the
            # request so the owned process loop can always drain its session.
            reconstruction.process.cancelled = True
        try:
            for sig in (signal.SIGINT, signal.SIGTERM):
                previous[sig] = signal.signal(sig, cancelled)
            reconstruction.run()
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)
        print(str(args.workspace / 'points.ply'))
        return 0
    except (OSError, ValueError, KeyError, OverflowError, NativeFailure) as error:
        if created_workspace:
            try:
                write_json(args.workspace / 'failure.json', dict(status='failed', reason=str(error)))
            except OSError:
                pass  # Preserve the original failure, even if storage is full.
        print('Reconstruction refused/failed: ' + str(error), file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
