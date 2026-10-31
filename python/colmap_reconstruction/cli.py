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

from .process import NativeFailure, NativeProcess


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
    parser.add_argument('--backend', choices=('experimental_fixed_cohort',), default='experimental_fixed_cohort', help='current adaptive native PHOTO/GEO then mapped fusion; never stock fallback')
    parser.add_argument('--photos', required=True, type=Path)
    parser.add_argument('--workspace', required=True, type=Path, help='NEW exclusively owned output directory; existing paths refuse')
    parser.add_argument('--host-budget-bytes', type=positive_int, help='optional upper bound on native observed FREE host capacity')
    parser.add_argument('--device-budget-bytes', type=positive_int, help='optional upper bound on native observed FREE device capacity')
    for name in ('loader-scratch-host', 'process-host', 'process-device',
                 'unknown-host', 'unknown-device', 'reserve-host', 'reserve-device'):
        parser.add_argument('--' + name + '-bytes', required=True, type=nonnegative_int,
                            help='explicit RAW uncalibrated allowance; native applies one aggregate 3%% margin')
    parser.add_argument('--unknown-provenance', required=True)
    parser.add_argument('--active-owners-ceiling', type=nonnegative_int, default=0,
                        help='0 means job count; admission is not a throughput optimum')
    parser.add_argument('--require-hard-memory-cap', action='store_true', help='unsupported in ordinary local mode: always refuses')
    parser.add_argument('--timeout-seconds', type=positive_float, default=3600)
    parser.add_argument('--threads', type=positive_int, default=1, help='SfM/feature/undistortion workers; independent of optimized solver concurrency')
    parser.add_argument('--gpu-index', type=nonnegative_int, default=0)
    parser.add_argument('--source-cap', type=int, choices=(8, 20), default=20)
    parser.add_argument('--num-samples', type=int, choices=range(1, 16), default=15)
    parser.add_argument('--random-seed', type=int, default=0)
    parser.add_argument('--cache-size-gib', type=positive_float, default=1)
    parser.add_argument('--max-image-size', type=int, default=-1, help='explicit undistortion resize only; -1 preserves size')
    parser.add_argument('--sparse-model', help='exact mapper model directory name or largest (most registered images, ties by lexicographically smallest name); required for multiple models')
    parser.add_argument('--tile', choices=(16, 32), type=int, default=16)
    parser.add_argument('--no-lookahead', action='store_true')
    return parser.parse_args(argv)


def digest(path, *, check=None):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while True:
            if check:
                check()
            chunk = stream.read(1024 * 1024)
            if not chunk:
                break
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
        self.workspace = args.workspace
        self.dense = self.workspace / 'dense'
        self.assets = self.workspace / 'planning'
        self.process = NativeProcess(self.workspace, args.timeout_seconds,
                                     host_budget_bytes=args.host_budget_bytes, threads=args.threads)
        self.executable_sha = digest(executable, check=self.process.remaining)

    def native(self, command, *arguments, capture=False):
        if digest(self.executable, check=self.process.remaining) != self.executable_sha:
            raise NativeFailure('native executable changed during reconstruction')
        return self.process.run([self.executable, command, *map(str, arguments)], capture=capture)

    def resources(self):
        # Reuse the canonical native host/cgroup + CUDA observer, not a
        # parallel Python resource calculator or management-index query.
        seconds = math.floor(self.process.remaining())
        if seconds < 1:
            raise NativeFailure('reconstruction deadline has less than one native second remaining')
        observation = json.loads(self.native('reconstruction_info', '--gpu_index',
            self.args.gpu_index, '--deadline_seconds', seconds, capture=True))
        if observation.get('allocation_fingerprint') != self.identity['allocation_fingerprint']:
            raise NativeFailure('native resource observation build changed')
        for side in ('host', 'device'):
            available = int(observation['available_' + side + '_bytes'])
            cap = getattr(self.args, side + '_budget_bytes')
            if cap is not None:
                available = min(available, cap)
            if available <= getattr(self.args, 'reserve_' + side + '_bytes'):
                raise NativeFailure(side + ' reserve unavailable before native stage')
        write_json(self.assets / ('resources-%03d.json' % self.process.sequence), observation)

    def dense_arguments(self):
        a = self.args
        seconds = math.floor(self.process.remaining())
        if seconds < 1:
            raise NativeFailure('reconstruction deadline has less than one native second remaining')
        argv = ['--workspace_path', str(self.dense), '--output_path', str(self.workspace / 'cohort'),
                '--resolve_workspace_plan', '1', '--report_path', str(self.assets / 'cohort.json'),
                '--auto_active_owners', '1', '--auto_phase_active_owners', '1',
                '--observe_capacities', '1', '--photo_source_storage', 'auto',
                '--active_owners_ceiling', str(a.active_owners_ceiling),
                '--output_policy', 'final_geo_only', '--deadline_seconds', str(seconds),
                '--tile_threads', str(a.tile), '--unknown_provenance', a.unknown_provenance]
        for name in ('loader_scratch_host', 'process_host', 'process_device',
                     'unknown_host', 'unknown_device', 'reserve_host', 'reserve_device'):
            argv += ['--' + name + '_bytes', str(getattr(a, name + '_bytes'))]
        for name, value in (('host', a.host_budget_bytes), ('device', a.device_budget_bytes)):
            if value is not None:
                argv += ['--' + name + '_capacity_bytes', str(value)]
        for key, value in self.graph['phases']['geo']['requested_options'].items():
            argv += ['--PatchMatchStereo.' + key,
                     str(value).lower() if type(value) is bool else str(value)]
        return argv

    def run(self):
        self.assets.mkdir()
        write_json(self.assets / 'intent.json', {key: str(value) if isinstance(value, Path) else value
                   for key, value in vars(self.args).items()})
        identity = json.loads(self.native('reconstruction_info', capture=True))
        if identity.get('schema_version') != 1 or identity.get('cuda_compiled') is not True:
            raise ValueError('native CUDA/MVS build required')
        if not re.fullmatch(r'[0-9a-f]{64}', identity.get('allocation_fingerprint', '')):
            raise ValueError('missing native allocation fingerprint')
        self.identity = identity
        write_json(self.assets / 'build.json', dict(identity=identity, executable=self.executable,
                   executable_sha256=self.executable_sha))
        self.resources()
        self.prepare()
        self.plan()
        self.dense_stage()
        self.resources()
        self.native('stereo_fusion', '--workspace_path', self.dense,
            '--workspace_format', 'COLMAP', '--input_type', 'geometric',
            '--output_path', self.workspace / 'points.ply', '--optimized_only', 'true',
            '--optimized_dispatch_receipt', self.assets / 'fusion.receipt',
            '--StereoFusion.max_image_size', '-1', '--StereoFusion.use_mmap', 'true',
            '--StereoFusion.use_cache', 'false', '--StereoFusion.num_threads', '1')
        receipt = json.loads((self.assets / 'fusion.receipt').read_bytes())
        if any(receipt.get(key) != expected for key, expected in dict(
                schema_version=1, phase='fusion', backend='cpu', cache=False,
                lifetime='mapped_inputs', fusion_mode='mmap').items()):
            raise ValueError('native mapped fusion dispatch binding missing/mismatched')
        count = validate_ply(self.workspace / 'points.ply', check=self.process.remaining)
        self.process.remaining()
        write_json(self.workspace / 'result.json', dict(status='completed', vertices=count,
            registered_images=self.registered_images, photo_references=len(self.graph['phases']['photo']['jobs']),
            geo_references=len(self.graph['phases']['geo']['jobs']), backend='experimental_fixed_cohort',
            dense_report=self.report,
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
        policy = 'largest' if selected == 'largest' else 'explicit' if selected is not None else 'single'
        if selected is None and len(candidates) == 1:
            selected = candidates[0]
        if not candidates or (policy != 'largest' and selected not in candidates):
            raise ValueError('select --sparse-model from native models: ' + ', '.join(candidates))
        analyzed = {}
        for name in candidates if policy == 'largest' else [selected]:
            counts = json.loads(self.native('model_analyzer', '--path', sparse / name,
                                           '--output_format', 'JSON', capture=True))
            if not isinstance(counts, dict) or any(type(counts.get(key)) is not int
                    for key in ('num_registered_images', 'num_points3D')):
                raise ValueError('invalid native model analyzer counts for sparse model ' + name)
            if counts['num_registered_images'] < 2 or counts['num_points3D'] < 1:
                raise ValueError('sparse model ' + name + ' has insufficient registered images/points')
            analyzed[name] = counts
        if policy == 'largest':
            selected = min(candidates, key=lambda name: (-analyzed[name]['num_registered_images'], name))
        counts = analyzed[selected]
        self.registered_images = counts['num_registered_images']
        write_json(self.assets / 'sparse-model.json', dict(name=selected, policy=policy, counts=counts))
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

    def check_inputs(self, files):
        if any(digest(path, check=self.process.remaining) != expected for path, expected in files.items()):
            raise ValueError('native input changed after planning')

    def dense_stage(self):
        self.check_inputs(self.input_files)
        self.resources()
        self.native('experimental_fixed_cohort', *self.dense_arguments())
        self.report = json.loads((self.assets / 'cohort.json').read_bytes())
        if (self.report.get('status') != 'completed'
                or self.report.get('effective_backend') != 'experimental_fixed_cohort'
                or self.report.get('allocation_fingerprint') != self.identity['allocation_fingerprint']):
            raise ValueError('native fixed-cohort execution binding missing/mismatched')
        for phase in ('photo', 'geo'):
            actual = self.report['phases'][phase]
            expected = self.graph['phases'][phase]['jobs']
            # Exact effective order (including any duplicates), per-reference
            # depth/sigma/options/PRNG; never globalize or sort to obtain parity.
            if actual != [dict(reference=j['reference'], sources=j['sources'], options=j['options'])
                          for j in expected]:
                raise ValueError('native fixed-cohort effective ' + phase + ' plan changed')
        self.check_inputs(self.input_files)
        # Publish only complete native outputs into the fresh undistorted tree.
        # Native final_geo_only retains PHOTO until the final GEO drain.
        for job in self.graph['phases']['geo']['jobs']:
            name = self.images[job['reference']]['name']
            for kind in ('depth_maps', 'normal_maps'):
                relative = Path('stereo') / kind / (name + '.geometric.bin')
                source = self.workspace / 'cohort' / relative
                image = self.images[job['reference']]
                validate_dense_map(source, image['width'], image['height'],
                                   3 if kind == 'normal_maps' else 1, check=self.process.remaining)
                target = self.dense / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                # link is an atomic no-clobber publication, not an overwrite.
                os.link(source, target)
        fusion = self.dense / 'stereo/fusion.cfg'
        names = [self.images[j['reference']]['name'] for j in self.graph['phases']['geo']['jobs']]
        eligible = [line.strip() for line in fusion.read_text().splitlines()
                    if line.strip() and not line.strip().startswith('#')]
        if eligible != names and (len(eligible) != len(names) or set(eligible) != set(names)):
            raise ValueError('fusion eligibility differs from complete GEO producers')


def validate_dense_map(path, width, height, channels, *, check=None):
    """Validate native float32 map framing and the entire finite payload."""
    with Path(path).open('rb') as stream:
        header = ('%d&%d&%d&' % (width, height, channels)).encode('ascii')
        if stream.read(len(header)) != header:
            raise ValueError('dense map dimensions differ from native plan')
        remaining = width * height * channels * 4
        while remaining:
            if check:
                check()
            chunk = stream.read(min(1024 * 1024, remaining))
            if not chunk or len(chunk) % 4 or any(not math.isfinite(v[0]) for v in struct.iter_unpack('<f', chunk)):
                raise ValueError('truncated/nonfinite dense map')
            remaining -= len(chunk)
        if stream.read(1):
            raise ValueError('unexpected dense map payload')


def validate_ply(path, *, check=None):
    """Read the native scalar-vertex PLY formats, including finite XYZ values."""
    with Path(path).open('rb') as stream:
        header = bytearray()
        while len(header) < 65536:
            if check:
                check()
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
            for index in range(vertices[0]):
                if check and index % 1024 == 0:
                    check()
                values = stream.readline(65536).split()
                if len(values) != len(properties) or not all(math.isfinite(float(v)) for v in values):
                    raise ValueError('truncated/nonfinite fused PLY')
        elif formats in ([b'binary_little_endian'], [b'binary_big_endian']):
            layout = struct.Struct(('<' if formats[0] == b'binary_little_endian' else '>') +
                                   ''.join(types[prop[0]] for prop in properties))
            for index in range(vertices[0]):
                if check and index % 1024 == 0:
                    check()
                payload = stream.read(layout.size)
                if len(payload) != layout.size:
                    raise ValueError('truncated fused PLY')
                if not all(math.isfinite(v) for v in layout.unpack(payload)):
                    raise ValueError('nonfinite fused PLY')
        else:
            raise ValueError('unsupported fused PLY format')
        if check:
            check()
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
        if not (args.unknown_host_bytes or args.unknown_device_bytes) or not args.unknown_provenance.strip():
            raise ValueError('positive explicit unknown allowance and provenance required')
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
    except (OSError, ValueError, KeyError, TypeError, AttributeError, OverflowError, NativeFailure) as error:
        if created_workspace:
            try:
                write_json(args.workspace / 'failure.json', dict(status='failed', reason=str(error)))
            except OSError:
                pass  # Preserve the original failure, even if storage is full.
        print('Reconstruction refused/failed: ' + str(error), file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
