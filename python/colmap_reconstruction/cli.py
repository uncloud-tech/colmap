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
import select
import re
import struct
import sys
import tempfile

from .process import NativeFailure, NativeProcess
from .resource_policy import select_policy, validate_overrides


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


def byte_count(text):
    value = nonnegative_int(text)
    if value > 2**64 - 1:
        raise argparse.ArgumentTypeError('must fit native uint64 bytes')
    return value


def positive_byte_count(text):
    value = byte_count(text)
    if value == 0:
        raise argparse.ArgumentTypeError('must be positive')
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
    parser.add_argument('--plan-only', action='store_true', help='inspect and display only; never create the requested workspace')
    parser.add_argument('--yes', action='store_true', help='approve the displayed plan noninteractively; separate from engineering consent')
    parser.add_argument('--backend', choices=('experimental_fixed_cohort',), default='experimental_fixed_cohort', help='current adaptive native PHOTO/GEO then mapped fusion; never stock fallback')
    parser.add_argument('--photos', required=True, type=Path)
    parser.add_argument('--workspace', required=True, type=Path, help='NEW exclusively owned output directory; existing paths refuse')
    parser.add_argument('--host-budget-bytes', type=positive_byte_count, help='optional upper bound on native observed FREE host capacity')
    parser.add_argument('--device-budget-bytes', type=positive_byte_count, help='optional upper bound on native observed FREE device capacity')
    for name in ('loader-scratch-host', 'process-host', 'process-device',
                 'unknown-host', 'unknown-device', 'reserve-host', 'reserve-device'):
        parser.add_argument('--' + name + '-bytes', type=byte_count,
                            help='expert RAW allowance override; otherwise versioned conservative policy, not calibration; native applies one aggregate 3%% margin')
    parser.add_argument('--unknown-provenance', help='required for explicit unknown byte overrides')
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
    parser.add_argument('--max-image-size', type=int, default=-1, help='expert undistorted-camera edge cap only; -1 means no resource resize, but undistortion geometry can change dimensions; never limits extraction')
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
        if (command in ('feature_extractor', 'exhaustive_matcher', 'mapper', 'image_undistorter')
                and hasattr(self, 'preview_options')):
            self.check_approval_binding()
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
        if (hasattr(self, 'approved_device_uuid')
                and observation.get('device_uuid') != self.approved_device_uuid):
            raise NativeFailure('selected device identity changed since preview; rerun to approve a fresh plan')
        effective = {}
        for side in ('host', 'device'):
            available = int(observation['available_' + side + '_bytes'])
            cap = getattr(self.args, side + '_budget_bytes')
            if cap is not None:
                available = min(available, cap)
            effective[side] = available
        observation['effective_capacity_bytes'] = effective
        observation['capacity_classification'] = 'measured-free-limited-by-user-cap'
        self.observation = dict(observation)
        write_json(self.assets / ('resources-%03d.json' % self.process.sequence), observation)
        for side in ('host', 'device'):
            if effective[side] <= getattr(self.args, 'reserve_' + side + '_bytes'):
                raise NativeFailure(side + ' reserve unavailable before native stage; '
                                    'lower workload or free capacity (budgets cannot raise FREE)')
        return effective

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

    def photo_snapshot(self):
        """Bind bytes, names and resolved targets, including discovery sidecars."""
        snapshot = {}
        for path in sorted(self.args.photos.rglob('*')):
            self.process.remaining()
            snapshot[str(path.relative_to(self.args.photos))] = dict(
                target=str(path.resolve(strict=True)),
                sha256=digest(path, check=self.process.remaining) if path.is_file() else None)
        return snapshot

    def check_approval_binding(self):
        self.process.remaining()
        if vars(self.args) != self.preview_options:
            raise ValueError('options changed since preview; rerun to review and approve a fresh plan')
        if digest(self.executable, check=self.process.remaining) != self.executable_sha:
            raise ValueError('native executable changed since preview; rerun to approve a fresh plan')
        if self.photo_snapshot() != self.approved_photos:
            raise ValueError('photo inputs/inventory changed since preview; rerun to approve a fresh plan')

    def preview(self):
        self.approved_photos = self.photo_snapshot()
        self.assets.mkdir()
        write_json(self.assets / 'intent.json', {key: str(value) if isinstance(value, Path) else value
                   for key, value in vars(self.args).items()})
        identity = json.loads(self.native('reconstruction_info', capture=True))
        if identity.get('schema_version') != 1 or identity.get('cuda_compiled') is not True:
            raise ValueError('plan unavailable: native CUDA/MVS build required; CPU fallback unsupported')
        if not re.fullmatch(r'[0-9a-f]{64}', identity.get('allocation_fingerprint', '')):
            raise ValueError('missing native allocation fingerprint')
        self.identity = identity
        self.resource_policy = select_policy(self.args, identity)
        write_json(self.assets / 'resource-policy.json', self.resource_policy)
        print('Resource policy: ' + json.dumps(self.resource_policy, sort_keys=True), file=sys.stderr)
        write_json(self.assets / 'build.json', dict(identity=identity, executable=self.executable,
                   executable_sha256=self.executable_sha))
        self.resources()
        self.approved_device_uuid = self.observation['device_uuid']
        self.extraction_limit = self.inspect_photos()
        if self.photo_snapshot() != self.approved_photos:
            raise ValueError('photo inputs changed during inspection; rerun for a fresh plan')
        self.preview_options = dict(vars(self.args))
        plan = dict(
            executable=dict(path=self.executable, sha256=self.executable_sha, build=self.identity),
            output_path=str(self.workspace), photo_count=len(self.resolution['original']['photos']),
            resolution=self.resolution, input_binding=self.approved_photos,
            options={key: str(value) if isinstance(value, Path) else value
                     for key, value in self.preview_options.items()},
            hardware_observation=self.observation,
            cpu_available=len(os.sched_getaffinity(0)),
            hardware_not_exported_by_native='CPU model, CUDA name/total/SM: UNKNOWN',
            resource_policy=self.resource_policy,
            capacity_semantics='FREE snapshot, not reserved; reobserved/admitted at native boundaries',
            effective_dispatch=dict(extraction=dict(backend='CUDA SIFT', threads=self.args.threads,
                    gpu_index=self.args.gpu_index),
                matching=dict(backend='CUDA SIFT exhaustive', threads=self.args.threads,
                    gpu_index=self.args.gpu_index),
                mapping=dict(backend='incremental CPU Ceres', threads=self.args.threads),
                undistortion=dict(threads=self.args.threads),
                stereo=dict(backend=self.args.backend, gpu_index=self.args.gpu_index,
                    source_cap=self.args.source_cap, num_samples=self.args.num_samples,
                    random_seed=self.args.random_seed, cache_size_gib=self.args.cache_size_gib,
                    tile=self.args.tile, lookahead=not self.args.no_lookahead),
                fusion=dict(backend='CPU', threads=1, use_mmap=True, use_cache=False)),
            phases=['CUDA SIFT extraction', 'exhaustive GPU matching', 'incremental CPU Ceres SfM',
                    'sparse model selection', 'geometric undistortion',
                    'adaptive PHOTO then GEO experimental_fixed_cohort', 'mapped CPU fusion'],
            dense_resolution='float32; stereo/fusion uncapped; no automatic resize or stock fallback',
            scheduling='native memory-admitted phase owners and PHOTO storage; ceiling is not measured concurrency',
            unresolved='UNKNOWN runtime, physical fit, post-SfM geometry, depth ranges and source graph; '
                       'remaining native dense options resolve after SfM under these options, not a final dense plan; '
                       'insufficient capacity or unsupported input refuses without quality change')
        write_json(self.assets / 'approved-plan.json', plan)
        print('Effective plan (approval required): ' + json.dumps(plan, sort_keys=True), flush=True)

    def run(self):
        self.prepare()
        self.plan()
        self.dense_stage()
        phase_capacity = self.resources()['host']
        # Use the same absolute phase capacity for best-effort direct-child RSS
        # monitoring. Native accounting owns the one 3% margin and reserve;
        # Python only hands off raw policy, never a second forecast/margin.
        self.process.host_budget_bytes = phase_capacity
        self.native('stereo_fusion', '--workspace_path', self.dense,
            '--workspace_format', 'COLMAP', '--input_type', 'geometric',
            '--output_path', self.workspace / 'points.ply', '--optimized_only', 'true',
            '--optimized_dispatch_receipt', self.assets / 'fusion.receipt',
            '--StereoFusion.max_image_size', '-1', '--StereoFusion.use_mmap', 'true',
            '--StereoFusion.use_cache', 'false', '--StereoFusion.num_threads', '1',
            '--StereoFusion.host_budget_bytes', phase_capacity,
            '--StereoFusion.policy_host_bytes', self.args.loader_scratch_host_bytes
                + self.args.process_host_bytes + self.args.unknown_host_bytes,
            '--StereoFusion.reserve_host_bytes', self.args.reserve_host_bytes)
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
            dense_report=self.report, resource_policy=self.resource_policy,
            performance_qualified=False, hard_memory_cap=False))

    def inspect_photos(self):
        summary = json.loads(self.native('photo_directory_summary', '--image_path',
                                         self.args.photos, capture=True))
        if (not isinstance(summary, dict) or type(summary.get('schema_version')) is not int
                or summary['schema_version'] != 1 or summary.get('header_only') is not True
                or summary.get('orientation_applied') is not False):
            raise ValueError('invalid native photo inspection schema/dimension semantics')
        build = summary.get('build')
        if (not isinstance(build, dict) or build.get('cuda_compiled') is not True
                or not isinstance(build.get('revision'), str) or not build['revision']
                or not self.identity['source_commit'].startswith(build['revision'])):
            raise ValueError('native photo inspection build mismatch')
        photos = summary.get('photos')
        if (not isinstance(photos, list) or not photos
                or type(summary.get('photo_count')) is not int
                or summary['photo_count'] != len(photos)):
            raise ValueError('empty or inconsistent native photo inspection')
        if summary.get('problems') != []:
            raise ValueError('photo inspection problems; repair unreadable/unknown-dimension inputs; '
                             'no reference skipping: ' + json.dumps(summary.get('problems')))
        if not isinstance(summary.get('non_image_files'), list):
            raise ValueError('invalid native non-image inventory')
        paths = set()
        working_dimensions = []
        for photo in photos:
            if (not isinstance(photo, dict) or not isinstance(photo.get('path'), str)
                    or not photo['path'] or photo['path'] in paths
                    or any(type(photo.get(key)) is not int or not 0 < photo[key] <= (2**31 - 1) // 2
                           for key in ('width', 'height'))):
                # The GPU first octave is doubled: never overflow its signed
                # maxd multiplication or use nonpositive SIFT's implicit 3200.
                raise ValueError('invalid/unsupported native photo dimensions or duplicate path')
            orientation = photo.get('orientation')
            if orientation is not None and (type(orientation) is not int or orientation <= 0):
                raise ValueError('invalid native photo EXIF orientation')
            # ImageReader derives gravity from EXIF 1/3/6/8. The extraction
            # worker rotates 6/8 by 90/270 degrees before CUDA SIFT, whose
            # GLTexInput::TruncateWidthCU otherwise silently drops columns.
            swapped = orientation in (6, 8)
            working_width = photo['height'] if swapped else photo['width']
            if working_width % 4:
                raise ValueError('unsupported CUDA SIFT input ' + photo['path']
                    + ': effective working width ' + str(working_width)
                    + (' after EXIF rotation' if swapped else '')
                    + ' must be a multiple of four (backend truncates columns); '
                      'no automatic resize/fallback. Use supported input dimensions '
                      'explicitly; originals are unchanged. Native odd-width support is pending.')
            working_dimensions.append(dict(path=photo['path'], width=working_width,
                height=photo['width'] if swapped else photo['height']))
            paths.add(photo['path'])
        extraction_limit = max(max(photo['width'], photo['height']) for photo in photos)
        selected = [photo['path'] for photo in photos]
        self.extraction_image_list = []
        effective_order = selected
        ordering = 'native filesystem path discovery order'
        if summary['non_image_files']:
            for name in selected:
                # ReadTextFileLines uses getline('\n') and trims exactly these
                # four ASCII characters. It does NOT interpret '#' comments.
                # Do not trim, escape, or rename a path into a different file.
                try:
                    name.encode('utf-8')
                    representable = '\n' not in name and '\0' not in name and name == name.strip(' \n\r\t')
                except UnicodeEncodeError:
                    representable = False
                if not representable:
                    raise ValueError('native image list cannot represent photo filename '
                                     + repr(name) + '; no rename/skip; originals unchanged')
            image_list = self.assets / 'extraction-images.txt'
            with image_list.open('xb') as stream:
                stream.write(('\n'.join(selected) + '\n').encode('utf-8'))
            self.extraction_image_list = ['--image_list_path', image_list]
            # ImageReader sorts nonempty image_names as std::string, unlike
            # default filesystem::path discovery. UTF-8 byte order matches it.
            effective_order = sorted(selected, key=lambda name: name.encode('utf-8'))
            ordering = 'native image-list UTF-8 byte string sort (may differ from discovery)'
        inventory = dict(selected_photos=selected, effective_order=effective_order,
                         ordering=ordering, excluded_non_image_files=summary['non_image_files'])
        write_json(self.assets / 'extraction-inventory.json', inventory)
        print('Extraction inventory: ' + json.dumps(inventory), file=sys.stderr)
        write_json(self.assets / 'photos.json', summary)
        resolution = dict(
            original=dict(semantics='encoded dimensions; originals unchanged', photos=photos),
            extraction=dict(backend='CUDA SIFT', max_image_size=extraction_limit,
                first_octave=-1, working_dimensions=working_dimensions,
                semantics='no thumbnail; EXIF gravity rotation; first octave upsamples 2x'),
            undistortion=dict(max_image_size=self.args.max_image_size,
                geometric_options=dict(blank_pixels=0.0, min_scale=0.2, max_scale=2.0),
                resolved_dimensions=None,
                semantics='pinhole geometry may change dimensions even with cap -1; resolved after SfM'),
            stereo=dict(max_image_size=-1, map_precision='float32'),
            fusion=dict(max_image_size=-1, semantics='consume geometric maps without additional resize'),
            automatic_resource_resize=False, non_image_files=summary['non_image_files'])
        self.resolution = resolution
        write_json(self.assets / 'resolution-intent.json', resolution)
        print('Resolution intent (not approval): ' + json.dumps(resolution, sort_keys=True), file=sys.stderr)
        return extraction_limit

    def prepare(self):
        a = self.args
        extraction_limit = self.extraction_limit if hasattr(self, 'extraction_limit') else self.inspect_photos()
        common = ['--database_path', self.workspace / 'database.db']
        phase_capacity = self.resources()['host']
        previous_host_budget = self.process.host_budget_bytes
        self.process.host_budget_bytes = phase_capacity
        try:
            self.native('feature_extractor', *common, '--image_path', a.photos, *self.extraction_image_list,
                '--FeatureExtraction.max_image_size', str(extraction_limit),
                '--FeatureExtraction.type', 'SIFT', '--FeatureExtraction.use_gpu', 'true',
                '--SiftExtraction.first_octave', '-1', '--FeatureExtraction.gpu_index', a.gpu_index,
                '--FeatureExtraction.num_threads', a.threads,
                '--FeatureExtraction.host_budget_bytes', phase_capacity,
                '--FeatureExtraction.policy_host_bytes', a.loader_scratch_host_bytes + a.process_host_bytes + a.unknown_host_bytes,
                '--FeatureExtraction.reserve_host_bytes', a.reserve_host_bytes)
        finally:
            self.process.host_budget_bytes = previous_host_budget
        matching_capacity = self.resources()
        previous_host_budget = self.process.host_budget_bytes
        self.process.host_budget_bytes = matching_capacity['host']
        try:
            # This frontend requires a native CUDA build and retains its default
            # GPU matching backend. Device policy is independent of host policy.
            self.native('exhaustive_matcher', *common, '--FeatureMatching.gpu_index', a.gpu_index,
                '--FeatureMatching.num_threads', a.threads,
                '--FeatureMatching.host_budget_bytes', matching_capacity['host'],
                '--FeatureMatching.policy_host_bytes', a.loader_scratch_host_bytes + a.process_host_bytes + a.unknown_host_bytes,
                '--FeatureMatching.reserve_host_bytes', a.reserve_host_bytes,
                '--FeatureMatching.device_budget_bytes', matching_capacity['device'],
                '--FeatureMatching.policy_device_bytes', a.process_device_bytes + a.unknown_device_bytes,
                '--FeatureMatching.reserve_device_bytes', a.reserve_device_bytes)
        finally:
            self.process.host_budget_bytes = previous_host_budget
        sparse = self.workspace / 'sparse'
        sparse.mkdir()
        mapping_capacity = self.resources()['host']
        previous_host_budget = self.process.host_budget_bytes
        self.process.host_budget_bytes = mapping_capacity
        try:
            # ba_gpu_index selects an ordinal only; default Ceres BA stays CPU.
            self.native('mapper', *common, '--image_path', a.photos, '--output_path', sparse,
                '--Mapper.num_threads', a.threads, '--Mapper.ba_gpu_index', a.gpu_index,
                '--Mapper.host_budget_bytes', mapping_capacity,
                '--Mapper.policy_host_bytes', a.loader_scratch_host_bytes + a.process_host_bytes + a.unknown_host_bytes,
                '--Mapper.reserve_host_bytes', a.reserve_host_bytes)
        finally:
            self.process.host_budget_bytes = previous_host_budget
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
        phase_capacity = self.resources()['host']
        previous_host_budget = self.process.host_budget_bytes
        self.process.host_budget_bytes = phase_capacity
        try:
            self.native('image_undistorter', '--image_path', a.photos, '--input_path', sparse / selected,
                '--output_path', self.dense, '--max_image_size', a.max_image_size,
                '--blank_pixels', '0', '--min_scale', '0.2', '--max_scale', '2',
                '--num_patch_match_src_images', a.source_cap, '--num_threads', a.threads,
                '--host_budget_bytes', phase_capacity,
                '--policy_host_bytes', a.loader_scratch_host_bytes + a.process_host_bytes + a.unknown_host_bytes,
                '--reserve_host_bytes', a.reserve_host_bytes)
        finally:
            # This FREE snapshot belongs only to the undistortion phase.
            self.process.host_budget_bytes = previous_host_budget

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
            if (self.graph['phases'][phase]['requested_options'].get('max_image_size') != -1
                    or any(job['options'].get('max_image_size') != -1 for job in jobs)):
                raise ValueError('native dense plan attempted an unrequested resolution cap; no reduced retry')
            if len(jobs) != self.registered_images or len({j['reference'] for j in jobs}) != len(jobs):
                raise ValueError('dense graph does not cover every registered selected-model reference')
        self.images = {image['index']: image for image in self.graph['images']}
        for image in self.images.values():
            path = Path(image['path']).resolve(strict=True)
            if not path.is_relative_to(self.dense / 'images'):
                raise ValueError('native image outside owned dense workspace')
        self.input_files = {entry['path']: entry['sha256'] for entry in self.graph['input_files']}
        self.check_inputs(self.input_files)
        effective = dict(images=[dict(name=image['name'], width=image['width'], height=image['height'])
                                for image in self.graph['images']],
            semantics='resolved undistorted geometry, shared by uncapped PHOTO/GEO and fusion inputs',
            stereo_max_image_size=-1, fusion_max_image_size=-1, map_precision='float32')
        write_json(self.assets / 'resolution-effective.json', effective)
        print('Resolved dense dimensions: ' + json.dumps(effective, sort_keys=True), file=sys.stderr)

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


def approve_plan(process):
    """Read one explicit TTY answer without escaping deadline/cancellation."""
    if not sys.stdin.isatty():
        raise ValueError('noninteractive execution requires explicit --yes plan approval')
    print('Approve this plan? Type yes to run; no/Enter rejects (Ctrl-D cancels).', flush=True)
    answer = bytearray()
    while True:
        remaining = process.remaining()
        ready, _, _ = select.select([sys.stdin], [], [], min(remaining, 0.2))
        if not ready:
            continue
        chunk = os.read(sys.stdin.fileno(), 1)
        if not chunk:
            raise NativeFailure('plan approval cancelled: EOF')
        if chunk == b'\n':
            break
        answer.extend(chunk)
        if len(answer) > 64:
            raise ValueError('plan rejected: expected yes')
    if answer.strip().lower() != b'yes':
        raise ValueError('plan rejected; no reconstruction started')
    process.remaining()


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
        validate_overrides(args)
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
        if args.workspace.is_relative_to(args.photos):
            raise ValueError('workspace must be outside photos to preserve input discovery')
        # tempfile probes its environment-selected directory before returning it.
        # Refuse input-tree candidates before even that probe can mutate inputs.
        for name in ('TMPDIR', 'TEMP', 'TMP'):
            candidate = os.environ.get(name)
            if candidate and Path(candidate).resolve().is_relative_to(args.photos):
                raise ValueError(name + ' must be outside photos for temporary query logs')
        print('EXPERIMENTAL: logical dense admission only, NOT hard RSS/VRAM caps. '
              'SfM/fusion physical terms unknown; monitored host RSS is best-effort.', file=sys.stderr)
        reconstruction = Reconstruction(args, str(Path(executable).resolve()))
        query_directory = tempfile.TemporaryDirectory(prefix='colmap-preview-')
        reconstruction.assets = Path(query_directory.name) / 'planning'
        reconstruction.process.workspace = Path(query_directory.name)
        previous = {}
        def cancelled(signum, frame):
            # Do not raise between Popen and child assignment: remember the
            # request so the owned process loop can always drain its session.
            reconstruction.process.cancelled = True
        try:
            for sig in (signal.SIGINT, signal.SIGTERM):
                previous[sig] = signal.signal(sig, cancelled)
            try:
                reconstruction.preview()
            except NativeFailure as error:
                # Query logs are temporary on refusal. Preserve a bounded native
                # diagnostic in the terminal before cleaning their directory.
                log = reconstruction.process.workspace / ('command-%03d.stderr' % reconstruction.process.sequence)
                detail = ''
                if log.is_file():
                    with log.open('rb') as stream:
                        stream.seek(max(0, log.stat().st_size - 4096))
                        detail = stream.read(4096).decode('utf-8', errors='replace').strip()
                raise NativeFailure('plan unavailable: ' + str(error) + ('; ' + detail if detail else '')) from error
            reconstruction.process.remaining()
            if args.plan_only:
                return 0
            if not args.yes:
                approve_plan(reconstruction.process)
            reconstruction.check_approval_binding()
            args.workspace.mkdir(mode=0o700)
            created_workspace = True
            shutil.copytree(query_directory.name, args.workspace, dirs_exist_ok=True)
            reconstruction.assets = args.workspace / 'planning'
            reconstruction.process.workspace = args.workspace
            if reconstruction.extraction_image_list:
                reconstruction.extraction_image_list = ['--image_list_path', reconstruction.assets / 'extraction-images.txt']
            reconstruction.run()
        finally:
            query_directory.cleanup()
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
