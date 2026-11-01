# SPDX-License-Identifier: BSD-3-Clause
"""Fresh fixed-cohort RAW logical inventory, experimental/unqualified.

All caller pixels and frozen copies, policy-selected returned host maps and
all PHOTO device snapshots stay charged throughout. OutputPolicy is the sole
native execution/inventory binding: default retains both host phases; final-only
returns GEO only (16 rather than 32 bytes per reference pixel). Private source
padding preserves ordinal multiplicity. Opt-in shared PHOTO charges a padded
host/device union once in PHOTO only; host validation-copy transients remain
per owner. Global padding can increase memory; default stays private. Explicit unknown pitch/array/context/allocator bytes are NOT
calibrated coefficients or physical-fit proof. Native admission applies the ONE
aggregate 3% margin; this producer does not apply any item margins.

Only loaded grayscale shape metadata is evaluated: no file-map identities,
pixel allocations, GPU queries, source sorting or execution. Source content is
bound with the existing resident allocation closure/algorithm, not a new digest
scheme. The pin includes the optional final-GEO-only disposition and existing
workspace adapter/shared validation and native observed-FREE admission entry,
pending independent review, NOT inherited current-phase-v1/optimized-PHOTO
qualification. Capacity observation/selection remains native-only; this producer
still emits the same raw inventory without any free-memory or context credit. This producer covers native
execution only; workspace decode/resize scratch and
cache overlap require the adapter's separate explicit lifecycle admission.
"""
from dataclasses import dataclass, fields
from enum import Enum
import math
from pathlib import Path
import runpy
import subprocess

from .model import _add, _mul, current_owned_shape_minimum, exact_int

_FIXED_COHORT_FINGERPRINT = '877f6a9fd56088aec736bad6791f60df172793c4fe4ba479f764969338739d61'
_PROFILE = 'fresh-fixed-cohort/current-owned-shape-lower-bound-v1/unqualified'
_ROOT = Path(__file__).resolve().parents[2]


class PhotoSourceStorage(Enum):
    PRIVATE = 'private'
    SHARED = 'shared'


class OutputPolicy(Enum):
    PHOTO_AND_GEO = 'photo_and_geo'
    FINAL_GEO_ONLY = 'final_geo_only'


@dataclass(frozen=True)
class ByteCharge:
    host: int = 0
    device: int = 0

    def __post_init__(self):
        exact_int(self.host)
        exact_int(self.device)

    def __add__(self, other):
        return ByteCharge(_add(self.host, other.host), _add(self.device, other.device))


@dataclass(frozen=True)
class ImageShape:
    width: int
    height: int
    channels: int
    bitmap_num_bytes: int

    def __post_init__(self):
        exact_int(self.width, 1, (1 << 31) - 1)
        exact_int(self.height, 1, (1 << 31) - 1)
        if type(self.channels) is not int or self.channels != 1:
            raise ValueError('native fresh-cohort requires loaded grayscale channels=1')
        # Bitmap(int,int,...) multiplies its dimensions in signed int before
        # vector resize. Refuse shapes that cannot use that allocation safely.
        exact_int(_mul(self.width, self.height), 1, (1 << 31) - 1)
        exact_int(self.bitmap_num_bytes, 1)
        if self.bitmap_num_bytes != _mul(self.width, self.height):
            raise ValueError('producer supports only observed tight grayscale bitmap storage')

    @property
    def num_bytes(self):
        # Bitmap::Allocate/NumBytes: tightly packed uint8 vector W*H*C.
        return self.bitmap_num_bytes


@dataclass(frozen=True)
class PhaseOptions:
    geom_consistency: bool
    filter: bool
    gpu_index: str
    depth_min: float
    depth_max: float
    sigma_spatial: float
    max_image_size: int = -1
    random_seed: int = -1
    window_radius: int = 5
    window_step: int = 1
    sigma_color: float = 0.2
    num_samples: int = 15
    owned_distinct_source_cache: bool = False
    owned_tex_lookahead: bool = False
    owned_cuda_graphs: bool = False
    owned_async_pool: bool = False
    ncc_sigma: float = 0.6
    min_triangulation_angle: float = 1.0
    incident_angle_sigma: float = 0.9
    num_iterations: int = 5
    geom_consistency_regularizer: float = 0.3
    geom_consistency_max_cost: float = 3.0
    filter_min_ncc: float = 0.1
    filter_min_triangulation_angle: float = 3.0
    filter_min_num_consistent: int = 2
    filter_geom_consistency_max_cost: float = 1.0
    write_consistency_graph: bool = False
    allow_missing_files: bool = False
    num_threads: int = 1
    cache_size: float = 32.0

    def validate(self, geo, device):
        bools = ('geom_consistency', 'filter', 'owned_distinct_source_cache',
                 'owned_tex_lookahead', 'owned_cuda_graphs', 'owned_async_pool',
                 'write_consistency_graph', 'allow_missing_files')
        for name in bools:
            if type(getattr(self, name)) is not bool:
                raise ValueError('phase bool must be exact')
        for name, minimum in (('random_seed', -1), ('max_image_size', -1),
                              ('window_radius', 1), ('window_step', 1),
                              ('num_samples', 1), ('num_iterations', 1),
                              ('filter_min_num_consistent', 0), ('num_threads', 1)):
            exact_int(getattr(self, name), minimum, (1 << 31) - 1)
        for field in fields(self):
            v = getattr(self, field.name)
            if field.name not in bools and field.name not in (
                    'gpu_index', 'random_seed', 'max_image_size', 'window_radius',
                    'window_step', 'num_samples', 'num_iterations',
                    'filter_min_num_consistent', 'num_threads'):
                if type(v) not in (int, float) or not math.isfinite(v):
                    raise ValueError('nonfinite/invalid phase numeric option')
        if (self.geom_consistency != geo or (geo and not self.filter) or
                self.gpu_index != device or type(device) is not str or
                not device.isascii() or not device.isdecimal() or
                str(int(device)) != device or int(device) > (1 << 31) - 1 or
                self.window_radius != 5 or self.window_step != 1 or
                self.max_image_size != -1 or self.write_consistency_graph or
                self.allow_missing_files or not 0 < self.depth_min < self.depth_max):
            raise ValueError('unsupported native fixed-cohort phase/options/device')
        if (any(getattr(self, n) <= 0 for n in ('sigma_spatial', 'sigma_color',
                'ncc_sigma', 'incident_angle_sigma', 'cache_size')) or
                any(getattr(self, n) < 0 for n in ('geom_consistency_regularizer',
                'geom_consistency_max_cost', 'filter_geom_consistency_max_cost')) or
                not 0 <= self.min_triangulation_angle < 180 or
                not 0 <= self.filter_min_triangulation_angle <= 180 or
                not -1 <= self.filter_min_ncc <= 1):
            raise ValueError('invalid native PatchMatchOptions')


@dataclass(frozen=True)
class Job:
    reference_image_idx: int
    photo_sources: tuple[int, ...]
    geo_sources: tuple[int, ...]
    photo_options: PhaseOptions
    geo_options: PhaseOptions

    def __post_init__(self):
        # Freeze without changing source order or duplicate ordinals.
        object.__setattr__(self, 'photo_sources', tuple(self.photo_sources))
        object.__setattr__(self, 'geo_sources', tuple(self.geo_sources))


@dataclass(frozen=True)
class SourceBinding:
    revision: str
    fingerprint: str


def _bind_source(root):
    root = Path(root).resolve()
    # Reuse the canonical compiled resident closure verbatim (including its
    # generator). Never execute a caller root's Python or accept a supplied hash.
    generator = runpy.run_path(str(_ROOT / 'scripts/resident/generate_resident_allocation_fingerprint.py'))
    try:
        fingerprint = generator['compute_fingerprint'](root)
    except (SystemExit, OSError) as error:
        raise ValueError('unavailable fixed-cohort source closure') from error
    if fingerprint != _FIXED_COHORT_FINGERPRINT:
        raise ValueError('unknown fixed-cohort source fingerprint; review/repin required')
    revision = subprocess.check_output(
        ['git', '-C', str(root), 'rev-parse', 'HEAD^{commit}'], text=True).strip()
    return SourceBinding(revision, fingerprint)


def active_peak(jobs, active):
    """Resourcewise independent top-active, not a top-by-device job subset."""
    return ByteCharge(
        sum_checked(sorted((j.host for j in jobs), reverse=True)[:active]),
        sum_checked(sorted((j.device for j in jobs), reverse=True)[:active]))


def sum_checked(values):
    total = 0
    for value in values:
        total = _add(total, value)
    return total


@dataclass(frozen=True)
class FixedCohortBudget:
    caller_inputs: ByteCharge
    frozen_inputs: ByteCharge
    host_outputs: ByteCharge
    snapshots: ByteCharge
    process: ByteCharge
    unknown: ByteCharge
    unknown_provenance: str
    headroom_reserve: ByteCharge
    max_active_owners: int
    photo_private: tuple[ByteCharge, ...]
    geo_private: tuple[ByteCharge, ...]
    geo_staging_device: tuple[int, ...]
    source: SourceBinding
    images: tuple[ImageShape, ...]
    jobs: tuple[Job, ...]
    already_live_process: tuple[ByteCharge, ByteCharge]
    already_live_process_provenance: tuple[str, str]
    photo_shared: ByteCharge = ByteCharge()
    photo_source_storage: PhotoSourceStorage = PhotoSourceStorage.PRIVATE
    output_policy: OutputPolicy = OutputPolicy.PHOTO_AND_GEO
    phase_active_owners: tuple[int, int] = (0, 0)
    profile: str = _PROFILE
    physical_fit_certified: bool = False
    calibration_certified: bool = False

    @property
    def retained(self):
        return (self.caller_inputs + self.frozen_inputs + self.host_outputs +
                self.snapshots + self.process + self.unknown)

    @property
    def photo_peak(self):
        return self.retained + self.photo_shared + active_peak(self.photo_private, self.phase_active_owners[0] or self.max_active_owners)

    @property
    def geo_peak(self):
        return self.retained + active_peak(self.geo_private, self.phase_active_owners[1] or self.max_active_owners)

    @property
    def peak(self):
        return ByteCharge(max(self.photo_peak.host, self.geo_peak.host),
                          max(self.photo_peak.device, self.geo_peak.device))

    def native_cpp_initializer(self):
        """Exact experimental typed API binding, not a wire/CLI protocol.

        Produces a C++ lambda expression returning native FixedCohortBudget.
        Compiled in CPU fixture tests against the real native public headers.
        Metadata remains on this typed Python record; no execution authority.
        """
        if not isinstance(self.photo_source_storage, PhotoSourceStorage):
            raise ValueError('expected typed PhotoSourceStorage')
        if not isinstance(self.output_policy, OutputPolicy):
            raise ValueError('expected typed OutputPolicy')
        # Fixed-width octal UTF-8 escapes preserve arbitrary provenance bytes
        # without C++ universal-character/surrogate or quote injection issues.
        def text(value):
            return '"' + ''.join('\\%03o' % b for b in value.encode('utf-8')) + '"'

        charge = lambda b: '{' + str(b.host) + 'ULL,' + str(b.device) + 'ULL}'
        lines = ['[] { colmap::mvs::owned::FixedCohortBudget b;']
        policy = ('kPhotoAndGeo' if self.output_policy == OutputPolicy.PHOTO_AND_GEO
                  else 'kFinalGeoOnly')
        lines.append(f'b.output_policy = colmap::mvs::owned::FixedCohortOutputPolicy::{policy};')
        storage = 'kShared' if self.photo_source_storage == PhotoSourceStorage.SHARED else 'kPrivate'
        lines.append(f'b.photo_source_storage = colmap::mvs::owned::FixedCohortPhotoSourceStorage::{storage};')
        for name in ('photo_shared', 'caller_inputs', 'frozen_inputs', 'host_outputs', 'snapshots',
                     'process', 'unknown', 'headroom_reserve'):
            lines.append(f'b.{name} = {charge(getattr(self, name))};')
        lines.append(f'b.unknown_provenance = {text(self.unknown_provenance)};')
        lines.append(f'b.max_active_owners = {self.max_active_owners};')
        lines.append(f'b.phase_active_owners = {{{self.phase_active_owners[0]},{self.phase_active_owners[1]}}};')
        for name in ('photo_private', 'geo_private'):
            lines.append(f'b.{name} = {{' + ','.join(map(charge, getattr(self, name))) + '};')
        lines.append('b.geo_staging_device = {' + ','.join(str(x)+'ULL' for x in self.geo_staging_device) + '};')
        for i in range(2):
            lines.append(f'b.already_live_process[{i}] = {charge(self.already_live_process[i])};')
            lines.append(f'b.already_live_process_provenance[{i}] = ' + text(self.already_live_process_provenance[i]) + ';')
        return '\n'.join(lines + ['return b; }()'])


def forecast_fixed_cohort(images, jobs, *, max_active_owners, source_root,
                          process, unknown, unknown_provenance, headroom_reserve,
                          already_live_process=(ByteCharge(), ByteCharge()),
                          already_live_process_provenance=('', ''),
                          output_policy=OutputPolicy.PHOTO_AND_GEO,
                          photo_source_storage=PhotoSourceStorage.PRIVATE,
                          phase_active_owners=(0, 0)):
    """Evaluate metadata only; capacities and the aggregate margin belong native.

    Process is caller-classified raw storage (not a new coefficient). Live
    process credits require independent per-observation provenance and remain
    INCLUDED in process. Unknown is positive, explicit and never inferred from
    free memory. Source/device/options/order binding does not hash caller pixels
    or calibration: native still validates actual inputs and frozen invocation.
    """
    if not isinstance(photo_source_storage, PhotoSourceStorage):
        raise ValueError('expected typed PhotoSourceStorage')
    if not isinstance(output_policy, OutputPolicy):
        raise ValueError('expected typed OutputPolicy')
    images, jobs = tuple(images), tuple(jobs)
    exact_int(max_active_owners, 1, max(2, len(jobs)))
    phase_active_owners = tuple(phase_active_owners)
    for count in phase_active_owners:
        exact_int(count, 0, len(jobs))
    if len(phase_active_owners) != 2:
        raise ValueError('expected PHOTO/GEO counts')
    if phase_active_owners != (0, 0):
        for count in phase_active_owners:
            exact_int(count, 1, len(jobs))
        if max(phase_active_owners) != max_active_owners:
            raise ValueError('phase count maximum must match max_active_owners')
    if not images or not jobs:
        raise ValueError('empty fixed cohort')
    for image in images:
        if not isinstance(image, ImageShape):
            raise ValueError('expected ImageShape')
    if not isinstance(unknown_provenance, str) or not unknown_provenance.strip() or '\x00' in unknown_provenance:
        raise ValueError('unknown pitch/context/allocator allowance needs provenance')
    if not isinstance(unknown, ByteCharge) or not (unknown.host or unknown.device):
        raise ValueError('positive explicit unknown allowance required')
    if not isinstance(process, ByteCharge) or not isinstance(headroom_reserve, ByteCharge):
        raise ValueError('expected typed raw process/headroom')
    live, provenance = tuple(already_live_process), tuple(already_live_process_provenance)
    if len(live) != 2 or len(provenance) != 2:
        raise ValueError('two observation credits required')
    for value, why in zip(live, provenance):
        if (not isinstance(value, ByteCharge) or not isinstance(why, str) or '\x00' in why or
                value.host > process.host or value.device > process.device or
                ((value.host or value.device) and not why.strip())):
            raise ValueError('invalid classified live process credit/provenance')
    producers = []
    for job in jobs:
        if not isinstance(job, Job):
            raise ValueError('expected Job')
        exact_int(job.reference_image_idx, 0, len(images) - 1)
        producers.append(job.reference_image_idx)
    if len(set(producers)) != len(producers):
        raise ValueError('duplicate producer')
    device = jobs[0].photo_options.gpu_index
    photo, geo, staging = [], [], []
    shared = photo_source_storage == PhotoSourceStorage.SHARED
    union = set()
    for job in jobs:
        ref = images[job.reference_image_idx]
        for geometry, sources, options in (
                (False, job.photo_sources, job.photo_options),
                (True, job.geo_sources, job.geo_options)):
            options.validate(geometry, device)
            if not sources:
                raise ValueError('empty ordered sources')
            for index in sources:
                exact_int(index, 0, len(images) - 1)
                if index == job.reference_image_idx or (geometry and index not in producers):
                    raise ValueError('self source or missing PHOTO producer')
            padded = _mul(_mul(max(images[i].width for i in sources),
                               max(images[i].height for i in sources)), len(sources))
            host, gpu = current_owned_shape_minimum(ref.width, ref.height, len(sources))
            if not geometry:
                union.update(sources)
            # Host ordered Image-copy validation transient remains per owner.
            bound = ByteCharge(_add(host, padded),
                               _add(_add(gpu, 0 if shared and not geometry else padded),
                                    _mul(8, padded) if geometry else 0))
            (geo if geometry else photo).append(bound)
            if geometry:
                staging.append(_mul(4, padded))
    union_bytes = (_mul(_mul(max(images[i].width for i in union),
                             max(images[i].height for i in union)), len(union))
                   if shared else 0)
    inputs = sum_checked(image.num_bytes for image in images)
    maps = _mul(sum_checked(images[i].num_bytes for i in producers), 16)
    host_phases = 2 if output_policy == OutputPolicy.PHOTO_AND_GEO else 1
    return FixedCohortBudget(
        ByteCharge(inputs), ByteCharge(inputs), ByteCharge(_mul(host_phases, maps)),
        ByteCharge(0, maps), process, unknown, unknown_provenance, headroom_reserve,
        max_active_owners, tuple(photo), tuple(geo), tuple(staging),
        _bind_source(source_root), images, jobs, live, provenance,
        ByteCharge(union_bytes, union_bytes), photo_source_storage, output_policy,
        phase_active_owners)
