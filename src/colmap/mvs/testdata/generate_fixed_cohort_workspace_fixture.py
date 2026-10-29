# SPDX-License-Identifier: BSD-3-Clause
"""Test-only literal workspace request -> C++ header, never an execution CLI.

Default: four 64x48 translated pinhole cameras observing a textured plane z=4.
External JSON has the same schema as synthetic_spec(); images are the COMPLETE
loaded workspace bank after native max_load_image_size resizing, in any explicit
order. Names, reference order and source ordinal multiplicity are preserved.
Every phase option must be supplied. GEO sources must close over four producers.
The caller freezes sparse/images separately; no image decoding or forecast
coefficients here. All RAW inventory comes from the canonical producer.

Preparation (no CUDA or dataset mutation):
  python <this-file> --sample --output /new/private/request.json
  # Fill literal workspace, COMPLETE loaded image shapes (e.g. max size 512),
  # four reference names, exact phase source orders/options, raw allowances.
  # Keep GEO closed on those four; PHOTO may use other loaded images.
  python <this-file> --spec /new/private/request.json --output /new/private/request.h
  cmake ... -DCOLMAP_FIXED_COHORT_EXTERNAL_HEADER=/new/private/request.h
  cmake --build <build> --target colmap_mvs_owned_fixed_cohort_pipeline_test
  COLMAP_TEST_FIXED_COHORT_WORKSPACE_GPU=1 <existing-test-binary> \\
      --gtest_filter='FixedCohortWorkspaceGpu.*' --gtest_output=xml:/new/result.xml

External data is read-only; no maps, sparse files, or logs are written there.
Freeze/identify input sparse/images, literal JSON/header, test source and test
ELF separately from the unchanged production allocation fingerprint. Runtime
captures profiler/ELF/hardware evidence through existing launch tooling. A
512-pixel run is explicitly a resized correctness fixture, not prior 1600-pixel
9-photo performance evidence. Unknown/process allowances must also cover the
sequential independent control's loader/banks/returned maps; they are never a
certified peak. External process/resource limits remain caller responsibility.
"""
import argparse
from dataclasses import asdict, fields, replace
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'python'))
from colmap_reconstruction.fixed_cohort import (  # noqa: E402
    ByteCharge, ImageShape, Job, PhaseOptions, forecast_fixed_cohort,
)


def synthetic_spec():
    photo = PhaseOptions(False, False, '0', 3, 5, 5, random_seed=0,
                         num_iterations=3, filter_min_ncc=0,
                         filter_min_triangulation_angle=1,
                         filter_min_num_consistent=1,
                         filter_geom_consistency_max_cost=2)
    ids = (3, 0, 2, 1)
    names = [f'plane{i}.png' for i in range(4)]
    refs = []
    for i, ref in enumerate(ids):
        sources = [names[ids[(i+j) % 4]] for j in (1, 2, 3)]
        refs.append(dict(reference=names[ref], photo_sources=sources,
                         geo_sources=sources,
                         photo_options=asdict(photo),
                         geo_options=asdict(replace(photo, geom_consistency=True, filter=True))))
    return dict(workspace='', max_load_image_size=-1,
                images=[dict(name=n, width=64, height=48) for n in names],
                references=refs, capacity=[2 << 30, 2 << 30],
                process=[64 << 20, 64 << 20], unknown=[128 << 20, 128 << 20],
                unknown_provenance='UNCALIBRATED fixture allowance: codec, OIIO, context, pitch, allocator; not physical-fit proof',
                headroom_reserve=[64 << 20, 64 << 20],
                loader_scratch_host=64 << 20, deadline_seconds=180)


def cpp(value):
    if isinstance(value, str):
        return '"' + ''.join('\\%03o' % b for b in value.encode()) + '"'
    if isinstance(value, bool):
        return str(value).lower()
    return repr(value)


def emit_header(spec, root=ROOT):
    images = spec['images']
    names = [i['name'] for i in images]
    if len(set(names)) != len(names) or len(spec['references']) != 4:
        raise ValueError('unique image names and exactly four producers required')
    shapes = tuple(ImageShape(i['width'], i['height'], 1, i['width']*i['height']) for i in images)
    option_names = {f.name for f in fields(PhaseOptions)}
    jobs = []
    for ref in spec['references']:
        for phase in ('photo_options', 'geo_options'):
            if set(ref[phase]) != option_names:
                raise ValueError('all explicit phase options required')
            if ref[phase]['random_seed'] < 0:
                raise ValueError('bitwise fixture needs an explicit repeatable seed')
            if ref[phase]['owned_cuda_graphs']:
                raise ValueError('freeze stream options; test explicitly varies only graphs and active owners')
        jobs.append(Job(names.index(ref['reference']),
                        tuple(names.index(n) for n in ref['photo_sources']),
                        tuple(names.index(n) for n in ref['geo_sources']),
                        PhaseOptions(**ref['photo_options']), PhaseOptions(**ref['geo_options'])))
    if spec['max_load_image_size'] != -1 and spec['max_load_image_size'] <= 0:
        raise ValueError('invalid native image size')
    for key in ('capacity', 'process', 'unknown', 'headroom_reserve'):
        if len(spec[key]) != 2:
            raise ValueError('host/device pair required')
        ByteCharge(*spec[key])
    if type(spec['deadline_seconds']) is not int or not 0 < spec['deadline_seconds'] <= 3600:
        raise ValueError('bounded integer deadline required')
    if type(spec['loader_scratch_host']) is not int or spec['loader_scratch_host'] <= 0:
        raise ValueError('positive explicit decoder scratch bound required')
    budgets = [forecast_fixed_cohort(shapes, jobs, max_active_owners=a, source_root=root,
                process=ByteCharge(*spec['process']), unknown=ByteCharge(*spec['unknown']),
                unknown_provenance=spec['unknown_provenance'],
                headroom_reserve=ByteCharge(*spec['headroom_reserve'])) for a in (1, 2)]
    identity = json.dumps(spec, sort_keys=True, separators=(',', ':'))
    lines = ['// Generated test-only canonical RAW request. Do not hand edit.',
             'WorkspaceFixture MakeWorkspaceFixture() { WorkspaceFixture f;',
             f'f.workspace = {cpp(spec["workspace"])};',
             f'f.max_size = {spec["max_load_image_size"]};',
             f'f.capacity = {{{spec["capacity"][0]}ULL,{spec["capacity"][1]}ULL}};',
             f'f.scratch = {spec["loader_scratch_host"]}ULL;',
             f'f.seconds = {spec["deadline_seconds"]};',
             f'f.spec = {cpp(identity)};',
             f'f.fingerprint = {cpp(budgets[0].source.fingerprint)};',
             f'f.revision = {cpp(budgets[0].source.revision)};']
    for filename in ('generate_fixed_cohort_workspace_fixture.py', '../owned_fixed_cohort_workspace_execution_test.cu'):
        path = Path(__file__).parent / filename
        if path.exists():
            lines.append(f'f.test_identities += {cpp(filename + ":" + hashlib.sha256(path.read_bytes()).hexdigest() + ";")};')
    for i in images:
        lines.append(f'f.shapes.push_back({{{cpp(i["name"])},{i["width"]},{i["height"]}}});')
    for ref in spec['references']:
        lines += ['{ FixedCohortWorkspaceReference r;', f'r.reference = {cpp(ref["reference"])};']
        for phase in ('photo', 'geo'):
            lines.append(f'r.{phase}_sources = {{' + ','.join(cpp(n) for n in ref[phase+'_sources']) + '};')
            for key, value in ref[phase+'_options'].items():
                lines.append(f'r.{phase}_options.{key} = {cpp(value)};')
        lines += ['f.references.push_back(r); }']
    for i, budget in enumerate(budgets):
        lines.append(f'f.budgets[{i}] = {budget.native_cpp_initializer()};')
    return '\n'.join(lines + ['return f; }', ''])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--sample', action='store_true', help='write literal JSON sample instead of C++')
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text()) if args.spec else synthetic_spec()
    text = json.dumps(spec, indent=2) + '\n' if args.sample else emit_header(spec)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text)
