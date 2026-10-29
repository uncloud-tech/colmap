# SPDX-License-Identifier: BSD-3-Clause
"""CPU-only public producer/native-header comparisons; no image allocation."""
import unittest

from colmap_reconstruction.model import current_owned_shape_minimum


class CanonicalShapeTest(unittest.TestCase):
    def test_native_lower_bound_worked_example(self):
        # Native BuildOwnedRotationMemoryPhases(5,4,3): init=2972,
        # steady=3052. Resident adds 40 + 2076 + 32 bytes.
        self.assertEqual(current_owned_shape_minimum(5, 4, 3), (480, 5200))


class FixedCohortTest(unittest.TestCase):
    def test_bitmap_native_signed_allocation_range(self):
        from colmap_reconstruction.fixed_cohort import ImageShape
        with self.assertRaises(ValueError):
            ImageShape(65536, 65536, 1, 4294967296)

    def test_bitmap_length_is_explicit_observed_metadata(self):
        from colmap_reconstruction.fixed_cohort import ImageShape
        self.assertEqual(ImageShape(5, 4, 1, 20).num_bytes, 20)
        with self.assertRaises(ValueError):
            ImageShape(5, 4, 1, 21)
        with self.assertRaises(ValueError):
            ImageShape(5, 4, 1, 19)

    def test_four_jobs_raw_retention_and_duplicates(self):
        from colmap_reconstruction.fixed_cohort import (
            ByteCharge, ImageShape, Job, PhaseOptions, forecast_fixed_cohort,
        )
        from pathlib import Path
        images = (ImageShape(5, 4, 1, 20), ImageShape(3, 3, 1, 9)) * 2
        ids = (3, 0, 2, 1)
        jobs = tuple(Job(ref, (ids[(i+1)%4], ids[(i+2)%4], ids[(i+1)%4]),
                         (ids[(i+1)%4], ids[(i+2)%4], ids[(i+1)%4]),
                         PhaseOptions(False, False, '0', 1, 5, 5),
                         PhaseOptions(True, True, '0', 1, 5, 5))
                     for i, ref in enumerate(ids))
        budget = forecast_fixed_cohort(
            images, jobs, max_active_owners=2,
            source_root=Path(__file__).resolve().parents[2],
            process=ByteCharge(100, 200), unknown=ByteCharge(300, 400),
            unknown_provenance='explicit test-only pitch/allocator allowance',
            headroom_reserve=ByteCharge(10, 20))
        self.assertEqual(budget.caller_inputs, ByteCharge(58, 0))
        self.assertEqual(budget.frozen_inputs, ByteCharge(58, 0))
        self.assertEqual(budget.host_outputs, ByteCharge(1856, 0))
        self.assertEqual(budget.snapshots, ByteCharge(0, 928))
        self.assertEqual(budget.geo_staging_device, (240, 240, 108, 240))
        self.assertEqual(budget.jobs, jobs)
        self.assertEqual(budget.photo_private[1], ByteCharge(540, 5260))
        self.assertEqual(budget.geo_private[1], ByteCharge(540, 5740))


class RefusalAndLifetimeTest(unittest.TestCase):
    def setUp(self):
        from pathlib import Path
        from colmap_reconstruction.fixed_cohort import ByteCharge, ImageShape, Job, PhaseOptions
        self.root = Path(__file__).resolve().parents[2]
        self.images = (ImageShape(5, 4, 1, 20), ImageShape(3, 3, 1, 9), ImageShape(2, 7, 1, 14))
        photo = PhaseOptions(False, False, '0', 1, 5, 5)
        geo = PhaseOptions(True, True, '0', 1, 5, 5)
        self.jobs = (Job(1, (0, 2, 0), (0, 0), photo, geo),
                     Job(0, (1,), (1,), photo, geo))
        self.kwargs = dict(max_active_owners=1, source_root=self.root,
                           process=ByteCharge(100, 200), unknown=ByteCharge(50, 60),
                           unknown_provenance='explicit uncalibrated test allowance',
                           headroom_reserve=ByteCharge(10, 20))

    def forecast(self, images=None, jobs=None, **kwargs):
        from colmap_reconstruction.fixed_cohort import forecast_fixed_cohort
        return forecast_fixed_cohort(self.images if images is None else images,
                                     self.jobs if jobs is None else jobs,
                                     **(self.kwargs | kwargs))

    def test_all_images_retained_but_only_producer_maps(self):
        from colmap_reconstruction.fixed_cohort import ByteCharge
        one = self.forecast()
        two = self.forecast(max_active_owners=2)
        # Image2 is PHOTO-only source, never a producer. Still copied twice.
        self.assertEqual(one.caller_inputs, ByteCharge(43))
        self.assertEqual(one.frozen_inputs, ByteCharge(43))
        self.assertEqual(one.host_outputs, ByteCharge(928))
        self.assertEqual(one.snapshots, ByteCharge(0, 464))
        self.assertEqual(one.retained, two.retained)
        self.assertLess(one.peak.host, two.peak.host)
        self.assertLess(one.peak.device, two.peak.device)
        self.assertFalse(one.physical_fit_certified)
        self.assertFalse(one.calibration_certified)
        self.assertIn('unqualified', one.profile)
        self.assertEqual(len(one.source.revision), 40)
        self.assertEqual(len(one.source.fingerprint), 64)

    def test_resourcewise_top_active_and_checked_overflow(self):
        from colmap_reconstruction.fixed_cohort import ByteCharge, active_peak
        from colmap_reconstruction.model import _add, _mul
        crossed = (ByteCharge(100, 1), ByteCharge(1, 200), ByteCharge(90, 2), ByteCharge(2, 150))
        self.assertEqual(active_peak(crossed, 1), ByteCharge(100, 200))
        self.assertEqual(active_peak(crossed, 2), ByteCharge(190, 350))
        with self.assertRaises(OverflowError):
            active_peak((ByteCharge((1 << 64)-1), ByteCharge(1)), 2)
        with self.assertRaises(OverflowError):
            _add((1 << 64)-1, 1)
        with self.assertRaises(OverflowError):
            _mul(1 << 63, 2)
        with self.assertRaises(OverflowError):
            current_owned_shape_minimum(1 << 28, 1, 2)

    def test_phase_channels_dimensions_options_device_and_order_refuse(self):
        from dataclasses import replace
        from colmap_reconstruction.fixed_cohort import ImageShape
        for args in ((0, 1, 1, 1), (1, -1, 1, 1), (1, 1, 3, 3), (True, 1, 1, 1)):
            with self.subTest(shape=args), self.assertRaises(ValueError):
                ImageShape(*args)
        bad_options = dict(geom_consistency=True, gpu_index='00', window_radius=4,
                           window_step=2, max_image_size=100, num_threads=-1,
                           allow_missing_files=True, write_consistency_graph=True,
                           depth_min=-1, depth_max=float('inf'), sigma_spatial=float('nan'),
                           sigma_color=0, num_samples=True, filter_min_ncc=2,
                           cache_size=0)
        for key, value in bad_options.items():
            with self.subTest(option=key), self.assertRaises(ValueError):
                first = replace(self.jobs[0], photo_options=replace(self.jobs[0].photo_options, **{key:value}))
                self.forecast(jobs=(first, self.jobs[1]))
        for change in (dict(geo_sources=(2,)), dict(photo_sources=(1,)),
                       dict(photo_sources=()), dict(geo_sources=(-1,)),
                       dict(geo_options=replace(self.jobs[0].geo_options, filter=False)),
                       dict(geo_options=replace(self.jobs[0].geo_options, gpu_index='1'))):
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.forecast(jobs=(replace(self.jobs[0], **change), self.jobs[1]))
        with self.assertRaises(ValueError):
            self.forecast(jobs=(self.jobs[0], self.jobs[0]))
        for active in (0, 3, True):
            with self.assertRaises(ValueError):
                self.forecast(max_active_owners=active)

    def test_unknown_allowance_and_classified_observation_credits(self):
        from colmap_reconstruction.fixed_cohort import ByteCharge
        for kwargs in (dict(unknown=ByteCharge()), dict(unknown_provenance='  '),
                       dict(already_live_process=(ByteCharge(101), ByteCharge())),
                       dict(already_live_process=(ByteCharge(1), ByteCharge())),
                       dict(already_live_process=(ByteCharge(),))):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.forecast(**kwargs)
        base = self.forecast()
        live = self.forecast(already_live_process=(ByteCharge(10, 20), ByteCharge(3, 4)),
                             already_live_process_provenance=('before copy measured process only',
                                                               'before GEO separately measured process only'))
        self.assertEqual(live.retained, base.retained)
        self.assertEqual(live.peak, base.peak)

    def test_missing_source_refuses_without_process_exit(self):
        import tempfile
        with tempfile.TemporaryDirectory(prefix='cohort-missing-', dir=self.root) as root:
            with self.assertRaisesRegex(ValueError, 'source closure'):
                self.forecast(source_root=root)

    def test_unknown_actual_source_content_refuses(self):
        import runpy
        import tempfile
        from pathlib import Path
        closure = runpy.run_path(str(self.root / 'scripts/resident/generate_resident_allocation_fingerprint.py'))['CLOSURE']
        with tempfile.TemporaryDirectory(prefix='cohort-source-', dir=self.root) as directory:
            root = Path(directory)
            for name in closure:
                target = root / name
                target.parent.mkdir(parents=True, exist_ok=True)
                if name.endswith('owned_fixed_cohort_pipeline.cu'):
                    target.write_bytes((self.root / name).read_bytes() + b'\n// unknown content\n')
                else:
                    target.symlink_to(self.root / name)
            with self.assertRaisesRegex(ValueError, 'unknown.*fingerprint'):
                self.forecast(source_root=root)


class NativeBindingTest(unittest.TestCase):
    def test_generated_four_job_budget_compiles_and_matches_native(self):
        import subprocess
        import tempfile
        from pathlib import Path
        from colmap_reconstruction.fixed_cohort import (
            ByteCharge, ImageShape, Job, PhaseOptions, forecast_fixed_cohort,
        )
        root = Path(__file__).resolve().parents[2]
        images = (ImageShape(5, 4, 1, 20), ImageShape(3, 3, 1, 9)) * 2
        ids = (3, 0, 2, 1)
        jobs = tuple(Job(ref, (ids[(i+1)%4], ids[(i+2)%4], ids[(i+1)%4]),
                         (ids[(i+1)%4], ids[(i+2)%4], ids[(i+1)%4]),
                         PhaseOptions(False, False, '0', 1, 5, 5),
                         PhaseOptions(True, True, '0', 1, 5, 5))
                     for i, ref in enumerate(ids))
        for active in (1, 2):
            budget = forecast_fixed_cohort(
                images, jobs, max_active_owners=active, source_root=root,
                process=ByteCharge(100, 200), unknown=ByteCharge(300, 400),
                unknown_provenance='explicit test-only allowance ' + chr(0x1f9ea),
                headroom_reserve=ByteCharge(10, 20),
                already_live_process=(ByteCharge(10, 20), ByteCharge(3, 4)),
                already_live_process_provenance=('copy classified', 'GEO classified'))
            checks = []
            for i, job in enumerate(jobs):
                ref = images[job.reference_image_idx]
                for geometry, sources in ((False, job.photo_sources), (True, job.geo_sources)):
                    bound = (budget.geo_private if geometry else budget.photo_private)[i]
                    padded = max(images[k].width for k in sources) * max(images[k].height for k in sources) * len(sources)
                    checks.append(f'''{{ auto m = ResidentPrivateLogicalMinimum({ref.width},{ref.height},{len(sources)},false);
                        assert(m.host + {padded} == {bound.host});
                        assert(m.device + {padded} + {8*padded if geometry else 0} == {bound.device}); }}''')
            # Degenerate + rotated dimensions compare canonical shape helper to native.
            for w, h, s in ((1, 1, 1), (1, 9, 2), (11, 1, 4), (7, 19, 5), (19, 7, 5)):
                host, device = current_owned_shape_minimum(w, h, s)
                checks.append(f'{{ auto m=ResidentPrivateLogicalMinimum({w},{h},{s},false); assert(m.host=={host} && m.device=={device}); }}')
            source = '''#include "colmap/mvs/owned_fixed_cohort_budget.h"
#include <cassert>
#include <iostream>
using namespace colmap::mvs::owned;
int main() {
auto b = ''' + budget.native_cpp_initializer() + ''';
''' + '\n'.join(checks) + f'''
auto p = FixedCohortPeaks(b);
assert(p.photo.host == {budget.photo_peak.host} && p.photo.device == {budget.photo_peak.device});
assert(p.geo.host == {budget.geo_peak.host} && p.geo.device == {budget.geo_peak.device});
assert(p.peak.host == {budget.peak.host} && p.peak.device == {budget.peak.device});
auto admitted = AdmitFixedCohortBudget(b, {{1000000,1000000}});
assert(admitted.host == MarginBytes(p.peak.host));
assert(admitted.device == MarginBytes(p.peak.device));
assert(b.host_outputs.host==1856 && b.snapshots.device==928);
assert(b.geo_staging_device == std::vector<uint64_t>({{240,240,108,240}}));
assert(b.photo_private.size()==4 && b.max_active_owners=={active});
assert(b.caller_inputs.host==58 && b.frozen_inputs.host==58);
assert(b.process.host==100 && b.process.device==200);
assert(b.unknown.host==300 && b.unknown.device==400);
assert(b.headroom_reserve.host==10 && b.headroom_reserve.device==20);
assert(b.already_live_process[0].host==10 && b.already_live_process[0].device==20);
assert(b.already_live_process[1].host==3 && b.already_live_process[1].device==4);
assert(b.already_live_process_provenance[0]=="copy classified");
assert(b.already_live_process_provenance[1]=="GEO classified");
assert(b.unknown_provenance == std::string("explicit test-only allowance ") +
       char(0xf0) + char(0x9f) + char(0xa7) + char(0xaa));
auto crossed=FixedCohortActivePeak({{{{100,1}},{{1,200}},{{90,2}},{{2,150}}}},2);
assert(crossed.host==190 && crossed.device==350);
CheckFixedCohortPhysicalCopies(b, b.snapshots.device, b.geo_staging_device);
// The original aggregate margin absorbs discrepancies; no second margin.
auto slack=MarginBytes(p.geo.device)-p.geo.device;
CheckFixedCohortPhysicalCopies(b,b.snapshots.device,{{240+slack,240,108,240}});
bool refused=false;
try {{ CheckFixedCohortPhysicalCopies(b,b.snapshots.device,{{241+slack,240,108,240}}); }}
catch(const std::invalid_argument&) {{ refused=true; }}
assert(refused);
refused=false;
try {{ AdmitFixedCohortBudget(b,{{admitted.host+b.headroom_reserve.host-1,1000000}}); }}
catch(const std::invalid_argument&) {{ refused=true; }}
assert(refused);
std::cout << admitted.host << " " << admitted.device;
}}
'''
            with tempfile.TemporaryDirectory(prefix='cohort-cpu-', dir=root) as directory:
                cpp, binary = Path(directory) / 'fixture.cc', Path(directory) / 'fixture'
                cpp.write_text(source)
                compiled = subprocess.run(['g++', '-std=c++17', '-pthread', '-Wall', '-Wextra', '-Werror',
                                '-I'+str(root/'src'), str(cpp), '-o', str(binary)],
                               capture_output=True, text=True)
                self.assertEqual(compiled.returncode, 0, compiled.stderr)
                result = subprocess.run([str(binary)], check=True, capture_output=True, text=True)
                # Independent once-margin totals for these fixed raw budgets.
                self.assertEqual(result.stdout, '3000 7487' if active == 1 else '3522 13093')


class WorkspaceFixtureBindingTest(unittest.TestCase):
    def test_canonical_generated_workspace_binding(self):
        import runpy
        from pathlib import Path
        root = Path(__file__).resolve().parents[2]
        tool = runpy.run_path(str(root / 'src/colmap/mvs/testdata/generate_fixed_cohort_workspace_fixture.py'))
        spec = tool['synthetic_spec']()
        for reference in spec['references']:
            for phase in ('photo_sources', 'geo_sources'):
                sources = reference[phase]
                self.assertEqual(len(sources), 4)
                self.assertEqual(len(set(sources)), 3)
                self.assertEqual(sources[0], sources[3])
                self.assertNotIn(reference['reference'], sources)
        header = tool['emit_header'](spec, root)
        self.assertIn('MakeWorkspaceFixture', header)
        self.assertIn('b.max_active_owners = 1', header)
        self.assertIn('b.max_active_owners = 2', header)
        self.assertIn(tool['cpp']('28dff08c2a392ed04fa046ade5e2717e51f922baece3b09dd91a4030ac8bfab0'), header)
        self.assertIn('b.caller_inputs = {12288ULL,0ULL}', header)
        self.assertIn('b.host_outputs = {393216ULL,0ULL}', header)
        self.assertIn('b.snapshots = {0ULL,196608ULL}', header)
        self.assertIn('b.geo_staging_device = {49152ULL,49152ULL,49152ULL,49152ULL}', header)
        self.assertIn('r.photo_options.random_seed = 0;', header)
        self.assertIn('r.geo_options.filter = true;', header)
        # External requests must freeze all options, not silently use defaults.
        spec['workspace'] = '/read-only/undistorted'
        del spec['references'][0]['photo_options']['num_samples']
        with self.assertRaisesRegex(ValueError, 'explicit phase options'):
            tool['emit_header'](spec, root)

    def test_external_binding_requires_repeatable_seed(self):
        import runpy
        from pathlib import Path
        root = Path(__file__).resolve().parents[2]
        tool = runpy.run_path(str(root / 'src/colmap/mvs/testdata/generate_fixed_cohort_workspace_fixture.py'))
        spec = tool['synthetic_spec']()
        spec['workspace'] = '/read-only/undistorted'
        spec['references'][0]['photo_options']['random_seed'] = -1
        with self.assertRaisesRegex(ValueError, 'explicit repeatable seed'):
            tool['emit_header'](spec, root)


if __name__ == '__main__':
    unittest.main()
