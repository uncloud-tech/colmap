# SPDX-License-Identifier: BSD-3-Clause
"""Real frontend processes with a synthetic native boundary, not GPU qualification."""
import json
import os
import pty
import select
import signal
import time
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from colmap_reconstruction.resource_policy import ALLOCATION_FINGERPRINT


@unittest.skipUnless(sys.platform == 'linux' and os.geteuid() != 0,
                     'frontend execution requires non-root Linux')
class ApprovalTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.photos = self.root / 'photos'
        self.photos.mkdir()
        (self.photos / 'photo.jpg').write_bytes(b'original')
        self.workspace = self.root / 'output'
        self.calls = self.root / 'calls.jsonl'
        self.native = self.root / 'native'
        self.native.write_text('#!' + sys.executable + '\n' + '''
import json, os, sys, signal
from pathlib import Path
with Path(os.environ['TEST_CALLS']).open('a') as log:
    log.write(json.dumps(sys.argv[1:]) + '\\n')
command = sys.argv[1]
if command == 'reconstruction_info':
    if len(sys.argv) > 2 and os.environ.get('TEST_MUTATE_ON_REOBSERVE'):
        calls = [json.loads(line) for line in Path(os.environ['TEST_CALLS']).read_text().splitlines()]
        if sum(call[0] == 'reconstruction_info' for call in calls) >= 3:
            Path(os.environ['TEST_MUTATE_ON_REOBSERVE']).write_bytes(b'changed at launch boundary')
    if len(sys.argv) > 2 and os.environ.get('TEST_BLOCK_QUERY'):
        with open(os.environ['TEST_BLOCK_QUERY'], 'w') as fifo:
            fifo.write(str(os.getpid()) + '\\n')
        signal.pause()
    if len(sys.argv) > 2 and os.environ.get('TEST_QUERY_FAIL'):
        print('CUDA device unavailable', file=sys.stderr)
        sys.exit(3)
    print(json.dumps(dict(schema_version=1, cuda_compiled=not bool(os.environ.get('TEST_NO_CUDA')),
        source_commit='a' * 40, allocation_fingerprint=FINGERPRINT,
        available_host_bytes=32 * 1024**3,
        available_device_bytes=(int(Path(os.environ['TEST_FREE']).read_text()) if Path(os.environ['TEST_FREE']).exists() else 16 * 1024**3),
        device_uuid=(Path(os.environ['TEST_DEVICE']).read_text() if Path(os.environ['TEST_DEVICE']).exists() else 'GPU-test'))))
elif os.environ.get('TEST_SFM_PROCEED') and command in ('feature_extractor', 'exhaustive_matcher'):
    pass
elif os.environ.get('TEST_SFM_PROCEED') and command == 'mapper':
    (Path(sys.argv[sys.argv.index('--output_path') + 1]) / '0').mkdir()
elif os.environ.get('TEST_SFM_PROCEED') and command == 'model_analyzer':
    print(json.dumps(dict(num_registered_images=2, num_points3D=1)))
elif os.environ.get('TEST_SFM_PROCEED') and command == 'image_undistorter':
    images = Path(sys.argv[sys.argv.index('--output_path') + 1]) / 'images'
    images.mkdir(parents=True)
    (images / 'photo.jpg').write_bytes(b'warped image fixture')
elif os.environ.get('TEST_SFM_PROCEED') and command == 'optimized_dense_plan':
    workspace = Path(sys.argv[sys.argv.index('--workspace_path') + 1])
    # Prescribed native response: these native quality options must survive dispatch.
    options = dict(max_image_size=-1, num_samples=7, random_seed=42, gpu_index='2',
                   num_iterations=5, window_radius=5, window_step=1)
    jobs = [dict(reference=0, sources=[1], options=options),
            dict(reference=1, sources=[0], options=options)]
    Path(sys.argv[sys.argv.index('--output_path') + 1]).write_text(json.dumps(dict(
        schema_version=1, workspace_path=str(workspace), build=dict(revision='a' * 8),
        phases={phase: dict(jobs=jobs, requested_options=options) for phase in ('photo', 'geo')},
        images=[dict(index=index, name='photo.jpg', path=str(workspace / 'images/photo.jpg'),
                     width=6405, height=4187) for index in (0, 1)], input_files=[])))
elif command == 'photo_directory_summary':
    print(json.dumps(dict(schema_version=1, header_only=True, orientation_applied=False,
        photo_count=1, photos=[dict(path='photo.jpg', width=4004, height=3001, orientation=None)],
        non_image_files=[], problems=[], build=dict(revision='a' * 8, cuda_compiled=True))))
else:
    print('TEST native execution reached', file=sys.stderr)
    sys.exit(17)
'''.replace('FINGERPRINT', repr(ALLOCATION_FINGERPRINT)))
        self.native.chmod(0o700)
        self.argv = [sys.executable, '-m', 'colmap_reconstruction.cli',
                     '--experimental-local', '--colmap', str(self.native),
                     '--photos', str(self.photos), '--workspace', str(self.workspace)]
        self.query_tmp = self.root / 'query-tmp'
        self.query_tmp.mkdir()
        self.device_state = self.root / 'device-state'
        self.env = dict(os.environ, TEST_CALLS=str(self.calls), TMPDIR=str(self.query_tmp),
                        TEST_DEVICE=str(self.device_state), TEST_FREE=str(self.root / 'free'))

    def invoke(self, *options, stdin=''):
        return subprocess.run(self.argv + list(options), input=stdin, text=True,
                              capture_output=True, env=self.env, timeout=10)

    def commands(self):
        return ([json.loads(line) for line in self.calls.read_text().splitlines()]
                if self.calls.exists() else [])

    def assert_preview_only(self):
        self.assertFalse(self.workspace.exists())
        self.assertEqual((self.photos / 'photo.jpg').read_bytes(), b'original')
        self.assertTrue(all(call[0] in ('reconstruction_info', 'photo_directory_summary')
                            for call in self.commands()))

    def interactive(self):
        master, slave = pty.openpty()
        self.addCleanup(os.close, master)
        process = subprocess.Popen(self.argv, stdin=slave, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, env=self.env)
        os.close(slave)
        self.addCleanup(lambda: process.poll() is None and process.kill())
        output = b''
        deadline = time.monotonic() + 10
        while b'Approve this plan?' not in output:
            self.assertGreater(deadline - time.monotonic(), 0, output.decode())
            ready, _, _ = select.select([process.stdout], [], [], 1)
            if ready:
                chunk = os.read(process.stdout.fileno(), 65536)
                self.assertTrue(chunk, output.decode())
                output += chunk
        return process, master, output

    def test_interactive_yes_launches_only_after_approval(self):
        process, terminal, preview = self.interactive()
        self.assert_preview_only()
        os.write(terminal, b'yes\n')
        tail, _ = process.communicate(timeout=10)
        self.assertEqual(process.returncode, 2, tail.decode())  # synthetic native exit 17
        self.assertIn('feature_extractor', [call[0] for call in self.commands()])
        self.assertTrue(self.workspace.is_dir())

    def test_changed_photo_refuses_before_output_creation(self):
        process, terminal, _ = self.interactive()
        (self.photos / 'photo.jpg').write_bytes(b'changed')
        os.write(terminal, b'yes\n')
        tail, _ = process.communicate(timeout=10)
        self.assertEqual(process.returncode, 2, tail.decode())
        self.assertIn(b'changed', tail)
        self.assertFalse(self.workspace.exists())
        self.assertNotIn('feature_extractor', [call[0] for call in self.commands()])

    def test_interactive_reject_cleans_preview_without_output(self):
        process, terminal, _ = self.interactive()
        self.assertTrue(list(self.query_tmp.iterdir()))
        os.write(terminal, b'no\n')
        tail, _ = process.communicate(timeout=10)
        self.assertEqual(process.returncode, 2, tail.decode())
        self.assertIn(b'plan rejected', tail)
        self.assert_preview_only()
        self.assertEqual(list(self.query_tmp.iterdir()), [])

    def test_interactive_eof_cancels_without_output(self):
        process, terminal, _ = self.interactive()
        os.write(terminal, b'\x04')
        tail, _ = process.communicate(timeout=10)
        self.assertEqual(process.returncode, 2, tail.decode())
        self.assertIn(b'EOF', tail)
        self.assert_preview_only()
        self.assertEqual(list(self.query_tmp.iterdir()), [])

    def test_signal_during_approval_cleans_without_output(self):
        process, terminal, _ = self.interactive()
        process.send_signal(signal.SIGTERM)
        tail, _ = process.communicate(timeout=10)
        self.assertEqual(process.returncode, 2, tail.decode())
        self.assertIn(b'cancelled', tail)
        self.assert_preview_only()
        self.assertEqual(list(self.query_tmp.iterdir()), [])

    def test_changed_executable_refuses_before_output(self):
        process, terminal, _ = self.interactive()
        with self.native.open('a') as stream:
            stream.write('\n# changed executable\n')
        os.write(terminal, b'yes\n')
        tail, _ = process.communicate(timeout=10)
        self.assertEqual(process.returncode, 2, tail.decode())
        self.assertIn(b'executable changed', tail)
        self.assert_preview_only()

    def test_added_photo_refuses_before_output(self):
        process, terminal, _ = self.interactive()
        (self.photos / 'new.jpg').write_bytes(b'new input')
        os.write(terminal, b'yes\n')
        tail, _ = process.communicate(timeout=10)
        self.assertEqual(process.returncode, 2, tail.decode())
        self.assertIn(b'inventory changed', tail)
        self.assert_preview_only()

    def test_yes_approves_but_does_not_replace_engineering_consent(self):
        self.argv.remove('--experimental-local')
        result = self.invoke('--yes')
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertIn('explicit --experimental-local consent required', result.stderr)
        self.assertFalse(self.workspace.exists())
        self.assertFalse(self.calls.exists())

    def test_admitted_extraction_preserves_selected_options(self):
        result = self.invoke('--yes', '--threads', '3', '--gpu-index', '2',
                             '--max-image-size', '1600', '--source-cap', '8',
                             '--num-samples', '7', '--random-seed', '42', '--tile', '32')
        self.assertEqual(result.returncode, 2)  # intentional native exit, not full qualification
        calls = self.commands()
        self.assertIn('feature_extractor', [call[0] for call in calls], result.stderr)
        extraction = next(call for call in calls if call[0] == 'feature_extractor')
        for option, expected in {'--image_path': str(self.photos),
                '--FeatureExtraction.max_image_size': '4004',
                '--FeatureExtraction.type': 'SIFT', '--FeatureExtraction.use_gpu': 'true',
                '--SiftExtraction.first_octave': '-1', '--FeatureExtraction.gpu_index': '2',
                '--FeatureExtraction.num_threads': '3'}.items():
            self.assertEqual(extraction[extraction.index(option) + 1], expected)
        plan = json.loads((self.workspace / 'planning/approved-plan.json').read_text())
        for key, expected in dict(threads=3, gpu_index=2, max_image_size=1600,
                source_cap=8, num_samples=7, random_seed=42, tile=32).items():
            self.assertEqual(plan['options'][key], expected)
        self.assertEqual(len([call for call in calls if call[0] == 'reconstruction_info']), 3)

    def test_approved_pipeline_commands_preserve_quality_and_dispatch_options(self):
        self.env['TEST_SFM_PROCEED'] = '1'
        result = self.invoke('--yes', '--threads', '3', '--gpu-index', '2',
                             '--max-image-size', '1600', '--source-cap', '8',
                             '--num-samples', '7', '--random-seed', '42', '--tile', '32',
                             '--no-lookahead')
        self.assertEqual(result.returncode, 2, result.stderr)  # synthetic dense exit 17
        calls = {call[0]: call for call in self.commands()}
        expected = {
            'exhaustive_matcher': {'--FeatureMatching.gpu_index': '2', '--FeatureMatching.num_threads': '3'},
            'mapper': {'--Mapper.num_threads': '3', '--Mapper.ba_gpu_index': '2'},
            'image_undistorter': {'--max_image_size': '1600', '--blank_pixels': '0',
                '--min_scale': '0.2', '--max_scale': '2', '--num_patch_match_src_images': '8',
                '--num_threads': '3'},
            'optimized_dense_plan': {'--source_cap': '8', '--num_samples': '7',
                '--gpu_index': '2', '--random_seed': '42', '--owned_tex_lookahead': 'false'},
            'experimental_fixed_cohort': {'--tile_threads': '32', '--output_policy': 'final_geo_only',
                '--auto_active_owners': '1', '--auto_phase_active_owners': '1',
                '--photo_source_storage': 'auto', '--PatchMatchStereo.max_image_size': '-1',
                '--PatchMatchStereo.num_samples': '7', '--PatchMatchStereo.random_seed': '42',
                '--PatchMatchStereo.gpu_index': '2', '--PatchMatchStereo.num_iterations': '5',
                '--PatchMatchStereo.window_radius': '5', '--PatchMatchStereo.window_step': '1'}}
        for command, options in expected.items():
            self.assertIn(command, calls, result.stderr)
            for name, value in options.items():
                self.assertEqual(calls[command][calls[command].index(name) + 1], value)
        plan = json.loads((self.workspace / 'planning/approved-plan.json').read_text())
        self.assertEqual(plan['effective_dispatch']['fusion'],
                         dict(backend='CPU', threads=1, use_mmap=True, use_cache=False))
        self.assertEqual(plan['resolution']['undistortion']['resolved_dimensions'], None)
        self.assertIn('Resolved dense dimensions', result.stderr)

    def test_input_change_during_reobservation_refuses_before_extraction(self):
        self.env['TEST_MUTATE_ON_REOBSERVE'] = str(self.photos / 'photo.jpg')
        result = self.invoke('--yes')
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertIn('photo inputs/inventory changed', result.stderr)
        self.assertNotIn('feature_extractor', [call[0] for call in self.commands()])

    def test_changed_device_refuses_instead_of_silently_switching_gpu(self):
        process, terminal, _ = self.interactive()
        self.device_state.write_text('GPU-replacement')
        os.write(terminal, b'yes\n')
        tail, _ = process.communicate(timeout=10)
        self.assertEqual(process.returncode, 2, tail.decode())
        self.assertIn(b'device identity changed', tail)
        self.assertNotIn('feature_extractor', [call[0] for call in self.commands()])

    def test_plan_only_without_cuda_build_is_honestly_unavailable(self):
        self.env['TEST_NO_CUDA'] = '1'
        result = self.invoke('--plan-only')
        self.assertEqual(result.returncode, 2)
        self.assertIn('plan unavailable', result.stderr)
        self.assertIn('CPU fallback unsupported', result.stderr)
        self.assertNotIn('Effective plan', result.stdout)
        self.assert_preview_only()
        self.assertEqual(list(self.query_tmp.iterdir()), [])

    def test_plan_only_failed_cuda_query_preserves_actionable_diagnostic(self):
        self.env['TEST_QUERY_FAIL'] = '1'
        result = self.invoke('--plan-only')
        self.assertEqual(result.returncode, 2)
        self.assertIn('CUDA device unavailable', result.stderr)
        self.assertIn('plan unavailable', result.stderr)
        self.assertNotIn('Effective plan', result.stdout)
        self.assert_preview_only()
        self.assertEqual(list(self.query_tmp.iterdir()), [])

    def test_signal_during_native_query_drains_child_and_cleans_preview(self):
        fifo = self.root / 'query-started'
        os.mkfifo(fifo)
        descriptor = os.open(fifo, os.O_RDWR | os.O_NONBLOCK)
        self.addCleanup(os.close, descriptor)
        self.env['TEST_BLOCK_QUERY'] = str(fifo)
        with tempfile.TemporaryFile() as output:
            process = subprocess.Popen(self.argv + ['--plan-only'], stdout=output,
                                       stderr=output, env=self.env)
            self.addCleanup(lambda: process.poll() is None and process.kill())
            ready, _, _ = select.select([descriptor], [], [], 10)
            self.assertTrue(ready)
            child_pid = int(os.read(descriptor, 100))
            process.send_signal(signal.SIGINT)
            process.wait(timeout=10)
            self.assertEqual(process.returncode, 2)
            with self.assertRaises(ProcessLookupError):
                os.kill(child_pid, 0)
        self.assert_preview_only()
        self.assertEqual(list(self.query_tmp.iterdir()), [])

    def test_small_free_fluctuation_does_not_request_reapproval(self):
        process, terminal, _ = self.interactive()
        (self.root / 'free').write_text(str(16 * 1024**3 - 1))
        os.write(terminal, b'yes\n')
        tail, _ = process.communicate(timeout=10)
        self.assertIn('feature_extractor', [call[0] for call in self.commands()])
        self.assertNotIn(b'Approve this plan?', tail)

    def test_capacity_loss_refuses_without_quality_fallback(self):
        process, terminal, _ = self.interactive()
        (self.root / 'free').write_text('1')
        os.write(terminal, b'yes\n')
        tail, _ = process.communicate(timeout=10)
        self.assertEqual(process.returncode, 2)
        self.assertIn(b'device reserve unavailable', tail)
        self.assertNotIn('feature_extractor', [call[0] for call in self.commands()])

    def test_approval_wait_respects_invocation_deadline(self):
        self.argv += ['--timeout-seconds', '1.5']
        process, terminal, _ = self.interactive()
        tail, _ = process.communicate(timeout=5)
        self.assertEqual(process.returncode, 2)
        self.assertIn(b'deadline expired', tail)
        self.assert_preview_only()
        self.assertEqual(list(self.query_tmp.iterdir()), [])

    def test_workspace_inside_photos_refuses_without_input_mutation(self):
        self.argv[self.argv.index('--workspace') + 1] = str(self.photos / 'output')
        result = self.invoke('--plan-only')
        self.assertEqual(result.returncode, 2)
        self.assertIn('outside photos', result.stderr)
        self.assertEqual([path.name for path in self.photos.iterdir()], ['photo.jpg'])
        self.assertFalse(self.calls.exists())

    def test_query_tmpdir_inside_photos_refuses_without_input_mutation(self):
        self.env['TMPDIR'] = str(self.photos)
        result = self.invoke('--plan-only')
        self.assertEqual(result.returncode, 2)
        self.assertIn('TMPDIR', result.stderr)
        self.assertEqual([path.name for path in self.photos.iterdir()], ['photo.jpg'])
        self.assertFalse(self.calls.exists())

    def test_noninteractive_requires_yes_even_with_piped_approval(self):
        result = self.invoke(stdin='yes\n')
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertIn('--yes', result.stderr)
        self.assert_preview_only()

    def test_plan_only_displays_honest_plan_without_output(self):
        result = self.invoke('--plan-only')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('Effective plan', result.stdout)
        self.assertIn('UNKNOWN', result.stdout)
        self.assertIn('4004', result.stdout)
        self.assertIn(str(self.workspace), result.stdout)
        self.assert_preview_only()


if __name__ == '__main__':
    unittest.main()
