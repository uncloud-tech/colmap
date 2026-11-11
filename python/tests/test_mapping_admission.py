# SPDX-License-Identifier: BSD-3-Clause
"""Frontend Mapper forwarding/accounting seam; NOT native/GPU qualification."""
import contextlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

from colmap_reconstruction.cli import Reconstruction, parse_args
from colmap_reconstruction.process import NativeFailure


class MappingAdmissionTest(unittest.TestCase):
    def mapper_handoff(self, ceiling, observed, previous=None, failure=None):
        with tempfile.TemporaryDirectory() as directory:
            reconstruction = Reconstruction.__new__(Reconstruction)
            reconstruction.workspace = Path(directory)
            reconstruction.assets = Path(directory)
            reconstruction.identity = dict(allocation_fingerprint='test', source_commit='test')
            reconstruction.process = SimpleNamespace(host_budget_bytes=previous,
                                                     remaining=lambda: 60, sequence=0)
            reconstruction.args = SimpleNamespace(photos=Path(directory), gpu_index='0', threads=2,
                host_budget_bytes=ceiling, device_budget_bytes=None, random_seed=7,
                loader_scratch_host_bytes=10, process_host_bytes=20, unknown_host_bytes=30,
                reserve_host_bytes=40, process_device_bytes=50, unknown_device_bytes=60,
                reserve_device_bytes=70, sparse_model=None)
            reconstruction.extraction_limit = 100
            reconstruction.extraction_image_list = []
            # Extraction mode selection is independent of this Mapper handoff.
            reconstruction.select_extraction = lambda capacity: dict(mode='resident')
            calls = []

            def native(command, *arguments, **kwargs):
                reconstruction.process.sequence += 1
                if command == 'reconstruction_info':
                    return json.dumps(dict(reconstruction.identity,
                        available_host_bytes=str(observed), available_device_bytes='2200'))
                if command == 'mapper':
                    calls.append((dict(zip(arguments[::2], arguments[1::2])),
                                  reconstruction.process.host_budget_bytes))
                    if failure is not None:
                        raise failure

            reconstruction.native = native
            expected = failure if failure is not None else ValueError('select --sparse-model')
            with self.assertRaisesRegex(type(expected), str(expected)):
                reconstruction.prepare()
            self.assertEqual(reconstruction.process.host_budget_bytes, previous)
            options, monitor = calls[0]
            self.assertEqual(options['--Mapper.policy_host_bytes'], 60)
            self.assertEqual(options['--Mapper.reserve_host_bytes'], 40)
            self.assertEqual(options['--Mapper.num_threads'], 2)
            self.assertEqual(options['--Mapper.ba_gpu_index'], '0')
            self.assertEqual(options['--Mapper.random_seed'], 7)
            self.assertNotIn('--Mapper.ba_use_gpu', options)
            self.assertNotIn('--Mapper.multiple_models', options)
            return options, monitor

    def test_explicit_mapper_ceiling_is_stable_across_smaller_observed_capacities(self):
        first = self.mapper_handoff(2000, 1200)
        second = self.mapper_handoff(2000, 1500)
        self.assertEqual([first[1], second[1]], [1200, 1500])
        self.assertEqual([first[0]['--Mapper.host_budget_bytes'],
                          second[0]['--Mapper.host_budget_bytes']], [2000, 2000])

    def test_explicit_ceiling_below_observed_capacity_bounds_accounting(self):
        options, monitor = self.mapper_handoff(1000, 1500)
        self.assertEqual(options['--Mapper.host_budget_bytes'], 1000)
        self.assertEqual(monitor, 1000)

    def test_absent_ceiling_keeps_dynamic_mapper_option_and_accounting(self):
        for observed in (1200, 1500):
            with self.subTest(observed=observed):
                options, monitor = self.mapper_handoff(None, observed)
                self.assertEqual(options['--Mapper.host_budget_bytes'], observed)
                self.assertEqual(monitor, observed)

    def test_monitor_restored_after_mapper_success_failure_or_cancellation(self):
        for ceiling in (None, 1000, 2000):
            for previous in (None, 12345):
                for failure in (None, NativeFailure('mapping failure'),
                                NativeFailure('reconstruction cancelled'), KeyboardInterrupt()):
                    with self.subTest(ceiling=ceiling, previous=previous, failure=repr(failure)):
                        self.mapper_handoff(ceiling, 1200, previous, failure)

    def test_cli_rejects_zero_and_invalid_explicit_ceiling(self):
        for value in ('0', '-1', 'invalid', '1.5', str(2**64)):
            with self.subTest(value=value), contextlib.redirect_stderr(io.StringIO()) as error:
                with self.assertRaises(SystemExit) as refused:
                    parse_args(['--photos', '/unused/photos', '--workspace', '/unused/workspace',
                                '--host-budget-bytes', value])
                self.assertEqual(refused.exception.code, 2)
                self.assertIn('--host-budget-bytes', error.getvalue())
