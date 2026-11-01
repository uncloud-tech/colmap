# SPDX-License-Identifier: BSD-3-Clause
"""Installed frontend CPU mapping handoff; no self-subprocess dispatch."""
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

from colmap_reconstruction.cli import Reconstruction
from colmap_reconstruction.process import NativeFailure


class MappingAdmissionTest(unittest.TestCase):
    def test_mapper_uses_host_policy_and_restores_monitor_after_success_or_failure(self):
        for previous, fail in ((None, True), (12345, True), (None, False), (12345, False)):
            with self.subTest(previous=previous, fail=fail), tempfile.TemporaryDirectory() as directory:
                reconstruction = Reconstruction.__new__(Reconstruction)
                reconstruction.workspace = Path(directory)
                reconstruction.process = SimpleNamespace(host_budget_bytes=previous)
                reconstruction.args = SimpleNamespace(photos=Path(directory), gpu_index='0', threads=2,
                    loader_scratch_host_bytes=10, process_host_bytes=20, unknown_host_bytes=30,
                    reserve_host_bytes=40, process_device_bytes=50, unknown_device_bytes=60,
                    reserve_device_bytes=70, sparse_model=None)
                capacities = iter((dict(host=1000, device=2000), dict(host=1100, device=2100),
                                   dict(host=1200, device=2200)))
                reconstruction.resources = lambda: next(capacities)
                calls = []

                def native(command, *arguments, **kwargs):
                    if command == 'mapper':
                        calls.append((dict(zip(arguments[::2], arguments[1::2])),
                                      reconstruction.process.host_budget_bytes))
                        if fail:
                            raise NativeFailure('mapping failure')

                reconstruction.native = native
                with self.assertRaisesRegex(NativeFailure if fail else ValueError,
                                            'mapping failure' if fail else 'select --sparse-model'):
                    reconstruction.prepare()
                self.assertEqual(reconstruction.process.host_budget_bytes, previous)
                options, monitor = calls[0]
                self.assertEqual(monitor, 1200)
                self.assertEqual(options['--Mapper.host_budget_bytes'], 1200)
                self.assertEqual(options['--Mapper.policy_host_bytes'], 60)
                self.assertEqual(options['--Mapper.reserve_host_bytes'], 40)
                self.assertEqual(options['--Mapper.num_threads'], 2)
                self.assertEqual(options['--Mapper.ba_gpu_index'], '0')
                self.assertNotIn('--Mapper.ba_use_gpu', options)
                self.assertNotIn('--Mapper.random_seed', options)
                self.assertNotIn('--Mapper.multiple_models', options)
