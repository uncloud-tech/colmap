# SPDX-License-Identifier: BSD-3-Clause
"""Installed frontend matching handoff and phase monitor lifetime."""
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

from colmap_reconstruction.cli import Reconstruction
from colmap_reconstruction.process import NativeFailure


class MatchingAdmissionTest(unittest.TestCase):
    def test_matching_budget_and_monitor_restored_on_success_and_failure(self):
        for previous, fail_matching in ((None, True), (12345, True), (None, False), (12345, False)):
            with self.subTest(previous=previous, fail_matching=fail_matching), tempfile.TemporaryDirectory() as directory:
                reconstruction = Reconstruction.__new__(Reconstruction)
                reconstruction.workspace = Path(directory)
                reconstruction.process = SimpleNamespace(host_budget_bytes=previous)
                reconstruction.args = SimpleNamespace(photos=Path(directory), gpu_index='0', threads=2,
                    loader_scratch_host_bytes=10, process_host_bytes=20, unknown_host_bytes=30,
                    reserve_host_bytes=40, process_device_bytes=50, unknown_device_bytes=60,
                    reserve_device_bytes=70)
                capacities = iter((dict(host=1000, device=2000), dict(host=1100, device=2100),
                                   dict(host=1200, device=2200)))
                reconstruction.resources = lambda: next(capacities)
                calls = []

                def native(command, *arguments, **kwargs):
                    if command == 'exhaustive_matcher':
                        calls.append((dict(zip(arguments[::2], arguments[1::2])),
                                      reconstruction.process.host_budget_bytes))
                        if fail_matching:
                            raise NativeFailure('matching failure')
                    elif command == 'mapper':
                        self.assertEqual(reconstruction.process.host_budget_bytes, previous)
                        raise NativeFailure('mapper sentinel')

                reconstruction.native = native
                with self.assertRaisesRegex(NativeFailure, 'matching failure' if fail_matching else 'mapper sentinel'):
                    reconstruction.prepare()
                self.assertEqual(reconstruction.process.host_budget_bytes, previous)
                self.assertEqual(len(calls), 1)
                options, monitor = calls[0]
                self.assertEqual(monitor, 1100)
                self.assertEqual(options['--FeatureMatching.host_budget_bytes'], 1100)
                self.assertEqual(options['--FeatureMatching.policy_host_bytes'], 60)
                self.assertEqual(options['--FeatureMatching.reserve_host_bytes'], 40)
                self.assertEqual(options['--FeatureMatching.device_budget_bytes'], 2100)
                self.assertEqual(options['--FeatureMatching.policy_device_bytes'], 110)
                self.assertEqual(options['--FeatureMatching.reserve_device_bytes'], 70)
                self.assertNotIn('--FeatureMatching.use_gpu', options)
                self.assertNotIn('--SiftMatching.max_ratio', options)
