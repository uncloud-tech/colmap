# SPDX-License-Identifier: BSD-3-Clause
"""CPU placement/report binding, not native or GPU execution proof."""

import copy
import unittest


class MultiGpuBindingTest(unittest.TestCase):
    def fixture(self):
        devices = [
            dict(ordinal=0, device_uuid="a" * 32),
            dict(ordinal=2, device_uuid="b" * 32),
        ]
        jobs = [
            dict(
                reference=3,
                sources=[1, 2, 1],
                options=dict(gpu_index="0", random_seed=7),
            ),
            dict(
                reference=1,
                sources=[3],
                options=dict(gpu_index="0", random_seed=7),
            ),
        ]
        graph = dict(
            phases={p: dict(jobs=copy.deepcopy(jobs)) for p in ("photo", "geo")}
        )
        scientific = [
            dict(
                reference=j["reference"],
                photo_sources=j["sources"],
                geo_sources=j["sources"],
                photo_options=j["options"],
                geo_options=j["options"],
            )
            for j in jobs
        ]
        plan = dict(
            schema_version=1,
            scientific_inputs=scientific,
            phases=[
                dict(
                    phase=p,
                    groups=[
                        dict(
                            ordinal=d["ordinal"],
                            uuid=d["device_uuid"],
                            cohorts=[dict(jobs=[i], active_owners=1)],
                        )
                        for i, d in enumerate(devices)
                    ],
                )
                for p in ("photo", "geo")
            ],
        )
        return devices, graph, plan

    def test_assignment_binds_only_placement_preserving_scientific_duplicates(
        self,
    ):
        from colmap_reconstruction.cli import bind_multi_gpu_plan

        devices, graph, plan = self.fixture()
        original = copy.deepcopy(graph)
        self.assertEqual(
            bind_multi_gpu_plan(plan, graph, devices),
            {"photo": {0: "0", 1: "2"}, "geo": {0: "0", 1: "2"}},
        )
        self.assertEqual(graph, original)

    def test_changed_readsets_uuid_or_duplicate_writer_refuse(self):
        from colmap_reconstruction.cli import bind_multi_gpu_plan

        for change in (
            "source_order",
            "uuid",
            "duplicate",
            "missing",
            "seed",
            "barrier",
        ):
            with self.subTest(change=change):
                devices, graph, plan = self.fixture()
                if change == "source_order":
                    plan["scientific_inputs"][0]["photo_sources"] = [1, 1, 2]
                elif change == "uuid":
                    plan["phases"][0]["groups"][1]["uuid"] = "c" * 32
                elif change == "duplicate":
                    plan["phases"][0]["groups"][1]["cohorts"][0]["jobs"] = [0]
                elif change == "missing":
                    plan["phases"][1]["groups"][1]["cohorts"] = []
                elif change == "seed":
                    plan["scientific_inputs"][0]["geo_options"] = dict(
                        gpu_index="0", random_seed=8
                    )
                elif change == "barrier":
                    plan["phases"].reverse()
                with self.assertRaises(ValueError):
                    bind_multi_gpu_plan(plan, graph, devices)

    def test_cohort_turnover_preserves_group_input_order(self):
        from colmap_reconstruction.cli import bind_multi_gpu_plan

        devices, graph, plan = self.fixture()
        for phase in graph["phases"].values():
            phase["jobs"] += copy.deepcopy(phase["jobs"])
        plan["scientific_inputs"] += copy.deepcopy(plan["scientific_inputs"])
        for phase in plan["phases"]:
            for index, group in enumerate(phase["groups"]):
                group["cohorts"].append(dict(jobs=[index + 2], active_owners=1))
        original = copy.deepcopy(graph)
        assignments = bind_multi_gpu_plan(plan, graph, devices)
        self.assertEqual(assignments["photo"], {0: "0", 2: "0", 1: "2", 3: "2"})
        self.assertEqual(graph, original)
        # Individually ordered cohorts are insufficient: a device must not run
        # later input jobs before its earlier cohort's jobs.
        for phase_index in (0, 1):
            changed = copy.deepcopy(plan)
            changed["phases"][phase_index]["groups"][0]["cohorts"].reverse()
            with self.subTest(phase=phase_index), self.assertRaises(ValueError):
                bind_multi_gpu_plan(changed, graph, devices)

    def test_dense_arguments_bind_approved_devices_and_frozen_host_cap(self):
        import sys
        import tempfile
        from pathlib import Path

        from colmap_reconstruction.cli import Reconstruction, parse_args

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = parse_args(["--photos", str(root), "--workspace", str(root)])
            reconstruction = Reconstruction(args, sys.executable)
            for name in (
                "loader_scratch_host",
                "process_host",
                "process_device",
                "unknown_host",
                "unknown_device",
                "reserve_host",
                "reserve_device",
            ):
                setattr(args, name + "_bytes", 0)
            args.unknown_provenance = "test-only"
            reconstruction.graph = {
                "phases": {"geo": {"requested_options": {"gpu_index": "0"}}}
            }
            reconstruction.stereo_devices = [
                dict(
                    ordinal=0,
                    device_uuid="a" * 32,
                    available_device_bytes="1024",
                ),
                dict(
                    ordinal=2,
                    device_uuid="b" * 32,
                    available_device_bytes="2048",
                ),
            ]
            reconstruction.dense_host_capacity = 4096
            argv = reconstruction.dense_arguments()
            self.assertIn("--stereo_devices", argv)
            self.assertEqual(
                argv[argv.index("--stereo_devices") + 1],
                "0:" + "a" * 32 + ":1024,2:" + "b" * 32 + ":2048",
            )
            self.assertEqual(
                argv[argv.index("--host_capacity_bytes") + 1], "4096"
            )

    def test_single_device_arguments_request_native_selection_and_frozen_disk(
        self,
    ):
        import sys
        import tempfile
        from pathlib import Path

        from colmap_reconstruction.cli import Reconstruction, parse_args

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = parse_args(["--photos", str(root), "--workspace", str(root)])
            r = Reconstruction(args, sys.executable)
            for name in (
                "loader_scratch_host",
                "process_host",
                "process_device",
                "unknown_host",
                "unknown_device",
                "reserve_host",
                "reserve_device",
            ):
                setattr(args, name + "_bytes", 0)
            args.unknown_provenance = "fixture"
            r.graph = {
                "phases": {"geo": {"requested_options": {"gpu_index": "0"}}}
            }
            r.stereo_devices = [
                dict(
                    ordinal=0,
                    device_uuid="a" * 32,
                    available_device_bytes="1024",
                )
            ]
            r.dense_host_capacity = 4096
            r.dense_device_capacity = 1000
            r.dense_disk_capacity = 8192
            argv = r.dense_arguments()
            self.assertEqual(
                argv[argv.index("--automatic_stereo_storage") + 1], "1"
            )
            self.assertEqual(
                argv[argv.index("--stereo_devices") + 1],
                "0:" + "a" * 32 + ":1000",
            )
            self.assertEqual(
                argv[argv.index("--stereo_disk_capacity_bytes") + 1], "8192"
            )
            self.assertEqual(
                argv[argv.index("--host_capacity_bytes") + 1], "4096"
            )

    def test_single_device_binding_rejects_changed_contract(
        self,
    ):
        from colmap_reconstruction.cli import bind_single_device_plan

        for kind in ("resident_snapshot", "single_device_host_exchange"):
            devices, graph, plan = self.fixture()
            devices = devices[:1]
            for phase in plan["phases"]:
                phase["groups"] = [
                    dict(
                        ordinal=0,
                        uuid="a" * 32,
                        capacity_bytes="1000",
                        cohorts=[dict(jobs=[0, 1], active_owners=1)],
                    )
                ]
            plan["execution_kind"] = kind
            plan["allocation_fingerprint"] = "fingerprint"
            key = (
                "snapshot"
                if kind == "resident_snapshot"
                else "single_device_host"
            )
            plan[key] = dict(
                effective_photo_source_storage="private",
                phases=[
                    dict(
                        phase=phase,
                        records=[
                            dict(
                                reference=j["reference"],
                                sources=j["sources"],
                                volume_mode="resident_private",
                                columns="0",
                                owned_cuda_graphs=False,
                                owned_async_pool=False,
                            )
                            for j in graph["phases"][phase]["jobs"]
                        ],
                    )
                    for phase in ("photo", "geo")
                ],
            )
            original = copy.deepcopy(plan)
            self.assertEqual(
                bind_single_device_plan(
                    plan, graph, devices, "fingerprint", "1000"
                )["geo"],
                {0: "0", 1: "0"},
            )
            for change in (
                "mode",
                "policy",
                "sources",
                "capacity",
                "device",
                "fingerprint",
            ):
                changed = copy.deepcopy(original)
                record = changed[key]["phases"][1]["records"][0]
                if change == "mode":
                    record["volume_mode"] = "unknown"
                elif change == "policy":
                    record["owned_async_pool"] = True
                elif change == "sources":
                    record["sources"] = [1, 1, 2]
                elif change == "capacity":
                    changed["phases"][0]["groups"][0]["capacity_bytes"] = "1001"
                elif change == "device":
                    changed["phases"][0]["groups"][0]["uuid"] = "b" * 32
                else:
                    changed["allocation_fingerprint"] = "changed"
                with (
                    self.subTest(kind=kind, change=change),
                    self.assertRaises(ValueError),
                ):
                    bind_single_device_plan(
                        changed, graph, devices, "fingerprint", "1000"
                    )

    def test_single_device_stage_prepares_and_binds_before_execution(self):
        import json
        import sys
        import tempfile
        from pathlib import Path
        from unittest.mock import patch

        from colmap_reconstruction.cli import Reconstruction, parse_args
        from colmap_reconstruction.process import NativeFailure

        devices, graph, plan = self.fixture()
        devices = devices[:1]
        for phase in plan["phases"]:
            phase["groups"] = [
                dict(
                    ordinal=0,
                    uuid="a" * 32,
                    capacity_bytes="1000",
                    cohorts=[dict(jobs=[0, 1], active_owners=1)],
                )
            ]
        plan["execution_kind"] = "single_device_host_exchange"
        plan["allocation_fingerprint"] = "fp"
        plan["single_device_host"] = dict(
            effective_photo_source_storage="private",
            phases=[
                dict(
                    phase=p,
                    records=[
                        dict(
                            reference=j["reference"],
                            sources=j["sources"],
                            volume_mode="resident_private",
                            columns="0",
                            owned_cuda_graphs=False,
                            owned_async_pool=False,
                        )
                        for j in graph["phases"][p]["jobs"]
                    ],
                )
                for p in ("photo", "geo")
            ],
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = parse_args(
                ["--photos", str(root), "--workspace", str(root), "--yes"]
            )
            for name in (
                "loader_scratch_host",
                "process_host",
                "process_device",
                "unknown_host",
                "unknown_device",
                "reserve_host",
                "reserve_device",
            ):
                setattr(args, name + "_bytes", 0)
            args.unknown_provenance = "fixture"
            r = Reconstruction(args, sys.executable)
            r.assets.mkdir()
            r.dense.mkdir()
            r.graph = graph
            graph["phases"]["geo"]["requested_options"] = {"gpu_index": "0"}
            r.identity = dict(allocation_fingerprint="fp")
            r.stereo_devices = [dict(devices[0], available_device_bytes="1000")]
            r.input_files = {}
            calls = []

            def native(command, *argv, capture=False):
                calls.append(argv)
                if "--prepare_only" in argv:
                    Path(argv[argv.index("--report_path") + 1]).write_text(
                        json.dumps(plan)
                    )
                else:
                    self.assertIn("--approved_multi_gpu_plan", argv)
                    raise NativeFailure("bounded execution failed")

            with (
                patch.object(
                    r,
                    "resources",
                    return_value={"host": 100000, "device": 1000},
                ),
                patch.object(r, "native", side_effect=native),
            ):
                with self.assertRaisesRegex(
                    NativeFailure, "bounded execution failed"
                ):
                    r.dense_stage()
            self.assertEqual(len(calls), 2)
            self.assertFalse((r.assets / "cohort.json").exists())

    def test_child_failure_after_bound_plan_never_publishes_maps(self):
        import json
        import sys
        import tempfile
        from pathlib import Path
        from unittest.mock import patch

        from colmap_reconstruction.cli import Reconstruction, parse_args
        from colmap_reconstruction.process import NativeFailure

        devices, graph, plan = self.fixture()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = parse_args(
                ["--photos", str(root), "--workspace", str(root), "--yes"]
            )
            for name in (
                "loader_scratch_host",
                "process_host",
                "process_device",
                "unknown_host",
                "unknown_device",
                "reserve_host",
                "reserve_device",
            ):
                setattr(args, name + "_bytes", 0)
            args.unknown_provenance = "fixture"
            reconstruction = Reconstruction(args, sys.executable)
            reconstruction.assets.mkdir()
            reconstruction.graph = graph
            graph["phases"]["geo"]["requested_options"] = {"gpu_index": "0"}
            reconstruction.stereo_devices = [
                dict(d, available_device_bytes="100000") for d in devices
            ]
            reconstruction.input_files = {}
            calls = []

            def native(command, *argv, capture=False):
                calls.append((command, argv))
                if command == "experimental_fixed_cohort":
                    if "--prepare_only" in argv:
                        Path(argv[argv.index("--report_path") + 1]).write_text(
                            json.dumps(plan)
                        )
                    else:
                        self.assertIn("--approved_multi_gpu_plan", argv)
                        raise NativeFailure("device child failed")

            with (
                patch.object(
                    reconstruction, "resources", return_value={"host": 100000}
                ),
                patch.object(reconstruction, "native", side_effect=native),
            ):
                with self.assertRaisesRegex(
                    NativeFailure, "device child failed"
                ):
                    reconstruction.dense_stage()
            self.assertEqual(len(calls), 2)
            self.assertFalse(reconstruction.dense.exists())

    def test_completed_report_binds_execution_options_before_publication(self):
        import contextlib
        import io
        import json
        import struct
        import sys
        import tempfile
        from pathlib import Path
        from unittest.mock import patch

        from colmap_reconstruction.cli import Reconstruction, parse_args

        for failure in ("none", "seed", "late_map"):
            changed_seed = failure == "seed"
            with (
                self.subTest(failure=failure),
                tempfile.TemporaryDirectory() as directory,
            ):
                devices, graph, plan = self.fixture()
                # Complete two-producer graph for the installed fixture.
                for phase in ("photo", "geo"):
                    graph["phases"][phase]["jobs"][0]["sources"] = [1, 1]
                plan["scientific_inputs"][0]["photo_sources"] = [1, 1]
                plan["scientific_inputs"][0]["geo_sources"] = [1, 1]
                root = Path(directory)
                args = parse_args(
                    ["--photos", str(root), "--workspace", str(root), "--yes"]
                )
                for name in (
                    "loader_scratch_host",
                    "process_host",
                    "process_device",
                    "unknown_host",
                    "unknown_device",
                    "reserve_host",
                    "reserve_device",
                ):
                    setattr(args, name + "_bytes", 0)
                args.unknown_provenance = "fixture"
                reconstruction = Reconstruction(args, sys.executable)
                reconstruction.assets.mkdir()
                reconstruction.graph = graph
                graph["phases"]["geo"]["requested_options"] = {"gpu_index": "0"}
                reconstruction.stereo_devices = [
                    dict(d, available_device_bytes="100000") for d in devices
                ]
                reconstruction.input_files = {}
                reconstruction.identity = {"allocation_fingerprint": "fixture"}
                reconstruction.images = {
                    3: dict(name="a.jpg", width=2, height=1),
                    1: dict(name="b.jpg", width=2, height=1),
                }
                (reconstruction.dense / "stereo").mkdir(parents=True)
                (reconstruction.dense / "stereo/fusion.cfg").write_text(
                    "a.jpg\nb.jpg\n"
                )

                def native(
                    command,
                    *argv,
                    capture=False,
                    plan=plan,
                    graph=graph,
                    devices=devices,
                    changed_seed=changed_seed,
                    failure=failure,
                    reconstruction=reconstruction,
                    root=root,
                ):
                    self.assertEqual(command, "experimental_fixed_cohort")
                    if "--prepare_only" in argv:
                        Path(argv[argv.index("--report_path") + 1]).write_text(
                            json.dumps(plan)
                        )
                        return
                    executed = {}
                    for phase in ("photo", "geo"):
                        executed[phase] = []
                        for i, entry in enumerate(
                            graph["phases"][phase]["jobs"]
                        ):
                            options = dict(
                                entry["options"],
                                gpu_index=str(devices[i]["ordinal"]),
                            )
                            if changed_seed and i == 1 and phase == "geo":
                                options["random_seed"] += 1
                            executed[phase].append(dict(entry, options=options))
                    (reconstruction.assets / "cohort.json").write_text(
                        json.dumps(
                            dict(
                                status="completed",
                                effective_backend="experimental_fixed_cohort",
                                allocation_fingerprint="fixture",
                                multi_gpu_plan=plan,
                                phases=executed,
                            )
                        )
                    )
                    for image in reconstruction.images.values():
                        for kind, channels in (
                            ("depth_maps", 1),
                            ("normal_maps", 3),
                        ):
                            path = (
                                root
                                / "cohort/stereo"
                                / kind
                                / (image["name"] + ".geometric.bin")
                            )
                            path.parent.mkdir(parents=True, exist_ok=True)
                            path.write_bytes(
                                f"2&1&{channels}&".encode()
                                + struct.pack(
                                    "<" + "f" * (2 * channels),
                                    *(
                                        [
                                            float("nan")
                                            if failure == "late_map"
                                            and image["name"] == "b.jpg"
                                            and channels == 3
                                            else 1.0
                                        ]
                                        * (2 * channels)
                                    ),
                                )
                            )

                with (
                    patch.object(
                        reconstruction,
                        "resources",
                        return_value={"host": 100000},
                    ),
                    patch.object(reconstruction, "native", side_effect=native),
                    contextlib.redirect_stdout(io.StringIO()),
                ):
                    if failure != "none":
                        with self.assertRaisesRegex(
                            ValueError,
                            "effective geo plan changed"
                            if changed_seed
                            else "nonfinite",
                        ):
                            reconstruction.dense_stage()
                    else:
                        reconstruction.dense_stage()
                for image in reconstruction.images.values():
                    for kind in ("depth_maps", "normal_maps"):
                        target = (
                            reconstruction.dense
                            / "stereo"
                            / kind
                            / (image["name"] + ".geometric.bin")
                        )
                        self.assertEqual(target.exists(), failure == "none")
                        if failure == "none":
                            source = root / "cohort/stereo" / kind / target.name
                            self.assertEqual(
                                target.read_bytes(), source.read_bytes()
                            )
                            self.assertEqual(
                                target.stat().st_ino, source.stat().st_ino
                            )


if __name__ == "__main__":
    unittest.main()
