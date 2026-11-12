# SPDX-License-Identifier: BSD-3-Clause
"""Public guard seams; synthetic bytes are not a reconstruction golden."""

import contextlib
import copy
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from colmap_reconstruction.byte_golden import (
    capture,
    compare,
    inventory,
    load_manifest,
    main,
)

HASH = "a" * 64


def wire(workspace):
    fields = """build_cuda_enabled 1
build_hip_enabled 0
opt_int max_image_size -1
opt_string gpu_index 1 30
opt_double depth_min 3ff0000000000000
opt_double depth_max 4000000000000000
opt_int random_seed 0
opt_int window_radius 5
opt_int window_step 1
opt_double sigma_spatial 3ff0000000000000
opt_double sigma_color 3ff0000000000000
opt_int num_samples 15
opt_bool owned_distinct_source_cache 1
opt_bool owned_tex_lookahead 1
opt_double ncc_sigma 3ff0000000000000
opt_double min_triangulation_angle 3ff0000000000000
opt_double incident_angle_sigma 3ff0000000000000
opt_int num_iterations 5
opt_bool geom_consistency 0
opt_double geom_consistency_regularizer 3ff0000000000000
opt_double geom_consistency_max_cost 3ff0000000000000
opt_bool filter 0
opt_double filter_min_ncc 3ff0000000000000
opt_double filter_min_triangulation_angle 3ff0000000000000
opt_int filter_min_num_consistent 2
opt_double filter_geom_consistency_max_cost 3ff0000000000000
opt_bool write_consistency_graph 0
opt_bool allow_missing_files 0
opt_int num_threads 1
opt_double cache_size 3ff0000000000000
"""
    for name, path in (
        ("workspace", workspace),
        ("config", workspace / "stereo/patch-match.cfg"),
    ):
        encoded = str(path).encode("utf-8")
        fields += f"opt_string {name} {len(encoded)} {encoded.hex()}\n"
    return "COLMAP_OPTIMIZED_PHOTO_OPTIONS_V1\n" + fields


def fixture(root, image_names=("0.jpg", "1.jpg")):
    def write(name, value):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(value) if not isinstance(value, str) else value
        )

    options = dict(
        random_seed=0,
        threads=1,
        backend="experimental_fixed_cohort",
        gpu_index=0,
        output_policy="photo_and_geo",
        workspace=str(root),
    )
    images = [
        dict(index=i, name=name, width=4, height=3)
        for i, name in enumerate(image_names)
    ]
    phases = {
        p: dict(
            requested_options=dict(random_seed=0),
            jobs=[
                dict(
                    reference=i,
                    sources=[j for j in range(len(images)) if j != i]
                    * (2 if len(images) == 2 else 1),
                    options=dict(random_seed=0, num_threads=1, gpu_index="0"),
                )
                for i in range(len(images))
            ],
        )
        for p in ("photo", "geo")
    }
    write("planning/intent.json", options)
    write(
        "planning/approved-plan.json",
        dict(
            options=options,
            input_binding={
                name: dict(target="/private/" + name, sha256=HASH)
                for name in image_names
            },
        ),
    )
    write(
        "planning/extraction-inventory.json",
        dict(effective_order=list(image_names)),
    )
    write(
        "planning/resolution-intent.json",
        dict(
            original=dict(
                photos=[
                    dict(path=name, width=4, height=3, orientation=None)
                    for name in image_names
                ]
            )
        ),
    )
    write(
        "planning/build.json",
        dict(executable_sha256=HASH, identity=dict(source_commit="b" * 40)),
    )
    native_phases = copy.deepcopy(phases)
    for phase, data in native_phases.items():
        for job in data["jobs"]:
            name = images[job["reference"]]["name"]
            suffix = "photometric" if phase == "photo" else "geometric"
            job.update(
                depth_min=1.0,
                depth_max=2.0,
                effective_options_wire_v1=wire(root / "dense"),
                outputs={
                    key: str(root / f"dense/stereo/{kind}/{name}.{suffix}.bin")
                    for key, kind in (
                        ("depth", "depth_maps"),
                        ("normal", "normal_maps"),
                    )
                },
            )
    write(
        "planning/dense-plan.json",
        dict(
            images=[
                dict(image, path=str(root / "dense/images" / image["name"]))
                for image in images
            ],
            phases=native_phases,
            workspace_path=str(root / "dense"),
            schema_version=1,
        ),
    )
    write(
        "planning/cohort.json",
        dict(
            status="completed",
            effective_backend=options["backend"],
            output_policy="photo_and_geo",
            phases={p: phases[p]["jobs"] for p in phases},
        ),
    )
    write(
        "planning/sparse-model.json",
        dict(name="0", counts=dict(num_registered_images=len(images))),
    )
    write(
        "result.json",
        dict(
            status="completed",
            registered_images=len(images),
            photo_references=len(images),
            geo_references=len(images),
        ),
    )
    write("database.db", "database")
    write("points.ply", "ply")
    write("points.ply.vis", "vis")
    for base in ("sparse/0", "dense/sparse"):
        for name in ("cameras", "images", "points3D", "rigs", "frames"):
            write(f"{base}/{name}.bin", name)
    write(
        "sparse/0/project.ini",
        f"database_path={root}/database.db\noutput_path={root}/sparse\n"
        "default_random_seed=0\n[Mapper]\nrandom_seed=0\nnum_threads=1\nhost_budget_bytes=100\n",
    )
    for image in images:
        write("dense/images/" + image["name"], "jpeg" + image["name"])
        for kind in ("depth_maps", "normal_maps"):
            for base, phase in (
                ("cohort", "photometric"),
                ("cohort", "geometric"),
                ("dense", "geometric"),
            ):
                write(
                    f"{base}/stereo/{kind}/{image['name']}.{phase}.bin",
                    phase + image["name"],
                )
    write("command-001.stderr", "timestamp 1")
    return dict(
        schema_version=1,
        frontend_source_commit="c" * 40,
        frontend_source_tree="d" * 40,
        frontend_artifacts=[
            dict(name="bin/colmap-reconstruct", sha256=HASH),
            dict(
                name="share/colmap/python/colmap_reconstruction/cli.py",
                sha256=HASH,
            ),
        ],
        native_source_commit="b" * 40,
        native_elf_sha256=HASH,
        libraries=[dict(name="libc", sha256=HASH)],
        hardware=dict(gpu="recorded GPU", cpu="recorded CPU"),
    )


class InventoryTest(unittest.TestCase):
    def test_inventory_keeps_directories_and_hardlink_publications(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "empty").mkdir()
            (root / "a").write_bytes(b"abc")
            (root / "b").hardlink_to(root / "a")
            members = inventory(root)
            self.assertEqual(members["empty"], {"type": "directory"})
            self.assertEqual(
                members["a"],
                {
                    "type": "file",
                    "length": 3,
                    "sha256": (
                        "ba7816bf8f01cfea414140de5dae2223"
                        "b00361a396177a9cb410ff61f20015ad"
                    ),
                },
            )
            self.assertEqual(members["a"], members["b"])

    def test_capture_and_operational_difference(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            qualification = fixture(root)
            baseline = capture(root, qualification)
            self.assertTrue(compare(baseline, baseline)["package_equal"])
            (root / "command-001.stderr").write_text("timestamp 2")
            (root / "sparse/0/project.ini").write_text(
                "default_random_seed=0\n[Mapper]\nrandom_seed=0\nnum_threads=1\nhost_budget_bytes=200\n"
            )
            verdict = compare(baseline, capture(root, qualification))
            self.assertTrue(verdict["product_equal"])
            self.assertFalse(verdict["package_equal"])

    def test_mutations_and_binding_changes_never_pass(self):
        for mutation in (
            "flip",
            "truncate",
            "rename",
            "swap",
            "missing",
            "extra",
            "directory",
            "link",
            "seed",
            "order",
            "missing_photo",
        ):
            with (
                self.subTest(mutation=mutation),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory)
                qualification = fixture(
                    root,
                    ("0.jpg", "1.jpg", "2.jpg")
                    if mutation == "order"
                    else ("0.jpg", "1.jpg"),
                )
                baseline = capture(root, qualification)
                a, b = root / "dense/images/0.jpg", root / "dense/images/1.jpg"
                if mutation == "flip":
                    a.write_bytes(b"X" + a.read_bytes()[1:])
                elif mutation == "truncate":
                    a.write_bytes(a.read_bytes()[:-1])
                elif mutation == "rename":
                    a.rename(a.with_name("renamed.jpg"))
                elif mutation == "swap":
                    first, second = a.read_bytes(), b.read_bytes()
                    a.write_bytes(second)
                    b.write_bytes(first)
                elif mutation == "missing":
                    a.unlink()
                elif mutation == "extra":
                    (root / "planning/unexpected").write_bytes(b"extra")
                elif mutation == "directory":
                    a.unlink()
                    a.mkdir()
                elif mutation == "link":
                    a.unlink()
                    a.symlink_to("1.jpg")
                elif mutation == "missing_photo":
                    (
                        root / "cohort/stereo/depth_maps/0.jpg.photometric.bin"
                    ).unlink()
                elif mutation == "seed":
                    path = root / "sparse/0/project.ini"
                    path.write_text(
                        path.read_text().replace(
                            "[Mapper]\nrandom_seed=0",
                            "[Mapper]\nrandom_seed=-1",
                        )
                    )
                else:
                    for filename in ("dense-plan", "cohort"):
                        path = root / f"planning/{filename}.json"
                        data = json.loads(path.read_text())
                        jobs = data["phases"]["photo"]
                        if filename == "dense-plan":
                            jobs = jobs["jobs"]
                        jobs[0]["sources"].reverse()
                        path.write_text(json.dumps(data))
                invalid_members = {
                    "rename",
                    "missing",
                    "directory",
                    "link",
                    "missing_photo",
                }
                if mutation in invalid_members or mutation == "seed":
                    message = (
                        "Mapper INI must record seed0/thread1"
                        if mutation == "seed"
                        else "missing/wrong-type required"
                    )
                    with self.assertRaisesRegex(ValueError, message):
                        capture(root, qualification)
                else:
                    candidate = capture(root, qualification)
                    verdict = compare(baseline, candidate)
                    self.assertFalse(verdict["product_equal"])
                    if mutation == "order":
                        self.assertFalse(verdict["binding_equal"])
                        self.assertEqual(verdict["changed"]["scientific"], [])
                    else:
                        self.assertTrue(verdict["binding_equal"])
                        self.assertTrue(verdict["changed"]["scientific"])

    def test_same_basename_publications_remain_distinct_and_swaps_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            names = ("west/shared.jpg", "east/shared.jpg")
            qualification = fixture(root, names)
            baseline = capture(root, qualification)
            paths = [
                f"cohort/stereo/depth_maps/{name}.geometric.bin"
                for name in names
            ]
            self.assertNotEqual(
                baseline["scientific"][paths[0]]["sha256"],
                baseline["scientific"][paths[1]]["sha256"],
            )
            self.assertEqual(
                baseline["scientific"][paths[0]]["length"],
                baseline["scientific"][paths[1]]["length"],
            )
            first, second = (root / name for name in paths)
            first_bytes, second_bytes = first.read_bytes(), second.read_bytes()
            first.write_bytes(second_bytes)
            second.write_bytes(first_bytes)
            verdict = compare(baseline, capture(root, qualification))
            self.assertFalse(verdict["product_equal"])
            self.assertEqual(verdict["changed"]["scientific"], sorted(paths))
            second.unlink()
            with self.assertRaises(ValueError):
                capture(root, qualification)

    def test_utf8_metadata_independent_of_locale(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            json_path, ini_path = root / "metadata.json", root / "project.ini"
            json_path.write_text(
                json.dumps({"name": "caf\u00e9"}, ensure_ascii=False),
                encoding="utf-8",
            )
            ini_path.write_text("label=caf\u00e9\n", encoding="utf-8")
            code = r"""
import sys
from pathlib import Path
from colmap_reconstruction.byte_golden import read_json, ini_binding
assert read_json(sys.argv[1])["name"] == "caf\u00e9"
assert ini_binding(Path(sys.argv[2]), {})["/label"] == "caf\u00e9"
"""
            result = subprocess.run(
                [sys.executable, "-c", code, str(json_path), str(ini_path)],
                env={
                    **os.environ,
                    "LC_ALL": "C",
                    "PYTHONUTF8": "0",
                    "PYTHONCOERCECLOCALE": "0",
                    "PYTHONOPTIMIZE": "0",
                },
                capture_output=True,
            )
            self.assertEqual(
                result.returncode, 0, result.stderr.decode(errors="replace")
            )

    def test_rejects_unsafe_and_malformed_manifests(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = capture(root, fixture(root))
            for mutation in (
                "traversal",
                "duplicate_domain",
                "negative_length",
                "empty_binding",
                "bad_seed",
                "missing_required",
                "wrong_domain",
                "dangling_link",
            ):
                with self.subTest(mutation=mutation):
                    candidate = copy.deepcopy(baseline)
                    if mutation == "traversal":
                        candidate["scientific"]["../escape"] = {
                            "type": "directory"
                        }
                    elif mutation == "duplicate_domain":
                        candidate["operational"]["database.db"] = candidate[
                            "scientific"
                        ]["database.db"]
                    elif mutation == "negative_length":
                        candidate["scientific"]["database.db"]["length"] = -1
                    elif mutation == "missing_required":
                        del candidate["scientific"]["database.db"]
                    elif mutation == "wrong_domain":
                        candidate["operational"]["database.db"] = candidate[
                            "scientific"
                        ].pop("database.db")
                    elif mutation == "dangling_link":
                        candidate["scientific"]["alias"] = dict(
                            type="symlink", target="absent"
                        )
                    elif mutation == "bad_seed":
                        candidate["binding"]["options"]["random_seed"] = -1
                    else:
                        candidate["binding"]["inputs"] = []
                    with self.assertRaises(ValueError):
                        compare(candidate, candidate)
            manifest = root.parent / (root.name + ".json")
            try:
                manifest.write_text('{"schema_version":1,"schema_version":1}')
                with self.assertRaises(ValueError):
                    load_manifest(manifest)
            finally:
                manifest.unlink()

    def test_compare_rejects_bad_outputs_and_missing_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = capture(root, fixture(root))
            for destination in ("../escape", "dense/stereo/wrong.bin"):
                with self.subTest(destination=destination):
                    candidate = copy.deepcopy(baseline)
                    candidate["binding"]["phases"]["photo"]["jobs"][0][
                        "outputs"
                    ]["depth"] = destination
                    with self.assertRaises(ValueError):
                        compare(candidate, candidate)
            for name in (
                "result.json",
                "planning/intent.json",
                "planning/approved-plan.json",
                "planning/build.json",
                "planning/dense-plan.json",
                "planning/cohort.json",
                "planning/sparse-model.json",
                "planning/extraction-inventory.json",
                "planning/resolution-intent.json",
            ):
                with self.subTest(metadata=name):
                    candidate = copy.deepcopy(baseline)
                    del candidate["operational"][name]
                    with self.assertRaises(ValueError):
                        compare(baseline, candidate)

    def test_automatic_device_and_policy_defaults_use_effective_options(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            qualification = fixture(root)
            path = root / "planning/intent.json"
            intent = json.loads(path.read_text())
            intent.update(gpu_index=None, process_host_bytes=None)
            path.write_text(json.dumps(intent))
            path = root / "planning/approved-plan.json"
            approved = json.loads(path.read_text())
            approved["options"]["process_host_bytes"] = 12345
            path.write_text(json.dumps(approved))
            self.assertTrue(
                compare(
                    capture(root, qualification), capture(root, qualification)
                )["product_equal"]
            )

    def test_source_shaped_metadata_different_roots_and_wire_negatives(self):
        with tempfile.TemporaryDirectory() as directory:
            first, second = Path(directory) / "a", Path(directory) / "b"
            first.mkdir()
            second.mkdir()
            qualification = fixture(first)
            fixture(second)
            baseline = capture(first, qualification)
            verdict = compare(baseline, capture(second, qualification))
            self.assertTrue(verdict["product_equal"])
            self.assertFalse(verdict["package_equal"])
            path = second / "planning/dense-plan.json"
            original = path.read_text()
            for mutation in ("nonpath", "config", "unknown", "trailing"):
                data = json.loads(original)
                job = data["phases"]["photo"]["jobs"][0]
                if mutation == "nonpath":
                    job["effective_options_wire_v1"] = job[
                        "effective_options_wire_v1"
                    ].replace("3ff0000000000000", "3ff0000000000001", 1)
                elif mutation == "config":
                    job["effective_options_wire_v1"] = wire(second / "wrong")
                elif mutation == "unknown":
                    job["effective_options_wire_v1"] = job[
                        "effective_options_wire_v1"
                    ].replace("opt_int window_step", "opt_int unknown")
                else:
                    job["effective_options_wire_v1"] += "extra\n"
                path.write_text(json.dumps(data))
                try:
                    candidate = capture(second, qualification)
                except ValueError:
                    continue
                self.assertFalse(compare(baseline, candidate)["product_equal"])

    def test_entrypoint_capture_is_exclusive_and_compare_is_read_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "workspace"
            root.mkdir()
            qualification = fixture(root)
            receipt = root.parent / "qualification.json"
            receipt.write_text(json.dumps(qualification))
            golden = root.parent / "golden.json"
            argv = ["capture", str(root), str(receipt), str(golden)]
            self.assertEqual(main(argv), 0)
            frozen = golden.read_bytes()
            with contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    main(argv)
            self.assertEqual(error.exception.code, 2)
            (root / "points.ply").write_bytes(b"corrupt")
            candidate = root.parent / "candidate.json"
            self.assertEqual(
                main(["capture", str(root), str(receipt), str(candidate)]), 0
            )
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(
                    main(["compare", str(golden), str(candidate)]), 1
                )
            self.assertEqual(golden.read_bytes(), frozen)
            with contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    main(
                        [
                            "capture",
                            str(root),
                            str(receipt),
                            str(root / "bad.json"),
                        ]
                    )
            for key in ("hardware", "libraries", "frontend_artifacts"):
                invalid = copy.deepcopy(qualification)
                invalid[key] = []
                with self.assertRaises(ValueError):
                    capture(root, invalid)

    def test_symlink_targets_and_escape(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            qualification = fixture(root)
            (root / "alias").symlink_to("dense/images/0.jpg")
            baseline = capture(root, qualification)
            (root / "alias").unlink()
            (root / "alias").symlink_to("dense/images/1.jpg")
            self.assertFalse(
                compare(baseline, capture(root, qualification))["product_equal"]
            )
            (root / "alias").unlink()
            (root / "alias").symlink_to("../outside")
            with self.assertRaises(ValueError):
                capture(root, qualification)
            (root / "alias").unlink()
            (root / "alias").symlink_to(str(root / "points.ply"))
            with self.assertRaises(ValueError):
                capture(root, qualification)
            (root / "alias").unlink()
            os.mkfifo(root / "fifo")
            with self.assertRaises(ValueError):
                capture(root, qualification)


if __name__ == "__main__":
    unittest.main()
