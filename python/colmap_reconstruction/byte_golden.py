# SPDX-License-Identifier: BSD-3-Clause
"""Developer-only raw byte guard; never reconstructs or promotes a baseline."""

import argparse
import hashlib
import json
import os
import re
import stat
from pathlib import Path, PurePosixPath


def sha256(path):
    value = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def safe_path(value):
    if (
        not isinstance(value, str)
        or not value
        or "\\" in value
        or "\0" in value
        or PurePosixPath(value).is_absolute()
        or any(p in ("", ".", "..") for p in value.split("/"))
    ):
        raise ValueError("unsafe relative path: " + repr(value))
    return value


def inventory(root):
    """All publications, including directories, links and hardlink aliases.

    Operate only on a quiescent, trusted local tree: not a hostile concurrent
    writer/sandbox boundary. Never follow directory symlinks.
    """
    root = Path(root).resolve(strict=True)
    members = {}

    def visit(directory):
        for path in sorted(directory.iterdir()):
            name = safe_path(path.relative_to(root).as_posix())
            mode = path.lstat().st_mode
            if stat.S_ISLNK(mode):
                target = os.readlink(path)
                if Path(target).is_absolute():
                    raise ValueError("absolute symlink: " + name)
                try:
                    path.resolve(strict=True).relative_to(root)
                except (ValueError, OSError, RuntimeError) as error:
                    raise ValueError(
                        "escaping/dangling symlink: " + name
                    ) from error
                members[name] = dict(type="symlink", target=target)
            elif stat.S_ISDIR(mode):
                members[name] = dict(type="directory")
                visit(path)
            elif stat.S_ISREG(mode):
                members[name] = dict(
                    type="file", length=path.stat().st_size, sha256=sha256(path)
                )
            else:
                raise ValueError("unsupported publication type: " + name)

    visit(root)
    return members


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key: " + key)
        result[key] = value
    return result


def read_json(path):
    return json.loads(
        Path(path).read_text(encoding="utf-8"),
        object_pairs_hook=unique_object,
        parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)),
    )


# Only these explicit operational intent fields are outside scientific binding.
# Unknown/new options remain scientific by default.
OPERATIONAL_OPTIONS = frozenset(
    (
        "photos",
        "workspace",
        "colmap",
        "yes",
        "timeout_seconds",
        "host_budget_bytes",
        "device_budget_bytes",
        "reserve_host_bytes",
        "reserve_device_bytes",
        "disk_budget_bytes",
        "reserve_disk_bytes",
        "loader_scratch_host_bytes",
        "process_host_bytes",
        "process_device_bytes",
        "unknown_host_bytes",
        "unknown_device_bytes",
        "unknown_provenance",
    )
)
INI_OPERATIONAL = frozenset(
    (
        ("", "database_path"),
        ("", "image_path"),
        ("", "output_path"),
        ("", "log_path"),
        ("Mapper", "host_budget_bytes"),
        ("Mapper", "policy_host_bytes"),
        ("Mapper", "reserve_host_bytes"),
    )
)


def scientific_options(options):
    return {k: v for k, v in options.items() if k not in OPERATIONAL_OPTIONS}


def ini_binding(path, expected_paths):
    # Parse scalar INI without interpolation, case folding or numeric coercion.
    # Unsupported/duplicate syntax refuses instead of silently losing fields.
    section = ""
    values = {}
    scientific = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        syntax = line.strip()
        if not syntax or syntax.startswith(("#", ";")):
            continue
        if syntax.startswith("[") and syntax.endswith("]"):
            section = syntax[1:-1]
            continue
        key, separator, value = line.partition("=")
        key = key.strip()
        token = (section, key)
        if not separator or not key or token in values:
            raise ValueError("malformed/duplicate INI field: " + str(path))
        values[token] = value
        if token in INI_OPERATIONAL:
            if token in expected_paths and value != expected_paths[token]:
                raise ValueError("INI path disagrees with recorded execution")
            if section == "Mapper" and not re.fullmatch(r"[0-9]+", value):
                raise ValueError("invalid INI operational byte capacity")
        else:
            scientific[section + "/" + key] = value
    if (
        "/default_random_seed" in scientific
        and scientific["/default_random_seed"] != "0"
    ):
        raise ValueError("INI default seed must be 0")
    if any(s == "Mapper" for s, _ in values) and (
        values.get(("Mapper", "random_seed")) != "0"
        or values.get(("Mapper", "num_threads")) != "1"
    ):
        raise ValueError("Mapper INI must record seed0/thread1")
    return scientific


def require_hash(value):
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(c not in "0123456789abcdef" for c in value)
    ):
        raise ValueError("invalid SHA256")


def qualification_binding(value):
    if (
        set(value)
        != {
            "schema_version",
            "frontend_source_commit",
            "frontend_source_tree",
            "frontend_artifacts",
            "native_source_commit",
            "native_elf_sha256",
            "libraries",
            "hardware",
        }
        or type(value["schema_version"]) is not int
        or value["schema_version"] != 1
    ):
        raise ValueError("invalid execution qualification schema")
    require_hash(value["native_elf_sha256"])
    for key in (
        "frontend_source_commit",
        "frontend_source_tree",
        "native_source_commit",
    ):
        commit = value[key]
        if (
            not isinstance(commit, str)
            or len(commit) != 40
            or any(c not in "0123456789abcdef" for c in commit)
        ):
            raise ValueError("invalid recorded Git identity")
    if (
        not isinstance(value["hardware"], dict)
        or not value["hardware"]
        or not value["libraries"]
    ):
        raise ValueError("missing recorded hardware/libraries")
    if not value["frontend_artifacts"]:
        raise ValueError("missing frontend artifact identities")
    names = set()
    for library in value["frontend_artifacts"] + value["libraries"]:
        if (
            set(library) != {"name", "sha256"}
            or not isinstance(library["name"], str)
            or not library["name"]
            or library["name"] in names
        ):
            raise ValueError("invalid/duplicate library identity")
        names.add(library["name"])
        require_hash(library["sha256"])
    required_frontend = {
        "bin/colmap-reconstruct",
        "share/colmap/python/colmap_reconstruction/cli.py",
    }
    if not required_frontend.issubset(
        {item["name"] for item in value["frontend_artifacts"]}
    ):
        raise ValueError("missing launcher/module identities")
    return value


def required_members(selected, names):
    required = {
        "database.db",
        "points.ply",
        "points.ply.vis",
        "sparse/" + selected + "/project.ini",
        "result.json",
    }
    required.update(
        "planning/" + name + ".json"
        for name in (
            "intent",
            "approved-plan",
            "build",
            "dense-plan",
            "cohort",
            "sparse-model",
            "extraction-inventory",
            "resolution-intent",
        )
    )
    for base in ("sparse/" + selected, "dense/sparse"):
        required.update(
            base + "/" + name + ".bin"
            for name in ("cameras", "images", "points3D", "rigs", "frames")
        )
    for name in names:
        required.add("dense/images/" + name)
        for kind in ("depth_maps", "normal_maps"):
            for base, phase in (
                ("cohort", "photometric"),
                ("cohort", "geometric"),
                ("dense", "geometric"),
            ):
                required.add(f"{base}/stereo/{kind}/{name}.{phase}.bin")
    return required


def operational_path(name):
    metadata = {
        "planning/" + item + ".json"
        for item in (
            "intent",
            "approved-plan",
            "build",
            "dense-plan",
            "cohort",
            "sparse-model",
            "extraction-inventory",
            "resolution-intent",
            "photos",
            "resolution-effective",
            "resource-policy",
            "multi-gpu-plan",
        )
    }
    metadata.update(
        (
            "result.json",
            "planning/fusion.receipt",
            "planning/extraction-images.txt",
            "planning/extraction-dimensions.txt",
        )
    )
    return bool(
        name in metadata
        or name.endswith("/project.ini")
        or name == "project.ini"
        or re.fullmatch(r"command-\d+\.(stdout|stderr)", name)
        or re.fullmatch(
            r"planning/(resources|extraction-selection)-\d+\.json", name
        )
        or name == "profiler"
        or name.startswith("profiler/")
    )


def wire_binding(wire, workspace):
    """V1 config path binding only; every other token remains byte-exact."""
    keys = [
        "int:max_image_size",
        "string:gpu_index",
        "double:depth_min",
        "double:depth_max",
        "int:random_seed",
        "int:window_radius",
        "int:window_step",
        "double:sigma_spatial",
        "double:sigma_color",
        "int:num_samples",
        "bool:owned_distinct_source_cache",
        "bool:owned_tex_lookahead",
        "double:ncc_sigma",
        "double:min_triangulation_angle",
        "double:incident_angle_sigma",
        "int:num_iterations",
        "bool:geom_consistency",
        "double:geom_consistency_regularizer",
        "double:geom_consistency_max_cost",
        "bool:filter",
        "double:filter_min_ncc",
        "double:filter_min_triangulation_angle",
        "int:filter_min_num_consistent",
        "double:filter_geom_consistency_max_cost",
        "bool:write_consistency_graph",
        "bool:allow_missing_files",
        "int:num_threads",
        "double:cache_size",
        "string:workspace",
        "string:config",
    ]
    lines = wire.splitlines(keepends=True)
    if lines[:3] != [
        "COLMAP_OPTIMIZED_PHOTO_OPTIONS_V1\n",
        "build_cuda_enabled 1\n",
        "build_hip_enabled 0\n",
    ]:
        raise ValueError("unsupported V1 wire header/build")
    position = 3
    for descriptor in keys:
        kind, key = descriptor.split(":")
        if key == "ncc_sigma":
            for optional in ("owned_cuda_graphs", "owned_async_pool"):
                if (
                    position < len(lines)
                    and lines[position] == f"opt_bool {optional} 1\n"
                ):
                    position += 1
        if position >= len(lines):
            raise ValueError("truncated V1 wire")
        row = lines[position]
        patterns = dict(
            int=r"-?(?:0|[1-9][0-9]*)",
            bool="[01]",
            double="[0-9a-f]{16}",
            string=r"(?:0|[1-9][0-9]*) [0-9a-f]*",
        )
        if not re.fullmatch(
            "opt_" + kind + " " + key + " (" + patterns[kind] + ")\n", row
        ):
            raise ValueError("invalid V1 field/order/token: " + key)
        if kind == "string":
            _, _, size, encoded = row[:-1].split(" ")
            decoded = bytes.fromhex(encoded)
            if len(decoded) != int(size):
                raise ValueError("V1 encoded string length mismatch")
            if key in ("workspace", "config"):
                expected = (
                    workspace
                    if key == "workspace"
                    else workspace + "/stereo/patch-match.cfg"
                )
                if decoded.decode("utf-8") != expected:
                    raise ValueError(
                        "V1 path does not bind actual graph: " + key
                    )
        position += 1
    if position != len(lines):
        raise ValueError("trailing/unknown V1 fields")
    return "".join(lines[:-2])


def publication_outputs(phase, name):
    suffix = "photometric" if phase == "photo" else "geometric"
    return {
        key: f"dense/stereo/{kind}/{name}.{suffix}.bin"
        for key, kind in (("depth", "depth_maps"), ("normal", "normal_maps"))
    }


def graph_binding(graph, all_members):
    """Bind explicit paths; all other graph metadata stays exact."""
    graph = json.loads(json.dumps(graph, allow_nan=False))
    workspace = graph["workspace_path"]
    if (
        not isinstance(workspace, str)
        or not workspace.startswith("/")
        or ".." in workspace.split("/")
    ):
        raise ValueError("invalid recorded native workspace")

    def relative(path):
        try:
            tail = PurePosixPath(path).relative_to(workspace).as_posix()
        except ValueError as error:
            raise ValueError("graph path escapes recorded workspace") from error
        return "dense/" + safe_path(tail)

    graph["workspace_path"] = "dense"
    image_names = {
        image["index"]: safe_path(image["name"]) for image in graph["images"]
    }
    for image in graph["images"]:
        image["path"] = relative(image["path"])
        if image["path"] != "dense/images/" + image["name"]:
            raise ValueError("graph image path/name mismatch")
    for phase, data in graph["phases"].items():
        for job in data["jobs"]:
            job["effective_options_wire_v1"] = wire_binding(
                job["effective_options_wire_v1"], workspace
            )
            job["outputs"] = {
                key: relative(path) for key, path in job["outputs"].items()
            }
            if job["outputs"] != publication_outputs(
                phase, image_names[job["reference"]]
            ):
                raise ValueError("wrong planned publication destination")
    for entry in graph.get("input_files", []):
        entry["path"] = relative(entry["path"])
        if all_members.get(entry["path"], {}).get("sha256") != entry["sha256"]:
            raise ValueError("graph consumed input hash mismatch")
    for entry in graph.get("future_photo_dependencies", []):
        entry["path"] = relative(entry["path"])
    return graph


def _capture(root, qualification):
    """Inventory a ceased workspace with recorded execution qualification."""
    root = Path(root).resolve(strict=True)
    all_members = inventory(root)

    def metadata(name):
        path = "planning/" + name + ".json"
        if all_members.get(path, {}).get("type") != "file":
            raise ValueError("missing regular metadata: " + path)
        return read_json(root / path)

    qualification_binding(qualification)
    intent = metadata("intent")
    approved = metadata("approved-plan")
    requested = scientific_options(intent)
    options = scientific_options(approved["options"])
    if requested.get("gpu_index") is None:
        requested["gpu_index"] = options.get("gpu_index")
    if options != requested:
        raise ValueError("intent/approved options mismatch")
    if (
        options.get("random_seed") != 0
        or options.get("threads") != 1
        or options.get("backend") != "experimental_fixed_cohort"
        or options.get("output_policy") != "photo_and_geo"
    ):
        raise ValueError("requires seed0/thread1/fixed-cohort/photo_and_geo")
    build = metadata("build")
    if (
        build["executable_sha256"] != qualification["native_elf_sha256"]
        or build["identity"]["source_commit"]
        != qualification["native_source_commit"]
    ):
        raise ValueError("recorded execution/native build mismatch")
    # Check an available native artifact; do not require it after closure.
    executable = build.get("executable")
    if (
        executable
        and Path(executable).is_file()
        and sha256(executable) != build["executable_sha256"]
    ):
        raise ValueError(
            "available native ELF disagrees with recorded execution"
        )
    raw_graph = metadata("dense-plan")
    recorded_workspace = approved["options"]["workspace"]
    if raw_graph["workspace_path"] != recorded_workspace + "/dense":
        raise ValueError("native graph workspace disagrees with approved run")
    graph = graph_binding(raw_graph, all_members)
    cohort = metadata("cohort")
    selection = metadata("sparse-model")
    selected = safe_path(selection["name"])
    if "/" in selected:
        raise ValueError("invalid selected sparse model")
    if all_members.get("result.json", {}).get("type") != "file":
        raise ValueError("missing result")
    result = read_json(root / "result.json")
    images = graph["images"]
    if (
        not images
        or len(images) != selection["counts"]["num_registered_images"]
    ):
        raise ValueError("missing registered image membership")
    indices = [image["index"] for image in images]
    names = [safe_path(image["name"]) for image in images]
    if len(set(indices)) != len(indices) or len(set(names)) != len(names):
        raise ValueError("duplicate image membership")
    if (
        result["status"] != "completed"
        or cohort["status"] != "completed"
        or cohort["effective_backend"] != options["backend"]
        or cohort.get("output_policy") != "photo_and_geo"
        or any(
            result[k] != len(images)
            for k in ("registered_images", "photo_references", "geo_references")
        )
    ):
        raise ValueError("incomplete run")
    phases = graph["phases"]
    for phase in ("photo", "geo"):
        jobs = phases[phase]["jobs"]
        if (
            phases[phase]["requested_options"].get("random_seed") != 0
            or len(jobs) != len(indices)
            or {j["reference"] for j in jobs} != set(indices)
            or cohort["phases"][phase]
            != [
                {k: j[k] for k in ("reference", "sources", "options")}
                for j in jobs
            ]
        ):
            raise ValueError("effective phase/order/options mismatch")
        for job in jobs:
            if (
                not job["sources"]
                or any(s not in indices for s in job["sources"])
                or job["options"].get("random_seed") != 0
                or job["options"].get("num_threads") != 1
                or str(job["options"].get("gpu_index"))
                != str(options.get("gpu_index"))
            ):
                raise ValueError("invalid source/seed/thread binding")
    order = metadata("extraction-inventory")["effective_order"]
    photos = metadata("resolution-intent")["original"]["photos"]
    by_name = {safe_path(photo["path"]): photo for photo in photos}
    if (
        len(by_name) != len(photos)
        or len(order) != len(set(order))
        or set(order) != set(by_name)
        or not order
    ):
        raise ValueError("invalid ordered encoded input membership")
    inputs = []
    for name in order:
        photo = by_name[name]
        entry = approved["input_binding"][name]
        require_hash(entry["sha256"])
        if any(
            type(photo.get(k)) is not int or photo[k] <= 0
            for k in ("width", "height")
        ):
            raise ValueError("invalid encoded dimensions")
        inputs.append(dict(name=name, sha256=entry["sha256"], dimensions=photo))
    for name in required_members(selected, names):
        if all_members.get(name, {}).get("type") != "file":
            raise ValueError(
                "missing/wrong-type required scientific publication: " + name
            )
    # Unknown publications default scientific; nothing can silently disappear.
    scientific, operational, inis = {}, {}, {}
    for name, member in all_members.items():
        if name.endswith("/project.ini") or name == "project.ini":
            if member["type"] != "file":
                raise ValueError("INI must be regular")
            expected_paths = {
                ("", "database_path"): recorded_workspace + "/database.db",
                ("", "output_path"): recorded_workspace + "/sparse",
            }
            if "photos" in approved["options"]:
                expected_paths[("", "image_path")] = approved["options"][
                    "photos"
                ]
            inis[name] = ini_binding(root / name, expected_paths)
            operational[name] = member
        elif operational_path(name):
            operational[name] = member
        else:
            scientific[name] = member
    return dict(
        schema_version=1,
        binding=dict(
            selected_model=selected,
            options=options,
            inputs=inputs,
            images=[
                {k: image[k] for k in ("index", "name", "width", "height")}
                for image in images
            ],
            phases=phases,
            graph_metadata={k: v for k, v in graph.items() if k != "phases"},
            ini=inis,
            qualification=qualification,
            qualification_sha256=hashlib.sha256(
                json.dumps(
                    qualification,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                ).encode("utf-8")
            ).hexdigest(),
        ),
        scientific=scientific,
        operational=operational,
    )


def _validate_manifest(value):
    json.dumps(value, allow_nan=False)
    if (
        not isinstance(value, dict)
        or set(value)
        != {"schema_version", "binding", "scientific", "operational"}
        or type(value["schema_version"]) is not int
        or value["schema_version"] != 1
    ):
        raise ValueError("invalid manifest schema")
    binding = value["binding"]
    if set(binding) != {
        "selected_model",
        "options",
        "inputs",
        "images",
        "phases",
        "graph_metadata",
        "ini",
        "qualification",
        "qualification_sha256",
    }:
        raise ValueError("invalid binding schema")
    qualification_binding(binding["qualification"])
    expected = hashlib.sha256(
        json.dumps(
            binding["qualification"],
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    if binding["qualification_sha256"] != expected:
        raise ValueError("execution qualification content hash mismatch")
    selected = safe_path(binding["selected_model"])
    if "/" in selected:
        raise ValueError("invalid selected model")
    options = binding["options"]
    if (
        options.get("random_seed") != 0
        or options.get("threads") != 1
        or options.get("backend") != "experimental_fixed_cohort"
        or options.get("output_policy") != "photo_and_geo"
        or type(options.get("gpu_index")) is not int
        or type(options.get("random_seed")) is not int
        or type(options.get("threads")) is not int
    ):
        raise ValueError("invalid fixed scientific options")
    if not all(binding[k] for k in ("inputs", "images", "phases", "ini")):
        raise ValueError("empty scientific binding")
    input_names = set()
    for entry in binding["inputs"]:
        if set(entry) != {"name", "sha256", "dimensions"}:
            raise ValueError("invalid input schema")
        safe_path(entry["name"])
        require_hash(entry["sha256"])
        if (
            entry["name"] in input_names
            or entry["dimensions"].get("path") != entry["name"]
        ):
            raise ValueError("invalid input membership")
        input_names.add(entry["name"])
        if any(
            type(entry["dimensions"].get(k)) is not int
            or entry["dimensions"][k] <= 0
            for k in ("width", "height")
        ):
            raise ValueError("invalid encoded shape")
    indices, names = set(), set()
    for image in binding["images"]:
        if set(image) != {"index", "name", "width", "height"}:
            raise ValueError("invalid image schema")
        safe_path(image["name"])
        if (
            type(image["index"]) is not int
            or image["index"] in indices
            or image["name"] in names
            or any(
                type(image[k]) is not int or image[k] <= 0
                for k in ("width", "height")
            )
        ):
            raise ValueError("invalid image membership/shape")
        indices.add(image["index"])
        names.add(image["name"])
    if set(binding["phases"]) != {"photo", "geo"}:
        raise ValueError("invalid phases")
    graph_metadata = binding["graph_metadata"]
    if (
        graph_metadata.get("schema_version") != 1
        or graph_metadata.get("workspace_path") != "dense"
        or [
            {k: image[k] for k in ("index", "name", "width", "height")}
            for image in graph_metadata.get("images", [])
        ]
        != binding["images"]
    ):
        raise ValueError("invalid graph metadata binding")
    image_names = {image["index"]: image["name"] for image in binding["images"]}
    for phase_name, phase in binding["phases"].items():
        jobs = phase["jobs"]
        if (
            len(jobs) != len(indices)
            or {j["reference"] for j in jobs} != indices
            or phase["requested_options"].get("random_seed") != 0
        ):
            raise ValueError("invalid phase membership")
        for job in jobs:
            if job["outputs"] != publication_outputs(
                phase_name, image_names[job["reference"]]
            ):
                raise ValueError("wrong planned publication destination")
            # Restore fixed path sentinels to validate the non-path wire shape.
            wire = job["effective_options_wire_v1"]
            for key, path in (
                ("workspace", "dense"),
                ("config", "dense/stereo/patch-match.cfg"),
            ):
                encoded = path.encode("utf-8")
                wire += f"opt_string {key} {len(encoded)} {encoded.hex()}\n"
            wire_binding(wire, "dense")
            if (
                not job["sources"]
                or any(s not in indices for s in job["sources"])
                or job["options"].get("random_seed") != 0
                or job["options"].get("num_threads") != 1
                or str(job["options"].get("gpu_index"))
                != str(options.get("gpu_index"))
            ):
                raise ValueError("invalid effective source/options")
    for name, ini in binding["ini"].items():
        safe_path(name)
        if not isinstance(ini, dict) or not ini:
            raise ValueError("invalid INI binding")
        if any(k.startswith("Mapper/") for k in ini) and (
            ini.get("Mapper/random_seed") != "0"
            or ini.get("Mapper/num_threads") != "1"
        ):
            raise ValueError("invalid Mapper INI seed/thread")
    seen = set()
    for domain in ("scientific", "operational"):
        if not isinstance(value[domain], dict) or not value[domain]:
            raise ValueError("empty domain")
        for name, member in value[domain].items():
            safe_path(name)
            if name in seen or not isinstance(member, dict):
                raise ValueError("duplicate/invalid member")
            if operational_path(name) != (domain == "operational"):
                raise ValueError("incorrect member domain")
            seen.add(name)
            kind = member.get("type")
            if kind == "file" and set(member) == {"type", "length", "sha256"}:
                if type(member["length"]) is not int or member["length"] < 0:
                    raise ValueError("invalid length")
                require_hash(member["sha256"])
            elif kind == "directory" and set(member) == {"type"}:
                pass
            elif kind == "symlink" and set(member) == {"type", "target"}:
                target = member["target"]
                if (
                    not isinstance(target, str)
                    or not target
                    or "\\" in target
                    or "\0" in target
                    or target.startswith("/")
                ):
                    raise ValueError("invalid link target")
                resolved = os.path.normpath(
                    str(PurePosixPath(name).parent / target)
                )
                if resolved != ".":
                    safe_path(resolved)
            else:
                raise ValueError("invalid member schema/type")
    for name in seen:
        parent = PurePosixPath(name).parent
        while str(parent) != ".":
            key = str(parent)
            member = value["scientific"].get(key, value["operational"].get(key))
            if member != {"type": "directory"}:
                raise ValueError("missing/non-directory parent")
            parent = parent.parent
    members = dict(value["scientific"], **value["operational"])
    for name in required_members(selected, names):
        if members.get(name, {}).get("type") != "file":
            raise ValueError("missing required publication: " + name)
    ini_names = {
        name
        for name in members
        if name.endswith("/project.ini") or name == "project.ini"
    }
    if ini_names != set(binding["ini"]):
        raise ValueError("missing/extra INI binding")

    def resolve_link(name):
        # Resolve paths against the frozen inventory, never the host filesystem.
        parts = name.split("/")
        expansions = 0
        i = 0
        while i < len(parts):
            key = "/".join(parts[: i + 1])
            member = members.get(key)
            if member is None:
                raise ValueError("dangling manifest link")
            if member["type"] == "symlink":
                expansions += 1
                if expansions > len(members):
                    raise ValueError("cyclic manifest link")
                target = os.path.normpath(
                    str(PurePosixPath(key).parent / member["target"])
                )
                if target != ".":
                    safe_path(target)
                parts = ([] if target == "." else target.split("/")) + parts[
                    i + 1 :
                ]
                i = 0
            else:
                if i < len(parts) - 1 and member["type"] != "directory":
                    raise ValueError("link through non-directory")
                i += 1

    for name, member in members.items():
        if member["type"] == "symlink":
            resolve_link(name)
    return value


def capture(root, qualification):
    try:
        value = _capture(root, qualification)
        return validate_manifest(value)
    except (KeyError, TypeError, AttributeError, IndexError) as error:
        raise ValueError("malformed execution metadata") from error


def validate_manifest(value):
    try:
        return _validate_manifest(value)
    except (KeyError, TypeError, AttributeError, IndexError) as error:
        raise ValueError("malformed manifest") from error


def load_manifest(path):
    return validate_manifest(read_json(path))


def compare(baseline, candidate):
    validate_manifest(baseline)
    validate_manifest(candidate)
    changed = {}
    for domain in ("scientific", "operational"):
        left, right = baseline[domain], candidate[domain]
        changed[domain] = sorted(
            k for k in left.keys() | right.keys() if left.get(k) != right.get(k)
        )
    binding_equal = json.dumps(
        baseline["binding"], sort_keys=True
    ) == json.dumps(candidate["binding"], sort_keys=True)
    product_equal = binding_equal and not changed["scientific"]
    return dict(
        product_equal=product_equal,
        package_equal=product_equal and not changed["operational"],
        binding_equal=binding_equal,
        changed=changed,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    create = sub.add_parser(
        "capture", help="explicit exclusive-create; never promotes a golden"
    )
    create.add_argument("workspace", type=Path)
    create.add_argument("qualification", type=Path)
    create.add_argument("output", type=Path)
    check = sub.add_parser(
        "compare", help="read-only; exit 1 on product mismatch"
    )
    check.add_argument("baseline", type=Path)
    check.add_argument("candidate", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "capture":
            if args.output.resolve().is_relative_to(args.workspace.resolve()):
                raise ValueError(
                    "manifest output must be outside inventoried workspace"
                )
            if args.output.exists() or args.output.is_symlink():
                raise ValueError("refusing existing output")
            value = capture(args.workspace, read_json(args.qualification))
            validate_manifest(value)
            with args.output.open("x") as stream:
                json.dump(
                    value, stream, sort_keys=True, indent=2, allow_nan=False
                )
                stream.write("\n")
            return 0
        verdict = compare(
            load_manifest(args.baseline), load_manifest(args.candidate)
        )
        print(json.dumps(verdict, sort_keys=True))
        return 0 if verdict["product_equal"] else 1
    except (ValueError, KeyError, TypeError, OSError) as error:
        parser.exit(2, "byte-golden refused: " + str(error) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
