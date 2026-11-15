#!/usr/bin/env bash
# SPDX-License-Identifier: BSD-3-Clause
# Actual terminal writer APIs and stereo_fusion caller; no GPU execution.
set -euo pipefail
cd "$(dirname "$0")/../.."
build=${TERMINAL_WRITER_BUILD_DIR:-build-terminal-writers-ci}
cmake -S . -B "$build" -GNinja -DCMAKE_BUILD_TYPE=Release \
  -DCUDA_ENABLED=OFF -DMVS_ENABLED=ON -DTESTS_ENABLED=ON \
  -DGUI_ENABLED=OFF -DONNX_ENABLED=OFF -DOPENGL_ENABLED=OFF -DCGAL_ENABLED=OFF
cmake --build "$build" --parallel 4 --target \
  colmap_util_ply_test colmap_mvs_fusion_test colmap_mvs_published_geo_test
export COLMAP_BINARY="$build/src/colmap/exe/colmap"
COLMAP_BINARY=$(realpath "$COLMAP_BINARY")
export COLMAP_BINARY
test -x "$COLMAP_BINARY"
sha256sum "$COLMAP_BINARY" "$build/src/colmap/util/ply_test" \
  "$build/src/colmap/mvs/fusion_test"
timeout 60s "$build/src/colmap/util/ply_test" \
  --gtest_filter='Ply.BinaryPointWriter*' \
  --gtest_output="xml:$build/ply-writer.xml"
# Both writer failure contracts and their actual CLI callers must execute.
timeout 180s "$build/src/colmap/mvs/fusion_test" \
  --gtest_filter='WritePointsVisibility.*:StereoFusion.PlyWriterCliFailureDoesNotPublishSuccessReceipt:StereoFusion.VisibilityWriterCliFailureDoesNotPublishSuccessReceipt' \
  --gtest_output="xml:$build/fusion-writer.xml"
# Missing/renamed tests and skipped integration must not masquerade as GREEN.
python3 - "$build" <<'PY'
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

build = Path(sys.argv[1])
for name, count in (("ply-writer.xml", 5), ("fusion-writer.xml", 5)):
    suite = ET.parse(build / name).getroot()
    assert int(suite.attrib["tests"]) == count, suite.attrib
    for field in ("failures", "errors"):
        assert int(suite.attrib.get(field, "0")) == 0, suite.attrib
    assert int(suite.attrib.get("skipped", "0")) == 0, suite.attrib
    assert not suite.findall(".//skipped"), "Integration skipped: not GREEN"
PY

# Real Mat::Write byte oracle and checked publisher, then actual metadata CLI.
TMPDIR="$(realpath "$build")" timeout 30s "$build/src/colmap/mvs/colmap_mvs_published_geo_test"
COLMAP_TEST_EXECUTABLE="$COLMAP_BINARY" PYTHONPATH="$PWD/python" python3 -m unittest \
  colmap_reconstruction.single_device_cli_test
