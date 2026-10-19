# SPDX-License-Identifier: BSD-3-Clause

import pickle

import pytest

import pycolmap

pytestmark = pytest.mark.skipif(
    not hasattr(pycolmap, "PatchMatchOptions"),
    reason="PatchMatchOptions not available (requires CUDA or HIP)",
)


def test_patch_match_options_init() -> None:
    options = pycolmap.PatchMatchOptions()
    assert options is not None


def test_stereo_fusion_options_init() -> None:
    options = pycolmap.StereoFusionOptions()
    assert options.use_mmap is False
    assert options.max_image_size == -1
    assert options.use_cache is False
    assert options.check()


def test_stereo_fusion_options_use_mmap_assignment() -> None:
    options = pycolmap.StereoFusionOptions()
    options.use_mmap = True
    assert options.use_mmap is True
    assert options.check()
    options.use_mmap = False
    assert options.use_mmap is False
    assert options.check()


@pytest.mark.parametrize(
    ("use_mmap", "max_image_size", "use_cache"),
    [(False, 640, True), (True, -1, False)],
)
def test_stereo_fusion_options_use_mmap_roundtrip(
    use_mmap: bool, max_image_size: int, use_cache: bool
) -> None:
    values = {
        "use_mmap": use_mmap,
        "max_image_size": max_image_size,
        "use_cache": use_cache,
    }
    options = pycolmap.StereoFusionOptions(values)
    serialized = options.todict()
    for name, value in values.items():
        assert serialized[name] == value

    merged = pycolmap.StereoFusionOptions()
    merged.mergedict(values)
    for restored in (
        pycolmap.StereoFusionOptions(serialized),
        pycolmap.StereoFusionOptions(**values),
        pickle.loads(pickle.dumps(options)),
        merged,
    ):
        assert restored.use_mmap is use_mmap
        assert restored.max_image_size == max_image_size
        assert restored.use_cache is use_cache
        assert restored.check()


@pytest.mark.parametrize(
    ("max_image_size", "use_cache"),
    [(-1, False), (640, False), (-1, True)],
)
def test_stereo_fusion_options_use_mmap_check(
    max_image_size: int, use_cache: bool
) -> None:
    options = pycolmap.StereoFusionOptions(
        use_mmap=True, max_image_size=max_image_size, use_cache=use_cache
    )
    assert options.check() is (max_image_size == -1 and not use_cache)
    options.use_mmap = False
    assert options.check()


@pytest.mark.parametrize(
    "name",
    [
        "patch_match_stereo",
        "stereo_fusion",
    ],
)
def test_public_api_callable(name: str) -> None:
    assert callable(getattr(pycolmap, name))
