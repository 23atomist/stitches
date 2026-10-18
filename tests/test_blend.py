import numpy as np
import pytest

from pcbstitch.blend import laplacian_pyramid, max_levels, multiband_blend, reconstruct

from .synth import make_scene


def test_pyramid_round_trip():
    img = make_scene(203, 151, seed=5).astype(np.float32)
    for levels in (1, 3, 5):
        lap = laplacian_pyramid(img, levels)
        assert len(lap) == levels
        assert np.abs(reconstruct(lap) - img).max() < 1e-2


def test_identical_inputs_are_unchanged():
    img = make_scene(256, 192, seed=6).astype(np.float32)
    mask_a = np.zeros((192, 256), np.float32)
    mask_a[:, :130] = 1.0
    out = multiband_blend([img, img.copy()], [mask_a, 1.0 - mask_a], levels=5)
    assert np.abs(out - img).max() < 0.5


def test_sharp_detail_is_not_doubled_but_colour_is_blended():
    base = make_scene(256, 192, seed=7).astype(np.float32)
    shifted = np.roll(base, 3, axis=1)  # a misaligned copy
    bright = base * 1.3
    mask = np.zeros((192, 256), np.float32)
    mask[:, :128] = 1.0
    out = multiband_blend([base, shifted], [mask, 1.0 - mask], levels=5)
    # far from the seam each side is the pure source
    assert np.abs(out[:, :60] - base[:, :60]).max() < 1.0
    assert np.abs(out[:, 200:] - shifted[:, 200:]).max() < 1.0
    out2 = multiband_blend([base, bright], [mask, 1.0 - mask], levels=5)
    ratio = out2[:, 120:136].mean() / base[:, 120:136].mean()
    assert 1.05 < ratio < 1.25  # brightness ramps across the seam instead of jumping
    assert np.abs(out2[:, :20] - base[:, :20]).max() < 2.0


def test_validation_and_levels():
    img = np.zeros((32, 32, 3), np.float32)
    with pytest.raises(ValueError):
        multiband_blend([], [], 3)
    with pytest.raises(ValueError):
        multiband_blend([img], [], 3)
    with pytest.raises(ValueError):
        multiband_blend([img], [np.zeros((16, 16), np.float32)], 3)
    assert max_levels(32, 32, 10) == 3
    assert max_levels(2048, 2048, 6) == 6
    out = multiband_blend([img], [np.ones((32, 32), np.float32)], 10)
    assert out.shape == img.shape
