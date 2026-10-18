import numpy as np
import pytest

from pcbstitch.camera import CameraModel, MosaicModel
from pcbstitch.compose import ComposeConfig, Composer
from pcbstitch.geometry import Rect
from pcbstitch.scan import load_session
from pcbstitch.tiles import TileSource

from .synth import default_scan


def test_translation_model_reproduces_tiles_exactly(tmp_path):
    scan, root = default_scan(tmp_path, static_rows=0, vignette=0.0, gain_spread=0.0, noise=0.0)
    src = TileSource(load_session(root))
    model = scan.truth_model().normalized()
    comp = Composer(src, model, None, ComposeConfig(block=128, pad=32, levels=3, workers=2))
    out = np.zeros((comp.height, comp.width, 3), np.uint8)
    comp.run(out)
    # every tile's central region is copied verbatim (up to JPEG error)
    for t, i in zip(scan.tiles, range(len(scan.tiles))):
        x, y = int(t.x - scan.tiles[0].x), int(t.y - scan.tiles[0].y)
        got = out[y + 100 : y + 140, x + 140 : x + 180].astype(int)
        want = src.load(i)[100:140, 140:180].astype(int)
        assert np.abs(got - want).mean() < 4.0
    assert comp.render_block(Rect(-10, -10, 0, 0)).shape == (0, 0, 3)
    assert comp.render_block(Rect(comp.width - 5, comp.height - 5, comp.width + 50, comp.height + 50)).shape == (5, 5, 3)
    with pytest.raises(ValueError):
        comp.run(np.zeros((10, 10, 3), np.uint8))
    with pytest.raises(ValueError):
        Composer(src, model, None, ComposeConfig(interpolation="nearest"))
    shifted = scan.truth_model()
    shifted.affines[:, 0, 2] -= 50
    with pytest.raises(ValueError):
        Composer(src, shifted, None, ComposeConfig())


def test_block_failure_surfaces_quickly(tmp_path):
    scan, root = default_scan(tmp_path, rows=1, cols=2, static_rows=0)
    src = TileSource(load_session(root))
    comp = Composer(src, scan.truth_model().normalized(), None, ComposeConfig(block=64, pad=8, levels=2, workers=2))
    calls = []

    def boom(block):
        calls.append(block)
        raise RuntimeError("bad block")

    comp.render_block = boom
    out = np.zeros((comp.height, comp.width, 3), np.uint8)
    with pytest.raises(RuntimeError):
        comp.run(out)
    assert len(calls) < len(comp.blocks())


def test_uncovered_pixels_are_black_and_exposure_applies(tmp_path):
    scan, root = default_scan(tmp_path, rows=1, cols=2, static_rows=0, vignette=0.0, gain_spread=0.0)
    src = TileSource(load_session(root))
    cam = CameraModel(160.0, 120.0, 200.0)
    model = MosaicModel.from_translations(cam, [[0.0, 0.0], [400.0, 300.0]], 320, 240)  # no overlap, gap between
    comp = Composer(src, model, None, ComposeConfig(block=256, pad=16, levels=2, workers=1, exposure=0.5))
    out = np.zeros((comp.height, comp.width, 3), np.uint8)
    comp.run(out)
    assert (out[250:290, 330:390] == 0).all()
    assert abs(out[100, 100].astype(int).mean() - 0.5 * src.load(0)[100, 100].astype(int).mean()) < 2
