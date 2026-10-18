import numpy as np
import pytest

from pcbstitch.camera import CameraModel, MosaicModel
from pcbstitch.matching import Match, PairMapper, match_pair_windows, ncc_match_checked, ransac_affine, translation_mapper, window_centres
from pcbstitch.tiles import rgb_to_gray

from .synth import cut_tiles, make_scene

SCENE = make_scene(900, 700, seed=11)


def test_translation_matches_are_exact():
    a = SCENE[200:440, 250:570]
    b = SCENE[198:438, 440:760]  # tile b sits at (+190, -2) relative to a
    ms = match_pair_windows(a, b, 0, 1, translation_mapper(184, 3), window=64, radius=16, max_windows=12)
    assert len(ms) >= 6
    for m in ms:
        assert abs((m.xi - m.xj) - 190) < 0.3 and abs((m.yi - m.yj) + 2) < 0.3
        assert m.score > 0.9 and m.i == 0 and m.j == 1


def test_matches_follow_a_warped_geometry():
    scan = cut_tiles(SCENE, rows=1, cols=2, tile_w=320, tile_h=240, step_x=190, step_y=0, vignette=0.0,
                     gain_spread=0.0, noise=0.5, jitter=0, backlash=0, k1=0.03, scale_jitter=0.02, seed=4)
    truth = scan.truth_model()
    a, b = scan.tiles[0].image, scan.tiles[1].image
    mapper = PairMapper(truth.pair_map(0, 1), truth.pair_map(1, 0))
    ms = match_pair_windows(a, b, 0, 1, mapper, window=48, radius=10, max_windows=20, spacing=24)
    assert len(ms) >= 8
    for m in ms:
        xj, yj = truth.pair_map(0, 1)(np.array([m.xi]), np.array([m.yi]))
        assert abs(xj[0] - m.xj) < 0.5 and abs(yj[0] - m.yj) < 0.5
    # a translation-only guess that is a few px off still finds the same correspondences
    rough = translation_mapper(188.0, 2.0)
    ms2 = match_pair_windows(a, b, 0, 1, rough, window=48, radius=12, max_windows=20, spacing=24)
    assert len(ms2) >= 8
    for m in ms2:
        xj, yj = truth.pair_map(0, 1)(np.array([m.xi]), np.array([m.yi]))
        assert abs(xj[0] - m.xj) < 1.5 and abs(yj[0] - m.yj) < 1.5


def test_rejects_flat_boundary_and_ambiguous_windows():
    flat = np.full((240, 320, 3), 100, np.uint8)
    assert match_pair_windows(flat, flat, 0, 1, translation_mapper(100, 0), window=64, radius=8) == []
    g = rgb_to_gray(SCENE)
    tmpl = g[100:164, 100:164]
    # peak sits on the search boundary -> rejected
    assert ncc_match_checked(tmpl, g[100:180, 100:180], 0.5, 0.02) is None
    assert ncc_match_checked(tmpl, g[90:174, 90:174], 0.5, 0.02) is not None
    # a vertical-stripe pattern is ambiguous along y
    stripes = np.tile((np.arange(200) % 10 < 5).astype(np.float32) * 100, (200, 1))
    assert ncc_match_checked(stripes[50:114, 50:114], stripes[30:134, 30:134], 0.5, 0.02) is None
    assert ncc_match_checked(np.zeros((8, 8), np.float32), np.zeros((20, 20), np.float32), 0.5, 0.02) is None
    with pytest.raises(ValueError):
        match_pair_windows(flat, flat[:100], 0, 1, translation_mapper(0, 0))


def test_window_centres_cover_tile():
    cx, cy = window_centres(320, 240, 64, 32)
    assert cx.min() == 32 and cx.max() == 288 and cy.max() == 192
    assert window_centres(50, 50, 64, 32)[0].size == 0


def test_ransac_keeps_affine_consensus():
    rng = np.random.default_rng(5)
    pts = rng.uniform(0, 300, (40, 2))
    aff = np.array([[1.01, 0.003, 190.0], [-0.002, 0.99, -3.0]])
    dst = pts @ aff[:, :2].T + aff[:, 2] + rng.normal(0, 0.3, pts.shape)
    matches = [Match(0, 1, *p, *d, 0.9) for p, d in zip(pts, dst)]
    for k in (3, 11, 25):  # parallax-like outliers
        m = matches[k]
        matches[k] = Match(0, 1, m.xi, m.yi, m.xj + 40, m.yj - 15, 0.95)
    kept = ransac_affine(matches, threshold=2.0)
    assert len(kept) == 37 and all(abs(m.xj - m.xi) < 200 for m in kept)
    assert ransac_affine(matches[:4], 2.0) == matches[:4]
    garbage = [Match(0, 1, float(i), float(i), rng.uniform(0, 300), rng.uniform(0, 300), 0.9) for i in range(12)]
    assert len(ransac_affine(garbage, 1.0)) <= 4
