import numpy as np
import pytest

from pcbstitch.registration import (
    auto_coarse_scale,
    ncc_match,
    overlap_ncc,
    phase_correlation,
)
from pcbstitch.tiles import rgb_to_gray

from .synth import make_scene

SCENE = make_scene(900, 700, seed=3)


def _pair(dx, dy, h=240, w=320):
    a = SCENE[200 : 200 + h, 250 : 250 + w]
    b = SCENE[200 + dy : 200 + dy + h, 250 + dx : 250 + dx + w]
    assert b.shape == a.shape
    return a, b


@pytest.mark.parametrize("dx,dy", [(190, 0), (-190, 7), (3, 140), (-5, -140), (0, 0), (230, -30)])
def test_phase_correlation_sign_convention(dx, dy):
    a, b = _pair(dx, dy)
    off = phase_correlation(rgb_to_gray(a), rgb_to_gray(b))
    assert (off.dx, off.dy) == (dx, dy)
    assert off.score > 0.05


def test_phase_correlation_prior_restricts_search():
    a, b = _pair(190, 4)
    off = phase_correlation(rgb_to_gray(a), rgb_to_gray(b), prior=(180, 0), radius=20)
    assert (off.dx, off.dy) == (190, 4)
    far = phase_correlation(rgb_to_gray(a), rgb_to_gray(b), prior=(-100, 60), radius=10)
    assert abs(far.dx + 100) <= 10 and abs(far.dy - 60) <= 10
    assert far.score < off.score
    none = phase_correlation(rgb_to_gray(a), rgb_to_gray(b), prior=(400, 400), radius=1)
    assert (none.dx, none.dy, none.score) == (400, 400, 0.0)


def test_overlap_ncc_sign_and_empty():
    a, b = _pair(100, -20)
    ga, gb = rgb_to_gray(a), rgb_to_gray(b)
    good, area = overlap_ncc(ga, gb, 100, -20)
    bad, _ = overlap_ncc(ga, gb, 96, -20)
    assert good > 0.95 and bad < good
    assert area == (320 - 100) * (240 - 20)
    assert overlap_ncc(ga, gb, 320, 0) == (0.0, 0)
    flat = np.zeros((50, 50), np.float32)
    assert overlap_ncc(flat, flat, 5, 5)[0] == 0.0
    with pytest.raises(ValueError):
        phase_correlation(np.zeros((4, 4), np.float32), np.zeros((5, 4), np.float32))


def test_ncc_match_exact_and_subpixel():
    g = rgb_to_gray(SCENE)
    search = g[50:250, 60:300]
    tmpl = search[40:140, 70:170]
    m = ncc_match(tmpl, search)
    assert (round(m.dx), round(m.dy)) == (70, 40)
    assert m.score > 0.99
    half = 0.5 * (g[50:250, 130:230] + g[50:250, 131:231])[40:140]
    m2 = ncc_match(half.astype(np.float32), search)
    assert abs(m2.dx - 70.5) < 0.2 and abs(m2.dy - 40) < 0.2
    flat = ncc_match(np.zeros((10, 10), np.float32), np.zeros((20, 20), np.float32))
    assert flat.score == 0.0
    with pytest.raises(ValueError):
        ncc_match(np.zeros((30, 30), np.float32), np.zeros((20, 20), np.float32))


def test_auto_coarse_scale():
    assert auto_coarse_scale(9152, 6944) == 8
    assert auto_coarse_scale(320, 240) == 1
    assert auto_coarse_scale(4000, 3000) == 4
