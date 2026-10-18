import numpy as np
import pytest

from pcbstitch.bundle import solve_bundle
from pcbstitch.camera import CameraModel, MosaicModel

from .synth import PhysicalCamera
from pcbstitch.matching import Match


def _synthetic_matches(truth: MosaicModel, rng, per_pair=20, noise=0.2):
    n = truth.n_tiles
    matches = []
    w, h = truth.tile_width, truth.tile_height
    for i in range(n):
        for j in range(n):
            if j == i + 1 or j == i + 3:
                for _ in range(per_pair):
                    xi, yi = rng.uniform(20, w - 20), rng.uniform(20, h - 20)
                    xj, yj = truth.pair_map(i, j)(np.array([xi]), np.array([yi]))
                    if 0 <= xj[0] < w and 0 <= yj[0] < h:
                        matches.append(Match(i, j, xi, yi, xj[0] + rng.normal(0, noise), yj[0] + rng.normal(0, noise), 0.9))
    return matches


def test_recovers_distortion_and_affines():
    rng = np.random.default_rng(0)
    cam = PhysicalCamera(0.03, 160.0, 120.0, 200.0, 0.012, -0.008).polynomial(320, 240)
    aff = np.zeros((6, 2, 3))
    for k in range(6):
        s = 1.0 + rng.uniform(-0.02, 0.02)
        rot = rng.uniform(-0.01, 0.01)
        aff[k] = [[s * np.cos(rot), -s * np.sin(rot), (k % 3) * 190.0 + rng.uniform(-5, 5)],
                  [s * np.sin(rot), s * np.cos(rot), (k // 3) * 100.0 + rng.uniform(-5, 5)]]
    aff[0] = [[1, 0, 0], [0, 1, 0]]
    truth = MosaicModel(cam, aff, 320, 240)
    matches = _synthetic_matches(truth, rng)
    matches[5] = Match(0, 1, 100, 100, 300, 50, 0.9)  # one wild outlier
    initial = MosaicModel.from_translations(CameraModel(160, 120, 200), aff[:, :, 2], 320, 240)
    res = solve_bundle(matches, initial)
    assert not res.inliers[5] and res.rms < 0.5
    assert np.abs(res.model.scales() - truth.scales()).max() < 1e-2
    assert abs(res.model.camera.max_displacement(320, 240) - truth.camera.max_displacement(320, 240)) < 1.5
    # what matters: corresponding points agree between tiles (optical centre vs
    # translation is a near-gauge freedom, so raw parameters are not compared)
    xs, ys = rng.uniform(10, 310, 50), rng.uniform(10, 230, 50)
    for i, j in [(0, 1), (1, 2), (0, 3), (3, 4), (2, 5)]:
        tx, ty = truth.pair_map(i, j)(xs, ys)
        rx, ry = res.model.pair_map(i, j)(xs, ys)
        inside = (tx > 0) & (tx < 320) & (ty > 0) & (ty < 240)
        assert np.hypot(tx - rx, ty - ry)[inside].max() < 1.0


def test_without_distortion_fit_and_validation():
    rng = np.random.default_rng(2)
    cam = CameraModel(160.0, 120.0, 200.0)
    aff = np.array([[[1.0, 0, 0], [0, 1.0, 0]], [[1.01, 0, 188.0], [0, 1.01, 2.0]]])
    truth = MosaicModel(cam, aff, 320, 240)
    matches = _synthetic_matches(truth, rng, per_pair=30, noise=0.1)
    initial = MosaicModel.from_translations(cam, [[0, 0], [190, 0]], 320, 240)
    res = solve_bundle(matches, initial, fit_distortion=False)
    assert res.model.camera.is_identity
    assert np.abs(res.model.affines[1] - aff[1]).max() < 0.3
    with pytest.raises(ValueError):
        solve_bundle([], initial)
    single = MosaicModel.from_translations(cam, [[0, 0]], 320, 240)
    res1 = solve_bundle([Match(0, 0, 10, 10, 10, 10, 1.0)], single)
    assert res1.model.n_tiles == 1
