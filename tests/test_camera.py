import numpy as np
import pytest

from pcbstitch.camera import CameraModel, MosaicModel, fit_camera, identity_camera
from pcbstitch.geometry import Rect


def test_distort_inverts_undistort():
    coeffs = np.array([[0.02, 0.003, -0.01, 0.004, 0.0, 0.002, 0.0], [-0.004, 0.015, 0.01, 0.0, 0.003, 0.0, -0.005]])
    cam = CameraModel(4576.0, 3472.0, 5744.0, coeffs)
    x = np.array([0.0, 9151.0, 4576.0, 100.0])
    y = np.array([0.0, 6943.0, 3472.0, 6000.0])
    xu, yu = cam.undistort(x, y)
    xb, yb = cam.distort(xu, yu)
    assert np.abs(xb - x).max() < 1e-3 and np.abs(yb - y).max() < 1e-3
    assert cam.max_displacement(9152, 6944) > 50
    ident = identity_camera(9152, 5984, 9152, 6944, Rect(0, 0, 9152, 5984))
    assert ident.is_identity and ident.max_displacement(9152, 5984) == 0
    assert np.allclose(ident.distort(x, y), (x, y))
    with pytest.raises(ValueError):
        CameraModel(1.0, 1.0, 1.0, np.zeros(3))
    d = cam.to_dict()
    assert len(d["coefficients"]) == 2 and len(d["monomials"]) == 7


def test_fit_camera_matches_a_radial_lens():
    from .synth import PhysicalCamera

    phys = PhysicalCamera(0.03, 160.0, 120.0, 200.0, 0.01, -0.005)
    poly = fit_camera(160.0, 120.0, 200.0, phys.undistort, 320, 240)
    xs, ys = np.meshgrid(np.linspace(0, 319, 9), np.linspace(0, 239, 7))
    px, py = phys.undistort(xs, ys)
    qx, qy = poly.undistort(xs, ys)
    assert np.hypot(px - qx, py - qy).max() < 0.05


def test_mosaic_model_round_trip_and_extent():
    cam = CameraModel(160.0, 120.0, 200.0, np.array([[0.01, 0, 0, 0.01, 0, 0.01, 0], [0, 0, 0.01, 0, 0.01, 0, 0.01]]))
    aff = np.array([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], [[1.01, 0.002, 190.0], [-0.002, 1.01, -3.0]]])
    model = MosaicModel(cam, aff, 320, 240)
    x = np.array([10.0, 300.0]); y = np.array([20.0, 200.0])
    mx, my = model.tile_to_mosaic(1, x, y)
    xb, yb = model.mosaic_to_tile(1, mx, my)
    assert np.allclose(xb, x, atol=1e-4) and np.allclose(yb, y, atol=1e-4)
    pm = model.pair_map(0, 1)
    xj, yj = pm(np.array([250.0]), np.array([100.0]))
    assert np.allclose(model.tile_to_mosaic(1, xj, yj), model.tile_to_mosaic(0, np.array([250.0]), np.array([100.0])), atol=1e-4)
    ext = model.extent()
    assert ext.x0 <= 0 and ext.y0 < 0 and ext.x1 > 500
    norm = model.normalized()
    assert norm.extent().x0 == 0 and norm.extent().y0 == 0
    assert np.allclose(norm.scales(), [1.0, np.sqrt(1.01**2 + 0.002**2)], atol=1e-6)
    assert abs(norm.rotations_deg()[1] - np.degrees(np.arctan2(-0.002, 1.01))) < 1e-6
    assert norm.centres().shape == (2, 2)
    d = norm.to_dict()
    assert set(d) >= {"camera", "max_distortion_px", "tile_scales", "affines"}
    with pytest.raises(ValueError):
        MosaicModel(cam, np.zeros((2, 3)), 320, 240)
    t = MosaicModel.from_translations(cam, [[0, 0], [100, 5]], 320, 240)
    assert t.n_tiles == 2 and t.affines[1, 0, 2] == 100
