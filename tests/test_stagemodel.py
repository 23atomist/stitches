import numpy as np
import pytest

from pcbstitch.stagemodel import fit_stage_model


def test_fit_recovers_matrix_with_outlier():
    rng = np.random.default_rng(0)
    true = np.array([[80.0, 1.0], [-0.8, 80.5]])
    ds = np.array([[32.5, 0.0]] * 20 + [[0.0, 22.1]] * 20)
    d = ds @ true.T + rng.normal(0, 0.5, ds.shape)
    d[5] += [300.0, -40.0]  # one garbage pair
    model, inliers = fit_stage_model(ds, d, outlier_px=4.0)
    assert np.allclose(model.matrix, true, atol=0.1)
    assert not inliers[5] and inliers.sum() == 39
    assert abs(model.px_per_unit - 80.25) < 0.2
    assert abs(model.rotation_deg - np.degrees(np.arctan2(-0.8, 80.0))) < 0.05
    pred = model.predict(np.array([[32.5, 0.0]]))
    assert np.allclose(pred, [[2600.0, -26.0]], atol=3)
    assert set(model.to_dict()) >= {"matrix", "px_per_unit", "rotation_deg", "n_inliers"}


def test_single_direction_is_minimum_norm():
    ds = np.array([[1.0, 0.0]] * 4)
    d = np.array([[200.0, -2.0]] * 4)
    model, inliers = fit_stage_model(ds, d)
    assert np.allclose(model.matrix[:, 0], [200.0, -2.0])
    assert np.allclose(model.matrix[:, 1], 0.0)
    assert inliers.all()


def test_bad_inputs():
    with pytest.raises(ValueError):
        fit_stage_model(np.zeros((0, 2)), np.zeros((0, 2)))
    with pytest.raises(ValueError):
        fit_stage_model(np.zeros((3, 2)), np.zeros((2, 2)))
