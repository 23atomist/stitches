import numpy as np

from pcbstitch.geometry import Rect
from pcbstitch.illumination import IlluminationModel, fit_illumination
from pcbstitch.scan import load_session
from pcbstitch.tiles import TileSource

from .synth import default_scan, vignette_map


def _fit(tmp_path, **kw):
    scan, root = default_scan(tmp_path, static_rows=0, **kw)
    session = load_session(root)
    src = TileSource(session)
    pairs = np.array([[p.i, p.j] for p in session.neighbor_pairs()])
    model = fit_illumination(src, scan.truth_model(), pairs, scale=2, grid=(12, 9), stride=2)
    return scan, session, src, pairs, model


def test_recovers_vignette_and_gains(tmp_path):
    scan, session, src, pairs, model = _fit(tmp_path)
    true_v = np.log(vignette_map(240, 320, scan.vignette_strength))
    est = np.log(model.flat_field(240, 320)[..., 1])
    est -= est.max() - true_v.max()
    diff = est - true_v
    assert np.abs(diff).max() < 0.1
    # a linear ramp is a true null-space of overlap data (it trades off against
    # position-correlated gains); the bowl itself must match closely once it is removed
    yy, xx = np.mgrid[0:240, 0:320]
    basis = np.stack([np.ones(diff.size), xx.ravel() / 320, yy.ravel() / 240], axis=1)
    coef, *_ = np.linalg.lstsq(basis, diff.ravel(), rcond=None)
    deramped = diff.ravel() - basis @ coef
    assert np.abs(deramped).max() < 0.04
    true_gain = np.log(np.array([t.gain for t in scan.tiles]))
    gdiff = model.log_gain - true_gain
    # same null-space as above: gains are recovered up to a position-linear term
    pos = scan.positions()
    gbasis = np.stack([np.ones(len(pos)), pos[:, 0] / 320, pos[:, 1] / 240], axis=1)
    for c in range(3):
        coef, *_ = np.linalg.lstsq(gbasis, gdiff[:, c], rcond=None)
        assert (gdiff[:, c] - gbasis @ coef).std() < 0.02


def test_correction_makes_overlaps_agree(tmp_path):
    scan, session, src, pairs, model = _fit(tmp_path)
    h, w = src.shape()
    before, after = [], []
    for i, j in pairs:
        d = (scan.positions()[j] - scan.positions()[i]).astype(int)
        ox0, ox1 = max(0, d[0]) + 2, min(w, w + d[0]) - 2
        oy0, oy1 = max(0, d[1]) + 2, min(h, h + d[1]) - 2
        ri = Rect(ox0, oy0, ox1, oy1)
        rj = Rect(ox0 - d[0], oy0 - d[1], ox1 - d[0], oy1 - d[1])
        a = src.load(i)[ri.y0:ri.y1, ri.x0:ri.x1].astype(np.float32)
        b = src.load(j)[rj.y0:rj.y1, rj.x0:rj.x1].astype(np.float32)
        ca = a * model.correction(i, ri)
        cb = b * model.correction(j, rj)
        ok = (a > 10) & (b > 10)
        before.append(np.abs(np.log(a[ok]) - np.log(b[ok])).mean())
        after.append(np.abs(np.log(ca[ok]) - np.log(cb[ok])).mean())
    assert np.mean(after) < 0.4 * np.mean(before)
    assert np.mean(after) < 0.06


def test_identity_and_degenerate_inputs(tmp_path):
    scan, root = default_scan(tmp_path, rows=1, cols=2, static_rows=0)
    src = TileSource(load_session(root))
    ident = fit_illumination(src, scan.truth_model(), np.zeros((0, 2)), scale=2)
    corr = ident.correction(0, Rect(0, 0, 320, 240))
    assert corr.shape == (240, 320, 3) and np.allclose(corr, 1.0)
    from pcbstitch.camera import MosaicModel

    apart = MosaicModel.from_translations(scan.camera(), np.array([[0.0, 0.0], [5000.0, 0.0]]), 320, 240)
    far = fit_illumination(src, apart, np.array([[0, 1]]), scale=2)
    assert np.allclose(far.log_gain, 0.0)
    d = IlluminationModel.identity(2, 240, 320).to_dict()
    assert d["grid"] == [2, 2] and len(d["tile_gain"]) == 2
