"""Bundle adjustment: fit lens distortion and per-tile affines to window matches."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

from .camera import N_MONOMIALS, MosaicModel
from .matching import Match

log = logging.getLogger(__name__)

N_SHARED = 2 * N_MONOMIALS  # shared rectification polynomial
N_TILE = 6
IDENTITY = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])


@dataclass(frozen=True)
class BundleResult:
    model: MosaicModel
    residuals: np.ndarray  # (M, 2) mosaic-space disagreement per match
    inliers: np.ndarray  # (M,) bool
    rms: float

    @property
    def residual_norms(self) -> np.ndarray:
        return np.linalg.norm(self.residuals, axis=1)


def _unpack(x: np.ndarray, n_tiles: int, base: MosaicModel, fit_distortion: bool) -> MosaicModel:
    cam = base.camera
    if fit_distortion:
        cam = cam.with_coeffs(base.camera.coeffs.ravel() + x[:N_SHARED])
    aff = np.empty((n_tiles, 2, 3))
    aff[0] = IDENTITY.reshape(2, 3)
    aff[1:] = x[N_SHARED:].reshape(n_tiles - 1, 2, 3)
    return MosaicModel(cam, aff, base.tile_width, base.tile_height)


def _residuals(model: MosaicModel, mi: np.ndarray, mj: np.ndarray, xi, yi, xj, yj, sw) -> np.ndarray:
    a = model.affines
    xu_i, yu_i = model.camera.undistort(xi, yi)
    xu_j, yu_j = model.camera.undistort(xj, yj)
    ai, aj = a[mi], a[mj]
    mx_i = ai[:, 0, 0] * xu_i + ai[:, 0, 1] * yu_i + ai[:, 0, 2]
    my_i = ai[:, 1, 0] * xu_i + ai[:, 1, 1] * yu_i + ai[:, 1, 2]
    mx_j = aj[:, 0, 0] * xu_j + aj[:, 0, 1] * yu_j + aj[:, 0, 2]
    my_j = aj[:, 1, 0] * xu_j + aj[:, 1, 1] * yu_j + aj[:, 1, 2]
    return np.concatenate([(mx_i - mx_j) * sw, (my_i - my_j) * sw])


def solve_bundle(
    matches: list[Match],
    initial: MosaicModel,
    fit_distortion: bool = True,
    translation_prior_weight: float = 0.1,
    linear_prior_weight: float | None = None,
    outlier_px: float = 4.0,
    f_scale: float = 2.0,
) -> BundleResult:
    """Solve for a ``MosaicModel`` that makes every match land on the same mosaic point.

    Tile 0 is the gauge (identity affine). Weak priors keep tiles with few matches
    near ``initial``: ``translation_prior_weight`` is the cost per px of translation
    and, unless given, the linear-part prior is the same cost applied at the tile
    edge (so a 1 % scale change on a 9000 px tile costs like 45 px of translation).
    A robust loss plus one outlier-rejection pass handle bad matches.
    """
    if linear_prior_weight is None:
        linear_prior_weight = translation_prior_weight * max(initial.tile_width, initial.tile_height) / 2.0
    n = initial.n_tiles
    if n < 1:
        raise ValueError("need at least one tile")
    if not matches:
        raise ValueError("no matches to fit")
    mi = np.array([m.i for m in matches])
    mj = np.array([m.j for m in matches])
    xi = np.array([m.xi for m in matches]); yi = np.array([m.yi for m in matches])
    xj = np.array([m.xj for m in matches]); yj = np.array([m.yj for m in matches])
    w = np.clip(np.array([m.score for m in matches]), 0.05, None)
    n_match = len(matches)

    prior_aff = initial.affines[1:].reshape(-1, N_TILE) if n > 1 else np.zeros((0, N_TILE))
    prior_w = np.tile([linear_prior_weight, linear_prior_weight, translation_prior_weight] * 2, (max(n - 1, 0), 1)).ravel()
    x0 = np.concatenate([np.zeros(N_SHARED), prior_aff.ravel()])

    def fun(x: np.ndarray, use: np.ndarray) -> np.ndarray:
        model = _unpack(x, n, initial, fit_distortion)
        sw = np.sqrt(w) * use
        data = _residuals(model, mi, mj, xi, yi, xj, yj, sw)
        prior = (x[N_SHARED:] - prior_aff.ravel()) * prior_w
        return np.concatenate([data, prior])

    n_params = N_SHARED + N_TILE * max(n - 1, 0)
    sparsity = lil_matrix((2 * n_match + prior_w.size, n_params), dtype=int)
    rows = np.arange(n_match)
    for r_off in (0, n_match):
        if fit_distortion:
            sparsity[r_off + rows, 0:N_SHARED] = 1
        for tiles in (mi, mj):
            for k in range(N_TILE):
                has = tiles > 0
                sparsity[r_off + rows[has], N_SHARED + (tiles[has] - 1) * N_TILE + k] = 1
    for k in range(prior_w.size):
        sparsity[2 * n_match + k, N_SHARED + k] = 1
    if not fit_distortion:
        sparsity[:, 0:N_SHARED] = 0
        sparsity[0:N_SHARED, 0:N_SHARED] = 1  # keep columns non-empty for the solver

    lower = np.full(n_params, -np.inf)
    upper = np.full(n_params, np.inf)
    if fit_distortion:
        lower[0:N_SHARED] = -0.25
        upper[0:N_SHARED] = 0.25

    def solve(x_start: np.ndarray, use: np.ndarray, loss: str) -> np.ndarray:
        res = least_squares(fun, x_start, args=(use,), jac_sparsity=sparsity, loss=loss, f_scale=f_scale,
                            x_scale="jac", method="trf", bounds=(lower, upper), max_nfev=300)
        return res.x

    def evaluate(x_cur: np.ndarray):
        model = _unpack(x_cur, n, initial, fit_distortion)
        raw = _residuals(model, mi, mj, xi, yi, xj, yj, np.ones(n_match)).reshape(2, n_match).T
        return model, raw, np.linalg.norm(raw, axis=1)

    # Stage 1: plain least squares from the prior (matches are already RANSAC-vetted,
    # and a convex start keeps the solver out of poor local minima). Stage 2: reject
    # outliers and refit with a robust loss.
    inliers = np.ones(n_match, dtype=bool)
    x = solve(x0, inliers.astype(float), "linear")
    model, raw, norms = evaluate(x)
    for _ in range(2):
        mad = 1.4826 * float(np.median(np.abs(norms[inliers] - np.median(norms[inliers])))) if inliers.any() else 0.0
        cutoff = max(outlier_px, 3.0 * mad)
        new_inliers = norms <= cutoff
        if new_inliers.sum() < max(3, n_match // 4):
            break
        inliers = new_inliers
        x = solve(x, inliers.astype(float), "soft_l1")
        model, raw, norms = evaluate(x)
    rms = float(np.sqrt(np.mean(norms[inliers] ** 2))) if inliers.any() else float("nan")
    log.info("bundle: %d/%d matches inlier, rms %.2f px, rectification up to %.0f px, scales %.4f..%.4f",
             int(inliers.sum()), n_match, rms, model.camera.max_displacement(model.tile_width, model.tile_height),
             float(model.scales().min()), float(model.scales().max()))
    return BundleResult(model, raw, inliers, rms)
