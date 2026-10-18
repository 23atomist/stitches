"""Data-driven illumination model: flat-field plus per-tile gain and gradient, per channel.

For two tiles seeing the same scene point, the scene radiance cancels:

    log I_i(u_i) − log I_j(u_j) = F(u_i) − F(u_j) + (a_i − a_j) + gradient terms

``F`` is a smooth log flat-field on a coarse bilinear grid in sensor coordinates,
``a`` a per-tile log gain and the gradient terms a ridge-regularised per-tile
linear ramp. The system is solved per channel with iteratively re-weighted least
squares so speculars and parallax do not bias it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np
import scipy.sparse as sp

from .camera import MosaicModel
from .geometry import Rect
from .tiles import TileSource

log = logging.getLogger(__name__)

MIN_LEVEL = 8.0
MAX_LEVEL = 250.0


@dataclass(frozen=True)
class IlluminationModel:
    """Multiplicative correction for every tile; see module docstring."""

    log_flat: np.ndarray  # (gy, gx, 3)
    log_gain: np.ndarray  # (N, 3)
    grad_x: np.ndarray  # (N, 3), per normalised x in [-0.5, 0.5]
    grad_y: np.ndarray  # (N, 3)
    tile_height: int
    tile_width: int

    @classmethod
    def identity(cls, n_tiles: int, tile_height: int, tile_width: int) -> IlluminationModel:
        return cls(np.zeros((2, 2, 3)), np.zeros((n_tiles, 3)), np.zeros((n_tiles, 3)),
                   np.zeros((n_tiles, 3)), tile_height, tile_width)

    def _log_flat_region(self, rect: Rect) -> np.ndarray:
        """Log flat-field sampled over ``rect`` (full-res cropped-tile coordinates)."""
        h, w = self.tile_height, self.tile_width
        gy, gx = self.log_flat.shape[:2]
        xs = (np.arange(rect.x0, rect.x1, dtype=np.float32) / max(w - 1, 1)) * (gx - 1)
        ys = (np.arange(rect.y0, rect.y1, dtype=np.float32) / max(h - 1, 1)) * (gy - 1)
        map_x = np.ascontiguousarray(np.broadcast_to(xs[None, :], (rect.height, rect.width)), dtype=np.float32)
        map_y = np.ascontiguousarray(np.broadcast_to(ys[:, None], (rect.height, rect.width)), dtype=np.float32)
        return cv2.remap(self.log_flat.astype(np.float32), map_x, map_y, cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REPLICATE)

    def flat_field(self, height: int, width: int) -> np.ndarray:
        """Linear flat-field over a whole tile, resampled to ``height × width`` (float32, ≤ ~1)."""
        model = IlluminationModel(self.log_flat, self.log_gain, self.grad_x, self.grad_y, height, width)
        return np.exp(model._log_flat_region(Rect(0, 0, width, height))).astype(np.float32)

    def correction(self, index: int, rect: Rect, exposure: float = 1.0) -> np.ndarray:
        """Multiplier for region ``rect`` (full-res cropped-tile coordinates) of tile ``index``."""
        xs = np.arange(rect.x0, rect.x1, dtype=np.float32)
        ys = np.arange(rect.y0, rect.y1, dtype=np.float32)
        map_x = np.ascontiguousarray(np.broadcast_to(xs[None, :], (rect.height, rect.width)))
        map_y = np.ascontiguousarray(np.broadcast_to(ys[:, None], (rect.height, rect.width)))
        return self.correction_map(index, map_x, map_y, exposure)

    def correction_map(self, index: int, map_x: np.ndarray, map_y: np.ndarray, exposure: float = 1.0) -> np.ndarray:
        """Multiplier at arbitrary tile pixel coordinates (float32 maps, as used by cv2.remap)."""
        h, w = self.tile_height, self.tile_width
        gy, gx = self.log_flat.shape[:2]
        fx = (map_x / max(w - 1, 1) * (gx - 1)).astype(np.float32)
        fy = (map_y / max(h - 1, 1) * (gy - 1)).astype(np.float32)
        log_f = cv2.remap(self.log_flat.astype(np.float32), fx, fy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        xn = (map_x - w / 2.0) / w
        yn = (map_y - h / 2.0) / h
        ramp = self.grad_x[index][None, None, :] * xn[..., None] + self.grad_y[index][None, None, :] * yn[..., None]
        total = log_f + ramp + self.log_gain[index][None, None, :]
        return (exposure * np.exp(-total)).astype(np.float32)

    def to_dict(self) -> dict:
        flat = np.exp(self.log_flat)
        return {
            "grid": list(self.log_flat.shape[:2]),
            "flat_field_min": flat.min(axis=(0, 1)).tolist(),
            "tile_gain": np.exp(self.log_gain).tolist(),
            "tile_grad_x": self.grad_x.tolist(),
            "tile_grad_y": self.grad_y.tolist(),
        }


def _bilinear_nodes(x: np.ndarray, y: np.ndarray, w: int, h: int, gx: int, gy: int):
    """Grid node indices (4 per sample) and weights for positions in a w×h tile."""
    fx = np.clip(x / max(w - 1, 1) * (gx - 1), 0, gx - 1 - 1e-6)
    fy = np.clip(y / max(h - 1, 1) * (gy - 1), 0, gy - 1 - 1e-6)
    ix, iy = np.floor(fx).astype(np.int64), np.floor(fy).astype(np.int64)
    tx, ty = fx - ix, fy - iy
    nodes = np.stack([iy * gx + ix, iy * gx + ix + 1, (iy + 1) * gx + ix, (iy + 1) * gx + ix + 1], axis=1)
    weights = np.stack([(1 - tx) * (1 - ty), tx * (1 - ty), (1 - tx) * ty, tx * ty], axis=1)
    return nodes, weights


def _sample_bilinear(img: np.ndarray, xs: np.ndarray, ys: np.ndarray, width: int = 4096) -> np.ndarray:
    """Bilinear samples of ``img`` at float positions (any count; remap maps are limited to 32767 wide)."""
    n = xs.shape[0]
    rows = -(-n // width)
    mx = np.zeros((rows, width), np.float32)
    my = np.zeros((rows, width), np.float32)
    mx.ravel()[:n] = xs
    my.ravel()[:n] = ys
    out = cv2.remap(img, mx, my, cv2.INTER_LINEAR)
    return out.reshape(rows * width, *img.shape[2:])[:n]


def _pair_samples(gi: np.ndarray, gj: np.ndarray, pair_map, stride: int):
    """Corresponding sample coordinates and log intensities for one pair.

    ``pair_map`` takes tile-``i`` pixel coordinates (at this resolution) to tile-``j``
    coordinates; ``gj`` is sampled bilinearly at the mapped (float) positions.
    """
    h, w = gi.shape[:2]
    xs = np.arange(1, w - 1, stride)
    ys = np.arange(1, h - 1, stride)
    xx, yy = np.meshgrid(xs, ys)
    xx, yy = xx.ravel().astype(np.float64), yy.ravel().astype(np.float64)
    xj, yj = pair_map(xx, yy)
    inside = (xj >= 1) & (xj <= w - 2) & (yj >= 1) & (yj <= h - 2)
    if inside.sum() < 4:
        return None
    xx, yy, xj, yj = xx[inside], yy[inside], xj[inside], yj[inside]
    vi = gi[yy.astype(int), xx.astype(int)]
    vj = _sample_bilinear(gj, xj, yj)
    ok = ((vi >= MIN_LEVEL) & (vi <= MAX_LEVEL) & (vj >= MIN_LEVEL) & (vj <= MAX_LEVEL)).all(axis=1)
    if ok.sum() < 4:
        return None
    return xx[ok], yy[ok], xj[ok], yj[ok], np.log(vi[ok]), np.log(vj[ok])


def fit_illumination(
    source: TileSource,
    model: MosaicModel,
    pairs: np.ndarray,
    scale: int = 16,
    grid: tuple[int, int] = (24, 18),
    smoothness: float = 0.1,
    gradient_ridge: float = 0.3,
    gain_ridge: float = 0.1,
    chroma_ridge: float = 1.0,
    stride: int = 2,
    iterations: int = 3,
    blur_sigma: float = 1.0,
    fit_gradients: bool = True,
) -> IlluminationModel:
    """Fit the illumination model from overlaps at ``1/scale`` resolution.

    ``model`` gives the geometry (how tile pixels correspond), ``pairs`` (M, 2) the
    tile index pairs whose overlaps supply samples.
    """
    n = len(source)
    th, tw = source.shape(1)
    pairs = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    if n < 2 or pairs.shape[0] == 0:
        log.info("illumination: nothing to fit (%d tiles, %d pairs)", n, pairs.shape[0])
        return IlluminationModel.identity(n, th, tw)
    gx, gy = grid
    h, w = source.shape(scale)

    def lowres(index: int) -> np.ndarray:
        img = source.load(index, scale).astype(np.float32)
        return cv2.GaussianBlur(img, (0, 0), blur_sigma) if blur_sigma > 0 else img

    cache: dict[int, np.ndarray] = {}
    rows_i, cols_i, vals_i = [], [], []
    rhs = []
    row = 0
    n_f = gx * gy
    col_a, col_bx, col_by = n_f, n_f + n, n_f + 2 * n
    for i, j in pairs:
        i, j = int(i), int(j)
        for k in (i, j):
            if k not in cache:
                cache[k] = lowres(k)
        full_map = model.pair_map(i, j)

        def low_map(x: np.ndarray, y: np.ndarray, fm=full_map) -> tuple[np.ndarray, np.ndarray]:
            xj, yj = fm(x * scale, y * scale)
            return xj / scale, yj / scale

        samp = _pair_samples(cache[i], cache[j], low_map, stride)
        if samp is None:
            continue
        xi, yi, xj, yj, li, lj = samp
        m = xi.shape[0]
        ni, wi = _bilinear_nodes(xi, yi, w, h, gx, gy)
        nj, wj = _bilinear_nodes(xj, yj, w, h, gx, gy)
        r = np.arange(row, row + m)
        rows_i += [np.repeat(r, 4), np.repeat(r, 4), r, r]
        cols_i += [ni.ravel(), nj.ravel(), np.full(m, col_a + i), np.full(m, col_a + j)]
        vals_i += [wi.ravel(), -wj.ravel(), np.ones(m), -np.ones(m)]
        if fit_gradients:
            rows_i += [r, r, r, r]
            cols_i += [np.full(m, col_bx + i), np.full(m, col_bx + j), np.full(m, col_by + i), np.full(m, col_by + j)]
            vals_i += [(xi - w / 2) / w, -(xj - w / 2) / w, (yi - h / 2) / h, -(yj - h / 2) / h]
        rhs.append(li - lj)
        row += m
    n_data = row
    if n_data == 0:
        log.warning("illumination: no usable overlap samples; using identity")
        return IlluminationModel.identity(n, th, tw)
    n_cols = n_f + 3 * n
    data_scale = 1.0 / np.sqrt(n_data)
    a_data = sp.csr_matrix(
        (np.concatenate(vals_i) * data_scale, (np.concatenate(rows_i), np.concatenate(cols_i))),
        shape=(n_data, n_cols),
    )
    y_data = np.concatenate(rhs) * data_scale  # (n_data, 3)

    # The data term is normalised to a mean over samples, so regularisers are
    # normalised to means over their rows too; the knobs are then scale-free.
    # Curvature (second differences) is penalised so a smooth vignetting bowl is
    # not shrunk; a tiny first-difference term pins nodes no overlap observes.
    n_curv = gx * max(gy - 2, 0) + gy * max(gx - 2, 0)
    n_edges = gx * (gy - 1) + gy * (gx - 1)
    lam_c = smoothness / np.sqrt(max(n_curv, 1))
    lam_1 = 0.01 * smoothness / np.sqrt(n_edges)
    lam_g = gradient_ridge / np.sqrt(2 * n)
    # A ramp in F is indistinguishable from position-correlated gains using
    # overlaps alone; a weak ridge on the gains picks the evenly-lit solution.
    lam_a = gain_ridge / np.sqrt(n)
    # Colour: the green channel is solved directly; red and blue are solved as
    # differences from green with an extra ridge pulling the flat-field's chroma
    # to zero. Lens shading is mostly achromatic, and this keeps a chance
    # correlation between white-balance drift and stage position from turning
    # into a colour ramp across the mosaic.
    lam_chroma = chroma_ridge / np.sqrt(n_f)
    reg_rows, reg_cols, reg_vals = [], [], []
    rr = 0
    for iy in range(gy):
        for ix in range(gx):
            node = iy * gx + ix
            if ix + 1 < gx:
                reg_rows += [rr, rr]; reg_cols += [node, node + 1]; reg_vals += [lam_1, -lam_1]; rr += 1
            if iy + 1 < gy:
                reg_rows += [rr, rr]; reg_cols += [node, node + gx]; reg_vals += [lam_1, -lam_1]; rr += 1
            if 0 < ix < gx - 1:
                reg_rows += [rr] * 3; reg_cols += [node - 1, node, node + 1]; reg_vals += [lam_c, -2 * lam_c, lam_c]; rr += 1
            if 0 < iy < gy - 1:
                reg_rows += [rr] * 3; reg_cols += [node - gx, node, node + gx]; reg_vals += [lam_c, -2 * lam_c, lam_c]; rr += 1
    for k in range(n):
        reg_rows += [rr]; reg_cols += [col_a + k]; reg_vals += [lam_a]; rr += 1
        reg_rows += [rr]; reg_cols += [col_bx + k]; reg_vals += [lam_g]; rr += 1
        reg_rows += [rr]; reg_cols += [col_by + k]; reg_vals += [lam_g]; rr += 1
    gauge = 10.0
    reg_rows += [rr] * n_f; reg_cols += list(range(n_f)); reg_vals += [gauge / n_f] * n_f; rr += 1
    reg_rows += [rr] * n; reg_cols += list(range(col_a, col_a + n)); reg_vals += [gauge / n] * n; rr += 1
    if not fit_gradients:
        for k in range(n):
            reg_rows += [rr]; reg_cols += [col_bx + k]; reg_vals += [gauge]; rr += 1
            reg_rows += [rr]; reg_cols += [col_by + k]; reg_vals += [gauge]; rr += 1
    a_reg = sp.csr_matrix((reg_vals, (reg_rows, reg_cols)), shape=(rr, n_cols))
    reg_normal = (a_reg.T @ a_reg).toarray()
    chroma_normal = reg_normal.copy()
    chroma_normal[:n_f, :n_f] += (lam_chroma**2) * np.eye(n_f)

    # unknown columns: green, red−green, blue−green
    y_solve = np.stack([y_data[:, 1], y_data[:, 0] - y_data[:, 1], y_data[:, 2] - y_data[:, 1]], axis=1)
    x = np.zeros((n_cols, 3))
    weights = np.ones((n_data, 3))
    for it in range(iterations):
        for c in range(3):
            aw = a_data.T.multiply(weights[:, c])
            normal = (aw @ a_data).toarray() + (reg_normal if c == 0 else chroma_normal) + 1e-9 * np.eye(n_cols)
            x[:, c] = np.linalg.solve(normal, aw @ y_solve[:, c])
        res = a_data @ x - y_solve
        mad = np.median(np.abs(res - np.median(res, axis=0)), axis=0) * 1.4826 + 1e-6
        weights = 1.0 / (1.0 + (res / (2.5 * mad)) ** 2)
    x = np.stack([x[:, 0] + x[:, 1], x[:, 0], x[:, 0] + x[:, 2]], axis=1)  # back to R, G, B
    log_flat = x[:n_f].reshape(gy, gx, 3)
    log_flat = log_flat - np.percentile(log_flat.reshape(-1, 3), 99, axis=0)
    model = IlluminationModel(log_flat, x[col_a:col_bx], x[col_bx:col_by], x[col_by:], th, tw)
    flat = np.exp(log_flat)
    log.info("illumination: %d samples, flat-field min %s, gain range %.3f..%.3f",
             n_data, np.round(flat.min(axis=(0, 1)), 3).tolist(),
             float(np.exp(model.log_gain).min()), float(np.exp(model.log_gain).max()))
    return model
