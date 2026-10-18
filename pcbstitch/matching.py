"""Dense point correspondences between overlapping tiles via windowed NCC."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

import cv2
import numpy as np

from .registration import Offset, _parabolic
from .tiles import rgb_to_gray

log = logging.getLogger(__name__)

PointMap = Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]

BOUNDARY_MARGIN = 2
DISTINCT_OFFSET = 8
RANSAC_ITERATIONS = 300
MIN_RANSAC_MATCHES = 6


@dataclass(frozen=True)
class PairMapper:
    """Current best guess of how tile ``i`` coordinates relate to tile ``j`` coordinates."""

    forward: PointMap  # i -> j
    backward: PointMap  # j -> i


def translation_mapper(dx: float, dy: float) -> PairMapper:
    """Mapper for a pure translation: tile ``j`` sits at ``(dx, dy)`` relative to tile ``i``."""

    def fwd(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return np.asarray(x, dtype=np.float64) - dx, np.asarray(y, dtype=np.float64) - dy

    def bwd(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return np.asarray(x, dtype=np.float64) + dx, np.asarray(y, dtype=np.float64) + dy

    return PairMapper(fwd, bwd)


@dataclass(frozen=True)
class Match:
    """Point ``(xi, yi)`` in tile ``i`` seen at ``(xj, yj)`` in tile ``j``."""

    i: int
    j: int
    xi: float
    yi: float
    xj: float
    yj: float
    score: float


def _distinct(res: np.ndarray, px: int, py: int, offset: int) -> float:
    """Peak height above the NCC surface ``offset`` px away along each axis (aperture check)."""
    h, w = res.shape
    neighbours = [
        res[py, max(px - offset, 0)], res[py, min(px + offset, w - 1)],
        res[max(py - offset, 0), px], res[min(py + offset, h - 1), px],
    ]
    return float(res[py, px] - max(neighbours))


def _prominence(res: np.ndarray, px: int, py: int, exclude: int) -> float:
    """Peak height above the best NCC value outside a ``± exclude`` px neighbourhood."""
    work = res.copy()
    y0, y1 = max(0, py - exclude), min(res.shape[0], py + exclude + 1)
    x0, x1 = max(0, px - exclude), min(res.shape[1], px + exclude + 1)
    work[y0:y1, x0:x1] = -np.inf
    rest = float(work.max()) if np.isfinite(work).any() else -1.0
    return float(res[py, px] - rest)


def ncc_match_checked(template: np.ndarray, search: np.ndarray, min_score: float, min_distinct: float):
    """Like ``ncc_match`` but returns ``None`` for weak, boundary or ambiguous peaks.

    Ambiguity is tested two ways: the peak must drop off within a few pixels along
    both axes (aperture problem on lines) and must beat every other candidate peak
    in the search area by ``min_distinct`` (repetitive texture).
    """
    if float(template.std()) < 1e-3 or float(search.std()) < 1e-3:
        return None
    res = np.nan_to_num(cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED), nan=-1.0)
    py, px = np.unravel_index(int(np.argmax(res)), res.shape)
    score = float(res[py, px])
    if score < min_score:
        return None
    if px < BOUNDARY_MARGIN or py < BOUNDARY_MARGIN or px >= res.shape[1] - BOUNDARY_MARGIN or py >= res.shape[0] - BOUNDARY_MARGIN:
        return None
    offset = min(DISTINCT_OFFSET, res.shape[1] // 2, res.shape[0] // 2)
    if _distinct(res, px, py, offset) < min_distinct:
        return None
    if _prominence(res, px, py, max(offset, template.shape[1] // 8)) < min_distinct:
        return None
    sx = _parabolic(res[py, px - 1], res[py, px], res[py, px + 1])
    sy = _parabolic(res[py - 1, px], res[py, px], res[py + 1, px])
    return Offset(px + sx, py + sy, score)


def _affine_from_points(src: np.ndarray, dst: np.ndarray) -> np.ndarray | None:
    """Least-squares 2×3 affine mapping ``src`` points to ``dst`` (None if degenerate)."""
    n = src.shape[0]
    a = np.hstack([src, np.ones((n, 1))])
    try:
        sol, *_ = np.linalg.lstsq(a, dst, rcond=None)
    except np.linalg.LinAlgError:
        return None
    if not np.all(np.isfinite(sol)):
        return None
    return sol.T


def ransac_affine(matches: list[Match], threshold: float, iterations: int = RANSAC_ITERATIONS, seed: int = 0) -> list[Match]:
    """Largest set of matches consistent with one affine map (rejects parallax and mismatches).

    Pairs with fewer than ``MIN_RANSAC_MATCHES`` matches are returned unchanged.
    """
    n = len(matches)
    if n < MIN_RANSAC_MATCHES:
        return list(matches)
    src = np.array([[m.xi, m.yi] for m in matches])
    dst = np.array([[m.xj, m.yj] for m in matches])
    rng = np.random.default_rng(seed)
    best: np.ndarray | None = None
    for _ in range(iterations):
        idx = rng.choice(n, 3, replace=False)
        if abs(np.linalg.det(np.hstack([src[idx], np.ones((3, 1))]))) < 1e-6:
            continue
        aff = _affine_from_points(src[idx], dst[idx])
        if aff is None:
            continue
        pred = src @ aff[:, :2].T + aff[:, 2]
        inl = np.linalg.norm(pred - dst, axis=1) <= threshold
        if best is None or inl.sum() > best.sum():
            best = inl
            if inl.sum() == n:
                break
    if best is None or best.sum() < 3:
        return []
    aff = _affine_from_points(src[best], dst[best])
    if aff is not None:
        pred = src @ aff[:, :2].T + aff[:, 2]
        best = np.linalg.norm(pred - dst, axis=1) <= threshold
    return [m for m, keep in zip(matches, best) if keep]


def window_centres(width: int, height: int, window: int, spacing: int) -> tuple[np.ndarray, np.ndarray]:
    """Candidate window centres on a regular grid that keeps the window inside the tile."""
    lo = window // 2
    xs = np.arange(lo, width - lo + 1, spacing)
    ys = np.arange(lo, height - lo + 1, spacing)
    if xs.size == 0 or ys.size == 0:
        return np.zeros(0), np.zeros(0)
    gx, gy = np.meshgrid(xs, ys)
    return gx.ravel().astype(np.float64), gy.ravel().astype(np.float64)


def _subsample(n: int, keep: int) -> np.ndarray:
    if n <= keep:
        return np.arange(n)
    return np.unique(np.linspace(0, n - 1, keep).round().astype(int))


def _warped_template(gray_i: np.ndarray, mapper: PairMapper, cx: int, cy: int, half: int) -> np.ndarray | None:
    """Tile-``i`` content resampled onto tile ``j``'s pixel grid around ``(cx, cy)``."""
    h, w = gray_i.shape
    us = np.arange(cx - half, cx + half, dtype=np.float64)
    vs = np.arange(cy - half, cy + half, dtype=np.float64)
    gu, gv = np.meshgrid(us, vs)
    bx, by = mapper.backward(gu, gv)
    if bx.min() < 0 or by.min() < 0 or bx.max() > w - 1 or by.max() > h - 1:
        return None
    return cv2.remap(gray_i, bx.astype(np.float32), by.astype(np.float32), cv2.INTER_LINEAR)


def match_pair_windows(
    img_i: np.ndarray,
    img_j: np.ndarray,
    i: int,
    j: int,
    mapper: PairMapper,
    window: int = 512,
    radius: int = 220,
    spacing: int | None = None,
    max_windows: int = 30,
    min_score: float = 0.5,
    min_distinct: float = 0.02,
) -> list[Match]:
    """NCC matches between tile ``i`` and tile ``j`` around the geometry in ``mapper``.

    Windows are centred on a grid over tile ``i``; each is warped through ``mapper``
    onto tile ``j``'s pixel grid (so scale, rotation and distortion differences do not
    bias the peak) and searched for within ``± radius`` of its predicted location.
    """
    h, w = img_i.shape[:2]
    if img_j.shape[:2] != (h, w):
        raise ValueError("tiles must have the same shape")
    half = window // 2
    cx, cy = window_centres(w, h, window, spacing or max(window // 2, 1))
    if cx.size == 0:
        return []
    px, py = mapper.forward(cx, cy)
    ok = (px - half - radius >= 0) & (px + half + radius <= w) & (py - half - radius >= 0) & (py + half + radius <= h)
    cx, cy, px, py = cx[ok], cy[ok], px[ok], py[ok]
    sel = _subsample(cx.size, max_windows)
    gray_i = rgb_to_gray(img_i)
    gray_j = rgb_to_gray(img_j)
    matches: list[Match] = []
    for k in sel:
        jx, jy = int(round(px[k])), int(round(py[k]))
        tmpl = _warped_template(gray_i, mapper, jx, jy, half)
        if tmpl is None:
            continue
        sx0, sy0 = jx - half - radius, jy - half - radius
        search = gray_j[sy0 : sy0 + window + 2 * radius, sx0 : sx0 + window + 2 * radius]
        m = ncc_match_checked(tmpl, search, min_score, min_distinct)
        if m is None:
            continue
        found_x, found_y = sx0 + m.dx + half, sy0 + m.dy + half  # where the template centre landed in j
        bx, by = mapper.backward(np.array([float(jx)]), np.array([float(jy)]))
        matches.append(Match(i, j, float(bx[0]), float(by[0]), float(found_x), float(found_y), m.score))
    return matches
