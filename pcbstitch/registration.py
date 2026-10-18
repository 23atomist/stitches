"""Whole-frame translation estimation (phase correlation) and NCC template matching.

Offsets follow one convention everywhere: ``(dx, dy)`` is the position of tile
``j``'s origin relative to tile ``i``'s origin in mosaic space, so pixel ``(x, y)``
of tile ``i`` shows the same scene point as pixel ``(x - dx, y - dy)`` of tile ``j``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np

log = logging.getLogger(__name__)

EPS = 1e-6


@dataclass(frozen=True)
class Offset:
    dx: float
    dy: float
    score: float


@dataclass(frozen=True)
class CoarseOffset:
    """Whole-frame translation estimate for a neighbour pair (full-resolution px)."""

    i: int
    j: int
    axis: str
    dx: float
    dy: float
    score: float


def _tukey(n: int, alpha: float) -> np.ndarray:
    """Flat-top window with cosine tapers over a fraction ``alpha`` of each end."""
    if alpha <= 0:
        return np.ones(n, np.float32)
    x = np.linspace(0.0, 1.0, n)
    w = np.ones(n)
    edge = alpha / 2.0
    lo = x < edge
    hi = x > 1.0 - edge
    w[lo] = 0.5 * (1 + np.cos(np.pi * (2 * x[lo] / alpha - 1)))
    w[hi] = 0.5 * (1 + np.cos(np.pi * (2 * (1 - x[hi]) / alpha - 1)))
    return w.astype(np.float32)


def _window2d(h: int, w: int, alpha: float = 0.25) -> np.ndarray:
    return _tukey(h, alpha)[:, None] * _tukey(w, alpha)[None, :]


def overlap_ncc(a: np.ndarray, b: np.ndarray, dx: int, dy: int) -> tuple[float, int]:
    """Pearson correlation of the regions where ``a`` and ``b`` overlap at offset ``(dx, dy)``.

    Returns ``(ncc, overlap_area_px)``; ncc is 0 when the overlap is empty or flat.
    """
    h, w = a.shape
    ox0, ox1 = max(0, dx), min(w, w + dx)
    oy0, oy1 = max(0, dy), min(h, h + dy)
    if ox1 - ox0 < 2 or oy1 - oy0 < 2:
        return 0.0, 0
    ra = a[oy0:oy1, ox0:ox1].astype(np.float64)
    rb = b[oy0 - dy : oy1 - dy, ox0 - dx : ox1 - dx].astype(np.float64)
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denom = float(np.sqrt((ra * ra).sum() * (rb * rb).sum()))
    if denom < EPS:
        return 0.0, ra.size
    return float((ra * rb).sum() / denom), ra.size


def _wrap(index: int, size: int) -> int:
    return index - size if index > size // 2 else index


def _top_candidates(corr: np.ndarray, count: int, suppress: int = 3) -> list[tuple[int, int, float]]:
    """``count`` strongest local peaks of ``corr`` as ``(px, py, value)``."""
    work = corr.copy()
    out: list[tuple[int, int, float]] = []
    for _ in range(count):
        py, px = np.unravel_index(int(np.argmax(work)), work.shape)
        val = float(work[py, px])
        if not np.isfinite(val):
            break
        out.append((int(px), int(py), val))
        y0, y1 = max(0, py - suppress), min(work.shape[0], py + suppress + 1)
        x0, x1 = max(0, px - suppress), min(work.shape[1], px + suppress + 1)
        work[y0:y1, x0:x1] = -np.inf
    return out


def phase_correlation(
    a: np.ndarray,
    b: np.ndarray,
    prior: tuple[float, float] | None = None,
    radius: float | None = None,
    candidates: int = 8,
    min_overlap_fraction: float = 0.04,
) -> Offset:
    """Whole-frame translation search; optionally restricted to ``prior ± radius``.

    Frames are zero-padded to twice their size (no circular aliasing for any
    shift inside the frame), the strongest phase-correlation peaks are taken as
    candidates and each is verified by the normalised cross-correlation of the
    actual overlap. ``score`` is that NCC (0 when nothing usable was found).
    """
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch {a.shape} vs {b.shape}")
    h, w = a.shape
    ph, pw = 2 * h, 2 * w
    win = _window2d(h, w)
    fa = np.fft.rfft2((a - a.mean()) * win, s=(ph, pw))
    fb = np.fft.rfft2((b - b.mean()) * win, s=(ph, pw))
    cross = fa * np.conj(fb)
    cross /= np.abs(cross) + EPS
    corr = np.fft.irfft2(cross, s=(ph, pw))
    # b(u) = a(u + d)  =>  peak at index d (wrapped modulo the padded size)
    dxs = np.array([_wrap(x, pw) for x in range(pw)], dtype=np.float32)
    dys = np.array([_wrap(y, ph) for y in range(ph)], dtype=np.float32)
    allowed = (np.abs(dxs)[None, :] < w) & (np.abs(dys)[:, None] < h)
    if prior is not None and radius is not None:
        allowed &= (np.abs(dys - prior[1])[:, None] <= radius) & (np.abs(dxs - prior[0])[None, :] <= radius)
    corr = np.where(allowed, corr, -np.inf)
    best = Offset(float(prior[0]), float(prior[1]), 0.0) if prior is not None else Offset(0.0, 0.0, 0.0)
    min_area = min_overlap_fraction * h * w
    for px, py, _ in _top_candidates(corr, candidates):
        dx, dy = int(dxs[px]), int(dys[py])
        ncc, area = overlap_ncc(a, b, dx, dy)
        if area >= min_area and ncc > best.score:
            best = Offset(float(dx), float(dy), ncc)
    return best


def _parabolic(vm1: float, v0: float, vp1: float) -> float:
    denom = vm1 - 2.0 * v0 + vp1
    if abs(denom) < EPS:
        return 0.0
    return float(np.clip(0.5 * (vm1 - vp1) / denom, -1.0, 1.0))


def ncc_match(template: np.ndarray, search: np.ndarray) -> Offset:
    """Sub-pixel position of ``template``'s top-left inside ``search`` and the NCC score."""
    th, tw = template.shape
    sh, sw = search.shape
    if th > sh or tw > sw:
        raise ValueError("template larger than search region")
    if float(template.std()) < 1e-3 or float(search.std()) < 1e-3:
        return Offset(0.0, 0.0, 0.0)
    res = cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED)
    res = np.nan_to_num(res, nan=-1.0)
    py, px = np.unravel_index(int(np.argmax(res)), res.shape)
    score = float(res[py, px])
    sx = sy = 0.0
    if 0 < px < res.shape[1] - 1:
        sx = _parabolic(res[py, px - 1], res[py, px], res[py, px + 1])
    if 0 < py < res.shape[0] - 1:
        sy = _parabolic(res[py - 1, px], res[py, px], res[py + 1, px])
    return Offset(px + sx, py + sy, score)


def auto_coarse_scale(width: int, height: int, target: int = 1200) -> int:
    """Power-of-two shrink that brings the longest side to roughly ``target`` pixels."""
    scale = 1
    while max(width, height) / scale > target * 1.5:
        scale *= 2
    return scale
