"""Find image regions that never change between tiles (hardware in the field of view)."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from .errors import StaticDetectionError
from .geometry import Rect
from .tiles import TileSource

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class StaticDetection:
    """Per-pixel std across tiles at ``scale`` and the crop derived from it."""

    std_map: np.ndarray
    scale: int
    threshold: float
    crop: Rect


def tile_std_map(source: TileSource, scale: int = 16) -> np.ndarray:
    """Standard deviation of luminance across all tiles, per pixel, at ``scale``."""
    n = len(source)
    acc = None
    acc2 = None
    for index in range(n):
        g = source.gray(index, scale).astype(np.float64)
        if acc is None:
            acc = np.zeros_like(g)
            acc2 = np.zeros_like(g)
        acc += g
        acc2 += g * g
    assert acc is not None and acc2 is not None
    var = acc2 / n - (acc / n) ** 2
    return np.sqrt(np.clip(var, 0.0, None)).astype(np.float32)


def _largest_run(valid: np.ndarray) -> tuple[int, int]:
    """Start and end (exclusive) of the longest run of True in a 1-D bool array."""
    best = (0, 0)
    start = None
    for i, v in enumerate(list(valid) + [False]):
        if v and start is None:
            start = i
        elif not v and start is not None:
            if i - start > best[1] - best[0]:
                best = (start, i)
            start = None
    return best


def crop_from_std_map(
    std_map: np.ndarray,
    scale: int,
    full_width: int,
    full_height: int,
    threshold: float,
    margin: int = 32,
) -> Rect:
    """Largest axis-aligned crop whose rows and columns are not static.

    A row (column) is static when the median std along it is below ``threshold``.
    """
    good_rows = np.median(std_map, axis=1) >= threshold
    good_cols = np.median(std_map, axis=0) >= threshold
    r0, r1 = _largest_run(good_rows)
    c0, c1 = _largest_run(good_cols)
    if r1 - r0 == 0 or c1 - c0 == 0:
        raise StaticDetectionError("every row or column looks static; pass --crop explicitly")
    h, w = std_map.shape
    y0 = 0 if r0 == 0 else r0 * scale + margin
    y1 = full_height if r1 == h else r1 * scale - margin
    x0 = 0 if c0 == 0 else c0 * scale + margin
    x1 = full_width if c1 == w else c1 * scale - margin
    rect = Rect(x0, y0, x1, y1)
    if rect.is_empty:
        raise StaticDetectionError(f"static-region crop collapsed to {rect}; pass --crop explicitly")
    return rect


def detect_static_crop(
    source: TileSource,
    scale: int = 16,
    relative_threshold: float = 0.35,
    min_threshold: float = 2.0,
    margin: int = 32,
) -> StaticDetection:
    """Detect static hardware in the frame and return the crop that excludes it.

    A pixel is static when its std across tiles is below
    ``max(min_threshold, relative_threshold * median std)``.
    """
    if len(source) < 3:
        log.info("fewer than 3 tiles; skipping static-region detection")
        full = Rect(0, 0, source.full_width, source.full_height)
        return StaticDetection(np.zeros(source.shape(scale), np.float32), scale, 0.0, full)
    std_map = tile_std_map(source, scale)
    threshold = max(min_threshold, relative_threshold * float(np.median(std_map)))
    crop = crop_from_std_map(std_map, scale, source.full_width, source.full_height, threshold, margin)
    frac = float((std_map < threshold).mean())
    log.info("static detection: threshold %.1f, %.1f%% static pixels, crop %s", threshold, 100 * frac, crop)
    return StaticDetection(std_map, scale, threshold, crop)
