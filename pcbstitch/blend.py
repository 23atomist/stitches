"""Multi-band (Laplacian pyramid) blending of pre-placed image crops with binary masks."""

from __future__ import annotations

import math

import cv2
import numpy as np

EPS = 1e-6


def max_levels(height: int, width: int, requested: int, min_size: int = 8) -> int:
    """Largest level count such that the coarsest level is still ≥ ``min_size`` px."""
    n = int(math.floor(math.log2(max(min(height, width), 1) / min_size))) + 1
    return max(1, min(requested, n))


def gaussian_pyramid(img: np.ndarray, levels: int) -> list[np.ndarray]:
    pyr = [img]
    for _ in range(levels - 1):
        pyr.append(cv2.pyrDown(pyr[-1]))
    return pyr


def _pyr_up(img: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    up = cv2.pyrUp(img, dstsize=(shape[1], shape[0]))
    return up.reshape(shape[0], shape[1], *img.shape[2:]) if img.ndim == 3 else up


def laplacian_pyramid(img: np.ndarray, levels: int) -> list[np.ndarray]:
    gauss = gaussian_pyramid(img, levels)
    lap = [gauss[i] - _pyr_up(gauss[i + 1], gauss[i].shape[:2]) for i in range(levels - 1)]
    lap.append(gauss[-1])
    return lap


def reconstruct(lap: list[np.ndarray]) -> np.ndarray:
    img = lap[-1]
    for level in reversed(lap[:-1]):
        img = level + _pyr_up(img, level.shape[:2])
    return img


def multiband_blend(images: list[np.ndarray], masks: list[np.ndarray], levels: int) -> np.ndarray:
    """Blend ``images`` (float32 H×W×C) using ``masks`` (float32 H×W) that partition the frame.

    Low frequencies are mixed over wide regions, high frequencies over narrow
    ones, so colour differences fade smoothly while detail never doubles.
    """
    if not images:
        raise ValueError("no images to blend")
    if len(images) != len(masks):
        raise ValueError("images and masks must have the same length")
    h, w = images[0].shape[:2]
    for img, m in zip(images, masks):
        if img.shape[:2] != (h, w) or m.shape != (h, w):
            raise ValueError("all images and masks must share the same height and width")
    levels = max_levels(h, w, levels)
    mask_pyrs = [gaussian_pyramid(m.astype(np.float32), levels) for m in masks]
    lap_pyrs = [laplacian_pyramid(img.astype(np.float32), levels) for img in images]
    blended: list[np.ndarray] = []
    for lv in range(levels):
        acc = np.zeros(mask_pyrs[0][lv].shape + images[0].shape[2:], dtype=np.float32)
        wsum = np.zeros(mask_pyrs[0][lv].shape, dtype=np.float32)
        for lap, mp in zip(lap_pyrs, mask_pyrs):
            m = mp[lv]
            acc += lap[lv] * (m[..., None] if acc.ndim == 3 else m)
            wsum += m
        norm = np.where(wsum > EPS, 1.0 / np.maximum(wsum, EPS), 0.0).astype(np.float32)
        blended.append(acc * (norm[..., None] if acc.ndim == 3 else norm))
    return reconstruct(blended)
