"""Writers for the finished mosaic: pyramidal BigTIFF, preview JPEG, Deep Zoom."""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
import pyvips

log = logging.getLogger(__name__)


def mosaic_to_vips(mosaic: np.ndarray) -> pyvips.Image:
    if mosaic.ndim != 3 or mosaic.shape[2] != 3 or mosaic.dtype != np.uint8:
        raise ValueError("mosaic must be uint8 H×W×3")
    arr = np.ascontiguousarray(mosaic)
    img = pyvips.Image.new_from_memory(arr.data, arr.shape[1], arr.shape[0], 3, "uchar")
    return img.copy(interpretation="srgb")


def write_tiff(mosaic: np.ndarray, path: Path, quality: int = 92, lossless: bool = False) -> Path:
    """Tiled, pyramidal BigTIFF; JPEG-compressed by default, deflate when lossless."""
    img = mosaic_to_vips(mosaic)
    kwargs = dict(tile=True, tile_width=256, tile_height=256, pyramid=True, bigtiff=True)
    if lossless:
        kwargs.update(compression="deflate", predictor="horizontal")
    else:
        kwargs.update(compression="jpeg", Q=int(quality))
    img.tiffsave(str(path), **kwargs)
    log.info("wrote %s (%dx%d)", path, img.width, img.height)
    return path


def write_preview(mosaic: np.ndarray, path: Path, max_dim: int = 4096, quality: int = 90) -> Path:
    img = mosaic_to_vips(mosaic)
    scale = min(1.0, max_dim / max(img.width, img.height))
    if scale < 1.0:
        img = img.resize(scale, kernel="lanczos3")
    img.jpegsave(str(path), Q=int(quality))
    log.info("wrote %s (%dx%d)", path, img.width, img.height)
    return path


def write_dzi(mosaic: np.ndarray, prefix: Path, quality: int = 90) -> Path:
    """Deep Zoom pyramid ``<prefix>.dzi`` + ``<prefix>_files/`` for web viewers."""
    img = mosaic_to_vips(mosaic)
    img.dzsave(str(prefix), suffix=f".jpg[Q={int(quality)}]", tile_size=256, overlap=1)
    log.info("wrote %s.dzi", prefix)
    return prefix.with_suffix(".dzi")


def downscale(mosaic: np.ndarray, factor: int) -> np.ndarray:
    """Box-filtered RGB uint8 downscale by an integer factor."""
    img = mosaic_to_vips(mosaic).shrink(factor, factor)
    return np.ndarray(buffer=img.write_to_memory(), dtype=np.uint8, shape=[img.height, img.width, 3])


def write_png(rgb: np.ndarray, path: Path) -> Path:
    if not cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)):
        raise OSError(f"could not write {path}")
    return path
