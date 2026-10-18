"""Decode tiles (optionally shrunk on load) with an LRU cache shared between threads."""

from __future__ import annotations

import logging
import math
import threading
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np
import pyvips

from .errors import ScanError
from .geometry import Rect
from .scan import ScanSession

log = logging.getLogger(__name__)
logging.getLogger("pyvips").setLevel(logging.WARNING)

JPEG_SUFFIXES = {".jpg", ".jpeg"}
MAX_JPEG_SHRINK = 8


def _vips_to_numpy(img: pyvips.Image) -> np.ndarray:
    return np.ndarray(
        buffer=img.write_to_memory(),
        dtype=np.uint8,
        shape=[img.height, img.width, img.bands],
    )


def decode_tile(path: Path, scale: int = 1) -> np.ndarray:
    """Decode ``path`` as RGB uint8, downsampled by the power-of-two ``scale``."""
    if scale < 1 or (scale & (scale - 1)) != 0:
        raise ValueError(f"scale must be a power of two, got {scale}")
    try:
        if path.suffix.lower() in JPEG_SUFFIXES and scale > 1:
            load_shrink = min(scale, MAX_JPEG_SHRINK)
            img = pyvips.Image.new_from_file(str(path), access="sequential", shrink=load_shrink)
            if scale > load_shrink:
                img = img.shrink(scale // load_shrink, scale // load_shrink)
        else:
            img = pyvips.Image.new_from_file(str(path), access="sequential")
            if scale > 1:
                img = img.shrink(scale, scale)
    except pyvips.Error as exc:
        raise ScanError(f"cannot decode {path}: {exc.message.strip()}") from exc
    if img.bands == 1:
        img = img.bandjoin([img, img])
    elif img.bands == 4:
        img = img.extract_band(0, n=3)
    elif img.bands != 3:
        raise ScanError(f"{path}: expected 1, 3 or 4 bands, got {img.bands}")
    if img.format == "ushort":
        img = (img / 256).cast("uchar")
    elif img.format != "uchar":
        raise ScanError(f"{path}: unsupported pixel format {img.format}; expected 8- or 16-bit")
    return _vips_to_numpy(img)


def image_size(path: Path) -> tuple[int, int]:
    """(width, height) of an image without decoding its pixels."""
    try:
        img = pyvips.Image.new_from_file(str(path), access="sequential")
    except pyvips.Error as exc:
        raise ScanError(f"cannot read {path}: {exc.message.strip()}") from exc
    return img.width, img.height


def scaled_dim(full: int, scale: int) -> int:
    return math.ceil(full / scale)


class TileSource:
    """Cropped, cached access to the tiles of a session at several scales.

    ``load(index, scale)`` returns the tile cropped to ``crop`` (full-resolution
    coordinates), downsampled by ``scale``. Arrays returned are shared with the
    cache and must be treated as read-only.
    """

    def __init__(
        self,
        session: ScanSession,
        crop: Rect | None = None,
        cache_bytes: int = 4 << 30,
    ) -> None:
        self.session = session
        self.full_width, self.full_height = image_size(session.tiles[0].path)
        mismatched = [t.path.name for t in session.tiles[1:] if image_size(t.path) != (self.full_width, self.full_height)]
        if mismatched:
            raise ScanError(f"tiles differ in size from {session.tiles[0].path.name}: {', '.join(mismatched[:5])}"
                            + (" ..." if len(mismatched) > 5 else ""))
        full = Rect(0, 0, self.full_width, self.full_height)
        self.crop = crop or full
        if self.crop.is_empty:
            raise ValueError(f"empty crop {self.crop}")
        if self.crop.intersect(full) != self.crop:
            raise ValueError(f"crop {self.crop.as_tuple()} extends outside the {self.full_width}x{self.full_height} frame")
        self.cache_bytes = cache_bytes
        self._cache: OrderedDict[tuple[int, int], np.ndarray] = OrderedDict()
        self._cache_size = 0
        self._lock = threading.Lock()
        self._inflight: dict[tuple[int, int], threading.Event] = {}

    def __len__(self) -> int:
        return len(self.session.tiles)

    def shape(self, scale: int = 1) -> tuple[int, int]:
        """(height, width) of a cropped tile at ``scale``."""
        c = self.crop.scaled(scale)
        return c.height, c.width

    def _decode_cropped(self, index: int, scale: int) -> np.ndarray:
        tile = self.session.tiles[index]
        img = decode_tile(tile.path, scale)
        want_w, want_h = scaled_dim(self.full_width, scale), scaled_dim(self.full_height, scale)
        if img.shape[1] != want_w or img.shape[0] != want_h:
            if scale == 1:
                raise ScanError(
                    f"{tile.path}: size {img.shape[1]}x{img.shape[0]} differs from "
                    f"{self.full_width}x{self.full_height} of the first tile"
                )
            img = cv2.resize(img, (want_w, want_h), interpolation=cv2.INTER_AREA)
        c = self.crop.scaled(scale)
        return np.ascontiguousarray(img[c.y0 : c.y1, c.x0 : c.x1])

    def load(self, index: int, scale: int = 1) -> np.ndarray:
        key = (index, scale)
        while True:
            with self._lock:
                cached = self._cache.get(key)
                if cached is not None:
                    self._cache.move_to_end(key)
                    return cached
                pending = self._inflight.get(key)
                if pending is None:
                    pending = threading.Event()
                    self._inflight[key] = pending
                    break
            pending.wait()
        try:
            img = self._decode_cropped(index, scale)
        except BaseException:
            with self._lock:
                self._inflight.pop(key, None)
            pending.set()
            raise
        with self._lock:
            self._insert(key, img)
            self._inflight.pop(key, None)
        pending.set()
        return img

    def _insert(self, key: tuple[int, int], img: np.ndarray) -> None:
        self._cache[key] = img
        self._cache_size += img.nbytes
        while self._cache_size > self.cache_bytes and len(self._cache) > 1:
            _, evicted = self._cache.popitem(last=False)
            self._cache_size -= evicted.nbytes

    def gray(self, index: int, scale: int = 1) -> np.ndarray:
        """Float32 luminance of a cropped tile (not cached)."""
        return rgb_to_gray(self.load(index, scale))

    def drop_scale(self, scale: int) -> None:
        """Evict every cached tile at ``scale``."""
        with self._lock:
            for key in [k for k in self._cache if k[1] == scale]:
                self._cache_size -= self._cache.pop(key).nbytes


def rgb_to_gray(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
