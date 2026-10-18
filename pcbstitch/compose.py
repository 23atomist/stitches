"""Block-wise compositing: warp tiles through the mosaic model, pick seams, blend."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Callable

import cv2
import numpy as np

from .blend import multiband_blend
from .camera import MosaicModel
from .geometry import Rect
from .illumination import IlluminationModel
from .tiles import TileSource

log = logging.getLogger(__name__)

MAP_STEP = 16
NO_TILE = -1
INTERPOLATION = {"linear": cv2.INTER_LINEAR, "cubic": cv2.INTER_CUBIC, "lanczos": cv2.INTER_LANCZOS4}


@dataclass(frozen=True)
class ComposeConfig:
    block: int = 2048
    pad: int = 384
    levels: int = 6
    workers: int = 4
    exposure: float = 1.0
    interpolation: str = "cubic"

    def effective_pad(self) -> int:
        """Padding large enough that pyramid support never reaches a block's interior."""
        return max(self.pad, (2 << self.levels) + 32)


class Composer:
    """Renders a normalised ``MosaicModel`` into an output array, block by block."""

    def __init__(
        self,
        source: TileSource,
        model: MosaicModel,
        illumination: IlluminationModel | None,
        config: ComposeConfig,
    ) -> None:
        if config.interpolation not in INTERPOLATION:
            raise ValueError(f"interpolation must be one of {sorted(INTERPOLATION)}")
        self.source = source
        self.model = model
        self.illumination = illumination
        self.config = config
        ext = model.extent()
        if ext.x0 < 0 or ext.y0 < 0:
            raise ValueError("model must be normalised (extent starting at 0, 0)")
        self.width, self.height = ext.x1, ext.y1
        self.full = Rect(0, 0, self.width, self.height)
        self.bboxes = [model.tile_bbox(i) for i in range(model.n_tiles)]

    def _maps(self, index: int, region: Rect) -> tuple[np.ndarray, np.ndarray]:
        """Tile pixel coordinates for every pixel of ``region`` (float32 maps for cv2.remap)."""
        xs = np.arange(region.x0, region.x1 + MAP_STEP, MAP_STEP, dtype=np.float64)
        ys = np.arange(region.y0, region.y1 + MAP_STEP, MAP_STEP, dtype=np.float64)
        gx, gy = np.meshgrid(xs, ys)
        tx, ty = self.model.mosaic_to_tile(index, gx, gy)
        u = (np.arange(region.width, dtype=np.float32) / MAP_STEP)[None, :]
        v = (np.arange(region.height, dtype=np.float32) / MAP_STEP)[:, None]
        gu = np.ascontiguousarray(np.broadcast_to(u, (region.height, region.width)))
        gv = np.ascontiguousarray(np.broadcast_to(v, (region.height, region.width)))
        map_x = cv2.remap(tx.astype(np.float32), gu, gv, cv2.INTER_LINEAR)
        map_y = cv2.remap(ty.astype(np.float32), gu, gv, cv2.INTER_LINEAR)
        return map_x, map_y

    def _render_tile(self, index: int, map_x: np.ndarray, map_y: np.ndarray) -> np.ndarray:
        img = self.source.load(index)
        warped = cv2.remap(img, map_x, map_y, INTERPOLATION[self.config.interpolation],
                           borderMode=cv2.BORDER_REPLICATE).astype(np.float32)
        if self.illumination is not None:
            warped *= self.illumination.correction_map(index, map_x, map_y, self.config.exposure)
        elif self.config.exposure != 1.0:
            warped *= self.config.exposure
        return warped

    def _labels_and_maps(self, region: Rect, cands: list[int]):
        """Seam labels for ``region``: each pixel goes to the candidate tile in which it is
        most central (normalised distance from the tile centre), or ``NO_TILE``."""
        w, h = self.model.tile_width, self.model.tile_height
        best = np.full((region.height, region.width), np.inf, dtype=np.float32)
        labels = np.full((region.height, region.width), NO_TILE, dtype=np.int32)
        maps: list[tuple[np.ndarray, np.ndarray]] = []
        for k, index in enumerate(cands):
            map_x, map_y = self._maps(index, region)
            maps.append((map_x, map_y))
            inside = (map_x >= 0) & (map_x <= w - 1) & (map_y >= 0) & (map_y <= h - 1)
            nx = (map_x - w / 2.0) / (w / 2.0)
            ny = (map_y - h / 2.0) / (h / 2.0)
            cost = np.where(inside, nx * nx + ny * ny, np.inf).astype(np.float32)
            better = cost < best
            best = np.where(better, cost, best)
            labels = np.where(better, k, labels)
        return labels, maps

    def render_block(self, block: Rect) -> np.ndarray:
        """Render one output block (mosaic coordinates) as uint8 RGB."""
        block = block.intersect(self.full)
        out = np.zeros((max(block.height, 0), max(block.width, 0), 3), np.uint8)
        if block.is_empty:
            return out
        padded = block.expand(self.config.effective_pad()).intersect(self.full)
        cands = [i for i, bb in enumerate(self.bboxes) if not bb.intersect(padded).is_empty]
        if not cands:
            return out
        labels, maps = self._labels_and_maps(padded, cands)
        used = [int(k) for k in np.unique(labels) if k != NO_TILE]
        if not used:
            return out
        inner = Rect(block.x0 - padded.x0, block.y0 - padded.y0, block.x1 - padded.x0, block.y1 - padded.y0)
        if len(used) == 1:
            blended = self._render_tile(cands[used[0]], *maps[used[0]])
        else:
            images = [self._render_tile(cands[k], *maps[k]) for k in used]
            masks = [(labels == k).astype(np.float32) for k in used]
            blended = multiband_blend(images, masks, self.config.levels)
        result = blended[inner.y0 : inner.y1, inner.x0 : inner.x1]
        covered = labels[inner.y0 : inner.y1, inner.x0 : inner.x1] != NO_TILE
        out[:] = np.where(covered[..., None], np.clip(np.rint(result), 0, 255), 0).astype(np.uint8)
        return out

    def blocks(self) -> list[Rect]:
        b = self.config.block
        return [
            Rect(x, y, min(x + b, self.width), min(y + b, self.height))
            for y in range(0, self.height, b)
            for x in range(0, self.width, b)
        ]

    def run(self, out: np.ndarray, progress: Callable[[int, int], None] | None = None) -> None:
        """Render every block into ``out`` (uint8 H×W×3), using a thread pool."""
        if out.shape != (self.height, self.width, 3):
            raise ValueError(f"output array must be {(self.height, self.width, 3)}, got {out.shape}")
        blocks = self.blocks()
        total = len(blocks)
        done = 0

        def work(block: Rect) -> None:
            out[block.y0 : block.y1, block.x0 : block.x1] = self.render_block(block)

        pool = ThreadPoolExecutor(max_workers=max(1, self.config.workers))
        try:
            futures = [pool.submit(work, b) for b in blocks]
            for fut in as_completed(futures):
                fut.result()  # re-raises the first failure instead of finishing every block
                done += 1
                if progress is not None:
                    progress(done, total)
        finally:
            pool.shutdown(wait=True, cancel_futures=True)
