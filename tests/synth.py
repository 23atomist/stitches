"""Synthetic PCB scenes and scan sessions with known ground truth."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pyvips

from pcbstitch.camera import CameraModel, MosaicModel, fit_camera

BOARD_RGB = (38, 108, 62)
MAT_RGB = (96, 98, 102)
COPPER_RGB = (196, 150, 70)
SILK_RGB = (235, 235, 230)
PAD_RGB = (205, 205, 200)
PART_RGB = (28, 28, 30)


def make_scene(width: int, height: int, seed: int = 0, n_traces: int = 60) -> np.ndarray:
    """A board-like RGB uint8 scene with thin traces, pads, parts and texture."""
    rng = np.random.default_rng(seed)
    img = np.empty((height, width, 3), np.uint8)
    img[:] = BOARD_RGB
    mat_w = width // 5
    img[:, :mat_w] = MAT_RGB
    for x in range(0, mat_w, 40):
        cv2.line(img, (x, 0), (x, height), (190, 190, 190), 1)
    for y in range(0, height, 40):
        cv2.line(img, (0, y), (mat_w, y), (190, 190, 190), 1)

    for _ in range(max(6, n_traces // 6)):
        x, y = int(rng.integers(mat_w, width)), int(rng.integers(0, height))
        w, h = int(rng.integers(40, 120)), int(rng.integers(30, 90))
        color = tuple(int(v) for v in rng.integers(20, 230, 3))
        cv2.rectangle(img, (x, y), (x + w, y + h), color, -1)
    for _ in range(n_traces):
        x0, y0 = rng.integers(mat_w, width), rng.integers(0, height)
        angle = rng.choice([0.0, np.pi / 2, np.pi / 4, -np.pi / 4]) + rng.normal(0, 0.03)
        length = rng.integers(60, max(61, width // 2))
        x1 = int(x0 + length * np.cos(angle))
        y1 = int(y0 + length * np.sin(angle))
        cv2.line(img, (int(x0), int(y0)), (x1, y1), COPPER_RGB, int(rng.integers(2, 4)), cv2.LINE_AA)
    for _ in range(n_traces // 2):
        c = (int(rng.integers(mat_w, width)), int(rng.integers(0, height)))
        cv2.circle(img, c, int(rng.integers(4, 9)), PAD_RGB, -1, cv2.LINE_AA)
    for _ in range(n_traces // 4):
        x, y = int(rng.integers(mat_w, width)), int(rng.integers(0, height))
        w, h = int(rng.integers(12, 40)), int(rng.integers(8, 24))
        cv2.rectangle(img, (x, y), (x + w, y + h), PART_RGB, -1)
        cv2.putText(img, "R1", (x + 2, y + h - 2), cv2.FONT_HERSHEY_PLAIN, 0.6, SILK_RGB, 1, cv2.LINE_AA)
    texture = rng.normal(0, 4.0, (height, width, 1)).astype(np.float32)
    return np.clip(img.astype(np.float32) + texture, 0, 255).astype(np.uint8)


@dataclass(frozen=True)
class PhysicalCamera:
    """Ground-truth lens: Brown radial term plus tilted-plane perspective (tests only)."""

    k1: float
    cx: float
    cy: float
    radius: float
    p1: float = 0.0
    p2: float = 0.0

    def undistort(self, x, y):
        nx = (np.asarray(x, dtype=np.float64) - self.cx) / self.radius
        ny = (np.asarray(y, dtype=np.float64) - self.cy) / self.radius
        f = 1.0 + self.k1 * (nx * nx + ny * ny)
        rx, ry = nx * f, ny * f
        w = 1.0 + self.p1 * rx + self.p2 * ry
        return self.cx + self.radius * rx / w, self.cy + self.radius * ry / w

    def polynomial(self, width: int, height: int) -> CameraModel:
        return fit_camera(self.cx, self.cy, self.radius, self.undistort, width, height)


@dataclass(frozen=True)
class SynthTile:
    row: int
    col: int
    x: float
    y: float
    scale: float
    gain: np.ndarray
    image: np.ndarray


@dataclass(frozen=True)
class SynthScan:
    scene: np.ndarray
    tiles: tuple[SynthTile, ...]
    tile_w: int
    tile_h: int
    static_rows: int
    vignette_strength: float
    k1: float
    tilt: tuple[float, float] = (0.0, 0.0)

    def positions(self) -> np.ndarray:
        return np.array([[t.x, t.y] for t in self.tiles], dtype=np.float64)

    def physical_camera(self) -> PhysicalCamera:
        return PhysicalCamera(self.k1, self.tile_w / 2.0, self.tile_h / 2.0,
                              float(np.hypot(self.tile_w / 2, self.tile_h / 2)), self.tilt[0], self.tilt[1])

    def camera(self) -> CameraModel:
        return self.physical_camera().polynomial(self.tile_w, self.tile_h)

    def truth_model(self, tile_height: int | None = None) -> MosaicModel:
        """Ground-truth geometry in tile 0's undistorted frame (before normalisation)."""
        s0 = self.tiles[0].scale
        aff = np.zeros((len(self.tiles), 2, 3))
        for k, t in enumerate(self.tiles):
            r = t.scale / s0
            aff[k] = [[r, 0.0, (t.x - self.tiles[0].x) / s0], [0.0, r, (t.y - self.tiles[0].y) / s0]]
        return MosaicModel(self.camera(), aff, self.tile_w, tile_height or self.tile_h)

    def scene_at(self, model: MosaicModel, mx: np.ndarray, my: np.ndarray) -> np.ndarray:
        """Ground-truth scene rendered at mosaic coordinates of a (normalised) recovered model."""
        a0 = model.affines[0]
        s0 = self.tiles[0].scale
        xu = mx - a0[0, 2]
        yu = my - a0[1, 2]
        sx = (self.tiles[0].x + s0 * xu).astype(np.float32)
        sy = (self.tiles[0].y + s0 * yu).astype(np.float32)
        return cv2.remap(self.scene, sx, sy, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def vignette_map(h: int, w: int, strength: float) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    rx = (xx - w / 2) / (w / 2)
    ry = (yy - h / 2) / (h / 2)
    r2 = rx * rx + ry * ry
    return (1.0 - strength * r2 / 2.0).astype(np.float32)


def cut_tiles(
    scene: np.ndarray,
    rows: int,
    cols: int,
    tile_w: int,
    tile_h: int,
    step_x: int,
    step_y: int,
    seed: int = 1,
    jitter: int = 3,
    backlash: int = 4,
    skew_y_per_col: float = 0.0,
    vignette: float = 0.35,
    gain_spread: float = 0.15,
    noise: float = 1.5,
    static_rows: int = 0,
    jpeg_like_blur: bool = False,
    k1: float = 0.0,
    scale_jitter: float = 0.0,
    tilt: tuple[float, float] = (0.0, 0.0),
) -> SynthScan:
    """Cut a serpentine grid of tiles out of ``scene`` with realistic nuisances.

    ``k1`` adds radial lens distortion shared by all tiles; ``scale_jitter`` varies
    each tile's magnification (focus breathing). Tiles are rendered by sampling the
    scene at ``position + scale · undistort(pixel)``.
    """
    rng = np.random.default_rng(seed)
    v = vignette_map(tile_h, tile_w, vignette)[..., None]
    cam = PhysicalCamera(k1, tile_w / 2.0, tile_h / 2.0, float(np.hypot(tile_w / 2, tile_h / 2)), tilt[0], tilt[1])
    px, py = np.meshgrid(np.arange(tile_w, dtype=np.float64), np.arange(tile_h, dtype=np.float64))
    ux, uy = cam.undistort(px, py)
    static_pattern = None
    if static_rows:
        static_pattern = rng.normal(28, 6, (static_rows, tile_w, 3)).astype(np.float32)
        cv2.line(static_pattern, (0, static_rows // 2), (tile_w, static_rows // 2), (60, 60, 60), 2)
    tiles: list[SynthTile] = []
    for r in range(rows):
        for c in range(cols):
            x = c * step_x + int(rng.integers(-jitter, jitter + 1)) + (backlash if r % 2 else 0)
            y = r * step_y + int(round(skew_y_per_col * c)) + int(rng.integers(-jitter, jitter + 1))
            x = int(np.clip(x, 0, scene.shape[1] - tile_w))
            y = int(np.clip(y, 0, scene.shape[0] - tile_h))
            s_i = 1.0 + (rng.uniform(-scale_jitter, scale_jitter) if scale_jitter else 0.0)
            if k1 == 0.0 and s_i == 1.0 and tilt == (0.0, 0.0):
                patch = scene[y : y + tile_h, x : x + tile_w].astype(np.float32)
            else:
                patch = cv2.remap(scene, (x + s_i * ux).astype(np.float32), (y + s_i * uy).astype(np.float32),
                                  cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE).astype(np.float32)
            gain = 1.0 + rng.uniform(-gain_spread, gain_spread, 3)
            img = patch * v * gain[None, None, :]
            img += rng.normal(0, noise, img.shape).astype(np.float32)
            if static_pattern is not None:
                img[-static_rows:] = static_pattern
            if jpeg_like_blur:
                img = cv2.GaussianBlur(img, (0, 0), 0.6)
            tiles.append(SynthTile(r, c, float(x), float(y), s_i, gain, np.clip(img, 0, 255).astype(np.uint8)))
    return SynthScan(scene, tuple(tiles), tile_w, tile_h, static_rows, vignette, k1, tilt)


def write_session(scan: SynthScan, root: Path, px_per_mm: float = 10.0, quality: int = 95) -> Path:
    """Write ``scan`` as a scan directory with ``raw/`` tiles and ``metadata.json``."""
    raw = root / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    waypoints = []
    for idx, t in enumerate(scan.tiles):
        img = pyvips.Image.new_from_memory(
            np.ascontiguousarray(t.image).data, t.image.shape[1], t.image.shape[0], 3, "uchar"
        )
        img.jpegsave(str(raw / f"tile_r{t.row}_c{t.col}.jpg"), Q=quality)
        waypoints.append(
            {
                "index": idx,
                "row": t.row,
                "col": t.col,
                "x_mm": round(t.col * 26.0, 3),
                "y_mm": round(t.row * 19.0, 3),
                "direction": "L2R" if t.row % 2 == 0 else "R2L",
            }
        )
    rows = max(t.row for t in scan.tiles) + 1
    cols = max(t.col for t in scan.tiles) + 1
    meta = {
        "session_id": root.name,
        "plan": {
            "rows": rows,
            "cols": cols,
            "total_tiles": len(scan.tiles),
            "step_x_mm": 26.0,
            "step_y_mm": 19.0,
            "waypoints": waypoints,
        },
    }
    (root / "metadata.json").write_text(json.dumps(meta, indent=2))
    return root


def default_scan(tmp_path: Path, **overrides) -> tuple[SynthScan, Path]:
    """A small 3×4 serpentine scan with every nuisance turned on."""
    params = dict(
        rows=3, cols=4, tile_w=320, tile_h=240, step_x=190, step_y=64,
        vignette=0.35, gain_spread=0.15, static_rows=32, backlash=4, jitter=3,
    )
    params.update(overrides)
    scene = make_scene(
        params["cols"] * params["step_x"] + params["tile_w"] + 40,
        params["rows"] * params["step_y"] + params["tile_h"] + 40,
        seed=overrides.get("seed", 0),
    )
    scan = cut_tiles(scene, **params)
    return scan, write_session(scan, tmp_path / "pcb_scan_test")
