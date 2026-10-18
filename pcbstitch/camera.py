"""Geometric model: shared camera rectification plus one affine transform per tile.

Tile pixel ``p`` maps to mosaic coordinates as ``A_i · rectify(p)``. ``rectify``
is shared by every tile (same camera, same board tilt) and is a smooth
polynomial displacement field in coordinates normalised by the frame's
half-diagonal: quadratic and cubic monomials capture radial and tangential
lens distortion, decentering and the foreshortening of a tilted board plane in
one well-conditioned set of parameters. ``A_i`` absorbs stage translation, axis
rotation and the per-tile magnification change caused by refocusing or by the
tilted board changing the working distance.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .geometry import Rect

DISTORT_ITERATIONS = 12
MAX_NORM = 4.0  # keeps the fixed-point iteration finite for wild trial coefficients
EDGE_SAMPLES = 64
N_MONOMIALS = 7  # x², xy, y², x³, x²y, xy², y³


def monomials(nx: np.ndarray, ny: np.ndarray) -> np.ndarray:
    """Stack of the 7 non-linear monomials, shape (7, ...)."""
    return np.stack([nx * nx, nx * ny, ny * ny, nx**3, nx * nx * ny, nx * ny * ny, ny**3])


@dataclass(frozen=True)
class CameraModel:
    """``rectify(p) = c + radius · (n + C · m(n))`` with ``n = (p − c) / radius``.

    ``coeffs`` is (2, 7): per output axis, weights of the monomials in ``m``.
    """

    cx: float
    cy: float
    radius: float
    coeffs: np.ndarray = field(default_factory=lambda: np.zeros((2, N_MONOMIALS)))

    def __post_init__(self) -> None:
        c = np.asarray(self.coeffs, dtype=np.float64)
        if c.shape != (2, N_MONOMIALS):
            raise ValueError(f"coeffs must have shape (2, {N_MONOMIALS})")
        object.__setattr__(self, "coeffs", c)

    @property
    def is_identity(self) -> bool:
        return not np.any(self.coeffs)

    def _displacement(self, nx: np.ndarray, ny: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        m = monomials(nx, ny)
        return np.tensordot(self.coeffs[0], m, axes=(0, 0)), np.tensordot(self.coeffs[1], m, axes=(0, 0))

    def undistort(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        nx = (np.asarray(x, dtype=np.float64) - self.cx) / self.radius
        ny = (np.asarray(y, dtype=np.float64) - self.cy) / self.radius
        dx, dy = self._displacement(nx, ny)
        return self.cx + self.radius * (nx + dx), self.cy + self.radius * (ny + dy)

    def distort(self, xu: np.ndarray, yu: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Inverse of ``undistort`` by fixed-point iteration (the displacement is a contraction)."""
        ux = (np.asarray(xu, dtype=np.float64) - self.cx) / self.radius
        uy = (np.asarray(yu, dtype=np.float64) - self.cy) / self.radius
        nx, ny = ux.copy(), uy.copy()
        if not self.is_identity:
            for _ in range(DISTORT_ITERATIONS):
                dx, dy = self._displacement(np.clip(nx, -MAX_NORM, MAX_NORM), np.clip(ny, -MAX_NORM, MAX_NORM))
                nx, ny = ux - dx, uy - dy
        return self.cx + self.radius * nx, self.cy + self.radius * ny

    def max_displacement(self, width: int, height: int) -> float:
        """Largest |rectify(p) − p| over the tile corners, in px."""
        xs = np.array([0.0, width - 1.0, 0.0, width - 1.0])
        ys = np.array([0.0, 0.0, height - 1.0, height - 1.0])
        xu, yu = self.undistort(xs, ys)
        return float(np.hypot(xu - xs, yu - ys).max())

    def with_coeffs(self, coeffs: np.ndarray) -> CameraModel:
        return CameraModel(self.cx, self.cy, self.radius, np.asarray(coeffs, dtype=np.float64).reshape(2, N_MONOMIALS))

    def to_dict(self) -> dict:
        return {"cx": self.cx, "cy": self.cy, "radius": self.radius, "coefficients": self.coeffs.tolist(),
                "monomials": ["x2", "xy", "y2", "x3", "x2y", "xy2", "y3"]}


def identity_camera(width: int, height: int, full_width: int | None = None, full_height: int | None = None,
                    crop: Rect | None = None) -> CameraModel:
    """No rectification, centred on the optical centre of the (uncropped) frame."""
    fw = full_width or width
    fh = full_height or height
    x0 = crop.x0 if crop else 0
    y0 = crop.y0 if crop else 0
    return CameraModel(fw / 2.0 - x0, fh / 2.0 - y0, float(np.hypot(fw / 2.0, fh / 2.0)))


def fit_camera(cx: float, cy: float, radius: float, undistort_fn, width: int, height: int, samples: int = 25) -> CameraModel:
    """Least-squares polynomial approximation of an arbitrary ``undistort_fn(x, y)`` (used by tests)."""
    xs, ys = np.meshgrid(np.linspace(0, width - 1, samples), np.linspace(0, height - 1, samples))
    xu, yu = undistort_fn(xs.ravel(), ys.ravel())
    nx, ny = (xs.ravel() - cx) / radius, (ys.ravel() - cy) / radius
    m = monomials(nx, ny).T
    tx = (xu - cx) / radius - nx
    ty = (yu - cy) / radius - ny
    cxs, *_ = np.linalg.lstsq(m, tx, rcond=None)
    cys, *_ = np.linalg.lstsq(m, ty, rcond=None)
    return CameraModel(cx, cy, radius, np.stack([cxs, cys]))


@dataclass(frozen=True)
class MosaicModel:
    """Camera model plus a (N, 2, 3) stack of per-tile affines (rectified tile → mosaic)."""

    camera: CameraModel
    affines: np.ndarray
    tile_width: int
    tile_height: int

    def __post_init__(self) -> None:
        if self.affines.ndim != 3 or self.affines.shape[1:] != (2, 3):
            raise ValueError("affines must have shape (N, 2, 3)")

    @property
    def n_tiles(self) -> int:
        return int(self.affines.shape[0])

    @classmethod
    def from_translations(cls, camera: CameraModel, translations: np.ndarray, tile_width: int, tile_height: int) -> MosaicModel:
        t = np.asarray(translations, dtype=np.float64).reshape(-1, 2)
        aff = np.tile(np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]), (t.shape[0], 1, 1))
        aff[:, :, 2] = t
        return cls(camera, aff, tile_width, tile_height)

    def tile_to_mosaic(self, index: int, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        xu, yu = self.camera.undistort(x, y)
        a = self.affines[index]
        return a[0, 0] * xu + a[0, 1] * yu + a[0, 2], a[1, 0] * xu + a[1, 1] * yu + a[1, 2]

    def mosaic_to_tile(self, index: int, mx: np.ndarray, my: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Actual (distorted) tile pixel coordinates that land on mosaic points ``(mx, my)``."""
        a = self.affines[index]
        inv = np.linalg.inv(np.vstack([a, [0.0, 0.0, 1.0]]))
        mx = np.asarray(mx, dtype=np.float64)
        my = np.asarray(my, dtype=np.float64)
        xu = inv[0, 0] * mx + inv[0, 1] * my + inv[0, 2]
        yu = inv[1, 0] * mx + inv[1, 1] * my + inv[1, 2]
        return self.camera.distort(xu, yu)

    def pair_map(self, i: int, j: int):
        """Function taking tile-``i`` pixel coordinates to tile-``j`` pixel coordinates."""

        def fn(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            mx, my = self.tile_to_mosaic(i, x, y)
            return self.mosaic_to_tile(j, mx, my)

        return fn

    def _boundary_points(self) -> tuple[np.ndarray, np.ndarray]:
        t = np.linspace(0.0, 1.0, EDGE_SAMPLES)
        w, h = self.tile_width - 1.0, self.tile_height - 1.0
        xs = np.concatenate([t * w, t * w, np.zeros_like(t), np.full_like(t, w)])
        ys = np.concatenate([np.zeros_like(t), np.full_like(t, h), t * h, t * h])
        return xs, ys

    def tile_bbox(self, index: int) -> Rect:
        xs, ys = self._boundary_points()
        mx, my = self.tile_to_mosaic(index, xs, ys)
        return Rect(int(np.floor(mx.min())), int(np.floor(my.min())), int(np.ceil(mx.max())) + 1, int(np.ceil(my.max())) + 1)

    def extent(self) -> Rect:
        boxes = [self.tile_bbox(i) for i in range(self.n_tiles)]
        return Rect(min(b.x0 for b in boxes), min(b.y0 for b in boxes), max(b.x1 for b in boxes), max(b.y1 for b in boxes))

    def normalized(self) -> MosaicModel:
        """Translate every affine so the mosaic starts at (0, 0)."""
        ext = self.extent()
        aff = self.affines.copy()
        aff[:, 0, 2] -= ext.x0
        aff[:, 1, 2] -= ext.y0
        return MosaicModel(self.camera, aff, self.tile_width, self.tile_height)

    def scales(self) -> np.ndarray:
        return np.sqrt(np.abs(np.linalg.det(self.affines[:, :, :2])))

    def rotations_deg(self) -> np.ndarray:
        return np.degrees(np.arctan2(self.affines[:, 1, 0], self.affines[:, 0, 0]))

    def centres(self) -> np.ndarray:
        """Mosaic position of every tile's centre pixel, (N, 2)."""
        out = np.zeros((self.n_tiles, 2))
        for i in range(self.n_tiles):
            mx, my = self.tile_to_mosaic(i, np.array([self.tile_width / 2.0]), np.array([self.tile_height / 2.0]))
            out[i] = (mx[0], my[0])
        return out

    def to_dict(self) -> dict:
        return {
            "camera": self.camera.to_dict(),
            "max_distortion_px": self.camera.max_displacement(self.tile_width, self.tile_height),
            "tile_scales": self.scales().tolist(),
            "tile_rotations_deg": self.rotations_deg().tolist(),
            "affines": self.affines.tolist(),
        }
