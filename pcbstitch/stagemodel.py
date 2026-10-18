"""Affine model from stage coordinates (mm or grid indices) to mosaic pixels."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class StageModel:
    """``pixel_offset = matrix @ stage_offset`` for any pair of tiles."""

    matrix: np.ndarray
    inlier_rms: float
    n_inliers: int
    n_pairs: int

    def predict(self, stage_delta: np.ndarray) -> np.ndarray:
        return np.asarray(stage_delta, dtype=np.float64) @ self.matrix.T

    @property
    def px_per_unit(self) -> float:
        return float(math.sqrt(abs(np.linalg.det(self.matrix))))

    @property
    def rotation_deg(self) -> float:
        """Rotation of the camera x axis relative to the stage x axis."""
        return float(math.degrees(math.atan2(self.matrix[1, 0], self.matrix[0, 0])))

    def to_dict(self) -> dict:
        return {
            "matrix": self.matrix.tolist(),
            "px_per_unit": self.px_per_unit,
            "rotation_deg": self.rotation_deg,
            "inlier_rms_px": self.inlier_rms,
            "n_inliers": self.n_inliers,
            "n_pairs": self.n_pairs,
        }


def fit_stage_model(
    stage_deltas: np.ndarray,
    pixel_offsets: np.ndarray,
    weights: np.ndarray | None = None,
    outlier_px: float = 12.0,
    iterations: int = 3,
) -> tuple[StageModel, np.ndarray]:
    """Robust weighted least squares for the 2×2 stage→pixel matrix.

    Returns the model and a boolean inlier mask over the input pairs.
    """
    ds = np.asarray(stage_deltas, dtype=np.float64)
    d = np.asarray(pixel_offsets, dtype=np.float64)
    if ds.shape != d.shape or ds.ndim != 2 or ds.shape[1] != 2:
        raise ValueError("stage_deltas and pixel_offsets must both be (M, 2)")
    m = ds.shape[0]
    if m == 0:
        raise ValueError("need at least one pair to fit a stage model")
    w = np.ones(m) if weights is None else np.clip(np.asarray(weights, dtype=np.float64), 1e-6, None)
    inliers = np.ones(m, dtype=bool)
    matrix = np.zeros((2, 2))
    for _ in range(iterations):
        sw = np.sqrt(w * inliers)
        a = ds * sw[:, None]
        for axis in range(2):
            matrix[axis], *_ = np.linalg.lstsq(a, d[:, axis] * sw, rcond=None)
        res = np.linalg.norm(d - ds @ matrix.T, axis=1)
        mad = float(np.median(np.abs(res[inliers] - np.median(res[inliers])))) if inliers.any() else 0.0
        cutoff = max(outlier_px, 3.0 * 1.4826 * mad)
        new_inliers = res <= cutoff
        if new_inliers.sum() < 1:
            break
        if np.array_equal(new_inliers, inliers):
            break
        inliers = new_inliers
    res = np.linalg.norm(d - ds @ matrix.T, axis=1)
    rms = float(np.sqrt(np.mean(res[inliers] ** 2))) if inliers.any() else float("nan")
    model = StageModel(matrix.copy(), rms, int(inliers.sum()), m)
    log.info("stage model: %.3f px/unit, rotation %.2f°, %d/%d inliers, rms %.2f px",
             model.px_per_unit, model.rotation_deg, model.n_inliers, m, rms)
    return model, inliers
