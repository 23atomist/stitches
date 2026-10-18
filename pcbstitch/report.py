"""Diagnostics: a JSON report and an annotated low-resolution overview."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from .bundle import BundleResult
from .camera import MosaicModel
from .geometry import Rect
from .matching import Match

GOOD = (40, 220, 60)
WEAK = (240, 200, 40)
BAD = (235, 50, 50)
MIN_GOOD_MATCHES = 4
WEAK_RMS_PX = 2.0  # surface relief parallax puts the floor near 1 px
BAD_RMS_PX = 4.0


def pair_stats(matches: list[Match], residuals: np.ndarray, inliers: np.ndarray) -> list[dict]:
    """Per neighbour pair: match counts, mean NCC and residual statistics (px)."""
    groups: dict[tuple[int, int], list[int]] = {}
    for k, m in enumerate(matches):
        groups.setdefault((m.i, m.j), []).append(k)
    out = []
    for (i, j), idx in sorted(groups.items()):
        idx_arr = np.array(idx)
        norms = np.linalg.norm(residuals[idx_arr], axis=1)
        inl = inliers[idx_arr]
        out.append(
            {
                "i": i, "j": j,
                "n_matches": int(len(idx)),
                "n_inliers": int(inl.sum()),
                "score_mean": float(np.mean([matches[k].score for k in idx])),
                "residual_rms_px": float(np.sqrt(np.mean(norms[inl] ** 2))) if inl.any() else None,
                "residual_max_px": float(norms[inl].max()) if inl.any() else None,
            }
        )
    return out


def pair_colour(stat: dict) -> tuple[int, int, int]:
    rms = stat["residual_rms_px"]
    if stat["n_inliers"] < MIN_GOOD_MATCHES or rms is None or rms > BAD_RMS_PX:
        return BAD
    if rms > WEAK_RMS_PX:
        return WEAK
    return GOOD


def summarise_pairs(stats: list[dict], bundle: BundleResult) -> dict:
    norms = bundle.residual_norms[bundle.inliers] if len(bundle.inliers) else np.zeros(0)
    rms_values = [s["residual_rms_px"] for s in stats if s["residual_rms_px"] is not None]
    return {
        "n_pairs": len(stats),
        "n_matches": int(len(bundle.inliers)),
        "n_inliers": int(bundle.inliers.sum()),
        "n_weak_pairs": int(sum(pair_colour(s) != GOOD for s in stats)),
        "residual_rms_px": bundle.rms,
        "residual_median_px": float(np.median(norms)) if norms.size else None,
        "residual_max_px": float(norms.max()) if norms.size else None,
        "pair_rms_max_px": float(max(rms_values)) if rms_values else None,
    }


def write_report(report: dict, path: Path) -> Path:
    path.write_text(json.dumps(report, indent=2, default=_json_default))
    return path


def _json_default(obj):
    if isinstance(obj, Rect):
        return obj.as_tuple()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"not JSON serialisable: {type(obj)!r}")


def draw_overview(small: np.ndarray, scale: int, model: MosaicModel, stats: list[dict]) -> np.ndarray:
    """Draw tile boxes and quality-coloured pair links on a downscaled mosaic."""
    canvas = np.ascontiguousarray(small).copy()
    centres = model.centres() / scale
    for i in range(model.n_tiles):
        r = model.tile_bbox(i).scaled(scale)
        cv2.rectangle(canvas, (r.x0, r.y0), (max(r.x1 - 1, r.x0), max(r.y1 - 1, r.y0)), (255, 255, 255), 1)
    for s in stats:
        a = tuple(int(v) for v in centres[s["i"]])
        b = tuple(int(v) for v in centres[s["j"]])
        cv2.line(canvas, a, b, pair_colour(s), 2)
    for i in range(model.n_tiles):
        cx, cy = (int(v) for v in centres[i])
        cv2.putText(canvas, str(i), (cx - 8, cy + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas
