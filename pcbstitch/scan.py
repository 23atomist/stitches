"""Discover the tiles of a scan session and their stage positions."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .errors import ScanError

log = logging.getLogger(__name__)

TILE_RE = re.compile(r"^tile_r(\d+)_c(\d+)\.(jpe?g|png|tiff?)$", re.IGNORECASE)
RAW_SUBDIR = "raw"
METADATA_NAME = "metadata.json"


@dataclass(frozen=True)
class Tile:
    """One captured frame and where the stage was when it was taken."""

    index: int
    row: int
    col: int
    path: Path
    stage_x_mm: float | None
    stage_y_mm: float | None


@dataclass(frozen=True)
class Pair:
    """A pair of grid neighbours; ``j`` is to the right of (``h``) or below (``v``) ``i``."""

    i: int
    j: int
    axis: str


@dataclass(frozen=True)
class ScanSession:
    """A loaded scan: tiles in grid order plus the raw planning metadata."""

    root: Path
    tiles: tuple[Tile, ...]
    rows: int
    cols: int
    plan: dict

    def grid(self) -> dict[tuple[int, int], Tile]:
        return {(t.row, t.col): t for t in self.tiles}

    def neighbor_pairs(self) -> tuple[Pair, ...]:
        """Right and down neighbour pairs over tiles that actually exist."""
        grid = self.grid()
        pairs: list[Pair] = []
        for t in self.tiles:
            right = grid.get((t.row, t.col + 1))
            if right is not None:
                pairs.append(Pair(t.index, right.index, "h"))
            below = grid.get((t.row + 1, t.col))
            if below is not None:
                pairs.append(Pair(t.index, below.index, "v"))
        return tuple(pairs)

    def has_stage_coords(self) -> bool:
        return all(t.stage_x_mm is not None and t.stage_y_mm is not None for t in self.tiles)

    def stage_coords(self) -> np.ndarray:
        """N×2 array of stage positions in mm, or grid indices when unknown."""
        if self.has_stage_coords():
            return np.array([[t.stage_x_mm, t.stage_y_mm] for t in self.tiles], dtype=np.float64)
        return np.array([[t.col, t.row] for t in self.tiles], dtype=np.float64)


def _find_tile_files(raw_dir: Path) -> dict[tuple[int, int], Path]:
    found: dict[tuple[int, int], Path] = {}
    for path in sorted(raw_dir.iterdir()):
        m = TILE_RE.match(path.name)
        if m is None:
            continue
        key = (int(m.group(1)), int(m.group(2)))
        if key in found:
            raise ScanError(f"duplicate tile for row {key[0]} col {key[1]}: {found[key]} and {path}")
        found[key] = path
    return found


def _read_metadata(root: Path) -> dict:
    meta_path = root / METADATA_NAME
    if not meta_path.exists():
        return {}
    try:
        return json.loads(meta_path.read_text())
    except json.JSONDecodeError as exc:
        raise ScanError(f"{meta_path} is not valid JSON: {exc}") from exc


def _waypoint_positions(plan: dict) -> dict[tuple[int, int], tuple[float, float]]:
    positions: dict[tuple[int, int], tuple[float, float]] = {}
    for wp in plan.get("waypoints", []):
        try:
            positions[(int(wp["row"]), int(wp["col"]))] = (float(wp["x_mm"]), float(wp["y_mm"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise ScanError(f"malformed waypoint {wp!r}: {exc}") from exc
    return positions


def load_session(root: str | Path) -> ScanSession:
    """Load a scan directory containing ``raw/tile_r*_c*.jpg`` and optional ``metadata.json``."""
    root = Path(root)
    raw_dir = root / RAW_SUBDIR
    if not raw_dir.is_dir():
        raw_dir = root
    if not raw_dir.is_dir():
        raise ScanError(f"scan directory not found: {root}")

    files = _find_tile_files(raw_dir)
    if not files:
        raise ScanError(f"no tile_r*_c*.jpg files found in {raw_dir}")

    meta = _read_metadata(root)
    plan = meta.get("plan", {}) if isinstance(meta, dict) else {}
    positions = _waypoint_positions(plan)
    if positions and not set(files).issubset(positions):
        missing = sorted(set(files) - set(positions))
        log.warning("%d tile files have no waypoint in metadata; using grid indices for them", len(missing))

    tiles: list[Tile] = []
    for index, (row, col) in enumerate(sorted(files)):
        pos = positions.get((row, col))
        tiles.append(
            Tile(
                index=index,
                row=row,
                col=col,
                path=files[(row, col)],
                stage_x_mm=pos[0] if pos else None,
                stage_y_mm=pos[1] if pos else None,
            )
        )

    rows = max(t.row for t in tiles) + 1
    cols = max(t.col for t in tiles) + 1
    expected = plan.get("total_tiles")
    if expected is not None and int(expected) != len(tiles):
        log.warning("metadata expects %s tiles but %d files were found", expected, len(tiles))
    return ScanSession(root=root, tiles=tuple(tiles), rows=rows, cols=cols, plan=plan)
