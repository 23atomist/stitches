import json

import numpy as np
import pytest

from pcbstitch.errors import ScanError
from pcbstitch.scan import load_session

from .synth import default_scan


def test_load_session_reads_grid_and_stage_coords(tmp_path):
    scan, root = default_scan(tmp_path)
    session = load_session(root)
    assert session.rows == 3 and session.cols == 4
    assert len(session.tiles) == 12
    assert session.has_stage_coords()
    coords = session.stage_coords()
    assert coords.shape == (12, 2)
    t = session.grid()[(1, 2)]
    assert (t.stage_x_mm, t.stage_y_mm) == (52.0, 19.0)


def test_neighbor_pairs_cover_right_and_down(tmp_path):
    _, root = default_scan(tmp_path)
    session = load_session(root)
    pairs = session.neighbor_pairs()
    h = [p for p in pairs if p.axis == "h"]
    v = [p for p in pairs if p.axis == "v"]
    assert len(h) == 3 * 3 and len(v) == 2 * 4
    grid = session.grid()
    for p in pairs:
        ti, tj = session.tiles[p.i], session.tiles[p.j]
        if p.axis == "h":
            assert (tj.row, tj.col) == (ti.row, ti.col + 1)
        else:
            assert (tj.row, tj.col) == (ti.row + 1, ti.col)
    assert grid[(0, 0)].index == 0


def test_missing_metadata_falls_back_to_grid_indices(tmp_path):
    _, root = default_scan(tmp_path)
    (root / "metadata.json").unlink()
    session = load_session(root)
    assert not session.has_stage_coords()
    assert np.array_equal(session.stage_coords()[:4], [[0, 0], [1, 0], [2, 0], [3, 0]])


def test_bad_metadata_raises(tmp_path):
    _, root = default_scan(tmp_path)
    (root / "metadata.json").write_text("{not json")
    with pytest.raises(ScanError):
        load_session(root)


def test_malformed_waypoint_raises(tmp_path):
    _, root = default_scan(tmp_path)
    meta = json.loads((root / "metadata.json").read_text())
    meta["plan"]["waypoints"][0] = {"row": "x"}
    (root / "metadata.json").write_text(json.dumps(meta))
    with pytest.raises(ScanError):
        load_session(root)


def test_empty_directory_raises(tmp_path):
    with pytest.raises(ScanError):
        load_session(tmp_path)
    with pytest.raises(ScanError):
        load_session(tmp_path / "nope")


def test_duplicate_tiles_raise(tmp_path):
    import shutil

    _, root = default_scan(tmp_path)
    shutil.copy(root / "raw" / "tile_r0_c1.jpg", root / "raw" / "tile_r00_c0.jpg")
    with pytest.raises(ScanError):
        load_session(root)
