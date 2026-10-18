import numpy as np
import pytest

from pcbstitch.errors import StaticDetectionError
from pcbstitch.geometry import Rect
from pcbstitch.scan import load_session
from pcbstitch.static import crop_from_std_map, detect_static_crop
from pcbstitch.tiles import TileSource

from .synth import default_scan


def test_detects_static_bar_at_bottom(tmp_path):
    scan, root = default_scan(tmp_path, static_rows=32)
    src = TileSource(load_session(root))
    det = detect_static_crop(src, scale=4, margin=4)
    assert det.crop.x0 == 0 and det.crop.x1 == 320 and det.crop.y0 == 0
    # bar occupies rows 208..240; crop must stop at or before it, minus the margin
    assert 180 <= det.crop.y1 <= 208 - 4 + 4


def test_no_static_region_keeps_full_frame(tmp_path):
    scan, root = default_scan(tmp_path, static_rows=0)
    src = TileSource(load_session(root))
    det = detect_static_crop(src, scale=4)
    assert det.crop == Rect(0, 0, 320, 240)


def test_crop_from_std_map_margins_and_errors():
    std = np.full((20, 30), 50.0, np.float32)
    std[:3, :] = 1.0   # static band on top
    std[:, -5:] = 1.0  # static band on right
    crop = crop_from_std_map(std, scale=8, full_width=240, full_height=160, threshold=5.0, margin=2)
    assert crop == Rect(0, 3 * 8 + 2, 25 * 8 - 2, 160)
    with pytest.raises(StaticDetectionError):
        crop_from_std_map(np.zeros((4, 4), np.float32), 8, 32, 32, threshold=5.0)
    tiny = np.full((4, 4), 50.0, np.float32)
    tiny[:2] = 0.0
    with pytest.raises(StaticDetectionError):
        crop_from_std_map(tiny, 1, 4, 4, threshold=5.0, margin=3)


def test_few_tiles_skips_detection(tmp_path):
    _, root = default_scan(tmp_path, rows=1, cols=2)
    src = TileSource(load_session(root))
    det = detect_static_crop(src)
    assert det.crop == Rect(0, 0, 320, 240)
