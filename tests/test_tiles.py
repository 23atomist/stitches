import numpy as np
import pytest

from pcbstitch.errors import ScanError
from pcbstitch.geometry import Rect
from pcbstitch.scan import load_session
from pcbstitch.tiles import TileSource, decode_tile, scaled_dim

from .synth import default_scan


def test_decode_matches_source_pixels(tmp_path):
    scan, root = default_scan(tmp_path)
    session = load_session(root)
    img = decode_tile(session.tiles[0].path)
    assert img.shape == (240, 320, 3) and img.dtype == np.uint8
    diff = np.abs(img.astype(int) - scan.tiles[0].image.astype(int))
    assert diff.mean() < 3.0  # JPEG q95


def test_decode_shrink_on_load_sizes(tmp_path):
    _, root = default_scan(tmp_path)
    session = load_session(root)
    for scale in (2, 4, 8):
        img = decode_tile(session.tiles[0].path, scale)
        assert img.shape[:2] == (scaled_dim(240, scale), scaled_dim(320, scale))
    with pytest.raises(ValueError):
        decode_tile(session.tiles[0].path, 3)


def test_tile_source_crop_and_cache(tmp_path):
    _, root = default_scan(tmp_path)
    session = load_session(root)
    src = TileSource(session, crop=Rect(10, 20, 310, 200), cache_bytes=10 << 20)
    assert src.shape() == (180, 300)
    assert src.shape(4) == (45, 75)
    a = src.load(3)
    b = src.load(3)
    assert a is b
    full = decode_tile(session.tiles[3].path)
    assert np.array_equal(a, full[20:200, 10:310])
    small = src.load(3, 4)
    assert small.shape == (45, 75, 3)
    assert src.gray(3, 4).dtype == np.float32
    src.drop_scale(4)
    assert (3, 4) not in src._cache and (3, 1) in src._cache


def test_tile_source_evicts_when_full(tmp_path):
    _, root = default_scan(tmp_path)
    session = load_session(root)
    src = TileSource(session, cache_bytes=2 * 320 * 240 * 3 + 1)
    for i in range(5):
        src.load(i)
    assert len(src._cache) == 2
    assert src._cache_size <= src.cache_bytes


def test_mismatched_tile_size_raises(tmp_path):
    _, root = default_scan(tmp_path)
    session = load_session(root)
    import pyvips

    bad = pyvips.Image.black(100, 80, bands=3)
    bad.jpegsave(str(session.tiles[1].path))
    with pytest.raises(ScanError):
        TileSource(session)


def test_empty_or_outside_crop_rejected(tmp_path):
    _, root = default_scan(tmp_path)
    session = load_session(root)
    with pytest.raises(ValueError):
        TileSource(session, crop=Rect(10, 10, 10, 50))
    with pytest.raises(ValueError):
        TileSource(session, crop=Rect(-5, 0, 320, 240))
    with pytest.raises(ValueError):
        TileSource(session, crop=Rect(0, 0, 400, 240))


def test_sixteen_bit_input_is_scaled(tmp_path):
    import pyvips

    _, root = default_scan(tmp_path, rows=1, cols=2)
    session = load_session(root)
    img = decode_tile(session.tiles[0].path)
    big = pyvips.Image.new_from_memory(np.ascontiguousarray(img).data, 320, 240, 3, "uchar").cast("ushort") * 256
    big.tiffsave(str(root / "raw" / "tile_r0_c0.tif"))
    session.tiles[0].path.unlink()
    again = load_session(root)
    out = decode_tile(again.tiles[0].path)
    assert out.dtype == np.uint8 and np.abs(out.astype(int) - img.astype(int)).max() <= 1
